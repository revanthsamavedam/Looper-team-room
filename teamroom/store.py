"""Room store — SQLite room log, teams, goals, memory, best practices.

Same pattern as the looper's SessionStore (one SQLite file, WAL mode,
one row per message). A room is a shared session every agent in it can
read and write; custom members live in the same file so they survive
restarts.

Scoping rules (the whole point of this file):
  - member memory is keyed by (team_id, handle): what a member did,
    decided and learned in one team never leaks into another team.
  - best practices are shared: one table, visible to every team.
  - a goal belongs to one team; its members are drawn from that team.

Gamification (same file, separate tables):
  - XP is per member, room-wide (the leaderboard ranks the whole
    room), earned only from real completed work — never invented.
    See XP_RULES below. Level is 1 + (xp // 100).
  - Badges are per-member milestones, awarded once each.
  - Goal progress counts completed agent replies on that goal
    ("tasks done") against the goal's target_tasks.

Backends: by default the store is a local SQLite file (the path
given to RoomStore). When TURSO_URL is set, the store talks to Turso
instead, through the libsql client — TURSO_URL is a libsql:// URL
(auth via TURSO_AUTH_TOKEN) for a remote database, or a local file
path (no token needed, handy for tests). The SQL is SQLite-compatible,
so the same schema and queries run on both. On Turso, every statement
is its own request (autocommit) — the local file's implicit
transaction does not carry over, so multi-statement operations are
not atomic there; the app tolerates that (small, idempotent writes).
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path

# XP is earned only from real completed work. Amounts are flat and
# public — see README "Gamification".
XP_TURN = 10            # member posted a reply in a completed turn
XP_DECISION = 5         # reply stated a Decision: line
XP_LEARNED = 5          # reply stated a Learned: line
XP_BEST_PRACTICE = 15   # reply stated a Best practice: line
XP_BUG = 15             # reply stated a Bug: line (usually QA)
XP_GOAL_DONE = 25       # member was on a goal when it was marked done
XP_PER_LEVEL = 100

# badge id -> (display name, what earns it)
BADGES = {
    "first-steps": ("First Steps", "completed a first turn"),
    "bug-hunter": ("Bug Hunter", "found a first bug"),
    "decider": ("Decider", "made a first decision"),
    "sharer": ("Sharer", "shared a first best practice"),
    "workhorse": ("Workhorse", "completed 10 turns"),
    "goal-getter": ("Goal Getter", "completed a goal"),
    "centurion": ("Centurion", "reached 100 XP"),
    "team-player": ("Team Player", "worked in 2 different teams"),
}


def level_for_xp(xp: int) -> dict:
    """Level 1 at 0-99 XP, level 2 at 100-199, and so on."""
    level = xp // XP_PER_LEVEL + 1
    return {"level": level, "xp": xp, "xp_into_level": xp % XP_PER_LEVEL,
            "xp_for_next": XP_PER_LEVEL}


# CREATE TABLE statements for the whole schema. Run one at a time, not
# via executescript: the libsql remote client silently drops
# executescript (tables never appear), and single statements work on
# both backends.
SCHEMA = [
    "CREATE TABLE IF NOT EXISTS rooms(id TEXT PRIMARY KEY, name TEXT, created_at REAL)",
    "CREATE TABLE IF NOT EXISTS room_messages(id INTEGER PRIMARY KEY AUTOINCREMENT, room_id TEXT, author TEXT, author_type TEXT, handle TEXT, content TEXT, ts REAL, goal_id TEXT)",
    "CREATE TABLE IF NOT EXISTS members(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, handle TEXT UNIQUE, specialty TEXT, system_prompt TEXT, source TEXT, created_at REAL)",
    "CREATE TABLE IF NOT EXISTS teams(id TEXT PRIMARY KEY, name TEXT, created_at REAL)",
    "CREATE TABLE IF NOT EXISTS team_members(team_id TEXT, handle TEXT, added_at REAL, PRIMARY KEY (team_id, handle))",
    "CREATE TABLE IF NOT EXISTS goals(id TEXT PRIMARY KEY, team_id TEXT, title TEXT, description TEXT, status TEXT, created_at REAL, target_tasks INTEGER NOT NULL DEFAULT 5)",
    "CREATE TABLE IF NOT EXISTS goal_members(goal_id TEXT, handle TEXT, PRIMARY KEY (goal_id, handle))",
    "CREATE TABLE IF NOT EXISTS member_memory(id INTEGER PRIMARY KEY AUTOINCREMENT, team_id TEXT, handle TEXT, kind TEXT, content TEXT, ts REAL)",
    "CREATE TABLE IF NOT EXISTS best_practices(id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, content TEXT, source_team_id TEXT, source_handle TEXT, created_at REAL)",
    "CREATE TABLE IF NOT EXISTS member_xp(handle TEXT PRIMARY KEY, xp INTEGER NOT NULL DEFAULT 0, turns INTEGER NOT NULL DEFAULT 0)",
    "CREATE TABLE IF NOT EXISTS member_badges(handle TEXT, badge TEXT, awarded_at REAL, PRIMARY KEY (handle, badge))",
    "CREATE TABLE IF NOT EXISTS projects(id TEXT PRIMARY KEY, name TEXT, description TEXT, goal_id TEXT, team_id TEXT, created_at REAL)",
    "CREATE TABLE IF NOT EXISTS project_files(id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT, path TEXT, content TEXT, updated_at REAL, UNIQUE(project_id, path))",
]


# Max characters per project file. Keeps a file small enough for the
# GitHub contents API (1 MB limit, content is base64-encoded on push).
MAX_PROJECT_FILE_CHARS = 500_000


def sanitize_project_path(path: str) -> str:
    """Normalise a project file path. Rejects absolute paths, '..'
    segments, empty paths and backslashes; returns a clean posix-style
    path like 'main.py' or 'src/app.py'."""
    raw = (path or "").strip().replace("\\", "/")
    if raw.startswith("/"):
        raise ValueError("invalid file path — use a relative path like main.py")
    parts = [p for p in raw.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts):
        raise ValueError("invalid file path — use a relative path like main.py")
    clean = "/".join(parts)
    if len(clean) > 200:
        raise ValueError("file path is too long (max 200 characters)")
    return clean


# Transient Turso/Hrana errors worth retrying: the connection dropped
# mid-request ("connection closed before message completed"), timeouts,
# resets. These are usually the network or the edge, not the query — a
# retry a moment later normally succeeds. Only reads (SELECT/PRAGMA)
# are retried: re-running a write after a lost response could apply it
# twice (double XP, duplicate message), so writes get one attempt and
# their callers handle the failure (endpoint -> 503, turn -> best-effort).
_TRANSIENT_PATTERNS = (
    "connection closed", "timed out", "timeout", "connection reset",
    "broken pipe", "temporarily unavailable", "too many requests",
    "connection aborted", "network unreachable", "name resolution",
    "503", "502", "504",
)


def _is_transient(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(p in msg for p in _TRANSIENT_PATTERNS)


class _LockedConnection:
    """Serialize every call to a DB connection through one lock.

    The server runs turns in a background thread while the SSE
    generator polls the store from the request thread, so two threads
    can touch the connection at once. The libsql remote client is not
    safe for that; sqlite3 tolerates it but does not need the risk.
    Cursors are safe to read after execute returns — both backends
    hold the result rows in memory by then. The lock is the store's
    own RLock, so methods that already hold it stay reentrant.
    """

    def __init__(self, conn, lock):
        self._conn = conn
        self._lock = lock

    def execute(self, sql, params=()):
        if sql.strip().upper().startswith(("SELECT", "PRAGMA")):
            return self._execute_read_with_retry(sql, params)
        with self._lock:
            return self._conn.execute(sql, params)

    def _execute_read_with_retry(self, sql, params=()):
        """A read retried on transient errors. Each attempt takes the
        lock briefly and releases it before the backoff sleep, so a
        retry never blocks the other thread's polls."""
        last: Exception | None = None
        for i in range(3):
            with self._lock:
                try:
                    return self._conn.execute(sql, params)
                except Exception as exc:  # noqa: BLE001 — retry transient only
                    if not _is_transient(exc) or i == 2:
                        raise
                    last = exc
            time.sleep(0.25 * (2 ** i))
        raise last  # pragma: no cover — the loop always returns or raises

    def executemany(self, sql, params):
        with self._lock:
            return self._conn.executemany(sql, params)

    def executescript(self, sql):
        with self._lock:
            return self._conn.executescript(sql)

    def commit(self):
        with self._lock:
            return self._conn.commit()

    def close(self):
        with self._lock:
            return self._conn.close()


class RoomStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._lock = threading.RLock()
        turso_url = (os.environ.get("TURSO_URL") or "").strip()
        if turso_url:
            # Persistent hosted store via the libsql client. TURSO_URL
            # is a libsql:// URL (auth via TURSO_AUTH_TOKEN) or a
            # local file path (no token needed — test hook). Do not
            # pass auth_token=None: libsql rejects a None token.
            import libsql
            token = (os.environ.get("TURSO_AUTH_TOKEN") or "").strip()
            if token:
                raw = libsql.connect(turso_url, auth_token=token)
            else:
                raw = libsql.connect(turso_url)
            self.backend = "turso"
            self.turso_url = turso_url
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            raw = sqlite3.connect(str(self.path), check_same_thread=False)
            raw.execute("PRAGMA journal_mode=WAL")
            self.backend = "sqlite"
            self.turso_url = None
        self.conn = _LockedConnection(raw, self._lock)
        self._init()

    def _init(self):
        with self._lock:
            for stmt in SCHEMA:
                self.conn.execute(stmt)
            self.conn.commit()
            # Migration: room_messages gained goal_id after v0.1.0.
            cols = {r[1] for r in self.conn.execute(
                "PRAGMA table_info(room_messages)").fetchall()}
            if "goal_id" not in cols:
                self.conn.execute("ALTER TABLE room_messages ADD COLUMN goal_id TEXT")
            # Migration: goals gained target_tasks (gamification).
            gcols = {r[1] for r in self.conn.execute(
                "PRAGMA table_info(goals)").fetchall()}
            if "target_tasks" not in gcols:
                self.conn.execute(
                    "ALTER TABLE goals ADD COLUMN target_tasks INTEGER NOT NULL DEFAULT 5")
            self.conn.commit()

    # -- rooms ------------------------------------------------------------
    def create_room(self, name: str, room_id: str | None = None) -> dict:
        rid = room_id or uuid.uuid4().hex[:12]
        with self._lock:
            self.conn.execute("INSERT OR IGNORE INTO rooms VALUES (?,?,?)",
                              (rid, name, time.time()))
            self.conn.commit()
            row = self.conn.execute("SELECT id, name, created_at FROM rooms WHERE id=?",
                                    (rid,)).fetchone()
        return {"id": row[0], "name": row[1], "created_at": row[2]}

    def get_room(self, room_id: str) -> dict | None:
        row = self.conn.execute("SELECT id, name, created_at FROM rooms WHERE id=?",
                                (room_id,)).fetchone()
        return {"id": row[0], "name": row[1], "created_at": row[2]} if row else None

    def list_rooms(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, name, created_at FROM rooms ORDER BY created_at").fetchall()
        return [{"id": r[0], "name": r[1], "created_at": r[2]} for r in rows]

    # -- messages ---------------------------------------------------------
    def append_message(self, room_id: str, author: str, author_type: str,
                       content: str, handle: str | None = None,
                       goal_id: str | None = None) -> dict:
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO room_messages(room_id, author, author_type, handle,"
                " content, ts, goal_id) VALUES (?,?,?,?,?,?,?)",
                (room_id, author, author_type, handle, content, time.time(), goal_id))
            self.conn.commit()
            mid = cur.lastrowid
        return self.get_message(mid)

    def get_message(self, mid: int) -> dict:
        row = self.conn.execute(
            "SELECT id, room_id, author, author_type, handle, content, ts, goal_id"
            " FROM room_messages WHERE id=?", (mid,)).fetchone()
        return {"id": row[0], "room_id": row[1], "author": row[2],
                "author_type": row[3], "handle": row[4],
                "content": row[5], "ts": row[6], "goal_id": row[7]}

    def messages(self, room_id: str, after: int = 0, limit: int = 200) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, room_id, author, author_type, handle, content, ts, goal_id"
            " FROM room_messages WHERE room_id=? AND id>? ORDER BY id LIMIT ?",
            (room_id, after, limit)).fetchall()
        return [{"id": r[0], "room_id": r[1], "author": r[2], "author_type": r[3],
                 "handle": r[4], "content": r[5], "ts": r[6], "goal_id": r[7]}
                for r in rows]

    def recent(self, room_id: str, limit: int = 10) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, room_id, author, author_type, handle, content, ts, goal_id"
            " FROM room_messages WHERE room_id=? ORDER BY id DESC LIMIT ?",
            (room_id, limit)).fetchall()
        rows.reverse()
        return [{"id": r[0], "room_id": r[1], "author": r[2], "author_type": r[3],
                 "handle": r[4], "content": r[5], "ts": r[6], "goal_id": r[7]}
                for r in rows]

    # -- custom members ---------------------------------------------------
    def add_member(self, name: str, handle: str, specialty: str,
                   system_prompt: str, source: str) -> dict:
        with self._lock:
            try:
                cur = self.conn.execute(
                    "INSERT INTO members(name, handle, specialty, system_prompt, source, created_at)"
                    " VALUES (?,?,?,?,?,?)",
                    (name, handle, specialty, system_prompt, source, time.time()))
                self.conn.commit()
            except (sqlite3.IntegrityError, ValueError) as exc:
                # sqlite3 raises IntegrityError for a duplicate handle;
                # libsql raises plain ValueError("UNIQUE constraint
                # failed: ..."). Only a uniqueness failure means the
                # handle exists — anything else propagates.
                if (isinstance(exc, sqlite3.IntegrityError)
                        or "UNIQUE constraint failed" in str(exc)):
                    raise ValueError(f"a member with handle @{handle} already exists")
                raise
            row = self.conn.execute(
                "SELECT id, name, handle, specialty, system_prompt, source, created_at"
                " FROM members WHERE id=?", (cur.lastrowid,)).fetchone()
        return {"id": row[0], "name": row[1], "handle": row[2], "specialty": row[3],
                "system_prompt": row[4], "source": row[5], "created_at": row[6]}

    def list_members(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, name, handle, specialty, system_prompt, source, created_at"
            " FROM members ORDER BY created_at").fetchall()
        return [{"id": r[0], "name": r[1], "handle": r[2], "specialty": r[3],
                 "system_prompt": r[4], "source": r[5], "created_at": r[6]} for r in rows]

    def get_member(self, handle: str) -> dict | None:
        row = self.conn.execute(
            "SELECT id, name, handle, specialty, system_prompt, source, created_at"
            " FROM members WHERE handle=?", (handle,)).fetchone()
        return {"id": row[0], "name": row[1], "handle": row[2], "specialty": row[3],
                "system_prompt": row[4], "source": row[5], "created_at": row[6]} if row else None

    def delete_member(self, handle: str) -> bool:
        with self._lock:
            cur = self.conn.execute("DELETE FROM members WHERE handle=?", (handle,))
            self.conn.commit()
            return cur.rowcount > 0

    # -- teams ------------------------------------------------------------
    def create_team(self, name: str, handles: list[str]) -> dict:
        tid = uuid.uuid4().hex[:12]
        with self._lock:
            self.conn.execute("INSERT INTO teams VALUES (?,?,?)",
                              (tid, name.strip(), time.time()))
            for h in handles:
                self.conn.execute("INSERT OR IGNORE INTO team_members VALUES (?,?,?)",
                                  (tid, h, time.time()))
            self.conn.commit()
        return self.get_team(tid)

    def get_team(self, team_id: str) -> dict | None:
        row = self.conn.execute("SELECT id, name, created_at FROM teams WHERE id=?",
                                (team_id,)).fetchone()
        if not row:
            return None
        return {"id": row[0], "name": row[1], "created_at": row[2],
                "members": self.team_members(team_id)}

    def list_teams(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, name, created_at FROM teams ORDER BY created_at").fetchall()
        return [{"id": r[0], "name": r[1], "created_at": r[2],
                 "members": self.team_members(r[0])} for r in rows]

    def team_members(self, team_id: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT handle FROM team_members WHERE team_id=? ORDER BY added_at",
            (team_id,)).fetchall()
        return [r[0] for r in rows]

    def add_team_member(self, team_id: str, handle: str) -> None:
        with self._lock:
            self.conn.execute("INSERT OR IGNORE INTO team_members VALUES (?,?,?)",
                              (team_id, handle, time.time()))
            self.conn.commit()

    def remove_team_member(self, team_id: str, handle: str) -> bool:
        with self._lock:
            cur = self.conn.execute(
                "DELETE FROM team_members WHERE team_id=? AND handle=?",
                (team_id, handle))
            self.conn.commit()
            return cur.rowcount > 0

    # -- goals ------------------------------------------------------------
    def create_goal(self, team_id: str, title: str, description: str,
                    handles: list[str], target_tasks: int = 5) -> dict:
        """A goal belongs to one team; its members must be team members.

        target_tasks is how many completed agent replies on the goal
        count as "done" for the progress bar (default 5)."""
        team_handles = set(self.team_members(team_id))
        outsiders = [h for h in handles if h not in team_handles]
        if outsiders:
            raise ValueError(
                f"@{', @'.join(outsiders)} {'is' if len(outsiders) == 1 else 'are'}"
                " not on this team — goal members must be drawn from the team")
        try:
            target_tasks = max(1, int(target_tasks))
        except (TypeError, ValueError):
            target_tasks = 5
        gid = uuid.uuid4().hex[:12]
        with self._lock:
            self.conn.execute(
                "INSERT INTO goals VALUES (?,?,?,?,?,?,?)",
                (gid, team_id, title.strip(), description.strip(), "active",
                 time.time(), target_tasks))
            for h in handles:
                self.conn.execute("INSERT OR IGNORE INTO goal_members VALUES (?,?)",
                                  (gid, h))
            self.conn.commit()
        return self.get_goal(gid)

    def get_goal(self, goal_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT id, team_id, title, description, status, created_at,"
            " target_tasks FROM goals WHERE id=?", (goal_id,)).fetchone()
        if not row:
            return None
        return {"id": row[0], "team_id": row[1], "title": row[2],
                "description": row[3], "status": row[4], "created_at": row[5],
                "target_tasks": row[6], "tasks_done": self.goal_task_count(goal_id),
                "members": self.goal_members(goal_id)}

    def list_goals(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, team_id, title, description, status, created_at,"
            " target_tasks FROM goals ORDER BY created_at").fetchall()
        return [{"id": r[0], "team_id": r[1], "title": r[2], "description": r[3],
                 "status": r[4], "created_at": r[5], "target_tasks": r[6],
                 "tasks_done": self.goal_task_count(r[0]),
                 "members": self.goal_members(r[0])} for r in rows]

    def goal_task_count(self, goal_id: str) -> int:
        """Completed agent replies on this goal — the progress numerator."""
        row = self.conn.execute(
            "SELECT COUNT(*) FROM room_messages"
            " WHERE goal_id=? AND author_type='agent'", (goal_id,)).fetchone()
        return row[0] if row else 0

    def goal_members(self, goal_id: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT handle FROM goal_members WHERE goal_id=?", (goal_id,)).fetchall()
        return [r[0] for r in rows]

    def set_goal_status(self, goal_id: str, status: str) -> dict | None:
        if status not in ("active", "done"):
            raise ValueError("status must be 'active' or 'done'")
        goal = self.get_goal(goal_id)
        if not goal:
            return None
        with self._lock:
            self.conn.execute("UPDATE goals SET status=? WHERE id=?", (status, goal_id))
            self.conn.commit()
        # Completing a goal pays XP to its members — once, on the
        # active -> done transition only.
        if goal["status"] == "active" and status == "done":
            for handle in goal["members"]:
                self.add_xp(handle, XP_GOAL_DONE)
                self.award_badge(handle, "goal-getter")
        return self.get_goal(goal_id)

    def delete_goal(self, goal_id: str) -> bool:
        with self._lock:
            self.conn.execute("DELETE FROM goal_members WHERE goal_id=?", (goal_id,))
            cur = self.conn.execute("DELETE FROM goals WHERE id=?", (goal_id,))
            self.conn.commit()
            return cur.rowcount > 0

    # -- projects -----------------------------------------------------------
    # A project is a real workspace for a goal (or a standalone piece of
    # work): its files live in project_files, in this same store, so they
    # survive restarts and cold starts — unlike anything an agent writes
    # to the ephemeral Vercel filesystem. The agents' file tools
    # (list_files / read_file / write_file) operate on the active
    # project's files; see teamroom/projects.py.
    def create_project(self, name: str, description: str = "",
                       goal_id: str | None = None,
                       team_id: str | None = None) -> dict:
        """Create a project, optionally linked to a goal (its team is
        then the goal's team unless a team is given explicitly)."""
        name = (name or "").strip()
        if not name:
            raise ValueError("a project needs a name")
        if goal_id:
            goal = self.get_goal(goal_id)
            if not goal:
                raise ValueError(f"no such goal {goal_id}")
            team_id = team_id or goal["team_id"]
        if team_id and not self.get_team(team_id):
            raise ValueError(f"no such team {team_id}")
        pid = uuid.uuid4().hex[:12]
        with self._lock:
            self.conn.execute(
                "INSERT INTO projects VALUES (?,?,?,?,?,?)",
                (pid, name, (description or "").strip(), goal_id, team_id,
                 time.time()))
            self.conn.commit()
        return self.get_project(pid)

    def get_project(self, project_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT id, name, description, goal_id, team_id, created_at"
            " FROM projects WHERE id=?", (project_id,)).fetchone()
        if not row:
            return None
        return {"id": row[0], "name": row[1], "description": row[2],
                "goal_id": row[3], "team_id": row[4], "created_at": row[5],
                "file_count": self.project_file_count(row[0]),
                "files": [f["path"] for f in self.list_project_files(row[0])]}

    def list_projects(self, goal_id: str | None = None,
                      team_id: str | None = None) -> list[dict]:
        sql = ("SELECT id, name, description, goal_id, team_id, created_at"
               " FROM projects")
        conds, params = [], []
        if goal_id:
            conds.append("goal_id=?")
            params.append(goal_id)
        if team_id:
            conds.append("team_id=?")
            params.append(team_id)
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        rows = self.conn.execute(sql + " ORDER BY created_at",
                                 tuple(params)).fetchall()
        return [{"id": r[0], "name": r[1], "description": r[2],
                 "goal_id": r[3], "team_id": r[4], "created_at": r[5],
                 "file_count": self.project_file_count(r[0]),
                 "files": [f["path"] for f in self.list_project_files(r[0])]}
                for r in rows]

    def get_project_for_goal(self, goal_id: str) -> dict | None:
        """The most recently created project linked to a goal, if any —
        the default project for that goal's turns."""
        row = self.conn.execute(
            "SELECT id FROM projects WHERE goal_id=? ORDER BY created_at DESC"
            " LIMIT 1", (goal_id,)).fetchone()
        return self.get_project(row[0]) if row else None

    def list_project_files(self, project_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT path, updated_at FROM project_files"
            " WHERE project_id=? ORDER BY path", (project_id,)).fetchall()
        return [{"path": r[0], "updated_at": r[1]} for r in rows]

    def project_file_count(self, project_id: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM project_files WHERE project_id=?",
            (project_id,)).fetchone()
        return row[0] if row else 0

    def get_project_file(self, project_id: str, path: str) -> dict | None:
        clean = sanitize_project_path(path)
        row = self.conn.execute(
            "SELECT path, content, updated_at FROM project_files"
            " WHERE project_id=? AND path=?", (project_id, clean)).fetchone()
        return ({"path": row[0], "content": row[1], "updated_at": row[2]}
                if row else None)

    def upsert_project_file(self, project_id: str, path: str,
                            content: str) -> dict:
        """Write (or overwrite) one file in a project. Returns the
        stored file row."""
        if not self.get_project(project_id):
            raise ValueError(f"no such project {project_id}")
        clean = sanitize_project_path(path)
        content = "" if content is None else str(content)
        if len(content) > MAX_PROJECT_FILE_CHARS:
            raise ValueError(
                f"file is too large (max {MAX_PROJECT_FILE_CHARS} characters)")
        with self._lock:
            self.conn.execute(
                "INSERT INTO project_files(project_id, path, content, updated_at)"
                " VALUES (?,?,?,?)"
                " ON CONFLICT(project_id, path) DO UPDATE SET"
                " content=excluded.content, updated_at=excluded.updated_at",
                (project_id, clean, content, time.time()))
            self.conn.commit()
        return self.get_project_file(project_id, clean)

    # -- team-scoped member memory ----------------------------------------
    def record_memory(self, team_id: str, handle: str, kind: str,
                      content: str) -> dict:
        """kind: did | decided | learned. Scoped by (team_id, handle)."""
        if kind not in ("did", "decided", "learned"):
            raise ValueError("memory kind must be did, decided or learned")
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO member_memory(team_id, handle, kind, content, ts)"
                " VALUES (?,?,?,?,?)",
                (team_id, handle, kind, content.strip(), time.time()))
            self.conn.commit()
            mid = cur.lastrowid
        return {"id": mid, "team_id": team_id, "handle": handle,
                "kind": kind, "content": content.strip()}

    def list_memory(self, team_id: str, handle: str | None = None,
                    limit: int = 50) -> list[dict]:
        if handle:
            rows = self.conn.execute(
                "SELECT id, team_id, handle, kind, content, ts FROM member_memory"
                " WHERE team_id=? AND handle=? ORDER BY id DESC LIMIT ?",
                (team_id, handle, limit)).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT id, team_id, handle, kind, content, ts FROM member_memory"
                " WHERE team_id=? ORDER BY id DESC LIMIT ?",
                (team_id, limit)).fetchall()
        return [{"id": r[0], "team_id": r[1], "handle": r[2], "kind": r[3],
                 "content": r[4], "ts": r[5]} for r in rows]

    # -- shared best practices --------------------------------------------
    def add_best_practice(self, title: str, content: str,
                          source_team_id: str | None = None,
                          source_handle: str | None = None) -> dict | None:
        """Shared across all teams. Identical content is recorded once."""
        title, content = title.strip(), content.strip()
        if not title or not content:
            raise ValueError("a best practice needs a title and content")
        with self._lock:
            dup = self.conn.execute(
                "SELECT id FROM best_practices WHERE content=?", (content,)).fetchone()
            if dup:
                return None
            cur = self.conn.execute(
                "INSERT INTO best_practices(title, content, source_team_id,"
                " source_handle, created_at) VALUES (?,?,?,?,?)",
                (title, content, source_team_id, source_handle, time.time()))
            self.conn.commit()
            pid = cur.lastrowid
        return {"id": pid, "title": title, "content": content,
                "source_team_id": source_team_id, "source_handle": source_handle}

    def list_best_practices(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, title, content, source_team_id, source_handle, created_at"
            " FROM best_practices ORDER BY created_at").fetchall()
        return [{"id": r[0], "title": r[1], "content": r[2],
                 "source_team_id": r[3], "source_handle": r[4], "created_at": r[5]}
                for r in rows]

    def delete_best_practice(self, practice_id: int) -> bool:
        with self._lock:
            cur = self.conn.execute("DELETE FROM best_practices WHERE id=?",
                                    (practice_id,))
            self.conn.commit()
            return cur.rowcount > 0

    # -- gamification: XP, levels, badges ---------------------------------
    def add_xp(self, handle: str, amount: int) -> dict:
        """Add XP to a member (room-wide). Returns the new totals."""
        with self._lock:
            self.conn.execute(
                "INSERT INTO member_xp(handle, xp, turns) VALUES (?,?,0)"
                " ON CONFLICT(handle) DO UPDATE SET xp = xp + ?",
                (handle, amount, amount))
            self.conn.commit()
        return self.get_xp(handle)

    def record_turn(self, handle: str) -> dict:
        """One completed turn for a member: +1 turn, +XP_TURN XP."""
        with self._lock:
            self.conn.execute(
                "INSERT INTO member_xp(handle, xp, turns) VALUES (?,?,1)"
                " ON CONFLICT(handle) DO UPDATE SET xp = xp + ?, turns = turns + 1",
                (handle, XP_TURN, XP_TURN))
            self.conn.commit()
        return self.get_xp(handle)

    def get_xp(self, handle: str) -> dict:
        row = self.conn.execute(
            "SELECT xp, turns FROM member_xp WHERE handle=?", (handle,)).fetchone()
        xp, turns = (row[0], row[1]) if row else (0, 0)
        out = level_for_xp(xp)
        out["handle"] = handle
        out["turns"] = turns
        out["badges"] = self.list_badges(handle)
        return out

    def leaderboard(self) -> list[dict]:
        """All members with XP, best first. Handles with no row are 0."""
        rows = self.conn.execute(
            "SELECT handle, xp, turns FROM member_xp ORDER BY xp DESC, turns DESC"
        ).fetchall()
        return [{"handle": r[0], **level_for_xp(r[1]), "turns": r[2],
                 "badges": self.list_badges(r[0])} for r in rows]

    def award_badge(self, handle: str, badge: str) -> bool:
        """Award a badge once. Returns True only when newly awarded."""
        if badge not in BADGES:
            raise ValueError(f"unknown badge {badge}")
        with self._lock:
            cur = self.conn.execute(
                "INSERT OR IGNORE INTO member_badges VALUES (?,?,?)",
                (handle, badge, time.time()))
            self.conn.commit()
            return cur.rowcount > 0

    def list_badges(self, handle: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT badge, awarded_at FROM member_badges WHERE handle=?"
            " ORDER BY awarded_at", (handle,)).fetchall()
        return [{"id": r[0], "name": BADGES[r[0]][0], "desc": BADGES[r[0]][1],
                 "awarded_at": r[1]} for r in rows if r[0] in BADGES]

    def teams_worked(self, handle: str) -> int:
        """How many different teams this member has memory in."""
        row = self.conn.execute(
            "SELECT COUNT(DISTINCT team_id) FROM member_memory WHERE handle=?",
            (handle,)).fetchone()
        return row[0] if row else 0
