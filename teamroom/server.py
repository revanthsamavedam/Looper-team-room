"""Team Room server — FastAPI app: UI, REST API, SSE stream.

Run:  python -m teamroom [--port 8765]   (from the custom-looper dir)
      or ./run-teamroom.sh
Env:  TEAMROOM_HOME   data dir (default ~/.looper-teamroom;
                    on Vercel: /tmp/looper-teamroom — only used when
                    the store is local; see TURSO_* below)
      TURSO_URL / TURSO_AUTH_TOKEN — when both are set, the store
                    talks to a Turso database via the libsql client
                    and the data persists across restarts/instances.
                    Without them the store is the local SQLite file.
      TEAMROOM_PORT   default 8765
      TEAMROOM_TURN_DEADLINE_S  cap on a turn, stream + work (default 45)
      LOOPER_MODEL / LOOPER_API_KEY / OPENAI_API_KEY — model config
      TEAMROOM_MOCK_MODEL=1 — scripted replies, verification only

BYOK: the browser may send its own OpenAI key per request
(X-OpenAI-Key / X-OpenAI-Model headers). The key lives only in the
in-memory provider for that turn — it is never stored, logged, or
written to the DB. See README "Bring your own key".

Serverless note: the message POST streams the turn (SSE) instead of
returning immediately, because a serverless function freezes after
its response — a background thread would never finish the turn. The
stream ends with turn_done, or turn_timeout when the soft deadline
is hit; the client then shows the partial results honestly.
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import threading
import time
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from . import agents, github, orchestrator
from .skills import parse_skill_text
from .store import RoomStore

TEAMROOM_HOME = Path(os.environ.get("TEAMROOM_HOME", Path.home() / ".looper-teamroom"))
STATIC_DIR = Path(__file__).parent / "static"
DEFAULT_ROOM_ID = "main"

store = RoomStore(TEAMROOM_HOME / "teamroom.db")
model_status = agents.probe_model()
store.create_room("Team Room", room_id=DEFAULT_ROOM_ID)

_run_lock = threading.Lock()
_run_active = False

app = FastAPI(title="Looper Team Room")


def _member_json(m: agents.Member) -> dict:
    xp = store.get_xp(m.handle)
    return {"name": m.name, "handle": m.handle, "colour": m.colour,
            "emoji": m.emoji, "specialty": m.specialty,
            "builtin": m.builtin, "source": m.source,
            "xp": xp["xp"], "level": xp["level"],
            "xp_into_level": xp["xp_into_level"],
            "xp_for_next": xp["xp_for_next"], "turns": xp["turns"],
            "badges": xp["badges"]}


def _known_handles() -> set[str]:
    return {m.handle for m in agents.list_members(store)}


def _validate_handles(handles: list[str]) -> list[str]:
    """Dedupe, lower-case, and reject handles that are not members."""
    known = _known_handles()
    clean = []
    for h in handles or []:
        h = str(h).strip().lower()
        if not h:
            continue
        if h not in known:
            raise HTTPException(400, f"no member with handle @{h}")
        if h not in clean:
            clean.append(h)
    return clean


# -- UI -------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


# -- status / roster ------------------------------------------------------
@app.get("/api/status")
def status():
    return {"model": model_status,
            "byok": True,
            "rooms": len(store.list_rooms()),
            "members": len(agents.list_members(store)),
            "teams": len(store.list_teams()),
            "goals": len(store.list_goals()),
            "projects": len(store.list_projects()),
            "mock": agents.MOCK_MODEL}


@app.get("/api/agents")
def list_agents():
    return [_member_json(m) for m in agents.list_members(store)]


@app.get("/api/leaderboard")
def leaderboard():
    """Members ranked by XP, best first. Members with no XP yet show 0."""
    xp_by_handle = {row["handle"]: row for row in store.leaderboard()}
    ranked = []
    for m in agents.list_members(store):
        xp = xp_by_handle.get(m.handle) or store.get_xp(m.handle)
        ranked.append({"name": m.name, "handle": m.handle, "colour": m.colour,
                       "emoji": m.emoji, "xp": xp["xp"], "level": xp["level"],
                       "turns": xp["turns"], "badges": xp["badges"]})
    ranked.sort(key=lambda r: (-r["xp"], -r["turns"], r["handle"]))
    return ranked


# -- rooms ----------------------------------------------------------------
@app.get("/api/rooms")
def list_rooms():
    return store.list_rooms()


@app.post("/api/rooms")
def create_room(payload: dict):
    name = (payload.get("name") or "Team Room").strip()
    return store.create_room(name)


@app.get("/api/rooms/{room_id}/messages")
def get_messages(room_id: str, after: int = 0, limit: int = 200):
    if not store.get_room(room_id):
        raise HTTPException(404, "no such room")
    return store.messages(room_id, after=after, limit=limit)


def _store_or_503(fn, what: str):
    """Run a store call, turning a store failure into a clean 503.

    A transient Turso hiccup used to surface as an uncaught ValueError
    and a 500 ("Could not send: HTTP 500"). The store retries transient
    reads internally; if it still fails, the user gets a clear,
    retryable message instead of a 500."""
    try:
        return fn()
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 — store failure -> 503, not 500
        raise HTTPException(
            503, f"the room's database hiccuped while {what} — "
                 f"please send again ({type(exc).__name__})")


@app.post("/api/rooms/{room_id}/messages")
def post_message(room_id: str, payload: dict, request: Request):
    global _run_active
    content = (payload.get("content") or "").strip()
    if not content:
        raise HTTPException(400, "message is empty")
    if not _store_or_503(lambda: store.get_room(room_id), "looking up the room"):
        raise HTTPException(404, "no such room")
    goal_id = payload.get("goal_id") or None
    if goal_id and not _store_or_503(lambda: store.get_goal(goal_id),
                                     "looking up the goal"):
        raise HTTPException(404, f"no such goal {goal_id}")
    project_id = payload.get("project_id") or None
    if project_id and not _store_or_503(lambda: store.get_project(project_id),
                                        "looking up the project"):
        raise HTTPException(404, f"no such project {project_id}")
    provider = _byok_provider(request)
    msg = _store_or_503(
        lambda: store.append_message(room_id, author="you", author_type="user",
                                     content=content, goal_id=goal_id),
        "saving your message")
    with _run_lock:
        if _run_active:
            raise HTTPException(409, "the room is already working — wait for it to finish")
        _run_active = True

    # One deadline for the whole turn, shared by the stream and the
    # work: the stream ends at it with turn_timeout, and the
    # orchestrator stops starting new work at it (plan, specialists,
    # integrate), so the turn aborts gracefully with partial results
    # instead of the hosting time limit killing the function mid-stream.
    # 45s leaves 15s of margin under Vercel's 60s function cap.
    deadline = time.monotonic() + float(
        os.environ.get("TEAMROOM_TURN_DEADLINE_S", "45"))

    def _run():
        global _run_active
        try:
            orchestrator.run_turn(store, room_id, content, model_status,
                                  goal_id=goal_id, provider=provider,
                                  deadline=deadline, project_id=project_id)
        finally:
            with _run_lock:
                _run_active = False

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    # Stream the turn as SSE: each new room message as it lands, then a
    # turn_done marker. Serverless functions freeze after the response,
    # so the turn must run inside this response — a fire-and-forget
    # background thread would be frozen and never finish.
    def gen():
        last = msg["id"] - 1  # include the user's own message, then the turn
        while True:
            try:
                new = store.messages(room_id, after=last, limit=100)
            except Exception:  # noqa: BLE001 — a failed poll must not kill
                # the stream; the next poll retries, and the deadline
                # below still ends the stream on time.
                new = []
            for m in new:
                yield f"data: {json.dumps(m)}\n\n"
                last = max(last, m["id"])
            if not thread.is_alive():
                yield 'data: {"type": "turn_done"}\n\n'
                return
            if time.monotonic() > deadline:
                yield ('data: {"type": "turn_timeout", "detail": "hosting time '
                       'limit reached — partial results above"}\n\n')
                return
            time.sleep(0.3)

    return StreamingResponse(gen(), media_type="text/event-stream")


def _byok_provider(request: Request):
    """Per-request OpenAI provider from the browser's BYOK key headers.

    The key arrives as X-OpenAI-Key / X-OpenAI-Model and lives only in
    the returned provider object, in memory, for this one turn. It is
    never stored, logged, or written to the DB. A bad model name is a
    plain 400 — the error never contains the key.
    """
    key = (request.headers.get("x-openai-key") or "").strip()
    if not key:
        return None
    model = (request.headers.get("x-openai-model") or "").strip()
    try:
        return agents.build_provider(key, model)
    except ValueError as exc:
        raise HTTPException(400, f"bad model/key: {exc}")


STREAM_MAX_AGE_S = 45.0  # a long-lived stream must end cleanly before
# the hosting time limit kills it — the browser's EventSource
# reconnects on its own, so a capped stream is just a reconnect
# cycle, while an uncapped one logs a 60s timeout every minute.


@app.get("/api/rooms/{room_id}/stream")
async def stream(room_id: str, after: int = 0, request: Request = None):
    if not store.get_room(room_id):
        raise HTTPException(404, "no such room")

    async def gen():
        last = after
        idle = 0
        started = time.monotonic()
        while True:
            if await request.is_disconnected():
                break
            if time.monotonic() - started > STREAM_MAX_AGE_S:
                # End cleanly before the hosting cap; the browser
                # reconnects and resumes from `last`.
                yield ": reconnect\n\n"
                break
            try:
                msgs = store.messages(room_id, after=last, limit=100)
            except Exception:  # noqa: BLE001 — a failed poll must not
                # kill the stream; the next poll retries.
                msgs = []
            for m in msgs:
                yield f"data: {json.dumps(m)}\n\n"
                last = max(last, m["id"])
            if not msgs:
                idle += 1
                if idle >= 30:  # ~15s of quiet -> heartbeat
                    yield ": heartbeat\n\n"
                    idle = 0
            await asyncio.sleep(0.5)

    return StreamingResponse(gen(), media_type="text/event-stream")


# -- custom members -------------------------------------------------------
@app.post("/api/members")
async def add_member(file: UploadFile | None = File(None),
                     text: str | None = Form(None)):
    raw, source = "", ""
    if file is not None:
        raw = (await file.read()).decode("utf-8", errors="replace")
        source = f"upload:{file.filename}"
    elif text and text.strip():
        raw = text
        source = "paste"
    else:
        raise HTTPException(400, "provide a skill file upload or pasted text")
    try:
        parsed = parse_skill_text(raw)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if agents.member_by_handle(store, parsed["handle"]):
        raise HTTPException(400,
                            f"a member with handle @{parsed['handle']} already exists")
    try:
        member = store.add_member(source=source, **parsed)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return member


@app.delete("/api/members/{handle}")
def delete_member(handle: str):
    member = store.get_member(handle)
    if member is None:
        raise HTTPException(404, f"no custom member @{handle}")
    store.delete_member(handle)
    return {"deleted": handle}


# -- teams ----------------------------------------------------------------
@app.get("/api/teams")
def list_teams():
    return store.list_teams()


@app.post("/api/teams")
def create_team(payload: dict):
    name = (payload.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "a team needs a name")
    handles = _validate_handles(payload.get("members"))
    return store.create_team(name, handles)


@app.get("/api/teams/{team_id}")
def get_team(team_id: str):
    team = store.get_team(team_id)
    if not team:
        raise HTTPException(404, "no such team")
    return team


@app.post("/api/teams/{team_id}/members")
def add_team_member(team_id: str, payload: dict):
    if not store.get_team(team_id):
        raise HTTPException(404, "no such team")
    handle = _validate_handles([payload.get("handle")])
    if not handle:
        raise HTTPException(400, "provide a member handle")
    store.add_team_member(team_id, handle[0])
    return store.get_team(team_id)


@app.delete("/api/teams/{team_id}/members/{handle}")
def remove_team_member(team_id: str, handle: str):
    if not store.get_team(team_id):
        raise HTTPException(404, "no such team")
    if not store.remove_team_member(team_id, handle):
        raise HTTPException(404, f"@{handle} is not on this team")
    return store.get_team(team_id)


# -- goals ----------------------------------------------------------------
@app.get("/api/goals")
def list_goals():
    return store.list_goals()


@app.post("/api/goals")
def create_goal(payload: dict):
    title = (payload.get("title") or "").strip()
    team_id = payload.get("team_id") or ""
    if not title:
        raise HTTPException(400, "a goal needs a title")
    if not store.get_team(team_id):
        raise HTTPException(404, "no such team — pick the goal's team first")
    handles = _validate_handles(payload.get("members"))
    try:
        return store.create_goal(team_id, title,
                                 (payload.get("description") or "").strip(), handles,
                                 target_tasks=payload.get("target_tasks") or 5)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/goals/{goal_id}")
def get_goal(goal_id: str):
    goal = store.get_goal(goal_id)
    if not goal:
        raise HTTPException(404, "no such goal")
    return goal


@app.patch("/api/goals/{goal_id}")
def update_goal(goal_id: str, payload: dict):
    if not store.get_goal(goal_id):
        raise HTTPException(404, "no such goal")
    try:
        return store.set_goal_status(goal_id, payload.get("status") or "")
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.delete("/api/goals/{goal_id}")
def delete_goal(goal_id: str):
    if not store.delete_goal(goal_id):
        raise HTTPException(404, "no such goal")
    return {"deleted": goal_id}


# -- projects ---------------------------------------------------------------
# A project is a real workspace: its files live in the store (Turso on
# Vercel), so they survive restarts and cold starts. When a turn runs
# with a project active, the agents' file tools write into it, and the
# Leader's plan includes a step that writes the deliverable files.
@app.get("/api/projects")
def list_projects():
    return store.list_projects()


@app.post("/api/projects")
def create_project(payload: dict):
    name = (payload.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "a project needs a name")
    try:
        return store.create_project(
            name, (payload.get("description") or "").strip(),
            goal_id=payload.get("goal_id") or None,
            team_id=payload.get("team_id") or None)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/projects/{project_id}")
def get_project(project_id: str):
    project = store.get_project(project_id)
    if not project:
        raise HTTPException(404, "no such project")
    return project


@app.get("/api/projects/{project_id}/files")
def list_project_files(project_id: str):
    if not store.get_project(project_id):
        raise HTTPException(404, "no such project")
    return store.list_project_files(project_id)


@app.get("/api/projects/{project_id}/files/{path:path}")
def get_project_file(project_id: str, path: str):
    if not store.get_project(project_id):
        raise HTTPException(404, "no such project")
    try:
        row = store.get_project_file(project_id, path)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if row is None:
        raise HTTPException(404, f"no such file in project: {path}")
    return row


@app.put("/api/projects/{project_id}/files/{path:path}")
def put_project_file(project_id: str, path: str, payload: dict):
    if not store.get_project(project_id):
        raise HTTPException(404, "no such project")
    try:
        return store.upsert_project_file(project_id, path,
                                         payload.get("content") or "")
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/projects/{project_id}/zip")
def download_project_zip(project_id: str):
    project = store.get_project(project_id)
    if not project:
        raise HTTPException(404, "no such project")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in store.list_project_files(project_id):
            row = store.get_project_file(project_id, f["path"])
            if row is not None:
                zf.writestr(row["path"], row["content"])
    buf.seek(0)
    filename = "".join(c if c.isalnum() or c in "-_." else "-"
                       for c in project["name"].strip().lower().replace(" ", "-")
                       ) or "project"
    return StreamingResponse(buf, media_type="application/zip",
                             headers={"Content-Disposition":
                                      f'attachment; filename="{filename}.zip"'})


@app.post("/api/projects/{project_id}/github")
def create_github_repo(project_id: str, payload: dict | None = None):
    """Create a GitHub repo for the project and push its files.

    Needs GITHUB_TOKEN in the environment (a repo-scoped token) — the
    endpoint reads it for this one call only and never stores it. The
    repo is private by default; pass {"private": false} to make it
    public. No MCP involved: plain GitHub REST API calls.
    """
    project = store.get_project(project_id)
    if not project:
        raise HTTPException(404, "no such project")
    body = payload or {}
    token = (os.environ.get("GITHUB_TOKEN") or "").strip()
    files = []
    for f in store.list_project_files(project_id):
        row = store.get_project_file(project_id, f["path"])
        if row is not None:
            files.append({"path": row["path"], "content": row["content"]})
    try:
        return github.create_repo_and_push(
            token, body.get("name") or project["name"],
            project.get("description") or "", files,
            private=body.get("private", True))
    except github.GitHubError as exc:
        # A missing token is a setup problem, not a server error — say
        # so plainly. Other GitHub failures are reported as-is.
        status = 400 if "GITHUB_TOKEN" in str(exc) else 502
        raise HTTPException(status, str(exc))


# -- team-scoped member memory --------------------------------------------
@app.get("/api/teams/{team_id}/memory")
def get_memory(team_id: str, handle: str | None = None, limit: int = 50):
    if not store.get_team(team_id):
        raise HTTPException(404, "no such team")
    return store.list_memory(team_id, handle=handle, limit=limit)


@app.post("/api/teams/{team_id}/memory")
def add_memory(team_id: str, payload: dict):
    if not store.get_team(team_id):
        raise HTTPException(404, "no such team")
    handle = (payload.get("handle") or "").strip().lower()
    if handle not in _known_handles():
        raise HTTPException(400, f"no member with handle @{handle}")
    content = (payload.get("content") or "").strip()
    if not content:
        raise HTTPException(400, "memory content is empty")
    try:
        return store.record_memory(team_id, handle,
                                   payload.get("kind") or "did", content)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


# -- shared best practices -------------------------------------------------
@app.get("/api/best-practices")
def list_best_practices():
    return store.list_best_practices()


@app.post("/api/best-practices")
def add_best_practice(payload: dict):
    try:
        created = store.add_best_practice(
            payload.get("title") or "", payload.get("content") or "",
            payload.get("source_team_id"), payload.get("source_handle"))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if created is None:
        return {"recorded": False, "note": "an identical best practice already exists"}
    return created


@app.delete("/api/best-practices/{practice_id}")
def delete_best_practice(practice_id: int):
    if not store.delete_best_practice(practice_id):
        raise HTTPException(404, "no such best practice")
    return {"deleted": practice_id}


# -- entrypoint -----------------------------------------------------------
def main(argv=None):
    import argparse
    import uvicorn

    ap = argparse.ArgumentParser(prog="teamroom")
    ap.add_argument("--port", type=int, default=int(os.environ.get("TEAMROOM_PORT", "8765")))
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args(argv)
    print(f"Looper Team Room on http://{args.host}:{args.port}/")
    if model_status.get("mock"):
        print("Model: MOCK (TEAMROOM_MOCK_MODEL=1 — scripted replies, verification only)")
    elif model_status.get("configured"):
        print(f"Model: {model_status['model']}")
    else:
        print(f"Model: NOT CONFIGURED — {model_status.get('error')}")
        print("  Set LOOPER_API_KEY (or LOOPER_MODEL, e.g. ollama/llama3.1) and restart.")
    print(f"Data: {'Turso ' + store.turso_url if store.backend == 'turso' else TEAMROOM_HOME / 'teamroom.db'} (backend: {store.backend})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
