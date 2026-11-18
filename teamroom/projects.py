"""Projects — a real workspace for the team's work.

A project is a set of files stored in the room store (Turso on
Vercel, SQLite locally), so they survive restarts and cold starts —
unlike anything an agent writes to the ephemeral Vercel filesystem.
The agents' file tools (list_files / read_file / write_file) operate
on the active project's files: when a turn runs with a project, the
same tool names the model already knows write into the project
instead of the local disk.

How it works: the active project lives in a ContextVar
(`active_project`), set for the duration of a turn by the
orchestrator (and by `run_member`, which is what the parallel path's
pool threads need). The looper's tool dispatch copies the caller's
context into each tool worker (see looper/loop.py), so the ContextVar
reaches the tool handlers even though dispatch runs in a pool. The
three file tools are re-registered here with project-aware handlers
that delegate to the original filesystem implementations when no
project is active — the looper core itself is untouched, and the
looper CLI never imports this module, so its tools stay
filesystem-based.

The `terminal` tool is deliberately NOT project-scoped: it runs shell
commands on the host, and project files live in the store, not on
disk — there is nothing for a shell command to run against on Vercel.
"""
from __future__ import annotations

import contextlib
import json
from contextvars import ContextVar
from dataclasses import dataclass

from looper import tools as _fs_tools  # noqa: F401 — import registers the file tools
from looper.registry import registry

from .store import RoomStore, sanitize_project_path

_PROJECT_TOOL_DESCRIPTIONS = {
    "list_files": ("List files in a directory (relative paths, max 100). "
                   "When a project is active, lists the project's files; "
                   "otherwise lists the local filesystem."),
    "read_file": ("Read a text file. Returns content (truncated at 20k "
                  "chars). When a project is active, reads from the "
                  "project's files; otherwise from the local filesystem."),
    "write_file": ("Write text to a file, creating parent dirs. Returns "
                   "bytes written. When a project is active, writes into "
                   "the project's files (paths like main.py) — the file "
                   "persists in the project; otherwise writes to the local "
                   "filesystem."),
}


@dataclass
class ProjectWorkspace:
    """A project's files, backed by the room store. The file tools
    call these methods when this project is the active one."""
    store: RoomStore
    project_id: str
    name: str = ""

    def list_files(self, path: str = ".", recursive: int = 0) -> str:
        prefix = _dir_prefix(path)
        if prefix is None:
            return json.dumps({"error": f"not a directory: {path}"})
        files = [f["path"] for f in self.store.list_project_files(self.project_id)]
        if prefix:
            files = [f for f in files if f.startswith(prefix + "/")]
        if not recursive:
            # only files directly under the prefix, not in subdirectories
            if prefix:
                files = [f for f in files
                         if "/" not in f[len(prefix) + 1:]]
            else:
                files = [f for f in files if "/" not in f]
        return json.dumps({"path": path, "files": sorted(files)[:100]})

    def read_file(self, path: str) -> str:
        try:
            row = self.store.get_project_file(self.project_id, path)
        except ValueError as exc:
            return json.dumps({"error": str(exc)})
        if row is None:
            return json.dumps({"error": f"no such file in project: {path}"})
        text = row["content"]
        return json.dumps({"path": row["path"], "content": text[:20000],
                           "truncated": len(text) > 20000})

    def write_file(self, path: str, content: str) -> str:
        try:
            row = self.store.upsert_project_file(self.project_id, path, content)
        except ValueError as exc:
            return json.dumps({"error": str(exc)})
        return json.dumps({"path": row["path"],
                           "bytes": len((content or "").encode())})


def _dir_prefix(path: str) -> str | None:
    """Normalise a list_files directory argument to a path prefix.
    Returns "" for the project root, or None for an invalid path."""
    raw = (path or "").strip().replace("\\", "/")
    if raw.startswith("/") or raw in ("", "."):
        if raw.startswith("/"):
            return None
        return ""
    parts = [p for p in raw.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts):
        return None
    return "/".join(parts)


_ACTIVE_PROJECT: ContextVar[ProjectWorkspace | None] = ContextVar(
    "teamroom_active_project", default=None)


@contextlib.contextmanager
def active_project(workspace: ProjectWorkspace | None):
    """Set the active project workspace for this thread (and, via the
    looper's context-propagating tool dispatch, for tool calls made
    under it). Always resets on exit, so a reused pool thread never
    leaks one turn's project into the next."""
    token = _ACTIVE_PROJECT.set(workspace)
    try:
        yield workspace
    finally:
        _ACTIVE_PROJECT.reset(token)


def current_project() -> ProjectWorkspace | None:
    return _ACTIVE_PROJECT.get()


def resolve_project(store: RoomStore, project_id: str | None,
                    goal_id: str | None = None) -> ProjectWorkspace | None:
    """The workspace for a turn: the explicitly named project wins;
    otherwise the goal's linked project (most recent) if the turn has
    a goal. Returns None when there is no project for this turn."""
    project = None
    if project_id:
        project = store.get_project(project_id)
    elif goal_id:
        project = store.get_project_for_goal(goal_id)
    if not project:
        return None
    return ProjectWorkspace(store, project["id"], project["name"])


# --- project-aware file tools -------------------------------------------
# Same names, same schemas as the looper's file tools — the model calls
# list_files / read_file / write_file exactly as before. When a project
# is active the calls land in the project store; otherwise they fall
# back to the original filesystem implementations.

def _project_list_files(path: str = ".", recursive: int = 0) -> str:
    ws = current_project()
    if ws is None:
        return _fs_tools.list_files(path=path, recursive=recursive)
    return ws.list_files(path=path, recursive=recursive)


def _project_read_file(path: str) -> str:
    ws = current_project()
    if ws is None:
        return _fs_tools.read_file(path=path)
    return ws.read_file(path=path)


def _project_write_file(path: str, content: str) -> str:
    ws = current_project()
    if ws is None:
        return _fs_tools.write_file(path=path, content=content)
    return ws.write_file(path=path, content=content)


_PROJECT_HANDLERS = {
    "list_files": _project_list_files,
    "read_file": _project_read_file,
    "write_file": _project_write_file,
}


def install_project_tools() -> None:
    """Re-register the three file tools with the project-aware
    handlers, keeping their original schemas (only the description
    gains the project note). Idempotent — safe to call more than once.
    """
    for name, handler in _PROJECT_HANDLERS.items():
        entry = registry.get(name)
        if entry is None:
            continue
        description = _PROJECT_TOOL_DESCRIPTIONS[name]
        schema = {**entry.schema, "description": description}
        registry.register(name=name, toolset="file", schema=schema,
                          handler=handler, description=description,
                          emoji=entry.emoji, override=True)


install_project_tools()
