"""Team Room package — a web chat room of looper specialists.

A room is a shared session: every agent in it reads and writes the
same SQLite message log (same pattern as the looper's SessionStore —
one file, WAL, one row per message). Custom members persist in the
same file, so they survive restarts.

Layout:
  store.py        SQLite room log + custom members
  skills.py       SKILL.md-style parsing for custom members
  agents.py       built-in specialists + member registry + Looper wiring
  orchestrator.py routing (keywords / @-mentions) + parallel runs + synthesis
  server.py       FastAPI app: UI, REST API, SSE stream
  static/         single-page chat UI, no build step
"""
from __future__ import annotations

__version__ = "0.1.0"
