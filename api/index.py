"""Vercel entrypoint — serves the Team Room FastAPI app.

Local runs are unaffected (./run-teamroom.sh). On Vercel the project
root is on sys.path, so `teamroom` and `looper` import normally.
State is persistent when TURSO_URL / TURSO_AUTH_TOKEN are set (the
store talks to Turso); without them it falls back to a SQLite file
in /tmp (TEAMROOM_HOME), which is ephemeral per instance — see
teamroom/README.md "Deploying to Vercel".
"""
from teamroom.server import app
