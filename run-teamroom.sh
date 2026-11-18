#!/bin/bash
# Start the Looper Team Room (local only, http://127.0.0.1:8765/).
# Uses the teamroom venv (.venv-teamroom) so the looper's stdlib-only
# system install is untouched. Model via LOOPER_MODEL / LOOPER_API_KEY.
set -e
cd "$(dirname "$0")"
if [ ! -d .venv-teamroom ]; then
  python3 -m venv .venv-teamroom
  .venv-teamroom/bin/pip install -q -r teamroom/requirements.txt
fi
exec .venv-teamroom/bin/python -m teamroom "$@"
