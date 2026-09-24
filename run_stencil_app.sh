#!/bin/bash
# Launch the GUI with a local virtualenv when one exists, otherwise python3.
ROOT="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$ROOT/stencil_app.py"

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
elif [[ -x "$ROOT/venv/bin/python" ]]; then
  PY="$ROOT/venv/bin/python"
else
  PY="$(command -v python3 || true)"
fi

if [[ -z "$PY" ]]; then
  echo "python3 not found. Install Python 3 or create .venv in $ROOT." >&2
  exit 1
fi

cd "$ROOT" || exit 1
exec "$PY" "$SCRIPT" "$@"
