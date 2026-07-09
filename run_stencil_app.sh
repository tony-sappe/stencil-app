#!/bin/bash

# Full path to your venv's Python (avoids activation issues in non-interactive shells)
VENV_PYTHON="/path/to/your/project/venv/bin/python"

# Full path to your script
SCRIPT="/path/to/your/project/stencil_app.py"

cd "$(dirname "$SCRIPT")" || exit 1

exec "$VENV_PYTHON" "$SCRIPT"
