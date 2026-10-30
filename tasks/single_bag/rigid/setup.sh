#!/usr/bin/env bash
set -euo pipefail
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${ISAAC_PYTHON:?Set ISAAC_PYTHON to the Python executable for pip-based Isaac Sim 6.2}"
exec "$ISAAC_PYTHON" -I -u -B "$task_dir/preflight.py"
