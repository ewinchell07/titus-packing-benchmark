#!/usr/bin/env bash
set -euo pipefail
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${ISAAC_PYTHON:?Set ISAAC_PYTHON to the Isaac Sim 6.2 Python executable}"
export OMNI_KIT_ACCEPT_EULA=YES
exec "$ISAAC_PYTHON" -I -u -B "$task_dir/newton170/runtime/launch.py" \
  "${TITUS_SOLVER_SITE:-$task_dir/.solver/site}" "$task_dir/pack_bag.py" "$@"
