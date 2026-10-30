#!/usr/bin/env bash
set -euo pipefail
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${ISAAC_PYTHON:?Set ISAAC_PYTHON to the Python executable for pip-based Isaac Sim 6.2}"
solver_site="${TITUS_SOLVER_SITE:-$task_dir/.solver/site}"
"$ISAAC_PYTHON" -c 'import importlib.metadata as m; assert m.version("isaacsim")=="6.2.0.0"; assert m.version("warp-lang")=="1.17.0"'
if [[ ! -d "$solver_site/newton" ]]; then
  if "$ISAAC_PYTHON" -m pip --version >/dev/null 2>&1; then
    "$ISAAC_PYTHON" -m pip install --target "$solver_site" --no-deps 'newton==1.7.0' 'mujoco==3.15.0' 'mujoco-warp==3.15.0'
  elif command -v uv >/dev/null 2>&1; then
    uv pip install --python "$ISAAC_PYTHON" --target "$solver_site" --no-deps 'newton==1.7.0' 'mujoco==3.15.0' 'mujoco-warp==3.15.0'
  else
    printf 'Install pip or uv to create the isolated solver overlay.\n' >&2
    exit 1
  fi
fi
exec "$ISAAC_PYTHON" -I -u -B "$task_dir/newton170/runtime/launch.py" "$solver_site" "$task_dir/preflight.py"
