#!/usr/bin/env bash
set -euo pipefail
: "${ISAAC_PYTHON:?Set ISAAC_PYTHON to Isaac Sim 6.2 Python}"
export OMNI_KIT_ACCEPT_EULA=YES
exec "$ISAAC_PYTHON" -u -B "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/pack_bag.py" "$@"
