#!/usr/bin/env bash
set -euo pipefail
adapter_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${GRASPGENX_DIR:?Set GRASPGENX_DIR to a new directory outside the Isaac environment}"
if [[ -e "$GRASPGENX_DIR" ]]; then
  echo 'Use a new directory; existing environments are not overwritten.' >&2
  exit 1
fi
command -v uv >/dev/null
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/NVlabs/GraspGenX.git "$GRASPGENX_DIR"
git -C "$GRASPGENX_DIR" checkout --detach b9429097728cb1c430dd78b92edf17ba318aad03
(cd "$GRASPGENX_DIR" && uv sync --locked --python 3.11 --extra serve)
uv pip install --python "$GRASPGENX_DIR/.venv/bin/python" usd-core==26.11
"$GRASPGENX_DIR/.venv/bin/python" "$adapter_dir/download_models.py" "$GRASPGENX_DIR/checkpoints"
