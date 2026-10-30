"""Launch the task-specific reference controller and retain its identity."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

from .paths import backend_root, task_info


def run_reference(task, backend, out, *, stop_after=None, no_close=False):
    root = backend_root(task, backend)
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(f"Choose a new output directory: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    if "ISAAC_PYTHON" not in os.environ:
        raise RuntimeError("Set ISAAC_PYTHON to the pinned Isaac Sim 6.2 Python")
    command = ["bash", str(root / "run_pack.sh"), "--out", str(out)]
    if stop_after:
        command += ["--stop-after", stop_after]
    if no_close:
        command += ["--no-close"]
    receipt = dict(
        task=task, task_status=task_info(task)["status"], backend=backend,
        controller="privileged reference", command=command,
        controller_sha256=hashlib.sha256((root / "pack_bag.py").read_bytes()).hexdigest(),
        qualification="Read acceptance.json; a diagnostic exit does not qualify packing",
    )
    result = subprocess.run(command, cwd=root)
    receipt["exit_code"] = result.returncode
    if out.is_dir():
        (out / "launch.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return result.returncode
