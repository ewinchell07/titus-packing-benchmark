"""Run the task's independent physical scorer."""

import os
from pathlib import Path
import subprocess

from .paths import backend_root


def evaluate(task, backend, episode):
    root = backend_root(task, backend)
    episode = Path(episode).resolve(strict=True)
    python = os.environ.get("ISAAC_PYTHON")
    if not python:
        raise RuntimeError("Set ISAAC_PYTHON to the pinned simulator Python")
    return subprocess.run([python, "-I", "-u", str(root / "score_pack.py"), str(episode)], cwd=root).returncode
