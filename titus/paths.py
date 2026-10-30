"""Resolve declared tasks and backends without importing simulator libraries."""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ("rigid", "soft")


def _link_shared(root):
    """Give a backend/task folder its view of the single shared assets and solver."""
    shared = {"assets": ROOT / "assets"}
    if (root / "newton170").is_dir():
        shared[".solver"] = ROOT / "solver"
        shared[".solver"].mkdir(exist_ok=True)  # Setup installs the overlay here if absent.
    for name, target in shared.items():
        link = root / name
        if not link.exists() and not link.is_symlink():
            link.symlink_to(os.path.relpath(target, root), target_is_directory=True)
    return root


def tasks():
    return json.loads((ROOT / "tasks/catalog.json").read_text())


def task_info(task):
    catalog = tasks()
    if task not in catalog:
        raise ValueError(f"Unknown task {task!r}; choose from {', '.join(catalog)}")
    return dict(catalog[task])


def backend_root(task, backend):
    task_info(task)
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}; choose rigid or soft")
    return _link_shared(ROOT / "tasks" / task / backend)


def population_backend_root(backend):
    if backend not in BACKENDS:
        raise ValueError("Choose rigid or soft")
    return _link_shared(ROOT / "backends" / backend)
