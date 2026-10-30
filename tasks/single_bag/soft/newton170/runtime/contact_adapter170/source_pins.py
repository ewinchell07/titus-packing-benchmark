"""Fail-closed exact-source contract for the owned Newton 1.7.0 adapter."""

from pathlib import Path
import hashlib
import importlib.metadata
import newton

PINNED_SOURCES = {
    '_src/geometry/kernels.py': 'ab58040a7ca391ff15816692a4979ea116e3cc2cc29964d8f3ca05c2a7f7a9be',
    '_src/geometry/tri_mesh_collision.py': '79033a24d8e8981cdd0145f47837e3b08e306662759ebfedb0b6b81182bde627',
    '_src/solvers/vbd/solver_vbd.py': '52d9de6f8cc662a95b4c55eaa657c3eedf5f6a6da699109dd9c2de36e36b932f',
    '_src/sim/collide.py': '186c92e89ad70dc9fe54f58cb8669dde5c878758a26c6c6afd65360f3a04002a',
    '_src/sim/contacts.py': '508067733fb71e9824da4fe539950f42d3a1ce72972f87eebbc5b4a83a78fcba',
    '_src/solvers/vbd/rigid_vbd_kernels.py': '8f0dc0fd50477025f461a9398f4a51f7dd8a499ae806230be82bc7e00fd1a412',
    '_src/geometry/narrow_phase.py': '2cee9b6f4b41eb36e7ca11d8ad64859ae927de6700e9d46f2cc84d4b1b4478c0',
}
PINNED_VERSIONS = {"newton": "1.7.0", "warp-lang": "1.17.0"}


def verify_source_pins():
    versions = {name: importlib.metadata.version(name) for name in PINNED_VERSIONS}
    if versions != PINNED_VERSIONS:
        raise RuntimeError(f"Adapter dependency mismatch: {versions!r}; expected {PINNED_VERSIONS!r}")
    root = Path(newton.__file__).resolve().parent
    actual = {rel: hashlib.sha256((root / rel).read_bytes()).hexdigest() for rel in PINNED_SOURCES}
    if actual != PINNED_SOURCES:
        raise RuntimeError(f"Installed Newton source differs from pinned 1.7.0 adapter: {actual!r}")
    return actual
