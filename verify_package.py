"""Verify supplied files without importing Isaac, Newton or any dependencies.

Generated runs and caches are ignored. This checks package integrity, not
simulation success or the authenticity of an independently supplied manifest.
"""

import hashlib
import json
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text())
    problems = []
    for name, expected in manifest["files_sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            problems.append({"file": name, "reason": "missing or outside package"})
            continue
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            problems.append({"file": name, "reason": "hash differs"})
    print(json.dumps({"passed": not problems, "release": manifest["release"],
                      "files_checked": len(manifest["files_sha256"]),
                      "problems": problems}, indent=2))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
