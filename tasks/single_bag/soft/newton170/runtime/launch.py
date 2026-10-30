"""Select the isolated, pinned solver packages before importing the task."""

from pathlib import Path
import runpy
import sys

site, script = [Path(arg).resolve(strict=True) for arg in sys.argv[1:3]]
sys.path.insert(0, str(site))
sys.path.insert(1, str(script.parent))
sys.argv = sys.argv[2:]
runpy.run_path(str(script), run_name="__main__")
