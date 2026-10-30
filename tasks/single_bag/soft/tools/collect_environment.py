import importlib.metadata as m
import json
from pathlib import Path
import platform
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
packages = [
    'isaacsim',
    'newton',
    'warp-lang',
    'mujoco',
    'mujoco-warp',
    'numpy',
    'scipy',
    'torch',
    'Pillow',
    'usd-core',
]
versions = {}
for name in packages:
    try:
        versions[name] = m.version(name)
    except m.PackageNotFoundError:
        versions[name] = None
report = dict(
    python=sys.version,
    platform=platform.platform(),
    machine=platform.machine(),
    packages=versions,
    gpu=subprocess.check_output(
        ['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'], text=True
    ).strip(),
    robot_engine='Isaac Sim native PhysX TGS',
    bag_engine='Newton 1.7 VBD',
    physics_dt_s=1 / 600,
    qualification_scope='Fixed reference case; calibration parameters preserved; effective bag material model',
)
(root / 'runtime-lock.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
