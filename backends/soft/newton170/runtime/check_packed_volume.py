"""Record the effective packed-volume repair against the original printed bag."""

from pathlib import Path
import hashlib
import json
import numpy as np
import trimesh
from pxr import Usd, UsdGeom

root = Path(__file__).resolve().parents[2]
mesh_path = root / "newton170/config/packed_tets.npz"
source_path = root / "newton170/config/source_tets.npz"
z = np.load(mesh_path)
p = z["points"].astype(float)
t = z["tets"].astype(int)
v = p[t]
m = np.stack((v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], v[:, 3] - v[:, 0]), axis=2)
assert np.all(np.linalg.det(m) > 0)
faces = np.concatenate([t[:, i] for i in [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]])
_, index, count = np.unique(np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True)
assert np.all(count <= 2)
surface = trimesh.Trimesh(p, faces[index[count == 1]], process=False)
stage = Usd.Stage.Open(str(root / "assets/bag_source/bag.usdc"))
skin = np.asarray(UsdGeom.Mesh(stage.GetPrimAtPath("/Bag/Visual/surface")).GetPointsAttr().Get(), float)
_, distances, _ = trimesh.proximity.closest_point(surface, skin)
report = dict(
    representation="Fully dynamic effective packed-content volume; no carrier or attachments",
    source_tets_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
    packed_tets_sha256=hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
    mass_kg=0.036,
    nodes=len(p),
    tets=len(t),
    volume_m3=float(np.linalg.det(m).sum() / 6),
    tet_condition_percentiles=np.quantile(np.linalg.cond(m), [0.5, 0.95, 0.99, 1]).tolist(),
    original_skin_to_packed_boundary_m=dict(
        p50=float(np.median(distances)), p95=float(np.quantile(distances, 0.95)), max=float(distances.max())
    ),
    original_printed_surface_changed=False,
    original_robot_changed=False,
    material=dict(E_pa=50000, nu=0.35, mu=0.5, k_damp=0.002),
    qualification="Effective contents model, not measured material identification. Native grasp task remains to be tested.",
)
(root / "newton170/config/packed_volume.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
