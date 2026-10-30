"""Bind the unchanged printed skin to the free packed FEM surface."""

from pathlib import Path
import hashlib
import json
import numpy as np
import trimesh
from pxr import Usd, UsdGeom

root = Path(__file__).resolve().parents[2]
z = np.load(root / "newton170/config/packed_tets.npz")
p, t = z["points"].astype(float), z["tets"].astype(int)
faces = np.concatenate([t[:, i] for i in [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]])
_, index, count = np.unique(np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True)
faces = faces[index[count == 1]]
surface = trimesh.Trimesh(p, faces, process=False)
stage = Usd.Stage.Open(str(root / "assets/bag_source/bag.usdc"))
skin = np.asarray(UsdGeom.Mesh(stage.GetPrimAtPath("/Bag/Visual/surface")).GetPointsAttr().Get(), float)
closest, distance, triangle = trimesh.proximity.closest_point(surface, skin)
ids = faces[triangle]
tri = p[ids]
weights = trimesh.triangles.points_to_barycentric(tri, closest)
e1 = tri[:, 1] - tri[:, 0]
normal = np.cross(e1, tri[:, 2] - tri[:, 0])
e1 /= np.linalg.norm(e1, axis=1)[:, None]
normal /= np.linalg.norm(normal, axis=1)[:, None]
e2 = np.cross(normal, e1)
offset = skin - closest
frame_offset = np.stack([np.sum(offset * a, axis=1) for a in [e1, e2, normal]], axis=1)
restored = np.einsum("ni,nij->nj", weights, tri) + sum(
    frame_offset[:, i, None] * a for i, a in enumerate([e1, e2, normal])
)
error = float(np.abs(restored - skin).max())
assert error < 1e-12
np.savez_compressed(
    root / "newton170/config/packed_skin_binding.npz",
    triangle_vertex_ids=ids,
    triangle_weights=weights,
    triangle_frame_offsets=frame_offset,
)
report = dict(
    vertices=len(skin),
    rest_reconstruction_max_abs_m=error,
    p95_boundary_offset_m=float(np.quantile(distance, 0.95)),
    max_boundary_offset_m=float(distance.max()),
    tet_sha256=hashlib.sha256((root / "newton170/config/packed_tets.npz").read_bytes()).hexdigest(),
    binding="Nearest surface triangle with local-frame offset; original topology, UVs and printing retained",
    physical_state_writes=0,
)
(root / "newton170/config/packed_skin_binding.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
