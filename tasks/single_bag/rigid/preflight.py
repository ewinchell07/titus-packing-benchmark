"""Static package/asset checks. Physical success is recorded separately."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

ROOT = Path(__file__).resolve().parent
expected = {"isaacsim": "6.2.0.0"}
actual = {name: importlib.metadata.version(name) for name in expected}
if actual != expected:
    raise RuntimeError(f"Runtime versions differ: {actual}")
if (ROOT / "manifest.json").exists():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    mismatches = [
        name
        for name, digest in manifest.items()
        if not (ROOT / name).is_file() or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest
    ]
    if mismatches:
        raise RuntimeError(f"Package files differ: {mismatches}")
audit = json.loads((ROOT / "scene/scene_lock.json").read_text())
for name, digest in audit["calibrated_robot_files"].items():
    if hashlib.sha256((ROOT / "assets/robot" / name).read_bytes()).hexdigest() != digest:
        raise RuntimeError(f"Delivered robot component changed: {name}")
stage = Usd.Stage.Open(str(ROOT / "scene/rigid_bag_box.usda"))
assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0 and UsdGeom.GetStageUpAxis(stage) == 'Z'
assert stage.GetEndTimeCode() >= 3600000.0 and stage.GetTimeCodesPerSecond() == 600.0
assert stage and stage.GetPrimAtPath("/World/Bag").IsActive()
assert stage.GetPrimAtPath("/World/DashCarton").IsActive()
assert not stage.GetPrimAtPath('/World/Table')
assert (
    UsdPhysics.CollisionAPI(stage.GetPrimAtPath('/World/RotaryTableV2/dish_collision'))
    .GetCollisionEnabledAttr()
    .Get()
)
assert (
    sum(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in Usd.PrimRange(stage.GetPrimAtPath("/World/DashCarton")))
    == 1
)
bag = stage.GetPrimAtPath('/World/Bag')
assert UsdPhysics.RigidBodyAPI(bag).GetRigidBodyEnabledAttr().Get()
assert not UsdPhysics.RigidBodyAPI(bag).GetKinematicEnabledAttr().Get()
assert abs(UsdPhysics.MassAPI(bag).GetMassAttr().Get() - 0.036) < 1e-6
receipt = json.loads((ROOT / 'geometry/coacd_receipt.json').read_text())
assert (
    sum(
        1
        for p in Usd.PrimRange(bag)
        if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()
    )
    == receipt['hull_count']
)
images = set()
for prim in stage.Traverse():
    for attr in prim.GetAttributes():
        value = attr.Get()
        if isinstance(value, Sdf.AssetPath) and Path(value.path).suffix.lower() in [
            ".png",
            ".jpg",
            ".jpeg",
            ".tif",
            ".exr",
        ]:
            if not value.resolvedPath or not Path(value.resolvedPath).is_file():
                raise RuntimeError(f"Missing image at {attr.GetPath()}: {value.path}")
            images.add(value.resolvedPath)
z = np.load(ROOT / "geometry/packed_tets.npz")
p, t = z["points"], z["tets"]
v = p[t]
assert np.all(np.einsum("ij,ij->i", v[:, 1] - v[:, 0], np.cross(v[:, 2] - v[:, 0], v[:, 3] - v[:, 0])) > 0)
print(
    json.dumps(
        dict(
            passed=True,
            scope="Static integrity and asset checks only",
            versions=actual,
            calibrated_robot_files=len(audit["calibrated_robot_files"]),
            resolved_images=len(images),
            bag_observation_landmarks=len(p),
            source_envelope_tetrahedra=len(t),
            physical_acceptance=False,
        ),
        indent=2,
    )
)
