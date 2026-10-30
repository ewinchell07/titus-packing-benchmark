"""Render recorded passing physics states; this viewer does not simulate a task."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("episode", type=Path)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--fps", type=int, default=10)
args = parser.parse_args()
args.episode = args.episode.resolve(strict=True)
args.out = args.out.resolve()
report = json.loads((args.episode / "report.json").read_text())
if not report.get("task_passed"):
    raise RuntimeError("Render a physically accepted packing episode")
args.out.mkdir(parents=True, exist_ok=False)
from isaacsim import SimulationApp

app = SimulationApp(
    {
        "headless": True,
        "fast_shutdown": True,
        "multi_gpu": False,
        "active_gpu": 0,
        "extra_args": ["--/UJITSO/enabled=false"],
    }
)
import numpy as np
import omni.usd
import omni.replicator.core as rep
from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt
from PIL import Image, ImageDraw

sys.path.insert(0, str(ROOT / "runtime"))
from visuals import add_camera_lights, Visuals

context = omni.usd.get_context()
context.open_stage(str(ROOT / "scene/rigid_bag_box.usda"))
stage = context.get_stage()
stage.SetEditTarget(stage.GetSessionLayer())
cache = UsdGeom.XformCache()
states = np.load(args.episode / "states.npz")
labels = list(states["body_labels"])
scales = {
    label: Gf.Transform(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(str(label)))).GetScale()
    for label in labels
}
for prim in list(stage.Traverse()):
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
    if prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
    if prim.IsA(UsdPhysics.Joint):
        prim.SetActive(False)
add_camera_lights(stage)
mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Bag/Visual/surface"))
mesh.MakeMatrixXform().Set(Gf.Matrix4d(1.0))
mesh.SetResetXformStack(True)
skin = object.__new__(Visuals)
skin.binding = np.load(ROOT / "geometry/packed_skin_binding.npz")
binding_audit = json.loads((ROOT / "geometry/packed_skin_binding.json").read_text())
assert binding_audit["tet_sha256"] == report["tet_mesh_sha256"]
product = rep.create.render_product("/World/InspectionCamera", (1024, 768))
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach([product])
for _ in range(4):
    app.update()
rows = json.loads((args.episode / "telemetry.json").read_text())
sample_times = states["t"]
provenance = []
failed = False
try:
    for frame in range(round(sample_times[-1] * args.fps) + 1):
        index = int(np.argmin(abs(sample_times - frame / args.fps)))
        poses, nodes = states["body_q"][index], states["nodes"][index]
        for label, pose in zip(labels, poses):
            prim = stage.GetPrimAtPath(str(label))
            matrix = Gf.Matrix4d().SetRotate(
                Gf.Rotation(Gf.Quatd(float(pose[6]), Gf.Vec3d(*map(float, pose[3:6]))))
            )
            matrix.SetTranslateOnly(Gf.Vec3d(*map(float, pose[:3])))
            transform = Gf.Matrix4d().SetScale(scales[label]) * matrix
            xform = UsdGeom.Xformable(prim)
            xform.MakeMatrixXform().Set(transform)
            xform.SetResetXformStack(True)
        points = skin.skin(nodes)
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
        mesh.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.array([points.min(0), points.max(0)])))
        rep.orchestrator.step(rt_subframes=4 if frame == 0 else 1, delta_time=0.0, pause_timeline=True)
        data = np.asarray(rgb.get_data())
        if data.ndim != 3 or data.size == 0:
            raise RuntimeError(f"No RGB frame: {data.shape}")
        image = Image.fromarray(data[..., :3])
        draw = ImageDraw.Draw(image)
        actual_time = float(sample_times[index])
        phase = min(rows, key=lambda row: abs(row["t_s"] - actual_time))["phase"]
        draw.rectangle((0, 0, 1024, 35), fill=(20, 24, 30))
        draw.text((12, 10), f"Recorded unassisted physics | {phase} | t={actual_time:.2f} s", fill="white")
        path = args.out / f"frame_{frame:04d}.png"
        image.save(path)
        provenance.append(
            dict(
                frame=frame,
                state_index=index,
                actual_time_s=actual_time,
                phase=phase,
                nodes_sha256=hashlib.sha256(nodes.tobytes()).hexdigest(),
                rigid_poses_sha256=hashlib.sha256(poses.tobytes()).hexdigest(),
                image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
        if frame % 20 == 0:
            print(json.dumps(dict(rendered=frame, phase=phase)), flush=True)
    details = []
    camera = UsdGeom.Camera(stage.GetPrimAtPath('/World/InspectionCamera'))
    box = poses[labels.index('/World/DashCarton')]
    target = box[:3] + np.array([0.0, 0.0, 0.13])
    for name, offset in [('final_inside', [0.32, 0.18, 0.58]), ('final_grasp', [-0.38, -0.32, 0.40])]:
        eye = target + offset
        view = (
            Gf.Matrix4d()
            .SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, target)), Gf.Vec3d(0, 0, 1))
            .GetInverse()
        )
        camera.MakeMatrixXform().Set(view)
        camera.CreateFocalLengthAttr(40.0)
        rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=True)
        image = Image.fromarray(np.asarray(rgb.get_data())[..., :3])
        path = args.out.parent / (name + '.png')
        image.save(path)
        details.append(
            dict(
                file=path.name,
                state_index=index,
                actual_time_s=actual_time,
                image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    (args.out / "provenance.json").write_text(
        json.dumps(
            dict(
                source_report_sha256=hashlib.sha256((args.episode / "report.json").read_bytes()).hexdigest(),
                physical_simulation=False,
                scope="Visualization of saved accepted physics states",
                frames=provenance,
                detail_views=details,
            ),
            indent=2,
        )
    )
except BaseException:
    import traceback

    failed = True
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
finally:
    app.close(exit_code=1 if failed else 0)
