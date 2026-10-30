"""Render the native rigid-body transforms without editing physics or skin geometry."""

from pathlib import Path
import numpy as np
from pxr import Gf, UsdGeom, UsdLux, Vt


def add_camera_lights(stage):
    camera = UsdGeom.Camera.Define(stage, "/World/InspectionCamera")
    view = (
        Gf.Matrix4d()
        .SetLookAt(Gf.Vec3d(0.65, -1.45, 1.7), Gf.Vec3d(-0.32, -0.02, 0.88), Gf.Vec3d(0, 0, 1))
        .GetInverse()
    )
    camera.MakeMatrixXform().Set(view)
    camera.CreateFocalLengthAttr(28.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    dome = UsdLux.DomeLight.Define(stage, "/World/InspectionDome")
    dome.CreateIntensityAttr(600.0)
    light = UsdLux.DistantLight.Define(stage, "/World/InspectionKey")
    light.CreateIntensityAttr(1800.0)
    light.CreateAngleAttr(10.0)
    light.AddRotateXYZOp().Set(Gf.Vec3f(25, -25, -40))


class Visuals:
    def __init__(self, world):
        import omni.replicator.core as rep

        self.world, self.rep = world, rep
        self.product = rep.create.render_product("/World/InspectionCamera", (1024, 768))
        self.rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        self.rgb.attach([self.product])
        from wrist_cameras import WristCameras

        self.wrist_cameras = WristCameras(world, rep)

    def skin(self, nodes):
        b = self.binding
        tri = nodes[b["triangle_vertex_ids"]]
        e1 = tri[:, 1] - tri[:, 0]
        normal = np.cross(e1, tri[:, 2] - tri[:, 0])
        length, area = np.linalg.norm(e1, axis=1), np.linalg.norm(normal, axis=1)
        if min(length.min(), area.min()) <= 1e-12:
            raise RuntimeError("Degenerate skin frame")
        e1 /= length[:, None]
        normal /= area[:, None]
        e2 = np.cross(normal, e1)
        offset = b["triangle_frame_offsets"]
        result = np.einsum("ni,nij->nj", b["triangle_weights"], tri)
        result += offset[:, 0, None] * e1 + offset[:, 1, None] * e2 + offset[:, 2, None] * normal
        return result.astype(np.float32)

    def render(self, path):
        from PIL import Image

        w = self.world
        before, before_velocity = w.read_rigid()
        before, before_velocity = before.copy(), before_velocity.copy()
        before_time = w.manager.get_simulation_time()
        self.wrist_cameras.update()
        playing = w.timeline.is_playing()
        w.timeline.pause()
        import usdrt
        from pxr import UsdUtils

        # Publish measured poses to OmniHydra's display-only world transforms.
        # The manual physics loop can leave native Fabric publication stale.
        # These attributes are not USD/native rigid-body pose commands.
        stage = usdrt.Usd.Stage.Attach(UsdUtils.StageCache.Get().GetId(w.native_stage).ToLongInt())
        if not hasattr(self, "display_scales"):
            cache = UsdGeom.XformCache()
            self.display_scales = [
                Gf.Transform(cache.GetLocalToWorldTransform(w.native_stage.GetPrimAtPath(label))).GetScale()
                for label in w.model.body_label
            ]
        for label, pose, scale in zip(w.model.body_label, before, self.display_scales):
            xform = usdrt.Rt.Xformable(stage.GetPrimAtPath(label))
            xform.CreateWorldPositionAttr(usdrt.Gf.Vec3d(*map(float, pose[:3])))
            xform.CreateWorldOrientationAttr(
                usdrt.Gf.Quatf(float(pose[6]), usdrt.Gf.Vec3f(*map(float, pose[3:6])))
            )
            xform.CreateWorldScaleAttr(usdrt.Gf.Vec3f(*map(float, scale)))
        self.rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=True)
        data = self.rgb.get_data()
        if data is None or np.asarray(data).size == 0:
            raise RuntimeError("No rendered RGB frame")
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.asarray(data)[..., :3]).save(path)
        self.last_camera_capture = {"cam_high": str(path), **self.wrist_cameras.save(path)}
        after, after_velocity = w.read_rigid()
        if (
            np.max(np.abs(before - after)) > 1e-7
            or np.max(np.abs(before_velocity - after_velocity)) > 1e-7
            or w.manager.get_simulation_time() != before_time
        ):
            raise RuntimeError("Rendering changed native physics state or time")
        if playing:
            w.timeline.play()
        return str(path)
