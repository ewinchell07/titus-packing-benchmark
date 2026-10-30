"""Synthetic hand-mounted RGB sensors, with camera-only pose publication.

Mount/optics are explicit simulation choices, not measured hardware calibration.
Native hand poses drive the cameras. No robot/object state is changed here.
"""

import json
from pathlib import Path
import numpy as np
from pxr import Gf, UsdGeom

CONFIG = {
    'calibration_status': 'simulation_configured_not_hardware_calibrated',
    'resolution_px': [256, 256],
    'focal_length_mm': 14.0,
    'horizontal_aperture_mm': 20.955,
    'vertical_aperture_mm': 20.955,
    'clipping_range_m': [0.005, 2.0],
    'hand_local_eye_m': [-0.065, 0.0, 0.025],
    'hand_local_look_at_m': [-0.016961, 0.0, 0.18],
    'hand_local_up': [0.0, 1.0, 0.0],
    'camera_convention': 'USD camera looks along local -Z with +Y up; Gf row-vector matrices',
    'physical_sensor_housing_and_collisions': False,
}


def pose_matrix(pose):
    m = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Quatd(float(pose[6]), Gf.Vec3d(*map(float, pose[3:6])))))
    m.SetTranslateOnly(Gf.Vec3d(*map(float, pose[:3])))
    return m


class WristCameras:
    def __init__(self, world, rep):
        self.world, self.rep = world, rep
        self.mount = (
            Gf.Matrix4d()
            .SetLookAt(
                Gf.Vec3d(*CONFIG['hand_local_eye_m']),
                Gf.Vec3d(*CONFIG['hand_local_look_at_m']),
                Gf.Vec3d(*CONFIG['hand_local_up']),
            )
            .GetInverse()
        )
        self.cameras = {}
        self.products = {}
        self.annotators = {}
        self.ops = {}
        for side in ('left', 'right'):
            key = 'cam_' + side + '_wrist'
            path = '/World/Rig/' + side + '_hand/PolicyWristCamera'
            camera = UsdGeom.Camera.Define(world.native_stage, path)
            camera.CreateFocalLengthAttr(CONFIG['focal_length_mm'])
            camera.CreateHorizontalApertureAttr(CONFIG['horizontal_aperture_mm'])
            camera.CreateVerticalApertureAttr(CONFIG['vertical_aperture_mm'])
            camera.CreateClippingRangeAttr(Gf.Vec2f(*CONFIG['clipping_range_m']))
            op = camera.MakeMatrixXform()
            op.Set(Gf.Matrix4d(1.0))
            camera.SetResetXformStack(True)
            product = rep.create.render_product(path, tuple(CONFIG['resolution_px']))
            annotator = rep.AnnotatorRegistry.get_annotator('rgb')
            annotator.attach([product])
            self.cameras[key] = camera
            self.products[key] = product
            self.annotators[key] = annotator
            self.ops[key] = op
        self.update()

    def update(self):
        poses = self.world.read_rigid()[0]
        self.transforms = {}
        for side in ('left', 'right'):
            key = 'cam_' + side + '_wrist'
            index = list(self.world.model.body_label).index('/World/Rig/' + side + '_hand')
            matrix = self.mount * pose_matrix(poses[index])
            self.ops[key].Set(matrix)
            self.transforms[key] = {
                'hand_pose_xyzw': poses[index].tolist(),
                'camera_to_world_row_matrix': np.asarray(matrix).tolist(),
            }

    def save(self, base_path):
        from PIL import Image

        base_path = Path(base_path)
        result = {}
        for side in ('left', 'right'):
            key = 'cam_' + side + '_wrist'
            data = np.asarray(self.annotators[key].get_data())
            if data.ndim != 3 or data.size == 0:
                raise RuntimeError('No rendered wrist RGB: ' + key)
            path = base_path.with_name(base_path.stem + '_' + side + '_wrist.png')
            Image.fromarray(data[..., :3]).save(path)
            result[key] = str(path)
        metadata = dict(
            CONFIG,
            sim_t_s=self.world.time,
            camera_to_hand_row_matrix=np.asarray(self.mount).tolist(),
            views=self.transforms,
        )
        base_path.with_name(base_path.stem + '_camera_metadata.json').write_text(
            json.dumps(metadata, indent=2) + '\n'
        )
        return result
