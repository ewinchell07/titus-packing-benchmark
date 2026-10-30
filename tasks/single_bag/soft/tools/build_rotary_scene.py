"""Restore the original dish and an upright free carton for the bimanual task."""

from pathlib import Path
import json
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Gf, Usd, UsdGeom, UsdPhysics

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'newton170/runtime'))
from robot_kinematics import transform

stage = Usd.Stage.Open(str(ROOT / 'scene/single_bag_box.usda'))
layer = stage.GetRootLayer()
rotary = '../assets/rotary_table/rotary_table.usda'
if rotary not in layer.subLayerPaths:
    layer.subLayerPaths.append(rotary)
stage.RemovePrim('/World/Table')
dish = stage.GetPrimAtPath('/World/RotaryTableV2/dish_collision')
UsdPhysics.CollisionAPI.Apply(dish).CreateCollisionEnabledAttr(True)
UsdPhysics.MeshCollisionAPI.Apply(dish).CreateApproximationAttr('none')
UsdGeom.Imageable(dish).CreateVisibilityAttr('invisible')
box = stage.GetPrimAtPath('/World/DashCarton')
box_rotation = Rotation.from_euler('z', 90, degrees=True)
box_matrix = np.eye(4)
box_matrix[:3, :3], box_matrix[:3, 3] = box_rotation.as_matrix(), [-0.32, 0.14, 0.88]
UsdGeom.Xformable(box).MakeMatrixXform().Set(Gf.Matrix4d(box_matrix.T.tolist()))
points = np.load(ROOT / 'newton170/config/packed_tets.npz')['points']
yaw = Rotation.from_euler('z', 18, degrees=True).as_matrix()
rotated = points @ yaw.T
origin = np.array([-0.40, -0.26, 0.863443 + 0.0005 - rotated[:, 2].min()])
origin[:2] -= (rotated.min(0)[:2] + rotated.max(0)[:2]) / 2
bag_pose = np.eye(4)
bag_pose[:3, :3], bag_pose[:3, 3] = yaw, origin
UsdGeom.Xformable(stage.GetPrimAtPath('/World/Bag')).MakeMatrixXform().Set(Gf.Matrix4d(bag_pose.T.tolist()))

# The source terminal flap had a world-space reset transform from another box
# pose. Reconstruct its hinge-relative pose from the actual authored joint.
source = Usd.Stage.Open(str(ROOT / 'assets/workcell/workcell_source.usdc'))
joint = UsdPhysics.RevoluteJoint(source.GetPrimAtPath('/World/DashCarton/terminal_tuck_crease'))
body0 = stage.GetPrimAtPath(joint.GetBody0Rel().GetTargets()[0])
body1 = stage.GetPrimAtPath(joint.GetBody1Rel().GetTargets()[0])
cache = UsdGeom.XformCache()
A = transform(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get())
B = transform(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get())
angle = float(
    source.GetPrimAtPath(joint.GetPath()).GetAttribute('drive:angular:physics:targetPosition').Get()
)
motion = np.eye(4)
motion[:3, :3] = Rotation.from_euler('x', angle, degrees=True).as_matrix()
desired = np.array(cache.GetLocalToWorldTransform(body0)).T @ A @ motion @ np.linalg.inv(B)
parent = np.array(cache.GetLocalToWorldTransform(body1.GetParent())).T
UsdGeom.Xformable(body1).MakeMatrixXform().Set(Gf.Matrix4d((np.linalg.inv(parent) @ desired).T.tolist()))
UsdGeom.Xformable(body1).SetResetXformStack(False)

# Rebuild the existing panel proxies from their source geometry after the hinge
# correction. These are the physical carton walls, not supporting fixtures.
audit = json.loads((ROOT / 'scene/carton_contact_panels.json').read_text())
cache.Clear()
inverse = cache.GetLocalToWorldTransform(box).GetInverse()
for row in audit['panel_colliders']:
    prim = stage.GetPrimAtPath(row['original'])
    prim.SetActive(True)
    points = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
    size = np.maximum(np.ptp(points, axis=0), 0.001)
    centre = (points.max(0) + points.min(0)) / 2
    relative = cache.GetLocalToWorldTransform(prim) * inverse
    matrix = (
        Gf.Matrix4d().SetScale(Gf.Vec3d(*size)) * Gf.Matrix4d().SetTranslate(Gf.Vec3d(*centre)) * relative
    )
    UsdGeom.Cube(stage.GetPrimAtPath(row['proxy'])).MakeMatrixXform().Set(matrix)
    prim.SetActive(False)
layer.Save()
normal = np.array([-0.1369258230519024, 0.0, -0.9905813035695554])
approach = np.array([0.0, -1.0, 0.0])
left_rotation = np.array(
    [
        [0.05724348564633941, 0.037639371975694724, 0.9976504703694242],
        [-0.9910317459524636, -0.11871464152472441, 0.06134258229051448],
        [0.12074461422966115, -0.992214750729363, 0.030506172647496083],
    ]
)
config = dict(
    task='bimanual_rotary_pack',
    bag_pinch_xy_m=[-0.40, -0.26],
    initial_left_opening_m=-0.0018,
    left_grip_point_box_m=[-0.08264847216615319, 0.06045039166259286, 0.2103596511317528],
    box_reset_xyz_m=[-0.32, 0.14, 0.88],
    box_reset_quaternion_xyzw=box_rotation.as_quat().tolist(),
    left_hand_rotation=left_rotation.tolist(),
    reset_scope='Upright carton at a contact-consistent left grasp with 0.5 mm jaw clearance; the original drives close and lift it. No supports or attachments.',
    rotary_table_source='Original delivered RotaryTableV2 dish, exact triangle collision mesh, stationary for this case',
    carton='One free rigid open carton; source printed panels and corrected terminal hinge pose; no flap articulation in this baseline',
)
(ROOT / 'config/task.json').write_text(json.dumps(config, indent=2) + '\n')
(ROOT / 'scene/bimanual_layout.json').write_text(
    json.dumps(
        dict(
            box_upright=True,
            terminal_flap_hinge_angle_deg=angle,
            physical_rotary_mesh=str(dish.GetPath()),
            artificial_table_removed=True,
            **config
        ),
        indent=2,
    )
    + '\n'
)
print(json.dumps(config, indent=2))
