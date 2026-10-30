"""Export the delivered YAM contact geometry and an observed bag for GraspGen-X."""

import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdGeom, UsdPhysics

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'newton170/runtime'))
from robot_kinematics import RobotKinematics
from visuals import Visuals

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('episode', type=Path)
ap.add_argument('--time', type=float, default=4.0)
ap.add_argument('--out', type=Path, required=True)
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
stage = Usd.Stage.Open(str(ROOT / 'scene/single_bag_box.usda'))
cache = UsdGeom.XformCache()
kin = RobotKinematics(stage)
z = np.load(args.episode / 'states.npz')
rows = json.loads((args.episode / 'telemetry.json').read_text())
idx = int(np.argmin(abs(z['t'] - args.time)))
row = min(rows, key=lambda r: abs(r['t_s'] - z['t'][idx]))
q = row['actual_joint_q'].copy()


def matrix(prim):
    return np.array(cache.GetLocalToWorldTransform(prim)).T


def apply(T, v):
    return v @ T[:3, :3].T + T[:3, 3]


def mesh(prim):
    if prim.IsA(UsdGeom.Mesh):
        m = UsdGeom.Mesh(prim)
        v = np.asarray(m.GetPointsAttr().Get(), float)
        counts = m.GetFaceVertexCountsAttr().Get()
        ids = m.GetFaceVertexIndicesAttr().Get()
        f = []
        j = 0
        for n in counts:
            f.extend([[ids[j], ids[j + i], ids[j + i + 1]] for i in range(1, n - 1)])
            j += n
        return v, np.array(f, int)
    if prim.IsA(UsdGeom.Cube):
        v = (
            np.array(
                [
                    [-1, -1, -1],
                    [-1, -1, 1],
                    [-1, 1, -1],
                    [-1, 1, 1],
                    [1, -1, -1],
                    [1, -1, 1],
                    [1, 1, -1],
                    [1, 1, 1],
                ],
                float,
            )
            * float(UsdGeom.Cube(prim).GetSizeAttr().Get())
            / 2
        )
        return v, np.array(
            [
                [0, 4, 6],
                [0, 6, 2],
                [1, 3, 7],
                [1, 7, 5],
                [0, 1, 5],
                [0, 5, 4],
                [2, 6, 7],
                [2, 7, 3],
                [0, 2, 3],
                [0, 3, 1],
                [4, 5, 7],
                [4, 7, 6],
            ]
        )
    raise ValueError('Unsupported collision primitive ' + str(prim.GetPath()))


def merge(parts):
    vs = []
    fs = []
    offset = 0
    for v, f in parts:
        vs.append(v)
        fs.append(f + offset)
        offset += len(v)
    return np.concatenate(vs), np.concatenate(fs)


def sample(v, f, count, rng):
    t = v[f]
    a = np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
    ids = rng.choice(len(f), count, p=a / a.sum())
    u = rng.random((count, 2))
    flip = u.sum(1) > 1
    u[flip] = 1 - u[flip]
    return t[ids, 0] + u[:, 0, None] * (t[ids, 1] - t[ids, 0]) + u[:, 1, None] * (t[ids, 2] - t[ids, 0])


q.update(right_finger_joint1=-0.05, right_finger_joint2=0.05)
poses = kin.forward(q)
H_G = np.eye(4)
H_G[:3, :3] = Rotation.from_euler('z', 90, degrees=True).as_matrix()
H_G[:3, 3] = [-0.016961, 0, 0]
G_H = np.linalg.inv(H_G)
H_W = np.linalg.inv(poses['/World/Rig/right_hand'])
parts = []
labels = []
for name in ['right_hand', 'right_left_finger', 'right_right_finger']:
    body = stage.GetPrimAtPath('/World/Rig/' + name)
    B_W = np.linalg.inv(matrix(body))
    for prim in Usd.PrimRange(body):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        v, f = mesh(prim)
        parts.append((apply(G_H @ H_W @ poses[str(body.GetPath())] @ B_W @ matrix(prim), v), f))
        labels.append(str(prim.GetPath()))
v, f = merge(parts)
with (args.out / 'gripper_open.obj').open('w') as out:
    for point in v:
        out.write('v ' + ' '.join(map(str, point)) + '\n')
    for face in f:
        out.write('f ' + ' '.join(map(str, face + 1)) + '\n')
contacts = json.loads((ROOT / 'assets/robot/physics/gripper_contacts.json').read_text())
landmarks = contacts['pad_local_landmarks_m']
neutral = q.copy()
neutral.update(right_finger_joint1=0.0, right_finger_joint2=0.0)
np0 = kin.forward(neutral)
H_W = np.linalg.inv(np0['/World/Rig/right_hand'])
pads = np.array(
    [
        apply(H_W @ np0['/World/Rig/' + name], np.array([landmarks[i]]))[0]
        for name, i in [('right_left_finger', 2), ('right_right_finger', 3)]
    ]
)
tcp = pads.mean(0)
gap = abs(pads[1, 1] - pads[0, 1])
size = np.array(contacts['pads'][2]['size'])
offset = apply(G_H, tcp[None])[0]
params = dict(
    extents_open=[float(gap + 0.1), float(2 * size[0]), float(2 * size[2])],
    offset_open=offset.tolist(),
    extents_mid=[float(gap + 0.047), float(2 * size[0]), float(2 * size[2])],
    offset_mid=offset.tolist(),
    gripper_type=0,
    fingertip_depth=float(offset[2] + size[2]),
)
assert np.max(abs(offset[:2])) < 1e-8
(args.out / 'sweep_volume.json').write_text(json.dumps(params, indent=2) + '\n')
(args.out / 'hand_from_graspgen.json').write_text(json.dumps(H_G.tolist(), indent=2) + '\n')
skin = object.__new__(Visuals)
skin.binding = np.load(ROOT / 'newton170/config/packed_skin_binding.npz')
bag_v = skin.skin(z['nodes'][idx])
_, bag_f = mesh(stage.GetPrimAtPath('/World/Bag/Visual/surface'))
rng = np.random.default_rng(0)
bag_pc = sample(bag_v, bag_f, 3500, rng)
body_poses = {str(n): p for n, p in zip(z['body_labels'], z['body_q'][idx])}
scene_parts = []
for prim in stage.Traverse():
    if (
        not prim.HasAPI(UsdPhysics.CollisionAPI)
        or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
    ):
        continue
    label = str(prim.GetPath())
    if not (
        label.startswith('/World/RotaryTableV2/')
        or label.startswith('/World/DashCarton/')
        or label.startswith('/World/Rig/left_')
    ):
        continue
    v0, f0 = mesh(prim)
    T = matrix(prim)
    parent = prim
    while parent and str(parent.GetPath()) not in body_poses:
        parent = parent.GetParent()
    if parent:
        pose = body_poses[str(parent.GetPath())]
        actual = np.eye(4)
        actual[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
        actual[:3, 3] = pose[:3]
        T = actual @ np.linalg.inv(matrix(parent)) @ T
    scene_parts.append((apply(T, v0), f0))
sv, sf = merge(scene_parts)
scene_pc = sample(sv, sf, 10000, rng)
np.savez_compressed(args.out / 'input.npz', object_points=bag_pc, scene_points=scene_pc)
receipt = dict(
    source_episode=str(args.episode.resolve()),
    state_index=idx,
    time_s=float(z['t'][idx]),
    state_sha256=hashlib.sha256((args.episode / 'states.npz').read_bytes()).hexdigest(),
    actual_joint_q=row['actual_joint_q'],
    robot_files=json.loads((ROOT / 'scene/scene_lock.json').read_text())['calibrated_robot_files'],
    collision_meshes=labels,
    physical_acceptance=False,
    scope='GraspGen-X proposal inputs from the actual delivered gripper and observed bag. Swept contact volume uses delivered effective pad landmarks and sizes; not a learned or hardware-calibrated contact model.',
)
(args.out / 'input.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(
    json.dumps(
        dict(exported=str(args.out), collision_parts=len(labels), gripper_vertices=len(v), params=params)
    )
)
