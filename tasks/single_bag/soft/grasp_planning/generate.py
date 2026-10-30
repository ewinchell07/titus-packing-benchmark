"""Generate GraspGen-X candidates for the delivered YAM; never command physics."""

import argparse, hashlib, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--repo', type=Path, required=True)
ap.add_argument('--checkpoints', type=Path, required=True)
ap.add_argument('--input', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--num-grasps', type=int, default=128)
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
os.environ['GRASPGENX_CHECKPOINT_DIR'] = str(args.checkpoints.resolve())
os.environ['GRASPGENX_GRIPPER_CFG_DIR'] = str(args.input.resolve())
sys.path.insert(0, str(args.repo.resolve()))
sys.path.insert(0, str(ROOT / 'newton170/runtime'))
import numpy as np, torch, trimesh
from pxr import Usd
from scipy.spatial.transform import Rotation
from robot_kinematics import RobotKinematics
from graspgenx.grasp_server import GraspGenXSampler
from graspgenx.utils.checkpoint_io import load_model_cfg
from graspgenx.utils.collision_filter import filter_colliding_grasps

np.random.seed(0)
torch.manual_seed(0)
params = json.loads((args.input / 'sweep_volume.json').read_text())
cfg = load_model_cfg(args.checkpoints / 'release/gen', args.checkpoints / 'release/dis')
sampler = GraspGenXSampler.from_sweep_volume(cfg, params)
actual_mesh = trimesh.load(args.input / 'gripper_open.obj', force='mesh')
sampler.gripper.collision_mesh = actual_mesh
sampler.gripper.visual_mesh = actual_mesh
z = np.load(args.input / 'input.npz')
start = time.monotonic()
poses, scores = GraspGenXSampler.run_inference(
    z['object_points'].astype(np.float32),
    sampler,
    num_grasps=args.num_grasps,
    topk_num_grasps=args.num_grasps,
    min_grasps=1,
    max_tries=1,
    remove_outliers=False,
)
poses = poses.cpu().numpy()
scores = scores.cpu().numpy()
elapsed = time.monotonic() - start
clear = filter_colliding_grasps(
    z['scene_points'],
    poses,
    gripper_collision_mesh=actual_mesh,
    collision_threshold=0.005,
    num_collision_samples=4000,
    batch_size=4,
)
H_G = np.array(json.loads((args.input / 'hand_from_graspgen.json').read_text()))
G_H = np.linalg.inv(H_G)
kin = RobotKinematics(Usd.Stage.Open(str(ROOT / 'scene/single_bag_box.usda')))
seed = json.loads((args.input / 'input.json').read_text())['actual_joint_q']
candidates = []
for i in np.argsort(-scores):
    T = poses[i] @ G_H
    pre = T.copy()
    pre[:3, 3] -= 0.05 * T[:3, 2]
    item = dict(
        rank=len(candidates) + 1,
        model_score=float(scores[i]),
        world_from_graspgen=poses[i].tolist(),
        world_from_native_hand=T.tolist(),
        hand_position_m=T[:3, 3].tolist(),
        hand_quaternion_xyzw=Rotation.from_matrix(T[:3, :3]).as_quat().tolist(),
        scene_pointcloud_clear=bool(clear[i]),
        ik_reachable=False,
        physical_acceptance=False,
    )
    local = (z['object_points'] - poses[i, :3, 3]) @ poses[i, :3, :3]
    occupancy = int(
        np.count_nonzero(
            np.all(
                np.abs(local - np.array(params['offset_open'])) <= np.array(params['extents_open']) / 2,
                axis=1,
            )
        )
    )
    item['swept_volume_object_points'] = occupancy
    if occupancy < 10:
        item['proposal_rejection'] = 'Fewer than 10 observed surface points in the jaw closure volume'
    if clear[i] and occupancy >= 10:
        try:
            q, _ = kin.solve(pre, seed)
            q, error = kin.solve(T, q)
            item.update(
                ik_reachable=True,
                ik_error=error,
                joint_targets={n: q[n] for n in q if n.startswith('right_joint_')},
            )
        except ValueError as e:
            item['ik_rejection'] = str(e)
    candidates.append(item)
report = dict(
    model='NVlabs/GraspGenX',
    source_commit='b9429097728cb1c430dd78b92edf17ba318aad03',
    generated=len(candidates),
    scene_pointcloud_clear=int(clear.sum()),
    ik_and_scene_candidates=sum(r['ik_reachable'] for r in candidates),
    generation_seconds=elapsed,
    gripper='Exact delivered YAM collision geometry plus effective pad swept volume',
    physical_acceptance=False,
    scope='Learned proposals; sampled scene collision filtering, observed points in the jaw closure volume, and native joint-limit IK. These do not establish a deformable-bag grasp or a collision-free full-arm trajectory.',
    input_sha256={
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.input.iterdir() if p.is_file()
    },
    candidates=candidates,
)
(args.out / 'candidates.json').write_text(json.dumps(report, indent=2) + '\n')
np.savez_compressed(args.out / 'raw_candidates.npz', poses=poses, scores=scores, scene_clear=clear)
print(json.dumps({k: v for k, v in report.items() if k not in ['candidates', 'input_sha256']}), flush=True)
