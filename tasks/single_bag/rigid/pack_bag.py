"""Use the calibrated left gripper to hold the carton and the right to pack one bag.

Objects remain free throughout. Commands go only to the original joint drives.
The carton upper rim starts between the left jaws; the runtime supplies no fixture.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--stop-after', choices=['box_hold', 'bag_hold', 'vertical', 'over_box'])
parser.add_argument(
    '--no-close', action='store_true', help='Open-right-jaw negative control; cannot pass packing acceptance'
)
parser.add_argument(
    '--grasp-candidates',
    type=Path,
    help='Optional GraspGen-X candidates.json; proposals still require physical validation',
)
parser.add_argument(
    '--candidate-rank',
    type=int,
    help='Rank to test; defaults to the highest-scored candidate that passed proposal filters',
)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
from isaacsim import SimulationApp

app = SimulationApp(
    {
        'headless': True,
        'fast_shutdown': True,
        'multi_gpu': False,
        'active_gpu': 0,
        'extra_args': ['--/UJITSO/enabled=false'],
    }
)
sys.path.insert(0, str(ROOT / 'runtime'))
from native_rigid_world import NativeRigidWorld as IsaacNewtonWorld
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

world = IsaacNewtonWorld(ROOT, app)
rows, frames = [], []
force_target = 3.0
failed = False
box_reference = None
selected_candidate = None
criteria = json.loads((ROOT / 'config/bimanual_acceptance.json').read_text())

from box_grasp_metrics import grasp_reference, grasp_motion


def record(phase):
    row = dict(phase=phase, **world.measure())
    rows.append(row)
    frames.append(world.frame())
    if len(rows) % 5 == 1:
        print(
            json.dumps(
                {
                    k: row[k]
                    for k in [
                        'phase',
                        't_s',
                        'centroid_m',
                        'finger_contact_force_N',
                        'left_box_contact_force_N',
                        'box_lowest_z_m',
                        'box_upright_error_deg',
                        'bag_inside_box_fraction',
                        'max_inverted_tets',
                    ]
                }
            ),
            flush=True,
        )
    return row


def execute(
    name, trajectory, seconds, right_opening=None, left_opening=None, force_hold=False, stop_on_grasp=False
):
    print(json.dumps(dict(phase=name, status='starting', t_s=world.time)), flush=True)
    start_right = world.target_opening
    start_left = -world.target_posture['left_finger_joint1']
    filtered = min(world.finger_contact_forces())
    missing = 0.0
    steps = round(seconds / world.dt)
    for tick in range(steps):
        t = (tick + 1) * world.dt
        u = min(1.0, t / seconds)
        smooth = u * u * (3 - 2 * u)
        posture = trajectory(t)
        left = start_left if left_opening is None else start_left + smooth * (left_opening - start_left)
        right = start_right if right_opening is None else start_right + smooth * (right_opening - start_right)
        if force_hold:
            measured = min(world.finger_contact_forces())
            filtered += min(1.0, world.dt / 0.02) * (measured - filtered)
            rate = np.clip(0.01 * (filtered - force_target), -0.02, 0.02)
            right = float(np.clip(world.target_opening + rate * world.dt, -0.003, 0.05))
        posture.update(left_finger_joint1=-left, left_finger_joint2=left)
        world.step(posture, right)
        if box_reference is not None:
            motion = world.track_box_hold(box_reference)
            if motion['maximum_box_slip_m'] > criteria['maximum_box_slip_relative_to_hand_m']:
                raise RuntimeError('Carton slipped in left gripper during ' + name)
            if (
                motion['maximum_box_relative_rotation_deg']
                > criteria['maximum_box_rotation_relative_to_hand_deg']
            ):
                raise RuntimeError('Carton rotated beyond acceptance during ' + name)
            if motion['maximum_box_upright_error_deg'] > criteria['maximum_box_upright_error_deg']:
                raise RuntimeError('Carton tilted beyond acceptance during ' + name)
            if motion['minimum_box_clearance_m'] < criteria['required_box_clearance_above_entire_rotary_m']:
                raise RuntimeError('Carton lost clearance during ' + name)
        if force_hold:
            missing = 0.0 if min(world.finger_contact_forces()) >= 0.03 else missing + world.dt
            if missing > 0.25:
                raise RuntimeError('Lost physical bag grasp during ' + name)
        if tick % 60 == 0 or tick == steps - 1:
            row = record(name)
            if world.time > 1.0 and row['box_lowest_z_m'] < 0.865:
                raise RuntimeError('Carton lost unsupported hold during ' + name)
            if box_reference is not None:
                if (
                    grasp_motion(row['left_hand_pose'], row['box_pose'], box_reference)['grasp_anchor_slip_m']
                    > criteria['maximum_box_slip_relative_to_hand_m']
                ):
                    raise RuntimeError('Carton slipped in left gripper during ' + name)
                if row['box_upright_error_deg'] > criteria['maximum_box_upright_error_deg']:
                    raise RuntimeError('Carton tilted beyond acceptance during ' + name)
        if stop_on_grasp and min(world.finger_contact_forces()) >= force_target:
            record(name)
            world.save(args.out, rows, frames)
            return
    world.save(args.out, rows, frames)
    if stop_on_grasp:
        raise RuntimeError('No bilateral bag grasp')


def hold(name, seconds, **options):
    posture = world.target_posture.copy()
    execute(name, lambda _: posture.copy(), seconds, **options)


def move_tcp(name, xyz, rotation, seconds, side='right', joint_space=False, **options):
    posture = world.target_posture.copy()
    pose = world.kin.forward(posture)['/World/Rig/' + side + '_hand']
    prior_rotation = Rotation.from_matrix(pose[:3, :3])
    prior_xyz = pose[:3, 3] + prior_rotation.apply(world.tcp_local)
    if joint_space:
        goal = np.eye(4)
        goal[:3, :3] = rotation.as_matrix()
        goal[:3, 3] = np.asarray(xyz) - rotation.apply(world.tcp_local)
        end, _ = world.kin.solve(goal, posture, side)
        names = [f'{side}_joint_{i}' for i in range(1, 7)]

        def trajectory(t):
            u = np.clip(t / seconds, 0.0, 1.0)
            u = u * u * (3 - 2 * u)
            return dict(posture, **{n: posture[n] + u * (end[n] - posture[n]) for n in names})

        execute(name, trajectory, seconds, **options)
        return
    slerp = Slerp([0.0, 1.0], Rotation.concatenate([prior_rotation, rotation]))
    times = np.linspace(0.0, seconds, max(11, round(seconds / 0.05) + 1))
    names = [f'{side}_joint_{i}' for i in range(1, 7)]
    samples = []
    seed = posture.copy()
    for t in times:
        u = t / seconds
        u = u * u * (3 - 2 * u)
        r = slerp(u)
        p = prior_xyz * (1 - u) + np.asarray(xyz) * u
        target = np.eye(4)
        target[:3, :3] = r.as_matrix()
        target[:3, 3] = p - r.apply(world.tcp_local)
        seed, _ = world.kin.solve(target, seed, side)
        samples.append([seed[n] for n in names])
    spline = CubicSpline(times, np.array(samples), bc_type=((1, np.zeros(6)), (1, np.zeros(6))))

    def trajectory(t):
        return dict(posture, **dict(zip(names, map(float, spline(t)))))

    execute(name, trajectory, seconds, **options)


def tcp(side='right'):
    pose = world.kin.forward(world.target_posture)['/World/Rig/' + side + '_hand']
    r = Rotation.from_matrix(pose[:3, :3])
    return pose[:3, 3] + r.apply(world.tcp_local), r


def stop_at(phase):
    if args.stop_after == phase:
        world.status = 'diagnostic_completed'
        world.render(args.out / 'scene.png')
        return True
    return False


def run():
    global box_reference, selected_candidate, force_target
    hold('box_close', 0.2, left_opening=-0.003)
    hold('settle_and_box_hold', 1.8)
    p, r = tcp('left')
    move_tcp('lift_box', p + [0, 0, 0.02], r, 1.0, side='left')
    hold('box_hold', 1.0)
    row = world.measure()
    box_reference = grasp_reference(row['left_hand_pose'], row['box_pose'], world.tcp_local)
    if stop_at('box_hold'):
        return
    # Fit a rigid pose to the free, settled bag to choose a grasp. This reads
    # nodes only; the object is never placed or reoriented by the controller.
    nodes = world.state.particle_q.numpy().astype(float)
    weights = world.mass / world.mass.sum()
    canonical = world.canonical
    c0, c1 = weights @ canonical, weights @ nodes
    U, _, Vt = np.linalg.svd((canonical - c0).T @ ((nodes - c1) * weights[:, None]))
    fit = Vt.T @ U.T
    if np.linalg.det(fit) < 0:
        Vt[-1] *= -1
        fit = Vt.T @ U.T
    # Tilt the grasp 45 degrees to fit the rigid cross-section within the jaws.
    # The positive-X end and the transfer route satisfy the original joint limits.
    C = Rotation.from_euler('x', world.task_config['rigid_grasp_roll_deg'], degrees=True)
    local_point = np.asarray(world.task_config['rigid_grasp_local_m'])
    grasp_point = c1 + fit @ (local_point - c0)
    grasp_rotation = Rotation.from_matrix(fit) * C
    approach_point = grasp_point - 0.05 * grasp_rotation.as_matrix()[:, 2]
    if args.grasp_candidates:
        proposals = json.loads(args.grasp_candidates.read_text())['candidates']
        eligible = [
            c
            for c in proposals
            if c['ik_reachable']
            and c['scene_pointcloud_clear']
            and (args.candidate_rank is None or c['rank'] == args.candidate_rank)
        ]
        if not eligible:
            raise ValueError('No eligible candidate with that rank')
        selected_candidate = eligible[0]
        hand = np.asarray(selected_candidate['world_from_native_hand'])
        grasp_rotation = Rotation.from_matrix(hand[:3, :3])
        grasp_point = hand[:3, 3] + grasp_rotation.apply(world.tcp_local)
        approach_point = grasp_point - 0.05 * hand[:3, 2]
    move_tcp('align', approach_point, grasp_rotation, 1.5, right_opening=0.05, joint_space=True)
    move_tcp('approach', grasp_point, grasp_rotation, 1.5, right_opening=0.05)
    if args.no_close:
        hold('open_gripper_control', 0.5, right_opening=0.05)
        p, _ = tcp()
        down = Rotation.from_euler('z', 18, degrees=True) * Rotation.from_euler('x', 180, degrees=True)
        move_tcp('lift_bag', p + [0, 0, 0.12], down, 2.0, right_opening=0.05)
        hold('bag_hold', 2.0, right_opening=0.05)
        world.status = 'negative_control_completed'
        world.render(args.out / 'scene.png')
        return
    hold('close_bag', 2.0, right_opening=-0.003, stop_on_grasp=True)
    hold('grasp_hold', 0.5, force_hold=True)
    p, r = tcp()
    measured = world.measure()
    bag_from_hand = (
        C
        if selected_candidate is None
        else Rotation.from_quat(measured['bag_pose'][3:]).inv()
        * Rotation.from_quat(measured['hand_pose'][3:])
    )
    desired_bag = Rotation.from_euler('y', -90, degrees=True)
    vertical = desired_bag * bag_from_hand
    move_tcp('clear_dish', p + np.array([0.0, -0.12, 0.06]), r, 1.5, force_hold=True)
    p, r = tcp()
    p[:2] = world.task_config['rigid_lift_xy_m']
    p[2] = 1.12
    move_tcp('lift_bag', p, vertical, 3.0, force_hold=True)
    hold('bag_hold', 2.0, force_hold=True)
    if stop_at('bag_hold'):
        return
    p, _ = tcp()
    p[2] = 1.12
    move_tcp('turn_bag_vertical', p, vertical, 0.5, force_hold=True)
    if stop_at('vertical'):
        return
    # An arched transfer clears the open lid before descending toward the
    # opening. A low horizontal transfer collides with the rear closure flap.
    p[2] = 1.40
    move_tcp('raise_for_insertion', p, vertical, 2.5, force_hold=True)
    box = world.measure()['box_pose']
    box_r = Rotation.from_quat(box[3:])
    vertical = box_r * desired_bag * bag_from_hand
    high = np.array(box[:3]) + box_r.apply([0.0, 0.0, 0.45])
    move_tcp('over_box_high', high, vertical, 3.0, force_hold=True)
    # Plan a fitting orientation from the observed rigid grasp, then realize it
    # through native robot drives. The bag is never reposed or attached.
    from insertion_plan import fit_orientation

    force_target = 4.0
    hold('secure_for_insertion', 0.5, force_hold=True)
    vertices = np.concatenate([world.hull_vertices, world.canonical])
    fitted = False
    for attempt in range(3):
        observed = world.measure()
        box = np.asarray(observed['box_pose'])
        box_r = Rotation.from_quat(box[3:])
        bag = np.asarray(observed['bag_pose'])
        bag_r = Rotation.from_quat(bag[3:])
        hand = np.asarray(observed['hand_pose'])
        hand_r = Rotation.from_quat(hand[3:])
        desired, plan_metrics = fit_orientation(vertices, box_r.inv() * bag_r)
        vertical = box_r * desired * bag_r.inv() * hand_r
        hand_tcp = hand[:3] + hand_r.apply(world.tcp_local)
        relative = box_r.inv().apply(
            (vertical * hand_r.inv()).apply(bag_r.apply(vertices) + bag[:3] - hand_tcp)
        )
        center_xy = -(relative.min(0) + relative.max(0))[:2] / 2
        target = np.array(box[:3]) + box_r.apply([*center_xy, 0.45])
        print(
            json.dumps(dict(insertion_plan=plan_metrics, attempt=attempt + 1, target_tcp=target.tolist())),
            flush=True,
        )
        move_tcp('align_packet_to_box', target, vertical, 1.5, force_hold=True)
        hold('aligned_hold', 0.4, force_hold=True)
        measured = world.measure()
        actual_r = Rotation.from_quat(measured['box_pose'][3:]).inv() * Rotation.from_quat(
            measured['bag_pose'][3:]
        )
        actual_extent = np.ptp(actual_r.apply(vertices), axis=0)
        if np.all(actual_extent <= np.array([0.1305, 0.1694, 0.207])):
            fitted = True
            break
    if not fitted:
        raise RuntimeError('Physical packet orientation did not reach insertion clearance')
    # Center the observed envelope, accounting for where the fingers actually
    # hold the packet. This shifts the robot target, not the object.
    observed = world.measure()
    box = np.asarray(observed['box_pose'])
    box_r = Rotation.from_quat(box[3:])
    bag = np.asarray(observed['bag_pose'])
    bag_r = Rotation.from_quat(bag[3:])
    hand = np.asarray(observed['hand_pose'])
    hand_r = Rotation.from_quat(hand[3:])
    hand_tcp = hand[:3] + hand_r.apply(world.tcp_local)
    relative = box_r.inv().apply(bag_r.apply(vertices) + bag[:3] - hand_tcp)
    center_xy = -(relative.min(0) + relative.max(0))[:2] / 2
    over = np.array(box[:3]) + box_r.apply([*center_xy, 0.35])
    move_tcp('over_box', over, vertical, 1.5, force_hold=True)
    if stop_at('over_box'):
        return
    # Keep the wrist above the open lid; the packet's lower end is inside the
    # opening before release and the remaining descent is passive gravity.
    insert = np.array(box[:3]) + box_r.apply([*center_xy, 0.29])
    move_tcp('insert', insert, vertical, 2.0, force_hold=True)
    hold('release_into_box', 0.7, right_opening=0.05)
    move_tcp('withdraw', over, vertical, 1.5, right_opening=0.05)
    hold('settle_in_box', 4.0, right_opening=0.05)
    world.status = 'completed'
    world.render(args.out / 'scene.png')


try:
    run()
except BaseException as error:
    import traceback

    failed = True
    world.status = 'failed'
    world.error = f'{type(error).__name__}: {error}'
    traceback.print_exc()
    try:
        world.render(args.out / 'failure.png')
    except Exception as render_error:
        print('Diagnostic render failed:', render_error, flush=True)
finally:
    world.save(args.out, rows, frames)
    report = json.loads((args.out / 'report.json').read_text())
    report.update(
        task='bimanual_rotary_pack',
        task_passed=False,
        configuration=world.task_config,
        no_close=args.no_close,
    )
    if selected_candidate is not None:
        import hashlib

        report['selected_grasp_candidate'] = dict(
            rank=selected_candidate['rank'],
            model_score=selected_candidate['model_score'],
            source_sha256=hashlib.sha256(args.grasp_candidates.read_bytes()).hexdigest(),
            world_from_native_hand=selected_candidate['world_from_native_hand'],
        )
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    accepted = False
    if world.status == 'completed':
        from score_pack import score

        acceptance = score(args.out)
        accepted = acceptance['passed']
        print(json.dumps(dict(acceptance=acceptance)), flush=True)
    if world.status == 'negative_control_completed':
        initial = [r for r in rows if r['phase'] == 'open_gripper_control'][-1]
        lifted = [r for r in rows if r['phase'] in ('lift_bag', 'bag_hold')]
        rise = max(r['centroid_m'][2] - initial['centroid_m'][2] for r in lifted)
        hand_rise = max(r['hand_pose'][2] - initial['hand_pose'][2] for r in lifted)
        accepted = rise < 0.02 and hand_rise > 0.08 and report['max_inverted_tets'] == 0
        (args.out / 'negative_control_check.json').write_text(
            json.dumps(
                dict(
                    passed=accepted,
                    task_passed=False,
                    maximum_bag_rise_m=rise,
                    hand_rise_m=hand_rise,
                    scope='Open right jaws; arm rises but bag remains on the dish',
                ),
                indent=2,
            )
            + '\n'
        )
    app.close(exit_code=0 if accepted or world.status == 'diagnostic_completed' else 1)
