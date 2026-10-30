"""Check the shared policy contract, privilege boundary and timing against a fake clock."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
from titus import PolicyRunner, Scene as PackingScene


class ClockScene:
    variant = 'test'
    task = 'single_bag'
    task_info = {'bags': 1, 'boxes': 1, 'status': 'test'}

    def __init__(self):
        self.t = 0.0
        self.commands = []

    def observe(self):
        return dict(
            t_s=self.t,
            actual_joint_q={},
            commanded_joints={},
            opening_target_m=0.05,
            left_opening_target_m=-0.003,
            timestep_s=1 / 600,
            box_pose=['privileged'],
            finger_contact_force_N=[0, 0],
        )

    def step(self, **command):
        self.commands.append(command)
        self.t += command['ticks'] / 600
        return self.observe()

    def render(self, path):
        return str(path)

    def render_views(self, path):
        return {key: str(path) + key for key in ('cam_high', 'cam_left_wrist', 'cam_right_wrist')}


def policy(o):
    assert 'box_pose' not in o and 'finger_contact_force_N' not in o and 'simulator_state' not in o
    assert set(o['rgb_paths']) == {'cam_high', 'cam_left_wrist', 'cam_right_wrist'}
    assert 'camera_metadata' not in o and o['rgb_path'] == o['rgb_paths']['cam_high']
    return [dict(ticks=20), dict(ticks=40, opening_m=0.03)]


with TemporaryDirectory() as directory:
    for mode, expected in [('freeze', 60), ('held_command', 6061)]:
        scene = ClockScene()
        out = Path(directory) / mode
        runner = PolicyRunner(scene, out, policy_name='contract-test', timing=mode)
        with patch('titus.policy.time.monotonic', side_effect=[0.0, 10.001]):
            runner.run_step(policy)
        record = json.loads((out / 'steps.jsonl').read_text())
        assert abs(scene.t - expected / 600) < 1e-9
        assert record['simulator_truth_before']['box_pose'] == ['privileged']
        assert all(c['ticks'] <= 6000 for c in scene.commands)
        assert record['modeled_delay_ticks'] == (6001 if mode == 'held_command' else 0)
        assert record['action_start_sim_s'] == (6001 / 600 if mode == 'held_command' else 0)
    scene = ClockScene()
    runner = PolicyRunner(
        scene, Path(directory) / 'oracle', policy_name='oracle', privileged=True, capture_rgb=False
    )

    def oracle(o):
        assert o['simulator_state']['box_pose'] == ['privileged']
        return dict(ticks=1)

    runner.run_step(oracle)
    try:
        runner.run_step(lambda o: dict(ticks=1.5))
    except ValueError:
        pass
    else:
        raise AssertionError('Fractional ticks were accepted')
    assert len(scene.commands) == 1
    scene = ClockScene()
    runner = PolicyRunner(
        scene, Path(directory) / 'external', policy_name='external', camera_views=('cam_high',)
    )

    def external(o):
        assert set(o['rgb_paths']) == {'cam_high'}
        return dict(ticks=1)

    runner.run_step(external)


original = dict(
    commanded_joints=dict(left_finger_joint1=0.003, right_finger_joint1=-0.04, right_finger_joint2=0.04),
    opening_target_m=0.02,
)
fixed = PackingScene.normalize(original)
assert (
    fixed['commanded_joints']['right_finger_joint1'] == -0.02
    and fixed['commanded_joints']['right_finger_joint2'] == 0.02
)
assert (
    fixed['left_opening_target_m'] == -0.003 and original['commanded_joints']['right_finger_joint1'] == -0.04
)

print(
    json.dumps(
        dict(
            passed=True,
            checks=[
                'policy privilege boundary',
                'explicit oracle mode',
                'freeze timing',
                'latency rounded to ticks',
                'delays longer than API limit',
                'chunk recording',
                'invalid tick rejection',
                'consistent gripper target observations',
            ],
        )
    )
)
