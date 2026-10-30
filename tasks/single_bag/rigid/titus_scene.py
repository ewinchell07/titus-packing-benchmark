"""Small Python interface for an LLM controller or a custom robotics experiment.

Set ISAAC_PYTHON and TITUS_SOLVER_SITE before constructing TitusScene.
Each reset starts a fresh simulator process, including fresh solver caches.
"""

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PREFIX = "TITUS_REPLY "


class TitusScene:
    def __init__(self, log_path="titus_simulator.log"):
        self.log_path = Path(log_path).resolve()
        self.process = None
        self.reset()

    def reset(self):
        """Restore the scene in a fresh process; returns the initial observation."""
        self.close()
        python = os.environ["ISAAC_PYTHON"]
        site = os.environ.get("TITUS_SOLVER_SITE", str(ROOT / ".solver/site"))
        self.log = self.log_path.open("a")
        env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="YES")
        self.process = subprocess.Popen(
            [python, "-I", "-u", "-B", str(Path(__file__)), "--worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        return self._reply()

    def _reply(self):
        for line in self.process.stdout:
            self.log.write(line)
            self.log.flush()
            if line.startswith(PREFIX):
                answer = json.loads(line[len(PREFIX) :])
                if "error" in answer:
                    raise RuntimeError(answer["error"])
                return answer["result"]
        raise RuntimeError(f"Simulator exited; inspect {self.log_path}")

    def _request(self, operation, **arguments):
        self.process.stdin.write(json.dumps(dict(operation=operation, **arguments)) + "\n")
        self.process.stdin.flush()
        return self._reply()

    def observe(self, include_bag_nodes=False):
        return self._request("observe", include_bag_nodes=include_bag_nodes)

    def move(self, joint_targets=None, opening_m=None, seconds=1.0, left_opening_m=None):
        """Smooth joint targets in radians; opening is one finger's travel in metres.

        The opening range is [-0.003, 0.05]. The neutral pad gap is 6.0894 mm;
        add twice opening_m to obtain nominal pad separation. Motion uses the
        original actuator gains and limits, with physical contact feedback.
        """
        return self._request(
            "move",
            joint_targets=joint_targets or {},
            opening_m=opening_m,
            left_opening_m=left_opening_m,
            seconds=seconds,
        )

    def step(self, joint_targets=None, opening_m=None, ticks=1, left_opening_m=None):
        """Hold targets for ticks of 1/600 s, suitable for closed-loop control."""
        return self._request(
            "step",
            joint_targets=joint_targets or {},
            opening_m=opening_m,
            left_opening_m=left_opening_m,
            ticks=ticks,
        )

    def solve_ik(self, position_m, quaternion_xyzw=None, side="right"):
        """Plan a hand-frame target; does not move the robot."""
        return self._request("ik", position_m=position_m, quaternion_xyzw=quaternion_xyzw, side=side)

    def render(self, path):
        """Save a camera image without advancing physics."""
        return self._request("render", path=str(Path(path).resolve()))

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.process.stdout.close()
            self.process.stdin.close()
            self.process = None
            self.log.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def worker():
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
    sys.path.insert(0, str(ROOT / "runtime"))
    from native_rigid_world import NativeRigidWorld as IsaacNewtonWorld
    import numpy as np
    from scipy.spatial.transform import Rotation

    world = IsaacNewtonWorld(ROOT, app)

    def observe(include_bag_nodes=False):
        result = world.measure()
        poses = world.state.body_q.numpy()
        result.update(
            box_pose_xyzw=poses[list(world.model.body_label).index("/World/DashCarton")].tolist(),
            commanded_joints=world.target_posture,
            opening_target_m=world.target_opening,
            joint_limits=world.kin.limits,
            timestep_s=world.dt,
        )
        if include_bag_nodes:
            result["bag_nodes_m"] = world.state.particle_q.numpy().tolist()
        return result

    def reply(value):
        print(PREFIX + json.dumps(dict(result=value)), flush=True)

    reply(observe())
    for line in sys.stdin:
        try:
            command = json.loads(line)
            op = command["operation"]
            if op == "observe":
                reply(observe(command.get("include_bag_nodes", False)))
                continue
            if op == "render":
                reply(world.render(command["path"]))
                continue
            if op == "ik":
                side = command.get("side", "right")
                if side not in ["left", "right"]:
                    raise ValueError("Unknown arm")
                xyz = np.asarray(command["position_m"], float)
                if xyz.shape != (3,) or not np.isfinite(xyz).all():
                    raise ValueError("Invalid position")
                seed = world.target_posture
                if command.get("quaternion_xyzw") is None:
                    q, error = world.kin.solve_position(xyz, seed, side=side)
                else:
                    target = np.eye(4)
                    target[:3, :3] = Rotation.from_quat(command["quaternion_xyzw"]).as_matrix()
                    target[:3, 3] = xyz
                    q, error = world.kin.solve(target, seed, side=side)
                reply(dict(joint_targets={n: q[n] for n in q if n.startswith(side + '_joint_')}, error=error))
                continue
            if op not in ["move", "step"]:
                raise ValueError("Unknown operation")
            start = world.target_posture.copy()
            target = start.copy()
            for name, value in command.get("joint_targets", {}).items():
                if name not in target or "finger" in name:
                    raise ValueError(f"Use opening_m for gripper control; unknown arm joint {name}")
                lo, hi = world.kin.limits[name]
                if not np.isfinite(value) or not lo <= value <= hi:
                    raise ValueError(f"Joint target outside source limits: {name}")
                target[name] = float(value)
            if command.get("left_opening_m") is not None:
                left = float(command["left_opening_m"])
                if not -0.003 <= left <= 0.05:
                    raise ValueError("Invalid left_opening_m")
                target.update(left_finger_joint1=-left, left_finger_joint2=left)
            opening = command.get("opening_m")
            opening = world.target_opening if opening is None else float(opening)
            if not -0.003 <= opening <= 0.05:
                raise ValueError("Invalid opening_m")
            ticks = (
                int(command.get("ticks", 1))
                if op == "step"
                else round(float(command.get("seconds", 1)) / world.dt)
            )
            if not 1 <= ticks <= round(10 / world.dt):
                raise ValueError("Command duration must be 1 tick to 10 seconds")
            start_opening = world.target_opening
            for i in range(ticks):
                u = (i + 1) / ticks if op == "move" else 1.0
                u = u * u * (3 - 2 * u)
                world.step(
                    {n: start[n] + u * (target[n] - start[n]) for n in start},
                    start_opening + u * (opening - start_opening),
                )
            reply(observe())
        except Exception as error:
            print(PREFIX + json.dumps(dict(error=f"{type(error).__name__}: {error}")), flush=True)
            if isinstance(error, RuntimeError):
                break
    app.close()


if __name__ == "__main__":
    if "--worker" in sys.argv:
        worker()
