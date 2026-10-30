"""Common scene and robot motion API; native physics runs in a subprocess."""

import importlib.util
from pathlib import Path
import tempfile

from .paths import backend_root, task_info


class Scene:
    """Open one task with the original calibrated robot.

    observe() exposes simulator truth for diagnostics and reference controllers.
    Use PolicyRunner for camera/proprioception-only policy observations.
    Commands affect robot joints. IK plans a hand pose but does not guarantee
    collision-free motion. Rendering and bag selection do not advance physics.
    """

    def __init__(self, backend="soft", task="single_bag", log_path="simulator.log", *, _instance=None):
        self.variant = backend
        self.task = task
        self.instance = _instance
        self.log_path = Path(log_path).resolve()
        self.recipe_manifest = _instance.manifest if _instance else None
        self.task_info = (dict(_instance.manifest["counts"], status="generated")
                          if _instance else task_info(task))
        root = _instance.root if _instance else backend_root(task, backend)
        spec = importlib.util.spec_from_file_location(
            f"titus_worker_{task}_{backend}", root / "titus_scene.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.client = module.TitusScene(Path(log_path))

    @classmethod
    def from_recipe(cls, recipe, *, seed=0, backend="soft", workspace="runs/scenes",
                    bags=None, boxes=None):
        """Populate a seeded scene and run its declared robot initialization."""
        from .scene_builder import materialize
        workspace = Path(workspace).resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        folder = Path(tempfile.mkdtemp(prefix=f"seed_{seed}_", dir=workspace))
        instance = materialize(recipe, folder / "scene", backend=backend,
                               seed=seed, bags=bags, boxes=boxes)
        scene = cls(backend, "recipe", folder / "simulator.log", _instance=instance)
        try:
            if instance.manifest["initialize_box_hold"]:
                scene.prepare_box_hold()
        except BaseException:
            scene.close()
            raise
        return scene

    def prepare_box_hold(self):
        """Physically close/lift the prepositioned carton using robot commands.

        This declared initialization is outside the evaluated policy. It writes
        no object pose and adds no support, weld or carrying force.
        """
        observation = self.move(left_opening_m=-.003, seconds=.2)
        hand = observation["left_hand_pose"]
        position = list(hand[:3])
        position[2] += .02
        plan = self.solve_ik(position, hand[3:], side="left")
        self.move(plan["joint_targets"], left_opening_m=-.003, seconds=1.)
        result = self.step(ticks=300)
        if self.instance:
            import json
            receipt = dict(scope="Declared robot-only reset initialization",
                           actions=["close left gripper for 0.2 s", "lift left hand 0.02 m over 1 s",
                                    "hold targets for 0.5 s"], final_time_s=result["t_s"],
                           object_pose_commands=0, object_force_commands=0)
            (self.instance.root / "initialization.json").write_text(json.dumps(receipt, indent=2) + "\n")
        return result

    @staticmethod
    def normalize(observation):
        result = dict(observation)
        commands = dict(result["commanded_joints"])
        opening = result["opening_target_m"]
        commands.update(right_finger_joint1=-opening, right_finger_joint2=opening)
        result["commanded_joints"] = commands
        result["left_opening_target_m"] = -commands["left_finger_joint1"]
        return result

    def observe(self, include_bag_nodes=False):
        return self.normalize(self.client.observe(include_bag_nodes=include_bag_nodes))

    def reset(self, seed=None):
        """Start a fresh native process at the declared fixed reset."""
        if seed is not None:
            if self.instance is None:
                raise ValueError("Seeded resets require Scene.from_recipe")
            previous = self.instance
            self.close()
            replacement = type(self).from_recipe(
                previous.recipe_path, seed=seed, backend=self.variant,
                workspace=previous.root.parents[1], **previous.manifest["count_overrides"],
            )
            self.__dict__.update(replacement.__dict__)
            return self.observe()
        result = self.normalize(self.client.reset())
        if self.instance and self.instance.manifest["initialize_box_hold"]:
            return self.prepare_box_hold()
        return result

    def select_bag(self, index):
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("Bag index must be an integer")
        if not 0 <= index < self.task_info["bags"]:
            raise ValueError("Bag index is outside this task's object count")
        if self.task_info["bags"] == 1:
            return self.observe()
        return self.normalize(self.client.select_bag(index))

    def step(self, joint_targets=None, opening_m=None, ticks=20, left_opening_m=None):
        """Apply robot targets for integer 1/600 s physics ticks."""
        if isinstance(ticks, bool) or not isinstance(ticks, int) or not 1 <= ticks <= 6000:
            raise ValueError("ticks must be an integer from 1 to 6000")
        return self.normalize(self.client.step(
            joint_targets=joint_targets, opening_m=opening_m, ticks=ticks,
            left_opening_m=left_opening_m,
        ))

    def move(self, joint_targets=None, opening_m=None, seconds=1.0, left_opening_m=None):
        """Interpolate joint targets; gripper values are one finger's travel."""
        return self.normalize(self.client.move(
            joint_targets=joint_targets, opening_m=opening_m, seconds=seconds,
            left_opening_m=left_opening_m,
        ))

    def solve_ik(self, position_m, quaternion_xyzw=None, side="right"):
        """Solve the native hand-frame WORLD pose (metres, quaternion XYZW).

        This is not a fingertip/TCP/grasp pose. Convert that pose with the
        declared hand-to-grasp transform first; see docs/GRASPGENX.md.
        IK enforces joint limits and pose residuals, not collision-free paths.
        """
        return self.client.solve_ik(position_m, quaternion_xyzw, side)

    def render(self, path):
        return self.client.render(path)

    def render_views(self, path):
        """Capture synchronized external and two wrist RGB images.

        Wrist mounts and optics are simulation settings, not measured camera
        calibration. A sibling camera_metadata.json records their provenance.
        """
        primary = Path(self.render(path))
        views = {"cam_high": primary}
        for side in ("left", "right"):
            views[f"cam_{side}_wrist"] = primary.with_name(primary.stem + f"_{side}_wrist.png")
        missing = [str(value) for value in views.values() if not value.is_file()]
        if missing:
            raise RuntimeError(f"Missing camera output: {missing}")
        return {key: str(value) for key, value in views.items()}

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
