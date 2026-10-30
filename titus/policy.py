"""Policy observation boundary, action chunks, inference timing and episode logs."""

import json
import math
from pathlib import Path
import time


class PolicyRunner:
    """Evaluate a policy callback independently of the reference controller.

    freeze pauses physics during inference. held_command advances physics for
    the measured inference duration under the previous command before applying
    the returned chunk. Rendering and IPC latency are excluded.
    """

    def __init__(self, scene, out, *, policy_name, timing="freeze", privileged=False,
                 grasp_assistance="none", capture_rgb=True, episode_metadata=None, evaluator=None,
                 camera_views=("cam_high", "cam_left_wrist", "cam_right_wrist")):
        if timing not in ("freeze", "held_command"):
            raise ValueError("Choose freeze or held_command timing")
        self.scene = scene
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=False)
        self.timing = timing
        self.privileged = privileged
        self.capture_rgb = capture_rgb
        if not camera_views or set(camera_views) - {"cam_high", "cam_left_wrist", "cam_right_wrist"}:
            raise ValueError("Choose external and/or left/right wrist camera keys")
        self.camera_views = tuple(camera_views)
        self.index = 0
        self.evaluator = evaluator
        self.last_evaluation = None
        info = scene.task_info
        resolved_scene = getattr(scene, "recipe_manifest", None)
        metadata = dict(
            task=scene.task, task_status=info["status"], variant=scene.variant,
            policy=policy_name, timing_mode=timing,
            privileged_simulator_state=privileged, grasp_assistance=grasp_assistance,
            scene_counts={"bags": info["bags"], "boxes": info["boxes"]},
            reset="seeded recipe" if resolved_scene is not None else "fixed declared task",
            seed_randomization=resolved_scene is not None,
            task_success="Apply the independent evaluator; this logger does not infer success",
            action_contract="Named joint positions in radians; finger travel in metres; integer 1/600 s ticks",
            user_metadata=episode_metadata or {},
            resolved_scene=resolved_scene,
            camera_views=list(self.camera_views) if capture_rgb else [],
        )
        (self.out / "episode.json").write_text(json.dumps(metadata, indent=2) + "\n")

    def _hold(self, ticks):
        while ticks:
            count = min(ticks, 6000)
            self.scene.step(ticks=count)
            ticks -= count

    def run_step(self, policy):
        before = self.scene.observe()
        fields = ("t_s", "actual_joint_q", "commanded_joints", "opening_target_m",
                  "left_opening_target_m", "timestep_s")
        observation = {key: before[key] for key in fields}
        if self.capture_rgb:
            views = self.scene.render_views(self.out / f"rgb_{self.index:06d}.png")
            observation["rgb_paths"] = {key: views[key] for key in self.camera_views}
            if "cam_high" in self.camera_views:
                observation["rgb_path"] = views["cam_high"]
        if self.privileged:
            observation["simulator_state"] = before
        start = time.monotonic()
        answer = policy(observation)
        latency = time.monotonic() - start
        actions = answer if isinstance(answer, list) else [answer]
        if not actions:
            raise ValueError("Policy returned an empty action chunk")
        actions = [dict(action) for action in actions]
        for action in actions:
            if set(action) - {"joint_targets", "opening_m", "left_opening_m", "ticks"}:
                raise ValueError("Unknown action field")
            ticks = action.get("ticks", 20)
            if isinstance(ticks, bool) or not isinstance(ticks, int) or not 1 <= ticks <= 6000:
                raise ValueError("ticks must be an integer from 1 to 6000")
            action["ticks"] = ticks
        delay = math.ceil(latency / before["timestep_s"]) if self.timing == "held_command" else 0
        self._hold(delay)
        action_start = self.scene.observe()["t_s"]
        for action in actions:
            after = self.scene.step(**action)
        if self.evaluator is not None:
            self.last_evaluation = self.evaluator.update()
        record = dict(
            index=self.index, observation=observation, inference_wall_s=latency,
            modeled_delay_ticks=delay, actions=actions, action_start_sim_s=action_start,
            final_sim_s=after["t_s"], simulator_truth_before=before, simulator_truth_after=after,
            evaluation=self.last_evaluation,
        )
        with (self.out / "steps.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        self.index += 1
        return after
