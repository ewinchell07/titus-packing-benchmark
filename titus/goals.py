"""Independent sampled placement goals for recipe-generated scenes."""

import json
import math
from pathlib import Path

from .paths import ROOT


def _inverse_rotate(point, quaternion):
    length = math.sqrt(sum(v*v for v in quaternion))
    if not math.isfinite(length) or length < 1e-12:
        raise ValueError("Invalid container orientation")
    x, y, z = [-v/length for v in quaternion[:3]]
    w = quaternion[3]/length
    px, py, pz = point
    tx, ty, tz = 2*(y*pz-z*py), 2*(z*px-x*pz), 2*(x*py-y*px)
    return [px+w*tx+y*tz-z*ty, py+w*ty+z*tx-x*tz, pz+w*tz+x*ty-y*tx]


class PlacementEvaluator:
    """Require assigned containment, release, contact and a sampled stable dwell.

    This general goal does not impose the reference script's prescribed lift
    trajectory. Updates must be at most max_sample_gap_s apart during the dwell.
    Truth is used only by this evaluator, not sent to the policy callback.
    """

    def __init__(self, scene, *, dwell_s=2., max_sample_gap_s=.2):
        if scene.recipe_manifest is None:
            raise ValueError("PlacementEvaluator requires a recipe-generated Scene")
        if dwell_s <= 0 or max_sample_gap_s <= 0:
            raise ValueError("Dwell and sample-gap durations must be positive")
        self.scene = scene
        self.dwell_s = dwell_s
        self.max_gap = max_sample_gap_s
        self.assets = json.loads((ROOT / "recipes/assets.json").read_text())
        self.since = {}
        self.last_time = None
        self.configuration = None

    def update(self):
        manifest = self.scene.recipe_manifest
        before = self.scene.observe()
        time = before["t_s"]
        configuration = manifest["configuration_sha256"]
        if (configuration != self.configuration or self.last_time is None
                or time < self.last_time or time-self.last_time > self.max_gap+1e-8):
            self.since = {}
        self.configuration = configuration
        actors = {actor["id"]: actor for actor in manifest["actors"]}
        active = before["selected_bag_index"]
        results = {}
        try:
            for bag_id, box_id in manifest["goal_assignments"].items():
                self.scene.select_bag(actors[bag_id]["bag_index"])
                observation = self.scene.observe(include_bag_nodes=True)
                if abs(observation["t_s"]-time) > 1e-8:
                    raise RuntimeError("Physics advanced during goal observation")
                pose = observation["objects"][box_id]["pose_xyzw"]
                points = observation["bag_nodes_m"]
                local = [_inverse_rotate([p[i]-pose[i] for i in range(3)], pose[3:]) for p in points]
                bounds = self.assets[actors[box_id]["asset"]]["interior_bounds_m"]
                inside = sum(all(bounds[0][i]-1e-6 <= p[i] <= bounds[1][i]+1e-6 for i in range(3)) for p in local)
                fraction = inside/len(local) if local else 0.
                contact = observation["container_contact_force_N"].get(actors[box_id]["prim_path"], 0.)
                gates = dict(contained=fraction == 1., released=max(observation["finger_contact_force_N"]) <= .02,
                             settled=observation["max_speed_m_s"] <= .03, target_contact=contact >= .05,
                             no_table_support=observation["support_contact_force_N"] <= .02)
                if all(gates.values()):
                    self.since.setdefault(bag_id, time)
                else:
                    self.since.pop(bag_id, None)
                dwell = time-self.since.get(bag_id, time)
                results[bag_id] = dict(target=box_id, gates=gates, inside_fraction=fraction,
                                       target_contact_N=contact, observed_stable_s=dwell,
                                       passed=all(gates.values()) and dwell >= self.dwell_s-1e-8)
        finally:
            self.scene.select_bag(active)
        self.last_time = time
        return dict(passed=bool(results) and all(r["passed"] for r in results.values()),
                    time_s=time, configuration_sha256=configuration, objects=results,
                    scope="Sampled assigned placement goal, not the reference trajectory qualification")
