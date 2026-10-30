"""Contract checks for external proposals; these are not native grasp trials."""

import copy
import json
import math
from pathlib import Path
import unittest

from titus.grasps import convert_graspgenx, opening_from_jaw_gap, require_fresh_capture
from titus.paths import ROOT


class GraspTests(unittest.TestCase):
    def setUp(self):
        folder = ROOT / "examples/graspgenx"
        self.result = json.loads((folder / "synthetic_proposals.json").read_text())
        self.context = json.loads((folder / "synthetic_context.json").read_text())

    def test_composes_source_and_inverse_hand_offset(self):
        converted = convert_graspgenx(self.result, self.context)
        grasp = converted["candidates"][0]
        # Fixture: source rotated 90 deg about world Z, grasp offset 0.1 source X,
        # and hand->grasp offset 0.08 local Z. Pregrasp offset is -0.06 local Z.
        for actual, expected in zip(grasp["grasp_hand_pose"]["position_m"], [.4, .3, .82]):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(grasp["pregrasp_hand_pose"]["position_m"][2], .76)
        q = grasp["grasp_hand_pose"]["quaternion_xyzw"]
        self.assertAlmostEqual(abs(q[2]), math.sqrt(.5))
        self.assertAlmostEqual(abs(q[3]), math.sqrt(.5))
        self.assertAlmostEqual(converted["pregrasp_opening_m"], .03)

    def test_rotated_hand_frame_and_180_degree_quaternion(self):
        self.context["world_from_source"] = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        for axis in range(3):
            transform = [[float(i==j) for j in range(4)] for i in range(4)]
            for i in range(3):
                transform[i][i] = 1 if i == axis else -1
            self.context["hand_from_grasp"] = transform
            q = convert_graspgenx(self.result, self.context)["candidates"][0]["grasp_hand_pose"]["quaternion_xyzw"]
            self.assertAlmostEqual(abs(q[axis]), 1.)
            self.assertAlmostEqual(q[3], 0.)

    def test_rejects_wrong_frames_scale_reflection_units_or_missing_context(self):
        contexts = []
        for key, value in [("units", "mm"), ("side", "both"), ("capture_time_s", float("nan")),
                           ("pregrasp_jaw_gap_m", .2), ("schema_version", True)]:
            context = copy.deepcopy(self.context); context[key] = value; contexts.append(context)
        for i, j, value in [(3,0,.2), (0,0,2), (0,1,1)]:
            context = copy.deepcopy(self.context);context["world_from_source"][i][j] = value;contexts.append(context)
        context = copy.deepcopy(self.context);del context["hand_from_grasp"];contexts.append(context)
        for context in contexts:
            with self.subTest(context=context), self.assertRaises(ValueError):
                convert_graspgenx(self.result, context)

    def test_scores_and_empty_results_are_explicit(self):
        low = copy.deepcopy(self.result["grasps"][0]);low[0][3] = .2
        self.result["grasps"].append(low);self.result["confidences"].append(.9)
        result = convert_graspgenx(self.result, self.context, min_confidence=.8, top_k=1)
        self.assertEqual(result["candidates"][0]["source_index"], 1)
        self.assertEqual(convert_graspgenx({"grasps": [], "confidences": []}, self.context)["selected_count"], 0)
        for invalid in ({"grasps": [], "confidences": [1.]},
                        dict(self.result, confidences=[float("nan"), .9]),
                        dict(self.result, gripper_name="different_hand")):
            with self.assertRaises(ValueError):
                convert_graspgenx(invalid, self.context)

    def test_capture_identity_and_age(self):
        bundle = convert_graspgenx(self.result, self.context)
        require_fresh_capture(bundle, 1.1, capture_id="synthetic-capture-001", max_age_s=.2)
        for time, identifier in [(1.3,"synthetic-capture-001"),(.9,"synthetic-capture-001"),(1.,"other-episode")]:
            with self.assertRaises(ValueError):
                require_fresh_capture(bundle, time, capture_id=identifier, max_age_s=.2)

    def test_jaw_gap_is_full_gap_not_one_finger(self):
        self.assertAlmostEqual(opening_from_jaw_gap(.0000894), -.003)
        self.assertAlmostEqual(opening_from_jaw_gap(.1060894), .05)
        with self.assertRaises(ValueError):
            opening_from_jaw_gap(0)


if __name__ == "__main__":
    unittest.main()
