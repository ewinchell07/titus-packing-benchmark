import unittest

from titus.goals import PlacementEvaluator, _inverse_rotate


class ObservedScene:
    recipe_manifest = {
        "configuration_sha256": "test-scene",
        "actors": [
            {"id": "bag_001", "kind": "bag", "bag_index": 0},
            {"id": "box_002", "kind": "box", "asset": "lundberg_carton", "prim_path": "/World/Box_2"},
        ],
        "goal_assignments": {"bag_001": "box_002"},
    }

    def __init__(self):
        self.t = 0.
        self.contact = .35
        self.finger_force = 0.

    def select_bag(self, index):
        assert index == 0

    def observe(self, include_bag_nodes=False):
        return {
            "t_s": self.t, "selected_bag_index": 0,
            "objects": {"box_002": {"pose_xyzw": [1, 2, 1, 0, 0, 0, 1]}},
            "bag_nodes_m": [[.98, 2, 1.05], [1.02, 2.02, 1.1]],
            "container_contact_force_N": {"/World/Box_2": self.contact, "/World/DashCarton": .8},
            "finger_contact_force_N": [self.finger_force, self.finger_force],
            "max_speed_m_s": 0., "support_contact_force_N": 0.,
        }


class GoalTests(unittest.TestCase):
    def test_requires_dwell_and_assigned_container_contact(self):
        scene = ObservedScene()
        evaluator = PlacementEvaluator(scene)
        for tick in range(21):
            scene.t = tick/10
            result = evaluator.update()
        self.assertTrue(result["passed"])
        scene.t += .1
        scene.contact = 0.
        self.assertFalse(evaluator.update()["passed"])

    def test_missing_samples_and_gripper_support_do_not_pass(self):
        scene = ObservedScene()
        evaluator = PlacementEvaluator(scene)
        evaluator.update()
        scene.t = 2.
        self.assertFalse(evaluator.update()["passed"])
        scene.finger_force = .1
        for tick in range(1, 25):
            scene.t = 2+tick/10
            self.assertFalse(evaluator.update()["passed"])

    def test_box_frame_rotation(self):
        q = [0, 0, 2**-.5, 2**-.5]
        point = _inverse_rotate([0, 1, 0], q)
        self.assertAlmostEqual(point[0], 1.)
        self.assertAlmostEqual(point[1], 0.)


if __name__ == "__main__":
    unittest.main()
