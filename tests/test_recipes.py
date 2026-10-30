import copy
import json
from pathlib import Path
import tempfile
import unittest

from titus.paths import ROOT
from titus.scene_builder import materialize, resolve_recipe


class RecipeTests(unittest.TestCase):
    recipe = ROOT / "recipes/randomized_packing.json"

    def test_repeatable_population_and_counts(self):
        first = resolve_recipe(self.recipe, seed=42, bags=18, boxes=3)
        self.assertEqual(first, resolve_recipe(self.recipe, seed=42, bags=18, boxes=3))
        self.assertNotEqual(first["configuration_sha256"], resolve_recipe(self.recipe, seed=43, bags=18, boxes=3)["configuration_sha256"])
        self.assertEqual(first["counts"], {"bags": 18, "boxes": 3})
        self.assertEqual(len(first["actors"]), 21)
        self.assertEqual(len({a["id"] for a in first["actors"]}), 21)
        self.assertEqual(set(first["goal_assignments"].values()), {"box_001", "box_002", "box_003"})
        for index, a in enumerate(first["actors"]):
            for b in first["actors"][index+1:]:
                overlap = all(a["bounds_m"][0][i] < b["bounds_m"][1][i]
                              and b["bounds_m"][0][i] < a["bounds_m"][1][i] for i in range(3))
                self.assertFalse(overlap, (a["id"], b["id"]))

    def test_materialized_scene_has_lineage_and_shared_resources(self):
        with tempfile.TemporaryDirectory() as directory:
            result = materialize(self.recipe, Path(directory)/"scene", backend="rigid", seed=7, bags=3, boxes=2)
            self.assertEqual(json.loads((result.root/"population.json").read_text())["counts"], {"bags": 3, "boxes": 2})
            self.assertTrue((result.root/"assets/robot").is_dir())
            self.assertEqual(len(result.manifest["calibrated_robot_files_sha256"]), 13)
            stage = (result.root/"scene/multi_bag_box.usda").read_text()
            self.assertIn('upAxis = "Z"', stage)
            self.assertIn('"Bag_3"', stage)
            self.assertIn('"Box_2"', stage)
            with self.assertRaises(FileExistsError):
                materialize(self.recipe, result.root, seed=7)

    def test_invalid_randomization_is_not_silently_accepted(self):
        original = json.loads(self.recipe.read_text())
        with tempfile.TemporaryDirectory() as directory:
            for field, value in [("robot_stiffness", 0), ("bag_yaw_deg", [float("nan"), 20]),
                                 ("layer_gap_m", -1), ("bags", 0), ("goal", "unknown")]:
                recipe = copy.deepcopy(original)
                recipe[field] = value
                path = Path(directory)/"invalid.json"
                path.write_text(json.dumps(recipe))
                with self.assertRaises(ValueError):
                    resolve_recipe(path)

    def test_fixed_bags_obey_dish_boundary(self):
        recipe = json.loads((ROOT / "recipes/single_bag.json").read_text())
        resolve_recipe(ROOT / "recipes/single_bag.json")
        recipe["fixed_bags"][0]["center_xy_m"] = [2., 0.]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "outside.json"
            path.write_text(json.dumps(recipe))
            with self.assertRaisesRegex(ValueError, "dish_radius_m"):
                resolve_recipe(path)


if __name__ == "__main__":
    unittest.main()
