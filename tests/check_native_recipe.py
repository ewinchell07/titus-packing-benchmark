"""One bounded generated-scene reset/step/render smoke check, not a packing trial."""

import argparse
import json
from pathlib import Path

from titus import PlacementEvaluator, Scene
from titus.paths import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("rigid", "soft"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    with Scene.from_recipe(ROOT/"recipes/randomized_packing.json", seed=7, backend=args.backend,
                           bags=3, boxes=2, workspace=args.out/"scenes") as scene:
        before = scene.observe()
        assert before["bag_count"] == 3 and len(before["bags"]) == 3
        assert len(before["objects"]) == 5
        assert set(before["objects"]) == {"bag_001", "bag_002", "bag_003", "box_001", "box_002"}
        selected = scene.select_bag(2)
        assert selected["t_s"] == before["t_s"] and selected["objects"] == before["objects"]
        stepped = scene.step(ticks=60)
        assert abs(stepped["t_s"]-before["t_s"]-.1) < 1e-8
        outcome = PlacementEvaluator(scene).update()
        assert not outcome["passed"], "Unpacked reset was incorrectly scored as success"
        scene.render(args.out/"scene.png")
        assert scene.observe()["t_s"] == stepped["t_s"]
        receipt = dict(passed=True, backend=args.backend, counts={"bags": 3, "boxes": 2},
                       configuration_sha256=scene.recipe_manifest["configuration_sha256"],
                       calibrated_robot_files=13, selection_advances_physics=False,
                       rendering_advances_physics=False, initialization="declared physical robot grip/lift",
                       placement_goal_passed=False,
                       scope="Generated scene and control API smoke check; no packing-success claim")
        (args.out/"receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
        (args.out/"population.json").write_text(json.dumps(scene.recipe_manifest, indent=2)+"\n")
        print(json.dumps(receipt))


if __name__ == "__main__":
    main()
