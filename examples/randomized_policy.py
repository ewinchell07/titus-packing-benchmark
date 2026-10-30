"""Populate a seeded scene and connect an LLM/VLA policy callback.

The example policy only holds current targets. It is an interface example and
does not claim packing success. Replace policy() with your model adapter.
"""

import argparse
from pathlib import Path

from titus import PlacementEvaluator, PolicyRunner, Scene
from titus.paths import ROOT


def policy(observation):
    # Camera path, measured joints, commanded joints, gripper targets and clock.
    # Model-specific image preprocessing and action normalization belong here.
    return {"ticks": 20}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", default=str(ROOT/"recipes/randomized_packing.json"))
    parser.add_argument("--backend", choices=("rigid", "soft"), default="soft")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bags", type=int)
    parser.add_argument("--boxes", type=int)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    with Scene.from_recipe(args.recipe, seed=args.seed, backend=args.backend, bags=args.bags,
                           boxes=args.boxes, workspace=args.out/"scene_instances") as scene:
        evaluator = PlacementEvaluator(scene)
        runner = PolicyRunner(scene, args.out/"episode", policy_name="hold-target-example",
                              timing="freeze", privileged=False, evaluator=evaluator)
        for _ in range(args.steps):
            runner.run_step(policy)
        print(runner.last_evaluation)


if __name__ == "__main__":
    main()
