"""One entry point for setup, reference runs, task status and evaluation."""

import argparse
import json
import subprocess

from .controllers import run_reference
from .evaluation import evaluate
from .paths import BACKENDS, backend_root, population_backend_root, tasks


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m titus")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="Show task counts and qualification status")
    grasps = commands.add_parser("grasps", help="Convert supplied GraspGenX poses; does not move the robot")
    grasps.add_argument("proposals")
    grasps.add_argument("--context", required=True)
    grasps.add_argument("--out", required=True)
    grasps.add_argument("--min-confidence", type=float)
    grasps.add_argument("--top-k", type=int, default=20)
    population = commands.add_parser("populate", help="Create a seeded scene without starting a simulator")
    population.add_argument("recipe")
    population.add_argument("--seed", type=int, default=0)
    population.add_argument("--backend", choices=BACKENDS, default="soft")
    population.add_argument("--bags", type=int)
    population.add_argument("--boxes", type=int)
    population.add_argument("--out", required=True)
    for name in ("setup", "reference", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--task", choices=tuple(tasks()), default=None if name == "setup" else "single_bag")
        command.add_argument("--backend", choices=BACKENDS, default="soft")
        if name == "reference":
            command.add_argument("--out", required=True)
            command.add_argument("--stop-after", choices=("box_hold", "bag_hold", "vertical", "over_box"))
            command.add_argument("--no-close", action="store_true")
        if name == "evaluate":
            command.add_argument("episode")
    args = parser.parse_args(argv)
    if args.command == "grasps":
        from pathlib import Path
        from .grasps import convert_graspgenx
        converted = convert_graspgenx(json.loads(Path(args.proposals).read_text()),
                                     json.loads(Path(args.context).read_text()),
                                     min_confidence=args.min_confidence, top_k=args.top_k)
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(converted, indent=2, allow_nan=False) + "\n")
        print(json.dumps(dict(output=str(output.resolve()), selected_count=converted["selected_count"],
                              scope=converted["scope"]), indent=2))
        return 0
    if args.command == "list":
        print(json.dumps(tasks(), indent=2))
        return 0
    if args.command == "populate":
        from .scene_builder import materialize
        instance = materialize(args.recipe, args.out, backend=args.backend, seed=args.seed,
                               bags=args.bags, boxes=args.boxes)
        print(json.dumps(dict(scene=str(instance.root), counts=instance.manifest["counts"],
                              seed=args.seed, configuration_sha256=instance.manifest["configuration_sha256"]), indent=2))
        return 0
    if args.command == "setup":
        root = backend_root(args.task, args.backend) if args.task else population_backend_root(args.backend)
        return subprocess.run(["bash", str(root / "setup.sh")], cwd=root).returncode
    if args.command == "reference":
        return run_reference(args.task, args.backend, args.out, stop_after=args.stop_after,
                             no_close=args.no_close)
    return evaluate(args.task, args.backend, args.episode)
