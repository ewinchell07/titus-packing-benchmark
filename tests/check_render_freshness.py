"""Native regression: camera geometry follows robot motion without stepping physics."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from titus import Scene
from titus.paths import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("rigid", "soft"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    with Scene.from_recipe(ROOT / "recipes/single_bag.json", backend=args.backend,
                           seed=7, workspace=args.out / "scenes") as scene:
        start = scene.observe()
        views = scene.render_views(args.out / "before.png")
        assert set(views) == {"cam_high", "cam_left_wrist", "cam_right_wrist"}
        for key, path in views.items():
            expected = (1024, 768) if key == "cam_high" else (256, 256)
            assert Image.open(path).size == expected
        metadata = json.loads((args.out / "before_camera_metadata.json").read_text())
        assert metadata["sim_t_s"] == start["t_s"]
        scene.render_views(args.out / "same_state.png")
        after_render = scene.observe()
        assert after_render["t_s"] == start["t_s"]
        assert after_render["native_elapsed_s"] == start["native_elapsed_s"]
        assert after_render["actual_joint_q"] == start["actual_joint_q"]
        joint = "right_joint_1"
        targets = {joint: start["commanded_joints"][joint] + 0.7}
        scene.move(targets, seconds=1.5)
        moved = scene.observe()
        scene.render_views(args.out / "after.png")
        final = scene.observe()
        assert final["t_s"] == moved["t_s"]
        assert final["native_elapsed_s"] == moved["native_elapsed_s"]
        assert final["actual_joint_q"] == moved["actual_joint_q"]
        pixels = [np.asarray(Image.open(args.out / name).convert("RGB"), dtype=float)
                  for name in ("before.png", "same_state.png", "after.png")]
        noise = float(np.mean(np.max(np.abs(pixels[1] - pixels[0]), axis=2) > 20))
        changed = float(np.mean(np.max(np.abs(pixels[2] - pixels[1]), axis=2) > 20))
        actual = abs(moved["actual_joint_q"][joint] - start["actual_joint_q"][joint])
        passed = actual > 0.5 and changed > max(0.005, 5 * noise + 0.001)
        receipt = dict(passed=passed, backend=args.backend, actual_motion_rad=actual,
                       camera_keys=list(views), synchronized_camera_metadata=True,
                       same_state_changed_pixel_fraction=noise,
                       moved_changed_pixel_fraction=changed,
                       rendering_advances_physics=False,
                       scope="Camera freshness after measured robot motion; not packing qualification")
        (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt), flush=True)
        assert passed, "Rendered geometry did not follow measured robot motion"


if __name__ == "__main__":
    main()
