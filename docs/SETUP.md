# Runtime and first run

Clone the tagged code repository and download its matching assets before using the SDK:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-client.txt
python scripts/fetch_assets.py
python verify_package.py
```

The fetch command verifies the pinned HF commit, each ZIP checksum, and every member. Existing matching files are retained; modified files are refused. `--offline-dir DIR` uses the same release ZIPs from local storage. `--component scene` downloads only the scene data for rigid-only use; full package verification expects both components. The runtime overlay is a separate dependency download, not model weights or demonstration data.


Use Linux x86-64 with an NVIDIA GPU, a compatible NVIDIA driver and a **pip-based Isaac Sim 6.2 environment with Python 3.12**. The original native checks used an RTX 4090 (24 GB) and driver 595.91.07. These are recorded test conditions, not a benchmark of the minimum hardware requirements. macOS can inspect recipes and convert grasp files but cannot run this native simulator package.

`ISAAC_PYTHON` must name the environment's Python executable, not the system Python and not a shell launcher. Isaac Sim is not included or installed by the SDK. Provision it using your Isaac Sim distribution, then use the supplied setup command to check compatibility. Each backend's `runtime-lock.json` records the tested environment. The enforced core pins are Isaac Sim `6.2.0.0`; the soft variant also requires Warp `1.17.0`, Newton `1.7.0`, MuJoCo `3.15.0` and MuJoCo-Warp `3.15.0`.

Newton 1.7.0 is pinned because this alpha preserves the tested SDK snapshot and its exact solver-file checksums. It is not a claim that 1.7.0 is the newest Newton release. A newer Newton version needs a separately validated compatibility and physics update; installing it over this snapshot will not reproduce this release.

Run from the cloned code repository after fetching both components:

```bash
python3 verify_package.py
export ISAAC_PYTHON=/absolute/path/to/isaac-environment/bin/python
python3 -m titus setup --backend rigid
python3 -m titus setup --backend soft
python3 -m titus reference --backend rigid --out runs/reference_rigid
```

The outer `python3` runs the lightweight SDK interface (Python 3.10+); native simulation uses `ISAAC_PYTHON` in a subprocess. Keep the two roles distinct. Setup verifies versions, asset hashes and USD composition. It does not run a grasp or certify GPU operation. The reference command executes and scores the previously qualified fixed task.

The soft solver overlay (Linux, Python 3.12) is supplied once at `solver/site` and shared by the soft backend and the soft reference task. If it is absent, soft setup installs the pinned packages there via pip or uv; mismatched existing packages fail preflight. `TITUS_SOLVER_SITE` can override the overlay path. Do not overwrite the robot's environment with model dependencies; keep GraspGenX/OpenPI in their own process/environment and exchange observations/actions or proposal files.

For new scenes, first run the small hold-target example:

```bash
python3 -m examples.randomized_policy --backend rigid --seed 42 \
  --bags 1 --boxes 1 --out runs/first_scene
```

This checks the recipe/control/camera workflow. It does not pack the bag. Edit the callback for your model, then increase counts or randomization ranges. The left carton-hold initialization is declared and logged; use `initialize_box_hold=false` in a recipe when the policy must acquire the carton itself.

`python3 -m titus populate ...` creates a disk instance for inspecting USD/configuration. It does not launch the simulator. For normal execution, `Scene.from_recipe(...)` creates and loads its own instance directly. There is no public `Scene.from_directory` loader in this release; reproduce a populated configuration by using its recipe, seed and count overrides.

If a native process fails, inspect the simulator log in the generated instance or reference output directory. A package hash failure is an integrity issue; a version error is an environment issue; an evaluated failed grasp is a task outcome. Preserve those distinctions when reporting results.
