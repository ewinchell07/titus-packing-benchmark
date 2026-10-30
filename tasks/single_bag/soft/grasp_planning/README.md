# Optional GraspGen-X proposals

This adapter runs NVIDIA GraspGen-X against the observed bag and the delivered
YAM gripper geometry. It generates poses; the Isaac/Newton simulator must still
test the approach, grasp, lift and release. Model scores are not success rates.
The deterministic reference case does not depend on this model.

The input exporter uses all 52 collision pieces from the delivered hand and
fingers, and a swept volume from its existing effective pad landmarks and sizes.
It exports the deformed printed bag surface, the carton, the dish and the left
arm as observed in a recorded physical episode. It does not replace the robot.

Install inference into its own Linux/NVIDIA Python 3.11 environment (requires
`git` and `uv`). The script fetches approximately 1.7 GB of public checkpoints,
pinned by revision and SHA-256. Isaac's Python remains separate.

```bash
export GRASPGENX_DIR=/absolute/path/to/new/graspgenx
bash grasp_planning/setup.sh

"$ISAAC_PYTHON" grasp_planning/export_inputs.py evidence/qualification/reset_1 \
  --time 4 --out runs/grasp_input
"$GRASPGENX_DIR/.venv/bin/python" grasp_planning/generate.py \
  --repo "$GRASPGENX_DIR" --checkpoints "$GRASPGENX_DIR/checkpoints" \
  --input runs/grasp_input --out runs/proposals

bash run_pack.sh --grasp-candidates runs/proposals/candidates.json \
  --candidate-rank 13 --stop-after bag_hold --out runs/candidate_trial
"$ISAAC_PYTHON" grasp_planning/score_hold.py runs/candidate_trial
```

Choose an eligible rank from your generated file; rank 13 is only an example,
not a promised successful grasp. The proposal filter checks sampled gripper-to-
scene clearance, at least 10 observed surface points inside the jaw closure
volume, and native joint-limit IK at the approach and grasp poses. It
does not establish a collision-free full-arm path or a stable deformable grasp.
The physical trial can reject either. The lift controller's reference turn was
designed for its sealed-end grasp; other proposals can need a different turn.

The exported convention maps GraspGen-X's closing axis to native hand Y and
approach axis to native hand Z. `hand_from_graspgen.json` records the exact frame
conversion. All robot commands use the original native drives. The optional
model is not loaded into the simulator and cannot attach or reposition objects.

Upstream: https://github.com/NVlabs/GraspGenX. Source and model revisions are in
`model-lock.json`. Consult upstream license/model terms when redistributing.

## Recorded result in this package

The qualified scene produced 512 proposals; 81 passed sampled scene clearance,
and three also passed closure-volume occupancy and native IK checks. The highest
eligible candidate (rank 244) did not pass the physical lift-and-hold trial.
See `evidence/graspgenx/physical_probe/`. The reference packing controller uses
its separately validated grasp. Generated candidates remain experimental.
