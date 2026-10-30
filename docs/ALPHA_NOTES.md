# Alpha scope and reporting

## What this release establishes

The scene package contains executable SDK code, physical assets, both native backends, and a scripted reference. It is based on SDK 2026.10.28.8 with physical assets and runtime code unchanged. Alpha packaging adds a six-bag/one-carton starter recipe and public documentation. Internal debugging reports, campaign archives, and private infrastructure are excluded.

The supplied reference controller is a privileged single-bag task. Its existence and previous execution establish a useful installation check, not six-bag model success. The six-bag starter recipe is generated without physics during packaging; it is not a validated collection of reachable or solved benchmark seeds.

## Limits to account for

- **Soft physics:** local tetrahedron collapse or inversion can occur under contact. A short successful replay does not qualify every policy, seed, solver setting, or GPU. Keep native checks enabled; preserve failures as invalid simulation episodes.
- **Solver comparisons:** this package preserves the supplied solver defaults. Experimental 48-iteration settings and controller patches used elsewhere are not included or silently adopted.
- **Task configuration:** the alpha starter disables automatic carton holding; the upstream randomized recipe and fixed reference differ. Report initialization and recipe identity.
- **Evaluation:** use the shipped SDK evaluator contract in `EVALUATION.md`. The independent development campaign has different reset and scoring rules and is not the released benchmark protocol. No campaign scores or frozen train/test leaderboard are promised by this alpha.
- **Assets:** one blue Sea Salt bag and one carton representation. Additional copies, colors, and randomization are not independent SKUs. The dish is stationary and carton flaps are fixed.
- **Sensors:** RGB only; simulation wrist optics are engineering settings rather than measured hardware calibration. The optional factory background changes RGB and has no collision.
- **UMI:** no validated transform into this robot, no corresponding robot joints, and no third camera. Retargeting and observation alignment remain explicit integration work.
- **Reproducibility:** pin files, software, seed, backend, hardware, and task configuration. Seeded scene generation is reproducible; GPU physics is not guaranteed bitwise deterministic.

## Reporting an experiment

Include the alpha release and file hashes; recipe/configuration and seed; backend, simulator, solver, and GPU; model/checkpoint and training split; camera/action adapter; initialization; inference timing mode and action cadence; evaluator version and sampling/dwell; simulation time; and completion or invalidity reason. Distinguish released-and-contained bags from grasps, lifts, partial entry, and attempts. Report physical-invalid episodes separately from valid unsuccessful policies.

The packaging validation is CPU-only and recorded in `validation.json`. Native setup and the reference task should be run in your provisioned environment before interpreting policy results.
