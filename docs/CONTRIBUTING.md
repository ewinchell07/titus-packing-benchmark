# Contributing to the alpha

Open a GitHub issue or pull request on `ewinchell07/titus-packing-benchmark` with the integration you are building or a minimal reproduction. Small, auditable contributions are especially useful:

- Model adapters with explicit image order, normalization, joint mapping, gripper units, and timing.
- Measured camera and robot calibration with frame definitions and provenance.
- Validated bimanual UMI-to-robot registration and retargeting, with the original data preserved.
- Contact-stability fixes demonstrated with unchanged failure checks and clearly recorded solver/material changes.
- New physical assets with provenance, units, collision geometry, and redistribution terms.
- Reproducible task/evaluator protocols and honest separation of invalid episodes from policy failures.

Provide the release hash, recipe, seed, backend/runtime versions, GPU, commands, and relevant logs. Do not include API keys, access tokens, private host addresses, or unrelated recordings. Keep policy observations separate from evaluator truth. Label privileged controllers and oracle experiments explicitly.

The alpha has no promised support SLA or finalized leaderboard. Propose protocol changes before comparing scores across different reset, sensing, timing, or scoring contracts.
