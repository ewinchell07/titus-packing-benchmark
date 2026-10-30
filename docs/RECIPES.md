# Recipe contract

Use `recipes/randomized_packing.json` as the editable template. All coordinates are metres in the original Z-up workcell.

| Field | Meaning |
| --- | --- |
| `bags`, `boxes` | Integer, `{"choices": [1, 3, 6]}`, or `{"min": 1, "max": 6}`. Boxes include the primary carton. CLI overrides are available. |
| `bag_region_xy_m` | X and Y sampling intervals for bag centres. |
| `box_region_xy_m` | X and Y sampling intervals for spare cartons. The primary left-grasp carton stays fixed. |
| `bag_yaw_deg`, `box_yaw_deg` | Finite ordered yaw intervals. |
| `max_bag_layers` | Number of available reset layers when one layer cannot fit all bags. Objects are free and settle physically. |
| `layer_gap_m`, `placement_gap_m` | Positive separation between reset layers and nonnegative XY separation. |
| `dish_radius_m` | Maximum XY bounding-corner radius for bags (including fixed poses) and spare cartons, at most 0.60 m. The primary held carton has its fixed separate pose. |
| `fixed_bags` | Optional exact list of `center_xy_m` and `yaw_deg`; its length must match the count. |
| `goal` | `pack_into_primary` or round-robin `distribute` across declared cartons. |
| `camera` | `eye_m`, `look_at_m`, `focal_length_mm`; each scalar may be fixed or `[min,max]`. |
| `lighting` | `dome_intensity`, `key_intensity`, fixed or `[min,max]`. |
| `initialize_box_hold` | Whether reset performs the recorded robot-only primary-carton grip/lift sequence. |

The sampler rejects initial object bounding-box overlap. It does not certify robot reachability, collision-free manipulation, pile stability, grasp quality or policy success. Impossible placement bounds fail with a clear error rather than silently moving objects outside the recipe.

`recipes/single_bag.json` is a small generic fixed-pose recipe. The separately qualified exact reference is selected with `python3 -m titus reference`; do not treat the similarly named recipe as another qualified execution.

The primary carton and calibrated robot physical parameters are not randomized. Robot initialization uses recorded reachable joint positions from the original calibrated robot. New physics randomizations should be explicit additions with their own metadata, not hidden changes to calibration.

Generated folders reference the SDK's shared resources. They are working instances, not standalone asset bundles. To reproduce on another machine, retain this SDK version, the recipe, resolved `population.json` and seed, then regenerate the instance. Each episode log includes the resolved population and count overrides. `Scene.reset(seed=...)` creates a fresh process and generated instance.

`population.json` has stable `bag_001`/`box_001` IDs and a `goal_assignments` mapping. Diagnostic observations expose these same IDs in `objects`. A configuration digest identifies the sampled setup; the backend, runtime lock and source-manifest digest are recorded separately.
