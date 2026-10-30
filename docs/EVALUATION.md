# Evaluation contract

The environment, reference controller, policy and evaluator are separate layers. A reference pass demonstrates feasibility of one controller in one fixed case. It is not evidence that an LLM, VLA or WAM succeeds, nor that every generated seed is solved.

## Policy interface

`PolicyRunner` supplies synchronized external, left-wrist and right-wrist RGB paths, measured joint state, commanded joint state, gripper targets and simulation time. Use `camera_views=("cam_high",)` for external-only input. Camera mounting metadata, object geometry, poses, contact forces and evaluator truth are excluded from the callback by default. Set `privileged=True` only for an explicitly labeled oracle experiment. This is an observation contract, not a security sandbox against policy code that deliberately accesses the environment object.

A callback returns one action dictionary or a list of action dictionaries. Fields are `joint_targets`, `opening_m`, `left_opening_m`, and integer `ticks` (default 20). Joints use radians and fingers use metres of travel per finger. Nominal gap is 0.0060894 + 2 * opening_m. Physics runs at 600 Hz; 20 ticks gives 30 Hz commands. Robot gains, joint limits, passive finger linkage and effort limits remain active.

Each VLA requires its own image preprocessing, action representation, normalization, embodiment mapping and checkpoint adapter. This SDK provides the environment boundary; it does not make ALOHA, DROID or another robot's policy directly compatible with YAM. The separate Pi0.5 experiment is not a dependency or a qualified controller in this package.

## Timing and records

`freeze` pauses simulation while model inference runs. `held_command` measures inference time, advances physics for the corresponding rounded-up ticks under the previous command, then applies the action chunk. Rendering and IPC latency are excluded; this is not a real-time hardware scheduler. Use the same timing mode, action rate, chunk length and observation inputs when comparing policies.

The episode directory contains metadata, resolved recipe/population, RGB frames, observations, actions, inference latency, evaluator outcomes and simulator truth in a separate field. Use `episode_metadata` for checkpoint, data split and training condition. Do not train camera-only policies on privileged fields accidentally.

## Goals and reference qualification

`PlacementEvaluator` checks each bag against its assigned carton: all sampled physical-envelope points inside, released right-finger contact, low speed, target-carton contact and no table support. It requires a two-second observed stable dwell. Samples more than 0.2 s apart reset the dwell. This is an independent sampled placement goal; it does not require the scripted reference trajectory or certify continuous collision-free manipulation between observations. It restores the selected bag and never advances physics.

The fixed single-bag reference has additional task-specific acceptance conditions. It is a privileged installation example, not a six-bag learned-policy baseline. Internal qualification recordings are not included in this public scene export.

## Experiment design

Freeze training and held-out recipe/seed lists and report the actual initialization, camera, timing, backend, and evaluator contracts. Keep rigid and soft results separate: the comparison changes the complete bag/contact representation, so it does not isolate deformation alone. This alpha does not define a frozen leaderboard or claim every generated configuration is reachable or solved.

Current supplied assets are the blue Sea Salt bag and carton. Multiple copies and visual randomization do not establish unseen-SKU coverage. The robot is the original calibrated native Isaac articulation; historical calibration is joint-tracking evidence and does not by itself establish hardware contact fidelity or sim-to-real transfer.
