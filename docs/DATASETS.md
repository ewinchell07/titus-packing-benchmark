# Using the real-world demonstrations

The collection links the existing datasets without copying, reformatting, relabeling, or changing their access settings. Request access on each dataset page if needed, then authenticate locally with `hf auth login`. Never put your token into code or a shared notebook.

## Robot teleoperation

Repository: [titus-robots/rotary-table-rice-cakes](https://huggingface.co/datasets/titus-robots/rotary-table-rice-cakes)

Pinned alpha revision: `85f8a92cacf20561a27955ad4f8217a233cf5ec1`.

```bash
hf download titus-robots/rotary-table-rice-cakes --repo-type dataset \
  --revision 85f8a92cacf20561a27955ad4f8217a233cf5ec1 --local-dir data/teleop
```

- LeRobot v3; 60 cleaned segments and 162,143 frames, approximately 45 minutes at 60 Hz.
- Three RGB streams: chest, left wrist, right wrist; 480×270 fisheye images encoded as H.264.
- State and action have 14 values: `[finger_joint1, joint1, ..., joint6]` per arm, left then right. Actions are absolute joint positions.
- Teleop dropouts were split, not spliced; the dataset preserves the segment-to-source mapping and QC reports. Keep segments from the same original recording in the same train/test split to avoid leakage.
- Inspect `meta/info.json`, `meta/stats.json`, conversion configuration, and the dataset card before writing an adapter. Compute training normalization on your training split only.

Do not assume that a 14-element policy output can be sent directly to the scene API. Map named arm joints, finger travel, measured versus commanded state, and camera order explicitly. The dataset's action frequency is 60 Hz; the SDK hold example uses 30 Hz unless changed.

## Handheld UMI demonstrations

Repository: [titus-robots/trumi-box-bags-2026-10-16](https://huggingface.co/datasets/titus-robots/trumi-box-bags-2026-10-16)

Pinned alpha revision: `7100a2251f34b818590a055d4eba2c7d9422a80a`.

```bash
hf download titus-robots/trumi-box-bags-2026-10-16 --repo-type dataset \
  --revision 7100a2251f34b818590a055d4eba2c7d9422a80a --local-dir data/umi
```

- 21 retained bimanual episodes, 471 seconds total: 16 one-box/six-bag episodes and 5 two-box/twelve-bag episodes.
- Two handheld GoPro views, 1352×1014 JPEG at 119.88 fps; poses and widths at 59.94 Hz.
- MCAP topics include `/cameras/camera{0,1}/image`, `robot{0,1}/eef/state`, and `robot{0,1}/gripper/state`.
- End-effector poses are metres and axis-angle rotation vectors in the mapping-tag frame, Z-up. Gripper width is in metres. These widths are not the SDK's per-finger travel commands.
- `episode_index.csv` links episodes to source demos; `calibration/` contains fisheye intrinsics, tag configuration, SLAM-to-tag registration, and gripper ranges.
- 21 of 42 recorded bimanual demos were retained where both cameras tracked. Use the supplied tracking metrics when selecting training data; do not assume uniform trajectory quality.

Read the MCAP and tabular metadata first. `dataset_plan.pkl` uses Python pickle and should only be deserialized when you trust the source and pinned revision.

## Combining the sources

The corpora share the packing activity but are not synchronized paired captures. UMI does not provide the robot's chest camera or joint trajectories. The supplied UMI calibration reaches its mapping-tag frame; a validated mapping-tag-to-YAM transform and matching TCP orientation convention are not included in this alpha.

Before treating UMI as robot training data, establish and validate that registration, retarget both arms with joint limits and collision-aware motion, align timestamps and action rates, and declare how missing/mismatched views are handled. Do not fabricate a chest view or silently duplicate a wrist view. Keep original recordings and derived conversions separately versioned.

The simulation uses pinhole wrist views and a 1024×768 external view, unlike either real camera setup. Adapt images deliberately and record cropping, resizing, distortion treatment, camera order, normalization statistics, and action conventions with each experiment.
