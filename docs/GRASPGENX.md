# Using supplied Grasp GenX proposals

This SDK imports **provided pose proposals**. It does not bundle or run Grasp GenX, supply a verified YAM Grasp GenX gripper configuration, or claim that an imported proposal will grasp the bag. The calibrated robot dynamics remain unchanged.

## Supported input

Export one object's GraspGenX Python-client result as JSON:

```python
# In your GraspGenX environment, after inference:
import json
from pathlib import Path

# grasps: (K,4,4); confidences: (K,). Convert numpy arrays to plain lists.
Path("proposals.json").write_text(json.dumps({
    "grasps": grasps.tolist(), "confidences": confidences.tolist()
}, allow_nan=False))
```

For a scene result, first select its instance ID and associate it with the correct Titus `bag_001`, etc. `infer_object` and `infer_scene_pc` return poses in the supplied point cloud's frame; `infer_scene_depth` uses the camera frame. Missing instances and empty candidate lists mean no proposal. The new server modes do not perform full-scene collision filtering. Scores rank proposals; they are not task-success probabilities. See the [official client/server contract](https://github.com/NVlabs/GraspGenX/blob/b9429097728cb1c430dd78b92edf17ba318aad03/client-server/README.md).

The importer accepts the `grasps` and `confidences` arrays, with an optional `gripper_name` consistency check. It does **not** guess Isaac YAML, GraspDataGen HDF5, quaternion lists, millimetres or an entire multi-instance response. Request the matrix arrays above when someone supplies outputs. The upstream mesh demo's `--output_file` writes a different YAML format; ask for the in-memory matrices and their frame metadata instead of renaming that file to JSON.

## Required context

Provide a separate `context.json` with these fields. The synthetic example in `examples/graspgenx/` illustrates the shape only; its transforms are **not YAM calibration**.

| Field | Contract |
| --- | --- |
| `schema_version`, `units` | `1`, `"m"`. All translations and offsets are metres. |
| `object_id`, `side` | Stable bag ID; `"right"` for the normal Titus packing task. `"left"` is supported for other tasks. |
| `source_frame` | Named frame in which the returned matrices are expressed, including any centering or normalization. |
| `world_from_source` | 4×4 transform from that source frame to the Titus Z-up world at capture time. |
| `hand_from_grasp` | 4×4 transform from the generator's gripper/grasp frame to `/World/Rig/right_hand` or `left_hand`. |
| `gripper_name`, `gripper_config_id` | Actual gripper used to condition inference; identify/hash its configuration and assets. |
| `frame_calibration_id` | Identify the reviewed hand/grasp/camera transform calibration. An ID records provenance, not validation by this importer. |
| `capture_id`, `capture_time_s` | Unique capture identifier across episodes/resets; simulation time of the source observation. |
| `pregrasp_offset_grasp_m` | Approach offset expressed in the generator's grasp frame. For a confirmed +Z approach, `[0,0,-0.06]` backs off 6 cm. |
| `pregrasp_jaw_gap_m` | Explicit nominal full jaw gap before contact, not a model-predicted width. |

All transforms use column vectors: `A_from_B @ point_B = point_A`. The conversion is:

```text
world_from_hand = world_from_source @ source_from_grasp @ inverse(hand_from_grasp)
```

Do not pass a grasp/fingertip pose directly to `Scene.solve_ik`: that method targets the native **hand frame**. GraspGenX's sweep-volume convention uses +Z approach and +X closing, according to its [type definition](https://github.com/NVlabs/GraspGenX/blob/b9429097728cb1c430dd78b92edf17ba318aad03/graspgenx/serving/types.py). The delivered YAM finger geometry/frame must be mapped to that convention; an identity transform or a Franka profile is not a valid substitute. Upstream documentation lists `yam_4310`, but this release has not established that its geometry, pads, open/close states or frame offset match our calibrated deliverable.

For wrist-camera metadata, `camera_to_world_row_matrix` is a USD row-vector matrix. Transpose it first. For a point cloud in the usual OpenCV optical frame (+X right, +Y down, +Z forward), also compose the optical-to-USD camera axis change:

```python
world_from_source = np.asarray(camera_to_world_row_matrix).T @ np.diag([1, -1, -1, 1])
```

Use that only when the cloud really uses those optical axes. For a centered cloud, compose its removed centroid back into `world_from_source`. The upstream [CLI loader](https://github.com/NVlabs/GraspGenX/blob/b9429097728cb1c430dd78b92edf17ba318aad03/client-server/graspgenx_client.py) explicitly centers its inputs; losing that translation moves every proposal to the wrong place. The SDK currently exports RGB and camera metadata, **not depth, segmentation or point clouds**. Those generator inputs must be supplied separately. Mesh-based proposals and simulator object poses are privileged inputs and should be labeled accordingly in evaluations.

## Convert, inspect, then integrate with your controller

This runnable algebra example requires only Python and never starts the simulator:

```bash
python3 -m titus grasps examples/graspgenx/synthetic_proposals.json \
  --context examples/graspgenx/synthetic_context.json \
  --top-k 10 --out runs/synthetic_hand_targets.json
```

For real data, substitute `proposals.json` and your reviewed `context.json`. Malformed transforms, scale/shear/reflection, nonfinite values, mismatched scores, missing frames and out-of-range jaw gaps are rejected. Empty output is handled explicitly. Source indices are retained after ranking.

Within an already running scene, the bridge to IK is:

```python
import json
from pathlib import Path
from titus.grasps import require_fresh_capture

bundle = json.loads(Path("runs/hand_targets.json").read_text())
# active_capture_id is retained by your capture code; do not read it from bundle.
require_fresh_capture(bundle, scene.observe()["t_s"],
                      capture_id=active_capture_id, max_age_s=0.1)
if not bundle["candidates"]:
    raise RuntimeError("No proposal: acquire another observation or change grasp sampling")
candidate = bundle["candidates"][0]
pre = scene.solve_ik(**candidate["pregrasp_hand_pose"], side=bundle["context"]["side"])
grasp = scene.solve_ik(**candidate["grasp_hand_pose"], side=bundle["context"]["side"])
# These are endpoint IK solutions, not a collision-checked trajectory.
```

Next, the controller must check the approach/retreat path against the carton, other arm, dish and clutter; execute robot joint/gripper actions; close under contact; verify lift; move to the carton; release; and evaluate the result. `Scene.move()` interpolates joints and is not a collision planner. Do not automatically execute the highest score. Re-observe moving or deforming bags and replan; timestamp checks cannot detect incorrect extrinsics or instantaneous shape changes.

The YAM command is one finger's travel: `opening_m = (full_gap_m - 0.0060894)/2`, limited to `[-0.003, 0.05]`. The importer supplies `pregrasp_opening_m`; the controller still chooses the close command and checks physical contact. No attachment, object teleportation or carrying force is introduced by this conversion.

CPU tests cover transformation composition, 180-degree rotations, widths, ranking, invalid inputs and capture identity/age. **No GraspGenX-generated grasp has been executed or qualified on this YAM scene in this release.** This is the explicit remaining integration boundary.
