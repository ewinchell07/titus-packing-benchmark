"""Convert supplied GraspGenX matrix proposals into Titus hand-pose targets.

Standard library only. This module neither runs a model nor moves a robot.
All matrices use column vectors: A_from_B maps B coordinates into A.
See docs/GRASPGENX.md for the required gripper and capture-frame calibration.
"""

import math


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _vector(value, size, name):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{name} must contain {size} numbers")
    return [_number(x, name) for x in value]


def _se3(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{name} must be a 4x4 rigid transform")
    m = [_vector(row, 4, name) for row in value]
    if any(abs(x-y) > 1e-5 for x, y in zip(m[3], [0, 0, 0, 1])):
        raise ValueError(f"{name} needs column-vector convention; last row must be [0,0,0,1]")
    for i in range(3):
        for j in range(3):
            if abs(sum(m[k][i]*m[k][j] for k in range(3)) - (i == j)) > 1e-4:
                raise ValueError(f"{name} rotation must be orthonormal (no scale or shear)")
    det = (m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
           - m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
           + m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]))
    if abs(det-1) > 1e-4:
        raise ValueError(f"{name} must be right handed (determinant +1)")
    return m


def _multiply(a, b):
    return [[sum(a[i][k]*b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def _inverse(m):
    r = [[m[j][i] for j in range(3)] for i in range(3)]
    return [r[i] + [-sum(r[i][k]*m[k][3] for k in range(3))] for i in range(3)] + [[0., 0., 0., 1.]]


def _pose(m):
    # Largest quaternion component avoids division by a near-zero trace at 180 degrees.
    terms = [1+m[0][0]-m[1][1]-m[2][2], 1-m[0][0]+m[1][1]-m[2][2],
             1-m[0][0]-m[1][1]+m[2][2], 1+m[0][0]+m[1][1]+m[2][2]]
    index = max(range(4), key=terms.__getitem__)
    q = [0.] * 4
    q[index] = .5 * math.sqrt(max(0., terms[index]))
    denominator = 4*q[index]
    if index == 3:
        q[:3] = [(m[2][1]-m[1][2])/denominator, (m[0][2]-m[2][0])/denominator,
                 (m[1][0]-m[0][1])/denominator]
    else:
        j, k = (index+1) % 3, (index+2) % 3
        q[j] = (m[j][index]+m[index][j])/denominator
        q[k] = (m[k][index]+m[index][k])/denominator
        q[3] = (m[k][j]-m[j][k])/denominator
    norm = math.sqrt(sum(x*x for x in q))
    return dict(position_m=[m[i][3] for i in range(3)], quaternion_xyzw=[x/norm for x in q])


def opening_from_jaw_gap(jaw_gap_m):
    """Nominal pad separation -> one-finger command; never clip an invalid width."""
    gap = _number(jaw_gap_m, "pregrasp_jaw_gap_m")
    opening = (gap - .0060894)/2
    if not .0000894 <= gap <= .1060894:
        raise ValueError("Nominal jaw gap must be within [0.0000894, 0.1060894] m")
    return opening


def convert_graspgenx(result, context, *, min_confidence=None, top_k=20):
    """Convert one object's raw grasps/confidences lists with explicit metadata.

    Python client arrays must first use .tolist(). A scene response must first
    select one instance. Isaac YAML is a different format and is not accepted.
    Every input pose is validated, including poses excluded by score filtering.
    """
    if not isinstance(result, dict) or not isinstance(context, dict):
        raise ValueError("result and context must be dictionaries")
    required = {"schema_version", "units", "object_id", "side", "source_frame",
                "gripper_name", "gripper_config_id", "frame_calibration_id", "capture_id",
                "capture_time_s", "world_from_source", "hand_from_grasp",
                "pregrasp_offset_grasp_m", "pregrasp_jaw_gap_m"}
    if required - context.keys():
        raise ValueError(f"Missing grasp context: {sorted(required-context.keys())}")
    if context["schema_version"] != 1 or isinstance(context["schema_version"], bool) or context["units"] != "m":
        raise ValueError("Expected context schema_version=1 and units='m'; convert units explicitly")
    for key in ("object_id", "source_frame", "gripper_name", "gripper_config_id", "frame_calibration_id", "capture_id"):
        if not isinstance(context[key], str) or not context[key].strip():
            raise ValueError(f"{key} must be a nonempty identifier")
    if context["side"] not in ("left", "right"):
        raise ValueError("side must be left or right")
    if "gripper_name" in result and result["gripper_name"] != context["gripper_name"]:
        raise ValueError("Result gripper_name differs from the declared gripper")
    capture_time = _number(context["capture_time_s"], "capture_time_s")
    if capture_time < 0:
        raise ValueError("capture_time_s must be nonnegative simulation time")
    world_source = _se3(context["world_from_source"], "world_from_source")
    grasp_hand = _inverse(_se3(context["hand_from_grasp"], "hand_from_grasp"))
    offset = _vector(context["pregrasp_offset_grasp_m"], 3, "pregrasp_offset_grasp_m")
    opening = opening_from_jaw_gap(context["pregrasp_jaw_gap_m"])
    grasps, scores = result.get("grasps"), result.get("confidences")
    if not isinstance(grasps, list) or not isinstance(scores, list) or len(grasps) != len(scores):
        raise ValueError("Expected matching grasps (K,4,4) and confidences (K,) lists")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if min_confidence is not None:
        min_confidence = _number(min_confidence, "min_confidence")
    candidates = []
    for index, (raw, raw_score) in enumerate(zip(grasps, scores)):
        source_grasp = _se3(raw, f"grasps[{index}]")
        score = _number(raw_score, f"confidences[{index}]")
        if min_confidence is not None and score < min_confidence:
            continue
        world_grasp = _multiply(world_source, source_grasp)
        world_pregrasp = [row[:] for row in world_grasp]
        for i in range(3):
            world_pregrasp[i][3] += sum(world_grasp[i][j]*offset[j] for j in range(3))
        candidates.append(dict(source_index=index, confidence=score,
                               grasp_hand_pose=_pose(_multiply(world_grasp, grasp_hand)),
                               pregrasp_hand_pose=_pose(_multiply(world_pregrasp, grasp_hand))))
    candidates.sort(key=lambda value: value["confidence"], reverse=True)
    return dict(schema_version=1, scope="Pose proposals only; no motion or grasp qualification",
                context=context, pregrasp_opening_m=opening,
                input_count=len(grasps), selected_count=min(top_k, len(candidates)),
                candidates=candidates[:top_k])


def require_fresh_capture(bundle, current_time_s, *, capture_id, max_age_s):
    """Reject a different capture, future timestamp or stale simulation-time pose.

    The caller retains the active capture ID across inference. This cannot detect
    bad extrinsics or motion/deformation below the chosen age threshold.
    """
    if capture_id != bundle["context"]["capture_id"]:
        raise ValueError("Grasp proposals belong to a different capture")
    now = _number(current_time_s, "current_time_s")
    age = now - _number(bundle["context"]["capture_time_s"], "capture_time_s")
    limit = _number(max_age_s, "max_age_s")
    if limit < 0 or age < -1e-9 or age > limit + 1e-9:
        raise ValueError(f"Grasp capture age {age:g} s is outside [0, {limit:g}]")
