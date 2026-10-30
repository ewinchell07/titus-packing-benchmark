"""Separate grasp-point sliding from rigid rotation about that grasp."""

import numpy as np
from scipy.spatial.transform import Rotation


def grasp_reference(hand, box, tcp_local):
    hand, box = np.asarray(hand), np.asarray(box)
    h, b = Rotation.from_quat(hand[3:]), Rotation.from_quat(box[3:])
    anchor_world = hand[:3] + h.apply(tcp_local)
    return dict(
        anchor_box_local=b.inv().apply(anchor_world - box[:3]).tolist(),
        anchor_hand_local=np.asarray(tcp_local).tolist(),
        box_origin_hand_local=h.inv().apply(box[:3] - hand[:3]).tolist(),
        box_rotation_hand=(h.inv() * b).as_quat().tolist(),
    )


def grasp_motion(hand, box, reference):
    hand, box = np.asarray(hand), np.asarray(box)
    h, b = Rotation.from_quat(hand[3:]), Rotation.from_quat(box[3:])
    anchor_world = box[:3] + b.apply(reference['anchor_box_local'])
    anchor_hand = h.inv().apply(anchor_world - hand[:3])
    origin_hand = h.inv().apply(box[:3] - hand[:3])
    return dict(
        grasp_anchor_slip_m=float(np.linalg.norm(anchor_hand - reference['anchor_hand_local'])),
        origin_displacement_m=float(np.linalg.norm(origin_hand - reference['box_origin_hand_local'])),
        relative_rotation_deg=float(
            (Rotation.from_quat(reference['box_rotation_hand']).inv() * h.inv() * b).magnitude() * 180 / np.pi
        ),
        upright_error_deg=float(np.degrees(np.arccos(np.clip(b.as_matrix()[2, 2], -1, 1)))),
    )
