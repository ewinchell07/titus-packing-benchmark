"""Independent acceptance of the recorded two-arm rotary-table packing case."""

import json
from pathlib import Path
import numpy as np
import sys
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT / 'runtime'))
from box_grasp_metrics import grasp_reference, grasp_motion


def score(directory):
    directory = Path(directory)
    report = json.loads((directory / 'report.json').read_text())
    rows = json.loads((directory / 'telemetry.json').read_text())
    criteria = json.loads((ROOT / 'config/bimanual_acceptance.json').read_text())
    phases = {r['phase'] for r in rows}
    required = {'box_hold', 'bag_hold', 'over_box', 'insert', 'release_into_box', 'withdraw', 'settle_in_box'}
    held = [r for r in rows if r['phase'] == 'bag_hold']
    boxed = [r for r in rows if r['t_s'] >= 4.0 - 1 / 600]
    settled = [r for r in rows if r['phase'] == 'settle_in_box' and r['t_s'] >= rows[-1]['t_s'] - 2.0]
    gates = dict(
        completed=report['status'] == 'completed',
        required_phases_present=required <= phases,
        no_object_attachments=report['physical_object_attachments'] == 0,
        no_controller_object_pose_writes=report['object_pose_writes_after_reset'] == 0,
        no_controller_object_forces=report['object_forces'] == 0,
        native_rigid_bag=report['bag_representation'] == 'rigid_coacd',
        native_linkage=report['max_native_mimic_error_m'] < 0.001,
        native_clock_matches=bool(rows) and max(abs(r['native_elapsed_s'] - r['t_s']) for r in rows) < 1e-5,
        calibrated_actuators_retained=report['native_robot']['max_efforts'][0][12:] == [10.0, 0.0, 10.0, 0.0],
    )
    metrics = {}
    continuous = report.get('continuous_box_hold_metrics')
    gates['continuous_box_hold_evidence'] = bool(
        continuous and continuous['samples'] >= max(1, int((report['simulated_seconds'] - 4.0) * 600) - 2)
    )
    if continuous:
        gates['continuous_box_rotation'] = (
            continuous['maximum_box_relative_rotation_deg']
            <= criteria['maximum_box_rotation_relative_to_hand_deg']
        )
        gates.update(
            continuous_box_slip=continuous['maximum_box_slip_m']
            <= criteria['maximum_box_slip_relative_to_hand_m'],
            continuous_box_upright=continuous['maximum_box_upright_error_deg']
            <= criteria['maximum_box_upright_error_deg'],
            continuous_box_clearance=continuous['minimum_box_clearance_m']
            >= criteria['required_box_clearance_above_entire_rotary_m'],
        )
    if boxed:
        first = boxed[0]
        reference = grasp_reference(
            first['left_hand_pose'], first['box_pose'], report['tcp_from_delivered_pad_landmarks_m']
        )
        relative = [grasp_motion(r['left_hand_pose'], r['box_pose'], reference) for r in boxed]
        metrics.update(
            minimum_box_clearance_m=min(r['box_lowest_z_m'] for r in boxed) - 0.8634426593780518,
            maximum_box_upright_error_deg=max(r['box_upright_error_deg'] for r in boxed),
            minimum_left_finger_box_force_N=min(min(r['left_box_contact_force_N']) for r in boxed),
            maximum_box_slip_m=max(r['grasp_anchor_slip_m'] for r in relative),
            maximum_box_origin_displacement_m=max(r['origin_displacement_m'] for r in relative),
            maximum_box_relative_rotation_deg=max(r['relative_rotation_deg'] for r in relative),
        )
        gates.update(
            box_held_above_entire_rotary=metrics['minimum_box_clearance_m']
            >= criteria['required_box_clearance_above_entire_rotary_m'],
            box_upright=metrics['maximum_box_upright_error_deg'] <= criteria['maximum_box_upright_error_deg'],
            bilateral_box_contact=metrics['minimum_left_finger_box_force_N']
            >= criteria['minimum_left_finger_box_force_N'],
            box_does_not_slip=metrics['maximum_box_slip_m']
            <= criteria['maximum_box_slip_relative_to_hand_m'],
        )
    else:
        gates['box_hold_evidence'] = False
    if held:
        metrics.update(
            bag_hold_duration_s=held[-1]['t_s'] - held[0]['t_s'],
            minimum_bag_hold_clearance_m=min(r['lowest_z_m'] for r in held) - 0.8634426593780518,
            minimum_bag_hold_finger_force_N=min(min(r['finger_contact_force_N']) for r in held),
            maximum_bag_hold_table_force_N=max(r['support_contact_force_N'] for r in held),
        )
        gates.update(
            bag_hold_duration=metrics['bag_hold_duration_s'] >= criteria['required_bag_hold_s'] - 2 / 600,
            bag_lift_clearance=metrics['minimum_bag_hold_clearance_m']
            >= criteria['minimum_bag_hold_clearance_m'],
            bilateral_bag_grasp=metrics['minimum_bag_hold_finger_force_N'] >= 0.03,
            no_table_support_during_bag_hold=metrics['maximum_bag_hold_table_force_N'] <= 0.02,
        )
    else:
        gates['bag_hold_evidence'] = False
    if settled:
        metrics.update(
            settled_duration_s=settled[-1]['t_s'] - settled[0]['t_s'],
            minimum_final_inside_fraction=min(r['bag_inside_box_fraction'] for r in settled),
            maximum_final_bag_speed_mps=max(r['max_speed_m_s'] for r in settled),
            maximum_final_right_finger_force_N=max(max(r['finger_contact_force_N']) for r in settled),
            minimum_final_carton_contact_force_N=min(r['bag_carton_contact_force_N'] for r in settled),
            maximum_final_table_contact_force_N=max(r['support_contact_force_N'] for r in settled),
        )
        gates['all_coacd_hull_vertices_inside_box'] = all(
            r['coacd_hulls_inside_box_fraction'] == 1.0 for r in settled
        )
        gates.update(
            final_settle_duration=metrics['settled_duration_s'] >= criteria['required_final_settle_s'] - 0.11,
            bag_inside_box=metrics['minimum_final_inside_fraction']
            >= criteria['minimum_final_bag_inside_box_fraction'],
            bag_settled=metrics['maximum_final_bag_speed_mps'] <= criteria['maximum_final_bag_speed_mps'],
            bag_released=metrics['maximum_final_right_finger_force_N']
            <= criteria['maximum_released_right_finger_force_N'],
            bag_supported_by_box=metrics['minimum_final_carton_contact_force_N']
            >= criteria['minimum_final_bag_carton_contact_force_N'],
            bag_not_supported_by_table=metrics['maximum_final_table_contact_force_N'] <= 0.02,
        )
    else:
        gates['final_settle_evidence'] = False
    result = dict(
        passed=all(gates.values()),
        gates=gates,
        metrics=metrics,
        continuous_box_hold_metrics=continuous,
        criteria=criteria,
    )
    (directory / 'acceptance.json').write_text(json.dumps(result, indent=2) + '\n')
    report['task_passed'] = result['passed']
    (directory / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return result


if __name__ == '__main__':
    import sys

    result = score(sys.argv[1])
    print(json.dumps(result, indent=2))
    sys.exit(0 if result['passed'] else 1)
