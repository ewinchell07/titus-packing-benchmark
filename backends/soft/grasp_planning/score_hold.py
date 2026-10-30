"""Accept only the physical lift-and-hold portion of a generated-grasp probe."""

import json, sys
from pathlib import Path

p = Path(sys.argv[1])
report = json.loads((p / 'report.json').read_text())
rows = json.loads((p / 'telemetry.json').read_text())
held = [r for r in rows if r['phase'] == 'bag_hold']
gates = dict(
    generated_grasp_used='selected_grasp_candidate' in report,
    diagnostic_completed=report['status'] == 'diagnostic_completed',
    zero_inverted_tets=report['max_inverted_tets'] == 0,
    no_attachments=report['physical_object_attachments'] == 0,
)
metrics = {}
if held:
    metrics = dict(
        hold_s=held[-1]['t_s'] - held[0]['t_s'],
        clearance_m=min(r['lowest_z_m'] for r in held) - 0.8634426593780518,
        bilateral_force_N=min(min(r['finger_contact_force_N']) for r in held),
        table_force_N=max(r['support_contact_force_N'] for r in held),
    )
    gates.update(
        hold_duration=metrics['hold_s'] >= 2.0 - 2 / 600,
        clearance=metrics['clearance_m'] >= 0.05,
        bilateral_contact=metrics['bilateral_force_N'] >= 0.03,
        unsupported=metrics['table_force_N'] <= 0.02,
    )
else:
    gates['hold_evidence'] = False
m = report.get('continuous_box_hold_metrics')
gates['box_held'] = bool(
    m
    and m['maximum_box_slip_m'] <= 0.01
    and m['maximum_box_upright_error_deg'] <= 8
    and m['maximum_box_relative_rotation_deg'] <= 8
    and m['minimum_box_clearance_m'] >= 0.01
)
a = dict(
    passed=all(gates.values()),
    scope='Generated grasp: physical bag lift and two-second hold only. Does not qualify packing.',
    gates=gates,
    metrics=metrics,
    selected_grasp=report.get('selected_grasp_candidate'),
)
(p / 'generated_grasp_acceptance.json').write_text(json.dumps(a, indent=2) + '\n')
print(json.dumps(a, indent=2))
sys.exit(0 if a['passed'] else 1)
