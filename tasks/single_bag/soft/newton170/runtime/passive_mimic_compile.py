"""Preserve zero-powered passive followers across Newton 1.7 MuJoCo export.

The pinned exporter emits an invalid actuator range [0, 0] even when the joint
has no actuator. Remove only that unused spec-level clamp before compilation.
Newton model gains/effort limits, mimic constraints and physical state stay intact.
"""

from contextlib import contextmanager
import numpy as np


@contextmanager
def passive_mimic_compile(model, receipt, *, rigid_finger_linkage=False):
    import mujoco

    expected = {}
    leaders = set()
    refs = model.joint_mimic_joint.numpy()
    starts = model.joint_qd_start.numpy()
    ke = model.joint_target_ke.numpy()
    kd = model.joint_target_kd.numpy()
    effort = model.joint_effort_limit.numpy()
    for j, ref in enumerate(refs):
        if ref < 0:
            continue
        d = int(starts[j])
        if ke[d] != 0 or kd[d] != 0 or effort[d] != 0:
            raise ValueError('Expected genuinely passive source follower')
        expected[model.joint_label[j].replace('/', '_')] = model.joint_label[j]
        leaders.add(model.joint_label[int(ref)].replace('/', '_'))
    if len(expected) != 2:
        raise ValueError('Expected exactly two source finger mimic followers')
    original = mujoco.MjSpec.compile

    def compile_checked(spec, *args, **kwargs):
        seen = []
        for joint in spec.joints:
            if rigid_finger_linkage and joint.name in (set(expected) | leaders):
                joint.solimp_limit = [0.9999, 0.9999, 0.001, 0.5, 2.0]
            if joint.name not in expected:
                continue
            if not np.array_equal(np.asarray(joint.actfrcrange), np.zeros(2)):
                raise RuntimeError('Pinned exporter no longer has the reviewed zero-range pattern')
            # No actuator may inject force into this source-passive joint.
            actuators = [a.name for a in spec.actuators if getattr(a, 'target', None) == joint.name]
            if actuators:
                raise RuntimeError('Passive follower unexpectedly has an actuator: ' + repr(actuators))
            joint.actfrclimited = False
            seen.append(
                dict(
                    joint=joint.name,
                    source_path=expected[joint.name],
                    source_motor_effort_limit=0.0,
                    source_motor_gains=[0.0, 0.0],
                    actuator_count=0,
                    exported_actuator_range_before=[0.0, 0.0],
                    exported_actuator_clamp_after=False,
                    physical_state_writes=0,
                )
            )
        if len(seen) != 2:
            raise RuntimeError('Expected passive followers missing from MuJoCo spec')
        if rigid_finger_linkage:
            matched = 0
            for equality in spec.equalities:
                if equality.name1 in expected and equality.name2 in leaders:
                    # Map the source's rigid finger linkage to a stiff passive
                    # equality. The MuJoCo default 20 ms soft equality allowed
                    # centimetres of follower error under real contact load.
                    equality.solimp = [0.9999, 0.9999, 0.001, 0.5, 2.0]
                    matched += 1
            if matched != 2:
                raise RuntimeError('Expected exactly two finger linkage equalities')
            for row in seen:
                row['passive_equality_solimp'] = [0.9999, 0.9999, 0.001, 0.5, 2.0]
                row['finger_limit_solimp'] = [0.9999, 0.9999, 0.001, 0.5, 2.0]
        result = original(spec, *args, **kwargs)
        for row in seen:
            joint_id = result.joint(row['joint']).id
            transmissions = np.asarray(result.actuator_trnid)
            if len(transmissions) and np.any(transmissions[:, 0] == joint_id):
                raise RuntimeError('Compiled passive follower has an actuator transmission')
            row['compiled_joint_id'] = int(joint_id)
            row['compiled_actuator_count'] = 0
            row['compiled_actuator_clamp'] = bool(result.jnt_actfrclimited[joint_id])
        receipt.extend(seen)
        return result

    mujoco.MjSpec.compile = compile_checked
    try:
        yield
    finally:
        mujoco.MjSpec.compile = original
