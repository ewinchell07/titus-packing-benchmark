"""CPU native capacity kernels/refresh hooks; seeded contact records, no physics step."""

from pathlib import Path
import argparse, json, traceback
import numpy as np
import warp as wp
import newton
from newton.solvers import SolverVBD
from rigid_soft_capacity_guard import RigidSoftCapacityGuard, RigidSoftCapacityError


def make(rowcap):
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    I = wp.mat33(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01)
    for x in [0.0, 0.2]:
        bid = b.add_body(xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()), mass=1.0, inertia=I)
        b.add_shape_box(body=bid, hx=0.02, hy=0.02, hz=0.02)
    q = np.array([[0, 0, 0], [0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]], float)
    b.add_soft_mesh(
        pos=wp.vec3(1.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0),
        vertices=q.tolist(),
        indices=[0, 1, 2, 3],
        density=1728.0,
        k_mu=200000 / 2.7,
        k_lambda=200000 * 0.35 / (1.35 * 0.3),
        particle_radius=0.0005,
    )
    b.color()
    m = b.finalize(device='cpu')
    pipeline = newton.CollisionPipeline(m, rigid_contact_max=8, soft_contact_max=8)
    solver = SolverVBD(
        m,
        iterations=1,
        rigid_body_contact_buffer_size=rowcap,
        rigid_body_particle_contact_buffer_size=rowcap,
        rigid_compliant_alm=True,
        collision_pipeline=pipeline,
    )
    return m, solver


def seed(m):
    c = newton.Contacts(8, 8, device='cpu')
    c.rigid_contact_count.assign([3])
    c.soft_contact_count.assign([3])
    for name, val in [('rigid_contact_shape0', 0), ('rigid_contact_shape1', 1), ('soft_contact_shape', 0)]:
        a = getattr(c, name).numpy()
        a[:3] = val
        getattr(c, name).assign(a)
    a = c.soft_contact_indices.numpy()
    a[:3] = [[0, -1, -1], [1, -1, -1], [2, -1, -1]]
    c.soft_contact_indices.assign(a)
    a = c.soft_contact_barycentric.numpy()
    a[:3] = [1.0, 0.0, 0.0]
    c.soft_contact_barycentric.assign(a)
    return c


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--out', type=Path, required=True)
    a = pa.parse_args()
    a.out.mkdir(exist_ok=True, parents=True)
    wp.config.kernel_cache_dir = str(a.out / 'warp_cache')
    wp.init()
    wp.set_device('cpu')
    r = {
        'schema': 'titus.newton-pre-force-capacity-cpu.v1',
        'status': 'running',
        'scope': __doc__,
        'cases': [],
        'physical_acceptance': False,
    }
    try:
        for kind in ['global_rigid', 'global_soft', 'global_reduction', 'body_body', 'body_particle']:
            m, s = make(1)
            g = RigidSoftCapacityGuard(m).attach_solver(s)
            c = seed(m)
            if kind == 'global_reduction':
                c._reduction_overflow.assign([1])
            elif kind.startswith('global_'):
                c = newton.Contacts(1, 1, device='cpu')
                getattr(c, kind.replace('global_', '') + '_contact_count').assign([2])
            try:
                if kind.startswith('global_'):
                    g.check_contacts(c, kind)
                elif kind == 'body_body':
                    s._refresh_rigid_contact_state(c, True)
                else:
                    s._refresh_body_particle_contact_state(c, True)
            except RigidSoftCapacityError as e:
                r['cases'].append({'name': kind, 'rejected': True, 'error': str(e), 'guard': g.report()})
            else:
                raise AssertionError(kind + ' truncated data was accepted')
        m, s = make(4)
        g = RigidSoftCapacityGuard(m).attach_solver(s)
        c = seed(m)
        g.check_contacts(c, 'positive_counts')
        s._refresh_rigid_contact_state(c, True)
        s._refresh_body_particle_contact_state(c, True)
        g.assert_healthy()
        r['positive_per_body_rows'] = g.report()
        p = newton.CollisionPipeline(m, rigid_contact_max=8, soft_contact_max=8)
        g.attach_pipeline(p)
        p.collide(m.state(), p.contacts())
        g.assert_healthy()
        assert g.calls.get('pipeline.collide', 0) == 1 and g.calls.get('raw_rigid_before_matching', 0) == 1
        r['positive_pipeline'] = g.report()
        r.update(
            status='cpu_capacity_controls_passed',
            qualification='Actual native list-build/refresh kernels exercised on CPU with explicitly seeded contacts. All4 negative controls reject before force iteration; no dynamics/GPU/full-world/performance claim.',
        )
    except BaseException:
        r.update(status='error', error=traceback.format_exc())
        traceback.print_exc()
    (a.out / 'report.json').write_text(json.dumps(r, indent=2) + '\n')
    print(json.dumps({'status': r['status'], 'error': r.get('error')}), flush=True)
    return 0 if r['status'] == 'cpu_capacity_controls_passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
