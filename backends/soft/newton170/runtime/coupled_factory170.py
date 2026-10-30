"""Pinned Newton1.7.0 two-way MuJoCo rigid + VBD volume factory.

For an owned Isaac6.2 NewtonStage _create_solver override. This changes solver
construction only; no positions, velocities, target object paths, holds or
attachments are authored by this module. Native/task acceptance is separate.
"""

from pathlib import Path
import hashlib, importlib.metadata, types
import numpy as np

# Populated from the installed exact SDK copies, not a remote latest branch.
PINNED_SOURCES = {
    '_src/solvers/coupled/solver_coupled.py': 'b1433e7a63f09c7d6014205a09501ba8a4015419decb7a72a533d75dc61faae4',
    '_src/solvers/coupled/solver_coupled_proxy.py': 'e481e07346fd4b569b22b34a722ae44a3ced10f8413f8c2da594a446e6afd2e1',
    '_src/solvers/coupled/model_view.py': 'ccd74638c9558cf4dda692ccc609a51bb46b8ea640203c41b07deb6970368ac4',
    '_src/solvers/coupled/interface.py': 'c8d1f160d4ffcac22841e870f3cd0375c7aad30f799549052692097ab4a1fdb9',
    '_src/solvers/mujoco/solver_mujoco.py': '0fb93442c5424e3e320a9b2a8c35308a9ac14be49fd04dcfb76073a32f9fe051',
    '_src/solvers/solver.py': '58d978d39a292b9193a8b727714a1d6ac5200f20b26250ccc32bf185df499cbc',
    '_src/solvers/vbd/solver_vbd.py': '52d9de6f8cc662a95b4c55eaa657c3eedf5f6a6da699109dd9c2de36e36b932f',
}


def _hash_array(a):
    if a is None:
        return None
    q = np.ascontiguousarray(a.numpy())
    return {'dtype': str(q.dtype), 'shape': list(q.shape), 'sha256': hashlib.sha256(q.tobytes()).hexdigest()}


def _model_snapshot(model):
    names = [
        'body_q',
        'body_qd',
        'body_mass',
        'body_inv_mass',
        'body_inertia',
        'body_inv_inertia',
        'body_com',
        'body_flags',
        'particle_q',
        'particle_qd',
        'particle_mass',
        'particle_inv_mass',
        'particle_radius',
        'particle_flags',
        'tet_indices',
        'tet_poses',
        'tet_activations',
        'tet_materials',
        'tri_indices',
        'tri_poses',
        'tri_materials',
        'joint_type',
        'joint_parent',
        'joint_child',
        'joint_X_p',
        'joint_X_c',
        'joint_axis',
        'joint_q_start',
        'joint_qd_start',
        'joint_target_ke',
        'joint_target_kd',
        'joint_effort_limit',
        'joint_limit_lower',
        'joint_limit_upper',
        'joint_mimic_joint',
        'joint_mimic_coeffs',
        'shape_flags',
        'shape_type',
        'shape_body',
        'shape_transform',
        'shape_scale',
        'shape_source_ptr',
    ]
    return {n: _hash_array(getattr(model, n, None)) for n in names}


def check_source_pins():
    import newton

    versions = {n: importlib.metadata.version(n) for n in ['newton', 'warp-lang', 'mujoco', 'mujoco-warp']}
    if versions != {'newton': '1.7.0', 'warp-lang': '1.17.0', 'mujoco': '3.15.0', 'mujoco-warp': '3.15.0'}:
        raise RuntimeError('Two-way adapter requires pinned Newton1.7.0/Warp1.17.0')
    root = Path(newton.__file__).resolve().parent
    actual = {n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in PINNED_SOURCES}
    if actual != PINNED_SOURCES:
        raise RuntimeError('Installed coupled solver source differs from reviewed exact SDK')
    return {'versions': versions, 'source_hashes': actual}


def register_builder_attributes(builder):
    """Call once in owned_add before original add_usd (VBD is native-registered)."""
    from newton.solvers import SolverMuJoCo

    SolverMuJoCo.register_custom_attributes(builder)


def _counts(model):
    return {
        n: int(getattr(model, n))
        for n in [
            'body_count',
            'shape_count',
            'joint_count',
            'joint_dof_count',
            'particle_count',
            'tet_count',
            'tri_count',
        ]
    }


def _reaction_snapshot(self):
    """Read-only exact SDK history/input buffers; no stepping or state writes.

    In lagged mode the newly harvested buffer is used on the next outer step,
    or the next relaxation pass when proxy_iterations>1. Source input body_f is
    what the rigid sub-solver was actually given on its most recent pass.
    """
    entries = {}
    for name in ['mjc', 'vbd']:
        e = self._entries[name]
        entries[name] = {
            'body_local_to_global': e.body_local_to_global.numpy().copy(),
            'particle_local_to_global': e.particle_local_to_global.numpy().copy(),
            'joint_coord_local_to_global': e.joint_coord_local_to_global.numpy().copy(),
            'joint_dof_local_to_global': e.joint_dof_local_to_global.numpy().copy(),
            'input_body_f': None if e.state_0.body_f is None else e.state_0.body_f.numpy().copy(),
        }
    mappings = []
    for m in self._proxy_mappings:
        mappings.append(
            {
                'source': m.src_name,
                'destination': m.dst_name,
                'source_global_ids': m.src_ids.numpy().copy(),
                'proxy_global_ids': m.proxy_ids_global.numpy().copy(),
                'proxy_local_ids': m.proxy_ids_local.numpy().copy(),
                'feedback_global_wrench': m.coupling_forces.numpy().copy(),
                'source_local_to_proxy_global': m.source_local_to_proxy_global.numpy().copy(),
                'destination_local_to_proxy_global': m.destination_local_to_proxy_global.numpy().copy(),
            }
        )
    return {
        'entries': entries,
        'body_proxy_mappings': mappings,
        'proxy_contact_count': self._titus_vbd_solver.contacts.soft_contact_count.numpy().copy(),
        'owned_contact_handoff': dict(self._titus_vbd_solver._titus_owned_contact_counters),
        'qualification': 'Raw actual staggered-impulse history and source input wrenches. Nonzero harvest alone is not proof of stable/accurate reciprocal contact; compare prior harvested mapped feedback to source input, with external body_f accounted for, and require reciprocal positive/negative contact controls.',
    }


def _owned_prepare(self, state, contacts, *, contacts_freshly_detected=False):
    # Stock proxy scheduler passes outer rigid contacts. Updated VBD owns its
    # refreshed local buffer and requires a None step argument instead.
    self._titus_original_prepare(state, contacts, contacts_freshly_detected=contacts_freshly_detected)
    self._titus_owned_contact_counters['prepare_calls'] += 1
    return None


def _owned_harvest(self, *args, **kwargs):
    # Do not forward None to stock harvest: it would silently return zero force.
    # Feed exactly the owned contact rows/adjacency that VBD actually consumed.
    if self.contacts is None:
        raise RuntimeError('Owned VBD contact buffer is absent')
    kwargs['contacts'] = self.contacts
    self._titus_owned_contact_counters['harvest_calls'] += 1
    return self._titus_original_harvest(*args, **kwargs)


def _entry_contact_observed_step(self, state_in, state_out, control, contacts, dt, *, observables=None):
    # Composite contact observations cannot reconcile independently detected
    # destination rows. Capture each real solver's own native record stream.
    tag = self._titus_entry_tag
    if tag == 'mjc' and self.use_mujoco_cpu:
        self._titus_outer._titus_rigid_observables = None
        self._titus_observed_steps += 1
        return self._titus_original_step(state_in, state_out, control, contacts, dt, observables=None)
    bound = self.contacts if tag == 'vbd' else contacts
    if bound is None:
        raise RuntimeError('Actual entry contact buffer missing: ' + tag)
    if tag == 'vbd' and contacts is not None:
        raise RuntimeError('Owned VBD received external contacts')
    if 'contact_f' not in self.supported_observables:
        raise RuntimeError('Entry native backend does not support contact_f: ' + tag)
    if observables is not None and not observables.is_requested('contact_f'):
        raise RuntimeError(
            'Use separate entry contact observables; composite body-only observables are not enabled by this diagnostic wrapper'
        )
    if observables is None:
        if getattr(self, '_titus_observed_contacts', None) is not bound:
            self._titus_contact_observables = self.observables(['contact_f'], contacts=bound)
            self._titus_observed_contacts = bound
        observables = self._titus_contact_observables
    owner = self._titus_outer
    setattr(owner, '_titus_' + ('vbd' if tag == 'vbd' else 'rigid') + '_observables', observables)
    setattr(owner, '_titus_' + ('vbd' if tag == 'vbd' else 'rigid') + '_observed_contacts', bound)
    self._titus_observed_steps += 1
    return self._titus_original_step(state_in, state_out, control, contacts, dt, observables=observables)


def make_coupled_solver(model, solver_cfg, *, adapter_factory, proxy_iterations=1, mujoco_options=None):
    """Return actual SolverCoupledProxy with accessible components and receipts.

    All rigid bodies/joints are owned by MuJoCo, all particles by VBD. The VBD
    view receives finite-inertia body proxies and returns their contact wrenches
    to MuJoCo via the SDK's lagged-impulse coupling. No external rigid-pose replay.
    """
    import newton
    from newton.solvers import SolverMuJoCo, SolverVBD
    from newton.solvers.experimental.coupled import SolverCoupledProxy

    pins = check_source_pins()
    if (
        isinstance(proxy_iterations, bool)
        or int(proxy_iterations) != proxy_iterations
        or not 1 <= proxy_iterations <= 4
    ):
        raise ValueError('Bounded proxy_iterations1..4 required')
    if model.body_count == 0 or model.joint_count == 0 or model.particle_count == 0 or model.tet_count == 0:
        raise ValueError('Expected one rigid-articulation/volume model')
    if adapter_factory is None:
        raise ValueError('A reviewed surface-only contact adapter factory is required')
    kwargs = dict(vars(solver_cfg)) if hasattr(solver_cfg, '__dict__') else dict(solver_cfg)
    kwargs.pop('solver_type', None)
    kwargs.pop('num_substeps', None)
    if kwargs.pop('collision_pipeline', None) is not None:
        raise ValueError('VBD needs its own model-view pipeline, not a supplied global pipeline')
    kwargs.setdefault('rigid_soft_enable_dat', True)
    if kwargs.get('integrate_with_external_rigid_solver', False):
        raise ValueError('One-way external-rigid integration is forbidden')
    kwargs['integrate_with_external_rigid_solver'] = False
    if not kwargs.get('particle_enable_self_contact', False):
        raise ValueError('VBD intervolume/self contact must be enabled')
    rigid_options = dict(mujoco_options or {})
    if rigid_options.get('use_mujoco_cpu', False):
        raw_meshes = [
            model.shape_label[i]
            for i, k in enumerate(model.shape_type.numpy())
            if int(k) == int(newton.GeoType.MESH)
        ]
        if raw_meshes:
            raise ValueError(
                'CPU rigid contact requires primitive/convex colliders; raw triangle meshes present: '
                + repr(raw_meshes[:8])
            )
    if rigid_options.get('use_mujoco_contacts', False):
        raise ValueError('Use Newton contacts for protected triangle/SDF source colliders')
    if rigid_options.get('disable_contacts', False):
        raise ValueError('Rigid contact may not be disabled')
    rigid_options['use_mujoco_contacts'] = False
    rigid_options.setdefault('nconmax', 65536)
    rigid_options.setdefault('njmax', 262144)
    rigid_options.setdefault('update_data_interval', 1)
    if int(rigid_options['update_data_interval']) != 1:
        raise ValueError('Coupled restarts require Newton-state synchronization each pass')
    before = _model_snapshot(model)
    body_ids = list(range(model.body_count))
    joint_ids = list(range(model.joint_count))
    particle_ids = list(range(model.particle_count))
    local_pipelines = []

    def make_vbd(view):
        import warp as wp

        # All real rigid bodies are source-owned by MuJoCo. Disable only their
        # rigid-rigid participation on this copy-on-write destination view;
        # retain all rigid-soft surface shapes and global source flags.
        local_flags = view._cow_array('shape_flags')
        flags = local_flags.numpy().copy()
        flags &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)
        local_flags.assign(flags)
        p = newton.CollisionPipeline(
            view,
            soft_contact_gap=0.001,
            soft_contact_max=262144,
            rigid_contact_max=65536,
            enable_rigid_soft_full_surface_contact=True,
            broad_phase='sap',
            verify_buffers=True,
        )
        local_pipelines.append(p)
        soft = SolverVBD(model=view, collision_pipeline=p, **kwargs)
        soft._titus_original_prepare = soft.coupling_prepare_proxy_contacts
        soft._titus_original_harvest = soft.coupling_harvest_proxy_wrenches
        soft._titus_owned_contact_counters = {
            'prepare_calls': 0,
            'harvest_calls': 0,
            'harvest_uses_owned_contacts': True,
        }
        soft.coupling_prepare_proxy_contacts = types.MethodType(_owned_prepare, soft)
        soft.coupling_harvest_proxy_wrenches = types.MethodType(_owned_harvest, soft)
        return soft

    outer = SolverCoupledProxy(
        model=model,
        entries=[
            SolverCoupledProxy.Entry(
                name='mjc',
                solver=lambda view: SolverMuJoCo(model=view, **rigid_options),
                bodies=body_ids,
                joints=joint_ids,
            ),
            SolverCoupledProxy.Entry(name='vbd', solver=make_vbd, particles=particle_ids),
        ],
        coupling=SolverCoupledProxy.Config(
            proxies=[
                SolverCoupledProxy.Proxy(
                    source='mjc',
                    destination='vbd',
                    bodies=body_ids,
                    mass_scale=1.0,
                    mode='staggered',
                    proxy_relaxation=1.0,
                    collision_pipeline=None,
                )
            ],
            iterations=int(proxy_iterations),
        ),
    )
    rigid = outer.solver('mjc')
    soft = outer.solver('vbd')
    if soft.integrate_with_external_rigid_solver:
        raise RuntimeError('VBD proxy body dynamics unexpectedly disabled')
    if soft.model is not outer.view('vbd') or rigid.model is not outer.view('mjc'):
        raise RuntimeError('Sub-solver view identity mismatch')
    surface_adapter = adapter_factory(soft)
    if getattr(surface_adapter, 'model', None) is not soft.model:
        raise RuntimeError('Surface adapter attached to wrong model view')
    if not local_pipelines or any(p.model is not soft.model for p in local_pipelines):
        raise RuntimeError('Proxy contact pipeline attached to wrong model view')
    after = _model_snapshot(model)
    if before != after:
        raise RuntimeError(
            'Coupled construction changed protected original model arrays: '
            + repr([n for n in before if before[n] != after[n]])
        )
    if soft.model.particle_count != model.particle_count or soft.model.tet_count != model.tet_count:
        raise RuntimeError('Soft view lost volume nodes/Tets')
    if rigid.model.particle_count != 0:
        raise RuntimeError('MuJoCo view unexpectedly owns soft particles')
    if not np.array_equal(soft.model.tet_indices.numpy(), model.tet_indices.numpy()):
        raise RuntimeError('Soft view Tet connectivity differs from common source')
    if not np.array_equal(soft.model.particle_mass.numpy(), model.particle_mass.numpy()):
        raise RuntimeError('Soft view particle masses differ from common source')
    outer._titus_vbd_solver = soft
    outer._titus_rigid_solver = rigid
    outer._titus_surface_adapter = surface_adapter
    outer._titus_vbd_observables = None
    outer._titus_rigid_observables = None
    for name, entry_solver in [('mjc', rigid), ('vbd', soft)]:
        entry_solver._titus_entry_tag = name
        entry_solver._titus_outer = outer
        entry_solver._titus_original_step = entry_solver.step
        entry_solver._titus_observed_steps = 0
        entry_solver.step = types.MethodType(_entry_contact_observed_step, entry_solver)
    outer._titus_view_models = {'mjc': rigid.model, 'vbd': soft.model}
    outer._titus_local_pipelines = {'vbd_owned': local_pipelines[0]}
    outer._titus_index_maps = {
        name: {
            key: getattr(outer._entries[name], key).numpy().copy()
            for key in [
                'body_local_to_global',
                'particle_local_to_global',
                'joint_coord_local_to_global',
                'joint_dof_local_to_global',
            ]
        }
        for name in ['mjc', 'vbd']
    }
    global_shape_ids = {label: i for i, label in enumerate(model.shape_label)}
    if len(global_shape_ids) != model.shape_count:
        raise RuntimeError('Duplicate global shape labels; explicit native map required')
    for name, entry_solver in [('mjc', rigid), ('vbd', soft)]:
        outer._titus_index_maps[name]['shape_local_to_global'] = np.array(
            [global_shape_ids[label] for label in entry_solver.model.shape_label], dtype=np.int64
        )
    outer._titus_reaction_snapshot = types.MethodType(_reaction_snapshot, outer)
    outer._titus_factory_receipt = {
        'schema': 'titus.native-two-way-coupled-factory.v1',
        'source_pins': pins,
        'actual_outer_solver': type(outer).__name__,
        'actual_components': {'mjc': type(rigid).__name__, 'vbd': type(soft).__name__},
        'global_counts': _counts(model),
        'entry_counts': {'mjc': _counts(rigid.model), 'vbd': _counts(soft.model)},
        'ownership': {
            'rigid_bodies_global': body_ids,
            'rigid_joints_global': joint_ids,
            'soft_particle_global_range': [0, model.particle_count],
        },
        'global_model_arrays_unchanged_during_construction': True,
        'global_model_array_digests': before,
        'proxy_policy': {
            'mode': 'staggered',
            'mass_scale': 1.0,
            'relaxation': 1.0,
            'iterations': int(proxy_iterations),
            'full_surface_contact': True,
            'soft_owned_pipeline': True,
            'rigid_soft_DAT': bool(soft.rigid_soft_enable_dat),
            'owned_contact_harvest': True,
            'VBD_view_rigid_rigid_disabled': True,
        },
        'contact_observables': 'Separate actual rigid entry filtered Contacts and owned VBD Contacts; no composite/global contact-row reconciliation',
        'actual_backend_contact_observable_support': {
            'mjc': 'contact_f' in rigid.supported_observables,
            'vbd': 'contact_f' in soft.supported_observables,
        },
        'global_state_writes_by_factory': 0,
        'global_object_attachments_added': 0,
        'mujoco_options': rigid_options,
        'qualification': 'Experimental Newton SDK SolverCoupledProxy. Internal dynamic proxy state transfer and explicit lagged contact-wrench feedback are solver coupling, not a prescribed gripper/object motion. Source model/ownership/view/source pins verified; native force/contact/stability/task qualification still required.',
    }
    return outer
