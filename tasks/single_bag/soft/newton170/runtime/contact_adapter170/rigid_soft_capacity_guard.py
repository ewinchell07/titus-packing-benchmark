"""Pinned pre-force capacity guards for actual Newton VBD rigid/soft lists.

The native per-body overflow kernel prints a warning only. These scoped wrappers
raise before the VBD force loop consumes truncated adjacency. No physics state
or buffer capacity is changed. CUDA graph capture is deliberately unsupported.
"""

from pathlib import Path
from collections import deque
import hashlib, importlib.metadata, types
import numpy as np
import newton

if __package__:
    from .source_pins import PINNED_SOURCES, verify_source_pins
else:
    from source_pins import PINNED_SOURCES, verify_source_pins


class RigidSoftCapacityError(RuntimeError):
    pass


def _guarded_collide(self, *args, **kwargs):
    g = self._titus_rigid_soft_capacity_guard
    g.before()
    result = g.original_collide[id(self)](*args, **kwargs)
    contacts = args[1] if len(args) > 1 else kwargs['contacts']
    g.check_contacts(contacts, 'pipeline.collide', self)
    return result


def _guarded_narrow(self, *args, **kwargs):
    g = self._titus_rigid_soft_capacity_guard
    g.before()
    count = int(kwargs['candidate_pair_count'].numpy()[0])
    cap = int(kwargs['candidate_pair'].shape[0])
    if count < 0 or count > cap:
        g._record({'phase': 'pre_narrow_candidate_pairs', 'count': count, 'capacity': cap}, True)
    result = g.original_narrow[id(self)](*args, **kwargs)
    writer = kwargs['writer_data']
    count = int(writer.contact_count.numpy()[0])
    cap = int(writer.contact_max)
    g._record(
        {'phase': 'raw_rigid_before_matching', 'count': count, 'capacity': cap},
        count < 0 or count > cap or (cap > 0 and count == cap),
    )
    return result


def _guarded_refresh_body_body(self, *args, **kwargs):
    g = self._titus_rigid_soft_capacity_guard
    g.before()
    contacts = args[0] if args else kwargs.get('contacts')
    g.check_contacts(contacts, 'pre_body_body_refresh')
    result = g.original_refresh_body_body(*args, **kwargs)
    g.check_body_rows(self, 'body_body')
    return result


def _guarded_refresh_body_particle(self, *args, **kwargs):
    g = self._titus_rigid_soft_capacity_guard
    g.before()
    contacts = args[0] if args else kwargs.get('contacts')
    g.check_contacts(contacts, 'pre_body_particle_refresh')
    result = g.original_refresh_body_particle(*args, **kwargs)
    g.check_body_rows(self, 'body_particle')
    return result


class RigidSoftCapacityGuard:
    def __init__(self, model):
        self.source_hashes = verify_source_pins()
        self.model = model
        self.original_collide = {}
        self.original_narrow = {}
        self.calls = {}
        self.first = []
        self.periodic = deque(maxlen=64)
        self.maxima = {}
        self.failure = None
        self.total_records = 0

    def before(self):
        if self.failure:
            raise RigidSoftCapacityError('Sticky capacity failure: ' + repr(self.failure))
        if self.model.device.is_capturing:
            raise RuntimeError('Pre-force synchronized capacity guard requires uncaptured execution')

    def _record(self, row, invalid=False):
        self.total_records += 1
        row['ordinal'] = self.total_records
        phase = row['phase']
        self.calls[phase] = self.calls.get(phase, 0) + 1
        if len(self.first) < 8:
            self.first.append(row)
        if self.total_records % 256 == 0:
            self.periodic.append(row)
        for k, v in row.items():
            if isinstance(v, (int, float)) and k not in ['ordinal']:
                key = phase + '.' + k
                if key not in self.maxima or v > self.maxima[key][k]:
                    self.maxima[key] = row
        if invalid:
            self.failure = row
            raise RigidSoftCapacityError('Contact capacity rejected before force consumption: ' + repr(row))

    def attach_pipeline(self, pipeline):
        if pipeline.model is not self.model:
            raise ValueError('Pipeline model identity mismatch')
        if hasattr(pipeline, '_titus_rigid_soft_capacity_guard'):
            if pipeline._titus_rigid_soft_capacity_guard is self:
                return self
            raise RuntimeError('Pipeline already has another capacity guard')
        narrow = pipeline.narrow_phase
        if not hasattr(narrow, '_titus_rigid_soft_capacity_guard'):
            self.original_narrow[id(narrow)] = narrow.launch_custom_write
            narrow._titus_rigid_soft_capacity_guard = self
            narrow.launch_custom_write = types.MethodType(_guarded_narrow, narrow)
        elif narrow._titus_rigid_soft_capacity_guard is not self:
            raise RuntimeError('Narrow phase already guarded')
        self.original_collide[id(pipeline)] = pipeline.collide
        pipeline._titus_rigid_soft_capacity_guard = self
        pipeline.collide = types.MethodType(_guarded_collide, pipeline)
        return self

    def attach_solver(self, solver):
        if type(solver).__name__ != 'SolverVBD' or solver.model is not self.model:
            raise ValueError('Requires actual common-model SolverVBD')
        if hasattr(solver, '_titus_rigid_soft_capacity_guard'):
            raise RuntimeError('Solver capacity guard already attached')
        self.original_refresh_body_body = solver._refresh_rigid_contact_state
        self.original_refresh_body_particle = solver._refresh_body_particle_contact_state
        solver._titus_rigid_soft_capacity_guard = self
        solver._refresh_rigid_contact_state = types.MethodType(_guarded_refresh_body_body, solver)
        solver._refresh_body_particle_contact_state = types.MethodType(_guarded_refresh_body_particle, solver)
        if getattr(solver, 'collision_pipeline', None) is None:
            raise RuntimeError('Requires solver-owned CollisionPipeline')
        self.attach_pipeline(solver.collision_pipeline)
        self.solver = solver
        return self

    @classmethod
    def for_native_stage(cls, ns):
        g = cls(ns.model).attach_solver(ns.solver)
        if getattr(ns, 'collision_pipeline', None) is not None:
            g.attach_pipeline(ns.collision_pipeline)
        g.check_contacts(
            getattr(ns, 'contacts', None), 'initial_import', getattr(ns, 'collision_pipeline', None)
        )
        return g

    def check_contacts(self, contacts, phase='manual', pipeline=None):
        self.before()
        if contacts is None:
            return
        reduction_overflow = int(contacts._reduction_overflow.numpy()[0])
        row = {'phase': phase, 'native_reduction_overflow': reduction_overflow}
        invalid = reduction_overflow != 0
        for kind, fields in [
            (
                'rigid',
                [
                    'rigid_contact_shape0',
                    'rigid_contact_shape1',
                    'rigid_contact_point0',
                    'rigid_contact_point1',
                    'rigid_contact_normal',
                ],
            ),
            (
                'soft',
                [
                    'soft_contact_indices',
                    'soft_contact_barycentric',
                    'soft_contact_shape',
                    'soft_contact_normal',
                ],
            ),
        ]:
            count = int(getattr(contacts, kind + '_contact_count').numpy()[0])
            cap = int(getattr(contacts, kind + '_contact_max'))
            actual = min(int(getattr(contacts, k).shape[0]) for k in fields)
            row[kind + '_count'] = count
            row[kind + '_declared_capacity'] = cap
            row[kind + '_smallest_record_capacity'] = actual
            # Reject saturation too: no native global overflow bit proves whether a
            # full counter was clamped/reduced during contact matching.
            invalid |= (
                count < 0
                or actual < cap
                or count > cap
                or (cap > 0 and count == cap)
                or (cap == 0 and count != 0)
            )
        if (
            pipeline is not None
            and hasattr(pipeline, 'broad_phase_pair_count')
            and hasattr(pipeline, 'broad_phase_shape_pairs')
        ):
            count = int(pipeline.broad_phase_pair_count.numpy()[0])
            cap = int(pipeline.broad_phase_shape_pairs.shape[0])
            row.update(broad_phase_pair_count=count, broad_phase_pair_capacity=cap)
            invalid |= count < 0 or count > cap
        self._record(row, bool(invalid))

    def check_body_rows(self, solver, kind):
        self.before()
        if not getattr(solver, '_integrates_rigid_bodies', False):
            return
        counts = getattr(solver, kind + '_contact_counts').numpy()
        cap = int(getattr(solver, kind + '_contact_buffer_pre_alloc'))
        maximum = int(getattr(solver, kind + '_contact_overflow_max').numpy()[0])
        indices = getattr(solver, kind + '_contact_indices')
        expected = len(counts) * cap
        bad = np.flatnonzero((counts < 0) | (counts > cap))
        row = {
            'phase': kind + '_rows',
            'body_rows': len(counts),
            'row_capacity': cap,
            'max_native_count': int(counts.max(initial=0)),
            'native_overflow_max': maximum,
            'total_native_body_contact_entries': int(counts.sum(dtype=np.int64)),
            'actual_index_capacity': int(indices.size),
            'expected_index_capacity': expected,
        }
        invalid = len(bad) > 0 or maximum != 0 or indices.size < expected
        if invalid:
            row['overflow_witnesses'] = [
                {'body': int(i), 'count': int(counts[i]), 'capacity': cap} for i in bad[:16]
            ]
        self._record(row, bool(invalid))

    def assert_healthy(self):
        if self.failure:
            raise RigidSoftCapacityError('Sticky capacity failure: ' + repr(self.failure))

    def report(self):
        return {
            'schema': 'titus.newton170-rigid-soft-capacity-guard.v1',
            'source_hashes': self.source_hashes,
            'calls_by_phase': self.calls,
            'total_checks': self.total_records,
            'first_checks': self.first,
            'periodic_checks': list(self.periodic),
            'maxima_witnesses': self.maxima,
            'failure': self.failure,
            'physical_state_writes': 0,
            'qualification': 'Raw rigid count checked before matching/sorting, global final counters+confirmed broad-phase pair counter checked after every collide; per-body lists checked after each native refresh before force iteration. Global saturation fails conservatively; exact-capacity does not itself prove truncation. Narrow-phase internal triangle candidate buffers require separate provenance/guards. CPU/GPU/performance acceptance recorded separately.',
        }
