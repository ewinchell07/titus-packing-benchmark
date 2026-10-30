"""Pinned Newton1.7.0 surface-only VF queries and fail-closed contact buffers.

Scope: finalized Tet-volume model whose collision triangles equal its exact Tet
boundary. Interior particles remain ordinary finite-mass elasticity DOFs. No
physical positions, velocities, masses, forces or attachments are written.
"""

from pathlib import Path
import hashlib, importlib.metadata, types
from collections import deque
import numpy as np
import warp as wp
import newton

if __package__:
    from .surface_detection import surface_vertex_triangle_collision_detection
    from .source_pins import PINNED_SOURCES, verify_source_pins
else:
    from surface_detection import surface_vertex_triangle_collision_detection
    from source_pins import PINNED_SOURCES, verify_source_pins


class ContactCapacityError(RuntimeError):
    pass


@wp.kernel
def _row_stats(
    counts: wp.array[wp.int32],
    caps: wp.array[wp.int32],
    stats: wp.array[wp.int32],
    overflow_slot: int,
    max_slot: int,
    cap_slot: int,
):
    i = wp.tid()
    c = counts[i]
    cap = caps[i]
    if c < 0 or cap < 0 or c > cap:
        wp.atomic_add(stats, overflow_slot, 1)
    wp.atomic_max(stats, max_slot, c)
    wp.atomic_max(stats, cap_slot, cap)


@wp.kernel
def _flag_stats(flags: wp.array[wp.int32], stats: wp.array[wp.int32]):
    i = wp.tid()
    if flags[i] != 0:
        wp.atomic_add(stats, 9, 1)


def _boundary(tets):
    f = np.concatenate([tets[:, ids] for ids in [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]])
    _, ix, c = np.unique(np.sort(f, axis=1), axis=0, return_index=True, return_counts=True)
    if np.any(c > 2):
        raise ValueError('Non-manifold Tet face incidence')
    return np.sort(f[ix[c == 1]], axis=1)


def _canon(rows):
    return sorted(map(tuple, np.asarray(rows, dtype=np.int64)))


def _guarded_edge(self, *args, **kwargs):
    owner = self._titus_surface_adapter
    owner.before_refresh(self, 'ee')
    owner.original_edges[id(self)](*args, **kwargs)
    owner.after_refresh(self, 'ee')


def _ensured_detector(self):
    owner = self._titus_surface_adapter
    detector = owner.original_pipeline_ensures[id(self)]()
    return owner.attach_detector(detector)


class SurfaceContactAdapter:
    """Install on the exact pipeline detector after constructing SolverVBD.

    Prefer for_solver(solver) before the first simulate call. Do not use CUDA graph capture:
    each refresh synchronizes a small statistics array to reject overflow before
    contact forces consume truncated lists. Performance is unqualified.
    """

    def __init__(self, model):
        self.model = model
        self.original_edges = {}
        self.original_pipeline_ensures = {}
        self.detectors = []
        self.refreshes = {'vf': 0, 'ee': 0}
        self.first_history = []
        self.periodic_history = deque(maxlen=64)
        self.maxima_witnesses = {}
        self.check_totals = {'VF_rows': 0, 'EE_rows': 0, 'triangle_rows': 0, 'overflow_events': 0}
        self.failure = None
        self.adapter_hashes = {
            n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
            for n in ['contact_adapter.py', 'surface_detection.py', 'surface_kernel.py', 'source_pins.py']
        }
        self.source_hashes = verify_source_pins()
        t = model.tet_indices.numpy().astype(np.int64)
        tri = model.tri_indices.numpy().astype(np.int64)
        if len(t) == 0 or tri.shape[1:] != (3,):
            raise ValueError('Adapter requires Tet volumes with explicit surface triangles')
        if _canon(np.sort(tri, axis=1)) != _canon(_boundary(t)):
            raise ValueError('Collision triangles differ from true Tet boundary; refuse masking')
        self.mask_numpy = np.zeros(model.particle_count, dtype=np.int32)
        self.mask_numpy[np.unique(tri)] = 1
        if not np.array_equal(np.unique(t), np.arange(model.particle_count)):
            raise ValueError('Model includes unsupported nonvolume particles')
        self.surface_mask = wp.array(self.mask_numpy, dtype=wp.int32, device=model.device)
        self.stats = wp.zeros(10, dtype=wp.int32, device=model.device)
        self.topology = {
            'particle_count': model.particle_count,
            'tet_count': model.tet_count,
            'collision_triangles': model.tri_count,
            'surface_query_vertices': int(self.mask_numpy.sum()),
            'interior_vertices_excluded_from_surface_query': int((self.mask_numpy == 0).sum()),
            'tri_sha256': hashlib.sha256(tri.tobytes()).hexdigest(),
            'tet_sha256': hashlib.sha256(t.tobytes()).hexdigest(),
        }

    def attach_detector(self, detector):
        if detector.model is not self.model:
            raise ValueError('Detector model identity mismatch')
        if hasattr(detector, '_titus_surface_adapter'):
            if detector._titus_surface_adapter is self:
                return detector
            raise RuntimeError('Detector already has another adapter')
        detector._titus_surface_adapter = self
        self.original_edges[id(detector)] = detector.edge_edge_collision_detection
        detector.vertex_triangle_collision_detection = types.MethodType(
            surface_vertex_triangle_collision_detection, detector
        )
        detector.edge_edge_collision_detection = types.MethodType(_guarded_edge, detector)
        self.detectors.append(detector)
        return detector

    def attach_pipeline(self, pipeline):
        if pipeline.model is not self.model:
            raise ValueError('Pipeline model identity mismatch')
        if hasattr(pipeline, '_titus_surface_adapter') and pipeline._titus_surface_adapter is not self:
            raise RuntimeError('Pipeline already adapted')
        if id(pipeline) not in self.original_pipeline_ensures:
            self.original_pipeline_ensures[id(pipeline)] = pipeline._ensure_soft_self_contact_detector
            pipeline._titus_surface_adapter = self
            pipeline._ensure_soft_self_contact_detector = types.MethodType(_ensured_detector, pipeline)
        detector = pipeline._ensure_soft_self_contact_detector()
        self.attach_detector(detector)
        self.pipeline = pipeline
        return self

    @classmethod
    def for_solver(cls, solver):
        owner = cls(solver.model)
        pipeline = getattr(solver, 'collision_pipeline', None)
        if pipeline is None:
            raise RuntimeError('Requires solver-owned CollisionPipeline; detached detector is unsupported')
        owner.attach_pipeline(pipeline)
        return owner.verify_solver(solver)

    def verify_solver(self, solver):
        if solver.model is not self.model:
            raise ValueError('Solver model identity mismatch')
        if not solver.particle_enable_self_contact:
            raise RuntimeError('Solver soft/self contact is disabled')
        pipeline = getattr(solver, 'collision_pipeline', None)
        if pipeline is None:
            raise RuntimeError('Requires solver-owned CollisionPipeline')
        if pipeline is not getattr(self, 'pipeline', None):
            raise RuntimeError('Solver uses an uninstrumented pipeline')
        self.attach_detector(pipeline._ensure_soft_self_contact_detector())
        return self

    def before_refresh(self, detector, phase):
        if self.failure:
            raise ContactCapacityError('Adapter already failed: ' + str(self.failure))
        if detector not in self.detectors:
            raise RuntimeError('Unregistered detector')
        if self.model.device.is_capturing:
            raise RuntimeError('Synchronous fail-closed contact guard requires uncaptured execution')

    def after_refresh(self, detector, phase):
        self.stats.zero_()
        families = (
            [('vertex_colliding_triangles_count', 'vertex_colliding_triangles_buffer_sizes', 0, 3, 6)]
            if phase == 'vf'
            else [('edge_colliding_edges_count', 'edge_colliding_edges_buffer_sizes', 2, 4, 7)]
        )
        if phase == 'vf' and detector.record_triangle_contacting_vertices:
            families.append(
                ('triangle_colliding_vertices_count', 'triangle_colliding_vertices_buffer_sizes', 1, 5, 8)
            )
        for count_name, cap_name, of, ms, cs in families:
            counts = getattr(detector, count_name)
            caps = getattr(detector, cap_name)
            if counts.shape != caps.shape:
                raise RuntimeError('Contact row capacity/count shape mismatch')
            self.check_totals[['VF_rows', 'triangle_rows', 'EE_rows'][of]] += counts.size
            if counts.size:
                wp.launch(
                    _row_stats,
                    dim=counts.size,
                    inputs=[counts, caps, self.stats, of, ms, cs],
                    device=self.model.device,
                )
        wp.launch(
            _flag_stats,
            dim=detector.resize_flags.size,
            inputs=[detector.resize_flags, self.stats],
            device=self.model.device,
        )
        values = self.stats.numpy().tolist()
        self.refreshes[phase] += 1
        row = {
            'phase': phase,
            'ordinal': self.refreshes[phase],
            'overflow_rows_VF': values[0],
            'overflow_rows_triangle': values[1],
            'overflow_rows_EE': values[2],
            'max_VF_count': values[3],
            'max_EE_count': values[4],
            'max_triangle_count': values[5],
            'max_VF_capacity': values[6],
            'max_EE_capacity': values[7],
            'max_triangle_capacity': values[8],
            'native_nonzero_resize_flags': values[9],
            'total_refresh_ordinal': sum(self.refreshes.values()),
        }
        if len(self.first_history) < 8:
            self.first_history.append(row)
        if row['total_refresh_ordinal'] % 256 == 0:
            self.periodic_history.append(row)
        for key in [
            'max_VF_count',
            'max_EE_count',
            'max_triangle_count',
            'overflow_rows_VF',
            'overflow_rows_triangle',
            'overflow_rows_EE',
            'native_nonzero_resize_flags',
        ]:
            if key not in self.maxima_witnesses or row[key] > self.maxima_witnesses[key][key]:
                self.maxima_witnesses[key] = row
        if any(values[i] for i in [0, 1, 2, 9]):
            self.check_totals['overflow_events'] += 1
            row['native_resize_flags'] = detector.resize_flags.numpy().tolist()
            row['overflow_witnesses'] = {}
            for count_name, cap_name, *_ in families:
                counts = getattr(detector, count_name).numpy()
                caps = getattr(detector, cap_name).numpy()
                bad = np.flatnonzero((counts < 0) | (caps < 0) | (counts > caps))
                row['overflow_witnesses'][count_name] = [
                    {'row': int(i), 'count': int(counts[i]), 'capacity': int(caps[i])} for i in bad[:16]
                ]
            self.failure = row
            raise ContactCapacityError('Truncated collision lists rejected before solve: ' + repr(row))

    def report(self):
        return {
            'schema': 'titus.newton170-surface-contact-adapter.v1',
            'owned_pipeline': getattr(self, 'pipeline', None) is not None,
            'source_hashes': self.source_hashes,
            'adapter_source_hashes': self.adapter_hashes,
            'topology': self.topology,
            'refresh_counts': self.refreshes,
            'capacity_failure': self.failure,
            'refresh_history_first': self.first_history,
            'refresh_history_periodic': list(self.periodic_history),
            'maxima_witnesses': self.maxima_witnesses,
            'checked_row_totals': self.check_totals,
            'telemetry_bound': 'First8 refresh rows + last64 periodic rows(every256) + one witness per7 metrics + fatal overflow row. Every refresh is still checked.',
            'physical_state_writes': 0,
            'contact_policy': 'Only actual surface vertices query actual surface triangles; stock topology filters and all surface VF/EE remain enabled. Every refresh checks native capacity counts and flags before solve.',
            'qualification': 'No material/control/geometry retune. No physics acceptance from detection alone. GPU and performance not qualified. CUDA graph capture deliberately rejected.',
        }
