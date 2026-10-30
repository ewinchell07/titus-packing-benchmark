Owned Newton 1.7.0 surface-contact adapter

This is an instance-only port for the exact candidate installed under `solver/site/newton`. It leaves the installed package, geometry, masses, material parameters, forces, positions and velocities unchanged. It is pinned to Newton 1.7.0, Warp 1.17.0 and all seven source hashes in `source_pins.py`; there is no pin-bypass argument.

```python
# Put solver_ep22_v1 on sys.path after selecting the candidate Newton site.
from contact_adapter170 import SurfaceContactAdapter, RigidSoftCapacityGuard

pipeline = newton.CollisionPipeline(model, **explicit_capacity_and_contact_options)
solver = newton.solvers.SolverVBD(
    model,
    collision_pipeline=pipeline,
    particle_enable_self_contact=True,
    **explicit_solver_options,
)
surface_guard = SurfaceContactAdapter.for_solver(solver)
capacity_guard = RigidSoftCapacityGuard(model).attach_solver(solver)
```

Call both constructors after SolverVBD configures its detector, and before the first collision/step. The pipeline must be the actual solver-owned `CollisionPipeline` with identical `model`; the detached detector path is rejected. The wrapper also instruments detector replacement and externally rebound Contacts buffers. A common-world model must contain only finite-mass volume particles and triangles equal to its exact Tet boundary; unsupported mixed surface/volume particle models fail rather than receive an invented mask.

The only VF-kernel modification is an early return for interior query vertices, resetting their output count/minimum-distance. All true surface VF, stock edge-edge, world/BVH grouping, adjacency, topology/rest-distance filters and reverse triangle records remain. Interior elasticity DOFs are untouched. The 1.7 launch uses `_vertex_collision_block_size()` (which supplies the new default) instead of directly reading the old optional block-size field. All four native resize flags are checked.

Each VF/EE refresh synchronizes tiny native counters and fails before force consumption if any row overflows or a native resize flag is nonzero. The companion guard retains raw rigid narrow-phase, global rigid/soft counters and the new contact-reduction loss flag, broad-phase capacity and per-body rigid/soft refresh checks. Global contact-list saturation is rejected conservatively. No buffer capacities are changed, and CUDA graph capture is deliberately rejected. Failure is sticky and retained in `report()`; telemetry is bounded while every refresh is checked.

The CPU controls exercise actual native collision kernels and capacity refreshes, without stepping physics. GPU, full-scene behavior, task completion and runtime cost require their own retained result. See `cpu_detection_01/report.json` and `cpu_capacity_01/report.json` once written. These are diagnostic adapter checks, not physical qualification.
