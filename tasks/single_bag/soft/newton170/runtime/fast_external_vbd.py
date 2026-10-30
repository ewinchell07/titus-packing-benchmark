"""Cache only the stock VBD particle iteration, between collision refreshes.

Collision detection and all capacity checks remain uncaptured. The graph records
the original Newton kernels in their original order; it adds no physics law.
"""

import warp as wp
from newton.solvers import SolverVBD
from newton._src.solvers.solver import SolverBase


class FastExternalVBD(SolverVBD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.integrate_with_external_rigid_solver or self.rigid_soft_enable_dat:
            raise ValueError("Iteration graphs require the external rigid path without rigid DAT")
        self._iteration_graphs = {}
        self._graph_warm = False
        self._graph_layout = None
        self.iteration_graph_launches = 0

    def step(self, *args, **kwargs):
        self._graph_layout = None
        result = super().step(*args, **kwargs)
        self._graph_warm = True
        return result

    @staticmethod
    def _array_layout(owner):
        return tuple(
            (name, value.ptr, tuple(value.shape))
            for name, value in sorted(vars(owner).items())
            if isinstance(value, wp.array)
        )

    def _solve_particle_iteration(self, state_in, state_out, contacts, dt):
        if not self._graph_warm:
            return super()._solve_particle_iteration(state_in, state_out, contacts, dt)
        frequency = SolverBase.CollisionFrequencyType
        allowed = {frequency.NONE, frequency.PRE_INIT, frequency.PRE_POST_INIT}
        if self._rigid_mode_this_step not in allowed or self._self_contact_mode_this_step not in allowed:
            raise RuntimeError("Iteration graphs require collision refreshes outside the iteration loop")
        if self._graph_layout is None:
            scalar_names = [
                "friction_epsilon",
                "particle_enable_self_contact",
                "particle_self_contact_margin",
                "dat_conservative_bound_relaxation",
                "rigid_soft_contact_use_log_barrier",
                "use_particle_tile_solve",
                "_particle_contact_use_gather",
            ]
            self._graph_layout = (
                self._array_layout(self),
                self._array_layout(self.model),
                self._array_layout(contacts),
                tuple((n, getattr(self, n)) for n in scalar_names),
                self.model.soft_contact_ke,
                self.model.soft_contact_kd,
                self.model.soft_contact_mu,
            )
        key = (self._graph_layout, self._array_layout(state_in), self._array_layout(state_out), float(dt))
        graph = self._iteration_graphs.get(key)
        if graph is None:
            with wp.ScopedCapture(device=self.device) as capture:
                super()._solve_particle_iteration(state_in, state_out, contacts, dt)
            graph = capture.graph
            self._iteration_graphs[key] = graph
        wp.capture_launch(graph)
        self.iteration_graph_launches += 1
