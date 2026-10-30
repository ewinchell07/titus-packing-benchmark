"""Plan a nearby collision-envelope orientation that fits the carton interior.

This computes robot targets from observed poses. It never edits object state.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


def fit_orientation(vertices, box_from_bag, limits=(0.1265, 0.1654, 0.203)):
    vertices = np.asarray(vertices)
    limits = np.asarray(limits)

    def orientation(v):
        return Rotation.from_rotvec(v) * box_from_bag

    def slack(v):
        return limits - np.ptp(orientation(v).apply(vertices), axis=0)

    best = None
    for initial in [
        np.zeros(3),
        np.array([0.2, 0, 0]),
        np.array([-0.2, 0, 0]),
        np.array([0, 0.2, 0]),
        np.array([0, -0.2, 0]),
    ]:
        result = minimize(
            lambda v: float(v @ v),
            initial,
            method='SLSQP',
            bounds=[(-0.7, 0.7)] * 3,
            constraints=[dict(type='ineq', fun=slack)],
            options=dict(maxiter=150, ftol=1e-11),
        )
        if min(slack(result.x)) >= -1e-7 and (best is None or result.fun < best.fun):
            best = result
    if best is None:
        raise RuntimeError('No nearby rigid packet orientation fits the carton')
    return orientation(best.x), dict(
        rotation_change_deg=float(np.degrees(np.linalg.norm(best.x))),
        projected_extents_m=np.ptp(orientation(best.x).apply(vertices), axis=0).tolist(),
        limits_m=limits.tolist(),
    )
