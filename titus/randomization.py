"""Deterministic, bounded placement sampling in metres."""

import itertools
import math


def sample_count(spec, rng, maximum):
    if isinstance(spec, dict):
        if set(spec) == {"choices"}:
            value = rng.choice(spec["choices"])
        elif set(spec) == {"min", "max"}:
            value = rng.randint(spec["min"], spec["max"])
        else:
            raise ValueError("Count must be an integer, choices, or min/max")
    else:
        value = spec
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"Count must be an integer from 1 to {maximum}")
    return value


def rotated_bounds(bounds, yaw_deg):
    angle = math.radians(yaw_deg)
    c, s = math.cos(angle), math.sin(angle)
    points = [(c*x-s*y, s*x+c*y, z) for x, y, z in itertools.product(*zip(*bounds))]
    return [[min(p[i] for p in points) for i in range(3)],
            [max(p[i] for p in points) for i in range(3)]]


def overlaps_xy(a, b, gap):
    return all(a[0][axis] < b[1][axis] + gap and b[0][axis] < a[1][axis] + gap
               for axis in (0, 1))


def placed_bounds(local, origin):
    return [[local[side][axis] + origin[axis] for axis in range(3)] for side in (0, 1)]


def sample_placement(bounds, region, yaw_range, base_z, occupied, rng, *, gap, radius):
    for _ in range(1500):
        center = [rng.uniform(*region[axis]) for axis in (0, 1)]
        yaw = rng.uniform(*yaw_range)
        local = rotated_bounds(bounds, yaw)
        origin = [center[i] - (local[0][i] + local[1][i]) / 2 for i in (0, 1)]
        origin.append(base_z - local[0][2])
        box = placed_bounds(local, origin)
        if any(math.hypot(x, y) > radius for x, y in itertools.product(
                [box[0][0], box[1][0]], [box[0][1], box[1][1]])):
            continue
        if any(overlaps_xy(box, other, gap) for other in occupied):
            continue
        return dict(center_xy_m=center, yaw_deg=yaw, base_z_m=base_z,
                    origin_m=origin, bounds_m=box)
    return None
