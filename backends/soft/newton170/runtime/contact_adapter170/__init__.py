"""Owned, exact-source-pinned Newton 1.7.0 instance contact adapter."""

from .contact_adapter import SurfaceContactAdapter, ContactCapacityError
from .rigid_soft_capacity_guard import RigidSoftCapacityGuard, RigidSoftCapacityError

__all__ = [
    "SurfaceContactAdapter",
    "ContactCapacityError",
    "RigidSoftCapacityGuard",
    "RigidSoftCapacityError",
]
