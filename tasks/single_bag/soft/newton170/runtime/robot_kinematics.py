"""Forward/inverse kinematics from the scene's native USD joint frames.

Planning only: these functions never write simulator state or object poses.
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def transform(position, quaternion):
    q = np.r_[np.asarray(quaternion.GetImaginary()), quaternion.GetReal()]
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(q).as_matrix()
    result[:3, 3] = position
    return result


class RobotKinematics:
    def __init__(self, stage):
        from pxr import UsdPhysics

        self.joints = []
        self.limits = {}
        for prim in stage.Traverse():
            if not prim.IsA(UsdPhysics.Joint):
                continue
            joint = UsdPhysics.Joint(prim)
            child = joint.GetBody1Rel().GetTargets()
            parent = joint.GetBody0Rel().GetTargets()
            if not child or not str(child[0]).startswith('/World/Rig/'):
                continue
            kind = prim.GetTypeName()
            name = prim.GetName()
            axis = prim.GetAttribute('physics:axis').Get()
            self.joints.append(
                dict(
                    name=name,
                    kind=kind,
                    parent=str(parent[0]) if parent else '',
                    child=str(child[0]),
                    A=transform(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get()),
                    B=np.linalg.inv(
                        transform(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get())
                    ),
                    axis=np.eye(3)['XYZ'.index(axis)] if axis else np.zeros(3),
                )
            )
            if 'Revolute' in kind or 'Prismatic' in kind:
                lo, hi = [prim.GetAttribute('physics:' + x + 'Limit').Get() for x in ['lower', 'upper']]
                scale = np.pi / 180 if 'Revolute' in kind else 1.0
                self.limits[name] = (float(lo) * scale, float(hi) * scale)

    def forward(self, q):
        poses = {'': np.eye(4)}
        pending = self.joints.copy()
        while pending:
            progressed = False
            for joint in pending.copy():
                if joint['parent'] not in poses:
                    continue
                motion = np.eye(4)
                value = q.get(joint['name'], 0.0)
                if 'Revolute' in joint['kind']:
                    motion[:3, :3] = Rotation.from_rotvec(joint['axis'] * value).as_matrix()
                elif 'Prismatic' in joint['kind']:
                    motion[:3, 3] = joint['axis'] * value
                poses[joint['child']] = poses[joint['parent']] @ joint['A'] @ motion @ joint['B']
                pending.remove(joint)
                progressed = True
            if not progressed:
                raise ValueError('Disconnected USD robot joint graph')
        return poses

    def solve(self, target, seed, side='right'):
        names = [f'{side}_joint_{i}' for i in range(1, 7)]
        bounds = np.asarray([self.limits[n] for n in names]).T
        bounds = np.where(np.isfinite(bounds), bounds, np.sign(bounds) * 6.28)
        hand = '/World/Rig/' + side + '_hand'

        def residual(values):
            q = dict(seed, **dict(zip(names, values)))
            pose = self.forward(q)[hand]
            return np.r_[
                (pose[:3, 3] - target[:3, 3]) * 10,
                Rotation.from_matrix(target[:3, :3] @ pose[:3, :3].T).as_rotvec(),
            ]

        result = least_squares(
            residual,
            [seed.get(n, 0.0) for n in names],
            bounds=bounds,
            max_nfev=500,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )
        q = dict(seed, **dict(zip(names, result.x)))
        pose = self.forward(q)[hand]
        error = dict(
            position_m=float(np.linalg.norm(pose[:3, 3] - target[:3, 3])),
            orientation_deg=float(
                np.degrees(np.linalg.norm(Rotation.from_matrix(target[:3, :3] @ pose[:3, :3].T).as_rotvec()))
            ),
        )
        if error['position_m'] > 0.0005 or error['orientation_deg'] > 0.5:
            raise ValueError('IK target unreachable: ' + str(error))
        return q, error

    def solve_position(self, target_xyz, seed, side='right'):
        """Find a reachable lift pose, allowing wrist rotation within source limits."""
        names = [f'{side}_joint_{i}' for i in range(1, 7)]
        bounds = np.asarray([self.limits[n] for n in names]).T
        start = np.asarray([seed[n] for n in names])
        hand = '/World/Rig/' + side + '_hand'
        prior = self.forward(seed)[hand][:3, :3]

        def residual(values):
            pose = self.forward(dict(seed, **dict(zip(names, values))))[hand]
            angle = Rotation.from_matrix(prior @ pose[:3, :3].T).as_rotvec()
            return np.r_[(pose[:3, 3] - target_xyz) * 1000, angle * 0.01, (values - start) * 0.0001]

        result = least_squares(
            residual, start, bounds=bounds, max_nfev=500, ftol=1e-12, xtol=1e-12, gtol=1e-12
        )
        q = dict(seed, **dict(zip(names, result.x)))
        pose = self.forward(q)[hand]
        error = dict(
            position_m=float(np.linalg.norm(pose[:3, 3] - target_xyz)),
            orientation_change_deg=float(
                np.degrees(np.linalg.norm(Rotation.from_matrix(prior @ pose[:3, :3].T).as_rotvec()))
            ),
        )
        if error['position_m'] > 0.0005:
            raise ValueError('Position IK unreachable: ' + str(error))
        return q, error
