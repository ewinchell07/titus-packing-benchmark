"""Native Isaac calibrated robot, free carton, and one CoACD rigid bag.

`state.particle_q` is a compatibility view of rigidly transformed observation
landmarks. These are not particles, constraints, or commanded physics states.
Only native joint-drive targets are commanded after initialization.
"""

import hashlib, importlib.metadata, itertools, json
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
from robot_kinematics import RobotKinematics
from box_grasp_metrics import grasp_motion


class ObservedArray:
    def __init__(self, value):
        self.value = np.asarray(value)

    def numpy(self):
        return self.value.copy()


class NativeRigidWorld:
    dt = 1 / 600

    def __init__(self, root, app):
        import omni.usd, omni.timeline, torch
        from isaacsim.core.simulation_manager import SimulationManager

        self.root, self.app, self.torch, self.manager = root, app, torch, SimulationManager
        self.task_config = json.loads((root / 'config/task.json').read_text())
        self.status, self.error, self.time = 'running', None, 0.0
        self.max_inverted, self.max_mimic_error = 0, 0.0
        self.box_hold_metrics = None
        self.versions = {n: importlib.metadata.version(n) for n in ['isaacsim', 'numpy', 'scipy', 'torch']}
        if self.versions['isaacsim'] != '6.2.0.0':
            raise RuntimeError('Use Isaac Sim 6.2.0.0')
        audit = json.loads((root / 'scene/scene_lock.json').read_text())
        for name, digest in audit['calibrated_robot_files'].items():
            if hashlib.sha256((root / 'assets/robot' / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError('Calibrated robot file changed: ' + name)
        self.stage_path = root / 'scene/rigid_bag_box.usda'
        self.stage = Usd.Stage.Open(str(self.stage_path))
        self.kin = RobotKinematics(self.stage)
        left = [0.8, 1.61612117, 0.58079654, 1.02635992, 0.93404287, 1.55928135]
        right = [-0.04787518, 1.40745401, 0.62199587, -0.11615930, -0.21305409, 0.23174639]
        seed = {
            f'{side}_joint_{i}': v
            for side, values in [('left', left), ('right', right)]
            for i, v in enumerate(values, 1)
        }
        seed.update(
            left_finger_joint1=0.00022357,
            left_finger_joint2=-0.00022357,
            right_finger_joint1=-0.04,
            right_finger_joint2=0.04,
        )
        neutral = self.kin.forward(dict(seed, right_finger_joint1=0.0, right_finger_joint2=0.0))
        inverse = np.linalg.inv(neutral['/World/Rig/right_hand'])
        landmarks = json.loads((root / 'assets/robot/physics/gripper_contacts.json').read_text())[
            'pad_local_landmarks_m'
        ]
        self.tcp_local = np.mean(
            [
                (inverse @ neutral['/World/Rig/' + name] @ np.r_[landmarks[i], 1])[:3]
                for name, i in [('right_left_finger', 2), ('right_right_finger', 3)]
            ],
            axis=0,
        )
        yaw = Rotation.from_euler('z', 18, degrees=True)
        R = yaw * Rotation.from_euler('x', 180, degrees=True)
        target = np.eye(4)
        target[:3, :3] = R.as_matrix()
        target[:3, 3] = np.r_[self.task_config['bag_pinch_xy_m'], 0.909] - R.apply(self.tcp_local)
        self.grasp, _ = self.kin.solve(target, seed)
        target[:3, 3] -= 0.04 * R.as_matrix()[:, 2]
        self.pregrasp, _ = self.kin.solve(target, self.grasp)
        L = Rotation.from_matrix(self.task_config['left_hand_rotation'])
        target[:3, :3] = L.as_matrix()
        target[:3, 3] = (
            np.asarray(self.task_config['box_reset_xyz_m'])
            + Rotation.from_quat(self.task_config['box_reset_quaternion_xyzw']).apply(
                self.task_config['left_grip_point_box_m']
            )
            - L.apply(self.tcp_local)
        )
        self.pregrasp, _ = self.kin.solve(target, self.pregrasp, 'left')
        o = self.task_config['initial_left_opening_m']
        self.pregrasp.update(left_finger_joint1=-o, left_finger_joint2=o)
        self.target_posture, self.target_opening = self.pregrasp.copy(), 0.05
        mesh = np.load(root / 'geometry/packed_tets.npz')
        self.canonical = mesh['points'].astype(float)
        self.tets = mesh['tets']
        v = self.canonical[self.tets]
        vol = np.einsum('ij,ij->i', v[:, 1] - v[:, 0], np.cross(v[:, 2] - v[:, 0], v[:, 3] - v[:, 0])) / 6
        hulls = np.load(root / 'geometry/coacd_hulls.npz')
        self.hull_vertices = np.concatenate([hulls[k] for k in hulls.files if k.startswith('vertices_')])
        self.mass = np.zeros(len(self.canonical))
        np.add.at(self.mass, self.tets.ravel(), np.repeat(0.036 * vol / vol.sum() / 4, 4))
        SimulationManager.switch_physics_engine('physx')
        self.context = omni.usd.get_context()
        self.context.open_stage(str(self.stage_path))
        for _ in range(3):
            app.update()
        self.native_stage = self.context.get_stage()
        self.native_stage.SetEditTarget(self.native_stage.GetSessionLayer())
        self.native_stage.SetTimeCodesPerSecond(600)
        self.native_stage.SetEndTimeCode(3600000.0)
        from visuals import add_camera_lights

        add_camera_lights(self.native_stage)
        self.visuals = None
        for label in [
            '/World/Rig/left_left_finger',
            '/World/Rig/left_right_finger',
            '/World/Rig/right_left_finger',
            '/World/Rig/right_right_finger',
            '/World/DashCarton',
            '/World/Bag',
        ]:
            PhysxSchema.PhysxContactReportAPI.Apply(
                self.native_stage.GetPrimAtPath(label)
            ).CreateThresholdAttr(0.0)
        scene = PhysxSchema.PhysxSceneAPI.Apply(self.native_stage.GetPrimAtPath('/World/PhysicsScene'))
        scene.CreateEnableGPUDynamicsAttr(True)
        scene.CreateBroadphaseTypeAttr('GPU')
        scene.CreateSolverTypeAttr('TGS')
        scene.CreateTimeStepsPerSecondAttr(600)
        SimulationManager.setup_simulation(dt=self.dt, device='cuda:0')
        self.timeline = omni.timeline.get_timeline_interface()
        self.timeline.play()
        SimulationManager.initialize_physics()
        self.native_start_time = float(SimulationManager.get_simulation_time())
        self.native_dt = float(SimulationManager.get_physics_dt())
        assert abs(self.native_dt - self.dt) < 1e-9
        sim = SimulationManager.get_physics_sim_view()
        self.rig = sim.create_articulation_view('/World/Rig')
        labels = [
            str(p.GetPath())
            for p in self.native_stage.Traverse()
            if p.HasAPI(UsdPhysics.RigidBodyAPI)
            and UsdPhysics.RigidBodyAPI(p).GetRigidBodyEnabledAttr().Get()
        ]
        self.rigid = sim.create_rigid_body_view(labels)
        self.native_map = np.array([list(self.rigid.prim_paths).index(n) for n in labels])
        self.model = SimpleNamespace(body_label=labels)
        self.native_names = [str(p).rsplit('/', 1)[-1] for p in np.asarray(self.rig.dof_paths).reshape(-1)]
        self.rig_ids = self.tensor([0], dtype=torch.int32)
        initial = self.native_values(self.pregrasp, 0.05)
        self.rig.set_dof_positions(self.tensor(initial), self.rig_ids)
        self.rig.set_dof_velocities(self.tensor(np.zeros_like(initial)), self.rig_ids)
        self.rig.set_dof_position_targets(self.tensor(initial), self.rig_ids)
        self.rig.set_dof_velocity_targets(self.tensor(np.zeros_like(initial)), self.rig_ids)
        sim.update_articulations_kinematic()
        self.previous_command = initial.copy()
        self.left_box_contacts = sim.create_rigid_contact_view(
            ['/World/Rig/left_left_finger', '/World/Rig/left_right_finger'],
            [['/World/DashCarton'], ['/World/DashCarton']],
            max_contact_data_count=2048,
        )
        self.right_bag_contacts = sim.create_rigid_contact_view(
            ['/World/Rig/right_left_finger', '/World/Rig/right_right_finger'],
            [['/World/Bag'], ['/World/Bag']],
            max_contact_data_count=2048,
        )
        self.bag_contacts = sim.create_rigid_contact_view(
            ['/World/Bag'],
            [['/World/DashCarton', '/World/RotaryTableV2/dish_collision', '/World/Floor']],
            max_contact_data_count=4096,
        )
        self._box_body = labels.index('/World/DashCarton')
        self._bag_body = labels.index('/World/Bag')
        self._left_hand_body = labels.index('/World/Rig/left_hand')
        self._right_hand_body = labels.index('/World/Rig/right_hand')
        cache = UsdGeom.XformCache()
        inv = cache.GetLocalToWorldTransform(
            self.native_stage.GetPrimAtPath('/World/DashCarton')
        ).GetInverse()
        corners = np.array(list(itertools.product([-0.5, 0.5], repeat=3)))
        pieces = []
        for p in Usd.PrimRange(self.native_stage.GetPrimAtPath('/World/DashCarton/ContactPanels')):
            if p.IsA(UsdGeom.Cube):
                T = np.asarray(cache.GetLocalToWorldTransform(p) * inv)
                pieces.append(
                    (np.c_[corners * float(UsdGeom.Cube(p).GetSizeAttr().Get()), np.ones(8)] @ T)[:, :3]
                )
        self._box_corners = np.concatenate(pieces)
        self.body_com = self.array(self.rigid.get_coms())[self.native_map, :3]
        self.native_report = dict(
            backend='Isaac Sim 6.2 native PhysX TGS articulation',
            physics_dt_s=self.dt,
            body_masses_kg=self.array(self.rigid.get_masses())[self.native_map].reshape(-1).tolist(),
            dof_names=self.native_names,
            max_efforts=self.array(self.rig.get_dof_max_forces()).tolist(),
            stiffness=self.array(self.rig.get_dof_stiffnesses()).tolist(),
            damping=self.array(self.rig.get_dof_dampings()).tolist(),
            rigid_paths=labels,
            controller_pose_writes_after_reset=0,
        )
        assert abs(self.native_report['body_masses_kg'][self._bag_body] - 0.036) < 1e-6
        self._observe_native()
        self.rest = self.state.particle_q.numpy()
        print(
            json.dumps(dict(native_robot_ready=self.native_report, bag='Native free CoACD rigid body')),
            flush=True,
        )

    @staticmethod
    def array(x):
        return x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)

    def tensor(self, x, dtype=None):
        return self.torch.as_tensor(np.asarray(x), dtype=dtype or self.torch.float32, device='cuda:0')

    def native_values(self, posture, opening):
        q = dict(posture, right_finger_joint1=-opening, right_finger_joint2=opening)
        return np.array([[q[n] for n in self.native_names]], np.float32)

    def read_rigid(self):
        return (
            self.array(self.rigid.get_transforms())[self.native_map],
            self.array(self.rigid.get_velocities())[self.native_map],
        )

    def _observe_native(self):
        self._native_poses, self._native_velocities = self.read_rigid()
        p = self._native_poses[self._bag_body]
        r = Rotation.from_quat(p[3:])
        nodes = r.apply(self.canonical) + p[:3]
        vel = self._native_velocities[self._bag_body]
        node_vel = vel[:3] + np.cross(
            np.broadcast_to(vel[3:], nodes.shape), r.apply(self.canonical - self.body_com[self._bag_body])
        )
        self.state = SimpleNamespace(
            body_q=ObservedArray(self._native_poses),
            particle_q=ObservedArray(nodes),
            particle_qd=ObservedArray(node_vel),
        )
        self.actual = dict(zip(self.native_names, self.array(self.rig.get_dof_positions())[0]))
        self.max_mimic_error = float(
            max(
                self.max_mimic_error,
                max(
                    abs(self.actual[f'{s}_finger_joint1'] + self.actual[f'{s}_finger_joint2'])
                    for s in ['left', 'right']
                ),
            )
        )
        if self.max_mimic_error > 0.001:
            raise RuntimeError('Native passive finger linkage error')

    def step(self, posture, opening):
        command = self.native_values(posture, opening)
        self.rig.set_dof_position_targets(self.tensor(command), self.rig_ids)
        self.rig.set_dof_velocity_targets(
            self.tensor((command - self.previous_command) / self.dt), self.rig_ids
        )
        self.previous_command = command.copy()
        before = self.manager.get_simulation_time()
        self.manager.step(steps=1)
        if abs(self.manager.get_simulation_time() - before - self.dt) > 1e-6:
            raise RuntimeError('Native clock mismatch')
        self._observe_native()
        self.target_posture, self.target_opening = posture.copy(), opening
        self.time += self.dt

    def finger_contact_forces(self):
        return np.linalg.norm(
            self.array(self.right_bag_contacts.get_normal_contact_force_matrix(self.dt))
            .reshape(2, -1, 3)
            .sum(1),
            axis=1,
        ).tolist()

    def track_box_hold(self, reference):
        box = self._native_poses[self._box_body]
        motion = grasp_motion(self._native_poses[self._left_hand_body], box, reference)
        clearance = (
            float((Rotation.from_quat(box[3:]).apply(self._box_corners) + box[:3])[:, 2].min())
            - 0.8634426593780518
        )
        if self.box_hold_metrics is None:
            self.box_hold_metrics = dict(
                samples=0,
                maximum_box_slip_m=0.0,
                maximum_box_origin_displacement_m=0.0,
                maximum_box_relative_rotation_deg=0.0,
                maximum_box_upright_error_deg=0.0,
                minimum_box_clearance_m=1e9,
            )
        m = self.box_hold_metrics
        m['samples'] += 1
        for dest, src in [
            ('maximum_box_slip_m', 'grasp_anchor_slip_m'),
            ('maximum_box_origin_displacement_m', 'origin_displacement_m'),
            ('maximum_box_relative_rotation_deg', 'relative_rotation_deg'),
            ('maximum_box_upright_error_deg', 'upright_error_deg'),
        ]:
            m[dest] = max(m[dest], motion[src])
        m['minimum_box_clearance_m'] = min(m['minimum_box_clearance_m'], clearance)
        return m

    def measure(self):
        nodes = self.state.particle_q.numpy()
        box = self._native_poses[self._box_body]
        r = Rotation.from_quat(box[3:])
        local = r.inv().apply(nodes - box[:3])
        inside = (
            (local[:, 0] >= -0.0682)
            & (local[:, 0] <= 0.0683)
            & (abs(local[:, 1]) <= 0.0877)
            & (local[:, 2] >= -0.0025)
            & (local[:, 2] <= 0.213)
        )
        bag = self._native_poses[self._bag_body]
        hull_local = r.inv().apply(Rotation.from_quat(bag[3:]).apply(self.hull_vertices) + bag[:3] - box[:3])
        hulls_inside = (
            (hull_local[:, 0] >= -0.0682)
            & (hull_local[:, 0] <= 0.0683)
            & (abs(hull_local[:, 1]) <= 0.0877)
            & (hull_local[:, 2] >= -0.0025)
            & (hull_local[:, 2] <= 0.213)
        )
        left_vectors = (
            self.array(self.left_box_contacts.get_normal_contact_force_matrix(self.dt))
            .reshape(2, -1, 3)
            .sum(1)
        )
        contact = self.array(self.bag_contacts.get_normal_contact_force_matrix(self.dt)).reshape(3, 3)
        magnitudes = np.linalg.norm(contact, axis=1)
        return dict(
            t_s=self.time,
            native_elapsed_s=float(self.manager.get_simulation_time() - self.native_start_time),
            centroid_m=((self.mass / self.mass.sum()) @ nodes).tolist(),
            lowest_z_m=float(nodes[:, 2].min()),
            max_inverted_tets=0,
            max_speed_m_s=float(np.linalg.norm(self.state.particle_qd.numpy(), axis=1).max()),
            hand_pose=self._native_poses[self._right_hand_body].tolist(),
            finger_contact_force_N=self.finger_contact_forces(),
            support_contact_force_N=float(magnitudes[1:].sum()),
            actual_joint_q={k: float(v) for k, v in self.actual.items()},
            box_pose=box.tolist(),
            box_lowest_z_m=float((r.apply(self._box_corners) + box[:3])[:, 2].min()),
            left_hand_pose=self._native_poses[self._left_hand_body].tolist(),
            left_box_contact_force_N=np.linalg.norm(left_vectors, axis=1).tolist(),
            left_box_normal_vectors_N=left_vectors.tolist(),
            box_upright_error_deg=float(np.degrees(np.arccos(np.clip(r.as_matrix()[2, 2], -1, 1)))),
            bag_inside_box_fraction=float(inside.mean()),
            bag_box_local_min_m=local.min(0).tolist(),
            bag_box_local_max_m=local.max(0).tolist(),
            bag_carton_contact_force_N=float(magnitudes[0]),
            box_linear_speed_m_s=float(np.linalg.norm(self._native_velocities[self._box_body, :3])),
            max_native_mimic_error_m=self.max_mimic_error,
            coacd_hulls_inside_box_fraction=float(hulls_inside.mean()),
            bag_representation='rigid_coacd',
            bag_pose=self._native_poses[self._bag_body].tolist(),
        )

    def frame(self):
        return self.time, self.state.particle_q.numpy(), self._native_poses.copy()

    def render(self, path):
        from visuals import Visuals

        if self.visuals is None:
            self.visuals = Visuals(self)
        return self.visuals.render(path)

    def save(self, out, rows, frames, no_close=False):
        (out / 'telemetry.json').write_text(json.dumps(rows, indent=2) + '\n')
        if frames:
            np.savez_compressed(
                out / 'states.npz',
                t=[f[0] for f in frames],
                nodes=[f[1] for f in frames],
                body_q=[f[2] for f in frames],
                tets=self.tets,
                rest_points=self.rest,
                particle_mass=self.mass,
                body_labels=np.array(self.model.body_label),
            )
        report = dict(
            versions=self.versions,
            simulated_seconds=self.time,
            status=self.status,
            error=self.error,
            accepted_grasp=False,
            no_close=no_close,
            object_attachments=0,
            object_pose_writes_after_reset=0,
            object_forces=0,
            physical_object_attachments=0,
            max_inverted_tets=0,
            mass_kg=0.036,
            backend='Isaac Sim 6.2 native PhysX calibrated robot + CoACD rigid bag',
            bag_representation='rigid_coacd',
            scope='One free rigid body; observation landmarks are rigidly transformed points, not soft particles. No external contact coupling or object commands.',
            native_robot=self.native_report,
            max_native_mimic_error_m=self.max_mimic_error,
            continuous_box_hold_metrics=self.box_hold_metrics,
            tcp_from_delivered_pad_landmarks_m=self.tcp_local.tolist(),
            robot_calibration=json.loads((self.root / 'scene/robot_calibration.json').read_text()),
            tet_mesh_sha256=hashlib.sha256((self.root / 'geometry/packed_tets.npz').read_bytes()).hexdigest(),
            coacd=json.loads((self.root / 'geometry/coacd_receipt.json').read_text()),
            runtime_settings=dict(dt=self.dt, robot_engine='Native PhysX TGS', bag_engine='Native PhysX TGS'),
            qualification='Recorded attempt; independent acceptance is required',
        )
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
