"""Calibrated native Isaac articulation coupled to a free Newton VBD packet.

The Newton collision boundary follows measured native rigid state. Contact-only
reaction wrenches return to those same native bodies on the next physics tick.
Only joint targets are commanded. No bag poses, carrier or attachments are used.
"""

import json
import itertools
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from single_bag_world import SingleBagWorld, signed_volume


class IsaacNewtonWorld(SingleBagWorld):
    external_native = True

    def __init__(self, root, app):
        self.app = app
        super().__init__(root)
        import omni.usd
        import omni.timeline
        from pxr import PhysxSchema, Usd
        import torch
        import warp as wp
        from isaacsim.core.simulation_manager import SimulationManager

        self.wp, self.torch, self.manager = wp, torch, SimulationManager
        SimulationManager.switch_physics_engine("physx")
        self.context = omni.usd.get_context()
        self.context.open_stage(str(self.stage_path))
        for _ in range(3):
            self.app.update()
        self.native_stage = self.context.get_stage()
        self.native_stage.SetEditTarget(self.native_stage.GetSessionLayer())
        self.native_stage.SetStartTimeCode(0.0)
        self.native_stage.SetEndTimeCode(3600000.0)
        self.native_stage.SetTimeCodesPerSecond(1 / self.dt)
        from visuals import add_camera_lights

        add_camera_lights(self.native_stage)
        self.visuals = None
        for label in ['/World/Rig/left_left_finger', '/World/Rig/left_right_finger', '/World/DashCarton']:
            PhysxSchema.PhysxContactReportAPI.Apply(
                self.native_stage.GetPrimAtPath(label)
            ).CreateThresholdAttr(0.0)
        scene = PhysxSchema.PhysxSceneAPI.Apply(self.native_stage.GetPrimAtPath("/World/PhysicsScene"))
        scene.CreateEnableGPUDynamicsAttr(True)
        scene.CreateBroadphaseTypeAttr("GPU")
        scene.CreateSolverTypeAttr("TGS")
        scene.CreateTimeStepsPerSecondAttr(round(1 / self.dt))
        SimulationManager.setup_simulation(dt=self.dt, device="cuda:0")
        self.timeline = omni.timeline.get_timeline_interface()
        self.timeline.play()
        SimulationManager.initialize_physics()
        self.native_dt = float(SimulationManager.get_physics_dt())
        if abs(self.native_dt - self.dt) > 1e-9:
            raise RuntimeError(f'Native/Newton timestep mismatch: {self.native_dt} versus {self.dt}')
        self.native_start_time = float(SimulationManager.get_simulation_time())
        sim = SimulationManager.get_physics_sim_view()
        self.rig = sim.create_articulation_view("/World/Rig")
        self.rigid = sim.create_rigid_body_view(list(self.model.body_label))
        self.left_box_contacts = sim.create_rigid_contact_view(
            ['/World/Rig/left_left_finger', '/World/Rig/left_right_finger'],
            [['/World/DashCarton'], ['/World/DashCarton']],
            max_contact_data_count=1024,
        )
        native_paths = list(self.rigid.prim_paths)
        self.native_map = np.array([native_paths.index(label) for label in self.model.body_label], int)
        self.native_names = [str(p).rsplit("/", 1)[-1] for p in np.asarray(self.rig.dof_paths).reshape(-1)]
        self.rig_ids = self.tensor([0], dtype=torch.int32)
        self.body_ids = self.tensor(np.arange(self.rigid.count), dtype=torch.int32)
        initial = self.native_values(self.pregrasp, 0.05)
        self.rig.set_dof_positions(self.tensor(initial), self.rig_ids)
        self.rig.set_dof_velocities(self.tensor(np.zeros_like(initial)), self.rig_ids)
        self.rig.set_dof_position_targets(self.tensor(initial), self.rig_ids)
        self.rig.set_dof_velocity_targets(self.tensor(np.zeros_like(initial)), self.rig_ids)
        sim.update_articulations_kinematic()
        self.previous_command = initial.copy()
        self.pending = np.zeros((self.model.body_count, 6), np.float32)
        self.feedback_steps = 0
        self.max_mimic_error = 0.0
        self._shape_body = self.model.shape_body.numpy()
        self._finger_shapes = np.array(
            [
                [f"/World/Rig/{finger}/" in label for label in self.model.shape_label]
                for finger in ["right_left_finger", "right_right_finger"]
            ],
            bool,
        )
        self._box_body = list(self.model.body_label).index("/World/DashCarton")
        self._left_hand_body = list(self.model.body_label).index('/World/Rig/left_hand')
        self.box_hold_metrics = None
        from pxr import UsdGeom

        cache = UsdGeom.XformCache()
        carton = self.native_stage.GetPrimAtPath("/World/DashCarton")
        inverse_carton = cache.GetLocalToWorldTransform(carton).GetInverse()
        cube_corners = np.array(list(itertools.product([-0.5, 0.5], repeat=3)))
        local_corners = []
        for prim in Usd.PrimRange(carton):
            if "/ContactPanels/" in str(prim.GetPath()) and prim.IsA(UsdGeom.Cube):
                matrix = np.asarray(cache.GetLocalToWorldTransform(prim) * inverse_carton)
                size = float(UsdGeom.Cube(prim).GetSizeAttr().Get())
                local_corners.append((np.c_[cube_corners * size, np.ones(8)] @ matrix)[:, :3])
        if not local_corners:
            raise RuntimeError("No explicit carton panel colliders")
        self._box_corners = np.concatenate(local_corners)
        self.contact_forces = np.empty((0, 6))
        self.contact_shapes = np.empty(0, int)
        self.contact_observables = None
        self.contacts = self.soft.contacts
        self.native_report = dict(
            backend="Isaac Sim 6.2 native PhysX TGS articulation",
            physics_dt_s=self.native_dt,
            box_contact_sensor_count=self.left_box_contacts.sensor_count,
            box_contact_filter_count=self.left_box_contacts.filter_count,
            body_masses_kg=self.array(self.rigid.get_masses()).reshape(-1).tolist(),
            dof_names=self.native_names,
            max_efforts=self.array(self.rig.get_dof_max_forces()).tolist(),
            stiffness=self.array(self.rig.get_dof_stiffnesses()).tolist(),
            damping=self.array(self.rig.get_dof_dampings()).tolist(),
            rigid_paths=native_paths,
            controller_pose_writes_after_reset=0,
        )
        poses, velocities = self.read_rigid()
        self._native_poses = poses
        self._native_velocities = velocities
        for state in [self.state, self.next_state]:
            state.body_q.assign(poses)
            state.body_qd.assign(velocities)
        self.solver.reset(self.state, flags=0)
        print(json.dumps(dict(native_robot_ready=self.native_report)), flush=True)

    @staticmethod
    def array(value):
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return value.numpy() if hasattr(value, "numpy") else np.asarray(value)

    def tensor(self, value, dtype=None):
        return self.torch.as_tensor(np.asarray(value), dtype=dtype or self.torch.float32, device="cuda:0")

    def native_values(self, posture, opening):
        q = dict(posture, right_finger_joint1=-opening, right_finger_joint2=opening)
        return np.array([[q[name] for name in self.native_names]], np.float32)

    def read_rigid(self):
        return (
            self.array(self.rigid.get_transforms())[self.native_map].astype(np.float32),
            self.array(self.rigid.get_velocities())[self.native_map].astype(np.float32),
        )

    def step(self, posture, opening):
        # Apply only the prior Newton contact reaction, at the COM used by Newton.
        poses, _ = self.read_rigid()
        com = Rotation.from_quat(poses[:, 3:]).apply(self.model.body_com.numpy()) + poses[:, :3]
        force = np.zeros((self.rigid.count, 3), np.float32)
        torque = force.copy()
        position = force.copy()
        force[self.native_map] = self.pending[:, :3]
        torque[self.native_map] = self.pending[:, 3:]
        position[self.native_map] = com
        self.rigid.apply_forces_and_torques_at_position(
            self.tensor(force), self.tensor(torque), self.tensor(position), self.body_ids, True
        )
        if np.any(self.pending):
            self.feedback_steps += 1
        command = self.native_values(posture, opening)
        velocity = (command - self.previous_command) / self.dt
        self.rig.set_dof_position_targets(self.tensor(command), self.rig_ids)
        self.rig.set_dof_velocity_targets(self.tensor(velocity), self.rig_ids)
        self.previous_command = command.copy()
        before_time = self.manager.get_simulation_time()
        self.manager.step(steps=1)
        native_delta = self.manager.get_simulation_time() - before_time
        if abs(native_delta - self.dt) > 1e-6:
            raise RuntimeError(
                f'Native physics advanced {native_delta} seconds for one {self.dt}-second Newton tick'
            )
        poses, velocities = self.read_rigid()
        self._native_poses = poses
        self._native_velocities = velocities
        if not np.isfinite(poses).all():
            raise RuntimeError("Nonfinite native rigid state")
        self.next_state.body_q.assign(poses)
        self.next_state.body_qd.assign(velocities)
        self.state.clear_forces()
        if self.contact_observables is None:
            self.contact_observables = self.soft.observables(["contact_f"], contacts=self.contacts)
        self.soft.step(
            self.state, self.next_state, self.control, None, self.dt, observables=self.contact_observables
        )
        self.state, self.next_state = self.next_state, self.state
        if np.max(np.abs(self.state.body_q.numpy() - poses)) > 1e-5:
            raise RuntimeError("Newton changed the authoritative native rigid pose")
        nodes = self.state.particle_q.numpy().astype(float)
        if not np.isfinite(nodes).all():
            raise RuntimeError("Nonfinite Newton packet state")
        self.max_inverted = max(
            self.max_inverted, int(np.count_nonzero(signed_volume(nodes, self.tets) * self.rest_volume <= 0))
        )
        if self.max_inverted:
            raise RuntimeError(f"Packet tetrahedra inverted: {self.max_inverted}")
        count = int(self.contacts.soft_contact_count.numpy()[0])
        self.contact_shapes = (
            self.contacts.soft_contact_shape[:count].numpy().astype(int) if count else np.empty(0, int)
        )
        offset = self.contacts.rigid_contact_max
        self.contact_forces = (
            self.contact_observables.contact_f[offset : offset + count].numpy().copy()
            if count
            else np.empty((0, 6))
        )
        bodies = self._shape_body[self.contact_shapes]
        self.pending.fill(0)
        mask = bodies >= 0
        np.add.at(self.pending, bodies[mask], self.contact_forces[mask])
        if not np.isfinite(self.pending).all():
            raise RuntimeError("Nonfinite contact reaction")
        actual = dict(zip(self.native_names, self.array(self.rig.get_dof_positions())[0]))
        error = max(
            abs(actual[f"{side}_finger_joint1"] + actual[f"{side}_finger_joint2"])
            for side in ["left", "right"]
        )
        self.max_mimic_error = max(self.max_mimic_error, float(error))
        if error > 0.001:
            raise RuntimeError(f"Native finger linkage error {error} m")
        self.target_posture, self.target_opening = posture.copy(), opening
        self.time += self.dt

    def finger_contact_forces(self):
        magnitudes = np.linalg.norm(self.contact_forces[:, :3], axis=1)
        return [float(magnitudes[mask[self.contact_shapes]].sum()) for mask in self._finger_shapes]

    def left_box_forces(self):
        matrix = self.array(self.left_box_contacts.get_normal_contact_force_matrix(self.dt))
        self.left_box_normal_vectors = matrix.reshape(2, -1, 3).sum(axis=1)
        return np.linalg.norm(self.left_box_normal_vectors, axis=1).tolist()

    def track_box_hold(self, reference):
        """Measure every native tick so a brief slip cannot hide between samples."""
        box = self._native_poses[self._box_body]
        hand = self._native_poses[self._left_hand_body]
        rotation = Rotation.from_quat(box[3:])
        from box_grasp_metrics import grasp_motion

        motion = grasp_motion(hand, box, reference)
        slip = motion["grasp_anchor_slip_m"]
        angle = float(np.degrees(np.arccos(np.clip(rotation.as_matrix()[2, 2], -1, 1))))
        clearance = float((rotation.apply(self._box_corners) + box[:3])[:, 2].min()) - 0.8634426593780518
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
        m['maximum_box_slip_m'] = max(m['maximum_box_slip_m'], slip)
        m['maximum_box_origin_displacement_m'] = max(
            m['maximum_box_origin_displacement_m'], motion['origin_displacement_m']
        )
        m['maximum_box_relative_rotation_deg'] = max(
            m['maximum_box_relative_rotation_deg'], motion['relative_rotation_deg']
        )
        m['maximum_box_upright_error_deg'] = max(m['maximum_box_upright_error_deg'], angle)
        m['minimum_box_clearance_m'] = min(m['minimum_box_clearance_m'], clearance)
        return m

    def measure(self):
        nodes = self.state.particle_q.numpy().astype(float)
        labels = [self.model.shape_label[i] for i in self.contact_shapes]
        magnitudes = np.linalg.norm(self.contact_forces[:, :3], axis=1)
        forces = self.finger_contact_forces()
        support = np.array(
            [
                label.startswith(("/World/Table/", "/World/Table", "/World/Floor", "/World/RotaryTableV2"))
                for label in labels
            ],
            bool,
        )
        hand = list(self.model.body_label).index("/World/Rig/right_hand")
        box_pose = self.state.body_q.numpy()[self._box_body]
        box_corners = Rotation.from_quat(box_pose[3:]).apply(self._box_corners) + box_pose[:3]
        box_rotation = Rotation.from_quat(box_pose[3:])
        box_nodes = box_rotation.inv().apply(nodes - box_pose[:3])
        inside = (
            (box_nodes[:, 0] >= -0.0682)
            & (box_nodes[:, 0] <= 0.0683)
            & (np.abs(box_nodes[:, 1]) <= 0.0877)
            & (box_nodes[:, 2] >= -0.0025)
            & (box_nodes[:, 2] <= 0.213)
        )
        box_contacts = np.array([label.startswith('/World/DashCarton') for label in labels], bool)
        left_hand = list(self.model.body_label).index('/World/Rig/left_hand')
        left_forces = self.left_box_forces()
        projected = self.array(self.rig.get_dof_projected_joint_forces())[0]
        return dict(
            t_s=self.time,
            centroid_m=((self.mass / self.mass.sum()) @ nodes).tolist(),
            native_elapsed_s=float(self.manager.get_simulation_time() - self.native_start_time),
            lowest_z_m=float(nodes[:, 2].min()),
            max_inverted_tets=self.max_inverted,
            max_speed_m_s=float(np.linalg.norm(self.state.particle_qd.numpy(), axis=1).max()),
            hand_pose=self.state.body_q.numpy()[hand].tolist(),
            finger_contact_force_N=forces,
            support_contact_force_N=float(magnitudes[support].sum()),
            actual_joint_q=dict(zip(self.native_names, self.array(self.rig.get_dof_positions())[0].tolist())),
            box_pose=box_pose.tolist(),
            box_lowest_z_m=float(box_corners[:, 2].min()),
            left_hand_pose=self.state.body_q.numpy()[left_hand].tolist(),
            left_box_contact_force_N=left_forces,
            left_box_normal_vectors_N=self.left_box_normal_vectors.tolist(),
            left_projected_finger_forces_N=[
                float(projected[self.native_names.index(name)])
                for name in ['left_finger_joint1', 'left_finger_joint2']
            ],
            box_upright_error_deg=float(
                np.degrees(np.arccos(np.clip(box_rotation.as_matrix()[2, 2], -1, 1)))
            ),
            bag_inside_box_fraction=float(inside.mean()),
            bag_box_local_min_m=box_nodes.min(0).tolist(),
            bag_box_local_max_m=box_nodes.max(0).tolist(),
            bag_carton_contact_force_N=float(magnitudes[box_contacts].sum()),
            box_linear_speed_m_s=float(np.linalg.norm(self._native_velocities[self._box_body, :3])),
            max_native_mimic_error_m=self.max_mimic_error,
            maximum_body_contact_reaction_N=float(np.linalg.norm(self.pending[:, :3], axis=1).max()),
        )

    def render(self, path):
        from visuals import Visuals

        if self.visuals is None:
            self.visuals = Visuals(self)
        return self.visuals.render(path)

    def save(self, out, rows, frames, no_close=False):
        super().save(out, rows, frames, no_close=no_close)
        path = out / "report.json"
        report = json.loads(path.read_text())
        report.update(
            backend="Isaac Sim 6.2 calibrated native articulation + Newton 1.7 VBD packet",
            scope="Two-way contact-wrench coupling; rigid poses read from native physics, never prescribed by the controller",
            native_robot=self.native_report,
            max_native_mimic_error_m=self.max_mimic_error,
            continuous_box_hold_metrics=self.box_hold_metrics,
            contact_feedback_steps=self.feedback_steps,
            physical_object_attachments=0,
            rigid_state_writes_after_reset="Solver-state transfer only; no controller pose writes",
            runtime_settings=dict(
                dt=self.dt,
                vbd_iterations=24,
                robot_engine="Isaac PhysX TGS",
                packet_engine="Newton 1.7 VBD",
                soft_contact_ke=self.model.soft_contact_ke,
                soft_contact_kd=self.model.soft_contact_kd,
                self_contact_rest_exclusion_m=0.003,
                rigid_soft_contacts="Surface particle contacts",
            ),
            packed_volume=json.loads((self.root / "newton170/config/packed_volume.json").read_text()),
        )
        path.write_text(json.dumps(report, indent=2) + "\n")
