"""Physics setup for simple_grasp.py; no playback, attachments or bag forces."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "contact_adapter170"))
from contact_adapter import SurfaceContactAdapter
from coupled_factory170 import make_coupled_solver, register_builder_attributes
from passive_mimic_compile import passive_mimic_compile
from rigid_soft_capacity_guard import RigidSoftCapacityGuard
from robot_kinematics import RobotKinematics


def signed_volume(points, tets):
    p = points[tets]
    return np.einsum("ij,ij->i", p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0])) / 6


class SingleBagWorld:
    dt = 1 / 600

    def __init__(self, root):
        if not getattr(self, "external_native", False):
            raise RuntimeError(
                "Use TitusScene: the supported robot backend is the calibrated native Isaac articulation"
            )
        import newton
        import warp as wp
        from fast_external_vbd import FastExternalVBD as SolverVBD
        from newton._src.solvers.solver import SolverBase
        from pxr import Usd

        self.newton = newton
        self.root = root
        self.task_config = json.loads((root / 'config/task.json').read_text())
        self.status, self.error = "running", None
        self.versions = {
            p: importlib.metadata.version(p)
            for p in ["newton", "warp-lang", "mujoco", "mujoco-warp", "isaacsim"]
        }
        expected = dict(zip(self.versions, ["1.7.0", "1.17.0", "3.15.0", "3.15.0", "6.2.0.0"]))
        required = (
            ["newton", "warp-lang", "isaacsim"] if getattr(self, "external_native", False) else list(expected)
        )
        if any(self.versions[key] != expected[key] for key in required):
            raise RuntimeError(f"Use the pinned Titus environment: {expected}; got {self.versions}")
        wp.init()
        wp.set_device("cuda:0")
        self.stage_path = root / "scene/single_bag_box.usda"
        stage = Usd.Stage.Open(str(self.stage_path))
        self.stage = stage
        self.kin = RobotKinematics(stage)
        # A single reset posture, not a replay trajectory. All subsequent motion is IK.
        left = [0.8, 1.61612117, 0.58079654, 1.02635992, 0.93404287, 1.55928135]
        right = [-0.04787518, 1.40745401, 0.62199587, -0.11615930, -0.21305409, 0.23174639]
        seed = {
            f"{side}_joint_{i}": value
            for side, values in [("left", left), ("right", right)]
            for i, value in enumerate(values, 1)
        }
        seed.update(
            left_finger_joint1=0.00022357,
            left_finger_joint2=-0.00022357,
            right_finger_joint1=-0.04,
            right_finger_joint2=0.04,
        )
        neutral = self.kin.forward(dict(seed, right_finger_joint1=0.0, right_finger_joint2=0.0))
        inverse_hand = np.linalg.inv(neutral["/World/Rig/right_hand"])
        landmarks = json.loads((root / "assets/robot/physics/gripper_contacts.json").read_text())[
            "pad_local_landmarks_m"
        ]
        self.tcp_local = np.mean(
            [
                (inverse_hand @ neutral["/World/Rig/" + finger] @ np.r_[landmarks[i], 1.0])[:3]
                for finger, i in [("right_left_finger", 2), ("right_right_finger", 3)]
            ],
            axis=0,
        )
        self.yaw = Rotation.from_euler("z", 18, degrees=True).as_matrix()
        self.R = self.yaw @ Rotation.from_euler("x", 180, degrees=True).as_matrix()
        self.pinch = np.r_[self.task_config['bag_pinch_xy_m'], 0.909]
        self.target = np.eye(4)
        self.target[:3, :3] = self.R
        self.target[:3, 3] = self.pinch - self.R @ self.tcp_local
        self.grasp, _ = self.kin.solve(self.target, seed)
        pre = self.target.copy()
        pre[:3, 3] -= 0.04 * self.R[:, 2]
        self.pregrasp, _ = self.kin.solve(pre, self.grasp)
        self.left_R = np.asarray(self.task_config['left_hand_rotation'])
        self.left_target = np.eye(4)
        self.left_target[:3, :3] = self.left_R
        self.left_target[:3, 3] = (
            np.asarray(self.task_config['box_reset_xyz_m'])
            + Rotation.from_quat(self.task_config['box_reset_quaternion_xyzw']).apply(
                self.task_config['left_grip_point_box_m']
            )
            - self.left_R @ self.tcp_local
        )
        self.pregrasp, _ = self.kin.solve(self.left_target, self.pregrasp, 'left')
        initial_left = self.task_config['initial_left_opening_m']
        self.pregrasp.update(left_finger_joint1=-initial_left, left_finger_joint2=initial_left)
        self.grasp.update({k: v for k, v in self.pregrasp.items() if k.startswith('left_')})

        self.mesh_path = root / "newton170/config/packed_tets.npz"
        mesh = np.load(self.mesh_path)
        self.canonical = mesh["points"].astype(float)
        self.tets = mesh["tetrahedra" if "tetrahedra" in mesh else "tets"].astype(np.int32)
        volumes = signed_volume(self.canonical, self.tets)
        if np.any(volumes <= 0):
            raise ValueError("Invalid source tetrahedra")
        points = self.canonical @ self.yaw.T
        origin = np.r_[self.pinch[:2], 0.863443 + 0.0005 - points[:, 2].min()]
        origin[:2] -= (points.min(0)[:2] + points.max(0)[:2]) / 2
        points += origin

        builder = newton.ModelBuilder()
        # Newton mesh/SDF pointers used by collision kernels retain builder-owned
        # resources. Keep the builder alive for the entire simulation.
        self.builder = builder
        SolverVBD.register_custom_attributes(builder)
        if not getattr(self, "external_native", False):
            register_builder_attributes(builder)
        builder.default_particle_radius = 0.0005
        builder.default_shape_cfg.ke = 1e5
        builder.default_shape_cfg.kd = 0.2
        builder.default_shape_cfg.mu = 0.5
        builder.default_shape_cfg.configure_sdf(force_sdf=True)
        self.imported = builder.add_usd(
            stage,
            load_visual_shapes=False,
            load_static_visual_shapes=False,
            enable_self_collisions=False,
            only_load_enabled_rigid_bodies=True,
        )
        for i, label in enumerate(builder.shape_label):
            if int(builder.shape_type[i]) in [int(newton.GeoType.MESH), int(newton.GeoType.CONVEX_MESH)]:
                builder.shape_force_sdf[i] = True
                builder.shape_sdf_max_resolution[i] = (
                    256 if ("finger" in label or "dish_collision" in label) else 128
                )
                builder.shape_sdf_target_voxel_size[i] = None
        # Preserve the workcell's intentional mount/column overlap filter.
        shapes = {label: i for i, label in enumerate(builder.shape_label)}
        pair = tuple(
            sorted(
                [
                    shapes["/World/Rig/rig_mount/mount_plate_collision"],
                    shapes["/World/RigSupport/column_collision"],
                ]
            )
        )
        if pair not in builder.shape_collision_filter_pairs:
            builder.add_shape_collision_filter_pair(*pair)
        E, nu = 50000.0, 0.35
        builder.add_soft_mesh(
            pos=wp.vec3(0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0),
            vertices=points.tolist(),
            indices=self.tets.ravel().tolist(),
            density=0.036 / volumes.sum(),
            k_mu=E / (2 * (1 + nu)),
            k_lambda=E * nu / ((1 + nu) * (1 - 2 * nu)),
            k_damp=0.002,
            particle_radius=0.0005,
            label="single_free_bag",
        )
        builder.color()
        self.model = builder.finalize(device="cuda:0")
        self.model.soft_contact_ke = 1e5
        self.model.soft_contact_kd = 0.2
        self.model.soft_contact_mu = 0.5
        self.mass = self.model.particle_mass.numpy().astype(float)
        if abs(self.mass.sum() - 0.036) > 1e-6 or self.model.attachment_particle_body_count:
            raise RuntimeError("Expected one free 36 g bag with no attachments")
        slot, frequency = SolverBase.CollisionSlot, SolverBase.CollisionFrequencyType
        options = dict(
            iterations=24,
            friction_epsilon=1e-4,
            particle_enable_self_contact=True,
            particle_self_contact_margin=0.001,
            particle_self_contact_gap=0.001,
            particle_vertex_contact_buffer_size=256,
            particle_edge_contact_buffer_size=256,
            particle_topological_contact_filter_threshold=2,
            particle_rest_shape_contact_exclusion_radius=0.003,
            rigid_body_particle_contact_buffer_size=16384,
            rigid_body_contact_buffer_size=512,
            rigid_soft_enable_dat=True,
            rigid_compliant_alm=True,
            rigid_avbd_alpha=0.0,
            collision_frequency_type={
                slot.RIGID: frequency.PRE_POST_INIT,
                slot.SOFT_SELF_CONTACT: frequency.PRE_POST_INIT,
            },
        )
        self.export_receipt = []
        if getattr(self, "external_native", False):
            flags = self.model.shape_flags.numpy()
            flags &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)
            self.model.shape_flags.assign(flags)
        self.pipeline = newton.CollisionPipeline(
            self.model,
            soft_contact_gap=0.001,
            soft_contact_max=262144,
            rigid_contact_max=0 if getattr(self, "external_native", False) else 65536,
            enable_rigid_soft_full_surface_contact=False,
            broad_phase="sap",
            verify_buffers=True,
        )
        if getattr(self, "external_native", False):
            options["rigid_soft_enable_dat"] = False
            self.solver = SolverVBD(
                self.model,
                collision_pipeline=self.pipeline,
                integrate_with_external_rigid_solver=True,
                **options,
            )
            self.soft = self.solver
            self.surface_adapter = SurfaceContactAdapter.for_solver(self.soft)
        else:
            with passive_mimic_compile(self.model, self.export_receipt, rigid_finger_linkage=True):
                self.solver = make_coupled_solver(
                    self.model,
                    options,
                    adapter_factory=SurfaceContactAdapter.for_solver,
                    proxy_iterations=1,
                    mujoco_options=dict(
                        use_mujoco_cpu=True, integrator="implicitfast", nconmax=4096, njmax=16384
                    ),
                )
            self.soft = self.solver._titus_vbd_solver
            self.soft_guard = RigidSoftCapacityGuard(self.soft.model).attach_solver(self.soft)
        self.outer_guard = RigidSoftCapacityGuard(self.model).attach_pipeline(self.pipeline)
        self.contacts = self.pipeline.contacts()
        self.control = self.model.control()
        self.state, self.next_state = self.model.state(), self.model.state()
        kinds = self.model.joint_type.numpy()
        joints = [
            j
            for j in range(self.model.joint_count)
            if self.model.joint_label[j].startswith("/World/Rig/Joints/")
            and int(kinds[j]) in [int(newton.JointType.REVOLUTE), int(newton.JointType.PRISMATIC)]
        ]
        self.names = [Path(self.model.joint_label[j]).name for j in joints]
        self.coords = self.model.joint_q_start.numpy()[joints].astype(int)
        self.target_q = self.control.joint_target_q.numpy().copy()
        q = self.state.joint_q.numpy()
        q[self.coords] = self._values(self.pregrasp, 0.05)
        self.state.joint_q.assign(q)
        self.state.joint_qd.zero_()
        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
        self.next_state.assign(self.state)
        self.solver.reset(self.state, flags=0)
        self.rest = self.state.particle_q.numpy().astype(float)
        self.rest_volume = signed_volume(self.rest, self.tets)
        self.target_posture, self.target_opening, self.time = self.pregrasp.copy(), 0.05, 0.0
        self.max_inverted = 0
        self.patch = np.argsort(np.linalg.norm(self.canonical[:, :2], axis=1))[:32]
        wp.synchronize()
        self._keepalive = locals().copy()

    def _values(self, posture, opening):
        values = dict(posture, right_finger_joint1=-opening, right_finger_joint2=opening)
        return np.array([values[name] for name in self.names])

    def plan_grasp(self):
        nodes = self.state.particle_q.numpy().astype(float)
        weights = self.mass / self.mass.sum()
        pinch = self.pinch + weights @ (nodes - self.rest)
        target = self.target.copy()
        target[:3, 3] = pinch - self.R @ self.tcp_local
        grasp, _ = self.kin.solve(target, self.grasp)
        pre = target.copy()
        pre[:3, 3] -= 0.04 * self.R[:, 2]
        aligned_pre, _ = self.kin.solve(pre, grasp)
        high = target[:3, 3] + [0, 0, 0.11]
        raised = target.copy()
        raised[:3, 3] = high
        lift, _ = self.kin.solve(raised, grasp)
        high[:2] += 0.16 * self.yaw[:2, 0]
        raised[:3, 3] = high
        over_table, _ = self.kin.solve(raised, lift)
        return aligned_pre, grasp, lift, over_table

    def step(self, posture, opening):
        # The only per-step write is to actuator targets. No particle/body poses.
        first = self.time == 0
        if first:
            print("first step: targets", flush=True)
        self.target_q[self.coords] = self._values(posture, opening)
        self.control.joint_target_q.assign(self.target_q)
        self.control.joint_target_qd.zero_()
        self.state.clear_forces()
        if first:
            print("first step: collision", flush=True)
        self.pipeline.collide(self.state, self.contacts)
        if first:
            print("first step: solver", flush=True)
        self.solver.step(self.state, self.next_state, self.control, self.contacts, self.dt)
        import mujoco

        rigid = self.solver._titus_rigid_solver
        for warning in [
            mujoco.mjtWarning.mjWARN_BADQPOS,
            mujoco.mjtWarning.mjWARN_BADQVEL,
            mujoco.mjtWarning.mjWARN_BADQACC,
        ]:
            if rigid.mj_data.warning[int(warning)].number:
                raise RuntimeError(
                    f"Native robot integration warning: {warning}; refusing MuJoCo's automatic state reset"
                )
        if first:
            print("first step: readback", flush=True)
        self.state, self.next_state = self.next_state, self.state
        nodes = self.state.particle_q.numpy()
        if not np.isfinite(nodes).all() or not np.isfinite(self.state.body_q.numpy()).all():
            raise RuntimeError("Nonfinite physics state")
        self.max_inverted = max(
            self.max_inverted, int(np.count_nonzero(signed_volume(nodes, self.tets) * self.rest_volume <= 0))
        )
        self.target_posture, self.target_opening = posture.copy(), opening
        self.time += self.dt
        if self.max_inverted:
            raise RuntimeError(
                f"Bag mesh inverted ({self.max_inverted} tetrahedra); stopping this failed attempt"
            )

    def measure(self):
        if self.time < 0.005:
            print("first measurement", flush=True)
        nodes = self.state.particle_q.numpy().astype(float)
        count = int(self.soft.contacts.soft_contact_count.numpy()[0])
        local_shapes = self.soft.contacts.soft_contact_shape.numpy()[:count]
        shape_map = self.solver._titus_index_maps["vbd"]["shape_local_to_global"]
        labels = [self.model.shape_label[int(shape_map[int(i)])] for i in local_shapes]
        start = self.soft.contacts.rigid_contact_max
        forces = self.solver._titus_vbd_observables.contact_f.numpy()[start : start + count, :3]
        norms = np.linalg.norm(forces, axis=1)
        finger_forces = [
            float(norms[np.array([f"/World/Rig/{finger}/" in label for label in labels], dtype=bool)].sum())
            for finger in ["right_left_finger", "right_right_finger"]
        ]
        return dict(
            t_s=self.time,
            centroid_m=((self.mass / self.mass.sum()) @ nodes).tolist(),
            lowest_z_m=float(nodes[:, 2].min()),
            max_inverted_tets=self.max_inverted,
            finger_contact_force_N=finger_forces,
            actual_joint_q=dict(zip(self.names, self.state.joint_q.numpy()[self.coords].tolist())),
        )

    def frame(self):
        return self.time, self.state.particle_q.numpy(), self.state.body_q.numpy()

    def save(self, out, rows, frames, no_close=False):
        (out / "telemetry.json").write_text(json.dumps(rows, indent=2) + "\n")
        if frames:
            np.savez_compressed(
                out / "states.npz",
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
            no_close=no_close,
            status=self.status,
            error=self.error,
            accepted_grasp=False,
            object_attachments=0,
            object_pose_writes_after_reset=0,
            object_forces=0,
            max_inverted_tets=self.max_inverted,
            mass_kg=float(self.mass.sum()),
            tet_mesh_sha256=hashlib.sha256(self.mesh_path.read_bytes()).hexdigest(),
            passive_follower_export_fix=self.export_receipt,
            robot_calibration=json.loads((self.root / "scene/robot_calibration.json").read_text()),
            calibrated_articulation_sha256=hashlib.sha256(
                (self.root / "assets/robot/physics/articulation.usdc").read_bytes()
            ).hexdigest(),
            tcp_from_delivered_pad_landmarks_m=self.tcp_local.tolist(),
            imported_drive_stiffness=self.model.joint_target_ke.numpy().tolist(),
            imported_drive_damping=self.model.joint_target_kd.numpy().tolist(),
            imported_motor_effort_limits=self.model.joint_effort_limit.numpy().tolist(),
            runtime_settings=dict(
                dt=self.dt,
                vbd_iterations=24,
                rigid_backend="MuJoCo native CPU",
                integrator="implicitfast",
                soft_contact_ke=self.model.soft_contact_ke,
                soft_contact_kd=self.model.soft_contact_kd,
                reference="newton/examples/softbody/example_softbody_franka.py",
            ),
            backend="Newton SolverCoupledProxy: MuJoCo articulated rigid robot + VBD soft bag",
            scope="Standalone Newton in Isaac Sim 6.2 Python; no Kit Stage integration asserted",
            qualification="Recorded attempt; task success must be assessed from telemetry",
        )
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
