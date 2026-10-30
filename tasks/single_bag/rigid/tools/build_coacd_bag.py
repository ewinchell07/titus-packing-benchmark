"""Build native rigid bag collisions from the same closed packed-content envelope."""

from pathlib import Path
import hashlib, importlib.metadata, json, time, itertools
import numpy as np
import trimesh, coacd
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf, Vt

ROOT = Path(__file__).resolve().parents[1]
z = np.load(ROOT / 'geometry/packed_tets.npz')
points = z['points'].astype(float)
tets = z['tets']
faces = np.concatenate([tets[:, ids] for ids in [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]])
_, idx, count = np.unique(np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True)
mesh = trimesh.Trimesh(points, faces[idx[count == 1]], process=True)
mesh.fix_normals()
assert mesh.is_watertight and mesh.volume > 0
params = dict(
    threshold=0.002,
    real_metric=True,
    max_convex_hull=32,
    preprocess_mode='off',
    resolution=2000,
    mcts_nodes=20,
    mcts_iterations=100,
    mcts_max_depth=3,
    pca=False,
    merge=True,
    decimate=True,
    max_ch_vertex=64,
    seed=0,
)
start = time.monotonic()
parts = coacd.run_coacd(
    coacd.Mesh(np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int32)), **params
)
elapsed = time.monotonic() - start
stage = Usd.Stage.CreateNew(str(ROOT / 'scene/rigid_bag_box.usda'))
stage.GetRootLayer().subLayerPaths = ['single_bag_box.usda']
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.SetStageUpAxis(stage, 'Z')
stage.SetTimeCodesPerSecond(600.0)
stage.SetStartTimeCode(0.0)
stage.SetEndTimeCode(3600000.0)
stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
bag = stage.OverridePrim('/World/Bag')
UsdPhysics.RigidBodyAPI.Apply(bag).CreateKinematicEnabledAttr(False)
mass = UsdPhysics.MassAPI.Apply(bag)
mass.CreateMassAttr(0.036)
mesh.density = 0.036 / mesh.volume
properties = mesh.mass_properties
I = np.asarray(properties['inertia'])
eigenvalues, R = np.linalg.eigh(I)
if np.linalg.det(R) < 0:
    R[:, 0] *= -1
q = Rotation.from_matrix(R).as_quat()
mass.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, properties['center_mass'])))
mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, eigenvalues)))
mass.CreatePrincipalAxesAttr(Gf.Quatf(float(q[3]), Gf.Vec3f(*map(float, q[:3]))))
# Disable any historical source simulation schemas. Only our hulls collide.
for prim in Usd.PrimRange(bag):
    if prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
    if prim != bag and prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
material = UsdShade.Material.Define(stage, '/World/Bag/CollisionMaterial')
physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
physics.CreateStaticFrictionAttr(0.5)
physics.CreateDynamicFrictionAttr(0.5)
physics.CreateRestitutionAttr(0.0)
rows = []
arrays = {}
for i, (v, f) in enumerate(parts):
    hull = trimesh.Trimesh(v, f, process=True)
    hull.fix_normals()
    assert hull.is_watertight and hull.is_convex and len(hull.vertices) <= 64
    p = UsdGeom.Mesh.Define(stage, f'/World/Bag/CoACD/hull_{i:03d}')
    p.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(hull.vertices.astype(np.float32)))
    p.CreateFaceVertexIndicesAttr(Vt.IntArray(hull.faces.astype(np.int32).ravel().tolist()))
    p.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(hull.faces)))
    UsdPhysics.CollisionAPI.Apply(p.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(p.GetPrim()).CreateApproximationAttr('convexHull')
    p.GetPrim().AddAppliedSchema('PhysxCollisionAPI')
    p.GetPrim().CreateAttribute('physxCollision:contactOffset', Sdf.ValueTypeNames.Float).Set(0.002)
    p.GetPrim().CreateAttribute('physxCollision:restOffset', Sdf.ValueTypeNames.Float).Set(0.0)
    UsdShade.MaterialBindingAPI.Apply(p.GetPrim()).Bind(material, materialPurpose='physics')
    p.CreateVisibilityAttr('invisible')
    rows.append(
        dict(
            path=str(p.GetPath()),
            vertices=len(hull.vertices),
            faces=len(hull.faces),
            volume_m3=float(hull.volume),
        )
    )
    arrays[f'vertices_{i}'] = hull.vertices
    arrays[f'faces_{i}'] = hull.faces
stage.GetRootLayer().Save()
np.savez_compressed(ROOT / 'geometry/coacd_hulls.npz', **arrays)
receipt = dict(
    representation='One freely dynamic native Isaac rigid body with CoACD convex collision pieces',
    coacd_version=importlib.metadata.version('coacd'),
    parameters=params,
    seconds=elapsed,
    source='Same watertight packed-content envelope used by the soft variant',
    source_sha256=hashlib.sha256((ROOT / 'geometry/packed_tets.npz').read_bytes()).hexdigest(),
    source_volume_m3=float(mesh.volume),
    mass_kg=0.036,
    friction=0.5,
    restitution=0.0,
    contact_offset_m=0.002,
    center_of_mass_m=properties['center_mass'].tolist(),
    inertia_kg_m2=I.tolist(),
    hull_count=len(rows),
    hulls=rows,
    printed_surface_changed=False,
    calibrated_robot_changed=False,
    physics_qualified=False,
)
(ROOT / 'geometry/coacd_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt), flush=True)
