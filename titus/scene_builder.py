"""Compile an asset/count/randomization recipe into an immutable reset scene.

    Population happens before physics starts. The generated objects are free
    bodies/soft volumes throughout simulation. No runtime object controller is
    introduced. Generated folders reference this SDK's versioned resources;
    archive the recipe, resolved population and SDK version to reproduce them.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import shutil

from .paths import ROOT, population_backend_root
from .randomization import placed_bounds, rotated_bounds, sample_count, sample_placement
from .randomization import overlaps_xy


@dataclass(frozen=True)
class SceneInstance:
    root: Path
    recipe_path: Path
    backend: str
    manifest: dict


def _matrix(origin, yaw):
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    x, y, z = origin
    return f"(({c}, {s}, 0, 0), ({-s}, {c}, 0, 0), (0, 0, 1, 0), ({x}, {y}, {z}, 1))"


def resolve_recipe(recipe_path, *, seed=0, bags=None, boxes=None):
    recipe_path = Path(recipe_path).resolve(strict=True)
    recipe = json.loads(recipe_path.read_text())
    allowed = {"schema_version", "name", "bag_asset", "box_asset", "bags", "boxes",
               "bag_region_xy_m", "box_region_xy_m", "bag_yaw_deg", "box_yaw_deg",
               "max_bag_layers", "layer_gap_m", "placement_gap_m", "dish_radius_m",
               "goal", "fixed_bags", "initialize_box_hold", "camera", "lighting"}
    if set(recipe) - allowed:
        raise ValueError(f"Unknown recipe fields: {sorted(set(recipe)-allowed)}")
    if recipe.get("schema_version") != 1:
        raise ValueError("Expected recipe schema_version 1")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer from 0 to 2**32-1")
    for name in ("bag_region_xy_m", "box_region_xy_m", "bag_yaw_deg", "box_yaw_deg"):
        if name not in recipe:
            continue
        ranges = recipe[name] if "region" in name else [recipe[name]]
        if "region" in name and len(ranges) != 2:
            raise ValueError(f"{name} requires X and Y ranges")
        for interval in ranges:
            if len(interval) != 2 or not all(math.isfinite(float(x)) for x in interval) or interval[0] > interval[1]:
                raise ValueError(f"Invalid finite ordered range in {name}")
    assets = json.loads((ROOT / "recipes/assets.json").read_text())
    bag_asset = recipe.get("bag_asset", "sea_salt_bag")
    box_asset = recipe.get("box_asset", "lundberg_carton")
    if bag_asset != "sea_salt_bag" or box_asset != "lundberg_carton":
        raise ValueError("This release supplies Sea Salt bags and Lundberg cartons; register and adapt new asset representations explicitly")
    rng = random.Random(seed)
    bag_count = sample_count(recipe.get("bags", 1) if bags is None else bags, rng, 64)
    box_count = sample_count(recipe.get("boxes", 1) if boxes is None else boxes, rng, 8)
    box_bounds = assets[box_asset]["local_bounds_m"]
    primary_origin = [-.32, .14, .88]
    primary_bounds = placed_bounds(rotated_bounds(box_bounds, 90.), primary_origin)
    actors = [dict(id="box_001", kind="box", asset=box_asset, prim_path="/World/DashCarton",
                   origin_m=primary_origin, yaw_deg=90., bounds_m=primary_bounds,
                   role="prepositioned_left_grasp")]
    gap = float(recipe.get("placement_gap_m", .008))
    radius = float(recipe.get("dish_radius_m", .58))
    max_layers = recipe.get("max_bag_layers", 6)
    if isinstance(max_layers, bool) or not isinstance(max_layers, int) or not 1 <= max_layers <= 16 or not math.isfinite(gap) or gap < 0 or not 0 < radius <= .60:
        raise ValueError("Invalid placement gap, dish radius or layer count")
    occupied_boxes = [primary_bounds]
    for index in range(1, box_count):
        placement = sample_placement(
            box_bounds, recipe.get("box_region_xy_m", [[.12, .36], [-.34, .35]]),
            recipe.get("box_yaw_deg", [85, 95]), .870,
            occupied_boxes, rng, gap=gap, radius=radius,
        )
        if placement is None:
            raise ValueError("Spare cartons do not fit the requested bounds without initial overlap; widen bounds or reduce boxes")
        occupied_boxes.append(placement["bounds_m"])
        actors.append(dict(id=f"box_{index+1:03d}", kind="box", asset=box_asset,
                           prim_path=f"/World/Box_{index+1}", role="spare", **placement))
    bag_bounds = assets[bag_asset]["local_bounds_m"]
    layer_height = bag_bounds[1][2] - bag_bounds[0][2] + float(recipe.get("layer_gap_m", .012))
    if not math.isfinite(layer_height) or layer_height <= bag_bounds[1][2] - bag_bounds[0][2]:
        raise ValueError("layer_gap_m must be positive")
    layers = [[] for _ in range(max_layers)]
    fixed = recipe.get("fixed_bags")
    if fixed is not None and len(fixed) != bag_count:
        raise ValueError("fixed_bags must match the resolved bag count")
    for index in range(bag_count):
        placement = None
        for layer in range(max_layers):
            base_z = .863943 + layer * layer_height
            if fixed is not None:
                item = fixed[index]
                center = item["center_xy_m"]
                if len(center) != 2 or not all(math.isfinite(float(x)) for x in [*center, item["yaw_deg"]]):
                    raise ValueError("Fixed bag pose needs two finite XY coordinates and a finite yaw")
                local = rotated_bounds(bag_bounds, item["yaw_deg"])
                origin = [center[i] - (local[0][i]+local[1][i])/2 for i in (0, 1)] + [base_z-local[0][2]]
                placement = dict(center_xy_m=center, yaw_deg=item["yaw_deg"], base_z_m=base_z,
                                 origin_m=origin, bounds_m=placed_bounds(local, origin))
                bounds = placement["bounds_m"]
                if any(math.hypot(x, y) > radius for x in (bounds[0][0], bounds[1][0])
                       for y in (bounds[0][1], bounds[1][1])):
                    raise ValueError("Fixed bag pose exceeds dish_radius_m")
                if any(overlaps_xy(placement["bounds_m"], other, gap) for other in occupied_boxes + layers[layer]):
                    raise ValueError("Fixed bag poses overlap another reset object")
            else:
                placement = sample_placement(
                    bag_bounds, recipe.get("bag_region_xy_m", [[-.48, .07], [-.43, .38]]),
                    recipe.get("bag_yaw_deg", [-25, 25]), base_z,
                    occupied_boxes + layers[layer], rng, gap=gap, radius=radius,
                )
            if placement is not None:
                layers[layer].append(placement["bounds_m"])
                break
        if placement is None:
            raise ValueError("Bags do not fit the requested region/layers without initial overlap; increase max_bag_layers or reduce bags")
        actors.append(dict(id=f"bag_{index+1:03d}", kind="bag", asset=bag_asset,
                           prim_path="/World/Bag" if index == 0 else f"/World/Bag_{index+1}",
                           bag_index=index, layer=layer, **placement))
    goal = recipe.get("goal", "pack_into_primary")
    if goal not in ("pack_into_primary", "distribute"):
        raise ValueError("goal must be pack_into_primary or distribute")
    assignments = {f"bag_{i+1:03d}": f"box_{(i % box_count if goal == 'distribute' else 0)+1:03d}"
                   for i in range(bag_count)}
    def scalar(value):
        if isinstance(value, list):
            if len(value) != 2 or value[0] > value[1]:
                raise ValueError("Visual randomization expects a scalar or ordered [min,max]")
            value = rng.uniform(*value)
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Visual parameters must be finite")
        return value
    visuals = {}
    if "camera" in recipe:
        camera = recipe["camera"]
        if set(camera) - {"eye_m", "look_at_m", "focal_length_mm"}:
            raise ValueError("Unknown camera field")
        eye = [scalar(x) for x in camera.get("eye_m", [.65, -1.45, 1.7])]
        target = [scalar(x) for x in camera.get("look_at_m", [-.25, 0, .9])]
        focal = scalar(camera.get("focal_length_mm", 28.))
        if len(eye) != 3 or len(target) != 3 or focal <= 0 or sum((a-b)**2 for a,b in zip(eye,target)) < 1e-6:
            raise ValueError("Invalid camera position, target or focal length")
        visuals["camera"] = dict(eye_m=eye, look_at_m=target, focal_length_mm=focal)
    if "lighting" in recipe:
        lighting = recipe["lighting"]
        if set(lighting) - {"dome_intensity", "key_intensity"}:
            raise ValueError("Unknown lighting field")
        visuals["lighting"] = {key: scalar(value) for key,value in lighting.items()}
        if any(value < 0 for value in visuals["lighting"].values()):
            raise ValueError("Light intensities must be nonnegative")
    resolved = dict(schema_version=1, name=recipe.get("name", "packing"), seed=seed,
                    counts=dict(bags=bag_count, boxes=box_count), actors=actors,
                    goal_assignments=assignments, initialize_box_hold=recipe.get("initialize_box_hold", True),
                    visuals=visuals,
                    robot_randomization=False, physics_assistance="none",
                    scope="Generated reset configuration; does not imply policy success or physical qualification",
                    recipe=recipe, count_overrides={"bags": bags, "boxes": boxes})
    resolved["configuration_sha256"] = hashlib.sha256(json.dumps(resolved, sort_keys=True).encode()).hexdigest()
    return resolved


def materialize(recipe_path, out, *, backend="soft", seed=0, bags=None, boxes=None):
    manifest = resolve_recipe(recipe_path, seed=seed, bags=bags, boxes=boxes)
    template = population_backend_root(backend)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name in ("assets", "geometry", ".solver", "runtime"):
        source = template / name
        if source.exists():
            (out / name).symlink_to(source, target_is_directory=True)
    if (template / "newton170").exists():
        (out / "newton170").mkdir()
        for name in ("runtime", "config"):
            (out / "newton170" / name).symlink_to(template / "newton170" / name, target_is_directory=True)
    for name in ("scene", "config"):
        shutil.copytree(template / name, out / name)
    shutil.copy2(template / "titus_scene.py", out / "titus_scene.py")
    task = json.loads((out / "config/task.json").read_text())
    task["right_reset_joint_positions_rad"] = json.loads((ROOT / "recipes/robot_ready.json").read_text())[backend]
    (out / "config/task.json").write_text(json.dumps(task, indent=2) + "\n")
    bags_config = [dict(center_xy_m=a["center_xy_m"], yaw_deg=a["yaw_deg"], base_z_m=a["base_z_m"], origin_m=a["origin_m"])
                   for a in manifest["actors"] if a["kind"] == "bag"]
    (out / "config/multi_bag.json").write_text(json.dumps({"bags": bags_config}, indent=2) + "\n")
    base = "rigid_bag_box.usda" if backend == "rigid" else "single_bag_box.usda"
    lines = ["#usda 1.0", "(", '    defaultPrim = "World"', '    upAxis = "Z"',
             "    metersPerUnit = 1", "    timeCodesPerSecond = 600", "    endTimeCode = 3600000",
             f"    subLayers = [@{base}@]", ")", 'over "World"', "{"]
    for actor in manifest["actors"]:
        if actor["id"] == "box_001":
            continue
        name = actor["prim_path"].rsplit("/", 1)[1]
        source = "/World/Bag" if actor["kind"] == "bag" else "/World/DashCarton"
        if actor["id"] == "bag_001":
            lines.append(f'    over "{name}"')
        else:
            lines.append(f'    def Xform "{name}" (prepend references = @{base}@<{source}>)')
        lines += ["    {", f"        matrix4d xformOp:transform = {_matrix(actor['origin_m'], actor['yaw_deg'])}",
                  '        uniform token[] xformOpOrder = ["xformOp:transform"]', "    }"]
    lines += ["}", ""]
    (out / "scene/multi_bag_box.usda").write_text("\n".join(lines))
    manifest["backend"] = backend
    audit = json.loads((template / "scene/scene_lock.json").read_text())
    manifest["calibrated_robot_files_sha256"] = audit["calibrated_robot_files"]
    manifest["runtime_lock"] = json.loads((template / "runtime-lock.json").read_text())
    manifest["backend_manifest_sha256"] = hashlib.sha256((template / "manifest.json").read_bytes()).hexdigest()
    (out / "population.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copy2(recipe_path, out / "recipe.json")
    return SceneInstance(out, Path(recipe_path).resolve(), backend, manifest)
