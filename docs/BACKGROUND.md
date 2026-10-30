# Optional factory background

The factory Gaussian splat is bundled as an optional visual asset. You can add it to an Isaac scene; the SDK's default recipes and reference tasks do not load it.

| File | Use |
| --- | --- |
| `assets/background/background.usda` | Reference this layer into your scene |
| `assets/background/factory.usdc` | Native Gaussian asset, with its estimated workcell alignment already authored |
| `assets/background/source/3DGS.sog` | Unmodified original capture for SOG-compatible tools |
| `assets/background/source/3DGS.voxel.zip` | Original spatial data supplied with the capture |
| `assets/background/source/cameras.json` | Original capture camera records |
| `assets/background/provenance.json` | Source identity, transforms and hashes |

## Add it in Isaac Sim

Start Isaac Sim 6.2 with particle-field support and geometry streaming enabled. These arguments apply to the current session:

```text
--enable omni.usd.schema.usd_particle_field
--enable omni.ujitso.client
--/UJITSO/enabled=true
--/UJITSO/geometry=true
```

Open your scene and drag `assets/background/background.usda` from the Content browser into the stage as a reference. Keep the reference transform at identity initially: the native asset already contains the estimated alignment to the Titus workcell.

Alternatively, run this in Isaac's Script Editor or your own simulator process after opening a scene. Replace the package path with your code checkout after fetching assets:

```python
from pathlib import Path
import omni.usd

package = Path("/path/to/Titus_Packing_SDK")
stage = omni.usd.get_context().get_stage()
background = stage.DefinePrim("/World/FactoryBackground", "Xform")
background.GetReferences().AddReference(
    str(package / "assets/background/background.usda")
)
```

The ordinary-Python `Scene` client runs Isaac in a separate process; the snippet belongs in that simulator process or the Isaac UI. This optional asset is not a recipe switch in the supplied SDK. Save your customized scene in a separate file.

The renderer can take several frames to load the splat. See NVIDIA's [particle-field import instructions](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/particle-fields.html#importing-3d-gaussian-splats-into-omniverse) and [geometry streaming settings](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_common.html#how-to-enable-geometry-streaming).

## Appearance and scope

The native asset retains 995,579 Gaussians from the original factory background. Its captured workcell region was removed and the nearby conveyor was lowered for the visual layout. The original source capture remains unmodified in `source/`.

Alignment is estimated, not a measured environment calibration. Capture gaps and fragments may be visible from new camera viewpoints. Check the placement and camera images after adding it. The source camera records use capture coordinates; the native USD already applies the workcell transform, so do not apply it a second time.

The splat has no collision, rigid-body or support behavior. It does not grasp, carry or stabilize objects. Robot calibration and the physical task assets remain unchanged. Enabling it changes RGB observations; the default reference results do not qualify a policy with this optional background enabled.
