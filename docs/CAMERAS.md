# Policy cameras

`Scene.render_views(path)` returns `cam_high`, `cam_left_wrist` and `cam_right_wrist` image paths. `PolicyRunner` supplies all three by default through `observation["rgb_paths"]`; `rgb_path` remains an alias for the external view when selected. Set `camera_views=("cam_high",)` for external-only policy input, or select either wrist. A missing requested output raises an error rather than silently masking the camera.

The external camera is 1024×768. The wrist cameras are 256×256, with a 14 mm focal length, 20.955 mm square aperture and 0.005–2 m clipping range. They use a pinhole model. Their housing has no physical geometry or collisions.

Each wrist view follows its measured native hand pose. Its hand-local eye is `[-0.065, 0, 0.025]` m, looking toward `[-0.016961, 0, 0.18]` m with `[0,1,0]` up. These are simulation design choices. They are not calibrated YAM camera extrinsics; no measured hardware camera calibration was supplied. Edit `CONFIG` in the chosen backend's `runtime/wrist_cameras.py` (rigid) or `newton170/runtime/wrist_cameras.py` (soft) to adapt the sensor to your hardware. Record the resulting code/configuration version with the experiment.

In the supplied held-carton reset, the left wrist can see a close carton flap while the right wrist sees the bag. Inspect framing and occlusion for your task; the external view supplies broader context.

All views are captured in one Replicator tick at zero elapsed physics time. The sibling `*_camera_metadata.json` records optics, hand poses, camera transforms and simulation time. That metadata is kept outside the policy callback's camera/proprioception observation. Fixed camera placement does not imply calibrated distortion, exposure, noise or rolling shutter.

Before each capture, the renderer publishes the measured physics poses to OmniHydra's display-only `_worldPosition`, `_worldOrientation` and `_worldScale` attributes, described in the [USDRT transform API](https://docs.omniverse.nvidia.com/kit/docs/usdrt.scenegraph/7.6.1/api/classusdrt_1_1_rt_xformable.html). It never writes USD or native rigid-body pose commands, attachments or forces, and every capture checks that native poses, velocities and simulation time are unchanged. Soft-bag skin vertices are drawn from the current simulated nodes.

The camera-freshness check (`tests/check_render_freshness.py`) compares images after measured arm motion against same-state render noise. It is a sensor integration check, not policy-success or hardware-transfer qualification.
