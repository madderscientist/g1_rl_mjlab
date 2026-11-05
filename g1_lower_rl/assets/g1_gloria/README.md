# G1 With Dual Gloria-M

Calibrated MuJoCo asset for the G1 with two rigidly closed Gloria-M grippers:

- 29 scalar hinges and 29 position actuators.
- 36 generalized positions, 35 generalized velocities and 44 named robot bodies.
- Joint order: lower-body15, left-arm7, right-arm7, matching motor IDs 0..28.
- Fixed gripper geometry includes masses, inertias, cameras and collision shapes.

## Geometry and calibration

`calibrated.urdf` defines joint anchors/axes, calibrated arm-link masses/inertias
and KWR57B + Gloria-M mount transforms. The MJCF uses these calibrated parameters;
`logo_link` and `head_link` are included in `torso_link`.

The grippers are fixed at `eccentric = 0`. Their eccentric bodies, sliders,
connecting rods and fingers form rigid assemblies with `*_gripper_base`;
grippers have no joint coordinates or policy actions.
Gripper, camera and finger collision shapes are boxes bounding their mesh AABBs.

`HOME_KEYFRAME` uses `shoulder_roll = +/-0.25` for thigh clearance.
Both this pose and the zero configuration are self-collision free.
MJCF intrinsic `xyz` and URDF extrinsic `rpy` rotations use their respective
conventions; the wrist-to-KWR57B mount is `quat="0.5 0.5 0.5 0.5"`.

## Meshes

STLs in `xmls/assets/` use the calibrated machine geometry:

| source | used for |
| --- | --- |
| `g1_description/meshes/*.STL` | body, legs, waist, arms |
| `g1_description/meshes/pelvis_ver0529.STL` | single-piece pelvis |
| `g1_description/meshes/*_wrist_*_link_5010.STL` | 5010 wrist shells |
| `Gloria-M/meshes/*.stl` | gripper base, eccentric, sliders, connecting rods |
| `camera_mount/mesh/cam-{main,ring}.STL` | gripper camera housing, scale `0.001 0.001 -0.001` |

`left_kwr57b_link` / `right_kwr57b_link` use cylinders with radius 0.0285 m
and length 0.053 m.

## Motion data and world model

Motion data reads the 29 canonical robot joints. Gripper positions and velocities
are excluded from the motion interface; clip selection is independent of gripper
opening. Simulation, FK, joint limits and default-pose buffers use this asset.

The [position WM v3](../../world_model/README.md) uses full state71, neural input145,
raw output76 and physical context48, with one held-target transition every20ms.
