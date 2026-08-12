# G1 With Dual Gloria-M

31-DoF MuJoCo model: the original 29 G1 joints plus `left_eccentric_joint` and
`right_eccentric_joint` for the two Gloria-M grippers.

`calibrated.urdf` is the unmodified calibrated source copied from
`Unitree_G1_Workspace/src/arm_gravity_compensation/config/calibrated.urdf`.
The MJCF reproduces it for every movable link: joint anchors, joint axes,
calibrated arm-link masses/inertias, and the KWR57B + Gloria-M mount transforms
all match the URDF to <1e-5 m / <1e-5 rad at zero configuration. Whole-robot mass
and COM match to 1 g / 0.00 mm (`logo_link` and `head_link` are folded into
`torso_link`, as in the stock mjlab G1 model).

## Meshes

All STLs in `xmls/assets/` come from
`Unitree_G1_Workspace/src/unitree_g1_description/model/`, so the visuals match
the real machine:

| source | used for |
| --- | --- |
| `g1_description/meshes/*.STL` | body, legs, waist, arms |
| `g1_description/meshes/pelvis_ver0529.STL` | pelvis (single piece, replaces the stock `pelvis.STL` + `pelvis_contour_link.STL`) |
| `g1_description/meshes/*_wrist_*_link_5010.STL` | 5010 wrist shells |
| `Gloria-M/meshes/*.stl` | gripper base, eccentric, sliders, connecting rods |
| `camera_mount/mesh/cam-{main,ring}.STL` | gripper camera housing (`scale="0.001 0.001 -0.001"`, as in the URDF) |

`left_kwr57b_link` / `right_kwr57b_link` have no mesh; the URDF draws them as a
cylinder (r = 0.0285, l = 0.053) and the MJCF does the same.

## Deliberate simplifications

- The gripper's internal spline/mimic chain (slider and connecting-rod links) is
  rigidly welded to `*_gripper_base` at its zero-configuration pose, so the
  fingers are not extra policy degrees of freedom. Their masses and inertias are
  kept, so the arm inertia is still exact.
- `*_gripper_collision`, `*_camera_collision` and `*_finger_collision` are boxes
  that tightly bound the corresponding mesh AABBs; the stock G1 rubber-hand
  capsules were removed.

## Notes

- MJCF `euler` uses MuJoCo's *intrinsic* `xyz` sequence, while URDF `rpy` is
  *extrinsic*. The wrist-to-KWR57B mount is therefore written as
  `quat="0.5 0.5 0.5 0.5"` rather than copying the URDF rpy values verbatim.
- The grippers reach ~130 mm further than the old rubber hands, so this asset
  ships its own `HOME_KEYFRAME` with `shoulder_roll = +/-0.25` (the stock G1
  value of +/-0.18 puts the grippers inside the thighs). Both `HOME_KEYFRAME` and
  the zero configuration are self-collision free.

### Joint-limit penalty

`left_eccentric_joint` / `right_eccentric_joint` have range `[0, 2.7638]` and are
initialised at `0`, which is the mechanically closed position but also the lower
*hard* limit. `G1_WITH_DUAL_GLORIA_M_ARTICULATION` uses
`soft_joint_pos_limit_factor=0.9`, which maps that range to a soft range of
roughly `[0.138, 2.626]` — so the initial state sits outside the soft limit and a
`joint_pos_limits` reward term would fire a constant penalty from step 0.

When adding a task for this robot, exclude the two gripper joints from any
joint-limit penalty, e.g.:

```python
joint_pos_limits = RewardTermCfg(
  func=mdp.joint_pos_limits,
  weight=-1.0,
  params={
    "asset_cfg": SceneEntityCfg("robot", joint_names=[r"^(?!.*_eccentric_joint$).*"])
  },
)
```

The same applies to any joint-deviation / action-rate term that should only cover
the 29 locomotion joints.

