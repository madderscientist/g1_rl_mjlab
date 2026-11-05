# Differentiable position-WM adapters

Adapters consume the [v3 contract](../README.md): actual target29, one 20ms step,
and next-state/foot-force predictions.

## State and geometry

| Full-state slice | Meaning |
|---|---|
| `0:3` | pelvis link-origin position, world |
| `3:7` | wxyz quaternion, pelvis to world |
| `7:36` | 29 true joint angles |
| `36:39` | pelvis link-origin linear velocity, world |
| `39:42` | pelvis angular velocity, body |
| `42:71` | 29 true joint velocities |

`StaticKinematicTemplate.from_mujoco(actual_model)` captures compiled 44-body
geometry, joint axes/references, local COMs and sole sites. Save its JSON or
`to_dict()` in provenance; `load`/`from_dict` then work without MuJoCo.
`from_xml()` uses nominal repository geometry. `TorchKinematics(template)(state)`
returns world body/site poses and velocities. Link origins are not COMs;
morphology/joint-axis changes require a new template.

## Control, forces and task signals

- `PositionTaskRollout` advances one 20ms step. `process_position_target` applies
  execution noise, scale/offset, clipping and encoder-bias subtraction.
  Optional `SubstepTorqueController`/`ControllerContext` convert external GC to
  a corrected target **once at the control boundary**.
  `TorchGravityTorque` is also available as a standalone numerical utility.
- `controller_from_metadata` reconstructs nominal gains/geometry/effort limits
  and GC divisors, with zero encoder bias. GC adds the context's torso COM offset
  to the template nominal COM.
- `prediction.foot_force_root` uses the next root frame; `.foot_force` is world
  force. Use world forces for contacts/vertical costs: inverted root Fz can be
  negative. Labels are final-substep forces, not control-period averages.
- `ForceContact` exposes hard events, 20ms timers and separate soft surrogates.
  Threshold/hysteresis approximates native `sensor.found > 0`; equivalence of
  first-touch/air-time events is not guaranteed.
- Motion, lower-body velocity and footstep adapters emit named observations,
  unweighted rewards/costs and hard events. The caller owns weights, commands,
  termination and resets. `soft_` signals do not replace hard timers/events.
  Contact-yaw cost uses contact-gated world-Z sole angular velocity.
  `MotionConfig.anchor_position_coarse_axes` selects `"xyz"` (default) or `"xy"`.
- Foot forces cannot represent self-collision, contact torque or CoP. Adapters
  do not claim to reproduce every native task-manager term.

## Validation

Tests cover native CPU MuJoCo FK/GC parity, action processing, contact-yaw and
two-step action gradients for all three task families. For compatible v3 weights:

```bash
python -m g1_lower_rl.world_model.adapters.checkpoint_smoke --help
```

`checkpoint_smoke.validate_checkpoint` checks checkpoint-to-task gradients without
optimizer updates. `--steps` counts 20ms control steps. Loading requires checkpoint
format2, physical contract3 and the fixed29 position interface.
Native closed-loop performance requires separate validation.
