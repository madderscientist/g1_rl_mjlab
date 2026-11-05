"""CPU-native kinematic parity and differentiable task/contact contracts."""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import mujoco
import numpy as np
import pytest
import torch

from g1_lower_rl.footstep_phase import FootstepPhaseCfg
from g1_lower_rl.world_model.adapters import (
  ContactConfig,
  ContactMemory,
  ControllerContext,
  FootstepConfig,
  FootstepGoal,
  FootstepMemory,
  FootstepTrackingAdapter,
  ForceContact,
  LowerBodyVelocityAdapter,
  MotionGoal,
  MotionTrackingAdapter,
  PositionTaskRollout,
  StaticKinematicTemplate,
  SubstepTorqueController,
  TorchGravityTorque,
  TorchKinematics,
  VelocityConfig,
  VelocityGoal,
  controller_from_metadata,
  process_position_target,
)
from g1_lower_rl.world_model.adapters.checkpoint_smoke import smoke_checkpoint
from g1_lower_rl.world_model.adapters.controller import ARM_JOINT_INDICES
from g1_lower_rl.world_model.adapters.math import multiply, rotate
from g1_lower_rl.world_model.model import (
  BODY_NAMES,
  JOINT_NAMES,
  PositionWorldModel,
  PositionWorldModelCfg,
  apply_state_delta,
  pack_inputs,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def model():
  return mujoco.MjModel.from_xml_path(str(ROOT / "g1_lower_rl/assets/g1_gloria/xmls/scene_g1_with_dual_gloria_m.xml"))


@pytest.fixture(scope="module")
def template(model):
  return StaticKinematicTemplate.from_mujoco(model)


def state(batch=2, dtype=torch.float64):
  result = torch.zeros(batch, 71, dtype=dtype)
  result[..., 2] = 0.79
  result[..., 3] = 1
  return result


@pytest.mark.parametrize("invalid", ["nonfinite", "zero_quaternion"])
def test_kinematics_eager_validation_still_rejects_invalid_states(template, invalid):
  value = state()
  if invalid == "nonfinite":
    value[0, 0] = torch.nan
    message = "state must be finite"
  else:
    value[0, 3:7] = 0
    message = "Root quaternion must be nonzero"
  with pytest.raises(ValueError, match=message):
    TorchKinematics(template)(value)


def test_motion_indices_are_nonpersistent_canonical_buffers(template):
  adapter = MotionTrackingAdapter(TorchKinematics(template))
  assert adapter.body_index_tensor.tolist() == list(adapter.body_indices)
  assert adapter.foot_index_tensor.tolist() == list(adapter.feet)
  assert adapter.policy_index_tensor.tolist() == list(range(29))
  assert not any("index_tensor" in key for key in adapter.state_dict())
  value = state()
  value[:, 7:36] = torch.randn(2, 29, dtype=value.dtype)
  torch.testing.assert_close(adapter.reference_tokens(value)[..., -29:], value[:, 7:36], rtol=0, atol=0)


def force(values):
  f = torch.zeros(*values.shape, 3, dtype=values.dtype)
  f[..., 2] = values
  return f


def contact_update(values, previous=None, dt=0.02):
  if previous is None:
    previous = ContactMemory.zeros(values)
  return ForceContact()(force(values), previous, dt)


def goal_for(s, fk, *, phase=0.0, frequency=1.0, ids=0, dx=0.0):
  k = fk(s)
  targets = torch.cat((k.foot_position[..., :2] + dx, torch.zeros_like(k.foot_position[..., :1])), dim=-1)
  shape = s.shape[:-1]
  return FootstepGoal(
    targets,
    torch.full((*shape, 2), ids, dtype=torch.long),
    s.new_full(shape, phase),
    s.new_full(shape, frequency),
    s.new_zeros((*shape, 2)),
  )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_level_batched_kinematics_matches_values_gradients_and_checkpoint_keys(template, dtype):
  torch.manual_seed(19)
  inputs = torch.randn(2, 3, 71, dtype=dtype, requires_grad=True)
  fk = TorchKinematics(template).to(dtype=dtype)
  expected = fk(inputs)
  expected_gradient, = torch.autograd.grad(sum(value.square().mean() for value in expected), inputs)
  keys = set(fk.state_dict())
  fk.level_batching = True
  actual = fk(inputs)
  actual_gradient, = torch.autograd.grad(sum(value.square().mean() for value in actual), inputs)
  for value, reference in zip(actual, expected, strict=True):
    torch.testing.assert_close(value, reference, rtol=2e-5, atol=2e-5)
  torch.testing.assert_close(actual_gradient, expected_gradient, rtol=3e-5, atol=3e-5)
  assert set(fk.state_dict()) == keys


@pytest.mark.parametrize("level_batching", [False, True])
def test_native_fk_random_all_bodies_feet_and_velocities(model, template, level_batching):
  """44 bodies and arbitrary root orientation, with both grippers fixed closed."""
  rng = np.random.default_rng(9701)
  states = state(12)
  states[:, :3] = torch.from_numpy(rng.normal(size=(12, 3)))
  quats = rng.normal(size=(12, 4))
  quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
  states[:, 3:7] = torch.from_numpy(quats)
  states[:, 7:36] = torch.from_numpy(rng.uniform(-0.7, 0.7, (12, 29)))
  states[:, 36:] = torch.from_numpy(rng.normal(size=(12, 35)))
  kinematics = TorchKinematics(template)
  kinematics.level_batching = level_batching
  result = kinematics(states)
  data = mujoco.MjData(model)
  joints = [model.joint(name).id for name in JOINT_NAMES]
  bodies = [model.body(name).id for name in BODY_NAMES]
  sites = [model.site(name).id for name in ("left_foot", "right_foot")]
  maximum_error = {}

  def record(name, predicted, native):
    maximum_error[name] = max(maximum_error.get(name, 0.0), float(np.max(np.abs(predicted - native))))

  for i, s in enumerate(states.numpy()):
    data.qpos[:7], data.qvel[:6] = s[:7], s[36:42]
    for j, native in enumerate(joints):
      data.qpos[model.jnt_qposadr[native]] = s[7 + j]
      data.qvel[model.jnt_dofadr[native]] = s[42 + j]
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(result.body_position[i].numpy(), data.xpos[bodies], atol=2e-14)
    record("body_position_m", result.body_position[i].numpy(), data.xpos[bodies])
    orientation_dot = np.abs((result.body_quaternion[i].numpy() * data.xquat[bodies]).sum(-1))
    np.testing.assert_allclose(orientation_dot, 1, atol=2e-14)
    sign = np.sign((result.body_quaternion[i].numpy() * data.xquat[bodies]).sum(-1, keepdims=True))
    record("body_quaternion_component_sign_aligned", result.body_quaternion[i].numpy(), sign * data.xquat[bodies])
    np.testing.assert_allclose(result.foot_position[i].numpy(), data.site_xpos[sites], atol=2e-14)
    record("foot_position_m", result.foot_position[i].numpy(), data.site_xpos[sites])
    for b, native in enumerate(bodies):
      jac_p, jac_r = np.zeros((3, model.nv)), np.zeros((3, model.nv))
      mujoco.mj_jacBody(model, data, jac_p, jac_r, native)
      np.testing.assert_allclose(result.body_linear_velocity[i, b].numpy(), jac_p @ data.qvel, atol=2e-13)
      np.testing.assert_allclose(result.body_angular_velocity[i, b].numpy(), jac_r @ data.qvel, atol=2e-13)
      record("body_linear_velocity_m_s", result.body_linear_velocity[i, b].numpy(), jac_p @ data.qvel)
      record("body_angular_velocity_rad_s", result.body_angular_velocity[i, b].numpy(), jac_r @ data.qvel)
    for foot, native in enumerate(sites):
      jac_p, jac_r = np.zeros((3, model.nv)), np.zeros((3, model.nv))
      mujoco.mj_jacSite(model, data, jac_p, jac_r, native)
      np.testing.assert_allclose(result.foot_linear_velocity[i, foot].numpy(), jac_p @ data.qvel, atol=2e-13)
      np.testing.assert_allclose(result.foot_angular_velocity[i, foot].numpy(), jac_r @ data.qvel, atol=2e-13)
      record("foot_linear_velocity_m_s", result.foot_linear_velocity[i, foot].numpy(), jac_p @ data.qvel)
      record("foot_angular_velocity_rad_s", result.foot_angular_velocity[i, foot].numpy(), jac_r @ data.qvel)
      matrix = np.empty(9)
      mujoco.mju_quat2Mat(matrix, result.foot_quaternion[i, foot].numpy())
      np.testing.assert_allclose(matrix, data.site_xmat[native], atol=2e-14)
      record("foot_rotation_matrix_component", matrix, data.site_xmat[native])
  print("NATIVE_FK_MAXERR=" + json.dumps(maximum_error, sort_keys=True))


def test_template_serialization_and_no_mujoco_calls(template):
  restored = StaticKinematicTemplate.from_dict(json.loads(json.dumps(template.to_dict())))
  assert restored == template
  with patch("mujoco.mj_forward", side_effect=AssertionError("Simulator called during forward")):
    fk = TorchKinematics(restored)
    torch.testing.assert_close(fk(state()).foot_position, TorchKinematics(template)(state()).foot_position)
  with pytest.raises(ValueError, match="version"):
    StaticKinematicTemplate.from_dict({**template.to_dict(), "version": 2})
  archive = io.BytesIO()
  torch.save(fk.state_dict(), archive)
  archive.seek(0)
  TorchKinematics(restored).load_state_dict(torch.load(archive, weights_only=True))


def test_namespaced_teacher_template():
  spec = mujoco.MjSpec.from_file(str(ROOT / "g1_lower_rl/assets/g1_gloria/xmls/scene_g1_with_dual_gloria_m.xml"))
  for element in (*spec.bodies, *spec.joints, *spec.sites):
    if element.name and element.name != "world":
      element.name = "robot/" + element.name
  for exclude in spec.excludes:
    exclude.bodyname1 = "robot/" + exclude.bodyname1
    exclude.bodyname2 = "robot/" + exclude.bodyname2
  for sensor in spec.sensors:
    sensor.objname = "robot/" + sensor.objname
  model = spec.compile()
  t = StaticKinematicTemplate.from_mujoco(model)
  assert t.source_prefix == "robot/" and len(t.source_body_ids) == 44
  assert t.source_joint_ids[0] == model.joint("robot/left_hip_pitch_joint").id
  k = TorchKinematics(t)(state())
  assert k.body_position.shape == (2, 44, 3)


def test_template_hinge_anchor_and_reference_offsets(model):
  """Exercise nonzero joint anchors/ref, not merely the zero-anchor robot XML."""
  edited = mujoco.MjModel.from_xml_path(str(ROOT / "g1_lower_rl/assets/g1_gloria/xmls/scene_g1_with_dual_gloria_m.xml"))
  j = edited.joint("left_knee_joint").id
  edited.jnt_pos[j] = [0.02, -0.01, 0.03]
  edited.qpos0[edited.jnt_qposadr[j]] = 0.15
  data = mujoco.MjData(edited)
  s = state(1)
  s[:, 10] = 0.7
  s[:, 47] = 1.2
  data.qpos[:], data.qvel[:] = s[0, :36].numpy(), s[0, 36:].numpy()
  mujoco.mj_forward(edited, data)
  result = TorchKinematics(StaticKinematicTemplate.from_mujoco(edited))(s)
  ids = [edited.body(name).id for name in BODY_NAMES]
  np.testing.assert_allclose(result.body_position[0].numpy(), data.xpos[ids], atol=2e-14)
  p, r = np.zeros((3, edited.nv)), np.zeros((3, edited.nv))
  native = edited.body("left_ankle_roll_link").id
  mujoco.mj_jacBody(edited, data, p, r, native)
  np.testing.assert_allclose(
    result.body_linear_velocity[0, BODY_NAMES.index("left_ankle_roll_link")], p @ data.qvel, atol=2e-14
  )


@pytest.mark.parametrize("shape", [(71,), (2, 3, 71)])
def test_fk_batch_and_zero_pose_gradients(template, shape):
  s = torch.zeros(shape, dtype=torch.float64)
  s[..., 3] = 1
  s.requires_grad_()
  k = TorchKinematics(template)(s)
  objective = (k.foot_position * s.new_tensor([0.2, 0.7, 1.0])).sum() + k.body_quaternion.square().sum()
  (grad,) = torch.autograd.grad(objective, s)
  assert torch.isfinite(grad).all()
  assert grad[..., 7:19].abs().sum() > 0
  assert k.body_position.shape == (*shape[:-1], 44, 3)


def test_fk_gradcheck(template):
  s = state(1) + torch.randn(1, 71, dtype=torch.float64) * 0.05
  s.requires_grad_()
  fk = TorchKinematics(template)
  assert torch.autograd.gradcheck(
    lambda x: torch.cat((fk(x).foot_position.flatten(-2), fk(x).foot_linear_velocity.flatten(-2)), -1), (s,), fast_mode=True
  )


def test_force_hysteresis_onset_timers_reset_and_soft_gradient():
  tracker = ForceContact(ContactConfig(on_force=10, off_force=5, temperature=2))
  memory = ContactMemory.zeros(torch.zeros(1, 2, dtype=torch.float64))
  sequence = [(0, False), (9, False), (10, True), (7, True), (4, False), (11, True)]
  updates = []
  for value, expected in sequence:
    u = tracker(force(torch.full((1, 2), float(value), dtype=torch.float64)), memory, 0.02)
    assert u.memory.contact.all() == expected
    memory = u.memory
    updates.append(u)
  assert updates[2].onset.all() and not updates[3].onset.any()
  assert updates[4].liftoff.all()
  torch.testing.assert_close(updates[2].memory.last_air_time, torch.full((1, 2), 0.04, dtype=torch.float64))
  torch.testing.assert_close(updates[4].memory.last_contact_time, torch.full((1, 2), 0.04, dtype=torch.float64))
  torch.testing.assert_close(updates[5].memory.last_air_time, torch.full((1, 2), 0.02, dtype=torch.float64))
  reset = memory.reset(torch.tensor([True]))
  assert not reset.contact.any() and reset.contact_time.sum() == 0
  f = force(torch.full((1, 2), 8.0, dtype=torch.float64)).requires_grad_()
  u = tracker(f, reset, 0.02)
  (grad,) = torch.autograd.grad(u.soft.sum(), f)
  assert (grad[..., 2] > 0).all() and grad[..., :2].abs().sum() == 0


def test_force_contact_normals_batch_dt_and_validation():
  f = torch.tensor([[[12.0, 0, 0], [0, 0, -20.0]], [[12.0, 0, 0], [0, 0, 12.0]]])
  memory = ContactMemory.zeros(f[..., 0])
  normal = torch.tensor([[1.0, 0, 0], [0, 0, 1.0]])
  update = ForceContact()(f, memory, torch.tensor([0.01, 0.02]), normal)
  assert update.memory.contact.tolist() == [[True, False], [True, True]]
  torch.testing.assert_close(update.memory.contact_time[:, 0], torch.tensor([0.01, 0.02]))
  with pytest.raises(ValueError, match="positive"):
    ForceContact()(f, memory, 0)
  with pytest.raises(ValueError):
    ContactConfig(on_force=5, off_force=5)


def test_contact_threshold_calibration_and_overlap_rejection():
  values = torch.tensor([[0.0, 1.0], [2.0, 3.0], [18.0, 20.0], [25.0, 30.0]])
  labels = values > 10
  config = ContactConfig.calibrate(force(values), labels, tail_quantile=0)
  assert config.off_force == 3 and config.on_force == 18 and config.temperature == 3.75
  with pytest.raises(ValueError, match="overlap"):
    ContactConfig.calibrate(force(values), ~labels)
  with pytest.raises(ValueError, match="both"):
    ContactConfig.calibrate(force(values), torch.ones_like(labels))


def test_velocity_rewards_native_formulas_frames_height_and_gradients(template):
  s = state()
  angle = torch.tensor(torch.pi / 2)
  s[:, 3], s[:, 6] = (angle / 2).cos(), (angle / 2).sin()
  s[:, 36:39] = torch.tensor([[0.0, 0.6, 0.1], [0.0, -0.2, 0.0]], dtype=s.dtype)
  s[:, 39:42] = torch.tensor([[0.2, 0.1, 0.4], [0.0, 0.0, -0.2]], dtype=s.dtype)
  s.requires_grad_()
  command = s.new_tensor([[0.7, 0, 0.5], [0.2, 0, 0.2]])
  goal = VelocityGoal(command, s.new_tensor(0.75), s.new_tensor([1.0, 2.0]))
  result = LowerBodyVelocityAdapter(TorchKinematics(template))(s, goal)
  linear = result.observations["base_lin_vel"]
  torch.testing.assert_close(linear[:, 0], s.new_tensor([0.6, -0.2]), atol=2e-8, rtol=2e-7)
  error = (command[:, :2] - linear[:, :2]).square().sum(-1) + 2 * linear[:, 2].square()
  torch.testing.assert_close(result.rewards["linear_velocity"], torch.exp(-error / 0.5**2))
  angular_error = (s[:, 41] - command[:, 2]).square() + 0.05 * s[:, 39:41].square().sum(-1)
  torch.testing.assert_close(result.rewards["angular_velocity"], torch.exp(-angular_error / 0.5**2))
  (grad,) = torch.autograd.grad(sum(value.sum() for value in result.rewards.values()), s)
  assert torch.isfinite(grad).all() and grad[:, 36:42].abs().sum() > 0 and grad[:, 7:19].abs().sum() > 0


def test_velocity_standing_phase_and_command_shortfall(template):
  s = state()
  goal = VelocityGoal(s.new_tensor([[0, 0, 0], [2, 0, 0]]), s.new_tensor(0.79), s.new_tensor([1, 1]))
  result = LowerBodyVelocityAdapter(TorchKinematics(template), VelocityConfig(std_scale=0.3))(s, goal)
  assert result.observations["phase"][0].sum() == 0
  torch.testing.assert_close(result.costs["velocity_shortfall"], s.new_tensor([0, 1]))


def test_motion_identity_reference_tokens_and_finite_perfect_orientation_gradients(template):
  fk = TorchKinematics(template)
  s = state().requires_grad_()
  adapter = MotionTrackingAdapter(fk)
  result = adapter(s, MotionGoal(s.detach(), window=s.detach().unsqueeze(-2).expand(2, 3, 71)))
  for reward in result.rewards.values():
    torch.testing.assert_close(reward, torch.ones_like(reward))
  assert result.observations["reference_tokens"].shape == (2, 3, 38)
  torch.testing.assert_close(result.observations["reference_tokens"][..., 6:9], s.new_tensor([0, 0, -1]).expand(2, 3, 3))
  (grad,) = torch.autograd.grad(sum(v.sum() for v in result.rewards.values()), s)
  assert torch.isfinite(grad).all()
  assert not any(event.any() for event in result.events.values())


def test_motion_reanchoring_translation_heading_and_joint_tracking_gradients(template):
  fk = TorchKinematics(template)
  ref = state()
  actual = ref.clone()
  actual[:, :2] += actual.new_tensor([2.0, -1.0])
  theta = actual.new_tensor(0.5)
  actual[:, 3], actual[:, 6] = (theta / 2).cos(), (theta / 2).sin()
  adapter = MotionTrackingAdapter(fk)
  result = adapter(actual, MotionGoal(ref))
  torch.testing.assert_close(result.rewards["body_position"], torch.ones(2, dtype=actual.dtype))
  torch.testing.assert_close(result.rewards["body_orientation"], torch.ones(2, dtype=actual.dtype))
  assert (result.rewards["anchor_position"] < 0.01).all()
  actual[:, 10] = 0.1
  actual.requires_grad_()
  result = adapter(actual, MotionGoal(ref))
  (grad,) = torch.autograd.grad(result.rewards["body_position"].sum(), actual)
  assert grad[:, 10].abs().sum() > 0


def test_motion_tokens_yaw_invariant_and_fixed_gripper_parent_gradients(template):
  fk = TorchKinematics(template)
  s = state()
  s[:, 36:39] = s.new_tensor([0.2, -0.1, 0.05])
  theta = s.new_tensor(0.8)
  rotation = torch.stack(((theta / 2).cos(), theta * 0, theta * 0, (theta / 2).sin())).expand(2, 4)
  transformed = s.clone()
  transformed[:, 3:7] = multiply(rotation, s[:, 3:7])
  transformed[:, 36:39] = rotate(rotation, s[:, 36:39])
  torch.testing.assert_close(MotionTrackingAdapter.reference_tokens(s), MotionTrackingAdapter.reference_tokens(transformed))
  s.requires_grad_()
  eccentric = [BODY_NAMES.index("left_eccentric"), BODY_NAMES.index("right_eccentric")]
  loss = (fk(s).body_quaternion[:, eccentric] * s.new_tensor([0.2, 0.3, 0.5, 0.9])).sum()
  (grad,) = torch.autograd.grad(loss, s)
  wrist_slots = [7 + JOINT_NAMES.index(f"{side}_wrist_yaw_joint") for side in ("left", "right")]
  assert grad[:, wrist_slots].abs().min() > 0
  assert all(template.body_joint[index] == -1 for index in eccentric)


def test_footstep_landing_after_air_once_per_target_and_timer_credit(template):
  fk, s = TorchKinematics(template), state(1)
  adapter = FootstepTrackingAdapter(fk)
  goal = goal_for(s, fk, phase=0.0, dx=0.03)
  memory = FootstepMemory.zeros(goal.targets)
  airborne = contact_update(s.new_tensor([[0, 20]]))
  out = adapter(s, goal, airborne, memory, 0.02)
  assert out.memory.saw_air.tolist() == [[True, False]]
  assert not out.signals.events["landing"].any()
  landed = contact_update(s.new_tensor([[20, 20]]), airborne.memory)
  out = adapter(s, goal, landed, out.memory, 0.02)
  assert out.signals.events["landing"].tolist() == [[True, False]]
  assert out.signals.costs["landing"].item() > 0
  again = adapter(s, goal, landed, out.memory, 0.02)
  assert not again.signals.events["landing"].any()
  # A target change while grounded cannot fabricate a landing or carry stale saw_air.
  next_goal = goal_for(s, fk, ids=1, phase=0.0)
  next_out = adapter(s, next_goal, landed, out.memory, 0.02)
  assert not next_out.memory.saw_air.any() and not next_out.signals.events["landing"].any()
  assert not next_out.memory.reset(torch.tensor([True])).saw_air.any()
  for _ in range(5):
    airborne = contact_update(s.new_tensor([[0, 20]]), airborne.memory)
  out = adapter(s, goal, airborne, memory, 0.02)
  assert out.signals.rewards["air_time"].item() > 0


def test_footstep_landing_error_gradient_and_standing_override(template):
  fk, s = TorchKinematics(template), state(1)
  adapter = FootstepTrackingAdapter(fk)
  goal = goal_for(s, fk, phase=0.0, dx=0.03)
  airborne = contact_update(s.new_tensor([[0, 20]]))
  out = adapter(s, goal, airborne, FootstepMemory.zeros(goal.targets), 0.02)
  landed = contact_update(s.new_tensor([[20, 20]]), airborne.memory)
  s.requires_grad_()
  out = adapter(s, goal, landed, out.memory, 0.02)
  (grad,) = torch.autograd.grad(out.signals.costs["landing"].sum(), s)
  assert torch.isfinite(grad).all() and grad[..., 7:13].abs().sum() > 0
  standing = goal_for(s.detach(), fk, frequency=0, dx=0.03)
  out = adapter(s, standing, landed, out.memory, 0.02)
  assert out.signals.observations["stance"].all()
  assert not out.signals.events["landing"].any()
  assert out.signals.rewards["schedule"].item() == 0


def test_footstep_soft_force_target_and_joint_gradients(template):
  fk, s = TorchKinematics(template), state(1).requires_grad_()
  goal = goal_for(s.detach(), fk, phase=0.0, dx=0.07)
  targets = goal.targets.clone().requires_grad_()
  goal = FootstepGoal(targets, goal.target_ids, goal.phase, goal.frequency, goal.ground_height)
  f = force(s.new_tensor([[8, 9]])).requires_grad_()
  u = ForceContact()(f, ContactMemory.zeros(s.new_zeros(1, 2)), 0.02)
  out = FootstepTrackingAdapter(fk)(s, goal, u, FootstepMemory.zeros(goal.targets), 0.02)
  objective = out.signals.rewards["soft_swing_position"].sum() + out.signals.costs["footprint_accuracy"].sum()
  sg, tg, fg = torch.autograd.grad(objective, (s, targets, f))
  assert all(torch.isfinite(g).all() for g in (sg, tg, fg))
  assert sg[..., 7:19].abs().sum() > 0 and tg.abs().sum() > 0 and fg[..., 1, 2].abs().sum() > 0
  assert out.signals.observations["foot_contact_forces"].shape == (1, 6)


def test_phase_windows_exact_boundaries_and_arbitrary_batches(template):
  cfg = FootstepPhaseCfg()
  phases = torch.tensor([*cfg.contact_start_rad, *cfg.liftoff_rad], dtype=torch.float64)
  s = state(4).reshape(2, 2, 71)
  fk = TorchKinematics(template)
  goal = goal_for(s, fk)
  goal = FootstepGoal(goal.targets, goal.target_ids, phases.reshape(2, 2), goal.frequency, goal.ground_height)
  update = contact_update(torch.full((2, 2, 2), 20.0, dtype=s.dtype))
  out = FootstepTrackingAdapter(fk, FootstepConfig(phase=cfg))(s, goal, update, FootstepMemory.zeros(goal.targets), 0.02)
  assert out.signals.observations["stance"].reshape(4, 2).tolist() == [[True, True], [True, True], [False, True], [True, False]]


def test_action_pathway_to_task_reward(template):
  """A differentiable next state must retain gradients all the way back to torque-like inputs."""
  s = state(1)
  action = torch.full((1, 29), 0.2, dtype=s.dtype, requires_grad=True)
  # Explicit toy actuator/dynamics integration, NOT a claim of physics fidelity.
  delta = torch.cat((s.new_zeros(1, 6), action * 0.02**2, s.new_zeros(1, 6), action * 0.02), dim=-1)
  predicted = apply_state_delta(s, delta)
  result = MotionTrackingAdapter(TorchKinematics(template))(predicted, MotionGoal(s))
  (grad,) = torch.autograd.grad(result.rewards["body_position"].sum() + result.rewards["body_linear_velocity"].sum(), action)
  assert torch.isfinite(grad).all() and grad[..., :12].abs().sum() > 0


def test_actual_world_model_action_to_task_gradients(template):
  torch.manual_seed(813)
  wm = PositionWorldModel(PositionWorldModelCfg(hidden_dim=16, interaction_dim=24, num_blocks=1)).double()
  wm.set_normalization(
    torch.zeros(145, dtype=torch.float64),
    torch.ones(145, dtype=torch.float64),
    torch.ones(70, dtype=torch.float64),
    torch.ones(6, dtype=torch.float64),
    torch.ones(70, dtype=torch.float64),
  )
  s = state(1)
  torque = torch.full((1, 29), 0.2, dtype=s.dtype, requires_grad=True)
  prediction = wm(s, torque, s.new_ones(1, 48))
  signals = MotionTrackingAdapter(TorchKinematics(template))(prediction.state, MotionGoal(s))
  objective = signals.rewards["body_position"].sum() + signals.rewards["body_linear_velocity"].sum()
  (grad,) = torch.autograd.grad(objective, torque)
  assert torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_gravity_torque_matches_native_all_joints_and_runtime_com_mass(model, template):
  rng = np.random.default_rng(282)
  s = state(5)
  s[:, 7:36] = torch.from_numpy(rng.uniform(-0.7, 0.7, (5, 29)))
  quats = rng.normal(size=(5, 4))
  quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
  s[:, 3:7] = torch.from_numpy(quats)
  ids = [model.body(name).id for name in BODY_NAMES]
  mass = torch.from_numpy(model.body_mass[ids].copy())
  torque = TorchGravityTorque(TorchKinematics(template))(s, mass)
  data = mujoco.MjData(model)
  joint_ids = [model.joint(name).id for name in JOINT_NAMES]
  dofs = [model.jnt_dofadr[j] for j in joint_ids]
  for i in range(5):
    data.qpos[:] = s[i, :36].numpy()
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(torque[i].numpy(), -data.qfrc_bias[dofs], atol=2e-13)
  # Runtime randomization and payload mass/COM must affect controller GC, not just dynamics.
  changed_mass = mass.clone()
  changed_mass[BODY_NAMES.index("left_gripper_base")] += 1.2
  changed_com = torch.tensor(template.body_com_position)
  changed_com[BODY_NAMES.index("torso_link"), 0] += 0.04
  s.requires_grad_()
  dynamic = TorchGravityTorque(TorchKinematics(template))(s, changed_mass, changed_com)
  assert not torch.allclose(dynamic, torque)
  (grad,) = torch.autograd.grad(dynamic.square().sum(), s)
  assert torch.isfinite(grad).all() and grad[:, 7:36].abs().sum() > 0


@pytest.mark.parametrize("dt", [0.005, 0.0025])
def test_controller_recomputes_gc_every_substep_nominal_vs_effective_gains(model, template, dt):
  fk = TorchKinematics(template)
  nominal = torch.linspace(20.0, 60.0, 29, dtype=torch.float64)
  controller = SubstepTorqueController(fk, nominal)
  s = state(1)
  s[:, 42:71] = 0.1
  action = torch.full((1, 29), 0.2, dtype=s.dtype, requires_grad=True)
  processed = controller.process_action(action, torch.ones(29, dtype=s.dtype), torch.zeros(29, dtype=s.dtype))
  mass = torch.tensor(model.body_mass[list(template.source_body_ids)], dtype=s.dtype)
  bias = torch.full((1, 29), 0.01, dtype=s.dtype)
  kp, kd = nominal * 1.3, torch.full((29,), 2.0, dtype=s.dtype)
  effort = torch.tensor([[-1000.0, 1000.0]], dtype=s.dtype).expand(29, 2)
  gain = torch.full((1, 29), 0.9, dtype=s.dtype)
  result = controller(s, processed, kp, kd, effort, mass, bias, gain)
  expected_target = processed - bias - result.gravity_torque * gain * controller.compensated_mask / nominal
  torch.testing.assert_close(result.corrected_target, expected_target)
  torch.testing.assert_close(result.torque, kp * (expected_target - s[:, 7:36]) - kd * s[:, 42:71])
  assert not result.saturated.any()
  (grad,) = torch.autograd.grad(result.torque.sum(), action)
  torch.testing.assert_close(grad, kp.unsqueeze(0))
  next_s = s.clone()
  next_s[:, 7:36] += next_s[:, 42:71] * dt
  next_result = controller(next_s, processed, kp, kd, effort, mass, bias, gain)
  assert not torch.allclose(next_result.corrected_target, result.corrected_target, atol=1e-10, rtol=1e-10)
  tiny_effort = torch.tensor([[-0.1, 0.2]], dtype=s.dtype).expand(29, 2)
  clipped = controller(s, processed, kp, kd, tiny_effort, mass, bias, gain)
  assert clipped.saturated.any() and (clipped.torque >= -0.1).all() and (clipped.torque <= 0.2).all()


@pytest.mark.parametrize("task", ["motion", "footstep", "velocity"])
@pytest.mark.parametrize("with_gc", [True, False])
def test_task_ready_two_tick_rollout_action_gradients_all_three_tasks(model, template, task, with_gc):
  torch.manual_seed(726)
  wm = PositionWorldModel(PositionWorldModelCfg(hidden_dim=16, interaction_dim=24, num_blocks=1)).double()
  wm.set_normalization(
    torch.zeros(145, dtype=torch.float64),
    torch.ones(145, dtype=torch.float64),
    torch.ones(70, dtype=torch.float64),
    torch.ones(6, dtype=torch.float64),
    torch.ones(70, dtype=torch.float64),
  )
  s, fk = state(1), TorchKinematics(template)
  if task == "motion":
    adapter, goal = MotionTrackingAdapter(fk), MotionGoal(s.clone())
  elif task == "velocity":
    adapter, goal = LowerBodyVelocityAdapter(fk), VelocityGoal(s.new_tensor([[0.3, 0.1, 0.2]]), s.new_tensor(0.75))
  else:
    adapter, goal = FootstepTrackingAdapter(fk), goal_for(s, fk, dx=0.05)
  controller = SubstepTorqueController(fk, s.new_full((29,), 40))
  composed = PositionTaskRollout(wm, adapter, controller=controller if with_gc else None)
  physics = s.new_ones(1, 48)
  physics[:, :44] = torch.tensor(model.body_mass[list(template.source_body_ids)], dtype=s.dtype)
  physics[:, 45:48] = 0
  parameters = ControllerContext(s.new_full((29,), 45), s.new_full((29,), 2), s.new_tensor([[-100, 100]]).expand(29, 2))
  memory = ContactMemory.zeros(s.new_zeros(1, 2))
  action1 = torch.full((1, 29), 0.02, dtype=s.dtype, requires_grad=True)
  action2 = torch.full((1, 29), 0.03, dtype=s.dtype, requires_grad=True)
  external = parameters if with_gc else None
  first = composed(s, action1, goal, physics, memory, controller_context=external)
  second = composed(
    first.prediction.state, action2, goal, physics, first.contact_memory, first.footstep_memory, controller_context=external
  )
  assert second.prediction.state.shape == (1, 71) and second.joint_target.shape == (1, 29)
  torch.testing.assert_close(second.last_action, action2)
  assert (second.contact_memory.air_time + second.contact_memory.contact_time > 0).all()
  if task == "footstep":
    assert second.footstep_memory is not None
    objective = second.signals.costs["footprint_accuracy"].sum() - second.signals.rewards["soft_swing_position"].sum()
  else:
    objective = -sum(value.sum() for value in second.signals.rewards.values())
  gradients = torch.autograd.grad(objective, (action1, action2))
  assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)


def controller_metadata():
  return {
    "kp_nominal": [40.0] * 29,
    "kd_nominal": [2.0] * 29,
    "effort_range_nominal": [[-100.0, 100.0]] * 29,
    "gravity_compensation_nominal_kp": list(range(20, 34)),
    "encoder_bias_range": [-0.01, 0.01],
    "kp_scale_range": [0.8, 1.2],
    "kd_scale_range": [0.8, 1.2],
    "effort_scale_range": [0.8, 1.2],
    "gravity_compensation_gain_range": [0.9, 1.01],
  }


def test_controller_metadata_factory_maps_gc14_and_preserves_nominal_effective_separation(template):
  setup = controller_from_metadata(controller_metadata(), template.to_dict(), dtype=torch.float64)
  expected = torch.arange(20, 34, dtype=torch.float64)
  torch.testing.assert_close(setup.controller.nominal_gc_kp[list(ARM_JOINT_INDICES)], expected)
  torch.testing.assert_close(setup.context.effective_kp, torch.full((29,), 40.0, dtype=torch.float64))
  assert setup.controller.compensated_mask[:15].eq(0).all()
  assert setup.controller.nominal_gc_kp[:15].eq(1).all()
  assert setup.context.encoder_bias.sum() == 0
  assert setup.context.gc_gain.eq(1).all()
  invalid = controller_metadata()
  invalid["gravity_compensation_nominal_kp"] = [1.0] * 13
  with pytest.raises(ValueError, match="14"):
    controller_from_metadata(invalid, template)


def test_real_checkpoint_load_and_all_three_task_smoke_without_simulator_imports(model, template):
  from g1_lower_rl.world_model.checkpoint import CHECKPOINT_VERSION, atomic_checkpoint, primitive_metadata

  torch.manual_seed(913)
  wm = PositionWorldModel(PositionWorldModelCfg(hidden_dim=16, interaction_dim=24, num_blocks=1)).double()
  context = torch.ones(1, 48, dtype=torch.float64)
  context[:, :44] = torch.tensor(model.body_mass[list(template.source_body_ids)], dtype=context.dtype)
  context[:, 45:] = 0
  mean = pack_inputs(state(1), torch.zeros(1, 29, dtype=context.dtype), context)[0]
  wm.set_normalization(
    mean,
    torch.ones_like(mean),
    torch.ones(70, dtype=mean.dtype),
    torch.ones(6, dtype=mean.dtype),
    torch.ones(70, dtype=mean.dtype),
  )
  payload = {
    "format_version": CHECKPOINT_VERSION,
    "timestamp_utc": "2026-10-26T11:21:00+00:00",
    "model_config": {"hidden_dim": 16, "interaction_dim": 24, "num_blocks": 1},
    "config": {"synthetic": False},
    "model": wm.state_dict(),
    "optimizer": torch.optim.Adam(wm.parameters()).state_dict(),
    "scaler": None,
    "normalization": {"source": "adapter_cpu_fixture"},
    "provenance": primitive_metadata(
      [
        {
          "collector": {
            "controller_template": controller_metadata(),
            "static_template": template.to_dict(),
            "physics_dt": 0.005,
            "control_dt": 0.02,
          },
          "rank": 0,
          "seed": 913,
          "validation_env_indices": [1, 2],
          "training_env_indices": [0, 1],
          "resume_replay": "fresh_trajectories_no_environment_restore",
        }
      ]
    ),
    "rng": [],
    "update": 0,
    "transitions": 0,
    "wall_seconds": 0.0,
    "world_size": 1,
    "best_score": 1.0,
    "best_metric": "adapter_fixture",
    "metrics": {},
    "runtime": {"collection_seconds": 0.0, "training_seconds": 0.0},
  }
  checkpoint = ROOT / "tests" / f".wm_adapter_checkpoint_{uuid4().hex}.pt"
  import builtins

  original_import = builtins.__import__

  def no_simulator_import(name, *args, **kwargs):
    if name == "mujoco" or name.startswith(("mujoco.", "mjlab")):
      raise AssertionError("Inference attempted a simulator import")
    return original_import(name, *args, **kwargs)

  try:
    atomic_checkpoint(checkpoint, payload)
    with patch("builtins.__import__", side_effect=no_simulator_import):
      report = smoke_checkpoint(checkpoint, steps=2)
    assert set(report) == {"motion", "footstep", "velocity"}
    assert all(result["action_gradient_l1"] > 0 for result in report.values())
    with pytest.raises(ValueError, match="rank"):
      smoke_checkpoint(checkpoint, rank=1)
  finally:
    checkpoint.unlink(missing_ok=True)


def test_position_target_preprocessing_matches_confirmed_order_and_bias_stays_outside_network():
  action = torch.full((2, 29), 0.4, dtype=torch.float64, requires_grad=True)
  scale, offset = torch.full_like(action, 2), torch.full_like(action, 0.1)
  noise, bias = torch.full_like(action, 0.2), torch.full_like(action, 0.03)
  limits = torch.tensor([[-1.0, 1.0]], dtype=action.dtype).expand(29, 2)
  target = process_position_target(action, scale, offset, limits, bias, noise)
  torch.testing.assert_close(target, torch.full_like(action, 0.97))
  with pytest.raises(ValueError, match="encoder_bias"):
    process_position_target(action, scale, offset, encoder_bias=bias[:1])


def test_contact_yaw_rate_cost_matches_native_formula_from_predicted_fk(template):
  from types import SimpleNamespace

  from g1_lower_rl.tasks.footstep_tracking.rewards import feet_contact_yaw_rate_cost

  fk, s = TorchKinematics(template), state(2)
  s[:, 39:42] = s.new_tensor([0.2, -0.3, 0.4])
  s[:, 44], s[:, 50] = 0.7, -0.9
  goal = goal_for(s, fk)
  contact = contact_update(s.new_tensor([[20, 0], [0, 20]]))
  result = FootstepTrackingAdapter(fk)(s, goal, contact, FootstepMemory.zeros(goal.targets), 0.02)
  sites = fk(s).foot_angular_velocity
  env = SimpleNamespace(
    num_envs=2,
    scene={
      "robot": SimpleNamespace(data=SimpleNamespace(site_ang_vel_w=sites)),
      "feet_ground_contact": SimpleNamespace(data=SimpleNamespace(found=contact.memory.contact.long())),
    },
  )
  expected = feet_contact_yaw_rate_cost(env, SimpleNamespace(name="robot", site_ids=[0, 1]))
  torch.testing.assert_close(result.signals.costs["feet_contact_yaw_rate"], expected)
