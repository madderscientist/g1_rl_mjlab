import copy
import io
import math
from unittest.mock import patch

import pytest
import torch
from torch import nn

from g1_lower_rl.world_model import (
  GatedResidualBlock,
  PositionWorldModel,
  PositionWorldModelCfg,
  apply_state_delta,
  integrate_state,
  pack_inputs,
  state_delta,
  state_features,
  state_update,
)
from g1_lower_rl.world_model.model import (
  BODY_NAMES,
  CONTEXT_DIM,
  CONTEXT_FIELDS,
  CONTROL_DT,
  DELTA_DIM,
  DELTA_FIELDS,
  FOOT_FORCE_DIM,
  INPUT_DIM,
  INPUT_FIELDS,
  JOINT_NAMES,
  STATE_DIM,
  STATE_FEATURE_DIM,
  STATE_FEATURE_FIELDS,
  STATE_FIELDS,
  TORQUE_DIM,
  _quat_mul,
  _rotation_quaternion,
  root_to_world,
  world_to_root,
)


def make_state(batch=3, dtype=torch.float32):
  state = torch.zeros(batch, STATE_DIM, dtype=dtype)
  state[:, 2] = 0.8
  state[:, 3] = 1
  return state


def make_inputs(batch=3, dtype=torch.float32):
  state = make_state(batch, dtype)
  context = torch.ones(batch, CONTEXT_DIM, dtype=dtype)
  context[:, CONTEXT_FIELDS["torso_com"]] = 0
  return pack_inputs(
    state,
    torch.zeros(batch, TORQUE_DIM, dtype=dtype),
    context,
  )


def make_model(dtype=torch.float32):
  model = PositionWorldModel(PositionWorldModelCfg(hidden_dim=32, interaction_dim=16, num_blocks=3)).to(dtype=dtype)
  model.set_normalization(
    torch.zeros(INPUT_DIM, dtype=dtype),
    torch.ones(INPUT_DIM, dtype=dtype),
    torch.ones(DELTA_DIM, dtype=dtype),
    torch.ones(FOOT_FORCE_DIM, dtype=dtype),
    torch.ones(DELTA_DIM, dtype=dtype),
  )
  return model


def zero_heads(model):
  with torch.no_grad():
    for head in (model.delta_head, model.force_head):
      head.weight.zero_()
      head.bias.zero_()


def test_trusted_training_path_matches_checked_forward_and_gradients():
  model = make_model()
  state = make_state().requires_grad_()
  target = torch.zeros(3, 29, requires_grad=True)
  context = make_inputs()[:, INPUT_FIELDS["context"]]
  checked = model(state, target, context).as_tensor()
  trusted = model(state, target, context, validate=False).as_tensor()
  torch.testing.assert_close(checked, trusted)
  checked_gradient = torch.autograd.grad(checked.square().sum(), target)[0]
  trusted_gradient = torch.autograd.grad(trusted.square().sum(), target)[0]
  torch.testing.assert_close(checked_gradient, trusted_gradient)
  torch.testing.assert_close(
    state_delta(state, model(state, target, context).state),
    state_delta(state, model(state, target, context).state, validate=False),
  )


def test_trusted_path_supports_fullgraph_compilation():
  model = make_model()
  state, inputs = make_state(), make_inputs()
  target = inputs[:, INPUT_FIELDS["joint_target"]].clone().requires_grad_()
  context = inputs[:, INPUT_FIELDS["context"]]
  compiled = torch.compile(
    lambda value: model(state, value, context, validate=False).as_tensor(), backend="eager", fullgraph=True
  )
  torch.testing.assert_close(compiled(target), model(state, target, context).as_tensor())
  compiled(target).sum().backward()
  assert target.grad is not None and torch.isfinite(target.grad).all()


def test_layout_is_exhaustive_and_closed_grippers_have_no_coordinates():
  for fields, size in (
    (STATE_FIELDS, STATE_DIM),
    (STATE_FEATURE_FIELDS, STATE_FEATURE_DIM),
    (DELTA_FIELDS, DELTA_DIM),
    (INPUT_FIELDS, INPUT_DIM),
    (CONTEXT_FIELDS, CONTEXT_DIM),
  ):
    indices = [i for part in fields.values() for i in range(part.start, part.stop)]
    assert indices == list(range(size))
  assert len(JOINT_NAMES) == len(set(JOINT_NAMES)) == TORQUE_DIM
  assert len(BODY_NAMES) == len(set(BODY_NAMES)) == 44
  assert JOINT_NAMES[0] == "left_hip_pitch_joint"
  assert JOINT_NAMES[12:15] == ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
  assert JOINT_NAMES[22] == "right_shoulder_pitch_joint"
  assert JOINT_NAMES[-1] == "right_wrist_yaw_joint"
  assert not any("eccentric" in name for name in JOINT_NAMES)
  state = make_state(2)
  target, context = torch.full((2, 29), 2.0), torch.full((2, 48), 3.0)
  blocks = (state_features(state), target, context)
  packed = pack_inputs(state, target, context)
  for field, block in zip(INPUT_FIELDS, blocks, strict=True):
    torch.testing.assert_close(packed[:, INPUT_FIELDS[field]], block)


def test_default_architecture_uses_1024_interactions_not_2048():
  model = PositionWorldModel()
  assert model.cfg == PositionWorldModelCfg(2048, 1024, 3)
  assert model.embedding.in_features == INPUT_DIM and model.embedding.out_features == 2048
  assert len(model.blocks) == 3
  for block in model.blocks:
    assert isinstance(block, GatedResidualBlock)
    assert block.gate.weight.shape == (1024, 2048)
    assert block.value.weight.shape == (1024, 2048)
    assert block.projection.weight.shape == (2048, 1024)
  first, second = model.blocks[0], model.blocks[1]
  assert isinstance(first, GatedResidualBlock) and isinstance(second, GatedResidualBlock)
  assert first.gate.weight is not second.gate.weight
  assert model.delta_head.out_features == DELTA_DIM
  assert model.force_head.out_features == FOOT_FORCE_DIM
  model.set_normalization(torch.zeros(145), torch.ones(145), torch.ones(70), torch.ones(6), torch.ones(70))
  inputs = make_inputs(batch=2)
  result = model(make_state(2), inputs[:, INPUT_FIELDS["joint_target"]], inputs[:, INPUT_FIELDS["context"]])
  assert result.state.shape == (2, 71)
  assert result.foot_force.shape == (2, 2, 3)
  assert result.as_tensor().shape == (2, 77)
  assert torch.isfinite(result.as_tensor()).all()
  torch.testing.assert_close(result.state[:, 3:7].norm(dim=-1), torch.ones(2))


def test_gated_block_equation_and_identity_residual():
  block = GatedResidualBlock(hidden_dim=4, interaction_dim=2).double()
  hidden = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
  normalized = block.norm(hidden)
  expected = hidden + block.projection(torch.nn.functional.silu(block.gate(normalized)) * block.value(normalized))
  torch.testing.assert_close(block(hidden), expected)
  with torch.no_grad():
    block.projection.weight.zero_()
    block.projection.bias.zero_()
  torch.testing.assert_close(block(hidden), hidden)
  block(hidden).sum().backward()
  torch.testing.assert_close(hidden.grad, torch.ones_like(hidden))


def test_gate_really_multiplies_separate_input_features():
  block = GatedResidualBlock(hidden_dim=2, interaction_dim=1).double()
  with torch.no_grad():
    block.gate.weight.copy_(torch.tensor([[1.0, 0.0]]))
    block.gate.bias.zero_()
    block.value.weight.copy_(torch.tensor([[0.0, 1.0]]))
    block.value.bias.zero_()
    block.projection.weight.copy_(torch.tensor([[1.0], [0.0]]))
    block.projection.bias.zero_()
  hidden = torch.tensor([[0.5, 2.0]], dtype=torch.float64)
  expected = hidden.clone()
  expected[:, 0] += torch.nn.functional.silu(hidden[:, 0]) * hidden[:, 1]
  with patch.object(block, "norm", nn.Identity()):
    torch.testing.assert_close(block(hidden), expected)


def test_zero_increment_preserves_state_and_normalizes_quaternion():
  state = make_state(dtype=torch.float64)
  state[:, 7:] = torch.randn_like(state[:, 7:])
  delta = torch.zeros(3, DELTA_DIM, dtype=torch.float64)
  torch.testing.assert_close(apply_state_delta(state, delta), state)
  scaled = state.clone()
  scaled[:, 3:7] *= 2
  torch.testing.assert_close(apply_state_delta(scaled, delta), state)


def test_rotation_increment_is_right_composed_in_root_body_frame():
  state = make_state(batch=1, dtype=torch.float64)
  state[:, 3:7] = torch.tensor([math.sqrt(0.5), 0, 0, math.sqrt(0.5)])
  delta = torch.zeros(1, DELTA_DIM, dtype=torch.float64)
  delta[:, 3] = math.pi / 2
  result = apply_state_delta(state, delta)
  torch.testing.assert_close(result[:, 3:7], torch.full((1, 4), 0.5, dtype=torch.float64))
  torch.testing.assert_close(state_delta(state, result), delta)


def test_tangent_delta_roundtrip_and_quaternion_double_cover():
  torch.manual_seed(12)
  state = make_state(batch=7, dtype=torch.float64)
  state[:, 3:7] = torch.randn(7, 4, dtype=torch.float64)
  state[:, 3:7] /= state[:, 3:7].norm(dim=-1, keepdim=True)
  delta = 0.1 * torch.randn(7, DELTA_DIM, dtype=torch.float64)
  next_state = apply_state_delta(state, delta)
  torch.testing.assert_close(state_delta(state, next_state), delta)
  next_state[:, 3:7] *= -1
  torch.testing.assert_close(state_delta(state, next_state), delta)


def test_rotation_zero_and_small_angle_gradients_are_finite_and_correct():
  state = make_state(batch=1, dtype=torch.float64)
  delta = torch.zeros(1, DELTA_DIM, dtype=torch.float64, requires_grad=True)
  assert torch.autograd.gradcheck(lambda value: apply_state_delta(state, value), (delta,))
  next_state = state.clone().requires_grad_()
  assert torch.autograd.gradcheck(lambda value: state_delta(state, value), (next_state,))
  tiny = delta.detach().clone()
  tiny[:, 3:6] = torch.tensor([1e-10, -2e-10, 3e-10])
  torch.testing.assert_close(state_delta(state, apply_state_delta(state, tiny)), tiny, atol=1e-12, rtol=1e-6)


def test_global_residual_is_in_physical_units_and_force_is_not_residual():
  model = make_model()
  zero_heads(model)
  delta_scale = torch.full((DELTA_DIM,), 0.2)
  force_scale = torch.arange(1, 7, dtype=torch.float32)
  model.set_normalization(
    torch.full((INPUT_DIM,), 0.5), torch.full((INPUT_DIM,), 2.0), delta_scale, force_scale, torch.ones(70)
  )
  with torch.no_grad():
    model.delta_head.bias.fill_(0.25)
    model.force_head.bias.copy_(torch.arange(6, dtype=torch.float32))
  inputs = make_inputs()
  result = model(make_state(), inputs[:, INPUT_FIELDS["joint_target"]], inputs[:, INPUT_FIELDS["context"]])
  torch.testing.assert_close(result.state, integrate_state(make_state(), torch.full((3, 70), 0.05)))
  expected_force = (torch.arange(6, dtype=torch.float32) * force_scale).reshape(1, 2, 3).expand(3, 2, 3)
  torch.testing.assert_close(result.foot_force_root, expected_force)


@pytest.mark.parametrize("shape", [(), (2,), (2, 3)])
def test_leading_dimensions_and_quaternion_sign_invariance(shape):
  model = make_model()
  inputs = make_inputs(batch=1)[0].expand(*shape, INPUT_DIM).clone()
  state = make_state(1)[0].expand(*shape, STATE_DIM).clone()
  target, context = inputs[..., INPUT_FIELDS["joint_target"]], inputs[..., INPUT_FIELDS["context"]]
  result = model(state, target, context)
  flipped = state.clone()
  flipped[..., 3:7] *= -1
  other = model(flipped, target, context)
  torch.testing.assert_close(result.foot_force, other.foot_force)
  torch.testing.assert_close(result.state[..., :3], other.state[..., :3])
  torch.testing.assert_close(result.state[..., 7:], other.state[..., 7:])
  torch.testing.assert_close(result.state[..., 3:7], -other.state[..., 3:7])
  assert result.as_tensor().shape == (*shape, STATE_DIM + FOOT_FORCE_DIM)


@pytest.mark.parametrize("steps", [4, 8])
def test_autoregressive_rollout_backpropagates_to_each_target_and_physical_context(steps):
  torch.manual_seed(42)
  model = make_model(dtype=torch.float64)
  seed = make_inputs(batch=2, dtype=torch.float64)
  state = make_state(2, dtype=torch.float64).requires_grad_()
  initial = state
  torque = (0.1 * torch.randn(steps, 2, 29, dtype=torch.float64)).requires_grad_()
  context = seed[:, INPUT_FIELDS["context"]].clone().requires_grad_()
  force_loss = torch.zeros((), dtype=torch.float64)
  for step in range(steps):
    prediction = model(state, torque[step], context)
    state = prediction.state
    force_loss = force_loss + prediction.foot_force.square().mean()
  loss = state.square().mean() + force_loss / steps
  loss.backward()
  for gradient in (initial.grad, torque.grad, context.grad):
    assert gradient is not None
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
  assert torque.grad is not None
  assert (torque.grad.abs().sum(dim=(1, 2)) > 0).all()
  for parameter in model.parameters():
    assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_normalization_fit_matches_samples_and_checkpoint_roundtrip():
  model = make_model(dtype=torch.float64)
  inputs = make_inputs(batch=5, dtype=torch.float64)
  inputs[:, INPUT_FIELDS["joint_target"]] = torch.randn(5, 29, dtype=torch.float64)
  delta = torch.linspace(0.01, 0.1, DELTA_DIM, dtype=torch.float64).expand(5, DELTA_DIM)
  states = make_state(5, dtype=torch.float64)
  next_states = integrate_state(states, delta)
  forces = torch.arange(30, dtype=torch.float64).reshape(5, 2, 3)
  model.fit_normalization(inputs, states, next_states, forces)
  torch.testing.assert_close(model.input_mean, inputs.mean(dim=0))
  torch.testing.assert_close(model.input_scale, inputs.std(dim=0, correction=0).clamp_min(1e-6))
  torch.testing.assert_close(model.delta_scale, delta[0])
  local = world_to_root(next_states[:, None, 3:7].expand(5, 2, 4), forces)
  torch.testing.assert_close(model.force_scale, local.flatten(-2).square().mean(0).sqrt())
  target, context = inputs[:, INPUT_FIELDS["joint_target"]], inputs[:, INPUT_FIELDS["context"]]
  expected = model(states, target, context).as_tensor()
  buffer = io.BytesIO()
  torch.save(model.state_dict(), buffer)
  buffer.seek(0)
  restored = PositionWorldModel(model.cfg).double()
  restored.load_state_dict(torch.load(buffer, weights_only=True))
  assert restored.normalization_ready
  torch.testing.assert_close(restored(states, target, context).as_tensor(), expected, atol=0, rtol=0)


def test_normalization_is_explicit_and_invalid_update_is_atomic():
  model = PositionWorldModel(PositionWorldModelCfg(32, 16, 3))
  with pytest.raises(RuntimeError, match="normalization"):
    model.predict_updates(make_inputs())
  before = model.input_mean.clone()
  with pytest.raises(ValueError, match="positive"):
    model.set_normalization(torch.ones(145), torch.ones(145), torch.zeros(70), torch.ones(6), torch.ones(70))
  torch.testing.assert_close(model.input_mean, before)
  assert not model.normalization_ready


def test_checkpoint_rejects_incompatible_physical_contract():
  model = make_model()
  saved = copy.deepcopy(model.state_dict())
  saved["_extra_state"]["rotation_increment"] = "world_frame_left_composed"
  with pytest.raises(RuntimeError, match="physical interface"):
    model.load_state_dict(saved)


def test_model_can_fit_position_target_transition_and_contact_force():
  torch.manual_seed(7)
  model = make_model()
  inputs = make_inputs(batch=32)
  inputs[:, INPUT_FIELDS["joint_target"]] = torch.randn(32, 29)
  delta = torch.zeros(32, DELTA_DIM)
  delta[:, DELTA_FIELDS["joint_velocity"]] = inputs[:, INPUT_FIELDS["joint_target"]] * 0.01
  states = make_state(32)
  target = integrate_state(states, delta)
  forces = torch.zeros(32, 2, 3)
  forces[:, :, 2] = inputs[:, INPUT_FIELDS["joint_target"]][:, :2].abs() + 1
  model.fit_normalization(inputs, states, target, forces)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

  def loss_value():
    result = model(states, inputs[:, INPUT_FIELDS["joint_target"]], inputs[:, INPUT_FIELDS["context"]])
    state_error = state_delta(result.state, target) / model.state_scale
    force_error = (result.foot_force.flatten(-2) - forces.flatten(-2)) / model.force_scale
    return state_error.square().mean() + force_error.square().mean()

  initial = loss_value().item()
  for _ in range(80):
    optimizer.zero_grad()
    loss = loss_value()
    loss.backward()
    optimizer.step()
  final = loss_value().item()
  assert math.isfinite(final) and final < initial * 0.2


@pytest.mark.parametrize(
  ("change", "error"),
  [
    ("short", "145"),
    ("nan", "finite"),
    ("bad_gravity", "unit norm"),
    ("negative_mass", "masses"),
    ("wrong_dtype", "dtype"),
  ],
)
def test_forward_rejects_invalid_inputs(change, error):
  model = make_model()
  inputs = make_inputs()
  if change == "short":
    inputs = inputs[:, :-1]
  elif change == "nan":
    inputs[:, 80] = float("nan")
  elif change == "bad_gravity":
    inputs[:, 1:4] = 0
  elif change == "negative_mass":
    inputs[:, 97] = -1
  else:
    inputs = inputs.double()
  with pytest.raises(ValueError, match=error):
    model.predict_updates(inputs)


def test_pack_rejects_implicit_broadcast_and_state_delta_rejects_dtype_mismatch():
  inputs = make_inputs(batch=2)
  with pytest.raises(ValueError, match="batch shape"):
    pack_inputs(
      make_state(2),
      torch.zeros(1, 29),
      inputs[:, INPUT_FIELDS["context"]],
    )
  with pytest.raises(ValueError, match="dtype"):
    apply_state_delta(make_state(dtype=torch.float32), torch.zeros(3, 70, dtype=torch.float64))


@pytest.mark.parametrize("field", ["hidden_dim", "interaction_dim", "num_blocks"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_config_requires_positive_integer_dimensions(field, value):
  with pytest.raises(ValueError, match="positive integer"):
    PositionWorldModelCfg(**{field: value})


def test_v3_dimensions_and_removed_inputs_are_exact():
  assert (STATE_DIM, STATE_FEATURE_DIM, CONTEXT_DIM, INPUT_DIM, DELTA_DIM, FOOT_FORCE_DIM) == (71, 68, 48, 145, 70, 6)
  assert set(INPUT_FIELDS) == {"state", "joint_target", "context"}
  assert set(CONTEXT_FIELDS) == {"body_mass", "foot_friction", "torso_com"}
  model = make_model()
  raw = model.predict_updates(make_inputs())
  assert raw.as_tensor().shape == (3, 76)
  assert model.get_extra_state()["control_dt"] == 0.02
  assert model.get_extra_state()["contract_version"] == 3


def test_zero_update_is_constant_velocity_and_target_encoding_roundtrips():
  torch.manual_seed(31)
  state = make_state(7, dtype=torch.float64)
  state[:, 3:7] = torch.randn(7, 4, dtype=state.dtype)
  state[:, 3:7] /= state[:, 3:7].norm(dim=-1, keepdim=True)
  state[:, 7:] = torch.randn_like(state[:, 7:])
  zero = torch.zeros(7, 70, dtype=state.dtype)
  advanced = integrate_state(state, zero)
  torch.testing.assert_close(advanced[:, :3], state[:, :3] + CONTROL_DT * state[:, 36:39])
  torch.testing.assert_close(advanced[:, 7:36], state[:, 7:36] + CONTROL_DT * state[:, 42:71])
  torch.testing.assert_close(state_update(state, advanced), zero, atol=1e-14, rtol=0)
  update = torch.randn_like(zero) * 0.03
  torch.testing.assert_close(state_update(state, integrate_state(state, update)), update, atol=1e-14, rtol=1e-12)
  assert torch.autograd.gradcheck(
    lambda value: integrate_state(make_state(1, torch.float64), value),
    (torch.zeros(1, 70, dtype=torch.float64, requires_grad=True),),
  )


def test_horizontal_translation_and_yaw_leave_inputs_invariant_and_outputs_equivariant():
  torch.manual_seed(32)
  state = make_state(4, torch.float64)
  state[:, 3:7] = _rotation_quaternion(torch.randn(4, 3, dtype=state.dtype))
  state[:, 7:] = torch.randn_like(state[:, 7:])
  rotation = _rotation_quaternion(state.new_tensor([[0.0, 0.0, 1.4]]).expand(4, 3))
  shifted = state.clone()
  shifted[:, :3] = root_to_world(rotation, state[:, :3]) + state.new_tensor([12.0, -8.0, 0.0])
  shifted[:, 3:7] = _quat_mul(rotation, state[:, 3:7])
  shifted[:, 36:39] = root_to_world(rotation, state[:, 36:39])
  context = make_inputs(4, torch.float64)[:, INPUT_FIELDS["context"]]
  target = torch.randn(4, 29, dtype=state.dtype)
  torch.testing.assert_close(pack_inputs(state, target, context), pack_inputs(shifted, target, context), atol=1e-14, rtol=1e-12)
  model = make_model(torch.float64)
  first, second = model(state, target, context), model(shifted, target, context)
  torch.testing.assert_close(
    second.state[:, :3], root_to_world(rotation, first.state[:, :3]) + state.new_tensor([12.0, -8.0, 0.0])
  )
  torch.testing.assert_close(second.state[:, 3:7], _quat_mul(rotation, first.state[:, 3:7]))
  torch.testing.assert_close(second.state[:, 36:39], root_to_world(rotation, first.state[:, 36:39]))
  torch.testing.assert_close(first.foot_force_root, second.foot_force_root)
  torch.testing.assert_close(second.foot_force, root_to_world(rotation[:, None, :].expand(4, 2, 4), first.foot_force))


def test_projected_gravity_is_smooth_across_vertical_and_distinguishes_inversion():
  state = make_state(4, torch.float64)
  pitch = state.new_tensor([0, math.radians(89.9), math.radians(90.1), math.pi])
  state[:, 3:7] = _rotation_quaternion(torch.stack((pitch * 0, pitch, pitch * 0), -1))
  features = state_features(state)
  torch.testing.assert_close(features[0, 1:4], state.new_tensor([0, 0, -1]))
  torch.testing.assert_close(features[3, 1:4], state.new_tensor([0, 0, 1]), atol=1e-14, rtol=0)
  assert float((features[1, 1:4] - features[2, 1:4]).norm()) < 0.0035
  model = make_model(torch.float64)
  zero_heads(model)
  with torch.no_grad():
    model.force_head.bias[2] = -10
  inputs = make_inputs(4, torch.float64)
  prediction = model(state, inputs[:, INPUT_FIELDS["joint_target"]], inputs[:, INPUT_FIELDS["context"]])
  assert prediction.foot_force[3, 0, 2] > 9.99


def test_legacy_torque_entry_point_fails_explicitly():
  from g1_lower_rl.world_model import TorqueWorldModel

  with pytest.raises(RuntimeError, match="retired"):
    TorqueWorldModel()
