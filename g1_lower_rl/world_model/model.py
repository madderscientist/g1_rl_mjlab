"""50 Hz position-target dynamics with gravity-based, root-frame features."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import NamedTuple

import torch
from torch import nn

STATE_DIM = 71
STATE_FEATURE_DIM = 68
DELTA_DIM = 70
JOINT_DIM = 29
TORQUE_DIM = JOINT_DIM  # Also used by the standalone legacy torque-controller utility.
CONTEXT_DIM = 48
FOOT_FORCE_DIM = 6
INPUT_DIM = 145
OUTPUT_DIM = DELTA_DIM + FOOT_FORCE_DIM
CONTROL_DT = 0.02
CONTRACT_VERSION = 3

STATE_FIELDS = {
  "root_position": slice(0, 3),
  "root_quaternion": slice(3, 7),
  "joint_position": slice(7, 36),
  "root_linear_velocity": slice(36, 39),
  "root_angular_velocity": slice(39, 42),
  "joint_velocity": slice(42, 71),
}
DELTA_FIELDS = {
  "root_position": slice(0, 3),
  "root_rotation": slice(3, 6),
  "joint_position": slice(6, 35),
  "root_linear_velocity": slice(35, 38),
  "root_angular_velocity": slice(38, 41),
  "joint_velocity": slice(41, 70),
}
INPUT_FIELDS = {
  "state": slice(0, 68),
  "joint_target": slice(68, 97),
  "context": slice(97, 145),
}
STATE_FEATURE_FIELDS = {
  "root_height": slice(0, 1),
  "projected_gravity": slice(1, 4),
  "root_linear_velocity": slice(4, 7),
  "root_angular_velocity": slice(7, 10),
  "joint_position": slice(10, 39),
  "joint_velocity": slice(39, 68),
}
CONTEXT_FIELDS = {
  "body_mass": slice(0, 44),
  "foot_friction": slice(44, 45),
  "torso_com": slice(45, 48),
}
JOINT_NAMES = (
  *(
    f"{side}_{name}_joint"
    for side in ("left", "right")
    for name in (
      "hip_pitch",
      "hip_roll",
      "hip_yaw",
      "knee",
      "ankle_pitch",
      "ankle_roll",
    )
  ),
  *(f"waist_{axis}_joint" for axis in ("yaw", "roll", "pitch")),
  *(
    name
    for side in ("left", "right")
    for name in (
      *(
        f"{side}_{joint}_joint"
        for joint in (
          "shoulder_pitch",
          "shoulder_roll",
          "shoulder_yaw",
          "elbow",
          "wrist_roll",
          "wrist_pitch",
          "wrist_yaw",
        )
      ),
    )
  ),
)
BODY_NAMES = (
  "pelvis",
  *(
    f"{side}_{name}_link"
    for side in ("left", "right")
    for name in (
      "hip_pitch",
      "hip_roll",
      "hip_yaw",
      "knee",
      "ankle_pitch",
      "ankle_roll",
    )
  ),
  "waist_yaw_link",
  "waist_roll_link",
  "torso_link",
  *(
    name
    for side in ("left", "right")
    for name in (
      *(
        f"{side}_{joint}_link"
        for joint in (
          "shoulder_pitch",
          "shoulder_roll",
          "shoulder_yaw",
          "elbow",
          "wrist_roll",
          "wrist_pitch",
          "wrist_yaw",
          "kwr57b",
        )
      ),
      f"{side}_gripper_base",
      f"{side}_eccentric",
      f"{side}_gloria_m_left_slider",
      f"{side}_gloria_m_right_slider",
      f"{side}_gloria_m_left_rod",
      f"{side}_gloria_m_right_rod",
    )
  ),
)


def _check_tensor(value: torch.Tensor, width: int, name: str) -> None:
  if value.ndim < 1 or value.shape[-1] != width:
    raise ValueError(f"{name} must have shape [..., {width}], got {tuple(value.shape)}")
  if not value.is_floating_point():
    raise TypeError(f"{name} must be floating point")
  if not torch.isfinite(value).all():
    raise ValueError(f"{name} must be finite")


def _check_matching(reference: torch.Tensor, value: torch.Tensor, name: str) -> None:
  if value.shape[:-1] != reference.shape[:-1]:
    raise ValueError(f"{name} batch shape must match {tuple(reference.shape[:-1])}, got {tuple(value.shape[:-1])}")
  if value.device != reference.device or value.dtype != reference.dtype:
    raise ValueError(f"{name} must share dtype and device with the reference tensor")


def _unit_quaternion(quaternion: torch.Tensor, validate: bool = True) -> torch.Tensor:
  norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
  if validate and (not torch.isfinite(norm).all() or (norm <= torch.finfo(quaternion.dtype).eps).any()):
    raise ValueError("Root quaternion must have a finite, nonzero norm")
  return quaternion / norm


def _quat_mul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
  lw, lv = left[..., :1], left[..., 1:]
  rw, rv = right[..., :1], right[..., 1:]
  return torch.cat((lw * rw - (lv * rv).sum(-1, keepdim=True), lw * rv + rw * lv + torch.linalg.cross(lv, rv, dim=-1)), dim=-1)


def _conjugate(quaternion: torch.Tensor) -> torch.Tensor:
  return torch.cat((quaternion[..., :1], -quaternion[..., 1:]), dim=-1)


def _rotate(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
  cross = torch.linalg.cross(quaternion[..., 1:], vector, dim=-1)
  return vector + 2 * (quaternion[..., :1] * cross + torch.linalg.cross(quaternion[..., 1:], cross, dim=-1))


def world_to_root(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
  """Rotate vectors using a wxyz body-to-world quaternion; supports foot axes."""
  return _rotate(_conjugate(_unit_quaternion(quaternion, validate=False)), vector)


def root_to_world(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
  return _rotate(_unit_quaternion(quaternion, validate=False), vector)


def _rotation_quaternion(rotation: torch.Tensor) -> torch.Tensor:
  angle = torch.linalg.vector_norm(rotation, dim=-1, keepdim=True)
  # sinc gives the analytic zero-angle limit without dividing by |rotation|.
  return torch.cat((torch.cos(angle / 2), 0.5 * torch.sinc(angle / (2 * math.pi)) * rotation), dim=-1)


def _apply_delta(state: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
  before = _unit_quaternion(state[..., 3:7], validate=False)
  quaternion = _quat_mul(before, _rotation_quaternion(delta[..., 3:6]))
  return torch.cat(
    (
      state[..., :3] + _rotate(before, delta[..., :3]),
      quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True),
      state[..., 7:36] + delta[..., 6:35],
      state[..., 36:39] + _rotate(before, delta[..., 35:38]),
      state[..., 39:] + delta[..., 38:],
    ),
    dim=-1,
  )


def apply_state_delta(state: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
  """Geometric delta utility, NOT the learned integrator; root increments are local."""
  _check_tensor(state, STATE_DIM, "state")
  _check_tensor(delta, DELTA_DIM, "delta")
  _check_matching(state, delta, "delta")
  _unit_quaternion(state[..., 3:7])
  return _apply_delta(state, delta)


def state_delta(state: torch.Tensor, next_state: torch.Tensor, *, validate: bool = True) -> torch.Tensor:
  """Physical state difference in the first state's root frame, with shortest rotation."""
  if validate:
    _check_tensor(state, STATE_DIM, "state")
    _check_tensor(next_state, STATE_DIM, "next_state")
    _check_matching(state, next_state, "next_state")
  before = _unit_quaternion(state[..., 3:7], validate=validate)
  after = _unit_quaternion(next_state[..., 3:7], validate=validate)
  relative = _unit_quaternion(_quat_mul(_conjugate(before), after), validate=False)
  relative = torch.where(relative[..., :1] < 0, -relative, relative)
  vector = relative[..., 1:]
  sine = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
  regular = 2 * torch.atan2(sine, relative[..., :1]) / sine.clamp_min(1e-4)
  small = 2 + sine.square() / 3 + 3 * sine.pow(4) / 20
  rotation = vector * torch.where(sine > 1e-4, regular, small)
  return torch.cat(
    (
      _rotate(_conjugate(before), next_state[..., :3] - state[..., :3]),
      rotation,
      next_state[..., 7:36] - state[..., 7:36],
      _rotate(_conjugate(before), next_state[..., 36:39] - state[..., 36:39]),
      next_state[..., 39:] - state[..., 39:],
    ),
    dim=-1,
  )


def state_features(state: torch.Tensor, *, validate: bool = True) -> torch.Tensor:
  if validate:
    _check_tensor(state, STATE_DIM, "state")
  quaternion = _unit_quaternion(state[..., 3:7], validate=validate)
  gravity = torch.zeros_like(state[..., :3])
  gravity[..., 2] = -1
  return torch.cat(
    (
      state[..., 2:3],
      _rotate(_conjugate(quaternion), gravity),
      _rotate(_conjugate(quaternion), state[..., 36:39]),
      state[..., 39:42],
      state[..., 7:36],
      state[..., 42:71],
    ),
    dim=-1,
  )


def state_update(state: torch.Tensor, next_state: torch.Tensor, *, validate: bool = True) -> torch.Tensor:
  """Supervision for velocity changes and semi-implicit integration corrections."""
  difference = state_delta(state, next_state, validate=validate)
  return torch.cat(
    (
      difference[..., :3] - CONTROL_DT * world_to_root(state[..., 3:7], next_state[..., 36:39]),
      difference[..., 3:6] - CONTROL_DT * next_state[..., 39:42],
      difference[..., 6:35] - CONTROL_DT * next_state[..., 42:71],
      difference[..., 35:],
    ),
    dim=-1,
  )


def integrate_state(state: torch.Tensor, update: torch.Tensor, *, validate: bool = True) -> torch.Tensor:
  """Advance exactly one 20ms control tick; zero update means constant velocity."""
  if validate:
    _check_tensor(state, STATE_DIM, "state")
    _check_tensor(update, DELTA_DIM, "update")
    _check_matching(state, update, "update")
  quaternion = _unit_quaternion(state[..., 3:7], validate=validate)
  linear = state[..., 36:39] + _rotate(quaternion, update[..., 35:38])
  angular = state[..., 39:42] + update[..., 38:41]
  joint_velocity = state[..., 42:71] + update[..., 41:70]
  orientation = _quat_mul(quaternion, _rotation_quaternion(CONTROL_DT * angular + update[..., 3:6]))
  return torch.cat(
    (
      state[..., :3] + CONTROL_DT * linear + _rotate(quaternion, update[..., :3]),
      _unit_quaternion(orientation, validate=False),
      state[..., 7:36] + CONTROL_DT * joint_velocity + update[..., 6:35],
      linear,
      angular,
      joint_velocity,
    ),
    dim=-1,
  )


def pack_inputs(
  state: torch.Tensor,
  joint_target: torch.Tensor,
  context: torch.Tensor,
  *,
  validate: bool = True,
) -> torch.Tensor:
  """Pack 68 invariant features, 29 actual position targets and 48 context values."""
  blocks = (
    ("joint_target", joint_target, JOINT_DIM),
    ("context", context, CONTEXT_DIM),
  )
  if validate:
    for name, value, width in blocks:
      _check_tensor(value, width, name)
      _check_matching(state, value, name)
    if (context[..., :45] < 0).any() or (context[..., :44].sum(-1) <= 0).any():
      raise ValueError("Body masses and shared friction must be nonnegative, with positive total mass")
  return torch.cat((state_features(state, validate=validate), joint_target, context), dim=-1)


def _canonical_inputs(inputs: torch.Tensor, validate: bool = True) -> torch.Tensor:
  if validate:
    _check_tensor(inputs, INPUT_DIM, "inputs")
    if not torch.allclose(inputs[..., 1:4].norm(dim=-1), torch.ones_like(inputs[..., 0]), atol=1e-4, rtol=1e-4):
      raise ValueError("Projected gravity must have unit norm")
    context = inputs[..., INPUT_FIELDS["context"]]
    if (context[..., :45] < 0).any() or (context[..., :44].sum(-1) <= 0).any():
      raise ValueError("Invalid body masses or shared friction")
  return inputs


@dataclass(frozen=True)
class PositionWorldModelCfg:
  hidden_dim: int = 2048
  interaction_dim: int = 1024
  num_blocks: int = 3

  def __post_init__(self) -> None:
    for name, value in asdict(self).items():
      if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


class DynamicsPrediction(NamedTuple):
  state: torch.Tensor
  foot_force_root: torch.Tensor

  @property
  def foot_force(self) -> torch.Tensor:
    """Task-facing ground-to-foot force, world frame at the predicted endpoint."""
    quaternion = self.state[..., 3:7].unsqueeze(-2).expand(*self.foot_force_root.shape[:-1], 4)
    return root_to_world(quaternion, self.foot_force_root)

  def as_tensor(self) -> torch.Tensor:
    """Return state followed by left XYZ then right XYZ foot force."""
    return torch.cat((self.state, self.foot_force.flatten(-2)), dim=-1)


class DynamicsUpdate(NamedTuple):
  delta: torch.Tensor
  foot_force_root: torch.Tensor

  def as_tensor(self) -> torch.Tensor:
    return torch.cat((self.delta, self.foot_force_root.flatten(-2)), dim=-1)


class GatedResidualBlock(nn.Module):
  """Pre-LN SwiGLU-style block; two learned projections interact elementwise."""

  def __init__(self, hidden_dim: int = 2048, interaction_dim: int = 1024) -> None:
    super().__init__()
    if hidden_dim <= 0 or interaction_dim <= 0:
      raise ValueError("Hidden and interaction dimensions must be positive")
    self.norm = nn.LayerNorm(hidden_dim)
    self.gate = nn.Linear(hidden_dim, interaction_dim)
    self.value = nn.Linear(hidden_dim, interaction_dim)
    self.projection = nn.Linear(interaction_dim, hidden_dim)

  def forward(self, hidden: torch.Tensor) -> torch.Tensor:
    normalized = self.norm(hidden)
    return hidden + self.projection(torch.nn.functional.silu(self.gate(normalized)) * self.value(normalized))


class PositionWorldModel(nn.Module):
  """A fixed-20ms position-control transition; only pack_inputs reaches the network."""

  input_mean: torch.Tensor
  input_scale: torch.Tensor
  delta_scale: torch.Tensor
  force_scale: torch.Tensor
  state_scale: torch.Tensor
  normalization_ready: torch.Tensor

  def __init__(self, cfg: PositionWorldModelCfg | None = None) -> None:
    super().__init__()
    self.cfg = cfg if cfg is not None else PositionWorldModelCfg()
    self.embedding = nn.Linear(INPUT_DIM, self.cfg.hidden_dim)
    self.blocks = nn.Sequential(
      *(GatedResidualBlock(self.cfg.hidden_dim, self.cfg.interaction_dim) for _ in range(self.cfg.num_blocks))
    )
    self.delta_head = nn.Linear(self.cfg.hidden_dim, DELTA_DIM)
    self.force_head = nn.Linear(self.cfg.hidden_dim, FOOT_FORCE_DIM)
    for head in (self.delta_head, self.force_head):
      nn.init.normal_(head.weight, std=1e-3)
      nn.init.zeros_(head.bias)
    self.register_buffer("input_mean", torch.zeros(INPUT_DIM))
    self.register_buffer("input_scale", torch.ones(INPUT_DIM))
    self.register_buffer("delta_scale", torch.ones(DELTA_DIM))
    self.register_buffer("force_scale", torch.ones(FOOT_FORCE_DIM))
    self.register_buffer("state_scale", torch.ones(DELTA_DIM))
    self.register_buffer("normalization_ready", torch.tensor(False))

  @torch.no_grad()
  def set_normalization(
    self,
    input_mean: torch.Tensor,
    input_scale: torch.Tensor,
    delta_scale: torch.Tensor,
    force_scale: torch.Tensor,
    state_scale: torch.Tensor,
  ) -> None:
    """Set fixed statistics; increments and forces use scale only, not centering."""
    stats = (
      ("input_mean", input_mean, INPUT_DIM),
      ("input_scale", input_scale, INPUT_DIM),
      ("delta_scale", delta_scale, DELTA_DIM),
      ("force_scale", force_scale, FOOT_FORCE_DIM),
      ("state_scale", state_scale, DELTA_DIM),
    )
    for name, value, width in stats:
      _check_tensor(value, width, name)
      if value.ndim != 1:
        raise ValueError(f"{name} must be a vector")
      if name != "input_mean" and (value <= 0).any():
        raise ValueError(f"{name} must be positive")
      converted = value.to(self.input_mean)
      if not torch.isfinite(converted).all() or (name != "input_mean" and (converted <= 0).any()):
        raise ValueError(f"{name} is not representable in model dtype")
    for name, value, _ in stats:
      buffer = self.get_buffer(name)
      buffer.copy_(value)
    self.normalization_ready.fill_(True)

  @torch.no_grad()
  def fit_normalization(
    self,
    inputs: torch.Tensor,
    states: torch.Tensor,
    next_states: torch.Tensor,
    foot_forces: torch.Tensor,
    min_scale: float = 1e-6,
  ) -> None:
    """Fit once on aligned training transitions, never on validation data or live rollout."""
    if not math.isfinite(min_scale) or min_scale <= 0:
      raise ValueError("min_scale must be positive and finite")
    canonical = _canonical_inputs(inputs)
    if inputs.ndim != 2 or inputs.shape[0] == 0:
      raise ValueError(f"Normalization requires a nonempty [samples, {INPUT_DIM}] batch")
    _check_tensor(next_states, STATE_DIM, "next_states")
    _check_tensor(states, STATE_DIM, "states")
    _check_matching(inputs, states, "states")
    _check_matching(inputs, next_states, "next_states")
    if not torch.allclose(canonical[..., :STATE_FEATURE_DIM], state_features(states), atol=1e-5, rtol=1e-5):
      raise ValueError("Packed features do not match the separately stored full states")
    if foot_forces.shape != (inputs.shape[0], 2, 3):
      raise ValueError("foot_forces must have shape [samples, 2, 3]")
    forces = foot_forces.flatten(-2)
    _check_tensor(forces, FOOT_FORCE_DIM, "foot_forces")
    _check_matching(inputs, forces, "foot_forces")
    deltas = state_update(states, next_states)
    quaternion = next_states[..., 3:7].unsqueeze(-2).expand(*foot_forces.shape[:-1], 4)
    local_forces = world_to_root(quaternion, foot_forces).flatten(-2)
    self.set_normalization(
      canonical.mean(dim=0),
      canonical.std(dim=0, correction=0).clamp_min(min_scale),
      deltas.square().mean(dim=0).sqrt().clamp_min(min_scale),
      local_forces.square().mean(dim=0).sqrt().clamp_min(min_scale),
      state_delta(states, next_states).square().mean(dim=0).sqrt().clamp_min(min_scale),
    )

  def predict_updates(self, inputs: torch.Tensor, *, validate: bool = True) -> DynamicsUpdate:
    if validate and not self.normalization_ready:
      raise RuntimeError("Set or fit world-model normalization before inference or training")
    if validate and (inputs.device != self.input_mean.device or inputs.dtype != self.input_mean.dtype):
      raise ValueError("Inputs must share model dtype and device")
    canonical = _canonical_inputs(inputs, validate=validate)
    hidden = self.blocks(self.embedding((canonical - self.input_mean) / self.input_scale))
    delta = self.delta_head(hidden) * self.delta_scale
    force = self.force_head(hidden) * self.force_scale
    return DynamicsUpdate(delta, force.unflatten(-1, (2, 3)))

  def forward(
    self,
    state: torch.Tensor,
    joint_target: torch.Tensor,
    context: torch.Tensor,
    *,
    validate: bool = True,
  ) -> DynamicsPrediction:
    update = self.predict_updates(pack_inputs(state, joint_target, context, validate=validate), validate=validate)
    return DynamicsPrediction(integrate_state(state, update.delta, validate=validate), update.foot_force_root)

  def get_extra_state(self) -> dict[str, object]:
    return {
      "contract_version": CONTRACT_VERSION,
      "config": asdict(self.cfg),
      "joint_names": JOINT_NAMES,
      "body_names": BODY_NAMES,
      "input_fields": {name: (part.start, part.stop) for name, part in INPUT_FIELDS.items()},
      "context_fields": {name: (part.start, part.stop) for name, part in CONTEXT_FIELDS.items()},
      "state_fields": {name: (part.start, part.stop) for name, part in STATE_FIELDS.items()},
      "state_feature_fields": {name: (part.start, part.stop) for name, part in STATE_FEATURE_FIELDS.items()},
      "delta_fields": {name: (part.start, part.stop) for name, part in DELTA_FIELDS.items()},
      "root_velocity_frames": ("world_linear", "body_angular"),
      "rotation_increment": "body_frame_right_composed",
      "foot_force": "left_then_right_root_at_next_state_ground_to_foot",
      "force_label": "final_physics_substep_contact_solution_not_control_average",
      "position_and_linear_velocity_updates": "current_root_frame",
      "context_com": "torso_local_offset_from_nominal",
      "action": "actual_position_target_held_for_control_tick",
      "control_dt": CONTROL_DT,
      "integration": "velocity_update_semi_implicit_plus_pose_correction",
    }

  def set_extra_state(self, state: object) -> None:
    if state != self.get_extra_state():
      raise RuntimeError("World-model checkpoint architecture or physical interface does not match this model")


TorqueWorldModelCfg = PositionWorldModelCfg


class TorqueWorldModel(PositionWorldModel):
  """Import-compatible tombstone for old callers; never reinterpret torque as position."""

  def __init__(self, cfg: PositionWorldModelCfg | None = None) -> None:
    raise RuntimeError(
      "The torque/substep WM interface is retired. Use PositionWorldModel with contract v3: "
      "state71, actual position target29, context48, fixed20ms. Legacy weights/data cannot resume."
    )
