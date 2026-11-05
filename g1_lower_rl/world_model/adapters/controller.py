"""External action preprocessing and standalone teacher PD/GC utilities.

Position WM v3 receives a held actual position target, never this utility's torque.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from g1_lower_rl.world_model.model import JOINT_NAMES, STATE_FIELDS, TORQUE_DIM

from .kinematics import TorchKinematics
from .math import rotate


def process_position_target(
  action: torch.Tensor,
  scale: torch.Tensor,
  offset: torch.Tensor,
  limits: torch.Tensor | None = None,
  encoder_bias: torch.Tensor | None = None,
  execution_noise: torch.Tensor | None = None,
) -> torch.Tensor:
  """Execution noise -> scale/offset -> clip -> subtract encoder bias, once at50Hz."""
  if action.shape[-1] != TORQUE_DIM or not torch.isfinite(action).all():
    raise ValueError(f"Actions must have {TORQUE_DIM} finite canonical values")
  value = action
  for name, tensor in (
    ("scale", scale),
    ("offset", offset),
    ("encoder_bias", encoder_bias),
    ("execution_noise", execution_noise),
  ):
    if tensor is not None and (
      tensor.shape not in ((TORQUE_DIM,), action.shape)
      or tensor.dtype != action.dtype
      or tensor.device != action.device
      or not torch.isfinite(tensor).all()
    ):
      raise ValueError(f"{name} must be finite, canonical and batch-compatible with actions")
  if execution_noise is not None:
    value = value + execution_noise
  target = value * scale + offset
  if limits is not None:
    if (
      limits.shape not in ((TORQUE_DIM, 2), (*action.shape, 2))
      or not torch.isfinite(limits).all()
      or limits.dtype != action.dtype
      or limits.device != action.device
      or (limits[..., 0] > limits[..., 1]).any()
    ):
      raise ValueError("Target limits must be finite ordered canonical lower/upper pairs")
    target = torch.maximum(torch.minimum(target, limits[..., 1]), limits[..., 0])
  return target if encoder_bias is None else target - encoder_bias


ARM_JOINT_INDICES = tuple(
  i for i, name in enumerate(JOINT_NAMES) if any(part in name for part in ("shoulder_", "elbow_", "wrist_"))
)


class TorchGravityTorque(nn.Module):
  """Physical gravity torque on canonical hinges; compensation has opposite sign."""

  distal_mask: torch.Tensor
  body_com_position: torch.Tensor
  gravity: torch.Tensor
  joint_body_indices: torch.Tensor

  def __init__(self, kinematics: TorchKinematics):
    super().__init__()
    self.kinematics = kinematics
    t = kinematics.template
    self.joint_bodies = tuple(t.body_joint.index(j) for j in range(TORQUE_DIM))
    self.register_buffer("joint_body_indices", torch.tensor(self.joint_bodies), persistent=False)
    mask = torch.zeros(TORQUE_DIM, 44, dtype=torch.float64)
    for j, anchor in enumerate(self.joint_bodies):
      for body in range(44):
        ancestor = body
        while ancestor >= 0 and ancestor != anchor:
          ancestor = t.parents[ancestor]
        mask[j, body] = ancestor == anchor
    self.register_buffer("distal_mask", mask)
    self.register_buffer("body_com_position", torch.tensor(t.body_com_position, dtype=torch.float64))
    self.register_buffer("gravity", torch.tensor(t.gravity, dtype=torch.float64))

  def forward(
    self,
    state: torch.Tensor,
    body_mass: torch.Tensor,
    body_com_position: torch.Tensor | None = None,
    gravity: torch.Tensor | None = None,
    *,
    validate: bool = True,
  ) -> torch.Tensor:
    """Mass [...,44]; optional runtime COM offsets [...,44,3] in each body frame."""
    if validate and (body_mass.shape[-1] != 44 or not torch.isfinite(body_mass).all() or (body_mass < 0).any()):
      raise ValueError("body_mass must contain 44 finite nonnegative canonical masses")
    k = self.kinematics(state, validate=validate)
    com_local = self.body_com_position.to(state) if body_com_position is None else body_com_position.to(state)
    com = k.body_position + rotate(k.body_quaternion, com_local.expand_as(k.body_position))
    ids = self.joint_body_indices
    q = k.body_quaternion[..., ids, :]
    axis = rotate(q, self.kinematics.joint_axis.to(state).expand(*state.shape[:-1], TORQUE_DIM, 3))
    anchor = k.body_position[..., ids, :] + rotate(
      q, self.kinematics.joint_position.to(state).expand(*state.shape[:-1], TORQUE_DIM, 3)
    )
    mass = body_mass.to(state).unsqueeze(-2) * self.distal_mask.to(state)
    weighted_com = torch.einsum("...jb,...bc->...jc", mass, com)
    lever = weighted_com - mass.sum(-1, keepdim=True) * anchor
    g = self.gravity.to(state) if gravity is None else gravity.to(state)
    g = torch.broadcast_to(g, state.shape[:-1] + (3,)).unsqueeze(-2).expand_as(axis)
    return (lever * torch.linalg.cross(g, axis, dim=-1)).sum(-1)


class ControllerOutput(NamedTuple):
  torque: torch.Tensor
  corrected_target: torch.Tensor
  gravity_torque: torch.Tensor
  unsaturated_torque: torch.Tensor
  saturated: torch.Tensor


class SubstepTorqueController(nn.Module):
  """Standalone PD/GC helper; v3 uses only its boundary-corrected target, not torque."""

  nominal_gc_kp: torch.Tensor
  compensated_mask: torch.Tensor

  def __init__(
    self,
    kinematics: TorchKinematics,
    nominal_gc_kp: torch.Tensor,
    compensated_joint_indices: tuple[int, ...] = ARM_JOINT_INDICES,
  ):
    super().__init__()
    if nominal_gc_kp.shape != (TORQUE_DIM,) or not torch.isfinite(nominal_gc_kp).all() or (nominal_gc_kp <= 0).any():
      raise ValueError(f"nominal_gc_kp must contain {TORQUE_DIM} positive finite original pre-randomization gains")
    if len(set(compensated_joint_indices)) != len(compensated_joint_indices) or any(
      i < 0 or i >= TORQUE_DIM for i in compensated_joint_indices
    ):
      raise ValueError("Invalid compensated joint indices")
    self.gravity_torque = TorchGravityTorque(kinematics)
    self.register_buffer("nominal_gc_kp", nominal_gc_kp.detach().clone())
    mask = torch.zeros(TORQUE_DIM, dtype=torch.float64)
    mask[list(compensated_joint_indices)] = 1
    self.register_buffer("compensated_mask", mask)

  @staticmethod
  def process_action(
    action: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor,
    limits: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Once per control tick: action*scale+offset, then optional target clipping.

    All arguments use the caller's policy joint order. Map/scatter into canonical
    joint order before forward; closed grippers have no coordinates or actuators.
    """
    target = action * scale + offset
    if limits is not None:
      target = torch.maximum(torch.minimum(target, limits[..., 1]), limits[..., 0])
    return target

  def forward(
    self,
    state: torch.Tensor,
    processed_target: torch.Tensor,
    effective_kp: torch.Tensor,
    effective_kd: torch.Tensor,
    effort_limits: torch.Tensor,
    body_mass: torch.Tensor,
    encoder_bias: torch.Tensor | None = None,
    gc_gain: torch.Tensor | None = None,
    body_com_position: torch.Tensor | None = None,
    gravity: torch.Tensor | None = None,
    *,
    gravity_state: torch.Tensor | None = None,
    validate: bool = True,
  ) -> ControllerOutput:
    """All joint arrays are canonical; limits [..., joints, 2] can be asymmetric.

    effective_kp/kd are saved runtime randomized gains, while nominal_gc_kp
    corresponds to the original action term's cached _kp used to divide GC.
    Legacy callers may recompute per physical substep. PositionTaskRollout instead
    uses corrected_target once at the control boundary and holds it for20ms.
    """
    for name, tensor in (
      ("processed_target", processed_target),
      ("effective_kp", effective_kp),
      ("effective_kd", effective_kd),
    ):
      if validate and (tensor.shape[-1] != TORQUE_DIM or not torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain {TORQUE_DIM} finite canonical values")
    if validate and ((effective_kp < 0).any() or (effective_kd < 0).any()):
      raise ValueError("Effective PD gains must be nonnegative")
    if validate and (
      effort_limits.shape[-2:] != (TORQUE_DIM, 2)
      or not torch.isfinite(effort_limits).all()
      or (effort_limits[..., 0] > effort_limits[..., 1]).any()
    ):
      raise ValueError(f"Expected finite ordered [...,{TORQUE_DIM},2] effort limits")
    gravity_torque = self.gravity_torque(
      state if gravity_state is None else gravity_state,
      body_mass,
      body_com_position,
      gravity,
      validate=validate,
    )
    gain = torch.ones_like(processed_target) if gc_gain is None else gc_gain
    bias = torch.zeros_like(processed_target) if encoder_bias is None else encoder_bias
    target = processed_target - bias - gravity_torque * gain * self.compensated_mask.to(state) / self.nominal_gc_kp.to(state)
    unsaturated = (
      effective_kp * (target - state[..., STATE_FIELDS["joint_position"]])
      - effective_kd * state[..., STATE_FIELDS["joint_velocity"]]
    )
    torque = torch.maximum(torch.minimum(unsaturated, effort_limits[..., 1]), effort_limits[..., 0])
    return ControllerOutput(torque, target, gravity_torque, unsaturated, torque != unsaturated)
