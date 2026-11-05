"""One50Hz policy action -> actual position target -> WM -> task signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import nn

from g1_lower_rl.world_model.model import (
  BODY_NAMES,
  CONTEXT_DIM,
  CONTEXT_FIELDS,
  CONTROL_DT,
  JOINT_DIM,
  DynamicsPrediction,
  PositionWorldModel,
)

from .contact import ContactMemory, ForceContact
from .controller import SubstepTorqueController, process_position_target
from .tasks import (
  FootstepGoal,
  FootstepMemory,
  FootstepTrackingAdapter,
  LowerBodyVelocityAdapter,
  MotionGoal,
  MotionTrackingAdapter,
  TaskSignals,
  VelocityGoal,
)


@dataclass(frozen=True)
class ControllerContext:
  """Optional external GC setup; none of these fields is a WM network input."""

  effective_kp: torch.Tensor
  effective_kd: torch.Tensor
  effort_limits: torch.Tensor
  encoder_bias: torch.Tensor | None = None
  gc_gain: torch.Tensor | None = None


class ControlStepOutput(NamedTuple):
  prediction: DynamicsPrediction
  signals: TaskSignals
  contact_memory: ContactMemory
  footstep_memory: FootstepMemory | None
  last_action: torch.Tensor
  joint_target: torch.Tensor


class PositionTaskRollout(nn.Module):
  """Caller-owned memories and exactly one held-target20ms transition per call.

  Force-derived contacts remain explicitly calibrated approximations, not native
  contact-found parity. Optional GC is computed once outside the network.
  """

  def __init__(
    self,
    world_model: PositionWorldModel,
    adapter: MotionTrackingAdapter | LowerBodyVelocityAdapter | FootstepTrackingAdapter,
    *,
    controller: SubstepTorqueController | None = None,
    contact: ForceContact | None = None,
  ):
    super().__init__()
    if controller is not None and controller.gravity_torque.kinematics.template != adapter.kinematics.template:
      raise ValueError("External GC and task must share the same static kinematic template")
    self.world_model, self.adapter, self.controller = world_model, adapter, controller
    self.contact = ForceContact() if contact is None else contact

  def forward(
    self,
    state: torch.Tensor,
    action: torch.Tensor,
    goal: MotionGoal | VelocityGoal | FootstepGoal,
    physical_context: torch.Tensor,
    contact_memory: ContactMemory,
    footstep_memory: FootstepMemory | None = None,
    *,
    action_scale: torch.Tensor | None = None,
    action_offset: torch.Tensor | None = None,
    target_limits: torch.Tensor | None = None,
    encoder_bias: torch.Tensor | None = None,
    execution_noise: torch.Tensor | None = None,
    controller_context: ControllerContext | None = None,
    ground_normal: torch.Tensor | None = None,
  ) -> ControlStepOutput:
    if action.shape != (*state.shape[:-1], JOINT_DIM) or physical_context.shape != (*state.shape[:-1], CONTEXT_DIM):
      raise ValueError("Expected batch-matched canonical action29 and physical context48")
    if (self.controller is None) != (controller_context is None):
      raise ValueError("External GC requires both controller and controller_context")
    if controller_context is not None and encoder_bias is not None:
      raise ValueError("Supply encoder bias either directly or in controller_context, not twice")
    scale = torch.ones_like(action) if action_scale is None else action_scale
    offset = torch.zeros_like(action) if action_offset is None else action_offset
    target = process_position_target(action, scale, offset, target_limits, encoder_bias, execution_noise)
    if self.controller is not None and controller_context is not None:
      torso = BODY_NAMES.index("torso_link")
      nominal = self.controller.gravity_torque.body_com_position.to(state).expand(*state.shape[:-1], 44, 3)
      actual_com = torch.cat(
        (
          nominal[..., :torso, :],
          (nominal[..., torso, :] + physical_context[..., CONTEXT_FIELDS["torso_com"]]).unsqueeze(-2),
          nominal[..., torso + 1 :, :],
        ),
        dim=-2,
      )
      control = self.controller(
        state,
        target,
        controller_context.effective_kp,
        controller_context.effective_kd,
        controller_context.effort_limits,
        physical_context[..., CONTEXT_FIELDS["body_mass"]],
        controller_context.encoder_bias,
        controller_context.gc_gain,
        actual_com,
      )
      target = control.corrected_target
    prediction = self.world_model(state, target, physical_context)
    update = self.contact(prediction.foot_force, contact_memory, CONTROL_DT, ground_normal)
    if isinstance(self.adapter, MotionTrackingAdapter) and isinstance(goal, MotionGoal):
      signals = self.adapter(prediction.state, goal, last_action=action)
      next_footstep = None
    elif isinstance(self.adapter, LowerBodyVelocityAdapter) and isinstance(goal, VelocityGoal):
      signals = self.adapter(prediction.state, goal, update, last_action=action)
      next_footstep = None
    elif isinstance(self.adapter, FootstepTrackingAdapter) and isinstance(goal, FootstepGoal):
      memory = FootstepMemory.zeros(goal.targets) if footstep_memory is None else footstep_memory
      output = self.adapter(prediction.state, goal, update, memory, CONTROL_DT, last_action=action)
      signals, next_footstep = output.signals, output.memory
    else:
      raise TypeError("Goal type must match the configured task adapter")
    return ControlStepOutput(prediction, signals, update.memory, next_footstep, action, target)
