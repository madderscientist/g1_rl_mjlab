"""Native motion-tracking rewards."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.tracking.mdp.commands import MotionCommand


def motion_anchor_pos_xy_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float = 1.2,
) -> torch.Tensor:
  command = cast("MotionCommand", env.command_manager.get_term(command_name))
  error = (command.anchor_pos_w[..., :2] - command.robot_anchor_pos_w[..., :2]).square().sum(-1)
  return torch.exp(-error / std**2)


def motion_anchor_lin_vel_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float = 0.4,
) -> torch.Tensor:
  """Track anchor velocity separately so small persistent errors do not accumulate drift."""
  command = cast("MotionCommand", env.command_manager.get_term(command_name))
  i = command.motion_anchor_body_index
  err = torch.sum(
    (command.body_lin_vel_w[:, i] - command.robot_body_lin_vel_w[:, i]).square(), dim=-1
  )
  return torch.exp(-err / std**2)
