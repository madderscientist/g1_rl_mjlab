"""依据刚完成控制拍的脚步执行快照检查终止条件"""

import math

import torch

from g1_lower_rl.tasks.footstep_tracking.reward_math import phase_windows


def footstep_distance_exceeded(env, command_name: str = "footsteps", max_distance: float = 1.0) -> torch.Tensor:
  """任一计划支撑脚距当前执行目标的 XY 距离过大时终止"""
  if not math.isfinite(max_distance) or max_distance <= 0:
    raise ValueError("max_distance must be finite and positive")
  command = env.command_manager.get_term(command_name)
  state = command.reward_state
  stance, _ = phase_windows(state.phase, phase_cfg=command.cfg.manager.phase)
  required = stance | (state.frequency == 0).unsqueeze(-1)
  positions = command.robot.data.site_pos_w[:, command.site_ids, :2]
  distance = torch.linalg.vector_norm(positions - state.targets_w[..., :2], dim=-1)
  return ((distance > max_distance) & required).any(dim=-1)