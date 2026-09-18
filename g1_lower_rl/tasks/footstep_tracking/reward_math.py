"""纯张量奖励原语，固定左右脚顺序并按完整步态周期计时"""

from __future__ import annotations

import math

import torch

from g1_lower_rl.footstep_phase import FootstepPhaseCfg, resolve_phase_cfg


def phase_windows(
  phase: torch.Tensor, stance_fraction: float | None = None, *, phase_cfg: FootstepPhaseCfg | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
  """根据共用接触时序返回支撑掩码和摆动进度"""
  cfg = resolve_phase_cfg(stance_fraction, phase_cfg)
  contact_start = phase.new_tensor(cfg.contact_start_rad)
  stance_durations = phase.new_tensor(cfg.stance_fractions)
  progress = torch.remainder(phase.unsqueeze(-1) - contact_start, 2 * math.pi) / (2 * math.pi)
  wrapped = torch.remainder(phase, 2 * math.pi).unsqueeze(-1)
  contact_start = torch.remainder(contact_start, 2 * math.pi)
  liftoff = torch.remainder(phase.new_tensor(cfg.liftoff_rad), 2 * math.pi)
  stance = torch.where(
    liftoff > contact_start,
    (wrapped >= contact_start) & (wrapped < liftoff),
    (wrapped >= contact_start) | (wrapped < liftoff),
  )
  swing_progress = ((progress - stance_durations) / (1.0 - stance_durations)).clamp(0.0, 1.0)
  return stance, swing_progress


def torque_square_cost(torques: torch.Tensor, weights: torch.Tensor, reference_torque: float = 100.0) -> torch.Tensor:
  """用统一尺度计算加权力矩平方，不按各轴峰值额定力矩归一化"""
  if not math.isfinite(reference_torque) or reference_torque <= 0:
    raise ValueError("reference_torque must be finite and positive")
  return ((torques / reference_torque).square() * weights).sum(dim=-1)


def joint_edge_cost(positions: torch.Tensor, limits: torch.Tensor, margin_fraction: float = 0.15) -> torch.Tensor:
  """内部安全区为零，每个关节到达硬边界时代价为一，越界后继续增大"""
  if not 0.0 < margin_fraction < 0.5:
    raise ValueError("margin_fraction must lie in (0, 0.5)")
  lower, upper = limits.unbind(-1)
  margin = (upper - lower) * margin_fraction
  lower_cost = ((lower + margin - positions) / margin).clamp_min(0).square()
  upper_cost = ((positions - upper + margin) / margin).clamp_min(0).square()
  return (lower_cost + upper_cost).sum(dim=-1)


def footprint_errors(actual: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """计算按左右脚排列的 XY/yaw 位姿之间的平面距离和最短航向角差"""
  distance = torch.linalg.vector_norm(actual[..., :2] - target[..., :2], dim=-1)
  yaw_delta = actual[..., 2] - target[..., 2]
  yaw_error = torch.atan2(yaw_delta.sin(), yaw_delta.cos())
  return distance, yaw_error


def footprint_accuracy_cost(
  distance: torch.Tensor,
  yaw_error: torch.Tensor,
  position_scale: float,
  yaw_scale: float,
) -> torch.Tensor:
  """等权组合线性位置误差与最短角差，不截断代价"""
  if not all(math.isfinite(value) and value > 0 for value in (position_scale, yaw_scale)):
    raise ValueError("Tracking scales must be finite and positive")
  wrapped_yaw = torch.atan2(yaw_error.sin(), yaw_error.cos())
  return 0.5 * distance / position_scale + 0.5 * wrapped_yaw.abs() / yaw_scale


def swing_tracking_score(error: torch.Tensor, stance: torch.Tensor, std: float = 0.1) -> torch.Tensor:
  """对计划摆动脚计算指数精度奖励，不乘摆动进度斜坡"""
  if not math.isfinite(std) or std <= 0:
    raise ValueError("Swing tracking std must be finite and positive")
  return (torch.exp(-(error / std).square()) * ~stance).sum(-1)


def contact_schedule_score(required_contact: torch.Tensor, contact: torch.Tensor) -> torch.Tensor:
  """所有脚的实际接触都符合计划模式时才给满分"""
  return (required_contact == contact).all(dim=-1).to(torch.float32)
