"""Tensor-only reward primitives with left/right ordering and cycle-based timing."""

from __future__ import annotations

import math

import torch

from g1_lower_rl.footstep_phase import FootstepPhaseCfg, resolve_phase_cfg


def phase_windows(
  phase: torch.Tensor, stance_fraction: float | None = None, *, phase_cfg: FootstepPhaseCfg | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return stance masks and swing progress from the shared contact schedule."""
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
  """Weighted torque squared, using one common scale rather than per-axis peak ratings."""
  if not math.isfinite(reference_torque) or reference_torque <= 0:
    raise ValueError("reference_torque must be finite and positive")
  return ((torques / reference_torque).square() * weights).sum(dim=-1)


def joint_edge_cost(positions: torch.Tensor, limits: torch.Tensor, margin_fraction: float = 0.15) -> torch.Tensor:
  """Zero in the interior, one per joint at a hard bound, increasing beyond it."""
  if not 0.0 < margin_fraction < 0.5:
    raise ValueError("margin_fraction must lie in (0, 0.5)")
  lower, upper = limits.unbind(-1)
  margin = (upper - lower) * margin_fraction
  lower_cost = ((lower + margin - positions) / margin).clamp_min(0).square()
  upper_cost = ((positions - upper + margin) / margin).clamp_min(0).square()
  return (lower_cost + upper_cost).sum(dim=-1)


def footprint_errors(actual: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """Planar distance and wrapped yaw error for [..., left/right, x/y/yaw] poses."""
  distance = torch.linalg.vector_norm(actual[..., :2] - target[..., :2], dim=-1)
  yaw_delta = actual[..., 2] - target[..., 2]
  yaw_error = torch.atan2(yaw_delta.sin(), yaw_delta.cos())
  return distance, yaw_error


def footprint_accuracy_score(
  distance: torch.Tensor,
  yaw_error: torch.Tensor,
  position_std: float,
  yaw_std: float,
) -> torch.Tensor:
  """平面位置与朝向各自评分后等权相加，单脚最高分保持为1"""
  if not all(math.isfinite(value) and value > 0 for value in (position_std, yaw_std)):
    raise ValueError("Tracking standard deviations must be finite and positive")
  position_score = torch.exp(-(distance / position_std).square())
  yaw_score = torch.exp(-(yaw_error / yaw_std).square())
  return 0.5 * position_score + 0.5 * yaw_score


def footprint_tracking_score(
  actual: torch.Tensor,
  target: torch.Tensor,
  required_contact: torch.Tensor,
  contact: torch.Tensor,
  position_std: float = 0.05,
  yaw_std: float = 0.15,
) -> torch.Tensor:
  """支撑脚的位置与朝向独立评分，未接触的脚不计分"""
  distance, yaw_error = footprint_errors(actual, target)
  score = footprint_accuracy_score(distance, yaw_error, position_std, yaw_std)
  count = required_contact.sum(dim=-1).clamp_min(1)
  return (score * required_contact * contact).sum(dim=-1) / count


def contact_schedule_score(required_contact: torch.Tensor, contact: torch.Tensor) -> torch.Tensor:
  """Full credit requires all feet to follow the scheduled contact mode."""
  return (required_contact == contact).all(dim=-1).to(torch.float32)
