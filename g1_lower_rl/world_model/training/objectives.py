"""Causal autoregression with logged actual position targets at fixed20ms."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from g1_lower_rl.world_model.model import (
  CONTROL_DT,
  DELTA_DIM,
  DELTA_FIELDS,
  INPUT_FIELDS,
  DynamicsPrediction,
  integrate_state,
  state_delta,
  world_to_root,
)
from g1_lower_rl.world_model.training.replay import ReplayWindow


@dataclass(frozen=True)
class LossConfig:
  gamma: float = 0.99
  gamma_reference_dt: float = CONTROL_DT
  force_weight: float = 1.0
  zero_force_weight: float = 1.0
  zero_force_threshold: float = 1e-3
  nonnegative_fz_weight: float = 0.01
  contact_threshold: float = 10.0

  def __post_init__(self) -> None:
    if not all(
      math.isfinite(value)
      for value in (
        self.gamma,
        self.gamma_reference_dt,
        self.force_weight,
        self.zero_force_weight,
        self.zero_force_threshold,
        self.nonnegative_fz_weight,
        self.contact_threshold,
      )
    ):
      raise ValueError("Loss configuration must be finite")
    if not 0 < self.gamma <= 1 or self.gamma_reference_dt <= 0:
      raise ValueError("gamma must be in (0, 1] and reference dt must be positive")
    if min(self.force_weight, self.zero_force_weight, self.zero_force_threshold, self.nonnegative_fz_weight) < 0:
      raise ValueError("Force loss weights and threshold cannot be negative")
    if self.contact_threshold <= 0:
      raise ValueError("Contact threshold must be positive")


def autoregressive_loss(
  forward: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], DynamicsPrediction],
  window: ReplayWindow,
  delta_scale: torch.Tensor,
  force_scale: torch.Tensor,
  cfg: LossConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
  state = window.state[:, 0]
  elapsed = torch.zeros(state.shape[0], device=state.device)
  state_sum = torch.zeros_like(elapsed)
  force_sum = torch.zeros_like(elapsed)
  zero_sum = torch.zeros_like(elapsed)
  negative_fz_sum = torch.zeros_like(elapsed)
  weight_sum = torch.zeros_like(elapsed)
  for step in range(window.inputs.shape[1]):
    logged = window.inputs[:, step]
    prediction = forward(state, logged[:, INPUT_FIELDS["joint_target"]], logged[:, INPUT_FIELDS["context"]])
    error = state_delta(prediction.state, window.next_state[:, step], validate=False) / delta_scale
    quaternion = window.next_state[:, step, 3:7].unsqueeze(-2).expand(*window.foot_force[:, step].shape[:-1], 4)
    target_force = world_to_root(quaternion, window.foot_force[:, step])
    force_error = (prediction.foot_force_root - target_force).flatten(-2) / force_scale
    zero = window.foot_force[:, step].norm(dim=-1) <= cfg.zero_force_threshold
    zero_error = (force_error.unflatten(-1, (2, 3)).square().mean(-1) * zero).sum(-1) / zero.sum(-1).clamp_min(1)
    normal_scale = force_scale.unflatten(-1, (2, 3)).square().mean(-1).sqrt().clamp_min(1e-6)
    negative_fz = prediction.foot_force[..., 2].clamp_max(0) / normal_scale
    weight = cfg.gamma ** (elapsed / cfg.gamma_reference_dt)
    state_sum = state_sum + weight * error.square().mean(-1)
    force_sum = force_sum + weight * force_error.square().mean(-1)
    zero_sum = zero_sum + weight * zero_error
    negative_fz_sum = negative_fz_sum + weight * negative_fz.square().mean(-1)
    weight_sum = weight_sum + weight
    elapsed = elapsed + CONTROL_DT
    state = prediction.state
  state_loss = (state_sum / weight_sum).mean()
  force_loss = (force_sum / weight_sum).mean()
  zero_loss = (zero_sum / weight_sum).mean()
  negative_fz_loss = (negative_fz_sum / weight_sum).mean()
  total = (
    state_loss
    + cfg.force_weight * force_loss
    + cfg.zero_force_weight * zero_loss
    + cfg.nonnegative_fz_weight * negative_fz_loss
  )
  return total, {
    "state_loss": state_loss.detach(),
    "force_loss": force_loss.detach(),
    "zero_force_loss": zero_loss.detach(),
    "nonnegative_fz_loss": negative_fz_loss.detach(),
  }


@torch.no_grad()
def physical_metric_sums(
  forward: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], DynamicsPrediction],
  window: ReplayWindow,
  zero_force_threshold: float = 1e-3,
  evaluation_steps: dict[str, int] | None = None,
  contact_threshold: float = 10.0,
  delta_scale: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
  """Raw sums/counts can be reduced across independent held-out trajectories."""
  sums: dict[str, torch.Tensor] = {}
  state = window.state[:, 0]
  baseline_state = state
  horizon = window.inputs.shape[1]
  evaluation_steps = {f"rollout_{horizon}": horizon} if evaluation_steps is None else evaluation_steps
  if any(not 1 <= count <= horizon for count in evaluation_steps.values()):
    raise ValueError("Evaluation endpoints must lie within the sampled horizon")
  for step in range(horizon):
    logged = window.inputs[:, step]
    prediction = forward(state, logged[:, INPUT_FIELDS["joint_target"]], logged[:, INPUT_FIELDS["context"]])
    state = prediction.state
    baseline_state = integrate_state(
      baseline_state, baseline_state.new_zeros(*baseline_state.shape[:-1], DELTA_DIM), validate=False
    )
    labels = (["one_step"] if step == 0 else []) + [
      label for label, count in evaluation_steps.items() if count == step + 1 and label != "one_step"
    ]
    if not labels:
      continue
    error = state_delta(state, window.next_state[:, step], validate=False)
    baseline_error = state_delta(baseline_state, window.next_state[:, step], validate=False)
    force_error = prediction.foot_force - window.foot_force[:, step]
    zero = window.foot_force[:, step].norm(dim=-1) <= zero_force_threshold
    true_contact = window.foot_force[:, step, :, 2] >= contact_threshold
    predicted_contact = prediction.foot_force[..., 2] >= contact_threshold
    swing = ~true_contact
    for label in labels:
      for name, part in DELTA_FIELDS.items():
        values = error[:, part]
        sums[f"{label}/{name}/squared"] = values.square().sum()
        sums[f"{label}/{name}/count"] = values.new_tensor(values.numel())
        baseline_values = baseline_error[:, part]
        sums[f"{label}/constant_velocity/{name}/squared"] = baseline_values.square().sum()
        sums[f"{label}/constant_velocity/{name}/count"] = baseline_values.new_tensor(baseline_values.numel())
      if delta_scale is not None:
        sums[f"{label}/state_normalized/squared"] = (error / delta_scale).square().sum()
        sums[f"{label}/state_normalized/count"] = error.new_tensor(error.numel())
        sums[f"{label}/constant_velocity_state_normalized/squared"] = (baseline_error / delta_scale).square().sum()
        sums[f"{label}/constant_velocity_state_normalized/count"] = error.new_tensor(error.numel())
      sums[f"{label}/force/squared"] = force_error.square().sum()
      sums[f"{label}/force/absolute"] = force_error.abs().sum()
      sums[f"{label}/force/count"] = force_error.new_tensor(force_error.numel())
      sums[f"{label}/zero_force/squared"] = (force_error.square() * zero[..., None]).sum()
      sums[f"{label}/zero_force/count"] = zero.sum().to(force_error.dtype) * 3
      sums[f"{label}/swing_force/squared"] = (force_error.square() * swing[..., None]).sum()
      sums[f"{label}/swing_force/absolute"] = (force_error.abs() * swing[..., None]).sum()
      sums[f"{label}/swing_force/count"] = swing.sum().to(force_error.dtype) * 3
      sums[f"{label}/contact/true_positive"] = (true_contact & predicted_contact).sum().to(force_error.dtype)
      sums[f"{label}/contact/false_positive"] = (~true_contact & predicted_contact).sum().to(force_error.dtype)
      sums[f"{label}/contact/false_negative"] = (true_contact & ~predicted_contact).sum().to(force_error.dtype)
      sums[f"{label}/contact/true_negative"] = (~true_contact & ~predicted_contact).sum().to(force_error.dtype)
      sums[f"{label}/negative_fz/absolute"] = prediction.foot_force[..., 2].clamp_max(0).abs().sum()
      sums[f"{label}/negative_fz/count"] = force_error.new_tensor(true_contact.numel())
      sums[f"{label}/seconds/sum"] = force_error.new_tensor(state.shape[0] * (step + 1) * CONTROL_DT)
      sums[f"{label}/seconds/count"] = force_error.new_tensor(state.shape[0])
  return sums


def finalize_metrics(sums: dict[str, torch.Tensor]) -> dict[str, float | None]:
  result: dict[str, float | None] = {}
  for key, value in sums.items():
    base, statistic = key.rsplit("/", 1)
    count = sums.get(f"{base}/count")
    if statistic == "count" or count is None:
      continue
    count_value = float(count)
    suffix = {"squared": "rmse", "absolute": "mae", "sum": "mean"}[statistic]
    result[f"{base}/{suffix}"] = (
      None if count_value == 0 else float((value / count).sqrt() if statistic == "squared" else value / count)
    )
  for key in sums:
    if not key.endswith("/contact/true_positive"):
      continue
    base = key.rsplit("/", 1)[0]
    tp, fp, fn, tn = (
      float(sums[f"{base}/{field}"])
      for field in (
        "true_positive",
        "false_positive",
        "false_negative",
        "true_negative",
      )
    )
    result[f"{base}/precision"] = None if tp + fp == 0 else tp / (tp + fp)
    result[f"{base}/recall"] = None if tp + fn == 0 else tp / (tp + fn)
    result[f"{base}/f1"] = None if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
    result[f"{base}/false_positive_rate"] = None if fp + tn == 0 else fp / (fp + tn)
    result[f"{base}/support"] = tp + fn
  for key, value in list(result.items()):
    if not key.endswith("/state_normalized/rmse"):
      continue
    label = key.removesuffix("/state_normalized/rmse")
    baseline = result[f"{label}/constant_velocity_state_normalized/rmse"]
    result[f"{label}/state_skill_vs_constant_velocity"] = (
      None if value is None or baseline is None or baseline == 0 else 1 - (value / baseline) ** 2
    )
  return result
