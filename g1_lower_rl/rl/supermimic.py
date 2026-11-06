"""Simulator-independent BPTT returns, fixed-target GAE and optimization."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch import nn


@contextmanager
def frozen(*modules: nn.Module):
  """Freeze weights, not the differentiable input-to-output computation."""
  parameters = list(dict.fromkeys(p for module in modules for p in module.parameters()))
  flags = [p.requires_grad for p in parameters]
  for parameter in parameters:
    parameter.requires_grad_(False)
  try:
    yield
  finally:
    for parameter, flag in zip(parameters, flags, strict=True):
      parameter.requires_grad_(flag)


def bptt_return(
  rewards: torch.Tensor,
  termination: torch.Tensor,
  terminal_value: torch.Tensor,
  gamma: float,
  *,
  truncation: torch.Tensor | None = None,
  bootstrap_values: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  if rewards.ndim != 2 or termination.shape != rewards.shape or terminal_value.shape != rewards.shape[:1]:
    raise ValueError("Expected rewards/termination [batch, decision_ticks] and terminal value [batch]")
  if not 0 < gamma <= 1:
    raise ValueError("gamma must be in (0,1]")
  if (truncation is None) != (bootstrap_values is None):
    raise ValueError("Truncation masks and their differentiable bootstrap values must be supplied together")
  if (
    truncation is not None
    and bootstrap_values is not None
    and (truncation.shape != rewards.shape or bootstrap_values.shape != rewards.shape)
  ):
    raise ValueError("Truncation/bootstrap shapes must match [batch, decision_ticks]")
  survival = torch.ones_like(terminal_value)
  result = torch.zeros_like(terminal_value)
  weights = []
  for t in range(rewards.shape[1]):
    weights.append(survival)
    result = result + gamma**t * survival * rewards[:, t]
    survival = survival * (1 - termination[:, t])
    if truncation is not None and bootstrap_values is not None:
      result = result + gamma ** (t + 1) * survival * truncation[:, t] * bootstrap_values[:, t]
      survival = survival * (1 - truncation[:, t])
  result = result + gamma ** rewards.shape[1] * survival * terminal_value
  return result, torch.stack(weights, dim=1)


@torch.no_grad()
def gae_targets(
  rewards: torch.Tensor,
  termination: torch.Tensor,
  values: torch.Tensor,
  gamma: float = 0.99,
  gae_lambda: float = 0.95,
  *,
  truncation: torch.Tensor | None = None,
) -> torch.Tensor:
  if rewards.ndim != 2 or termination.shape != rewards.shape or values.shape != (rewards.shape[0], rewards.shape[1] + 1):
    raise ValueError("GAE expects rewards/dones [N,H] and values [N,H+1]")
  if not 0 < gamma <= 1 or not 0 <= gae_lambda <= 1:
    raise ValueError("Invalid GAE discount/lambda")
  if truncation is not None and truncation.shape != rewards.shape:
    raise ValueError("Truncation mask must match GAE rewards")
  advantage = torch.zeros_like(rewards[:, 0])
  targets = torch.empty_like(rewards)
  for t in reversed(range(rewards.shape[1])):
    discount = gamma * (1 - termination[:, t])
    delta = rewards[:, t] + discount * values[:, t + 1] - values[:, t]
    continuation = 1 if truncation is None else 1 - truncation[:, t]
    advantage = delta + discount * gae_lambda * continuation * advantage
    targets[:, t] = values[:, t] + advantage
  return targets


def cosine_learning_rate(iteration: int, base: float, world_size: int, decay_steps: int = 250000) -> float:
  if iteration < 0 or base <= 0 or world_size < 1 or decay_steps < 1:
    raise ValueError("Invalid learning-rate schedule")
  progress = min(iteration / decay_steps, 1)
  return base * math.sqrt(world_size) * (0.03 + 0.97 * (1 + math.cos(math.pi * progress)) / 2)


def optimizer_step(parameters: Iterable[nn.Parameter], optimizer: torch.optim.Optimizer, *, actor: bool = False) -> float:
  params = [p for p in parameters if p.requires_grad]
  gradients = [p.grad for p in params if p.grad is not None]
  if not gradients:
    raise RuntimeError("Optimization phase produced no gradients")
  if dist.is_initialized():
    # One collective per optimizer step; preserve missing gradients rather than inventing updates.
    present = torch.tensor([p.grad is not None for p in params], device=gradients[0].device, dtype=torch.int64)
    counts = present.clone()
    dist.all_reduce(counts)
    if not torch.all((counts == 0) | (counts == dist.get_world_size())):
      raise RuntimeError("Ranks produced inconsistent missing-gradient masks")
    flat = nn.utils.parameters_to_vector(gradients)
    dist.all_reduce(flat)
    flat.div_(dist.get_world_size())
    offset = 0
    for gradient in gradients:
      gradient.copy_(flat[offset : offset + gradient.numel()].view_as(gradient))
      offset += gradient.numel()
  norm = nn.utils.get_total_norm(gradients, error_if_nonfinite=False)
  if not torch.isfinite(norm):
    # Squaring finite FP32 gradients can overflow although the true norm is finite.
    norm = nn.utils.get_total_norm([g.double() for g in gradients], error_if_nonfinite=True)
    logging.getLogger(__name__).warning(
      "FP32 gradient norm overflow with finite gradients; using FP64 norm reduction only: actor=%s norm=%s",
      actor,
      float(norm),
    )
  nn.utils.clip_grads_with_norm_(params, 1.0, norm)
  if actor:
    for gradient in gradients:
      gradient.clamp_(-0.1, 0.1)
  optimizer.step()
  return float(norm)
