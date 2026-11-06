"""Native behavior-policy samples and an exact fixed-Gaussian KL trust region."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import nn


@dataclass(frozen=True)
class NativePolicyBatch:
  history: torch.Tensor
  reference: torch.Tensor
  mean: torch.Tensor

  def __len__(self) -> int:
    return len(self.mean)


class NativePolicyRollout:
  def __init__(self, train_envs: int):
    self.train_envs = train_envs
    self.records: list[dict[str, torch.Tensor]] = []

  @torch.no_grad()
  def append(self, **record: torch.Tensor) -> None:
    if not record or any(len(value) < self.train_envs for value in record.values()):
      raise ValueError("Native policy record does not contain all training worlds")
    self.records.append({key: value[: self.train_envs].detach().clone() for key, value in record.items()})

  @torch.no_grad()
  def _stack(self) -> dict[str, torch.Tensor]:
    if not self.records:
      raise ValueError("A fresh nonempty native rollout is required")
    stacked = {key: torch.stack([row[key] for row in self.records]) for key in self.records[0]}
    if any(not torch.isfinite(value).all() for value in stacked.values() if value.is_floating_point()):
      raise FloatingPointError("Nonfinite native policy rollout")
    self.records.clear()
    return stacked

  @torch.no_grad()
  def finish_policy(self) -> NativePolicyBatch:
    stacked = self._stack()
    return NativePolicyBatch(**{key: stacked[key].flatten(0, 1) for key in ("history", "reference", "mean")})


@torch.no_grad()
def policy_kl(
  predict: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
  batch: NativePolicyBatch,
  std: float,
  microbatch: int,
) -> torch.Tensor:
  sums = batch.mean.new_zeros(2, dtype=torch.float64)
  for begin in range(0, len(batch), microbatch):
    end = begin + microbatch
    mean = predict(batch.history[begin:end], batch.reference[begin:end])
    kl = 0.5 * ((mean - batch.mean[begin:end]) / std).square().sum(-1)
    sums[0] += kl.double().sum()
    sums[1] += kl.numel()
  if dist.is_initialized():
    dist.all_reduce(sums)
  return sums[0] / sums[1]


@torch.no_grad()
def constrain_policy_kl(
  actor: nn.Module,
  optimizer: torch.optim.Optimizer,
  before: list[torch.Tensor],
  optimizer_before: dict[str, Any],
  measure: Callable[[], torch.Tensor],
  limit: float,
) -> dict[str, float]:
  parameters = list(actor.parameters())
  if len(parameters) != len(before) or not math.isfinite(limit) or limit <= 0:
    raise ValueError("Policy KL trust region requires matching weights and a positive finite limit")
  proposed = [parameter.detach().clone() for parameter in parameters]
  for attempt in range(13):
    fraction = 0.5**attempt
    for parameter, old, new in zip(parameters, before, proposed, strict=True):
      parameter.copy_(torch.lerp(old, new, fraction))
    kl = measure()
    if not torch.isfinite(kl):
      for parameter, old in zip(parameters, before, strict=True):
        parameter.copy_(old)
      optimizer.load_state_dict(optimizer_before)
      raise FloatingPointError("Nonfinite policy KL; actor and Adam restored")
    if kl <= limit:
      return {"kl": float(kl), "step_fraction": fraction, "rejected": 0.0}
  for parameter, old in zip(parameters, before, strict=True):
    parameter.copy_(old)
  optimizer.load_state_dict(optimizer_before)
  logging.getLogger(__name__).warning("Policy update rejected: rollout KL exceeded %g after backtracking", limit)
  return {"kl": float(measure()), "step_fraction": 0.0, "rejected": 1.0}
