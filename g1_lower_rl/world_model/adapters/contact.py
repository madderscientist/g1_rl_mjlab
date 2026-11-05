"""Force-derived hard contacts and explicit, caller-owned event/timer memory."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import nn


@dataclass(frozen=True)
class ContactConfig:
  """Newton thresholds are calibration parameters, NOT simulator contact-found parity."""

  on_force: float = 10.0
  off_force: float = 5.0
  temperature: float = 2.0

  def __post_init__(self) -> None:
    if not all(math.isfinite(x) for x in (self.on_force, self.off_force, self.temperature)):
      raise ValueError("Contact calibration must be finite")
    if not 0 <= self.off_force < self.on_force or self.temperature <= 0:
      raise ValueError("Require 0 <= off_force < on_force and temperature > 0")

  @classmethod
  def calibrate(
    cls,
    foot_force: torch.Tensor,
    contact_found: torch.Tensor,
    tail_quantile: float = 0.05,
  ) -> ContactConfig:
    """Fit a conservative hysteresis gap from paired teacher samples (world +z ground).

    Off is the (1-q) quantile of noncontact normal force; on is the q quantile
    of contact force. Overlapping distributions are rejected rather than claiming
    contact-found equivalence that the learned force cannot represent.
    Use only training/calibration splits, then freeze and serialize the result.
    """
    if not 0 <= tail_quantile < 0.5:
      raise ValueError("tail_quantile must lie in [0, 0.5)")
    if foot_force.shape[-2:] != (2, 3) or contact_found.shape != foot_force.shape[:-1]:
      raise ValueError("Expected paired [..., 2, 3] force and [..., 2] contact labels")
    if contact_found.dtype != torch.bool or not torch.isfinite(foot_force).all():
      raise ValueError("Calibration needs boolean contact labels and finite forces")
    normal = foot_force[..., 2].detach().clamp_min(0).double()
    positive, negative = normal[contact_found], normal[~contact_found]
    if positive.numel() == 0 or negative.numel() == 0:
      raise ValueError("Calibration needs both contact and noncontact examples")
    on = float(torch.quantile(positive, tail_quantile))
    off = float(torch.quantile(negative, 1 - tail_quantile))
    if on <= off:
      raise ValueError("Contact/noncontact force distributions overlap; inspect teacher calibration")
    return cls(on_force=on, off_force=off, temperature=max((on - off) / 4, 1e-6))


class ContactMemory(NamedTuple):
  contact: torch.Tensor
  air_time: torch.Tensor
  contact_time: torch.Tensor
  last_air_time: torch.Tensor
  last_contact_time: torch.Tensor

  @classmethod
  def zeros(cls, reference: torch.Tensor, initially_contact: bool = False) -> ContactMemory:
    """reference has shape [..., 2] and floating dtype."""
    if reference.shape[-1] != 2 or not reference.is_floating_point():
      raise ValueError("Expected floating point [..., 2] timer reference")
    return cls(
      torch.full_like(reference, initially_contact, dtype=torch.bool), *(torch.zeros_like(reference) for _ in range(4))
    )

  def reset(self, mask: torch.Tensor, initially_contact: bool = False) -> ContactMemory:
    """Functional per-environment reset; mask has the batch shape (without feet)."""
    mask = mask.unsqueeze(-1)
    return ContactMemory(
      torch.where(mask, torch.full_like(self.contact, initially_contact), self.contact),
      *(torch.where(mask, torch.zeros_like(x), x) for x in self[1:]),
    )


class ContactUpdate(NamedTuple):
  memory: ContactMemory
  onset: torch.Tensor
  liftoff: torch.Tensor
  soft: torch.Tensor
  normal_force: torch.Tensor
  force: torch.Tensor


class ForceContact(nn.Module):
  def __init__(self, config: ContactConfig | None = None):
    super().__init__()
    self.config = ContactConfig() if config is None else config

  def forward(
    self,
    foot_force: torch.Tensor,
    previous: ContactMemory,
    dt: float | torch.Tensor,
    normal: torch.Tensor | None = None,
  ) -> ContactUpdate:
    """Advance once per supplied force sample; positive world-z means ground-to-foot.

    dt may be a scalar or batch-shaped tensor. Onset sample starts contact_time at
    dt; liftoff starts air_time at dt. Completed durations are captured BEFORE reset.
    A sloped-ground normal may be supplied in world coordinates.
    """
    if foot_force.shape[-2:] != (2, 3) or previous.contact.shape != foot_force.shape[:-1]:
      raise ValueError("Force must be [..., 2, 3] and memory must match")
    dt = torch.as_tensor(dt, dtype=foot_force.dtype, device=foot_force.device)
    if not torch.isfinite(dt).all() or (dt <= 0).any():
      raise ValueError("dt must be finite and positive")
    dt = torch.broadcast_to(dt, foot_force.shape[:-2]).unsqueeze(-1)
    if not torch.isfinite(foot_force).all():
      raise ValueError("Force must be finite")
    if normal is None:
      force = foot_force[..., 2]
    else:
      norm = torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
      if not torch.isfinite(normal).all() or (norm <= 0).any():
        raise ValueError("Ground normals must be finite and nonzero")
      force = (foot_force * normal / norm).sum(-1)
    force = force.clamp_min(0)
    threshold = torch.where(previous.contact, self.config.off_force, self.config.on_force)
    hard = force >= threshold
    onset, liftoff = hard & ~previous.contact, ~hard & previous.contact
    zero = torch.zeros_like(force)
    memory = ContactMemory(
      hard,
      torch.where(hard, zero, previous.air_time + dt),
      torch.where(hard, previous.contact_time + dt, zero),
      torch.where(onset, previous.air_time, previous.last_air_time),
      torch.where(liftoff, previous.contact_time, previous.last_contact_time),
    )
    soft = torch.sigmoid((force - threshold) / self.config.temperature)
    return ContactUpdate(memory, onset, liftoff, soft, force, foot_force)
