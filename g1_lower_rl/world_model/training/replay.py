"""Device-resident, environment-isolated 20ms control-transition replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from g1_lower_rl.world_model.model import CONTRACT_VERSION, INPUT_DIM, INPUT_FIELDS, STATE_DIM, STATE_FEATURE_DIM, pack_inputs


class TransitionLike(Protocol):
  state: torch.Tensor
  inputs: torch.Tensor
  next_state: torch.Tensor
  foot_force: torch.Tensor
  valid: torch.Tensor
  segment: torch.Tensor
  contract_version: int


@dataclass
class ReplayWindow:
  state: torch.Tensor
  inputs: torch.Tensor
  next_state: torch.Tensor
  foot_force: torch.Tensor
  env: torch.Tensor
  sequence: torch.Tensor


class RingReplay:
  """Each column is one environment; a window never wraps logical history.

  Capacity is measured in 20ms control steps per environment, not physical substeps.
  Training and validation columns remain fixed for the lifetime of the replay.
  """

  def __init__(self, capacity: int, num_envs: int, validation_envs: int, device: str | torch.device = "cpu") -> None:
    if capacity <= 0 or not 0 < validation_envs < num_envs:
      raise ValueError("Positive capacity and disjoint nonempty train/validation environment splits are required")
    self.capacity, self.num_envs = capacity, num_envs
    self.train_envs = num_envs - validation_envs
    self.device = torch.device(device)
    self.inputs = torch.empty(capacity, num_envs, INPUT_DIM, device=self.device)
    self.state = torch.empty(capacity, num_envs, STATE_DIM, device=self.device)
    self.next_state = torch.empty(capacity, num_envs, STATE_DIM, device=self.device)
    self.foot_force = torch.empty(capacity, num_envs, 2, 3, device=self.device)
    self.valid = torch.zeros(capacity, num_envs, dtype=torch.bool, device=self.device)
    self.segment = torch.zeros(capacity, num_envs, dtype=torch.int64, device=self.device)
    self.total_steps = 0
    self._candidates: dict[tuple[int, bool], torch.Tensor] = {}

  @property
  def size(self) -> int:
    return min(self.capacity, self.total_steps)

  @property
  def bytes_allocated(self) -> int:
    return sum(
      t.numel() * t.element_size()
      for t in (self.state, self.inputs, self.next_state, self.foot_force, self.valid, self.segment)
    )

  def _env_slice(self, validation: bool) -> slice:
    return slice(self.train_envs, self.num_envs) if validation else slice(0, self.train_envs)

  @torch.no_grad()
  def append(self, batch: TransitionLike) -> int:
    if batch.contract_version != CONTRACT_VERSION:
      raise ValueError("Replay requires control-step position protocol v3")
    steps = batch.inputs.shape[0]
    shapes = {
      "state": (steps, self.num_envs, STATE_DIM),
      "inputs": (steps, self.num_envs, INPUT_DIM),
      "next_state": (steps, self.num_envs, STATE_DIM),
      "foot_force": (steps, self.num_envs, 2, 3),
      "valid": (steps, self.num_envs),
      "segment": (steps, self.num_envs),
    }
    for name, shape in shapes.items():
      value = getattr(batch, name)
      if value.shape != shape or value.device != self.device:
        raise ValueError(f"{name} must be {shape} on {self.device}")
    if batch.valid.dtype != torch.bool or batch.segment.dtype != torch.int64:
      raise TypeError("valid must be bool and segment must be int64")
    for name in ("state", "inputs", "next_state", "foot_force"):
      if getattr(batch, name).dtype != torch.float32:
        raise TypeError(f"{name} must be float32")
    valid_inputs = batch.inputs[batch.valid]
    initial_states = batch.state[batch.valid]
    valid_states = batch.next_state[batch.valid]
    finite = torch.isfinite(valid_inputs).all() & torch.isfinite(valid_states).all() & torch.isfinite(initial_states).all()
    finite &= torch.isfinite(batch.foot_force[batch.valid]).all()
    quaternions_ok = (initial_states[:, 3:7].norm(dim=-1) > 1e-6).all() & (valid_states[:, 3:7].norm(dim=-1) > 1e-6).all()
    if not bool(finite & quaternions_ok):
      raise ValueError("Valid replay transitions require finite values and nonzero quaternions")
    expected = pack_inputs(
      initial_states, valid_inputs[:, INPUT_FIELDS["joint_target"]], valid_inputs[:, INPUT_FIELDS["context"]]
    )
    if not torch.allclose(valid_inputs[:, :STATE_FEATURE_DIM], expected[:, :STATE_FEATURE_DIM], atol=1e-5, rtol=1e-5):
      raise ValueError("Replay features do not match full initial states")
    valid_count = int(batch.valid.sum())
    # Trim oversized writes before index_copy_: repeated destination indices are nondeterministic.
    keep = min(steps, self.capacity)
    if keep:
      source = slice(steps - keep, steps)
      rows = torch.arange(self.total_steps + steps - keep, self.total_steps + steps, device=self.device) % self.capacity
      for name in shapes:
        getattr(self, name).index_copy_(0, rows, getattr(batch, name)[source])
    self.total_steps += steps
    self._candidates.clear()
    return valid_count

  def candidates(self, horizon: int, *, validation: bool = False) -> torch.Tensor:
    """Return flattened chronological start/env indices, cached until append."""
    if horizon <= 0 or horizon > self.capacity:
      raise ValueError("horizon must be in [1, capacity]")
    key = (horizon, validation)
    if key in self._candidates:
      return self._candidates[key]
    if self.size < horizon:
      result = torch.empty(0, dtype=torch.int64, device=self.device)
    else:
      rows = torch.arange(self.total_steps - self.size, self.total_steps, device=self.device) % self.capacity
      envs = self._env_slice(validation)
      valid = self.valid[rows, envs]
      segment = self.segment[rows, envs]
      prefix = torch.cat((torch.zeros_like(valid[:1], dtype=torch.int64), (~valid).cumsum(0)))
      good = (prefix[horizon:] - prefix[:-horizon]) == 0
      if horizon > 1:
        changes = segment[1:] != segment[:-1]
        changes_prefix = torch.cat((torch.zeros_like(segment[:1]), changes.cumsum(0)))
        good &= (changes_prefix[horizon - 1 :] - changes_prefix[: 1 - horizon]) == 0
      result = good.flatten().nonzero(as_tuple=False).flatten()
    self._candidates[key] = result
    return result

  def sample(
    self, batch_size: int, horizon: int, *, validation: bool = False, generator: torch.Generator | None = None
  ) -> ReplayWindow:
    if batch_size <= 0:
      raise ValueError("batch_size must be positive")
    choices = self.candidates(horizon, validation=validation)
    if choices.numel() == 0:
      raise RuntimeError(f"No contiguous horizon-{horizon} windows in {'validation' if validation else 'training'} split")
    selected = choices[torch.randint(choices.numel(), (batch_size,), device=self.device, generator=generator)]
    count = self.num_envs - self.train_envs if validation else self.train_envs
    starts = selected // count
    env = selected % count + (self.train_envs if validation else 0)
    sequence = starts[:, None] + torch.arange(horizon, device=self.device) + self.total_steps - self.size
    rows = sequence % self.capacity
    columns = env[:, None]
    return ReplayWindow(
      self.state[rows, columns],
      self.inputs[rows, columns],
      self.next_state[rows, columns],
      self.foot_force[rows, columns],
      env,
      sequence,
    )

  def training_chunks(self, chunk_steps: int = 32):
    """Yield valid training transitions only, without materializing the entire dataset."""
    for start in range(0, self.size, chunk_steps):
      rows = (
        torch.arange(
          self.total_steps - self.size + start,
          self.total_steps - self.size + min(start + chunk_steps, self.size),
          device=self.device,
        )
        % self.capacity
      )
      valid = self.valid[rows, : self.train_envs]
      yield (
        self.state[rows, : self.train_envs][valid],
        self.inputs[rows, : self.train_envs][valid],
        self.next_state[rows, : self.train_envs][valid],
        self.foot_force[rows, : self.train_envs][valid],
      )
