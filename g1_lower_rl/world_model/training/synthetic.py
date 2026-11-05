"""Explicit simulator-free smoke source; never substituted for real collection."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from g1_lower_rl.world_model.model import (
  CONTRACT_VERSION,
  CONTROL_DT,
  JOINT_DIM,
  STATE_DIM,
  STATE_FIELDS,
  pack_inputs,
)


@dataclass
class SyntheticBatch:
  state: torch.Tensor
  inputs: torch.Tensor
  next_state: torch.Tensor
  foot_force: torch.Tensor
  valid: torch.Tensor
  segment: torch.Tensor
  contract_version: int = CONTRACT_VERSION


class SyntheticCollector:
  def __init__(self, *, num_envs: int, device: torch.device, seed: int, physics_dt: float) -> None:
    if num_envs < 2 or physics_dt != 0.005:
      raise ValueError("Synthetic v3 fixtures require >=2 environments and fixed5ms native physics metadata")
    self.num_envs, self.device, self.dt = num_envs, device, CONTROL_DT
    self.generator = torch.Generator(device=device).manual_seed(seed)
    self.state = torch.zeros(num_envs, STATE_DIM, device=device)
    self.state[:, 2], self.state[:, 3] = 0.8, 1.0
    self.step = 0
    self.counters = {"control_steps": 0}
    self.metadata = {
      "source": "synthetic_smoke_not_simulator",
      "recipe": "joint_velocity=target-current; alternating_zero_force",
      "contract_version": CONTRACT_VERSION,
      "physics_dt": physics_dt,
      "control_dt": CONTROL_DT,
    }

  def collect(self, num_control_steps: int = 1) -> SyntheticBatch:
    if num_control_steps < 1:
      raise ValueError("num_control_steps must be positive")
    inputs, initial, states, forces, segments = [], [], [], [], []
    for _ in range(num_control_steps):
      target = torch.randn(self.num_envs, JOINT_DIM, device=self.device, generator=self.generator)
      context = torch.ones(self.num_envs, 48, device=self.device)
      context[:, 44:] = 0
      packed = pack_inputs(self.state, target, context)
      next_state = self.state.clone()
      next_state[:, STATE_FIELDS["joint_velocity"]] = target - self.state[:, STATE_FIELDS["joint_position"]]
      next_state[:, STATE_FIELDS["joint_position"]] += next_state[:, STATE_FIELDS["joint_velocity"]] * self.dt
      force = torch.zeros(self.num_envs, 2, 3, device=self.device)
      if self.step % 2:
        force[:, :, 2] = target[:, :2].abs() * 10
      initial.append(self.state)
      inputs.append(packed)
      states.append(next_state)
      forces.append(force)
      segments.append(torch.full((self.num_envs,), self.step // 128, dtype=torch.int64, device=self.device))
      self.state = next_state
      self.step += 1
      self.counters["control_steps"] += self.num_envs
    return SyntheticBatch(
      state=torch.stack(initial),
      inputs=torch.stack(inputs),
      next_state=torch.stack(states),
      foot_force=torch.stack(forces),
      valid=torch.ones(len(inputs), self.num_envs, dtype=torch.bool, device=self.device),
      segment=torch.stack(segments),
    )

  def close(self) -> None:
    pass
