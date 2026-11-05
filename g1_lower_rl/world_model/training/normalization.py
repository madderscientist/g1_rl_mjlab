"""Frozen training-only sufficient statistics, shared across DDP ranks."""

from __future__ import annotations

import torch
import torch.distributed as dist

from g1_lower_rl.world_model.model import DELTA_DIM, INPUT_DIM, PositionWorldModel, state_delta, state_update, world_to_root
from g1_lower_rl.world_model.training.replay import RingReplay


@torch.no_grad()
def fit_normalization(
  model: PositionWorldModel, replay: RingReplay, input_floor: float = 1e-3, delta_floor: float = 1e-3, force_floor: float = 1.0
) -> dict[str, object]:
  if min(input_floor, delta_floor, force_floor) <= 0:
    raise ValueError("Normalization floors must be positive")
  # Float64 sums prevent cancellation for nearly constant physical parameters.
  stats = torch.zeros(1 + 2 * INPUT_DIM + 2 * DELTA_DIM + 6, dtype=torch.float64, device=replay.device)
  update_part = slice(1 + 2 * INPUT_DIM, 1 + 2 * INPUT_DIM + DELTA_DIM)
  state_part = slice(update_part.stop, -6)
  for states, inputs, targets, forces in replay.training_chunks():
    canonical = inputs.double()
    delta = state_update(states, targets).double()
    physical = state_delta(states, targets).double()
    quaternion = targets[..., 3:7].unsqueeze(-2).expand(*forces.shape[:-1], 4)
    local_forces = world_to_root(quaternion, forces).flatten(-2).double()
    stats[0] += inputs.shape[0]
    stats[1 : 1 + INPUT_DIM] += canonical.sum(0)
    stats[1 + INPUT_DIM : 1 + 2 * INPUT_DIM] += canonical.square().sum(0)
    stats[update_part] += delta.square().sum(0)
    stats[state_part] += physical.square().sum(0)
    stats[-6:] += local_forces.square().sum(0)
  if dist.is_initialized():
    dist.all_reduce(stats)
  count = int(stats[0])
  if count <= 0:
    raise RuntimeError("No valid training transitions for normalization")
  mean = stats[1 : 1 + INPUT_DIM] / count
  variance = stats[1 + INPUT_DIM : 1 + 2 * INPUT_DIM] / count - mean.square()
  model.set_normalization(
    mean.float(),
    variance.clamp_min(0).sqrt().clamp_min(input_floor).float(),
    (stats[update_part] / count).sqrt().clamp_min(delta_floor).float(),
    (stats[-6:] / count).sqrt().clamp_min(force_floor).float(),
    (stats[state_part] / count).sqrt().clamp_min(delta_floor).float(),
  )
  return {
    "samples": count,
    "split": "training_environments_only",
    "frozen": True,
    "input_floor": input_floor,
    "delta_floor": delta_floor,
    "force_floor": force_floor,
    "state_scale": "physical_local_differences_separate_from_integration_residual_scales",
    "force_scale_frame": "next_root",
  }
