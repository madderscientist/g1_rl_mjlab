"""MLP actor and observation layouts for native position-control training."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from torch import nn

from g1_lower_rl.world_model.adapters.math import conjugate, rotate

HISTORY = 10
OBS_DIM = 93
REFERENCE_DIM = 38
REFERENCE_OFFSETS = (-15, -13, -11, -9, -8, -6, -5, -3, -2, -1, 0, 1, 2, 3, 5, 6, 8, 9, 11, 13, 15)
NATIVE_TERMS = (("joint_pos", 29), ("joint_vel", 29), ("base_ang_vel", 3), ("projected_gravity", 3), ("actions", 29))
MLP_PARAMETER_BUDGET = 9_132_861


def actor_architecture_metadata() -> dict:
  return {
    "encoder": [798, 2048, 1024, 512, 256, 32],
    "bottleneck": "continuous_tanh32",
    "decoder": [962, 2048, 1024, 512, 256, 29],
    "activation": "SiLU",
    "interface": "unchanged_history10_reference21x38_residual29",
  }


def make_actor(device: str | torch.device) -> PositionMlpActor:
  return PositionMlpActor().to(device)


def _register_observation_scales(actor: nn.Module) -> None:
  actor.register_buffer(
    "history_scale",
    torch.cat(
      [
        torch.full((HISTORY * width,), 1 / scale)
        for (_, width), scale in zip(NATIVE_TERMS, (1.0, 5.0, 2.0, 1.0, 1.0), strict=True)
      ]
    ),
  )
  actor.register_buffer("reference_scale", torch.tensor([1.0] * 3 + [0.5] * 3 + [1.0] * 32).repeat(len(REFERENCE_OFFSETS)))


class PositionMlpActor(nn.Module):
  """The original SuperMimic MLP widths on the current reference/residual interface."""

  history_scale: torch.Tensor
  reference_scale: torch.Tensor

  def __init__(self):
    super().__init__()

    def network(input_dim: int, output_dim: int, last_gain: float | None = None) -> nn.Sequential:
      layers: list[nn.Module] = []
      for width in (2048, 1024, 512, 256):
        layers.extend((nn.Linear(input_dim, width), nn.SiLU()))
        input_dim = width
      last = nn.Linear(input_dim, output_dim)
      if last_gain is not None:
        nn.init.orthogonal_(last.weight, gain=last_gain)
        nn.init.zeros_(last.bias)
      layers.append(last)
      return nn.Sequential(*layers)

    self.encoder = network(len(REFERENCE_OFFSETS) * REFERENCE_DIM, 32)
    self.decoder = network(HISTORY * OBS_DIM + 32, 29, last_gain=0.01)
    _register_observation_scales(self)

  def forward(self, observations: TensorDict, stochastic_output: bool = False) -> torch.Tensor:
    if stochastic_output:
      raise ValueError("Position MLP is deterministic; execution noise belongs to the controller")
    history = torch.cat([observations["rg_" + name] for name, _ in NATIVE_TERMS], dim=-1)
    latent = torch.tanh(self.encoder(observations["rg_reference"] * self.reference_scale))
    return self.decoder(torch.cat((latent, history * self.history_scale), dim=-1))


def unflatten_native_history(observation: torch.Tensor) -> torch.Tensor:
  if observation.ndim != 2 or observation.shape[-1] != HISTORY * OBS_DIM:
    raise ValueError("Expected native term-major 10-frame canonical29 observation")
  parts, offset = [], 0
  for _, width in NATIVE_TERMS:
    parts.append(observation[:, offset : offset + HISTORY * width].reshape(-1, HISTORY, width))
    offset += HISTORY * width
  return torch.cat(parts, dim=-1)


def actor_observations(history: torch.Tensor, reference: torch.Tensor) -> TensorDict:
  if history.ndim != 3 or history.shape[1:] != (HISTORY, OBS_DIM):
    raise ValueError("Expected [batch,10,93] observation history")
  if reference.shape != (history.shape[0], len(REFERENCE_OFFSETS) * REFERENCE_DIM):
    raise ValueError("Expected 21 canonical38 reference tokens")
  values, offset = {}, 0
  for name, width in NATIVE_TERMS:
    values["rg_" + name] = history[..., offset : offset + width].flatten(1)
    offset += width
  values["rg_reference"] = reference
  return TensorDict(values, batch_size=history.shape[:1])


def proprioception(
  state: torch.Tensor,
  previous_action: torch.Tensor,
  encoder_bias: torch.Tensor,
  default_position: torch.Tensor,
) -> torch.Tensor:
  gravity = torch.zeros_like(state[..., :3])
  gravity[..., 2] = -1
  return torch.cat(
    (
      state[..., 7:36] + encoder_bias - default_position,
      state[..., 42:71],
      state[..., 39:42],
      rotate(conjugate(state[..., 3:7]), gravity),
      previous_action,
    ),
    dim=-1,
  )


def reference_tokens(state: torch.Tensor) -> torch.Tensor:
  gravity = torch.zeros_like(state[..., :3])
  gravity[..., 2] = -1
  inverse = conjugate(state[..., 3:7])
  return torch.cat(
    (
      rotate(inverse, state[..., 36:39]),
      state[..., 39:42],
      rotate(inverse, gravity),
      state[..., 7:36],
    ),
    dim=-1,
  )


def residual_target(
  action: torch.Tensor,
  reference_position: torch.Tensor,
  scale: torch.Tensor,
  bias: torch.Tensor,
  limits: torch.Tensor,
  limited: torch.Tensor,
) -> torch.Tensor:
  target = reference_position + action * scale - bias
  return torch.where(limited, target.clamp(limits[:, 0], limits[:, 1]), target)
