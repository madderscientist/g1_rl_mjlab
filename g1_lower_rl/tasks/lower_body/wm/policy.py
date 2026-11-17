"""MLP actor, history layout, and native-aligned observation/target helpers for velocity WM."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn

from g1_lower_rl.rl.supermimic import constrain_policy_kl, lower_body_actuator_target
from g1_lower_rl.world_model.adapters.math import conjugate, rotate, unit

DEFAULT_HISTORY = 10
ACTION_DIM = 15
ARM_DIM = 14
OBS_DIM = 57
GAIT_PERIOD = 0.6
CONTROL_DT = 0.02

OBS_TERMS: tuple[tuple[str, int], ...] = (
  ("base_ang_vel", 3),
  ("projected_gravity", 3),
  ("command_twist", 3),
  ("command_height", 1),
  ("phase", 2),
  ("joint_pos", 15),
  ("joint_vel", 15),
  ("actions", 15),
)
TERM_OFFSETS: tuple[float, ...] = (0.0, 0.0, 0.0, 0.68, 0.0, 0.0, 0.0, 0.0)
TERM_DIVISORS: tuple[float, ...] = (2.0, 1.0, 1.0, 0.10, 1.0, 1.0, 5.0, 1.0)
NOISE_STDS: tuple[float, ...] = (0.115, 0.029, 0.0, 0.0, 0.0, 0.006, 0.87, 0.0)


def observation_noise_std(device: torch.device | str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
  return torch.cat(
    [
      torch.full((width,), std, device=device, dtype=dtype)
      for (_, width), std in zip(OBS_TERMS, NOISE_STDS, strict=True)
    ]
  )


def actor_architecture_metadata(actor: VelocityMlpActor) -> dict[str, object]:
  return {
    "type": "velocity_history_mlp",
    "history_length": actor.history_length,
    "obs_dim": OBS_DIM,
    "input_dim": actor.history_length * OBS_DIM,
    "hidden_dims": list(actor.hidden_dims),
    "action_dim": ACTION_DIM,
    "activation": "SiLU",
    "term_order": [name for name, _ in OBS_TERMS],
    "term_offsets": list(TERM_OFFSETS),
    "term_divisors": list(TERM_DIVISORS),
  }


class VelocityMlpActor(nn.Module):
  """Pure MLP policy mapping [batch, history_length, 57] -> [batch, 15] lower-body actions."""

  frame_offset: torch.Tensor
  frame_scale: torch.Tensor

  def __init__(
    self,
    history_length: int = DEFAULT_HISTORY,
    hidden_dims: tuple[int, ...] = (1024, 512, 256),
  ):
    super().__init__()
    if history_length < 1 or not hidden_dims or min(hidden_dims) < 1:
      raise ValueError("history_length and hidden_dims must be positive")
    self.history_length = int(history_length)
    self.hidden_dims = tuple(int(w) for w in hidden_dims)
    self.register_buffer(
      "frame_offset",
      torch.cat(
        [
          torch.full((width,), offset)
          for (_, width), offset in zip(OBS_TERMS, TERM_OFFSETS, strict=True)
        ]
      ),
    )
    self.register_buffer(
      "frame_scale",
      torch.cat(
        [
          torch.full((width,), 1.0 / divisor)
          for (_, width), divisor in zip(OBS_TERMS, TERM_DIVISORS, strict=True)
        ]
      ),
    )
    layers: list[nn.Module] = []
    in_dim = self.history_length * OBS_DIM
    for width in self.hidden_dims:
      layers.extend((nn.Linear(in_dim, width), nn.SiLU()))
      in_dim = width
    last = nn.Linear(in_dim, ACTION_DIM)
    nn.init.orthogonal_(last.weight, gain=0.01)
    nn.init.zeros_(last.bias)
    layers.append(last)
    self.net = nn.Sequential(*layers)

  def forward(self, history: torch.Tensor) -> torch.Tensor:
    if history.ndim != 3 or history.shape[1:] != (self.history_length, OBS_DIM):
      raise ValueError(
        f"Expected [batch, {self.history_length}, {OBS_DIM}] observation history, got {tuple(history.shape)}"
      )
    scaled = ((history - self.frame_offset) * self.frame_scale).flatten(1)
    return self.net(scaled)


def migrate_actor_state_dict(
  actor: VelocityMlpActor,
  saved_state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], bool]:
  """Exact function-preserving linear surgery when migrating uncentered legacy actor weights."""
  state = {k: v.clone() for k, v in saved_state.items()}
  new_offset = actor.frame_offset.to(device=state["frame_scale"].device, dtype=state["frame_scale"].dtype)
  new_scale = actor.frame_scale.to(device=state["frame_scale"].device, dtype=state["frame_scale"].dtype)
  old_offset = state.get("frame_offset", torch.zeros_like(new_offset))
  old_scale = state["frame_scale"]
  if torch.allclose(old_offset, new_offset) and torch.allclose(old_scale, new_scale):
    state["frame_offset"] = old_offset
    return state, False
  w0 = state["net.0.weight"]
  b0 = state["net.0.bias"]
  ratio = (old_scale / new_scale).repeat(actor.history_length)
  delta_off = ((new_offset - old_offset) * old_scale).repeat(actor.history_length)
  state["net.0.bias"] = b0 + w0 @ delta_off
  state["net.0.weight"] = w0 * ratio.unsqueeze(0)
  state["frame_offset"] = new_offset
  state["frame_scale"] = new_scale
  return state, True


def make_actor(
  device: str | torch.device,
  history_length: int = DEFAULT_HISTORY,
  hidden_dims: tuple[int, ...] = (1024, 512, 256),
) -> VelocityMlpActor:
  return VelocityMlpActor(history_length=history_length, hidden_dims=hidden_dims).to(device)


def unflatten_native_history(observation: torch.Tensor, history_length: int = DEFAULT_HISTORY) -> torch.Tensor:
  """Convert mjlab term-major [B, H * 57] concatenated observations to frame-major [B, H, 57]."""
  if observation.ndim != 2 or observation.shape[-1] != history_length * OBS_DIM:
    raise ValueError(f"Expected [batch, {history_length * OBS_DIM}] native actor observation, got {tuple(observation.shape)}")
  parts: list[torch.Tensor] = []
  offset = 0
  for _, width in OBS_TERMS:
    span = history_length * width
    parts.append(observation[:, offset : offset + span].reshape(-1, history_length, width))
    offset += span
  return torch.cat(parts, dim=-1)


def phase_clock(
  episode_step: torch.Tensor,
  phase_offset: torch.Tensor,
  twist: torch.Tensor,
  *,
  dt: float = CONTROL_DT,
  period: float = GAIT_PERIOD,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  clock = (episode_step.to(twist.dtype) * dt) / period + phase_offset.to(twist.dtype)
  angle = torch.remainder(clock, 1.0) * (2.0 * torch.pi)
  raw = torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)
  standing = torch.linalg.vector_norm(twist, dim=-1) < command_threshold
  return torch.where(standing.unsqueeze(-1), torch.zeros_like(raw), raw)


def velocity_observation(
  state: torch.Tensor,
  previous_action: torch.Tensor,
  twist: torch.Tensor,
  height_cmd: torch.Tensor,
  episode_step: torch.Tensor,
  phase_offset: torch.Tensor,
  encoder_bias_lower: torch.Tensor,
  default_lower: torch.Tensor,
  imu_bias: torch.Tensor | None = None,
) -> torch.Tensor:
  """Construct one 57D velocity-tracking actor observation frame matching make_observations()."""
  q = unit(state[..., 3:7])
  gravity = torch.zeros_like(state[..., :3])
  gravity[..., 2] = -1.0
  ang_vel = state[..., 39:42]
  proj_grav = rotate(conjugate(q), gravity)
  if imu_bias is not None:
    ang_vel = ang_vel + imu_bias[..., :3]
    proj_grav = proj_grav + imu_bias[..., 3:6]
  h_cmd = height_cmd.unsqueeze(-1) if height_cmd.ndim == twist.ndim - 1 else height_cmd
  phase = phase_clock(episode_step, phase_offset, twist)
  joint_pos = state[..., 7 : 7 + ACTION_DIM] + encoder_bias_lower - default_lower
  joint_vel = state[..., 42 : 42 + ACTION_DIM]
  return torch.cat(
    (ang_vel, proj_grav, twist, h_cmd, phase, joint_pos, joint_vel, previous_action),
    dim=-1,
  )


velocity_target = lower_body_actuator_target


@dataclass(frozen=True)
class VelocityPolicyBatch:
  history: torch.Tensor
  mean: torch.Tensor

  def __len__(self) -> int:
    return len(self.mean)


class VelocityPolicyRollout:
  def __init__(self, train_envs: int):
    self.train_envs = train_envs
    self.records: list[dict[str, torch.Tensor]] = []

  @torch.no_grad()
  def append(self, *, history: torch.Tensor, mean: torch.Tensor) -> None:
    if len(history) < self.train_envs or len(mean) < self.train_envs:
      raise ValueError("Native velocity policy record does not contain all training worlds")
    self.records.append(
      {
        "history": history[: self.train_envs].detach().clone(),
        "mean": mean[: self.train_envs].detach().clone(),
      }
    )

  @torch.no_grad()
  def finish(self) -> VelocityPolicyBatch:
    if not self.records:
      raise ValueError("A fresh nonempty native velocity rollout is required")
    history = torch.stack([row["history"] for row in self.records]).flatten(0, 1)
    mean = torch.stack([row["mean"] for row in self.records]).flatten(0, 1)
    if not torch.isfinite(history).all() or not torch.isfinite(mean).all():
      raise FloatingPointError("Nonfinite native velocity policy rollout")
    self.records.clear()
    return VelocityPolicyBatch(history=history, mean=mean)


@torch.no_grad()
def velocity_policy_kl(
  predict: Callable[[torch.Tensor], torch.Tensor],
  batch: VelocityPolicyBatch,
  std: float,
  microbatch: int,
) -> torch.Tensor:
  sums = batch.mean.new_zeros(2, dtype=torch.float64)
  for begin in range(0, len(batch), microbatch):
    end = begin + microbatch
    mean = predict(batch.history[begin:end])
    kl = 0.5 * ((mean - batch.mean[begin:end]) / std).square().sum(-1)
    sums[0] += kl.double().sum()
    sums[1] += kl.numel()
  if dist.is_initialized():
    dist.all_reduce(sums)
  return sums[0] / sums[1]


__all__ = [
  "ACTION_DIM",
  "ARM_DIM",
  "CONTROL_DT",
  "DEFAULT_HISTORY",
  "GAIT_PERIOD",
  "OBS_DIM",
  "OBS_TERMS",
  "TERM_DIVISORS",
  "TERM_OFFSETS",
  "VelocityMlpActor",
  "VelocityPolicyBatch",
  "VelocityPolicyRollout",
  "actor_architecture_metadata",
  "constrain_policy_kl",
  "make_actor",
  "migrate_actor_state_dict",
  "observation_noise_std",
  "unflatten_native_history",
  "velocity_observation",
  "velocity_policy_kl",
  "velocity_target",
]

