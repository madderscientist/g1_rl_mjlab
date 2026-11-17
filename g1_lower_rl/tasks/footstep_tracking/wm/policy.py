"""Twin-Mirror 10-frame history MLP actor, zero-DC encoding, and KL trust region for footstep WM."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn

from g1_lower_rl.assets import ARM_JOINTS, LOWER_BODY_JOINTS, WHOLE_BODY_JOINTS
from g1_lower_rl.footsteps.tensor_math import to_local
from g1_lower_rl.rl.footstep_model import nominal_joint_positions
from g1_lower_rl.rl.mirror import _pair_joints
from g1_lower_rl.rl.supermimic import constrain_policy_kl, lower_body_actuator_target
from g1_lower_rl.world_model.adapters.math import conjugate, rotate, unit
from g1_lower_rl.world_model.model import JOINT_NAMES

DEFAULT_HISTORY = 10
ACTION_DIM = 15
ARM_DIM = 14
JOINT_DIM = 29
PROPRIO_RAW_DIM = 70
FRAME_DIM = PROPRIO_RAW_DIM + ACTION_DIM  # 85D per history tick (zero at upright default standing)
CMD_ENC_DIM = 13  # Zero-centered sagittal-equivariant command encoding
RAW_OBS_DIM = 79  # Deployment contract: 70D proprio + [phase, freq, L_next(3), R_next(3), theta]

if tuple(WHOLE_BODY_JOINTS) != JOINT_NAMES:
  raise AssertionError("WHOLE_BODY_JOINTS must match canonical PositionWorldModel JOINT_NAMES")
if tuple(LOWER_BODY_JOINTS) != JOINT_NAMES[:ACTION_DIM] or tuple(ARM_JOINTS) != JOINT_NAMES[ACTION_DIM:]:
  raise AssertionError("LOWER_BODY_JOINTS[:15] + ARM_JOINTS[15:29] must match JOINT_NAMES")


def proprio_noise_std(device: torch.device | str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
  """Per-tick raw sensor noise std on the 70D proprioception slice before frame scaling."""
  return torch.tensor(
    [0.006] * JOINT_DIM
    + [0.87] * JOINT_DIM
    + [0.115] * 3
    + [0.029] * 3
    + [0.115] * 3
    + [0.029] * 3,
    device=device,
    dtype=dtype,
  )


def build_twin_mirror_indices(
  history_length: int = DEFAULT_HISTORY,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Construct exact sagittal reflection permutations and signs for [H*85 + 13] -> [15]."""
  q_perm, q_sign = _pair_joints(list(WHOLE_BODY_JOINTS))
  act_perm, act_sign = _pair_joints(list(LOWER_BODY_JOINTS))

  frame_perm = (
    q_perm
    + [JOINT_DIM + i for i in q_perm]
    + list(range(58, 70))
    + [70 + i for i in act_perm]
  )
  frame_sign = (
    q_sign
    + q_sign
    + [-1.0, 1.0, -1.0]  # pelvis_ang_vel
    + [1.0, -1.0, 1.0]   # pelvis_grav - (0, 0, -1)
    + [-1.0, 1.0, -1.0]  # torso_ang_vel
    + [1.0, -1.0, 1.0]   # torso_grav - (0, 0, -1)
    + act_sign
  )
  if len(frame_perm) != FRAME_DIM or len(frame_sign) != FRAME_DIM:
    raise AssertionError("Incorrect 85D frame mirror layout")

  hist_perm: list[int] = []
  hist_sign: list[float] = []
  for h in range(history_length):
    base = h * FRAME_DIM
    hist_perm.extend(base + idx for idx in frame_perm)
    hist_sign.extend(frame_sign)

  cmd_base = history_length * FRAME_DIM
  # Command [0..12]:
  #   0: sin(phase) * move_gate          -> (-1)
  #   1: cos(phase) * move_gate          -> (-1)
  #   2: 2 * move_gate - 1               -> (+1)
  #   3: ((freq - 1.2) / 0.5) * move_gate-> (+1)
  #   4..7:  L_next [x/0.35, (y-0.12)/0.25, sin(yaw)/0.5, (cos(yaw)-1)/0.2] <-> 8..11
  #   8..11: R_next [x/0.35, (y+0.12)/0.25, sin(yaw)/0.5, (cos(yaw)-1)/0.2] <-> 4..7
  #   12: theta / 0.8                    -> (-1)
  cmd_perm = [
    cmd_base + 0,
    cmd_base + 1,
    cmd_base + 2,
    cmd_base + 3,
    cmd_base + 8,
    cmd_base + 9,
    cmd_base + 10,
    cmd_base + 11,
    cmd_base + 4,
    cmd_base + 5,
    cmd_base + 6,
    cmd_base + 7,
    cmd_base + 12,
  ]
  cmd_sign = [
    -1.0,
    -1.0,
    1.0,
    1.0,
    1.0,
    -1.0,
    -1.0,
    1.0,
    1.0,
    -1.0,
    -1.0,
    1.0,
    -1.0,
  ]

  return (
    torch.tensor(hist_perm + cmd_perm, dtype=torch.long),
    torch.tensor(hist_sign + cmd_sign, dtype=torch.float32),
    torch.tensor(act_perm, dtype=torch.long),
    torch.tensor(act_sign, dtype=torch.float32),
  )


def encode_proprio_frame(
  raw_proprio_70: torch.Tensor,
  prev_action_15: torch.Tensor,
  default_joint_pos_29: torch.Tensor,
) -> torch.Tensor:
  """Zero-center and physically scale one 70D proprioception + 15D previous-action frame."""
  q_rel = raw_proprio_70[..., 0:29] - default_joint_pos_29
  dq_scaled = raw_proprio_70[..., 29:58] * 0.2
  pelvis_ang = raw_proprio_70[..., 58:61] * 0.5
  pelvis_grav_rel = raw_proprio_70[..., 61:64].clone()
  pelvis_grav_rel[..., 2] = pelvis_grav_rel[..., 2] + 1.0
  torso_ang = raw_proprio_70[..., 64:67] * 0.5
  torso_grav_rel = raw_proprio_70[..., 67:70].clone()
  torso_grav_rel[..., 2] = torso_grav_rel[..., 2] + 1.0
  return torch.cat(
    (q_rel, dq_scaled, pelvis_ang, pelvis_grav_rel, torso_ang, torso_grav_rel, prev_action_15),
    dim=-1,
  )


def encode_command(
  phase: torch.Tensor,
  frequency: torch.Tensor,
  local_footprints: torch.Tensor,
  theta: torch.Tensor,
) -> torch.Tensor:
  """Encode [phase, frequency, local_footprints(2, 3), theta] into zero-centered 13D vector.

  When `frequency == 0` (`STANDING`), `anchor` is frozen in world space so `local_footprints`
  carry zero drift feedback and only inject static asymmetric bias; `fp_gate` zeroes them at `f == 0`
  while preserving 100% of footprints throughout active `STARTING` / `WALKING` / `STOPPING` (`f >= 0.005`).
  """
  move_gate = (frequency / 0.005).clamp(0.0, 1.0)
  fp_gate = move_gate
  sin_p = torch.sin(phase) * move_gate
  cos_p = torch.cos(phase) * move_gate
  mode_centered = 2.0 * move_gate - 1.0
  freq_centered = ((frequency - 1.2) / 0.5) * move_gate

  lx = (local_footprints[..., 0, 0] / 0.35) * fp_gate
  ly = ((local_footprints[..., 0, 1] - 0.12) / 0.25) * fp_gate
  lsin = (torch.sin(local_footprints[..., 0, 2]) / 0.5) * fp_gate
  lcos = ((torch.cos(local_footprints[..., 0, 2]) - 1.0) / 0.2) * fp_gate

  rx = (local_footprints[..., 1, 0] / 0.35) * fp_gate
  ry = ((local_footprints[..., 1, 1] + 0.12) / 0.25) * fp_gate
  rsin = (torch.sin(local_footprints[..., 1, 2]) / 0.5) * fp_gate
  rcos = ((torch.cos(local_footprints[..., 1, 2]) - 1.0) / 0.2) * fp_gate

  theta_scaled = theta / 0.8
  return torch.stack(
    (sin_p, cos_p, mode_centered, freq_centered, lx, ly, lsin, lcos, rx, ry, rsin, rcos, theta_scaled),
    dim=-1,
  )


def raw_proprio_from_state(
  state: torch.Tensor,
  encoder_bias_29: torch.Tensor,
  torso_quat: torch.Tensor,
  torso_ang_vel_w: torch.Tensor,
  imu_bias_12: torch.Tensor | None = None,
) -> torch.Tensor:
  """Construct 70D raw proprioception slice matching footstep_env_cfg actor observation."""
  q = state[..., 7:36] + encoder_bias_29
  dq = state[..., 42:71]
  pelvis_q = unit(state[..., 3:7])
  gravity = torch.zeros_like(state[..., :3])
  gravity[..., 2] = -1.0
  pelvis_ang = state[..., 39:42]
  pelvis_grav = rotate(conjugate(pelvis_q), gravity)
  torso_q = unit(torso_quat)
  torso_ang = rotate(conjugate(torso_q), torso_ang_vel_w)
  torso_grav = rotate(conjugate(torso_q), gravity)
  if imu_bias_12 is not None:
    pelvis_ang = pelvis_ang + imu_bias_12[..., 0:3]
    pelvis_grav = pelvis_grav + imu_bias_12[..., 3:6]
    torso_ang = torso_ang + imu_bias_12[..., 6:9]
    torso_grav = torso_grav + imu_bias_12[..., 9:12]
  return torch.cat((q, dq, pelvis_ang, pelvis_grav, torso_ang, torso_grav), dim=-1)


def command_from_world_queue(
  phase: torch.Tensor,
  frequency: torch.Tensor,
  queue: torch.Tensor,
  first_side: torch.Tensor,
  anchor: torch.Tensor,
  theta: torch.Tensor,
) -> torch.Tensor:
  """Project next [L_next, R_next] world footprints into anchor frame and encode 13D command."""
  sides = torch.arange(2, device=first_side.device)
  indices = (sides[None, :] != first_side[:, None]).long()
  future_world = queue.gather(1, indices[:, :, None].expand(-1, -1, 3))
  local_fp = to_local(future_world, anchor[:, None, :])
  return encode_command(phase, frequency, local_fp, theta)


footstep_target = lower_body_actuator_target


class FootstepTwinMirrorMlpActor(nn.Module):
  """Analytically sagittal-symmetric Twin-Mirror MLP mapping (history[B, H, 85], cmd[B, 13]) -> [B, 15]."""

  default_joint_pos: torch.Tensor
  obs_mirror_perm: torch.Tensor
  obs_mirror_sign: torch.Tensor
  act_mirror_perm: torch.Tensor
  act_mirror_sign: torch.Tensor

  def __init__(
    self,
    history_length: int = DEFAULT_HISTORY,
    hidden_dims: tuple[int, ...] = (1024, 512, 256),
    default_joint_pos: Sequence[float] | None = None,
  ):
    super().__init__()
    if history_length < 1 or not hidden_dims or min(hidden_dims) < 1:
      raise ValueError("history_length and hidden_dims must be positive")
    self.history_length = int(history_length)
    self.hidden_dims = tuple(int(w) for w in hidden_dims)
    defaults = torch.tensor(
      nominal_joint_positions() if default_joint_pos is None else default_joint_pos,
      dtype=torch.float32,
    )
    if defaults.shape != (JOINT_DIM,) or not torch.isfinite(defaults).all():
      raise ValueError("default_joint_pos must contain 29 finite joint angles")
    self.register_buffer("default_joint_pos", defaults)

    obs_perm, obs_sign, act_perm, act_sign = build_twin_mirror_indices(self.history_length)
    self.register_buffer("obs_mirror_perm", obs_perm, persistent=False)
    self.register_buffer("obs_mirror_sign", obs_sign, persistent=False)
    self.register_buffer("act_mirror_perm", act_perm, persistent=False)
    self.register_buffer("act_mirror_sign", act_sign, persistent=False)

    in_dim = self.history_length * FRAME_DIM + CMD_ENC_DIM
    layers: list[nn.Module] = []
    for width in self.hidden_dims:
      layers.extend((nn.Linear(in_dim, width), nn.SiLU()))
      in_dim = width
    last = nn.Linear(in_dim, ACTION_DIM)
    nn.init.orthogonal_(last.weight, gain=0.01)
    nn.init.zeros_(last.bias)
    layers.append(last)
    self.net = nn.Sequential(*layers)

  def forward(self, history: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
    if not torch.jit.is_tracing():
      if history.ndim != 3 or history.shape[1:] != (self.history_length, FRAME_DIM):
        raise ValueError(f"Expected [batch, {self.history_length}, {FRAME_DIM}] history, got {tuple(history.shape)}")
      if command.ndim != 2 or command.shape[-1] != CMD_ENC_DIM:
        raise ValueError(f"Expected [batch, {CMD_ENC_DIM}] command, got {tuple(command.shape)}")
    x = torch.cat((history.flatten(1), command), dim=-1)
    xm = x.index_select(-1, self.obs_mirror_perm) * self.obs_mirror_sign
    y, ym = self.net(torch.cat((x, xm), dim=0)).chunk(2, dim=0)
    ym_unmirrored = ym.index_select(-1, self.act_mirror_perm) * self.act_mirror_sign
    return 0.5 * (y + ym_unmirrored)


def actor_architecture_metadata(actor: FootstepTwinMirrorMlpActor) -> dict[str, object]:
  return {
    "type": "footstep_twin_mirror_history_mlp",
    "history_length": actor.history_length,
    "frame_dim": FRAME_DIM,
    "cmd_dim": CMD_ENC_DIM,
    "input_dim": actor.history_length * FRAME_DIM + CMD_ENC_DIM,
    "hidden_dims": list(actor.hidden_dims),
    "action_dim": ACTION_DIM,
    "activation": "SiLU",
    "twin_mirror": True,
    "zero_centered_standing": True,
  }


def make_actor(
  device: str | torch.device,
  history_length: int = DEFAULT_HISTORY,
  hidden_dims: tuple[int, ...] = (1024, 512, 256),
) -> FootstepTwinMirrorMlpActor:
  return FootstepTwinMirrorMlpActor(history_length=history_length, hidden_dims=hidden_dims).to(device)


@dataclass(frozen=True)
class FootstepPolicyBatch:
  history: torch.Tensor
  command: torch.Tensor
  mean: torch.Tensor

  def __len__(self) -> int:
    return len(self.mean)


class FootstepPolicyRollout:
  def __init__(self, train_envs: int):
    self.train_envs = train_envs
    self.records: list[dict[str, torch.Tensor]] = []

  @torch.no_grad()
  def append(self, *, history: torch.Tensor, command: torch.Tensor, mean: torch.Tensor) -> None:
    if min(len(history), len(command), len(mean)) < self.train_envs:
      raise ValueError("Native footstep policy record does not contain all training worlds")
    self.records.append(
      {
        "history": history[: self.train_envs].detach().clone(),
        "command": command[: self.train_envs].detach().clone(),
        "mean": mean[: self.train_envs].detach().clone(),
      }
    )

  @torch.no_grad()
  def finish(self) -> FootstepPolicyBatch:
    if not self.records:
      raise ValueError("A fresh nonempty native footstep rollout is required")
    history = torch.stack([r["history"] for r in self.records]).flatten(0, 1)
    command = torch.stack([r["command"] for r in self.records]).flatten(0, 1)
    mean = torch.stack([r["mean"] for r in self.records]).flatten(0, 1)
    if not (torch.isfinite(history).all() and torch.isfinite(command).all() and torch.isfinite(mean).all()):
      raise FloatingPointError("Nonfinite native footstep policy rollout")
    self.records.clear()
    return FootstepPolicyBatch(history=history, command=command, mean=mean)


@torch.no_grad()
def footstep_policy_kl(
  predict: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
  batch: FootstepPolicyBatch,
  std: float,
  microbatch: int,
) -> torch.Tensor:
  sums = batch.mean.new_zeros(2, dtype=torch.float64)
  for begin in range(0, len(batch), microbatch):
    end = begin + microbatch
    mean = predict(batch.history[begin:end], batch.command[begin:end])
    kl = 0.5 * ((mean - batch.mean[begin:end]) / std).square().sum(-1)
    sums[0] += kl.double().sum()
    sums[1] += kl.numel()
  if dist.is_initialized():
    dist.all_reduce(sums)
  return sums[0] / sums[1]


__all__ = [
  "ACTION_DIM",
  "ARM_DIM",
  "CMD_ENC_DIM",
  "DEFAULT_HISTORY",
  "FRAME_DIM",
  "JOINT_DIM",
  "PROPRIO_RAW_DIM",
  "RAW_OBS_DIM",
  "FootstepPolicyBatch",
  "FootstepPolicyRollout",
  "FootstepTwinMirrorMlpActor",
  "actor_architecture_metadata",
  "build_twin_mirror_indices",
  "command_from_world_queue",
  "constrain_policy_kl",
  "encode_command",
  "encode_proprio_frame",
  "footstep_policy_kl",
  "footstep_target",
  "make_actor",
  "proprio_noise_std",
  "raw_proprio_from_state",
]
