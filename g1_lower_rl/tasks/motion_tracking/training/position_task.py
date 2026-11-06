"""Analytic task semantics and split-safe histories around the position WM."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Literal, NamedTuple

import torch
from torch import nn

from g1_lower_rl.world_model.adapters.kinematics import TorchKinematics
from g1_lower_rl.world_model.adapters.math import conjugate, heading, rotate, wrap, yaw
from g1_lower_rl.world_model.adapters.tasks import MotionConfig, MotionGoal, MotionTrackingAdapter
from g1_lower_rl.world_model.model import PositionWorldModel, state_features
from g1_lower_rl.world_model.training.replay import RingReplay

from .position_policy import HISTORY, OBS_DIM

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnvCfg

TaskDefinition = Literal["recovery_xy_light_v1"]
RECOVERY_XY_TASK: TaskDefinition = "recovery_xy_light_v1"


def position_task_contract(definition: TaskDefinition, gamma: float = 0.99) -> dict[str, str | float]:
  if definition != RECOVERY_XY_TASK or not 0 < gamma < 1:
    raise ValueError("Unknown position-task definition or invalid discount")
  velocity_weight, position_weight = 0.2, 0.15
  maximum = 0.02 * (5 + (velocity_weight + position_weight))
  contract: dict[str, str | float] = {
    "definition": definition, "termination_probability": 0.005,
    "episode_length_s": 20.0,
    "pelvis_height_threshold": 0.25, "pelvis_gravity_z_threshold": 0.8,
    "anchor_velocity_weight": velocity_weight, "anchor_velocity_std": 0.4,
    "anchor_position_coarse_weight": position_weight, "anchor_position_coarse_std": 1.2,
    "maximum_step_reward": maximum, "sampler_g_max": maximum / (1 - gamma),
    "anchor_position_coarse_axes": "xy", "anchor_velocity_axes": "xyz",
  }
  return contract


def configure_native_task(
  cfg: ManagerBasedRlEnvCfg, definition: TaskDefinition, gamma: float = 0.99,
) -> dict[str, str | float]:
  from mjlab.managers.reward_manager import RewardTermCfg
  from mjlab.managers.termination_manager import TerminationTermCfg

  from g1_lower_rl.tasks.motion_tracking.mdp.rewards import motion_anchor_lin_vel_error_exp, motion_anchor_pos_xy_error_exp
  from g1_lower_rl.tasks.motion_tracking.mdp.terminations import with_probabilistic_termination

  contract = position_task_contract(definition, gamma)
  cfg.episode_length_s = float(contract["episode_length_s"])
  cfg.commands["motion"] = replace(cfg.commands["motion"], gamma=gamma, g_max=float(contract["sampler_g_max"]))
  cfg.rewards["motion_anchor_lin_vel"] = RewardTermCfg(
    func=motion_anchor_lin_vel_error_exp, weight=float(contract["anchor_velocity_weight"]),
    params={"command_name": "motion", "std": float(contract["anchor_velocity_std"])},
  )
  cfg.rewards["motion_global_root_pos_coarse"] = replace(
    cfg.rewards["motion_global_root_pos"], weight=float(contract["anchor_position_coarse_weight"]),
    params={"command_name": "motion", "std": float(contract["anchor_position_coarse_std"])},
    func=motion_anchor_pos_xy_error_exp,
  )
  for name in ("root_height", "root_tilt"):
    term = cfg.terminations[name]
    cfg.terminations[name] = TerminationTermCfg(
      func=with_probabilistic_termination,
      params={"base_func": term.func, "p_term": float(contract["termination_probability"]), **term.params},
    )
  return contract


class PolicyStart(NamedTuple):
  state: torch.Tensor
  history: torch.Tensor
  context: torch.Tensor
  frame: torch.Tensor
  bias: torch.Tensor
  remaining: torch.Tensor
  clip_end: torch.Tensor


class PositionReplay(RingReplay):
  def __init__(self, capacity: int, num_envs: int, validation_envs: int, device: str | torch.device):
    super().__init__(capacity, num_envs, validation_envs, device)
    self.latest = torch.empty(capacity, num_envs, OBS_DIM, device=self.device)
    self.frame = torch.empty(capacity, num_envs, dtype=torch.long, device=self.device)
    self.bias = torch.empty(capacity, num_envs, 29, device=self.device)
    self.remaining = torch.empty_like(self.frame)
    self.clip_end = torch.empty(capacity, num_envs, dtype=torch.bool, device=self.device)

  def append_control(self, batch, record: dict[str, torch.Tensor]) -> None:
    if batch.state.shape[0] != 1:
      raise ValueError("Policy replay appends one control tick at a time")
    row = self.total_steps % self.capacity
    for key in ("latest", "frame", "bias", "remaining", "clip_end"):
      destination, source = getattr(self, key)[row], record[key]
      if source.shape != destination.shape or source.dtype != destination.dtype or source.device != self.device:
        raise ValueError(f"Incorrect policy replay field: {key}")
      if source.is_floating_point() and not torch.isfinite(source).all():
        raise FloatingPointError(f"Nonfinite policy replay field: {key}")
      destination.copy_(source)
    self.append(batch)

  def sample_starts(self, batch_size: int, *, validation: bool = False) -> PolicyStart:
    window = self.sample(batch_size, HISTORY, validation=validation)
    rows = window.sequence % self.capacity
    env = window.env[:, None]
    return PolicyStart(
      window.state[:, -1], self.latest[rows, env], window.inputs[:, -1, 97:],
      self.frame[rows[:, -1], window.env], self.bias[rows[:, -1], window.env],
      self.remaining[rows[:, -1], window.env],
      self.clip_end[rows[:, -1], window.env],
    )


class PositionTask(nn.Module):
  def __init__(
    self, kinematics: TorchKinematics, default_position: torch.Tensor, soft_limits: torch.Tensor,
    definition: TaskDefinition = RECOVERY_XY_TASK, gamma: float = 0.99,
  ):
    super().__init__()
    self.definition = definition
    self.contract = position_task_contract(definition, gamma)
    self.adapter = MotionTrackingAdapter(
      kinematics, MotionConfig(anchor_position_coarse_axes="xy"),
      default_joint_position=default_position,
    )
    self.register_buffer("soft_limits", soft_limits.clone())

  def reward(self, state: torch.Tensor, reference: torch.Tensor, action: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
    rewards = self.adapter(state, MotionGoal(reference)).rewards
    tracking = (
      0.5 * (rewards["anchor_position"] + rewards["anchor_orientation"])
      + rewards["body_position"] + rewards["body_orientation"]
      + rewards["body_linear_velocity"] + rewards["body_angular_velocity"]
    )
    tracking = (tracking + float(self.contract["anchor_velocity_weight"]) * rewards["anchor_linear_velocity"]
                + float(self.contract["anchor_position_coarse_weight"]) * rewards["anchor_position_coarse"])
    q = state[..., 7:36]
    limit = (self.soft_limits[..., 0] - q).clamp_min(0) + (q - self.soft_limits[..., 1]).clamp_min(0)
    return 0.02 * (tracking - 0.1 * (action - previous).square().sum(-1) - 2 * limit.sum(-1))

  def termination(self, state: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    probability = float(self.contract["termination_probability"])
    return probability * torch.sigmoid(20 * self.failure_margin(state, reference))

  @staticmethod
  def failure_margin(state: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    gravity = torch.zeros_like(state[..., :3])
    gravity[..., 2] = -1
    tilt = (
      rotate(conjugate(state[..., 3:7]), gravity)[..., 2]
      - rotate(conjugate(reference[..., 3:7]), gravity)[..., 2]
    ).abs() / 0.8
    height = (state[..., 2] - reference[..., 2]).abs() / 0.25
    return torch.maximum(height, tilt) - 1

  @staticmethod
  def critic_features(
    model: PositionWorldModel, state: torch.Tensor, context: torch.Tensor, reference: torch.Tensor, window: torch.Tensor,
  ) -> torch.Tensor:
    feature = (state_features(state, validate=False) - model.input_mean[:68]) / model.input_scale[:68]
    physical = (context - model.input_mean[97:]) / model.input_scale[97:]
    offset = rotate(conjugate(heading(reference[..., 3:7])), state[..., :3] - reference[..., :3])
    yaw_error = wrap(yaw(state[..., 3:7]) - yaw(reference[..., 3:7])).unsqueeze(-1)
    return torch.cat((feature, physical, window, offset, yaw_error), dim=-1)
