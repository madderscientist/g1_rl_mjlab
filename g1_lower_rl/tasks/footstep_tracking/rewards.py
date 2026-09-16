"""mjlab reward terms; goals are supplied by a separate footstep command provider."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from g1_lower_rl.assets import LOWER_BODY_JOINTS
from g1_lower_rl.footstep_phase import FootstepPhaseCfg, resolve_phase_cfg
from g1_lower_rl.tasks.footstep_tracking.reward_math import (
  contact_schedule_score,
  footprint_accuracy_score,
  footprint_errors,
  footprint_tracking_score,
  joint_edge_cost,
  phase_windows,
  torque_square_cost,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.reward_manager import RewardTermCfg


@dataclass
class FootstepRewardState:
  """Pre-queue-advance snapshot, ordered [left, right], in world coordinates."""

  phase: torch.Tensor
  targets_w: torch.Tensor
  target_ids: torch.Tensor
  ground_height: torch.Tensor
  frequency: torch.Tensor

  def __post_init__(self) -> None:
    if self.phase.ndim != 1:
      raise ValueError("phase must have shape [num_envs]")
    num_envs = self.phase.shape[0]
    if self.targets_w.shape != (num_envs, 2, 3):
      raise ValueError("targets_w must have shape [num_envs, 2, 3] for world XY/yaw")
    if self.target_ids.shape != (num_envs, 2) or self.target_ids.dtype != torch.long:
      raise ValueError("target_ids must be int64 with shape [num_envs, 2]")
    if self.ground_height.shape != (num_envs, 2):
      raise ValueError("ground_height must have shape [num_envs, 2]")
    if self.frequency.shape != (num_envs,) or (self.frequency < 0).any():
      raise ValueError("frequency must be nonnegative with shape [num_envs]")
    if (self.target_ids < 0).any():
      raise ValueError("target_ids must be nonnegative")
    for tensor in (self.phase, self.targets_w, self.target_ids, self.ground_height, self.frequency):
      if tensor.device != self.phase.device or not torch.isfinite(tensor).all():
        raise ValueError("Reward state must be finite and on one device")


class LowerBodyTorqueCost:
  """Sum of actual actuator torques squared over exactly the 15 controlled joints."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    asset_cfg = cfg.params["asset_cfg"]
    asset = env.scene[asset_cfg.name]
    names = [asset.actuator_names[index] for index in asset_cfg.actuator_ids]
    if len(names) != 15 or set(names) != set(LOWER_BODY_JOINTS):
      raise ValueError("Torque cost must select exactly the 15 leg and waist actuators")
    self.actuator_ids = list(asset_cfg.actuator_ids)
    coefficients = cfg.params.get("copper_weights")
    if coefficients is not None and set(coefficients) != set(names):
      raise ValueError("copper_weights must name every controlled actuator exactly once")
    weights = [1.0 if coefficients is None else coefficients[name] for name in names]
    if not all(math.isfinite(value) and value > 0 for value in weights):
      raise ValueError("Copper weights must be finite and positive")
    self.weights = torch.tensor(weights, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    reference_torque: float = 100.0,
    copper_weights: dict[str, float] | None = None,
  ) -> torch.Tensor:
    del copper_weights
    torques = env.scene[asset_cfg.name].data.actuator_force[:, self.actuator_ids]
    return torque_square_cost(torques, self.weights, reference_torque)


def joint_zero_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Absolute joint angle penalty; deliberately not relative to the default pose."""
  return env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids].square().sum(-1)


class JointEdgeCost:
  """Soft margin against the actual hard joint limits, not the nominal pose."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    asset_cfg = cfg.params["asset_cfg"]
    limits = env.scene[asset_cfg.name].data.joint_pos_limits[:, asset_cfg.joint_ids]
    if not torch.isfinite(limits).all() or not (limits[..., 1] > limits[..., 0]).all():
      raise ValueError("Selected joints need finite, strictly ordered hard limits")

  def __call__(self, env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, margin_fraction: float = 0.15) -> torch.Tensor:
    data = env.scene[asset_cfg.name].data
    return joint_edge_cost(
      data.joint_pos[:, asset_cfg.joint_ids],
      data.joint_pos_limits[:, asset_cfg.joint_ids],
      margin_fraction,
    )


def fall_cost(env: ManagerBasedRlEnv) -> torch.Tensor:
  """A terminal event cost, independent of dt; timeouts are not falls."""
  return env.termination_manager.terminated.to(torch.float32) / env.step_dt


class FootstepReward:
  """Phase-aware foot geometry and contact terms, including latched landing accuracy."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    if cfg.params["component"] not in {"landing", "support", "approach", "schedule", "clearance", "slip"}:
      raise ValueError("Unknown footstep reward component")
    asset_cfg = cfg.params["asset_cfg"]
    asset = env.scene[asset_cfg.name]
    if [asset.site_names[index] for index in asset_cfg.site_ids] != ["left_foot", "right_foot"]:
      raise ValueError("Foot sites must be ordered [left_foot, right_foot]")
    shape = (env.num_envs, 2)
    self.last_ids = torch.full(shape, -1, dtype=torch.long, device=env.device)
    self.saw_air = torch.zeros(shape, dtype=torch.bool, device=env.device)
    self.previous_contact = torch.ones_like(self.saw_air)
    self.landed = torch.zeros_like(self.saw_air)
    self.landing_score = torch.zeros(shape, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    selected = slice(None) if env_ids is None else env_ids
    self.last_ids[selected] = -1
    self.saw_air[selected] = False
    self.previous_contact[selected] = True
    self.landed[selected] = False
    self.landing_score[selected] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    asset_cfg: SceneEntityCfg,
    component: str,
    stance_fraction: float | None = None,
    position_std: float = 0.08,
    yaw_std: float = 0.2,
    landing_window: float = 0.25,
    clearance: float = 0.08,
    phase_cfg: FootstepPhaseCfg | None = None,
  ) -> torch.Tensor:
    state = env.command_manager.get_term(command_name).reward_state
    if not isinstance(state, FootstepRewardState):
      raise TypeError("Footstep command must publish a FootstepRewardState as reward_state")
    if position_std <= 0 or yaw_std <= 0 or clearance <= 0 or not 0 < landing_window <= 1:
      raise ValueError("Invalid tracking, clearance or landing-window parameters")
    timing = resolve_phase_cfg(stance_fraction, phase_cfg)
    stance, swing_progress = phase_windows(state.phase, phase_cfg=timing)
    standing = state.frequency == 0
    stance = stance | standing.unsqueeze(-1)
    swing_progress = torch.where(standing.unsqueeze(-1), 0.0, swing_progress)
    contact = env.scene[sensor_name].data.found > 0
    if contact.shape != stance.shape:
      raise ValueError("Contact sensor must provide [num_envs, 2] in left/right foot order")
    data = env.scene[asset_cfg.name].data
    position = data.site_pos_w[:, asset_cfg.site_ids]
    quaternion = data.site_quat_w[:, asset_cfg.site_ids]
    scalar, roll, pitch, yaw = quaternion.unbind(-1)
    foot_yaw = torch.atan2(2 * (scalar * yaw + roll * pitch), 1 - 2 * (pitch.square() + yaw.square()))
    actual = torch.cat((position[..., :2], foot_yaw.unsqueeze(-1)), dim=-1)
    distance, yaw_error = footprint_errors(actual, state.targets_w)

    if component == "landing":
      changed = (self.last_ids != state.target_ids) | standing.unsqueeze(-1)
      self.landed[changed] = False
      self.saw_air[changed] = False
      self.landing_score[changed] = 0.0
      self.saw_air |= ~contact & ~stance
      first_contact = contact & ~self.previous_contact & self.saw_air & ~self.landed
      touchdown = state.phase.new_tensor([timing.left_stance_phase, timing.right_stance_phase])
      phase_error = state.phase.unsqueeze(-1) - touchdown
      distance_to_touchdown = torch.atan2(phase_error.sin(), phase_error.cos()).abs() / (2 * math.pi)
      swing_duration = 1.0 - state.phase.new_tensor(timing.stance_fractions)
      allowed_window = torch.maximum(
        landing_window * swing_duration, state.phase.new_tensor(timing.contact_half_width / (2 * math.pi))
      )
      permitted = distance_to_touchdown <= allowed_window
      score = footprint_accuracy_score(distance, yaw_error, position_std, yaw_std)
      self.landing_score[first_contact] = (score * permitted)[first_contact]
      self.landed |= first_contact
      self.previous_contact.copy_(contact)
      self.last_ids.copy_(state.target_ids)
      landing = (self.landing_score * stance * contact).sum(-1) / stance.sum(-1).clamp_min(1)
      hold = footprint_tracking_score(actual, state.targets_w, stance, contact, position_std, yaw_std)
      return torch.where(standing, hold, landing)
    if component == "support":
      return footprint_tracking_score(actual, state.targets_w, stance, contact, position_std, yaw_std)
    if component == "approach":
      return (distance * ~stance * swing_progress.square()).sum(-1)
    if component == "schedule":
      return contact_schedule_score(stance, contact)
    if component == "clearance":
      target_height = clearance * torch.sin(torch.pi * swing_progress)
      height = position[..., 2] - state.ground_height
      return (((target_height - height).clamp_min(0) / clearance).square() * ~stance).sum(-1)
    if component == "slip":
      speed = torch.linalg.vector_norm(data.site_lin_vel_w[:, asset_cfg.site_ids, :2], dim=-1)
      yaw_rate = data.site_ang_vel_w[:, asset_cfg.site_ids, 2].abs()
      return ((speed + 0.05 * yaw_rate) * contact).sum(-1)
    raise ValueError(f"Unknown component {component}")
