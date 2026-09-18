"""适配 mjlab 的奖励项，目标由独立脚步命令提供器发布"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from g1_lower_rl.assets import LOWER_BODY_JOINTS
from g1_lower_rl.footstep_phase import FootstepPhaseCfg, resolve_phase_cfg
from g1_lower_rl.tasks.footstep_tracking.reward_math import (
  contact_schedule_score,
  footprint_accuracy_cost,
  joint_edge_cost,
  phase_windows,
  swing_tracking_score,
  torque_square_cost,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.reward_manager import RewardTermCfg


@dataclass
class FootstepRewardState:
  """队列推进前的世界系执行快照，固定按左脚、右脚排列"""

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
  """仅对15个受控关节对应的实际执行器力矩平方求和"""

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


def body_tilt_angle_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """计算相对竖直方向的弧度倾角平方，不受世界航向角影响"""
  data = env.scene[asset_cfg.name].data
  orientation = data.body_link_quat_w[:, asset_cfg.body_ids].squeeze(1)
  gravity = quat_apply_inverse(orientation, data.gravity_vec_w)
  angle = torch.atan2(torch.linalg.vector_norm(gravity[:, :2], dim=-1), -gravity[:, 2])
  return angle.square()


def head_height_reward(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  command_name: str = "footsteps",
  sensor_name: str = "feet_ground_contact",
  height_cap: float = 1.254,
) -> torch.Tensor:
  """奖励头部几何体的离地高度，腾空或失败时不计奖励"""
  if not math.isfinite(height_cap) or height_cap <= 0:
    raise ValueError("height_cap must be finite and positive")
  state = env.command_manager.get_term(command_name).reward_state
  data = env.scene[asset_cfg.name].data
  head_z = data.geom_pos_w[:, asset_cfg.geom_ids, 2].squeeze(-1)
  height = head_z - state.ground_height.mean(-1)
  supported = (env.scene[sensor_name].data.found > 0).any(-1)
  valid = supported & ~env.termination_manager.terminated
  return (height / height_cap).clamp(0.0, 1.0) * valid


class FilteredPelvisUpright:
  """按步态频率调整低通滤波器，惩罚滤波后的骨盆重力向量偏斜"""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    asset_cfg = cfg.params["asset_cfg"]
    asset = env.scene[asset_cfg.name]
    if [asset.body_names[index] for index in asset_cfg.body_ids] != ["pelvis"]:
      raise ValueError("Filtered pelvis reward must select only the pelvis body")
    self.filtered_gravity = torch.zeros((env.num_envs, 3), device=env.device)
    self.initialized = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    self.last_step = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    selected = slice(None) if env_ids is None else env_ids
    self.filtered_gravity[selected] = 0.0
    self.initialized[selected] = False
    self.last_step[selected] = -1

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "footsteps",
    cutoff_ratio: float = 0.25,
    standing_cutoff_hz: float = 0.2,
  ) -> torch.Tensor:
    if not all(math.isfinite(value) and value > 0 for value in (cutoff_ratio, standing_cutoff_hz, env.step_dt)):
      raise ValueError("Filter cutoffs and step_dt must be finite and positive")
    data = env.scene[asset_cfg.name].data
    orientation = data.body_link_quat_w[:, asset_cfg.body_ids].squeeze(1)
    gravity = quat_apply_inverse(orientation, data.gravity_vec_w)
    frequency = env.command_manager.get_term(command_name).reward_state.frequency
    cutoff = torch.where(frequency > 0, frequency * cutoff_ratio, standing_cutoff_hz)
    alpha = -torch.expm1(-2 * math.pi * cutoff * env.step_dt)
    fresh = ~self.initialized
    self.filtered_gravity.copy_(torch.where(fresh.unsqueeze(-1), gravity, self.filtered_gravity))
    update = self.last_step != env.common_step_counter
    filtered = self.filtered_gravity + alpha.unsqueeze(-1) * (gravity - self.filtered_gravity)
    self.filtered_gravity.copy_(torch.where(update.unsqueeze(-1), filtered, self.filtered_gravity))
    self.initialized |= update
    self.last_step.masked_fill_(update, env.common_step_counter)
    return self.filtered_gravity[:, :2].square().sum(-1)


def joint_zero_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """惩罚绝对关节角，不使用相对默认姿态的偏差"""
  return env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids].square().sum(-1)


class JointEdgeCost:
  """对实际关节硬限位设置软边缘代价，不约束名义姿态"""

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
  """与控制步长无关的终止事件代价，时间上限截断不算跌倒"""
  return env.termination_manager.terminated.to(torch.float32) / env.step_dt


class FootstepReward:
  """锁存计划落脚误差，初始支撑只使用实时误差"""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    self.component = cfg.params["component"]
    if self.component not in {
      "landing", "support", "swing_position", "swing_yaw", "schedule", "clearance", "slip"
    }:
      raise ValueError("Unknown footstep reward component")
    asset_cfg = cfg.params["asset_cfg"]
    asset = env.scene[asset_cfg.name]
    if [asset.site_names[index] for index in asset_cfg.site_ids] != ["left_foot", "right_foot"]:
      raise ValueError("Foot sites must be ordered [left_foot, right_foot]")
    if self.component != "landing":
      return
    shape = (env.num_envs, 2)
    self.last_ids = torch.full(shape, -1, dtype=torch.long, device=env.device)
    self.saw_air = torch.zeros(shape, dtype=torch.bool, device=env.device)
    self.previous_contact = torch.ones_like(self.saw_air)
    self.landed = torch.zeros_like(self.saw_air)
    self.landing_expected = torch.zeros_like(self.saw_air)
    self.landing_cost = torch.zeros(shape, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if self.component != "landing":
      return
    selected = slice(None) if env_ids is None else env_ids
    self.last_ids[selected] = -1
    self.saw_air[selected] = False
    self.previous_contact[selected] = True
    self.landed[selected] = False
    self.landing_expected[selected] = False
    self.landing_cost[selected] = 0.0

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
    swing_position_std: float = 0.1,
    swing_yaw_std: float = 0.1,
    landing_miss_cost: float = 1.0,
  ) -> torch.Tensor:
    if component != self.component:
      raise ValueError("Reward component must match its construction config")
    state = env.command_manager.get_term(command_name).reward_state
    if not isinstance(state, FootstepRewardState):
      raise TypeError("Footstep command must publish a FootstepRewardState as reward_state")
    if position_std <= 0 or yaw_std <= 0 or clearance <= 0 or not 0 < landing_window <= 1:
      raise ValueError("Invalid tracking, clearance or landing-window parameters")
    if not math.isfinite(landing_miss_cost) or landing_miss_cost < 0:
      raise ValueError("landing_miss_cost must be finite and nonnegative")
    timing = resolve_phase_cfg(stance_fraction, phase_cfg)
    contact = env.scene[sensor_name].data.found > 0
    if contact.shape != (env.num_envs, 2):
      raise ValueError("Contact sensor must provide [num_envs, 2] in left/right foot order")
    data = env.scene[asset_cfg.name].data
    if component == "slip":
      speed = torch.linalg.vector_norm(data.site_lin_vel_w[:, asset_cfg.site_ids, :2], dim=-1)
      yaw_rate = data.site_ang_vel_w[:, asset_cfg.site_ids, 2].abs()
      return ((speed + 0.05 * yaw_rate) * contact).sum(-1)
    stance, swing_progress = phase_windows(state.phase, phase_cfg=timing)
    standing = state.frequency == 0
    stance = stance | standing.unsqueeze(-1)
    if component == "schedule":
      return contact_schedule_score(stance, contact)
    if component == "clearance":
      position = data.site_pos_w[:, asset_cfg.site_ids]
      swing_progress = torch.where(standing.unsqueeze(-1), 0.0, swing_progress)
      target_height = clearance * torch.sin(torch.pi * swing_progress)
      height = position[..., 2] - state.ground_height
      return (((target_height - height).clamp_min(0) / clearance).square() * ~stance).sum(-1)
    if component == "swing_position":
      position = data.site_pos_w[:, asset_cfg.site_ids]
      distance = torch.linalg.vector_norm(position[..., :2] - state.targets_w[..., :2], dim=-1)
      return swing_tracking_score(distance, stance, swing_position_std)
    quaternion = data.site_quat_w[:, asset_cfg.site_ids]
    scalar, roll, pitch, yaw = quaternion.unbind(-1)
    foot_yaw = torch.atan2(2 * (scalar * yaw + roll * pitch), 1 - 2 * (pitch.square() + yaw.square()))
    yaw_delta = foot_yaw - state.targets_w[..., 2]
    yaw_error = torch.atan2(yaw_delta.sin(), yaw_delta.cos())
    if component == "swing_yaw":
      return swing_tracking_score(yaw_error, stance, swing_yaw_std)
    position = data.site_pos_w[:, asset_cfg.site_ids]
    distance = torch.linalg.vector_norm(position[..., :2] - state.targets_w[..., :2], dim=-1)
    if component == "landing":
      changed = (self.last_ids != state.target_ids) | standing.unsqueeze(-1)
      self.landing_expected.copy_(torch.where(changed, (self.last_ids >= 0) & ~standing.unsqueeze(-1), self.landing_expected))
      self.landing_expected |= ~stance
      self.landed &= ~changed
      self.saw_air &= ~changed
      self.landing_cost.masked_fill_(changed, 0.0)
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
      cost = footprint_accuracy_cost(distance, yaw_error, position_std, yaw_std)
      self.landing_cost.copy_(torch.where(first_contact, cost + ~permitted * landing_miss_cost, self.landing_cost))
      self.landed |= first_contact
      self.previous_contact.copy_(contact)
      self.last_ids.copy_(state.target_ids)
      landing_cost = torch.where(self.landed, self.landing_cost, cost + self.landing_expected * landing_miss_cost)
      return (torch.where(standing.unsqueeze(-1), cost, landing_cost) * stance).sum(-1) / stance.sum(-1).clamp_min(1)
    if component == "support":
      cost = footprint_accuracy_cost(distance, yaw_error, position_std, yaw_std)
      return (cost * stance).sum(-1) / stance.sum(-1).clamp_min(1)
    raise ValueError(f"Unknown component {component}")
