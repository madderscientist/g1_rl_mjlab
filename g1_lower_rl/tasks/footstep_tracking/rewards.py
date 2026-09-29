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
from g1_lower_rl.tasks.lower_body.mdp import soft_landing as velocity_soft_landing
from g1_lower_rl.tasks.footstep_tracking.reward_math import (
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
  """15轴实际执行器铜损，接近硬限位且继续向外施力时单独放大"""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    asset_cfg = cfg.params["asset_cfg"]
    asset = env.scene[asset_cfg.name]
    names = [asset.actuator_names[index] for index in asset_cfg.actuator_ids]
    if len(names) != 15 or set(names) != set(LOWER_BODY_JOINTS):
      raise ValueError("Torque cost must select exactly the 15 leg and waist actuators")
    self.actuator_ids = list(asset_cfg.actuator_ids)
    self.joint_ids = [asset.joint_names.index(name) for name in names]
    self.at_lower_limit = None
    self.at_upper_limit = None
    self.pushing_limit = None
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
    command_name: str = "footsteps",
    standing_scale: float = 1.0,
    limit_scale: float = 1.0,
    limit_margin: float = 0.0,
  ) -> torch.Tensor:
    del copper_weights
    if not math.isfinite(standing_scale) or standing_scale <= 0:
      raise ValueError("Standing torque scale must be finite and positive")
    if not math.isfinite(limit_scale) or limit_scale < 1.0:
      raise ValueError("Limit torque scale must be finite and at least 1")
    if not math.isfinite(limit_margin) or limit_margin < 0.0:
      raise ValueError("Limit margin must be finite and nonnegative")
    data = env.scene[asset_cfg.name].data
    torques = data.actuator_force[:, self.actuator_ids]
    positions = data.joint_pos[:, self.joint_ids]
    limits = data.joint_pos_limits[:, self.joint_ids]
    self.at_lower_limit = positions <= limits[..., 0] + limit_margin
    self.at_upper_limit = positions >= limits[..., 1] - limit_margin
    self.pushing_limit = (self.at_lower_limit & (torques < 0)) | (self.at_upper_limit & (torques > 0))
    weights = self.weights * torch.where(self.pushing_limit, limit_scale, 1.0)
    cost = torque_square_cost(torques, weights, reference_torque)
    if standing_scale != 1.0:
      frequency = env.command_manager.get_term(command_name).reward_state.frequency
      cost = cost * torch.where(frequency == 0, standing_scale, 1.0)
    return cost


def body_tilt_angle_l2(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """计算相对竖直方向的弧度倾角平方，不受世界航向角影响"""
  data = env.scene[asset_cfg.name].data
  orientation = data.body_link_quat_w[:, asset_cfg.body_ids].squeeze(1)
  gravity = quat_apply_inverse(orientation, data.gravity_vec_w)
  angle = torch.atan2(torch.linalg.vector_norm(gravity[:, :2], dim=-1), -gravity[:, 2])
  return angle.square()


def _head_height_above_ground(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, command_name: str) -> torch.Tensor:
  state = env.command_manager.get_term(command_name).reward_state
  data = env.scene[asset_cfg.name].data
  return data.geom_pos_w[:, asset_cfg.geom_ids, 2].squeeze(-1) - state.ground_height.mean(-1)


def head_height_reward(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  command_name: str = "footsteps",
  sensor_name: str = "feet_ground_contact",
  height_cap: float = 1.254,
) -> torch.Tensor:
  """线性奖励离地高度，腾空或失败时不计奖励"""
  if not math.isfinite(height_cap) or height_cap <= 0:
    raise ValueError("height_cap must be finite and positive")
  height = _head_height_above_ground(env, asset_cfg, command_name)
  supported = (env.scene[sensor_name].data.found > 0).any(-1)
  valid = supported & ~env.termination_manager.terminated
  return (height / height_cap).clamp(0.0, 1.0) * valid


def head_height_shortfall(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  command_name: str = "footsteps",
  minimum_height: float = 1.15,
  height_scale: float = 0.2,
) -> torch.Tensor:
  """惩罚低于最低头高的线性缺口，不因失去脚部接触而免罚"""
  if not all(math.isfinite(value) and value > 0 for value in (minimum_height, height_scale)):
    raise ValueError("Minimum head height and scale must be finite and positive")
  height = _head_height_above_ground(env, asset_cfg, command_name)
  return (minimum_height - height).clamp_min(0.0) / height_scale


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


def _standing_position_score(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg,
                             command_name: str, position_scale: float) -> torch.Tensor:
  if not math.isfinite(position_scale) or position_scale <= 0:
    raise ValueError("Standing position scale must be finite and positive")
  state = env.command_manager.get_term(command_name).reward_state
  position = env.scene[asset_cfg.name].data.site_pos_w[:, asset_cfg.site_ids, :2]
  error = (position - state.targets_w[..., :2]).square().sum(-1).mean(-1)
  return (1.0 + error / position_scale**2).reciprocal()


def standing_linear_velocity_reward(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  command_name: str = "footsteps",
  std: float = math.sqrt(0.2),
  z_penalty: float = 1.5,
  position_scale: float = 0.1,
) -> torch.Tensor:
  """零执行频率时奖励身体低线速度，并按双脚偏离冻结目标的距离衰减"""
  if not math.isfinite(std) or std <= 0 or not math.isfinite(z_penalty) or z_penalty < 0:
    raise ValueError("Velocity std must be positive and z penalty nonnegative; both must be finite")
  state = env.command_manager.get_term(command_name).reward_state
  velocity = env.scene[asset_cfg.name].data.root_link_lin_vel_b
  error = velocity[:, :2].square().sum(-1) + z_penalty * velocity[:, 2].square()
  position_score = _standing_position_score(env, asset_cfg, command_name, position_scale)
  return torch.exp(-error / std**2) * position_score * (state.frequency == 0)


def standing_angular_velocity_reward(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  command_name: str = "footsteps",
  std: float = 0.7,
  xy_penalty: float = 0.05,
  position_scale: float = 0.1,
) -> torch.Tensor:
  """零执行频率时奖励身体低角速度，并按双脚偏离冻结目标的距离衰减"""
  if not math.isfinite(std) or std <= 0 or not math.isfinite(xy_penalty) or xy_penalty < 0:
    raise ValueError("Velocity std must be positive and xy penalty nonnegative; both must be finite")
  state = env.command_manager.get_term(command_name).reward_state
  velocity = env.scene[asset_cfg.name].data.root_link_ang_vel_b
  error = velocity[:, 2].square() + xy_penalty * velocity[:, :2].square().sum(-1)
  position_score = _standing_position_score(env, asset_cfg, command_name, position_scale)
  return torch.exp(-error / std**2) * position_score * (state.frequency == 0)


def feet_hold_position_cost(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  command_name: str = "footsteps",
  deadband: float = 0.03,
  distance_scale: float = 0.1,
) -> torch.Tensor:
  """停脚时限制相对冻结目标的水平漂移，保留小幅平衡调整空间"""
  if not math.isfinite(deadband) or deadband < 0 or not math.isfinite(distance_scale) or distance_scale <= 0:
    raise ValueError("Hold deadband must be nonnegative and distance scale positive; both must be finite")
  state = env.command_manager.get_term(command_name).reward_state
  position = env.scene[asset_cfg.name].data.site_pos_w[:, asset_cfg.site_ids, :2]
  distance = torch.linalg.vector_norm(position - state.targets_w[..., :2], dim=-1)
  cost = ((distance - deadband).clamp_min(0.0) / distance_scale).square().mean(-1)
  return cost * (state.frequency == 0)


def feet_flatness_cost(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  command_name: str = "footsteps",
  sensor_name: str = "feet_ground_contact",
  phase_cfg: FootstepPhaseCfg | None = None,
  angle_scale: float = math.pi / 12,
) -> torch.Tensor:
  """计划支撑或实际触地时约束脚掌法向，静止检查双脚，不限制航向角"""
  if not math.isfinite(angle_scale) or angle_scale <= 0:
    raise ValueError("Flatness angle scale must be finite and positive")
  state = env.command_manager.get_term(command_name).reward_state
  data = env.scene[asset_cfg.name].data
  orientation = data.site_quat_w[:, asset_cfg.site_ids]
  gravity = data.gravity_vec_w.unsqueeze(1).expand(-1, 2, -1)
  local_gravity = quat_apply_inverse(orientation, gravity)
  angle = torch.atan2(torch.linalg.vector_norm(local_gravity[..., :2], dim=-1), -local_gravity[..., 2])
  stance, _ = phase_windows(state.phase, phase_cfg=phase_cfg)
  contact = env.scene[sensor_name].data.found > 0
  active = stance | contact | (state.frequency == 0).unsqueeze(-1)
  cost = (angle / angle_scale).square()
  return (cost * active).sum(-1) / active.sum(-1).clamp_min(1)


def stance_knee_bend_cost(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  command_name: str = "footsteps",
  phase_cfg: FootstepPhaseCfg | None = None,
  standing_limit: float = math.pi / 6,
  walking_limit: float = math.pi / 4,
  angle_scale: float = math.pi / 4,
) -> torch.Tensor:
  """只惩罚计划支撑腿过度屈膝，停脚检查双腿，不限制摆动腿"""
  if (not all(math.isfinite(value) and value >= 0 for value in (standing_limit, walking_limit))
      or not math.isfinite(angle_scale) or angle_scale <= 0):
    raise ValueError("Knee limits must be nonnegative and angle scale positive; all must be finite")
  state = env.command_manager.get_term(command_name).reward_state
  standing = state.frequency == 0
  stance, _ = phase_windows(state.phase, phase_cfg=phase_cfg)
  stance = stance | standing.unsqueeze(-1)
  knee = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
  limit = torch.where(standing, standing_limit, walking_limit).unsqueeze(-1)
  cost = ((knee - limit).clamp_min(0.0) / angle_scale).square()
  return (cost * stance).sum(-1) / stance.sum(-1).clamp_min(1)


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


def soft_landing(env: ManagerBasedRlEnv, sensor_name: str, command_name: str = "footsteps") -> torch.Tensor:
  """复用速度任务的首次触地冲击代价，以脚步频率判断运动状态"""
  moving = env.command_manager.get_term(command_name).reward_state.frequency > 0
  return velocity_soft_landing(env, sensor_name) * moving


class FootstepReward:
  """每个目标仅在摆动离地后的首次触地评价落脚误差"""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    self.component = cfg.params["component"]
    if self.component not in {
      "landing", "swing_position", "swing_yaw", "schedule", "clearance", "slip",
      "slip_still", "air_time", "stationary", "swing_contact",
    }:
      raise ValueError("Unknown footstep reward component")
    asset_cfg = cfg.params["asset_cfg"]
    asset = env.scene[asset_cfg.name]
    if [asset.site_names[index] for index in asset_cfg.site_ids] != ["left_foot", "right_foot"]:
      raise ValueError("Foot sites must be ordered [left_foot, right_foot]")
    self._air_time_constants = None
    if self.component != "landing":
      return
    shape = (env.num_envs, 2)
    self.last_ids = torch.full(shape, -1, dtype=torch.long, device=env.device)
    self.saw_air = torch.zeros(shape, dtype=torch.bool, device=env.device)
    self.previous_contact = torch.ones_like(self.saw_air)
    self.landed = torch.zeros_like(self.saw_air)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if self.component != "landing":
      return
    selected = slice(None) if env_ids is None else env_ids
    self.last_ids[selected] = -1
    self.saw_air[selected] = False
    self.previous_contact[selected] = True
    self.landed[selected] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    asset_cfg: SceneEntityCfg,
    component: str,
    stance_fraction: float | None = None,
    position_std: float = 0.05,
    yaw_std: float = 0.2,
    clearance: float = 0.11,
    air_time_scale: float = 0.4,
    phase_cfg: FootstepPhaseCfg | None = None,
    swing_position_std: float = 0.2,
    swing_yaw_std: float = 0.1,
    swing_progress_power: float = 2.0,
    landing_start_step: int | None = None,
    landing_ramp_steps: int = 1,
    still_speed_deadband: float = 0.02,
  ) -> torch.Tensor:
    if component != self.component:
      raise ValueError("Reward component must match its construction config")
    state = env.command_manager.get_term(command_name).reward_state
    if not isinstance(state, FootstepRewardState):
      raise TypeError("Footstep command must publish a FootstepRewardState as reward_state")
    if not all(math.isfinite(value) and value > 0 for value in (position_std, yaw_std, clearance)):
      raise ValueError("Invalid tracking or clearance parameters")
    timing = resolve_phase_cfg(stance_fraction, phase_cfg)
    sensor = env.scene[sensor_name].data
    contact = sensor.found > 0
    if contact.shape != (env.num_envs, 2):
      raise ValueError("Contact sensor must provide [num_envs, 2] in left/right foot order")
    data = env.scene[asset_cfg.name].data
    moving = state.frequency > 0
    if component == "stationary":
      in_contact = sensor.current_contact_time > 0
      return (~in_contact).float().sum(-1) * ~moving
    if component == "slip_still":
      if not math.isfinite(still_speed_deadband) or still_speed_deadband < 0:
        raise ValueError("Standing speed deadband must be finite and nonnegative")
      speed = torch.linalg.vector_norm(data.site_lin_vel_w[:, asset_cfg.site_ids], dim=-1)
      return (speed - still_speed_deadband).clamp_min(0.0).sum(-1) * ~moving
    if component == "slip":
      speed = torch.linalg.vector_norm(data.site_lin_vel_w[:, asset_cfg.site_ids, :2], dim=-1)
      return (speed.square() * contact).sum(-1) * moving
    stance, swing_progress = phase_windows(state.phase, phase_cfg=timing)
    standing = state.frequency == 0
    stance = stance | standing.unsqueeze(-1)
    if component in {"schedule", "air_time"}:
      in_contact = sensor.current_contact_time > 0
      matched = (stance == contact).all(-1) & (stance == in_contact).all(-1)
      if component == "schedule":
        return matched.to(torch.float32) * moving
      if not math.isfinite(air_time_scale) or air_time_scale <= 0:
        raise ValueError("Air-time scale must be finite and positive")
      single_support = stance.sum(-1) == 1
      held_time = torch.where(stance, sensor.current_contact_time, sensor.current_air_time).min(dim=-1).values
      constant_key = (timing, state.frequency.device, state.frequency.dtype)
      if self._air_time_constants is None or self._air_time_constants[0] != constant_key:
        with torch.inference_mode(False):
          swing_fractions = state.frequency.new_tensor([1 - fraction for fraction in timing.stance_fractions])
        self._air_time_constants = constant_key, swing_fractions
      else:
        swing_fractions = self._air_time_constants[1]
      normalized = held_time.unsqueeze(-1) * state.frequency.unsqueeze(-1) / swing_fractions
      credit = torch.minimum(normalized.clamp(0.0, 1.0), swing_progress)
      return air_time_scale * (credit * ~stance).sum(-1) * matched * single_support * moving
    if component in {"swing_contact", "clearance"}:
      lift_profile = torch.sin(math.pi * swing_progress).square()
      if component == "swing_contact":
        return (contact * ~stance * lift_profile).sum(-1)
      position = data.site_pos_w[:, asset_cfg.site_ids]
      height = position[..., 2] - state.ground_height
      target = clearance * lift_profile * ~stance
      return ((height - target).abs() / clearance).sum(-1)
    if component != "swing_yaw":
      position = data.site_pos_w[:, asset_cfg.site_ids]
      distance = torch.linalg.vector_norm(position[..., :2] - state.targets_w[..., :2], dim=-1)
    if component != "swing_position":
      quaternion = data.site_quat_w[:, asset_cfg.site_ids]
      scalar, roll, pitch, yaw = quaternion.unbind(-1)
      foot_yaw = torch.atan2(2 * (scalar * yaw + roll * pitch), 1 - 2 * (pitch.square() + yaw.square()))
      yaw_delta = foot_yaw - state.targets_w[..., 2]
      yaw_error = torch.atan2(yaw_delta.sin(), yaw_delta.cos())
    if component in {"swing_position", "swing_yaw"}:
      error, std = (distance, swing_position_std) if component == "swing_position" else (yaw_error, swing_yaw_std)
      score = swing_tracking_score(error, stance, std,
                                   swing_progress=swing_progress, progress_power=swing_progress_power)
      return score * (contact & stance).any(-1)
    if component == "landing":
      if not math.isfinite(env.step_dt) or env.step_dt <= 0:
        raise ValueError("Landing event cost requires a positive finite control step")
      if landing_start_step is not None and (landing_start_step < 0 or landing_ramp_steps <= 0):
        raise ValueError("Landing ramp start must be nonnegative and duration positive")
      changed = (self.last_ids != state.target_ids) | standing.unsqueeze(-1)
      self.landed &= ~changed
      self.saw_air &= ~changed
      self.saw_air |= ~contact & ~stance
      first_contact = contact & ~self.previous_contact & self.saw_air & ~self.landed
      cost = footprint_accuracy_cost(distance, yaw_error, position_std, yaw_std)
      self.landed |= first_contact
      self.previous_contact.copy_(contact)
      self.last_ids.copy_(state.target_ids)
      scale = 1.0 if landing_start_step is None else min(
        1.0, max(0.0, (env.common_step_counter - 1 - landing_start_step) / landing_ramp_steps)
      )
      return scale * (cost * first_contact).sum(-1) / env.step_dt
    raise ValueError(f"Unknown component {component}")
