"""上肢扰动与开局随机化事件。

策略不控制手臂，但 mjlab 会用 ``EntityData.joint_pos_target`` 驱动**每一个**内置执行器，
而 ``clear_state()`` 在 reset 时会把它清零。没有 ``hold_arm_pose`` 的话，手臂会被直接拉直垂下。
把这个目标位姿随机重采——再叠上关节力矩——顺便就构成了下肢必须抗住的扰动，对应
实际使用时上层手臂策略在自顾自地动。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.envs.mdp import apply_body_impulse
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform
from mjlab.utils.lab_api.string import resolve_matching_names_values

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_POSE_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")


def _env_ids(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> torch.Tensor:
  if env_ids is None:
    return torch.arange(env.num_envs, device=env.device)
  return env_ids.to(env.device)


##
# 逐环境状态。挂在 env 上而不是各事件实例上，是为了让多个事件共享同一份。
##


def disturbance_level(env: ManagerBasedRlEnv) -> torch.Tensor:
  """每个环境自己的扰动强度系数，形状 (num_envs,)。

  三个扰动事件共用同一个系数，而不是各自采一个：只有共用才会出现“整段安静”的 episode。
  各自独立采样的话，三者同时接近零的概率是乘积，几乎碰不到。
  """
  level = getattr(env, "_disturbance_level", None)
  if level is None:
    level = torch.ones(env.num_envs, device=env.device)
    env._disturbance_level = level  # type: ignore[attr-defined]  # noqa: SLF001
  return level


def resample_disturbance_level(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  level_range: tuple[float, float] = (0.2, 1.0),
  quiet_prob: float = 0.0,
) -> None:
  """reset 事件：给这些环境重新采一个扰动强度。

  之前所有环境共用一个全局档位，策略只能看到“当前这个强度”。改成逐环境采样之后，
  同一批里同时存在安静的和被狠推的 episode，既不会把干净工况忘掉，也不用靠全局课程
  去回调。全局课程仍然决定上包络，这里决定每个环境占其中多少。

  ``quiet_prob``：把这么大比例的 episode 的系数直接钉成 0。均匀采样下“真正安静”
  （level<0.05）只占 5%，而站立只占指令窗口的 10%，两者相乘后“没人推的站立”在整批
  样本里不到 0.5%——而那正是要学“站稳”的那部分数据。
  """
  ids = _env_ids(env, env_ids)
  level = sample_uniform(*level_range, (len(ids),), env.device)
  if quiet_prob > 0.0:
    quiet = torch.rand(len(ids), device=env.device) < quiet_prob
    level = torch.where(quiet, torch.zeros_like(level), level)
  disturbance_level(env)[ids] = level


def gait_phase_offset(env: ManagerBasedRlEnv) -> torch.Tensor:
  """每个环境自己的步态相位起点，取值 [0, 1)。形状 (num_envs,)。"""
  offset = getattr(env, "_gait_phase_offset", None)
  if offset is None:
    offset = torch.zeros(env.num_envs, device=env.device)
    env._gait_phase_offset = offset  # type: ignore[attr-defined]  # noqa: SLF001
  return offset


def resample_gait_phase(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
  """reset 事件：重新采一个步态相位起点。

  相位时钟本来就是 ``episode_length_buf * dt``，每局必然从 0 开始。而相位 0 对应
  “左脚支撑、右脚摆动”，于是 85% 带着运动指令 reset 的环境都是右脚先迈，与实测到的
  “负向转弯普遍差 30%”“直行漂移永远偏正”自洽。
  """
  ids = _env_ids(env, env_ids)
  gait_phase_offset(env)[ids] = torch.rand(len(ids), device=env.device)


##
# 开局随机化。幅值 = 课程包络 x 逐环境系数。
##


def _level_weights(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  scale: float,
  keys: tuple[str, ...],
  unscaled: tuple[str, ...],
) -> torch.Tensor:
  """逐环境、逐通道的幅值系数，形状 (N, len(keys))。

  把采样结果乘上 s 等价于从 U(s*lo, s*hi) 采样，所以直接缩样本就行。
  ``unscaled`` 里的通道恒为 1。
  """
  factor = (disturbance_level(env)[env_ids] * scale).unsqueeze(-1)
  mask = torch.tensor(
    [0.0 if k in unscaled else 1.0 for k in keys], device=env.device
  ).unsqueeze(0)
  return 1.0 + (factor - 1.0) * mask


def scaled_reset_root_state(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  pose_range: dict[str, tuple[float, float]],
  velocity_range: dict[str, tuple[float, float]] | None = None,
  scale: float = 1.0,
  unscaled_pose_keys: tuple[str, ...] = ("yaw",),
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
  """mjlab 的 ``reset_root_state_uniform``，但幅值乘上“课程包络 x 逐环境系数”。

  开局姿态是这个任务里最硬的一道题：实测从零训练时中位存活只有 42 步（0.84 s），
  而且塌陷比例在所有高度指令上均匀（各档 15.9%-17.2%）——不是被要求蹲太低，而是
  复位那一刻就被扔进了爬不回来的状态，于是什么都学不到。

  yaw 默认不缩：它不是难度，是朝向覆盖。缩了会让开局朝向集中在一个方向上，
  重新引入左右/方向破缺。
  """
  ids = _env_ids(env, env_ids)
  asset: Entity = env.scene[asset_cfg.name]
  default_root_state = asset.data.default_root_state
  assert default_root_state is not None
  root_states = default_root_state[ids].clone()

  ranges = torch.tensor(
    [pose_range.get(k, (0.0, 0.0)) for k in _POSE_KEYS], device=env.device
  )
  pose = sample_uniform(
    ranges[:, 0], ranges[:, 1], (len(ids), 6), device=env.device
  ) * _level_weights(env, ids, scale, _POSE_KEYS, unscaled_pose_keys)

  positions = root_states[:, 0:3] + pose[:, 0:3] + env.scene.env_origins[ids]
  orientations = quat_mul(
    root_states[:, 3:7],
    quat_from_euler_xyz(pose[:, 3], pose[:, 4], pose[:, 5]),
  )

  ranges = torch.tensor(
    [(velocity_range or {}).get(k, (0.0, 0.0)) for k in _POSE_KEYS], device=env.device
  )
  velocities = root_states[:, 7:13] + sample_uniform(
    ranges[:, 0], ranges[:, 1], (len(ids), 6), device=env.device
  ) * _level_weights(env, ids, scale, _POSE_KEYS, ())

  asset.write_root_link_pose_to_sim(
    torch.cat([positions, orientations], dim=-1), env_ids=ids
  )
  asset.write_root_link_velocity_to_sim(velocities, env_ids=ids)


def scaled_reset_joints(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  position_range: tuple[float, float],
  velocity_range: tuple[float, float],
  scale: float = 1.0,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
  """mjlab 的 ``reset_joints_by_offset``，但初速度乘上“课程包络 x 逐环境系数”。

  只缩速度：关节角的小扰动只是让每局不从同一个姿势开始，不算难度；初速度才是真扰动。
  """
  ids = _env_ids(env, env_ids)
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  default_joint_vel = asset.data.default_joint_vel
  soft_limits = asset.data.soft_joint_pos_limits
  assert default_joint_pos is not None
  assert default_joint_vel is not None
  assert soft_limits is not None

  joint_pos = default_joint_pos[ids][:, asset_cfg.joint_ids].clone()
  joint_pos += sample_uniform(*position_range, joint_pos.shape, env.device)
  limits = soft_limits[ids][:, asset_cfg.joint_ids]
  joint_pos = joint_pos.clamp_(limits[..., 0], limits[..., 1])

  factor = (disturbance_level(env)[ids] * scale).unsqueeze(-1)
  joint_vel = default_joint_vel[ids][:, asset_cfg.joint_ids].clone()
  joint_vel += sample_uniform(*velocity_range, joint_vel.shape, env.device) * factor

  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, list):
    joint_ids = torch.tensor(joint_ids, device=env.device)
  asset.write_joint_state_to_sim(
    joint_pos.view(len(ids), -1),
    joint_vel.view(len(ids), -1),
    env_ids=ids,
    joint_ids=joint_ids,
  )


##
# 扰动。
##


class scaled_body_impulse(apply_body_impulse):
  """mjlab 的 ``apply_body_impulse``，再乘上逐环境的扰动强度。

  只在“本拍刚触发”的环境上重写一次力，所以不会逐拍复利地把力缩到零。
  """

  def __call__(self, env: ManagerBasedRlEnv, env_ids, **kwargs) -> None:
    was_active = self._active.clone()
    super().__call__(env, env_ids, **kwargs)
    triggered = (self._active & ~was_active).nonzero(as_tuple=False).flatten()
    if len(triggered) == 0:
      return
    level = disturbance_level(env)[triggered].view(-1, 1, 1)
    wrench = self._asset.data.body_external_wrench[triggered][:, self._body_ids]
    self._asset.write_external_wrench_to_sim(
      wrench[..., :3] * level,
      wrench[..., 3:] * level,
      env_ids=triggered,
      body_ids=self._body_ids,
    )


class hold_arm_pose:
  """把手臂保持在一个随机采样的 PD 目标上。

  用两次：一次 ``mode="reset"`` 且 ``write_state=True``，让 episode 直接从采样位姿开始而
  不是猛地弹过去；另一次 ``mode="interval"`` 且 ``write_state=False``，让手臂在 episode 中
  真的摆起来。
  """

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    asset: Entity = env.scene[asset_cfg.name]
    ids, names = asset.find_joints(asset_cfg.joint_names)
    self.joint_ids = torch.tensor(ids, device=env.device)
    _, _, ranges = resolve_matching_names_values(cfg.params["ranges"], names)
    self.lower = torch.tensor([r[0] for r in ranges], device=env.device)
    self.upper = torch.tensor([r[1] for r in ranges], device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    ranges: dict[str, tuple[float, float]],
    write_state: bool = False,
    blend: float = 1.0,
    scale_by_level: bool = False,
  ) -> None:
    del ranges  # 已在 __init__ 里消费。
    asset: Entity = env.scene[asset_cfg.name]
    ids = _env_ids(env, env_ids)
    target = sample_uniform(
      self.lower, self.upper, (len(ids), len(self.joint_ids)), env.device
    )
    # 直接下标写：set_joint_position_target() 会把 env_ids 和 joint_ids 广播在一起，
    # 只有写全部环境时才对。
    current = asset.data.joint_pos_target[ids.unsqueeze(1), self.joint_ids]
    # 留给课程把手臂动作淡入。blend=0 时手臂不动，这是策略还在找步态时需要的：
    # 每 1-3 s 换一次位姿正好落在步态周期上，会把步态淹掉。
    weight = torch.full((len(ids), 1), float(blend), device=env.device)
    if scale_by_level:
      weight = weight * disturbance_level(env)[ids].unsqueeze(1)
    target = current + weight * (target - current)
    asset.data.joint_pos_target[ids.unsqueeze(1), self.joint_ids] = target
    if write_state:
      asset.write_joint_state_to_sim(
        target,
        torch.zeros_like(target),
        env_ids=ids,
        joint_ids=self.joint_ids,
      )


class arm_torque_impulse:
  """向手臂关节注入瞬时随机力矩，配合 ``mode="step"`` 使用。

  ``qfrc_applied`` 不会被物理步清掉，所以力矩会保持一个采样时长，然后在进入下一段冷却
  前显式清零。
  """

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    asset: Entity = env.scene[asset_cfg.name]
    ids, _ = asset.find_joints(asset_cfg.joint_names)
    joint_ids = torch.tensor(ids, device=env.device)
    self.dof_ids = asset.data.indexing.joint_v_adr[joint_ids]
    assert self.dof_ids.shape == joint_ids.shape
    self.active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.timer = torch.zeros(env.num_envs, device=env.device)
    self.torque = torch.zeros(env.num_envs, len(ids), device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    torque_range: tuple[float, float],
    duration_s: tuple[float, float],
    cooldown_s: tuple[float, float],
  ) -> None:
    del env_ids  # step 事件总是对所有环境执行。
    asset: Entity = env.scene[asset_cfg.name]

    self.timer -= env.step_dt
    flip = self.timer <= 0.0
    if flip.any():
      ids = flip.nonzero(as_tuple=False).flatten()
      starting = ~self.active[ids]
      self.active[ids] = starting
      sampled = sample_uniform(
        *torque_range, (len(ids), self.torque.shape[1]), env.device
      )
      level = disturbance_level(env)[ids].unsqueeze(1)
      self.torque[ids] = sampled * starting.float().unsqueeze(1) * level
      span = torch.empty(len(ids), device=env.device)
      self.timer[ids] = torch.where(
        starting,
        span.uniform_(*duration_s),
        span.clone().uniform_(*cooldown_s),
      )
    asset.data.data.qfrc_applied[:, self.dof_ids] = self.torque

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.active[env_ids] = False
    self.timer[env_ids] = 0.0
    self.torque[env_ids] = 0.0
