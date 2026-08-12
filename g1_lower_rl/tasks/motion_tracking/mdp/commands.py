"""GMT 式通用动作跟踪命令。

与 mjlab 自带的单动作 ``MotionCommand`` 的两点关键差别，也正是「换一段没训过的动作
还能跟住」的来源：

1. **多动作 + 自适应采样**。语料里的每条动作都参与训练，复位时按「最近失败率」加权
   挑一条（GMT 的 Adaptive Sampling）。均匀采样会让策略把预算浪费在早就学会的简单
   动作上，难的那些始终学不动。

2. **偏航/平移不变的前瞻观测**。策略看到的不是「参考轨迹在世界里的绝对位姿」，而是
   未来若干帧在**根坐标系**下的：离地高度、重力方向、线/角速度、关节角。绝对位置和
   朝向被彻底剔除，于是同一段动作平移或转向之后对策略而言完全相同——这是泛化到新动作
   的前提。用投影重力而不是 roll/pitch 角，是为了避开角度回绕。

参考：GMT: General Motion Tracking for Humanoid Whole-Body Control (arXiv:2506.14770)。
奖励/终止项沿用 mjlab tracking 的实现，所以这里刻意保持了和 ``MotionCommand`` 一致的
属性名。
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_error_magnitude,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
  yaw_quat,
)

from g1_lower_rl.tasks.motion_tracking.mdp.motion_corpus import MotionCorpus

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  """把世界系向量转到 q 所定义的局部系。"""
  return quat_apply(quat_inv(q), v)


class GeneralMotionCommand(CommandTerm):
  cfg: "GeneralMotionCommandCfg"
  _env: "ManagerBasedRlEnv"

  def __init__(self, cfg: "GeneralMotionCommandCfg", env: "ManagerBasedRlEnv"):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self.robot_anchor_body_index = self.robot.body_names.index(cfg.anchor_body_name)
    self.motion_anchor_body_index = cfg.body_names.index(cfg.anchor_body_name)
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )
    # 前瞻观测只喂策略实际驱动的那些关节，维度才对得上动作空间。
    self.policy_joint_indexes = torch.tensor(
      self.robot.find_joints(cfg.policy_joint_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )

    self.motion = MotionCorpus(cfg.motion_dir, self.body_indexes, device=self.device)
    print(f"[GMT] 载入语料: {self.motion.describe()}")

    self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    # 相位用浮点：每拍推进 self.speed 帧，取参考帧时再取整。
    self.phase = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
    self.speed = torch.ones(self.num_envs, dtype=torch.float, device=self.device)

    self.lookahead = torch.tensor(
      cfg.lookahead_steps, dtype=torch.long, device=self.device
    )

    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0

    # 每条动作的失败率（EMA）。采样概率正比于它，难动作会被更频繁地练到。
    self.motion_failed = torch.zeros(
      self.motion.num_motions, dtype=torch.float, device=self.device
    )
    self._current_failed = torch.zeros_like(self.motion_failed)

    self._ghost_model = None

    for key in (
      "error_anchor_pos",
      "error_anchor_rot",
      "error_body_pos",
      "error_body_rot",
      "error_joint_pos",
      "error_joint_vel",
      "sampling_entropy",
    ):
      self.metrics[key] = torch.zeros(self.num_envs, device=self.device)

  ##
  # 帧寻址
  ##

  @property
  def time_steps(self) -> torch.Tensor:
    """当前参考帧在拼接语料里的全局下标。"""
    return self.motion.start_idx[self.motion_ids] + self.phase.long()

  def _lookahead_indexes(self) -> torch.Tensor:
    """(num_envs, K) 前瞻帧的全局下标，超出本条动作末尾的部分钳到末帧。

    前瞻步长也乘速度：放慢时该看到的是同一段「未来多少秒」的动作，而不是同样帧数。
    """
    phase = self.phase.unsqueeze(1) + self.lookahead.unsqueeze(0) * self.speed.unsqueeze(1)
    last = (self.motion.num_frames[self.motion_ids] - 1).unsqueeze(1).float()
    phase = torch.minimum(phase, last)
    return self.motion.start_idx[self.motion_ids].unsqueeze(1) + phase.long()

  ##
  # 参考量（属性名与 mjlab MotionCommand 对齐，奖励/终止项可直接复用）
  ##

  @property
  def command(self) -> torch.Tensor:
    """偏航与平移不变的前瞻观测，(num_envs, K * 39)。"""
    idx = self._lookahead_indexes()
    root_quat = self.motion.body_quat_w[idx, 0]  # (N, K, 4)

    height = self.motion.body_pos_w[idx, 0, 2:3]
    gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(
      root_quat.shape[0], root_quat.shape[1], 3
    )
    proj_gravity = quat_rotate_inverse(root_quat, gravity)
    lin_vel = quat_rotate_inverse(root_quat, self.motion.body_lin_vel_w[idx, 0])
    ang_vel = quat_rotate_inverse(root_quat, self.motion.body_ang_vel_w[idx, 0])
    # 参考速度要跟着播放倍率缩放：放慢后位置推进变慢，速度也必须变慢，
    # 否则“该到哪”和“该多快”互相矛盾，策略无法同时满足。
    scale = self.speed[:, None, None]
    lin_vel = lin_vel * scale
    ang_vel = ang_vel * scale
    joint_pos = self.motion.joint_pos[idx][:, :, self.policy_joint_indexes]

    return torch.cat(
      [height, proj_gravity, lin_vel, ang_vel, joint_pos], dim=-1
    ).flatten(1)

  @property
  def joint_pos(self) -> torch.Tensor:
    return self.motion.joint_pos[self.time_steps]

  @property
  def joint_vel(self) -> torch.Tensor:
    return self.motion.joint_vel[self.time_steps] * self.speed[:, None]

  @property
  def body_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]
    )

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self.time_steps]

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[self.time_steps] * self.speed[:, None, None]

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[self.time_steps] * self.speed[:, None, None]

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index]
      + self._env.scene.env_origins
    )

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

  ##
  # 机器人实测量
  ##

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.body_indexes]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.body_indexes]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

  ##
  # CommandTerm 接口
  ##

  def _update_metrics(self) -> None:
    self.metrics["error_anchor_pos"] = torch.norm(
      self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
    )
    self.metrics["error_anchor_rot"] = quat_error_magnitude(
      self.anchor_quat_w, self.robot_anchor_quat_w
    )
    self.metrics["error_body_pos"] = torch.norm(
      self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_rot"] = quat_error_magnitude(
      self.body_quat_relative_w, self.robot_body_quat_w
    ).mean(dim=-1)
    self.metrics["error_joint_pos"] = torch.norm(
      self.joint_pos - self.robot_joint_pos, dim=-1
    )
    self.metrics["error_joint_vel"] = torch.norm(
      self.joint_vel - self.robot_joint_vel, dim=-1
    )

  def _sample_motions(self, env_ids: torch.Tensor) -> torch.Tensor:
    if self.cfg.sampling_mode == "uniform" or self.motion.num_motions == 1:
      probs = torch.ones(self.motion.num_motions, device=self.device)
    else:
      probs = self.motion_failed + self.cfg.adaptive_uniform_ratio / float(
        self.motion.num_motions
      )
    probs = probs / probs.sum()
    self.metrics["sampling_entropy"][:] = -(probs * (probs + 1e-12).log()).sum() / max(
      math.log(self.motion.num_motions), 1e-6
    )
    return torch.multinomial(probs, len(env_ids), replacement=True)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # 统计这批复位里哪些是「失败」（被终止而非超时），记到对应动作头上。
    terminated = self._env.termination_manager.terminated[env_ids]
    if torch.any(terminated):
      failed_ids = self.motion_ids[env_ids][terminated]
      self._current_failed[:] = torch.bincount(
        failed_ids, minlength=self.motion.num_motions
      ).float()

    self.motion_ids[env_ids] = self._sample_motions(env_ids)
    lo, hi = self.cfg.speed_range
    self.speed[env_ids] = sample_uniform(lo, hi, (len(env_ids),), device=self.device)

    if self.cfg.sampling_mode == "start":
      self.phase[env_ids] = 0.0
    else:
      # 从动作中间任意位置起步：只从头开始的话，后半段几乎见不到。
      frac = sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)
      self.phase[env_ids] = frac * (
        self.motion.num_frames[self.motion_ids[env_ids]] - 1
      ).float()

    self._write_state_from_motion(env_ids)

  def _write_state_from_motion(self, env_ids: torch.Tensor) -> None:
    """把机器人摆到参考帧的位姿上，并加一点随机扰动。"""
    root_pos = self.body_pos_w[:, 0].clone()
    root_ori = self.body_quat_w[:, 0].clone()
    root_lin_vel = self.body_lin_vel_w[:, 0].clone()
    root_ang_vel = self.body_ang_vel_w[:, 0].clone()

    ranges = torch.tensor(
      [
        self.cfg.pose_range.get(k, (0.0, 0.0))
        for k in ("x", "y", "z", "roll", "pitch", "yaw")
      ],
      device=self.device,
    )
    samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_pos[env_ids] += samples[:, 0:3]
    root_ori[env_ids] = quat_mul(
      quat_from_euler_xyz(samples[:, 3], samples[:, 4], samples[:, 5]),
      root_ori[env_ids],
    )

    ranges = torch.tensor(
      [
        self.cfg.velocity_range.get(k, (0.0, 0.0))
        for k in ("x", "y", "z", "roll", "pitch", "yaw")
      ],
      device=self.device,
    )
    samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_lin_vel[env_ids] += samples[:, :3]
    root_ang_vel[env_ids] += samples[:, 3:]

    joint_pos = self.joint_pos.clone()
    joint_vel = self.joint_vel.clone()
    joint_pos += sample_uniform(
      self.cfg.joint_position_range[0],
      self.cfg.joint_position_range[1],
      joint_pos.shape,
      device=self.device,
    )
    limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos[env_ids] = torch.clip(joint_pos[env_ids], limits[:, :, 0], limits[:, :, 1])

    self.robot.write_joint_state_to_sim(
      joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids
    )
    self.robot.write_root_state_to_sim(
      torch.cat(
        [
          root_pos[env_ids],
          root_ori[env_ids],
          root_lin_vel[env_ids],
          root_ang_vel[env_ids],
        ],
        dim=-1,
      ),
      env_ids=env_ids,
    )
    self.robot.clear_state(env_ids=env_ids)

  def _update_command(self) -> None:
    self.phase += self.speed
    done = torch.where(self.phase >= self.motion.num_frames[self.motion_ids])[0]
    if done.numel() > 0:
      self._resample_command(done)

    n_bodies = len(self.cfg.body_names)
    anchor_pos = self.anchor_pos_w[:, None, :].repeat(1, n_bodies, 1)
    anchor_quat = self.anchor_quat_w[:, None, :].repeat(1, n_bodies, 1)
    robot_anchor_pos = self.robot_anchor_pos_w[:, None, :].repeat(1, n_bodies, 1)
    robot_anchor_quat = self.robot_anchor_quat_w[:, None, :].repeat(1, n_bodies, 1)

    # 参考姿态整体对齐到机器人当前的水平位置与偏航，只保留高度差。
    # 这样「站在哪里、朝哪边」不进入奖励，策略学的是相对运动。
    delta_pos = robot_anchor_pos
    delta_pos[..., 2] = anchor_pos[..., 2]
    delta_ori = yaw_quat(quat_mul(robot_anchor_quat, quat_inv(anchor_quat)))

    self.body_quat_relative_w = quat_mul(delta_ori, self.body_quat_w)
    self.body_pos_relative_w = delta_pos + quat_apply(
      delta_ori, self.body_pos_w - anchor_pos
    )

    self.motion_failed = (
      self.cfg.adaptive_alpha * self._current_failed
      + (1 - self.cfg.adaptive_alpha) * self.motion_failed
    )
    self._current_failed.zero_()

  def _debug_vis_impl(self, visualizer) -> None:
    """把参考动作画成半透明 ghost，叠在实际机器人上做对照。"""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    if self._ghost_model is None:
      # 碰撞几何设成全透明，只留视觉几何，否则 ghost 会糊成一团。
      self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
      for gi in range(self._ghost_model.ngeom):
        if (
          self._ghost_model.geom_contype[gi] != 0
          or self._ghost_model.geom_conaffinity[gi] != 0
        ):
          self._ghost_model.geom_rgba[gi, 3] = 0
        else:
          self._ghost_model.geom_rgba[gi] = (0.2, 0.8, 1.0, 0.45)

    indexing = self._env.scene[self.cfg.entity_name].indexing
    free_q = indexing.free_joint_q_adr.cpu().numpy()
    joint_q = indexing.joint_q_adr.cpu().numpy()

    for batch in env_indices:
      qpos = np.zeros(self._env.sim.mj_model.nq)
      qpos[free_q[0:3]] = self.body_pos_w[batch, 0].cpu().numpy()
      qpos[free_q[3:7]] = self.body_quat_w[batch, 0].cpu().numpy()
      qpos[joint_q] = self.joint_pos[batch].cpu().numpy()
      visualizer.add_ghost_mesh(qpos, model=self._ghost_model, label=f"ghost_{batch}")


@dataclass(kw_only=True)
class GeneralMotionCommandCfg(CommandTermCfg):
  motion_dir: str
  """存放 NPZ 动作的目录，目录里所有动作一起参与训练。"""
  anchor_body_name: str
  body_names: tuple[str, ...]
  """参与跟踪的刚体；**第一个必须是根节点**（前瞻观测按它取根位姿）。"""
  policy_joint_names: tuple[str, ...]
  entity_name: str
  lookahead_steps: tuple[int, ...] = (1, 5, 10, 15, 20, 30, 40, 55, 75, 95)
  """前瞻帧偏移（控制步）。50 Hz 下最远看到约 1.9 秒后。"""
  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  joint_position_range: tuple[float, float] = (-0.1, 0.1)
  adaptive_uniform_ratio: float = 0.1
  adaptive_alpha: float = 0.01
  sampling_mode: Literal["adaptive", "uniform", "start"] = "adaptive"

  speed_range: tuple[float, float] = (1.0, 1.0)
  """播放倍率的采样区间，每回合一个。<1 即放慢。

  原速的 LAFAN1 里有大量动作在这台机器上物理不可达（肩关节 1.4 Hz 以上就力矩饱和）。
  早期对照：整体放慢 1.5 倍后，iter 12000 的回合长度从 12.91 涨到 50.44。
  这里改成每回合随机，让同一段动作能以不同速度反复练到，而不用把语料扩容好几倍。
  """

  def build(self, env: "ManagerBasedRlEnv") -> GeneralMotionCommand:
    return GeneralMotionCommand(self, env)
