"""下肢任务的指令项。

速度指令加上高度指令，合起来就是上层发给机器人的四个浮点数：``[vx, vy, omega, height]``。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg

from g1_lower_rl.tasks.lower_body.mdp.observations import height_above_feet

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class ScenarioVelocityCommand(UniformVelocityCommand):
  """在均匀采样之上补两个基类表达不了的场景，并把直行环境的航向钉住。

  基类的 vx/vy/ω 各自独立均匀采样，于是“原地转弯”（vxy=0、ω≠0）和“只走不转”
  （ω=0）都只能靠碰巧采到，实测纯直行只占 4.9%。这里显式指定两者的占比。

  **曾经这里还有一层 OU 游走**（两次重采之间让 vx/vy 连续漂移，理由是遥控杆/规划器
  发的是连续量）。已删除：它让窗口内的指令不再是常数，于是“这段时间实际走了多远”
  这个最直接的跟踪量没法定义——``track_linear_velocity_avg`` 需要窗口内指令恒定。
  换来的连续性收益也存疑，2 s 相关时间下策略看到的仍然主要是噪声。
  """

  cfg: ScenarioVelocityCommandCfg

  def __init__(self, cfg: ScenarioVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.is_turning_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.is_forced_straight_env = torch.zeros_like(self.is_turning_env)
    self.is_straight_env = torch.zeros_like(self.is_turning_env)
    self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    r = torch.empty(len(env_ids), device=self.device)
    self.is_turning_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_turning_envs
    self.vel_command_b[env_ids[self.is_turning_env[env_ids]], :2] = 0.0
    # 强制直行：ω 置零、vx/vy 保留。和原地转弯对称。
    #
    # 基类的 vx/vy/ω 是各自独立均匀采的，实测这意味着 **76.5% 的环境都在边走边转**，
    # 而“纯直行”（|ω| < 0.1）只占 4.9%——完全是 ω 采样碰巧落进窄带，没有任何机制保证。
    # 行走环境里 |ω| 中位数 0.822 rad/s，已经顶到量程上限。后果就是“保持直线”这件事
    # 本身没被演示过，与实测“直行 20 s 偏 27°、两脚摩擦不同则偏 83°”吻合。
    #
    # 排掉原地转弯的环境：那些环境 vxy 已经是 0，再把 ω 置零就变成站立了。
    forced = (r.uniform_(0.0, 1.0) <= self.cfg.rel_straight_envs) & (
      ~self.is_turning_env[env_ids]
    )
    self.is_forced_straight_env[env_ids] = forced
    self.vel_command_b[env_ids[forced], 2] = 0.0

    # 被要求保持航向的环境。基类会随机采一个航向目标；这里改成把目标钉在机器人当前
    # 朝向上，``heading_error`` 于是变成“自重采样以来累计偏了多少”，奖励才能对它收费。
    # 偏航漂移是一个直流偏置，埋在比它大一个数量级的步态振荡里，光罚瞬时角速度看不见。
    #
    # 只在“从非直行变为直行”时才重新钉。每次重采样都重新钉的话，累计窗口只有 3-8 s，
    # 按实测的 1.3 deg/s 算误差上限才 0.18 rad，信号太弱；连续多段直行指令合并计算后，
    # 窗口可以长到整个 episode。
    was_straight = self.is_straight_env[env_ids].clone()
    straight = (
      self.vel_command_b[env_ids, 2].abs() < self.cfg.straight_threshold
    ) | self.is_standing_env[env_ids]
    self.is_straight_env[env_ids] = straight
    repin = env_ids[straight & (~was_straight | (self.command_counter[env_ids] == 0))]
    if len(repin) > 0:
      self.heading_target[repin] = self.robot.data.heading_w[repin]

  def _update_metrics(self) -> None:
    super()._update_metrics()
    max_command_step = self.cfg.resampling_time_range[1] / self._env.step_dt
    self.metrics["error_heading"] += (
      torch.rad2deg(self.heading_error.abs()) * self.is_straight_env.float()
    ) / max_command_step


@dataclass(kw_only=True)
class ScenarioVelocityCommandCfg(UniformVelocityCommandCfg):
  rel_turning_envs: float = 0.0
  """被指定为原地转弯的环境占比。"""
  rel_straight_envs: float = 0.0
  """被强制 ω = 0（只走不转）的环境占比。与 ``rel_turning_envs`` 互斥。"""
  straight_threshold: float = 0.1
  """yaw 指令低于此值时，该环境被要求保持航向。"""

  def build(self, env: ManagerBasedRlEnv) -> ScenarioVelocityCommand:
    return ScenarioVelocityCommand(self, env)


class BaseHeightCommand(CommandTerm):
  """骨盆相对足底的目标高度，采样之后缓变而不是直接跳变。

  采样出来的目标是跳的，但对外发布的指令按 ``max_rate`` 向它靠拢。阶跃只能训出阶跃
  响应；而斜坡一是上层规划器实际会发的东西，二是它才能把“原地蹲起”（速度指令为零、
  高度持续变化）变成一个能学的行为，而不是一个瞬态。
  """

  cfg: BaseHeightCommandCfg

  def __init__(self, cfg: BaseHeightCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self._site_ids, _ = self.robot.find_sites(list(cfg.site_names))
    self.height_command = torch.zeros(self.num_envs, 1, device=self.device)
    self.height_target = torch.zeros(self.num_envs, 1, device=self.device)
    self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.height_command

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)
    self.height_target[env_ids, 0] = r.uniform_(*self.cfg.ranges.height)
    # 一个 episode 的第一次采样：直接落在目标上，不要从 0 慢慢爬上去。
    first = env_ids[self.command_counter[env_ids] == 0]
    self.height_command[first] = self.height_target[first]

  def _update_command(self) -> None:
    step = self.cfg.max_rate * self._env.step_dt
    delta = (self.height_target - self.height_command).clamp(-step, step)
    self.height_command += delta

  def _update_metrics(self) -> None:
    max_command_step = self.cfg.resampling_time_range[1] / self._env.step_dt
    height = height_above_feet(self.robot, self._site_ids)
    self.metrics["error_height"] += (
      torch.abs(self.height_command[:, 0] - height) / max_command_step
    )


@dataclass(kw_only=True)
class BaseHeightCommandCfg(CommandTermCfg):
  entity_name: str
  site_names: tuple[str, ...]
  """高度相对哪几个足底 site 测量。"""

  max_rate: float = 0.15
  """发布指令的变化率，单位 m/s。蹲下去 15 cm 大约要 1 s。"""

  @dataclass
  class Ranges:
    height: tuple[float, float]

  ranges: Ranges

  def build(self, env: ManagerBasedRlEnv) -> BaseHeightCommand:
    return BaseHeightCommand(self, env)
