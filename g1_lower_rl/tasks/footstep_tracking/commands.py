"""把独立脚步算法接入训练时序，不在默认的步末命令回调中重复推进"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from g1_lower_rl.footsteps import FootstepManagerCfg, RandomCommandCfg
from g1_lower_rl.footsteps.footprint_geometry import load_footprint_geometry
from g1_lower_rl.footsteps.tensor_manager import TensorFootstepManager
from g1_lower_rl.tasks.footstep_tracking.rewards import FootstepRewardState


def foot_poses(data, site_ids) -> torch.Tensor:
  """从足底 site 提取世界 XY 和绕竖直轴的航向"""
  scalar, roll, pitch, yaw = data.site_quat_w[:, site_ids].unbind(-1)
  heading = torch.atan2(2 * (scalar * yaw + roll * pitch), 1 - 2 * (pitch.square() + yaw.square()))
  return torch.cat((data.site_pos_w[:, site_ids, :2], heading.unsqueeze(-1)), dim=-1)


@dataclass(kw_only=True)
class FootstepCommandCfg(CommandTermCfg):
  """平地训练适配参数，算法默认值与 Web 一致，自动调度由 source 负责"""

  resampling_time_range: tuple[float, float] = (math.inf, math.inf)
  manager: FootstepManagerCfg = field(default_factory=FootstepManagerCfg)
  source: RandomCommandCfg = field(default_factory=RandomCommandCfg)
  entity_name: str = "robot"
  sensor_name: str = "feet_ground_contact"
  compile_backend: bool = True

  def build(self, env) -> FootstepCommand:
    """构造持有每环境独立随机流和队列的命令项"""
    return FootstepCommand(self, env)


class FootstepCommand(CommandTerm):
  """仅在终止与奖励计算前推进一次，reset 后等待正向运动学刷新再读脚位"""

  cfg: FootstepCommandCfg

  def __init__(self, cfg: FootstepCommandCfg, env):
    """批量脚步状态与物理状态驻留同一设备，不逐环境创建Python执行器"""
    super().__init__(cfg, env)
    if not math.isclose(env.step_dt, cfg.manager.control_dt, abs_tol=1e-10):
      raise ValueError("Footstep control_dt must match environment step_dt")
    self.robot = env.scene[cfg.entity_name]
    self.site_ids = self.robot.find_sites(("left_foot", "right_foot"), preserve_order=True)[0]
    self.batch = TensorFootstepManager(cfg.manager, cfg.source, self.num_envs, self.device, env.cfg.seed or 0,
                      compiled=cfg.compile_backend and torch.device(self.device).type == "cuda")
    self.pending_reset = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
    self._needs_reset = True
    self._footprint_geometry = None
    self.last_step = -1
    self.failed = self.batch.state["failed"]
    self._command = self.batch.state["command"]
    self.future_world = self.batch.state["future_world"]
    self.future_sides = self.batch.state["future_sides"]
    self.support_world = self.batch.state["supports"]
    self.reward_state = FootstepRewardState(
      phase=torch.zeros(self.num_envs, device=self.device),
      frequency=torch.zeros(self.num_envs, device=self.device),
      targets_w=torch.zeros((self.num_envs, 2, 3), device=self.device),
      target_ids=torch.zeros((self.num_envs, 2), dtype=torch.long, device=self.device),
      ground_height=torch.zeros((self.num_envs, 2), device=self.device),
    )

  @property
  def command(self) -> torch.Tensor:
    """返回相位、频率及左支撑/右支撑基准和左/右下一落点，形状 N 乘 14"""
    return self._command

  def reset(self, env_ids) -> dict:
    """只标记局部重置，不读取尚未刷新正向运动学的脚位"""
    selected = slice(None) if env_ids is None else env_ids
    if isinstance(selected, torch.Tensor) and selected.numel() == 0:
      return {}
    self.pending_reset[selected] = True
    self.failed[selected] = False
    self._needs_reset = True
    return {}

  def _resample_command(self, env_ids) -> None:
    """兼容命令管理器的重置入口，调度重采样仍由随机源负责"""
    self.reset(env_ids)

  def _update_metrics(self) -> None:
    """当前适配器不另维护与奖励重复的累计指标"""

  def _publish(self) -> None:
    """只做设备内快照复制，奖励缓冲始终保持稳定引用"""
    for name, source in (("phase", "completed_phase"), ("frequency", "completed_frequency"),
                         ("targets_w", "completed_targets"), ("target_ids", "completed_ids")):
      getattr(self.reward_state, name).copy_(self.batch.state[source])

  def _update_command(self) -> None:
    """在 sim.forward 之后用真实初始脚位重建待重置环境的队列"""
    if not self._needs_reset:
      return
    self.batch.reset(foot_poses(self.robot.data, self.site_ids), self.pending_reset)
    self.pending_reset.zero_()
    self._needs_reset = False
    self._publish()

  def compute(self, dt: float) -> None:
    """默认步末回调只完成 reset，不推进时间或重采样行走意图"""
    self._update_command()

  def _debug_vis_impl(self, visualizer) -> None:
    """绘制下一拍策略输入的双脚支撑基准及各自下一落点，不推进状态"""
    if self._footprint_geometry is None:
      self._footprint_geometry = load_footprint_geometry()["feet"]
    colors = ((0.05, 0.65, 1.0), (1.0, 0.3, 0.12))
    for env_index in visualizer.get_env_indices(self.num_envs):
      if self.pending_reset[env_index]:
        continue
      future = self.future_world[env_index].detach().cpu().numpy()
      supports = self.support_world[env_index].detach().cpu().numpy()
      targets = [(side, pose, False) for side, pose in enumerate(future)]
      targets.extend((side, pose, True) for side, pose in enumerate(supports))
      for side, pose, support in targets:
        suffix = "_support" if support else "1"
        label = f"footstep_{env_index}_{'L' if side == 0 else 'R'}{suffix}"
        color = colors[side] + (0.7,)
        cosine, sine = math.cos(pose[2]), math.sin(pose[2])
        rotation = np.array(((cosine, -sine), (sine, cosine)))
        height = float(self.reward_state.ground_height[env_index, side]) + (0.025 if support else 0.015)
        for capsule_index, capsule in enumerate(self._footprint_geometry[side]["capsules"]):
          endpoints = np.array((capsule["start"], capsule["end"])) @ rotation.T + pose[:2]
          start, end = np.column_stack((endpoints, np.full(2, height)))
          capsule_label = f"{label}_capsule_{capsule_index}"
          visualizer.add_cylinder(start, end, radius=capsule["radius"], color=color, label=capsule_label)
          visualizer.add_sphere(start, radius=capsule["radius"], color=color, label=f"{capsule_label}_start")
          visualizer.add_sphere(end, radius=capsule["radius"], color=color, label=f"{capsule_label}_end")
        if support:
          lower, upper = np.asarray(self._footprint_geometry[side]["bounds"]) + np.array(([-0.015, -0.015], [0.015, 0.015]))
          corners = np.array(((lower[0], lower[1]), (upper[0], lower[1]), (upper[0], upper[1]), (lower[0], upper[1])))
          corners = corners @ rotation.T + pose[:2]
          corners = np.column_stack((corners, np.full(4, height)))
          for edge, start in enumerate(corners):
            visualizer.add_cylinder(start, corners[(edge + 1) % 4], radius=0.004, color=color, label=f"{label}_border_{edge}")
        origin = np.array((pose[0], pose[1], height + 0.035))
        direction = np.array((cosine, sine, 0.0))
        visualizer.add_arrow(origin, origin + 0.16 * direction, color=color, width=0.014 if support else 0.008, label=label)

  def finish_step(self) -> torch.Tensor:
    """在奖励前消费刚结束的一拍，超时变成局部终止并保留本拍奖励快照"""
    if self.last_step == self._env.common_step_counter:
      return self.failed
    if self._needs_reset:
      raise RuntimeError("Reset the environment before stepping footstep commands")
    self._env.sim.forward()
    feet = foot_poses(self.robot.data, self.site_ids)
    contact = self._env.scene[self.cfg.sensor_name].data.found > 0
    self.batch.advance(feet, contact)
    self.last_step = self._env.common_step_counter
    self._publish()
    return self.failed


def footstep_execution_failed(env, command_name: str = "footsteps") -> torch.Tensor:
  """首个终止项推进本拍脚步，并返回接触确认超时的环境掩码"""
  return env.command_manager.get_term(command_name).finish_step()
