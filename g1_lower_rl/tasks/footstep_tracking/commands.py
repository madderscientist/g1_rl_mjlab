"""把独立脚步算法接入训练时序，不在默认的步末命令回调中重复推进"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from g1_lower_rl.footsteps import FootstepManager, FootstepManagerCfg, RandomCommandCfg, RandomCommandSource
from g1_lower_rl.footsteps.footprint_geometry import load_footprint_geometry
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

  def build(self, env) -> FootstepCommand:
    """构造持有每环境独立随机流和队列的命令项"""
    return FootstepCommand(self, env)


class FootstepCommand(CommandTerm):
  """仅在终止与奖励计算前推进一次，reset 后等待正向运动学刷新再读脚位"""

  cfg: FootstepCommandCfg

  def __init__(self, cfg: FootstepCommandCfg, env):
    """分配设备侧观测和奖励缓冲，NumPy 生成状态按环境隔离"""
    super().__init__(cfg, env)
    if not math.isclose(env.step_dt, cfg.manager.control_dt, abs_tol=1e-10):
      raise ValueError("Footstep control_dt must match environment step_dt")
    if (
      not cfg.manager.frequency_range[0]
      <= cfg.source.frequency_range[0]
      <= cfg.source.frequency_range[1]
      <= cfg.manager.frequency_range[1]
    ):
      raise ValueError("Source frequency range must fit the manager execution range")
    self.robot = env.scene[cfg.entity_name]
    self.site_ids = self.robot.find_sites(("left_foot", "right_foot"), preserve_order=True)[0]
    seeds = np.random.SeedSequence(env.cfg.seed).generate_state(self.num_envs)
    self.managers = [FootstepManager(cfg.manager, int(seed)) for seed in seeds]
    self.sources = [RandomCommandSource(cfg.source, int(seed)) for seed in seeds]
    self.pending_reset = np.ones(self.num_envs, dtype=bool)
    self._footprint_geometry = None
    self.last_step = -1
    self.failed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
    self._command = torch.zeros((self.num_envs, 14), device=self.device)
    self.reward_state = FootstepRewardState(
      phase=torch.zeros(self.num_envs, device=self.device),
      frequency=torch.zeros(self.num_envs, device=self.device),
      targets_w=torch.zeros((self.num_envs, 2, 3), device=self.device),
      target_ids=torch.zeros((self.num_envs, 2), dtype=torch.long, device=self.device),
      ground_height=torch.zeros((self.num_envs, 2), device=self.device),
    )
    # NumPy 持有算法状态，复用同 dtype 的主机缓冲，避免逐环境向设备写标量
    self._buffers = {"command": self._command, "failed": self.failed}
    self._buffers.update({name: getattr(self.reward_state, name) for name in ("phase", "frequency", "targets_w", "target_ids")})
    self._host = {name: torch.zeros_like(buffer, device="cpu").numpy() for name, buffer in self._buffers.items()}

  @property
  def command(self) -> torch.Tensor:
    """返回相位、实际频率和冻结参考系四步，形状为 N 乘 14"""
    return self._command

  def reset(self, env_ids) -> dict:
    """只标记局部重置，不读取尚未刷新正向运动学的脚位"""
    selected = slice(None) if env_ids is None else env_ids
    if isinstance(selected, torch.Tensor):
      selected = selected.cpu().numpy()
    self.pending_reset[selected] = True
    self._host["failed"][selected] = False
    self.failed.copy_(torch.from_numpy(self._host["failed"]))
    return {}

  def _resample_command(self, env_ids) -> None:
    """兼容命令管理器的重置入口，调度重采样仍由随机源负责"""
    self.reset(env_ids)

  def _update_metrics(self) -> None:
    """当前适配器不另维护与奖励重复的累计指标"""

  def _store_command(self, index: int, command) -> None:
    """将已有命令写入固定缓冲，不重新排序队列或计算坐标变换"""
    row = self._host["command"][index]
    row[:2] = command.phase, command.frequency
    row[2:] = command.footsteps.ravel()

  def _publish(self) -> None:
    """批量复制已打包缓冲，不创建中间设备张量或异步复用尚未传完的数据"""
    for name, buffer in self._buffers.items():
      buffer.copy_(torch.from_numpy(self._host[name]))

  def _update_command(self) -> None:
    """在 sim.forward 之后用真实初始脚位重建待重置环境的队列"""
    selected = np.flatnonzero(self.pending_reset)
    if not len(selected):
      return
    # reset 写入 qpos 后 site 尚未刷新，只在步末 compute 中读取选中的环境
    feet = foot_poses(self.robot.data, self.site_ids)[torch.as_tensor(selected, device=self.device)].detach().cpu().numpy()
    for index, poses in zip(selected.tolist(), feet):
      heading = math.atan2(np.sin(poses[:, 2]).sum(), np.cos(poses[:, 2]).sum())
      manager = self.managers[index]
      self._store_command(index, manager.reset(poses, self.sources[index].reset(heading_origin=heading)))
      self._host["targets_w"][index] = manager.targets
      self._host["target_ids"][index] = manager.target_ids
      self._host["phase"][index] = manager.phase
      self._host["frequency"][index] = manager.frequency
    self.pending_reset[selected] = False
    self._publish()

  def compute(self, dt: float) -> None:
    """默认步末回调只完成 reset，不推进时间或重采样行走意图"""
    self._update_command()

  def _debug_vis_impl(self, visualizer) -> None:
    """绘制未来四步和本拍奖励快照的左右执行目标，不推进脚步状态"""
    if self._footprint_geometry is None:
      self._footprint_geometry = load_footprint_geometry()["feet"]
    colors = ((0.05, 0.65, 1.0), (1.0, 0.3, 0.12))
    for env_index in visualizer.get_env_indices(self.num_envs):
      if self.pending_reset[env_index]:
        continue
      command = self.managers[env_index].command()
      current_targets = self.reward_state.targets_w[env_index].detach().cpu().numpy()
      targets = [(slot // 2, slot % 2, pose, False) for slot, pose in enumerate(command.footsteps_w)]
      targets.extend((side, 0, pose, True) for side, pose in enumerate(current_targets))
      for side, future, pose, current in targets:
        suffix = "_current" if current else str(future + 1)
        label = f"footstep_{env_index}_{'L' if side == 0 else 'R'}{suffix}"
        color = tuple(channel * (1.0 if future == 0 else 0.65) for channel in colors[side]) + (0.7,)
        cosine, sine = math.cos(pose[2]), math.sin(pose[2])
        rotation = np.array(((cosine, -sine), (sine, cosine)))
        height = float(self.reward_state.ground_height[env_index, side]) + (0.025 if current else 0.015)
        for capsule_index, capsule in enumerate(self._footprint_geometry[side]["capsules"]):
          endpoints = np.array((capsule["start"], capsule["end"])) @ rotation.T + pose[:2]
          start, end = np.column_stack((endpoints, np.full(2, height)))
          capsule_label = f"{label}_capsule_{capsule_index}"
          visualizer.add_cylinder(start, end, radius=capsule["radius"], color=color, label=capsule_label)
          visualizer.add_sphere(start, radius=capsule["radius"], color=color, label=f"{capsule_label}_start")
          visualizer.add_sphere(end, radius=capsule["radius"], color=color, label=f"{capsule_label}_end")
        if current:
          lower, upper = np.asarray(self._footprint_geometry[side]["bounds"]) + np.array(([-0.015, -0.015], [0.015, 0.015]))
          corners = np.array(((lower[0], lower[1]), (upper[0], lower[1]), (upper[0], upper[1]), (lower[0], upper[1])))
          corners = corners @ rotation.T + pose[:2]
          corners = np.column_stack((corners, np.full(4, height)))
          for edge, start in enumerate(corners):
            visualizer.add_cylinder(start, corners[(edge + 1) % 4], radius=0.004, color=color, label=f"{label}_border_{edge}")
        origin = np.array((pose[0], pose[1], height + 0.035))
        direction = np.array((cosine, sine, 0.0))
        visualizer.add_arrow(origin, origin + 0.16 * direction, color=color, width=0.014 if current else 0.008, label=label)

  def finish_step(self) -> torch.Tensor:
    """在奖励前消费刚结束的一拍，超时变成局部终止并保留本拍奖励快照"""
    if self.last_step == self._env.common_step_counter:
      return self.failed
    if self.pending_reset.any():
      raise RuntimeError("Reset the environment before stepping footstep commands")
    self._env.sim.forward()
    feet = foot_poses(self.robot.data, self.site_ids)
    contact = self._env.scene[self.cfg.sensor_name].data.found > 0
    # 脚位与接触同批回读，只同步一次；接触仍来自最后物理子步的传感器记录
    feedback = torch.cat((feet, contact.unsqueeze(-1)), dim=-1).detach().cpu().numpy()
    for index, (manager, source) in enumerate(zip(self.managers, self.sources)):
      # advance 超时会抛出异常，先保存停止拍的旧目标，不能用异常后的队列补算
      if manager.mode == "stopping":
        self._host["phase"][index] = manager.phase + 2 * math.pi * manager.frequency * self._env.step_dt
        self._host["frequency"][index] = manager.frequency
        self._host["targets_w"][index] = manager.targets
        self._host["target_ids"][index] = manager.target_ids
      try:
        update = manager.advance(feet_w=feedback[index, :, :3], contacts=feedback[index, :, 3])
      except TimeoutError:
        self._host["failed"][index] = True
        self._host["command"][index] = 0
        continue
      for name in ("phase", "frequency", "targets_w", "target_ids"):
        self._host[name][index] = getattr(update.completed, name)
      manager.apply_request(source.advance(self._env.step_dt, mode=update.command.mode, frequency=update.command.frequency))
      # 普通方向与频率意图不修改已发布四步，只有停走模式切换需要刷新输出
      command = update.command if manager.mode == update.command.mode else manager.command()
      self._store_command(index, command)
    self.last_step = self._env.common_step_counter
    self._publish()
    return self.failed


def footstep_execution_failed(env, command_name: str = "footsteps") -> torch.Tensor:
  """首个终止项推进本拍脚步，并返回接触确认超时的环境掩码"""
  return env.command_manager.get_term(command_name).finish_step()
