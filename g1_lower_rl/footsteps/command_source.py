"""供训练和预览复用的随机行走意图，不持有或修改脚步管理器"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace

import numpy as np

from g1_lower_rl.footsteps.config import RandomCommandCfg
from g1_lower_rl.footsteps.sampler import wrap_angle


@dataclass(frozen=True)
class GaitRequest:
  """世界系移动方向、脚掌朝向、正频率目标及行走意图，不是电机命令"""

  movement_direction: float = 0.0
  foot_heading: float = 0.0
  frequency: float = 1 / 0.6
  walking: bool = True

  def __post_init__(self) -> None:
    """验证有限角度和正频率，停止由 walking=False 表达而不是零目标频率"""
    if not all(math.isfinite(value) for value in (self.movement_direction, self.foot_heading, self.frequency)):
      raise ValueError("Request angles and frequency must be finite")
    if self.frequency <= 0 or not isinstance(self.walking, bool):
      raise ValueError("Request frequency must be positive and walking must be boolean")


class RandomCommandSource:
  """根据执行状态产生下一拍意图，唯一时钟来自调用方提供的模拟 dt"""

  def __init__(self, cfg: RandomCommandCfg | None = None, seed: int | None = None):
    """初始化独立调度随机流，使用前需 reset 指定航向基准"""
    self.cfg = cfg or RandomCommandCfg()
    self.rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(2)[0])
    self.initialized = False

  def _directions(self) -> tuple[float, float]:
    """相对 reset 航向独立采样两个整体方向，逐脚扰动仍由采样器负责"""
    return tuple(
      float(wrap_angle(self.heading_origin + self.rng.uniform(*bounds)))
      for bounds in (self.cfg.direction_range, self.cfg.foot_heading_range)
    )

  def reset(self, heading_origin: float = 0.0, request: GaitRequest | None = None) -> GaitRequest:
    """清空调度时钟，可从外部意图接续随机模式，避免切换时突变"""
    if not math.isfinite(heading_origin):
      raise ValueError("heading_origin must be finite")
    if request is not None and not self.cfg.frequency_range[0] <= request.frequency <= self.cfg.frequency_range[1]:
      raise ValueError("Request frequency outside source range")
    self.heading_origin = heading_origin
    self.request = request or GaitRequest(*self._directions(), self.cfg.initial_frequency)
    self.elapsed = 0.0
    self.rate = 0.0
    self.rate_remaining = 0.0
    self.command_at = self.rng.uniform(*self.cfg.command_interval_s)
    self.restart_at = None
    self.automatic_commands = self.cfg.automatic_commands
    self.automatic_restart = self.cfg.automatic_restart
    self.initialized = True
    return self.request

  def set_automation(self, enabled: bool) -> None:
    """切换自动换向及停走决策，不关闭频率游走，不操作执行器内部状态"""
    if not self.initialized:
      raise RuntimeError("Call reset before using command source")
    if not isinstance(enabled, bool):
      raise TypeError("enabled must be boolean")
    self.automatic_commands = self.automatic_restart = enabled
    self.command_at = self.elapsed + self.rng.uniform(*self.cfg.command_interval_s)
    self.restart_at = None

  def set_request(self, request: GaitRequest) -> None:
    """同步外部手动意图，后续随机演化从该意图继续"""
    if not self.initialized:
      raise RuntimeError("Call reset before using command source")
    if not self.cfg.frequency_range[0] <= request.frequency <= self.cfg.frequency_range[1]:
      raise ValueError("Request frequency outside source range")
    self.request = request
    self.rate_remaining = 0.0
    self.restart_at = None

  def advance(self, dt: float, *, mode: str, frequency: float, random_frequency: bool = True) -> GaitRequest:
    """读取刚结束拍的执行模式和实际频率，返回下一拍意图，不积分步态相位"""
    if not self.initialized:
      raise RuntimeError("Call reset before using command source")
    if not math.isfinite(dt) or dt <= 0 or not math.isfinite(frequency) or frequency < 0:
      raise ValueError("Expected positive dt and nonnegative actual frequency")
    if mode not in ("walking", "starting", "stopping", "standing"):
      raise ValueError("Invalid execution mode")
    self.elapsed += dt
    if mode == "standing":
      if self.automatic_restart and not self.request.walking:
        # 保持时间从真实进入站立后的首次反馈开始，而非从停止请求开始
        if self.restart_at is None:
          self.restart_at = self.elapsed + self.rng.uniform(*self.cfg.hold_time_s)
        if self.elapsed + 1e-10 >= self.restart_at:
          target = self.cfg.initial_frequency if random_frequency else self.request.frequency
          self.request = GaitRequest(*self._directions(), target, True)
          self.command_at = self.elapsed + self.rng.uniform(*self.cfg.command_interval_s)
      return self.request
    self.restart_at = None
    if mode != "walking" or not self.request.walking:
      return self.request
    if random_frequency:
      if self.rate_remaining <= 1e-10:
        self.rate = self.rng.uniform(*self.cfg.frequency_rate_range)
        self.rate_remaining = self.cfg.frequency_rate_interval_s
      lower, upper = self.cfg.frequency_range
      width = upper - lower
      candidate = frequency + self.rate * dt
      if width == 0:
        target = lower
      else:
        remainder = (candidate - lower) % (2 * width)
        target = lower + (remainder if remainder <= width else 2 * width - remainder)
        if remainder > width:
          self.rate = -self.rate
        elif remainder == 0:
          self.rate = abs(self.rate)
        elif remainder == width:
          self.rate = -abs(self.rate)
      self.request = replace(self.request, frequency=target)
      self.rate_remaining -= dt
    if self.automatic_commands and self.elapsed >= self.command_at:
      self.command_at = self.elapsed + self.rng.uniform(*self.cfg.command_interval_s)
      if self.rng.random() < self.cfg.stop_probability:
        self.request = replace(self.request, walking=False)
      else:
        direction, heading = self._directions()
        self.request = replace(self.request, movement_direction=direction, foot_heading=heading)
    return self.request

  def state_dict(self) -> dict:
    """保存调度时钟、意图和随机流，用于可信本地检查点"""
    if not self.initialized:
      raise RuntimeError("Call reset before saving command source")
    return copy.deepcopy(self.__dict__)

  def load_state_dict(self, state: dict) -> None:
    """恢复同配置下的调度状态，不涉及脚步管理器快照"""
    if state.get("cfg") != self.cfg or not state.get("initialized"):
      raise ValueError("Command source checkpoint configuration mismatch")
    self.__dict__.update(copy.deepcopy(state))
