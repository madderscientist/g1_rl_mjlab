"""独立脚步生成与时钟参数，默认值不代表硬件能力边界"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from g1_lower_rl.footstep_phase import FootstepPhaseCfg

DEFAULT_FOOT_WIDTH = 0.24
"""Nominal stance width and default sampling centre; G1 zero-joint foot spacing is 0.23701291 m."""


def ordered_range(name: str, bounds: tuple[float, float], *, positive: bool = False) -> None:
  """验证区间恰有两个有限且有序的端点，可额外要求端点严格为正"""
  if len(bounds) != 2 or not all(math.isfinite(value) for value in bounds) or bounds[0] > bounds[1]:
    raise ValueError(f"{name} must contain two finite ordered bounds")
  if positive and bounds[0] <= 0:
    raise ValueError(f"{name} must be strictly positive")


@dataclass(frozen=True)
class FootstepSamplerCfg:
  """距离、方向和朝向采样参数，以及平面脚侧与间距约束"""

  distance_range: tuple[float, float] = (0.05, 0.48)
  distance_mean: float = 0.25
  distance_std: float = 0.10
  direction_noise: tuple[float, float] = (-2 * math.pi / 9, 2 * math.pi / 9)
  yaw_noise: tuple[float, float] = (-math.pi / 6, math.pi / 6)
  min_width: float = 0.12
  max_width: float = 0.36
  max_yaw_change: float = math.pi / 6

  @property
  def width_center(self) -> float:
    """Centre of the symmetric lateral sampling interval, not a hard minimum."""
    return (self.min_width + self.max_width) / 2

  def __post_init__(self) -> None:
    """检查截断正态参数和横向边界，不包含整体方向重采样规则"""
    ordered_range("distance_range", self.distance_range, positive=True)
    if not math.isfinite(self.distance_mean) or self.distance_mean <= 0:
      raise ValueError("distance_mean must be finite and positive")
    if not math.isfinite(self.distance_std) or self.distance_std <= 0:
      raise ValueError("distance_std must be finite and positive")
    for name in ("direction_noise", "yaw_noise"):
      ordered_range(name, getattr(self, name))
    if not 0 < self.min_width <= self.max_width <= self.distance_range[1]:
      raise ValueError("Expected 0 < min_width <= max_width <= maximum distance")
    if not 0 < self.max_yaw_change < math.pi / 2:
      raise ValueError("max_yaw_change must lie in (0, pi/2)")


@dataclass(frozen=True)
class RandomCommandCfg:
  """随机指令的分布和调度参数，不包含步态相位或脚印几何"""

  frequency_range: tuple[float, float] = (0.8, 1.8)
  initial_frequency: float = 1 / 0.6
  frequency_rate_range: tuple[float, float] = (-0.3, 0.3)
  frequency_rate_interval_s: float = 2.0
  direction_range: tuple[float, float] = (-math.pi, math.pi)
  foot_heading_range: tuple[float, float] = (0.0, 0.0)
  command_interval_s: tuple[float, float] = (3.0, 8.0)
  stop_probability: float = 0.30
  hold_time_s: tuple[float, float] = (2.0, 5.0)
  automatic_commands: bool = True
  automatic_restart: bool = True

  def __post_init__(self) -> None:
    """校验随机分布、重采样周期和概率，不引用执行器配置"""
    ordered_range("frequency_range", self.frequency_range, positive=True)
    ordered_range("frequency_rate_range", self.frequency_rate_range)
    for name in ("direction_range", "foot_heading_range"):
      ordered_range(name, getattr(self, name))
    for name in ("command_interval_s", "hold_time_s"):
      ordered_range(name, getattr(self, name), positive=True)
    if not self.frequency_range[0] <= self.initial_frequency <= self.frequency_range[1]:
      raise ValueError("initial_frequency must lie within frequency_range")
    if not self.frequency_rate_range[0] <= 0 <= self.frequency_rate_range[1]:
      raise ValueError("frequency_rate_range must include zero")
    if not math.isfinite(self.frequency_rate_interval_s) or self.frequency_rate_interval_s <= 0:
      raise ValueError("frequency_rate_interval_s must be finite and positive")
    if not 0 <= self.stop_probability <= 1:
      raise ValueError("stop_probability must lie in [0,1]")


@dataclass(frozen=True)
class FootstepManagerCfg:
  """执行器的步态时序与输入限幅，不包含随机指令分布"""

  phase: FootstepPhaseCfg = field(default_factory=FootstepPhaseCfg)
  sampler: FootstepSamplerCfg = field(default_factory=FootstepSamplerCfg)
  control_dt: float = 0.02
  frequency_range: tuple[float, float] = (0.8, 1.8)
  initial_frequency: float = 1 / 0.6
  frequency_slew_rate: float = 0.2
  hold_width: float = DEFAULT_FOOT_WIDTH
  stop_deceleration: float = 0.3
  stop_frequency_floor: float = 0.6
  start_acceleration: float = 0.4
  contact_confirm_steps: int = 2
  landing_timeout_s: float = 0.5
  require_contact_confirmation: bool = True

  def __post_init__(self) -> None:
    """验证频率、停走和时间参数，防止单个控制拍跳过接触事件"""
    ordered_range("frequency_range", self.frequency_range, positive=True)
    if not self.sampler.min_width <= self.hold_width <= self.sampler.max_width:
      raise ValueError("hold_width must lie within the sampler width range")
    for name in ("control_dt", "frequency_slew_rate", "stop_deceleration", "start_acceleration", "landing_timeout_s"):
      if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
        raise ValueError(f"{name} must be finite and positive")
    if not self.frequency_range[0] <= self.initial_frequency <= self.frequency_range[1]:
      raise ValueError("initial_frequency must lie within frequency_range")
    if not 0 < self.stop_frequency_floor <= self.frequency_range[0]:
      raise ValueError("stop_frequency_floor must be positive and no greater than the walking minimum")
    if not isinstance(self.contact_confirm_steps, int) or self.contact_confirm_steps < 1:
      raise ValueError("contact_confirm_steps must be a positive integer")
    width = self.phase.contact_half_width
    centers = self.phase.right_stance_phase - self.phase.left_stance_phase
    smallest_interval = min(width, centers - 2 * width, 2 * math.pi - centers - 2 * width)
    if 2 * math.pi * self.frequency_range[1] * self.control_dt >= smallest_interval:
      raise ValueError("control_dt/frequency may skip contact events; reduce dt or the frequency maximum")
