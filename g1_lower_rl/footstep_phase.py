"""生成器、奖励和部署共用的相位配置，不依赖 Torch 或仿真器"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FootstepPhaseCfg:
  """用弧度定义左右理论落地中心和共用双支撑半宽"""

  # 相位递增并对 2*pi 取模，以下区间左闭右开
  # [0, 0.4*pi) 右脚支撑、左脚摆动
  # [0.4*pi, 0.6*pi) 双支撑，左脚理论落地中心为 pi/2
  # [0.6*pi, 1.4*pi) 左脚支撑、右脚摆动
  # [1.4*pi, 1.6*pi) 双支撑，右脚理论落地中心为 3*pi/2
  # [1.6*pi, 2*pi) 右脚支撑、左脚摆动，并跨零点继续
  # 相位零和 pi 分别为左右摆动中点，实际足高与接地仍取决于机器人状态
  # stance_phase 是双支撑中心，半宽向两侧展开；f=0 时始终要求双脚接触
  left_stance_phase: float = math.pi / 2
  right_stance_phase: float = 3 * math.pi / 2
  contact_half_width: float = 0.1 * math.pi

  def __post_init__(self) -> None:
    """校验中心顺序和共享半宽，确保两个双支撑窗口之间保留单支撑区间"""
    if not all(math.isfinite(value) for value in (self.left_stance_phase, self.right_stance_phase, self.contact_half_width)):
      raise ValueError("Phase centers and half-width must be finite")
    if not 0 <= self.left_stance_phase < self.right_stance_phase < 2 * math.pi:
      raise ValueError("Expected 0 <= left_stance_phase < right_stance_phase < 2*pi")
    gap = self.right_stance_phase - self.left_stance_phase
    if not 0 < 2 * self.contact_half_width < min(gap, 2 * math.pi - gap):
      raise ValueError("contact_half_width must be positive and leave single-support intervals between windows")

  @property
  def double_support_windows_rad(self) -> tuple[tuple[float, float], tuple[float, float]]:
    """返回左右落地中心两侧的双支撑窗口，端点允许跨越一个周期"""
    return (
      (self.left_stance_phase - self.contact_half_width, self.left_stance_phase + self.contact_half_width),
      (self.right_stance_phase - self.contact_half_width, self.right_stance_phase + self.contact_half_width),
    )

  @property
  def contact_start_rad(self) -> tuple[float, float]:
    """返回左脚和右脚开始计划接触的相位边界"""
    return self.left_stance_phase - self.contact_half_width, self.right_stance_phase - self.contact_half_width

  @property
  def liftoff_rad(self) -> tuple[float, float]:
    """返回左脚和右脚的起脚边界，即另一侧双支撑窗口的结束点"""
    return self.right_stance_phase + self.contact_half_width, self.left_stance_phase + self.contact_half_width

  @property
  def touchdown_cycles(self) -> tuple[float, float]:
    """将左右理论落地中心转换为完整周期比例"""
    return self.left_stance_phase / (2 * math.pi), self.right_stance_phase / (2 * math.pi)

  @property
  def stance_fractions(self) -> tuple[float, float]:
    """计算左右脚各自在完整周期中的计划支撑占比"""
    gap = self.right_stance_phase - self.left_stance_phase
    width = 2 * self.contact_half_width
    return (gap + width) / (2 * math.pi), (2 * math.pi - gap + width) / (2 * math.pi)

  def in_double_support(self, phase_rad: float) -> bool:
    """判断累计弧度相位是否位于理论双支撑窗口，不判断实测接触"""
    if not math.isfinite(phase_rad):
      raise ValueError("Phase must be finite")
    wrapped = phase_rad % (2 * math.pi)
    for start, end in self.double_support_windows_rad:
      start_rad = start % (2 * math.pi)
      end_rad = end % (2 * math.pi)
      inside = start_rad <= wrapped < end_rad if start_rad < end_rad else wrapped >= start_rad or wrapped < end_rad
      if inside:
        return True
    return False

  def to_metadata(self) -> dict:
    """导出相位参数及派生边界，供模型和部署端校验同一份约定"""
    return {
      "parameterization": "stance_centers_shared_half_width_v1",
      "units": "radians",
      "interval_convention": "[center-half_width, center+half_width); interpreted modulo 2*pi",
      "left_stance_phase": self.left_stance_phase,
      "right_stance_phase": self.right_stance_phase,
      "contact_half_width": self.contact_half_width,
      "double_support_windows_rad": [list(window) for window in self.double_support_windows_rad],
      "left_touchdown_rad": self.left_stance_phase,
      "right_touchdown_rad": self.right_stance_phase,
      "contact_start_rad": list(self.contact_start_rad),
      "liftoff_rad": list(self.liftoff_rad),
      "stance_fractions": list(self.stance_fractions),
      "standing_override": "f=0 requires both feet in contact regardless of frozen phase",
    }


DEFAULT_PHASE_CFG = FootstepPhaseCfg()


def resolve_phase_cfg(stance_fraction: float | None = None, phase_cfg: FootstepPhaseCfg | None = None) -> FootstepPhaseCfg:
  """解析显式相位配置或旧支撑占比参数，禁止同时指定两种来源"""
  if stance_fraction is not None and phase_cfg is not None:
    raise ValueError("Specify phase_cfg or the legacy stance_fraction, not both")
  if phase_cfg is not None:
    return phase_cfg
  if stance_fraction is None:
    return DEFAULT_PHASE_CFG
  if not 0.5 < stance_fraction < 1.0:
    raise ValueError("stance_fraction must lie in (0.5, 1.0)")
  return FootstepPhaseCfg(contact_half_width=(stance_fraction - 0.5) * math.pi)
