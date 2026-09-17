"""参考 Mind Your Steps 的平面脚印采样，不依赖训练框架"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
from numpy.typing import ArrayLike

from g1_lower_rl.footsteps.config import FootstepSamplerCfg


def wrap_angle(angle):
  """将标量或数组角度归一到 [-pi, pi)，单位为弧度"""
  return (angle + math.pi) % (2 * math.pi) - math.pi


def pose_array(value: ArrayLike, shape: tuple[int, ...]) -> np.ndarray:
  """检查 XY/yaw 位姿的形状与有限性，并返回独立的 float64 数组"""
  result = np.asarray(value, dtype=np.float64)
  if result.shape != shape or not np.isfinite(result).all():
    raise ValueError(f"Expected finite XY/yaw poses with shape {shape}")
  return result.copy()


def to_local(poses: ArrayLike, anchor: ArrayLike) -> np.ndarray:
  """将世界系中的单个或多个 XY/yaw 位姿转换到共同水平参考系"""
  values = np.asarray(poses, dtype=np.float64)
  if values.ndim < 1 or values.shape[-1] != 3 or not np.isfinite(values).all():
    raise ValueError("Expected finite [...,3] XY/yaw poses")
  origin = pose_array(anchor, (3,))
  delta = values[..., :2] - origin[:2]
  cosine, sine = math.cos(origin[2]), math.sin(origin[2])
  return np.stack(
    (
      cosine * delta[..., 0] + sine * delta[..., 1],
      -sine * delta[..., 0] + cosine * delta[..., 1],
      wrap_angle(values[..., 2] - origin[2]),
    ),
    axis=-1,
  )


def from_local(pose: ArrayLike, anchor: ArrayLike) -> np.ndarray:
  """将单个局部 XY/yaw 位姿旋转并平移回世界系"""
  offset = pose_array(pose, (3,))
  origin = pose_array(anchor, (3,))
  cosine, sine = math.cos(origin[2]), math.sin(origin[2])
  return np.array(
    [
      origin[0] + cosine * offset[0] - sine * offset[1],
      origin[1] + sine * offset[0] + cosine * offset[1],
      wrap_angle(origin[2] + offset[2]),
    ]
  )


class FootstepSampler:
  """持有独立随机数状态和方向指令，按前一个异侧脚印生成目标"""

  def __init__(self, cfg: FootstepSamplerCfg | None = None, seed: int | np.random.SeedSequence | None = None):
    """初始化采样配置与随机种子，默认移动方向和脚掌朝向均为零"""
    self.cfg = cfg or FootstepSamplerCfg()
    self.rng = np.random.default_rng(seed)
    self.movement_direction = 0.0
    self.foot_heading = 0.0

  def set_direction(self, movement_direction: float, foot_heading: float) -> None:
    """分别设置世界系移动方向和脚掌朝向，输入为有限弧度值"""
    if not math.isfinite(movement_direction) or not math.isfinite(foot_heading):
      raise ValueError("Directions must be finite radians")
    self.movement_direction = float(wrap_angle(movement_direction))
    self.foot_heading = float(wrap_angle(foot_heading))

  def sample_distance(self) -> float:
    """从截断正态分布采样候选距离，返回尚未经过站距投影的米值"""
    lower, upper = self.cfg.distance_range
    if lower == upper:
      return lower
    distribution = NormalDist(self.cfg.distance_mean, self.cfg.distance_std)
    # 将右尾区间镜像到左尾，避免两个 CDF 值都舍入为一
    reflect = lower > self.cfg.distance_mean
    if reflect:
      lower, upper = 2 * self.cfg.distance_mean - upper, 2 * self.cfg.distance_mean - lower
    probability_low, probability_high = distribution.cdf(lower), distribution.cdf(upper)
    if probability_high <= probability_low:
      raise ValueError("Truncated normal interval has insufficient numerical probability; adjust distance_mean/std")
    probability = float(self.rng.uniform(probability_low, probability_high))
    probability = min(math.nextafter(1.0, 0.0), max(math.nextafter(0.0, 1.0), probability))
    distance = distribution.inv_cdf(probability)
    return 2 * self.cfg.distance_mean - distance if reflect else distance

  def sample(self, previous: ArrayLike, side: int) -> np.ndarray:
    """以前一个异侧世界位姿为参考，生成左脚 side=0 或右脚 side=1 的世界 XY/yaw"""
    reference = pose_array(previous, (3,))
    if side not in (0, 1):
      raise ValueError("side must be 0 (left) or 1 (right)")
    distance = self.sample_distance()
    maximum = self.cfg.distance_range[1]
    direction = self.movement_direction + self.rng.uniform(*self.cfg.direction_noise) - reference[2]
    lateral_sign = 1 if side == 0 else -1
    lateral = float(
      lateral_sign * np.clip(
        self.cfg.width_center + lateral_sign * distance * math.sin(direction), self.cfg.min_width, self.cfg.max_width
      )
    )
    # 补足最小站距后，再裁剪前后分量以满足总距离上限
    forward_limit = math.sqrt(max(0.0, maximum**2 - lateral**2))
    forward = float(np.clip(distance * math.cos(direction), -forward_limit, forward_limit))
    yaw = self.foot_heading + self.rng.uniform(*self.cfg.yaw_noise)
    delta_yaw = float(np.clip(wrap_angle(yaw - reference[2]), -self.cfg.max_yaw_change, self.cfg.max_yaw_change))
    return from_local([forward, lateral, delta_yaw], reference)
