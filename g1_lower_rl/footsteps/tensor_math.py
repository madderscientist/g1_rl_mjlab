"""批量脚步生成与观测共用的设备驻留几何计算"""

from __future__ import annotations

import math
from statistics import NormalDist

import torch

from g1_lower_rl.footsteps.config import FootstepSamplerCfg


def wrap(angle: torch.Tensor) -> torch.Tensor:
  return (angle + math.pi).remainder(2 * math.pi) - math.pi


def from_local(pose: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
  cosine, sine = anchor[..., 2].cos(), anchor[..., 2].sin()
  return torch.stack((anchor[..., 0] + cosine * pose[..., 0] - sine * pose[..., 1],
                      anchor[..., 1] + sine * pose[..., 0] + cosine * pose[..., 1],
                      wrap(anchor[..., 2] + pose[..., 2])), dim=-1)


def to_local(pose: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
  delta = pose[..., :2] - anchor[..., :2]
  cosine, sine = anchor[..., 2].cos(), anchor[..., 2].sin()
  return torch.stack((cosine * delta[..., 0] + sine * delta[..., 1],
                      -sine * delta[..., 0] + cosine * delta[..., 1],
                      wrap(pose[..., 2] - anchor[..., 2])), dim=-1)


class TensorFootstepSampler:
  """使用给定的均匀分位数，通过逆累积分布采样截断正态落点"""

  def __init__(self, cfg: FootstepSamplerCfg):
    self.cfg = cfg
    lower, upper = cfg.distance_range
    self.reflect = lower > cfg.distance_mean
    if self.reflect:
      lower, upper = 2 * cfg.distance_mean - upper, 2 * cfg.distance_mean - lower
    distribution = NormalDist(cfg.distance_mean, cfg.distance_std)
    self.probability_low = distribution.cdf(lower)
    self.probability_high = distribution.cdf(upper)
    if lower != upper and self.probability_high <= self.probability_low:
      raise ValueError("Truncated normal interval has insufficient numerical probability")

  def parameters(self, device):
    """返回可原地更新的距离分布及站距参数，不包含任务课程"""
    cfg = self.cfg
    return torch.tensor((*cfg.distance_range, cfg.distance_mean, cfg.distance_std,
                         self.probability_low, self.probability_high, float(self.reflect),
                         cfg.min_width, cfg.max_width), dtype=torch.float64, device=device)

  def sample(self, previous, side, direction, heading, uniform, parameters=None, yaw_noise=None):
    cfg = self.cfg
    minimum_width, maximum_width, maximum_distance = cfg.min_width, cfg.max_width, cfg.distance_range[1]
    width_center = cfg.width_center
    if parameters is not None:
      lower, upper, mean, std, probability_low, probability_high, reflect, minimum_width, maximum_width = parameters.unbind()
      probability = probability_low + uniform[..., 0] * (probability_high - probability_low)
      probability = probability.clamp(torch.finfo(probability.dtype).tiny, 1 - torch.finfo(probability.dtype).eps)
      sampled = torch.special.ndtri(probability) * std + mean
      sampled = torch.where(reflect > 0, 2 * mean - sampled, sampled)
      distance = torch.where(lower == upper, lower, sampled)
      maximum_distance = upper
      width_center = (minimum_width + maximum_width) / 2
    elif cfg.distance_range[0] == cfg.distance_range[1]:
      distance = torch.full_like(direction, cfg.distance_range[0])
    else:
      probability = self.probability_low + uniform[..., 0] * (self.probability_high - self.probability_low)
      probability = probability.clamp(torch.finfo(probability.dtype).tiny, 1 - torch.finfo(probability.dtype).eps)
      distance = torch.special.ndtri(probability) * cfg.distance_std + cfg.distance_mean
      if self.reflect:
        distance = 2 * cfg.distance_mean - distance
    angle = direction + cfg.direction_noise[0] + uniform[..., 1] * (cfg.direction_noise[1] - cfg.direction_noise[0]) - previous[..., 2]
    sign = 1 - 2 * side
    lateral = sign * (width_center + sign * distance * angle.sin()).clamp(minimum_width, maximum_width)
    forward_limit = (maximum_distance ** 2 - lateral.square()).clamp_min(0).sqrt()
    forward = (distance * angle.cos()).clamp(-forward_limit, forward_limit)
    yaw_lower, yaw_upper = cfg.yaw_noise if yaw_noise is None else yaw_noise.unbind()
    yaw = heading + yaw_lower + uniform[..., 2] * (yaw_upper - yaw_lower)
    delta_yaw = wrap(yaw - previous[..., 2]).clamp(-cfg.max_yaw_change, cfg.max_yaw_change)
    return from_local(torch.stack((forward, lateral, delta_yaw), dim=-1), previous)