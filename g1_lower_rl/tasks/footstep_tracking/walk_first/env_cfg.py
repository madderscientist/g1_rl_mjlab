"""先学直行，再扩展全向和变化脚印，保留宽容的精度引导"""

from dataclasses import replace

from g1_lower_rl.tasks.footstep_tracking.env_cfg import footstep_env_cfg
from g1_lower_rl.tasks.footstep_tracking.walk_first.curriculum import make_walk_first_curriculum


def _staged_env_cfg(play: bool):
  """两阶段共用固定请求步频和逐步扩展的脚印课程"""
  cfg = footstep_env_cfg(play=play)
  command = cfg.commands["footsteps"]
  sampler = replace(
    command.manager.sampler,
    distance_range=(0.27, 0.27), distance_mean=0.27, distance_std=0.08,
    min_width=0.24, max_width=0.24,
    direction_noise=(0.0, 0.0), yaw_noise=(0.0, 0.0),
  )
  command.manager = replace(
    command.manager, sampler=sampler,
    frequency_range=(1.2, 1.2), initial_frequency=1.2,
  )
  command.source = replace(
    command.source, frequency_range=(1.2, 1.2), initial_frequency=1.2,
    frequency_rate_range=(0.0, 0.0), direction_range=(0.0, 0.0),
    direction_change_range=(0.0, 0.0), foot_heading_range=(0.0, 0.0),
  )
  cfg.curriculum = make_walk_first_curriculum()
  return cfg


def walk_first_env_cfg(play: bool = False):
  """第一阶段用宽容的精度引导学习行走"""
  cfg = _staged_env_cfg(play)
  cfg.rewards.pop("footstep_landing")
  for name, weight, parameter, width in (
    ("footstep_swing_position", 1.0, "swing_position_std", 0.4),
    ("footstep_swing_yaw", 0.5, "swing_yaw_std", 0.3),
  ):
    reward = cfg.rewards[name]
    reward.weight = weight
    reward.params[parameter] = width
    reward.params["swing_progress_power"] = 1.0
  return cfg