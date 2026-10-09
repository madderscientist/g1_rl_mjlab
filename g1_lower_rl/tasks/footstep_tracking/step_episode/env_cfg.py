"""Single-step scene configuration with unchanged shared rewards."""

from dataclasses import fields, replace
import math

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from g1_lower_rl.assets import LOWER_BODY_JOINTS
from g1_lower_rl.tasks.footstep_tracking.walking.precision import precision_env_cfg

from .commands import StepEpisodeCommandCfg, step_episode_complete
from .rewards import StepLandingBonus, standing_action_change_cost, standing_joint_velocity_cost, step_failure_cost


def step_episode_env_cfg(play=False):
  cfg = precision_env_cfg(play=play, landing_start_step=0, landing_ramp_steps=1)
  original = cfg.commands["footsteps"]
  command = StepEpisodeCommandCfg(**{field.name: getattr(original, field.name) for field in fields(original)})
  command.sample_hold = not play
  sampler = replace(command.manager.sampler, distance_range=(.12, .35), distance_mean=.24,
    distance_std=.06, min_width=.12, max_width=.30, direction_noise=(0., 0.), yaw_noise=(0., 0.))
  command.manager = replace(command.manager, sampler=sampler, start_duration_s=.25, stop_duration_s=.25)
  command.source = replace(command.source, initial_standing_probability=1., initial_frequency=None,
    automatic_commands=False, automatic_restart=False, frequency_rate_range=(0., 0.),
    direction_range=(-math.pi, math.pi), foot_heading_range=(0., 0.))
  cfg.commands["footsteps"] = command
  cfg.curriculum = {}
  if play:
    cfg.events["reset_arm_pose"].params.update(target_scale=0., ramp_duration_s=0.)
    for name in ("arm_pose_ramp", "arm_pose_drift", "arm_torque"):
      cfg.events.pop(name, None)
  else:
    cfg.events["reset_arm_pose"].params["target_scale"] = 1.
    cfg.events["arm_pose_drift"].params["target_scale"] = 1.
  cfg.events["reset_base"].params.update(pose_range={}, velocity_range={})
  cfg.events["reset_robot_joints"].params.update(position_range=(0., 0.), velocity_range=(0., 0.))
  cfg.terminations["step_complete"] = TerminationTermCfg(func=step_episode_complete)
  cfg.rewards["fall"].func = step_failure_cost
  cfg.rewards["fall"].weight = -100.
  for name in ("footstep_swing_position", "footstep_swing_yaw"):
    cfg.rewards[name].params["swing_progress_power"] = 1.
  cfg.rewards["step_landing_bonus"] = RewardTermCfg(func=StepLandingBonus, weight=10., params={
    "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot"), preserve_order=True),
  })
  cfg.rewards["standing_joint_velocity"] = RewardTermCfg(func=standing_joint_velocity_cost, weight=-.2, params={
    "asset_cfg": SceneEntityCfg("robot", joint_names=LOWER_BODY_JOINTS, preserve_order=True),
  })
  cfg.rewards["standing_action_change"] = RewardTermCfg(func=standing_action_change_cost, weight=-.04)
  cfg.episode_length_s = 12.
  return cfg