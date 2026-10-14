"""Explicit post-150351 theta stage; not enabled by default task profiles."""

from dataclasses import replace

from mjlab.envs.mdp import generated_commands
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg

from . import make_footstep_env_cfg
from .commands import TorsoHeadingCommandCfg
from .curriculum import FREQUENCY_RATE_STAGES, NUM_STEPS_PER_ENV, _select_stage, arm_target_scale, torso_heading_range
from .rewards_cfg import make_torso_heading_reward
from .walking.curriculum import WALK_FIRST_STAGES, WALK_FIRST_YAW_STAGES


def make_theta_stage_env_cfg(profile, theta_stage, rank, play=False, *, precision_stage=None):
  if profile not in ("precision", "step-episode") or rank not in (0, 1):
    raise ValueError("Theta stage requires a walking/step profile and task rank 0 or 1")
  origin = theta_stage["task_start_steps"][rank]
  if not isinstance(origin, int) or origin < 0:
    raise ValueError("Theta stage requires a nonnegative task origin")
  cfg = make_footstep_env_cfg(profile, play=play, precision_stage=precision_stage)
  cfg.commands["torso_heading"] = TorsoHeadingCommandCfg(
    curriculum_start_step=origin, angle_limit=theta_stage["stages"][0][1],
    resampling_time_range=tuple(theta_stage["resampling_time_range_s"]),
    speed_range=tuple(theta_stage["speed_range_rad_s"]),
  )
  for group in cfg.observations.values():
    original = group.terms
    term = ObservationTermCfg(func=generated_commands, params={"command_name": "torso_heading"})
    group.terms = {}
    for name, value in original.items():
      if name == "privileged":
        group.terms["theta"] = term
      group.terms[name] = value
    if "theta" not in group.terms:
      group.terms["theta"] = term
  cfg.rewards.pop("waist_yaw_zero")
  cfg.rewards.pop("standing_pelvis_upright", None)
  cfg.rewards["torso_feet_heading"] = make_torso_heading_reward()
  iteration = origin // NUM_STEPS_PER_ENV
  if profile == "precision":
    _, angle, distance, width, change = _select_stage(WALK_FIRST_STAGES, iteration)
    _, yaw = _select_stage(WALK_FIRST_YAW_STAGES, iteration)
    _, frequency_rate = _select_stage(FREQUENCY_RATE_STAGES, iteration)
    command = cfg.commands["footsteps"]
    command.manager = replace(command.manager, sampler=replace(command.manager.sampler,
      distance_range=distance, min_width=width[0], max_width=width[1], yaw_noise=(-yaw, yaw)))
    command.source = replace(command.source, direction_range=(-angle, angle),
      direction_change_range=(-change, change), frequency_rate_range=frequency_rate)
    landing = cfg.rewards["footstep_landing"].params
    landing["landing_start_step"] = max(0, origin - landing["landing_ramp_steps"])
  if not play or profile == "precision":
    scale = arm_target_scale(iteration)
    for name in ("reset_arm_pose", "arm_pose_drift"):
      if name in cfg.events:
        cfg.events[name].params["target_scale"] = scale
  cfg.curriculum = {"torso_heading_range": CurriculumTermCfg(func=torso_heading_range, params={
    "stages": tuple(tuple(stage) for stage in theta_stage["stages"]), "num_steps_per_env": NUM_STEPS_PER_ENV,
  })}
  return cfg