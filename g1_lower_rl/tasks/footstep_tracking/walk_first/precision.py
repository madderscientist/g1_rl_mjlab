"""Precision training with walk-first commands and a gradual landing cost."""

from g1_lower_rl.tasks.footstep_tracking.curriculum import NUM_STEPS_PER_ENV
from g1_lower_rl.tasks.footstep_tracking.walk_first.env_cfg import _staged_env_cfg


def precision_env_cfg(play: bool = False, *, landing_start_step: int = 0,
                      landing_ramp_steps: int = 2000 * NUM_STEPS_PER_ENV):
  if landing_start_step < 0 or landing_ramp_steps <= 0:
    raise ValueError("Landing ramp start must be nonnegative and duration positive")
  cfg = _staged_env_cfg(play)
  landing = cfg.rewards.pop("footstep_landing")
  landing.weight = -0.1
  landing.params.update(landing_start_step=landing_start_step, landing_ramp_steps=landing_ramp_steps)
  cfg.rewards["footstep_landing"] = landing
  return cfg