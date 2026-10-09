"""第二阶段保留脚印课程，启用步频游走并渐入落脚代价"""

from dataclasses import replace

from g1_lower_rl.tasks.footstep_tracking.curriculum import FREQUENCY_RATE_STAGES, NUM_STEPS_PER_ENV, make_curriculum

from .env_cfg import _staged_env_cfg


def precision_env_cfg(play: bool = False, *, landing_start_step: int = 0,
                      landing_ramp_steps: int = 2000 * NUM_STEPS_PER_ENV):
  if landing_start_step < 0 or landing_ramp_steps <= 0:
    raise ValueError("Landing ramp start must be nonnegative and duration positive")
  cfg = _staged_env_cfg(play)
  command = cfg.commands["footsteps"]
  command.manager = replace(command.manager, frequency_range=(0.8, 1.8))
  command.source = replace(command.source, frequency_range=(0.8, 1.8),
                           frequency_rate_range=FREQUENCY_RATE_STAGES[0][1])
  cfg.curriculum["command_frequency_rate"] = make_curriculum()["command_frequency_rate"]
  landing = cfg.rewards.pop("footstep_landing")
  landing.weight = -0.1
  landing.params.update(landing_start_step=landing_start_step, landing_ramp_steps=landing_ramp_steps)
  cfg.rewards["footstep_landing"] = landing
  return cfg