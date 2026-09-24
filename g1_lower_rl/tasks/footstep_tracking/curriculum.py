"""脚步任务的指令采样区间和手臂目标幅度课程，不调整奖励"""

import math

from mjlab.managers.curriculum_manager import CurriculumTermCfg

NUM_STEPS_PER_ENV = 64
ARM_TARGET_SCALE_STAGES = ((800, 0.2), (1500, 0.5), (2000, 1.0))
# 移动方向和原方向变化角度区间
DIRECTION_CHANGE_STAGES = (
  (0, (0.0, 0.0)),
  (1500, (-math.pi / 4, math.pi / 4)),
  (2500, (-math.pi / 2, math.pi / 2)),
  (4000, (-math.pi, math.pi)),
)
# 步频随机游走变化区间
FREQUENCY_RATE_STAGES = (
  (0, (-0.01, 0.01)),
  (800, (-0.15, 0.15)),
  (2000, (-0.3, 0.3)),
)


def _select_stage(stages, iteration):
  """选取已经达到的最后一个阶段，不对相邻阶段插值"""
  selected = stages[0]
  for stage in stages[1:]:
    if iteration < stage[0]:
      break
    selected = stage
  return selected


def command_direction(env, env_ids, stages=DIRECTION_CHANGE_STAGES, num_steps_per_env=NUM_STEPS_PER_ENV):
  """独立更新方向变化角度区间，不改频率区间或当前执行状态"""
  del env_ids
  iteration_start, bounds = _select_stage(stages, env.common_step_counter // num_steps_per_env)
  target = env.command_manager.get_term("footsteps").batch.change_ranges[0]
  target.copy_(target.new_tensor(bounds))
  return {"iteration_start": iteration_start, "min_rad": bounds[0], "max_rad": bounds[1]}


def command_frequency_rate(env, env_ids, stages=FREQUENCY_RATE_STAGES, num_steps_per_env=NUM_STEPS_PER_ENV):
  """独立更新频率变化率区间，不改方向区间或当前执行状态"""
  del env_ids
  iteration_start, bounds = _select_stage(stages, env.common_step_counter // num_steps_per_env)
  target = env.command_manager.get_term("footsteps").batch.change_ranges[1]
  target.copy_(target.new_tensor(bounds))
  return {"iteration_start": iteration_start, "min_hz_s": bounds[0], "max_hz_s": bounds[1]}


def arm_target_scale(iteration: int) -> float:
  """前800轮固定0.2，随后线性插值到1500轮0.5及2000轮1"""
  if iteration <= ARM_TARGET_SCALE_STAGES[0][0]:
    return ARM_TARGET_SCALE_STAGES[0][1]
  for (start, lower), (end, upper) in zip(ARM_TARGET_SCALE_STAGES, ARM_TARGET_SCALE_STAGES[1:]):
    if iteration < end:
      return lower + (upper - lower) * (iteration - start) / (end - start)
  return ARM_TARGET_SCALE_STAGES[-1][1]


def arm_pose_amplitude(env, env_ids, num_steps_per_env=NUM_STEPS_PER_ENV):
  """只更新后续reset采样的最终角度系数，不改变进行中的五秒渐变"""
  del env_ids
  scale = arm_target_scale(env.common_step_counter // num_steps_per_env)
  env.event_manager.get_term_cfg("reset_arm_pose").params["target_scale"] = scale
  return {"target_scale": scale}


def make_curriculum() -> dict[str, CurriculumTermCfg]:
  """分别注册方向、频率变化率与手臂幅度课程及日志"""
  return {
    "arm_pose_amplitude": CurriculumTermCfg(func=arm_pose_amplitude, params={
      "num_steps_per_env": NUM_STEPS_PER_ENV,
    }),
    "command_direction": CurriculumTermCfg(func=command_direction, params={
      "stages": DIRECTION_CHANGE_STAGES, "num_steps_per_env": NUM_STEPS_PER_ENV,
    }),
    "command_frequency_rate": CurriculumTermCfg(func=command_frequency_rate, params={
      "stages": FREQUENCY_RATE_STAGES, "num_steps_per_env": NUM_STEPS_PER_ENV,
    }),
  }