"""先直行、再全向变步长和方向、最后随机化逐脚朝向"""

import math

from mjlab.managers.curriculum_manager import CurriculumTermCfg

from g1_lower_rl.tasks.footstep_tracking.curriculum import NUM_STEPS_PER_ENV, _select_stage, arm_pose_amplitude


WALK_FIRST_STAGES = (
  (0, 0.0, (0.27, 0.27), (0.24, 0.24), 0.0),
  (1500, math.pi / 6, (0.27, 0.27), (0.21, 0.27), 0.0),
  (1750, math.pi / 3, (0.27, 0.27), (0.21, 0.27), 0.0),
  (2000, 2 * math.pi / 3, (0.27, 0.27), (0.21, 0.27), 0.0),
  (2250, math.pi, (0.27, 0.27), (0.21, 0.27), 0.0),
  (2500, math.pi, (0.24, 0.30), (0.18, 0.30), math.pi / 12),
  (3000, math.pi, (0.20, 0.36), (0.12, 0.36), math.pi / 6),
  (3500, math.pi, (0.16, 0.40), (0.12, 0.36), math.pi / 3),
  (4000, math.pi, (0.12, 0.48), (0.12, 0.36), math.pi),
)

WALK_FIRST_YAW_STAGES = (
  (0, 0.0),
  (4500, math.radians(10)),
  (5000, math.radians(20)),
  (5500, math.radians(30)),
)


def walk_first_stage(iteration: int, stages=WALK_FIRST_STAGES):
  """按PPO迭代选择采样区间，不按物理控制拍推进档位"""
  return _select_stage(stages, iteration)


def walk_first_commands(env, env_ids, stages=WALK_FIRST_STAGES, num_steps_per_env=NUM_STEPS_PER_ENV,
                        yaw_stages=WALK_FIRST_YAW_STAGES):
  """更新后续采样范围，不修改已经承诺的脚印或进行中的步态"""
  del env_ids
  iteration = env.common_step_counter // num_steps_per_env
  start, angle, distance, width, change_angle = walk_first_stage(iteration, stages)
  yaw_start, yaw_angle = _select_stage(yaw_stages, iteration)
  batch = env.command_manager.get_term("footsteps").batch
  batch.initial_direction_range.copy_(batch.initial_direction_range.new_tensor((-angle, angle)))
  batch.change_ranges[0].copy_(batch.change_ranges.new_tensor((-change_angle, change_angle)))
  batch.set_sampling_ranges(distance, width, yaw_noise=(-yaw_angle, yaw_angle))
  return {"iteration_start": start, "initial_direction_half_range_rad": angle,
          "yaw_iteration_start": yaw_start, "foot_yaw_half_range_rad": yaw_angle,
          "direction_change_half_range_rad": change_angle, "distance_min_m": distance[0],
          "distance_max_m": distance[1], "width_min_m": width[0], "width_max_m": width[1]}


def make_walk_first_curriculum():
  """保留固定步频与宽容奖励，扩大空间指令和手臂目标幅度"""
  return {
    "walk_first_commands": CurriculumTermCfg(func=walk_first_commands, params={
      "stages": WALK_FIRST_STAGES, "num_steps_per_env": NUM_STEPS_PER_ENV, "yaw_stages": WALK_FIRST_YAW_STAGES,
    }),
    "arm_pose_amplitude": CurriculumTermCfg(func=arm_pose_amplitude, params={
      "num_steps_per_env": NUM_STEPS_PER_ENV,
    }),
  }