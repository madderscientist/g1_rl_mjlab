"""动作与指令配置。

策略输出 15 个关节位置增量（12 腿 + 3 腰），指令是上层发下来的
``[vx, vy, omega, height]`` 四个浮点数。
"""

from __future__ import annotations

import math

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg

from g1_lower_rl.assets import LOWER_BODY_ACTION_SCALE, LOWER_BODY_ACTUATOR_EXPR
from g1_lower_rl.tasks.lower_body import mdp
from g1_lower_rl.tasks.lower_body.cfg.constants import (
  FOOT_SITES,
  HEIGHT_STAGES,
  STANDING_RATIO,
  STRAIGHT_RATIO,
  TURNING_RATIO,
  VELOCITY_STAGES,
)


def make_actions() -> dict[str, ActionTermCfg]:
  return {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=LOWER_BODY_ACTUATOR_EXPR,
      scale=dict(LOWER_BODY_ACTION_SCALE),
      use_default_offset=True,
    )
  }


def make_commands() -> dict[str, CommandTermCfg]:
  return {
    "twist": mdp.ScenarioVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(3.0, 8.0),
      rel_standing_envs=STANDING_RATIO,
      # yaw 直接下指令，不走航向控制器。mjlab 的 rel_heading_envs 默认是 1.0，会把采样出来的
      # yaw 速率换成 stiffness * heading_error——机器人一转它就衰减到零。结果是策略在训练里
      # 根本没见过持续的 yaw 指令，而实机上操作员发的恰恰就是持续值。
      # heading_command 仍为 True：heading_target / heading_error 还要给航向保持用。
      heading_command=True,
      rel_heading_envs=0.0,
      rel_turning_envs=TURNING_RATIO,
      # 强制 ω = 0、只走不转。不显式指定的话实测只有 4.9% 的环境是纯直行，
      # 而 76.5% 在边走边转（行走环境里 |ω| 中位数 0.822 rad/s，已顶到量程上限）。
      # 这些环境里 track_angular_velocity_avg 的指令是 0，于是它直接变成一道
      # “累计航向漂移惩罚”——不需要额外的项。
      rel_straight_envs=STRAIGHT_RATIO,
      debug_vis=True,
      viz=mdp.ScenarioVelocityCommandCfg.VizCfg(z_offset=1.15),
      # 量程由 command_vel 课程逐档放开，这里是第一档。
      ranges=mdp.ScenarioVelocityCommandCfg.Ranges(
        **VELOCITY_STAGES[0][1],
        heading=(-math.pi, math.pi),
      ),
    ),
    "height": mdp.BaseHeightCommandCfg(
      entity_name="robot",
      resampling_time_range=(3.0, 8.0),
      site_names=FOOT_SITES,
      # 在编译出来的模型上量过：足底 site 就在鞋底，所以这个指令就是卷尺量的骨盆离地高度。
      # 下界对应膝关节弯 115 度，是关节行程能给到的最深。
      #
      # 上界 0.80 故意高于 0.7919 m 的运动学硬上限（完全直腿）。够不到的那 8 mm 在 std=0.05
      # 下只损失 2.6% 的奖励，代价可忽；好处是这一档的指令永远满足不了，策略只能尽量往高里顶，
      # 配合 straight_knee 就是“尽量高而直”。量程由 command_height 课程逐档放开。
      ranges=mdp.BaseHeightCommandCfg.Ranges(height=HEIGHT_STAGES[0][1]),
    ),
  }
