"""任务级常量：关节集合、课程档位表、闸门。

档位表放在这里而不是各自的配置模块里，因为它们是**跨模块共享**的：``events`` 用第一档
作初值，``curriculum`` 用整张表推进，``play`` 用最后一档还原训练末期的工况。
"""

from __future__ import annotations

import math

from mjlab.managers.scene_entity_config import SceneEntityCfg

# 策略控制的关节，顺序与真机电机编号一致。
LOWER_BODY_JOINT_EXPR = (
  r".*_hip_(pitch|roll|yaw)_joint",
  r".*_knee_joint",
  r".*_ankle_(pitch|roll)_joint",
  r"waist_(yaw|roll|pitch)_joint",
)
# 无论高度指令是多少都应该待在原位的关节。hip_pitch、knee、ankle_pitch 故意不在里面：
# 蹲起靠它们承担，由 track_base_height 管，而不是由姿态奖励管。
LATERAL_JOINT_EXPR = (
  r".*_hip_roll_joint",
  r".*_hip_yaw_joint",
  r".*_ankle_roll_joint",
)
ARM_JOINT_EXPR = (
  r".*_shoulder_(pitch|roll|yaw)_joint",
  r".*_elbow_joint",
  r".*_wrist_(roll|pitch|yaw)_joint",
)


def joints(*expr: str) -> SceneEntityCfg:
  """每个项都新建一份实体配置。管理器会就地解析它们，多个项共用同一个实例会让第二次
  解析的一致性检查失败。"""
  return SceneEntityCfg("robot", joint_names=expr)


# 课程档位写成「迭代数 x ITER」换算成环境步数，因为课程的判据是 ``common_step_counter``，
# 而它每调一次 ``env.step()`` 加一。所以 ITER 必须等于该任务的 ``num_steps_per_env``：
# 调大了步长却不改这里，整条课程会按比例提前触发，而且不会报错。GRU 版用了别的步长，
# 靠 ``g1_gloria._rescale_curriculum_steps`` 把 step 缩回去；``tasks/__init__`` 里有断言兜底。
ITER = 24

##
# 扰动课程档位，形式为 (迭代数, 幅值)。
##

# 上肢扰动。三条都是撞出来的经验：步态没出现之前扰动必须关着；一档内把幅值翻倍会永久
# 搞崩步态；档位上限必须落在策略能顶着走的范围内——30 N / 5 Nm 时它就不往前走、改成
# 原地踏步了。推进还受跟随质量抓闸，见 TRACKING_GATE。
ARM_TORQUE_LEVELS = ((0, 0.0), (500, 1.5), (1500, 2.5), (2600, 3.2), (3700, 4.0))
ARM_DRIFT_LEVELS = ((0, 0.0), (550, 0.3), (1650, 0.6), (2800, 1.0))
BODY_IMPULSE_LEVELS = ((0, 0.0), (600, 8.0), (1800, 12.0), (2900, 17.0), (3950, 20.0))

# 开局姿态的难度包络，取值是配置里那组幅值的倍率。0 就是干净开局。
# 实测：开局直接给到满幅（倾角 20 度 + 初速度）时，从零训练的中位存活只有 42 步
# （0.84 s），第 700 迭代仍卡在 51 步；而开局干净的那版同期已经跑满 969 步。
RESET_LEVELS = ((0, 0.0), (650, 0.2), (2000, 0.6), (2700, 1.0))

##
# 指令量程课程。
##

# 末档上界 1.3 m/s 不是保守：腿长 0.79 m 对应的行走 Froude 数上限约 1.97 m/s，
# 而 foot_gait 的 threshold=0.56 结构上就不允许腾空相，实测从没有策略超过过 1.1 m/s。
# 指令给到跑不到的量程只会把跟随奖励推进没有梯度的平地。
VELOCITY_STAGES = (
  (0, {"lin_vel_x": (-0.5, 0.8), "lin_vel_y": (-0.4, 0.4), "ang_vel_z": (-0.8, 0.8)}),
  (
    1850,
    {"lin_vel_x": (-0.8, 1.1), "lin_vel_y": (-0.6, 0.6), "ang_vel_z": (-1.1, 1.1)},
  ),
  (
    2850,
    {
      "lin_vel_x": (-1.0, 1.3),
      "lin_vel_y": (-0.8, 0.8),
      "ang_vel_z": (-math.pi / 2, math.pi / 2),
    },
  ),
)

# 高度指令的量程逐档放开。第一档卡在标称站姿附近，先把站住学会。
# 末档必须与 ``VELOCITY_STAGES`` 的档错开。实测过一次两者同落在第 3000 迭代：单拍奖励从
# 32.6 掉到 7.2、速度跟随误差从 0.90 涨到 1.60，前进能力直接报废且 2200 迭代没恢复。
HEIGHT_STAGES = ((0, (0.70, 0.77)), (1300, (0.65, 0.80)), (3050, (0.50, 0.80)))

# 高度跟随奖励核的 std 逐档收紧：宽核负责自举，窄核负责精度。
# 0.05 是想要的终值（那时候比指令低 5 cm 只剩 0.37 的奖励，能把“蹲着走”压掉），但从第 0
# 迭代就用它会把从零训练锁死——exp(-e²/std²) 在 e >> std 时是数值上的平地，高度误差
# 0.15 m 处 std=0.05 只给 9.9e-5 而 std=0.08 给 0.022，相差 226 倍，而成型期的机器人
# 正好就在那个区间。收紧还要过闸门，见 HEIGHT_STD_GATE。
HEIGHT_STD_STAGES = ((0, 0.08), (1500, 0.065), (3500, 0.05))

# 左右镜像数据增强（DUP）的强度逐档抬高，取值是逐样本的镜像概率。由 scripts/train.py
# 读走，档位进度由 common_step_counter 推算，因而续训会自动落回正确档位。
#
# 时机是测出来的：沿 air3yaw2 血统逐检查点量左右腿腾空占比，前 800 迭代两只脚都是精确的
# 0（蹭地走，根本没有步态）；1100 迭代第一次抬脚，而那一次就是单侧的（左 0.0% / 右 8.6%），
# 之后再没收敛回来。破缺不是逐步积累的，是在步态诞生那一刻一次成型的。
MIRROR_STAGES = ((0, 0.0), (800, 0.2), (1500, 0.3), (2000, 0.5))

##
# 闸门。
##

# 只有当至少一半环境已经把高度误差压进 5 cm，才允许收紧奖励核。
# 用绝对精度而不是“相对当前 std 的精度”作判据：收紧的前提是它确实已经站到位了，
# 否则收紧只会把它推回没有梯度的那片平地。
HEIGHT_STD_GATE = {"max_error": 0.05, "min_fraction": 0.5, "ema": 0.01}

# 只有当至少这么大比例的运动环境把速度跟随误差控制在 max_error 内，扰动课程才往下一档走。
TRACKING_GATE = {
  "command_name": "twist",
  "min_command": 0.3,
  "max_error": 0.6,
  "min_fraction": 0.3,
  "ema": 0.02,
}

##
# 指令场景占比，固定值，不随训练变化。
##

# 这是整套设置里最贵的一个数：站立环境把两个跟随奖励（合计 5.0）白送给“冻住不动”，
# 同时所有步态奖励都被 |指令| > 0.1 的门槛关掉，于是它们的最优解就是雕像。
# 曾试过让它随存活率从 0.9 动态降下来，但把从零训练锁死在雕像解上了。
STANDING_RATIO = 0.1  # 站着不动
TURNING_RATIO = 0.1  # 原地转弯
STRAIGHT_RATIO = 0.15  # 直走（强制 ω = 0）
