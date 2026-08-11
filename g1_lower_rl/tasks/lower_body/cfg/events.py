"""事件配置：开局随机化、上肢扰动、域随机化。

**顺序有意义。** ``disturbance_level`` 必须排在所有复位事件最前面——事件按配置顺序
执行，而后面两个复位事件要用它采出来的系数。
"""

from __future__ import annotations

from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from g1_lower_rl.tasks.lower_body import mdp
from g1_lower_rl.tasks.lower_body.cfg.constants import (
  ARM_DRIFT_LEVELS,
  ARM_JOINT_EXPR,
  ARM_TORQUE_LEVELS,
  BODY_IMPULSE_LEVELS,
  LOWER_BODY_JOINT_EXPR,
  RESET_LEVELS,
  joints,
)


def make_events() -> dict[str, EventTermCfg]:
  return {
    # 每个 episode 给本环境采一个扰动强度系数，开局姿态和三个扭动/推力事件共用它。
    # 全局课程仍然决定上包络，这里决定每个环境占其中多少，于是同一批里总有安静的
    # episode 可供“温故”。
    "disturbance_level": EventTermCfg(
      func=mdp.resample_disturbance_level,
      mode="reset",
      params={"level_range": (0.0, 1.0), "quiet_prob": 0.25},
    ),
    # 开局故意不给一个干净的站姿：整个身体相对地面带倾角、带初速度。要的就是让策略学会
    # 从差状态里抓回来，而不是只会从标准姿势起步。但幅值不能一上来就给满：乘上“课程包络
    # x 逐环境系数”，第一档是 0（干净开局）。yaw 不参与缩放——它是朝向覆盖，不是难度。
    "reset_base": EventTermCfg(
      func=mdp.scaled_reset_root_state,
      mode="reset",
      params={
        "pose_range": {
          "yaw": (-3.14, 3.14),
          "roll": (-0.35, 0.35),
          "pitch": (-0.35, 0.35),
        },
        "velocity_range": {
          "x": (-0.6, 0.6),
          "y": (-0.6, 0.6),
          "z": (-0.3, 0.3),
          "roll": (-0.8, 0.8),
          "pitch": (-0.8, 0.8),
          "yaw": (-0.8, 0.8),
        },
        "scale": RESET_LEVELS[0][1],  # 由课程逐档抬高。
      },
    ),
    # 关节只给小幅扰动。“从恶劣初始条件恢复”靠的是上面那个整身倾角，这里只是不让每局
    # 都从完全相同的关节角开始。初速度同样受课程与逐环境系数缩放，关节角不缩。
    "reset_robot_joints": EventTermCfg(
      func=mdp.scaled_reset_joints,
      mode="reset",
      params={
        "position_range": (-0.15, 0.15),
        "velocity_range": (-0.5, 0.5),
        "scale": RESET_LEVELS[0][1],  # 由课程逐档抬高。
        "asset_cfg": joints(*LOWER_BODY_JOINT_EXPR),
      },
    ),
    # 步态相位的起点也逐环境随机。相位时钟是 episode_length_buf * dt，每局必从 0 开始，
    # 而相位 0 对应右脚先迈，于是 85% 带运动指令 reset 的环境都是右脚先迈。
    "gait_phase": EventTermCfg(func=mdp.resample_gait_phase, mode="reset", params={}),
    # 每次 reset 都必须跑，否则手臂会被驱到零位。
    "reset_arm_pose": EventTermCfg(
      func=mdp.hold_arm_pose,
      mode="reset",
      params={
        "asset_cfg": joints(*ARM_JOINT_EXPR),
        "ranges": {},  # 按机器人设置。
        "write_state": True,
      },
    ),
    # episode 中途让手臂真的摆起来。
    "arm_pose_drift": EventTermCfg(
      func=mdp.hold_arm_pose,
      mode="interval",
      interval_range_s=(1.0, 4.0),
      params={
        "asset_cfg": joints(*ARM_JOINT_EXPR),
        "ranges": {},  # 按机器人设置。
        "write_state": False,
        "blend": ARM_DRIFT_LEVELS[0][1],  # 由课程逐档抬高。
        "scale_by_level": True,
      },
    ),
    "arm_torque": EventTermCfg(
      func=mdp.arm_torque_impulse,
      mode="step",
      params={
        "asset_cfg": joints(*ARM_JOINT_EXPR),
        "torque_range": (-ARM_TORQUE_LEVELS[0][1], ARM_TORQUE_LEVELS[0][1]),
        "duration_s": (0.2, 0.6),
        # 站立时的倒立摆模态实测 0.64~0.73 Hz（周期 1.4 s），而原来的 (1.0, 3.0) 平均
        # 间隔只有 2 s，与周期同量级——等于每次都在摆动衰减完之前重新激发它，
        # 策略看不到“稳下来”长什么样。
        "cooldown_s": (1.5, 4.0),
      },
    ),
    "body_impulse": EventTermCfg(
      func=mdp.scaled_body_impulse,
      mode="step",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # 按机器人设置。
        "force_range": (-BODY_IMPULSE_LEVELS[0][1], BODY_IMPULSE_LEVELS[0][1]),
        "torque_range": (-2.0, 2.0),
        "duration_s": (0.1, 0.4),
        "cooldown_s": (2.5, 5.5),  # 同上。
        "body_point_offset": (0.0, 0.0, 0.1),
      },
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(5.0, 6.0),
      params={
        "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.4, 0.4),
          "roll": (-0.52, 0.52),
          "pitch": (-0.52, 0.52),
          "yaw": (-0.78, 0.78),
        },
      },
    ),
    "payload_mass": EventTermCfg(
      mode="startup",
      func=dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # 按机器人设置。
        "operation": "add",
        "ranges": (0.0, 2.0),
      },
    ),
    # 每只脚一个事件，各自内部共享一个采样值，于是两只脚的摩擦相互独立。合成一个事件的话
    # 机器人在接触处完全对称，永远不需要主动保持航向——而那正是它在实机上栽掉的地方。
    "foot_friction_left": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # 按机器人设置。
        "operation": "abs",
        "ranges": (0.3, 1.6),
        "shared_random": True,
      },
    ),
    "foot_friction_right": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # 按机器人设置。
        "operation": "abs",
        "ranges": (0.3, 1.6),
        "shared_random": True,
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.015, 0.015),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # 按机器人设置。
        "operation": "add",
        "ranges": {0: (-0.05, 0.05), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
      },
    ),
  }
