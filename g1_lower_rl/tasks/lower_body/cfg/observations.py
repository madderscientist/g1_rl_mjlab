"""观测配置。

actor 看到的每一项都能在真机上由骨盆 IMU、关节编码器和上一拍动作重建出来，所以整个
观测向量可以被部署端逐项复现。**顺序是契约的一部分**：部署端必须按同样的顺序拼观测。
特权量只放在 critic 组里。
"""

from __future__ import annotations

import copy

from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.noise import GaussianNoiseCfg as Gnoise
from mjlab.utils.noise import NoiseModelWithAdditiveBiasCfg

from g1_lower_rl.tasks.lower_body import mdp
from g1_lower_rl.tasks.lower_body.cfg.constants import (
  ARM_JOINT_EXPR,
  LOWER_BODY_JOINT_EXPR,
  joints,
)

# 延时是机器人的属性，不是每一拍的属性：每步重采一次滞后量等于给关节速度叠了一层白噪声，
# 步态在那种信号下根本学不出来。这里让它几乎保持恒定。
_DELAY = {"delay_min_lag": 0, "delay_max_lag": 2, "delay_hold_prob": 0.98}


def _imu_noise(std: float, bias_std: float) -> NoiseModelWithAdditiveBiasCfg:
  """逐拍高斯噪声，再叠一个每个 episode 固定的偏置。"""
  return NoiseModelWithAdditiveBiasCfg(
    noise_cfg=Gnoise(std=std),
    bias_noise_cfg=Gnoise(std=bias_std),
  )


def make_observations() -> dict[str, ObservationGroupCfg]:
  # 噪声 std 取的是 velocity 任务那组均匀分布边界的方差等效值（std = 半宽 / sqrt(3)）；
  # 直接把半宽当高斯 std 用会让观测噪声比预期大约 1.7 倍。
  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=_imu_noise(std=0.115, bias_std=0.03),
      **_DELAY,
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=_imu_noise(std=0.029, bias_std=0.01),
      **_DELAY,
    ),
    "command_twist": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},
    ),
    "command_height": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "height"},
    ),
    "phase": ObservationTermCfg(
      func=mdp.shifted_phase,
      params={"period": 0.6, "command_name": "twist"},
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={"biased": True, "asset_cfg": joints(*LOWER_BODY_JOINT_EXPR)},
      noise=Gnoise(std=0.006),
      **_DELAY,
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      params={"asset_cfg": joints(*LOWER_BODY_JOINT_EXPR)},
      noise=Gnoise(std=0.87),
      **_DELAY,
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  # 特权观测。不会离开训练，所以想放什么都行。深拷贝一份，让 critic 的实体配置和 actor 的
  # 是两个独立对象。
  critic_terms = copy.deepcopy(actor_terms)
  for term in critic_terms.values():
    term.delay_max_lag = 0
    term.delay_hold_prob = 0.0
  critic_terms.update(
    {
      "base_lin_vel": ObservationTermCfg(
        func=mdp.builtin_sensor,
        params={"sensor_name": "robot/imu_lin_vel"},
      ),
      "base_height": ObservationTermCfg(
        func=mdp.base_height,
        params={"asset_cfg": SceneEntityCfg("robot", site_names=())},  # 按机器人设置。
      ),
      "arm_joint_pos": ObservationTermCfg(
        func=mdp.joint_pos_rel,
        params={"asset_cfg": joints(*ARM_JOINT_EXPR)},
      ),
      "arm_joint_vel": ObservationTermCfg(
        func=mdp.joint_vel_rel,
        params={"asset_cfg": joints(*ARM_JOINT_EXPR)},
      ),
      "foot_height": ObservationTermCfg(
        func=mdp.foot_height,
        params={"asset_cfg": SceneEntityCfg("robot", site_names=())},  # 按机器人设置。
      ),
      "foot_air_time": ObservationTermCfg(
        func=mdp.foot_air_time,
        params={"sensor_name": "feet_ground_contact"},
      ),
      "foot_contact": ObservationTermCfg(
        func=mdp.foot_contact,
        params={"sensor_name": "feet_ground_contact"},
      ),
      "foot_contact_forces": ObservationTermCfg(
        func=mdp.foot_contact_forces,
        params={"sensor_name": "feet_ground_contact"},
      ),
    }
  )

  return {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=1,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=1,
    ),
  }
