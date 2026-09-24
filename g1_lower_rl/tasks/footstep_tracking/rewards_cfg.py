"""脚步奖励率初值，部署前仍需通过行走训练验证权重"""

from __future__ import annotations

import math

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from g1_lower_rl.assets import LOWER_BODY_JOINTS
from g1_lower_rl.footstep_phase import FootstepPhaseCfg, resolve_phase_cfg
from g1_lower_rl.tasks.footstep_tracking import rewards
from g1_lower_rl.tasks.footstep_tracking.terminations import footstep_distance_exceeded
from g1_lower_rl.tasks.lower_body import mdp
from g1_lower_rl.tasks.lower_body.cfg.terminations import make_terminations as lower_body_terminations
from g1_lower_rl.tasks.motion_tracking.mdp.terminations import with_probabilistic_termination


PROB_TERMINATION = 1.0 / (50.0 * 4.0)


def make_rewards(
  command_name: str = "footsteps",
  sensor_name: str = "feet_ground_contact",
  stance_fraction: float | None = None,
  position_std: float = 0.08,
  yaw_std: float = 0.2,
  copper_weights: dict[str, float] | None = None,
  *,
  phase_cfg: FootstepPhaseCfg | None = None,
) -> dict[str, RewardTermCfg]:
  """使用密集摆动奖励、落脚事件代价和停脚姿态软约束"""
  phase_cfg = resolve_phase_cfg(stance_fraction, phase_cfg)
  terms = {
    name: RewardTermCfg(
      func=rewards.FootstepReward,
      weight=weight,
      params={
        "command_name": command_name,
        "sensor_name": sensor_name,
        "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot"), preserve_order=True),
        "component": component,
        "phase_cfg": phase_cfg,
        "position_std": position_std,
        "yaw_std": yaw_std,
        "swing_position_std": 0.2,
        "swing_yaw_std": 0.1,
        "clearance": 0.11,
        "air_time_scale": 0.4,
      },
    )
    for name, component, weight in (
      ("footstep_landing", "landing", -1.0),
      ("footstep_swing_position", "swing_position", 5.0),
      ("footstep_swing_yaw", "swing_yaw", 5.0),
      ("contact_schedule", "schedule", 1.0),
      ("foot_air_time", "air_time", 3.2),
      ("swing_clearance", "clearance", -0.5),
      ("swing_contact", "swing_contact", -0.5),
      ("foot_slip", "slip", -2.0),
      ("feet_slip_still", "slip_still", -4.0),
      ("stand_still_feet", "stationary", -0.5),
    )
  }
  terms.update(
    {
      "stand_still_linear_velocity": RewardTermCfg(
        func=rewards.standing_linear_velocity_reward,
        weight=2.0,
        params={
          "asset_cfg": SceneEntityCfg("robot"),
          "command_name": command_name,
          "std": math.sqrt(0.2),
          "z_penalty": 1.5,
        },
      ),
      "stand_still_angular_velocity": RewardTermCfg(
        func=rewards.standing_angular_velocity_reward,
        weight=0.5,
        params={
          "asset_cfg": SceneEntityCfg("robot"),
          "command_name": command_name,
          "std": 0.7,
          "xy_penalty": 0.05,
        },
      ),
      "feet_flatness": RewardTermCfg(
        func=rewards.feet_flatness_cost,
        weight=-0.2,
        params={
          "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot"), preserve_order=True),
          "command_name": command_name,
          "sensor_name": sensor_name,
          "phase_cfg": phase_cfg,
          "angle_scale": math.radians(15),
        },
      ),
      "feet_hold_position": RewardTermCfg(
        func=rewards.feet_hold_position_cost,
        weight=-0.2,
        params={
          "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot"), preserve_order=True),
          "command_name": command_name,
          "deadband": 0.03,
          "distance_scale": 0.1,
        },
      ),
      "stance_knee_bend": RewardTermCfg(
        func=rewards.stance_knee_bend_cost,
        weight=-0.2,
        params={
          "asset_cfg": SceneEntityCfg("robot", joint_names=("left_knee_joint", "right_knee_joint"), preserve_order=True),
          "command_name": command_name,
          "phase_cfg": phase_cfg,
          "standing_limit": math.radians(30),
          "walking_limit": math.radians(45),
          "angle_scale": math.radians(45),
        },
      ),
      "soft_landing": RewardTermCfg(
        func=rewards.soft_landing,
        weight=-2e-3,
        params={"sensor_name": sensor_name, "command_name": command_name},
      ),
      "lower_body_copper_proxy": RewardTermCfg(
        func=rewards.LowerBodyTorqueCost,
        weight=-0.25,
        params={
          "asset_cfg": SceneEntityCfg("robot", actuator_names=list(LOWER_BODY_JOINTS), preserve_order=True),
          "reference_torque": 100.0,
          "copper_weights": copper_weights,
          "command_name": command_name,
          "standing_scale": 2.0,
          "limit_scale": 10.0,
          "limit_margin": math.radians(0.1),
        },
      ),
      "waist_yaw_zero": RewardTermCfg(
        func=rewards.joint_zero_l2,
        weight=-0.4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=("waist_yaw_joint",))},
      ),
      "waist_roll_pitch_edges": RewardTermCfg(
        func=rewards.JointEdgeCost,
        weight=-2.0,
        params={
          "asset_cfg": SceneEntityCfg("robot", joint_names=("waist_roll_joint", "waist_pitch_joint")),
          "margin_fraction": 0.15,
        },
      ),
      "torso_upright": RewardTermCfg(
        func=rewards.body_tilt_angle_l2,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",))},
      ),
      "body_ang_vel": RewardTermCfg(
        func=mdp.body_angular_velocity_penalty,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",))},
      ),
      "head_height": RewardTermCfg(
        func=rewards.head_height_reward,
        weight=0.4,
        params={
          "asset_cfg": SceneEntityCfg("robot", geom_names=("head_collision",)),
          "command_name": command_name,
          "sensor_name": sensor_name,
          "height_cap": 1.254,
        },
      ),
      "head_height_low": RewardTermCfg(
        func=rewards.head_height_shortfall,
        weight=-1.0,
        params={
          "asset_cfg": SceneEntityCfg("robot", geom_names=("head_collision",)),
          "command_name": command_name,
          "minimum_height": 1.15,
          "height_scale": 0.2,
        },
      ),
      "pelvis_upright_filtered": RewardTermCfg(
        func=rewards.FilteredPelvisUpright,
        weight=-1.0,
        params={
          "asset_cfg": SceneEntityCfg("robot", body_names=("pelvis",)),
          "command_name": command_name,
          "cutoff_ratio": 0.25,
          "standing_cutoff_hz": 0.2,
        },
      ),
      "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.02),
      "controlled_joint_acc": RewardTermCfg(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LOWER_BODY_JOINTS)},
      ),
      "self_collisions": RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-2.0,
        params={"sensor_name": "self_collision", "force_threshold": 10.0},
      ),
      "fall": RewardTermCfg(func=rewards.fall_cost, weight=-10.0),
    }
  )
  return terms


def make_terminations():
  """仅倾倒和塌低共享概率终止，离轨及超时保持硬终止"""
  terms = {
    "footstep_distance": TerminationTermCfg(
      func=footstep_distance_exceeded,
      params={"command_name": "footsteps", "max_distance": 1.0},
    ),
    **lower_body_terminations(),
  }
  for name in ("fell_over", "collapsed"):
    term = terms[name]
    terms[name] = TerminationTermCfg(
      func=with_probabilistic_termination,
      params={"base_func": term.func, "p_term": PROB_TERMINATION, **term.params},
    )
  return terms
