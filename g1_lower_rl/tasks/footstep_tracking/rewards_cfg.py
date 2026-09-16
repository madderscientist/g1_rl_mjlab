"""Initial footstep reward rates; weights need locomotion validation before deployment."""

from __future__ import annotations

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from g1_lower_rl.assets import LOWER_BODY_JOINTS
from g1_lower_rl.footstep_phase import FootstepPhaseCfg, resolve_phase_cfg
from g1_lower_rl.tasks.footstep_tracking import rewards
from g1_lower_rl.tasks.lower_body import mdp
from g1_lower_rl.tasks.lower_body.cfg.terminations import make_terminations as lower_body_terminations


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
        "landing_window": 0.25,
        "clearance": 0.08,
      },
    )
    for name, component, weight in (
      ("footstep_landing", "landing", 4.0),
      ("footstep_support", "support", 1.0),
      ("footstep_approach", "approach", -0.5),
      ("contact_schedule", "schedule", 2.0),
      ("swing_clearance", "clearance", -0.5),
      ("foot_slip", "slip", -2.0),
    )
  }
  terms.update(
    {
      "soft_landing": RewardTermCfg(
        func=mdp.soft_landing,
        weight=-2e-3,
        params={"sensor_name": sensor_name},
      ),
      "lower_body_copper_proxy": RewardTermCfg(
        func=rewards.LowerBodyTorqueCost,
        weight=-0.5,
        params={
          "asset_cfg": SceneEntityCfg("robot", actuator_names=list(LOWER_BODY_JOINTS), preserve_order=True),
          "reference_torque": 100.0,
          "copper_weights": copper_weights,
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
        func=mdp.body_orientation_l2,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",))},
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
  """Keep lower_body fall thresholds; low-height termination is not height control."""
  return lower_body_terminations()
