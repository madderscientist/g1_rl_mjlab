import pytest

from g1_lower_rl.tasks.footstep_tracking import rewards, rewards_cfg
from g1_lower_rl.tasks.footstep_tracking.theta_stage import make_theta_stage_env_cfg


@pytest.mark.parametrize("profile,rank", [("precision", 0), ("step-episode", 1)])
@pytest.mark.parametrize("play", [False, True])
@pytest.mark.parametrize("heading", [
  {"weight": -10.},
  {"weight": -10., "linear": True},
  {"weight": -10., "linear": True, "reference": "feet_bisector"},
])
def test_theta_rewards_do_not_constrain_pelvis_yaw(profile, rank, play, heading):
  stage = {
    "task_start_steps": [9622528, 9525760],
    "stages": [[0, .35]],
    "resampling_time_range_s": [8., 12.],
    "speed_range_rad_s": [.1, .5],
    "heading_reward_weight": -12.,
    "standing_torso_upright": {"weight": -10.},
    "standing_pelvis_upright_linear": {"weight": -10.},
    "standing_pelvis_feet_heading": heading,
    "feet_contact_yaw_rate": {"weight": -.005, "yaw_rate_scale_deg_s": 1.},
  }
  cfg = make_theta_stage_env_cfg(profile, stage, rank, play=play)
  assert "standing_pelvis_feet_heading" not in cfg.rewards
  assert cfg.rewards["standing_pelvis_upright_linear"] == rewards_cfg.make_standing_pelvis_upright_reward(-10.)
  assert cfg.rewards["standing_torso_upright"] == rewards_cfg.make_standing_torso_upright_reward(-10.)
  assert cfg.rewards["torso_feet_heading"] == rewards_cfg.make_torso_heading_reward(-12.)
  assert cfg.rewards["feet_contact_yaw_rate"] == rewards_cfg.make_feet_contact_yaw_rate_reward(-.005)


def test_pelvis_yaw_reward_implementations_and_factory_are_removed():
  for name in (
    "standing_pelvis_heading_error_l2",
    "standing_pelvis_heading_error_l1",
    "standing_pelvis_bisector_heading_error_l1",
  ):
    assert not hasattr(rewards, name)
  assert not hasattr(rewards_cfg, "make_standing_pelvis_heading_reward")
