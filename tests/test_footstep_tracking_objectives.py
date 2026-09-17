import math
from types import SimpleNamespace

import pytest
import torch

from g1_lower_rl.tasks.footstep_tracking.reward_math import footprint_accuracy_cost, swing_tracking_score
from g1_lower_rl.tasks.footstep_tracking.rewards import FootstepReward, FootstepRewardState
from g1_lower_rl.tasks.footstep_tracking.rewards_cfg import make_rewards


def test_accuracy_cost_is_linear_near_zero_and_far_from_target():
  distance = torch.tensor([0.0, 0.01, 0.02, 0.08, 0.8], requires_grad=True)
  cost = footprint_accuracy_cost(distance, torch.zeros_like(distance), 0.08, 0.2)
  torch.testing.assert_close(cost, torch.tensor([0.0, 0.0625, 0.125, 0.5, 5.0]))
  cost.sum().backward()
  torch.testing.assert_close(distance.grad, torch.full_like(distance, 6.25))


def test_accuracy_cost_uses_linear_shortest_yaw_error():
  yaw = torch.tensor([0.0, 0.02, -0.04, 2 * math.pi - 0.04, math.pi])
  cost = footprint_accuracy_cost(torch.zeros_like(yaw), yaw, 0.08, 0.2)
  torch.testing.assert_close(cost, torch.tensor([0.0, 0.05, 0.1, 0.1, math.pi / 0.4]))


def test_swing_score_matches_paper_kernel_and_ignores_stance_foot():
  error = torch.tensor([[0.0, 9.0], [0.1, 0.0], [9.0, -0.1], [0.0, 0.0]])
  stance = torch.tensor([[False, True], [False, True], [True, False], [True, True]])
  score = 5 * swing_tracking_score(error, stance)
  torch.testing.assert_close(score, torch.tensor([5.0, 5 * math.exp(-1), 5 * math.exp(-1), 0.0]))


@pytest.mark.parametrize("scale", [0.0, -1.0, float("inf"), float("nan")])
def test_tracking_scales_must_be_positive_and_finite(scale):
  error = torch.zeros(1, 2)
  with pytest.raises(ValueError):
    footprint_accuracy_cost(error, error, scale, 0.2)
  with pytest.raises(ValueError):
    footprint_accuracy_cost(error, error, 0.08, scale)
  with pytest.raises(ValueError):
    swing_tracking_score(error, error.bool(), scale)


def make_reward_env():
  state = FootstepRewardState(
    phase=torch.zeros(2), frequency=torch.ones(2), targets_w=torch.zeros(2, 2, 3),
    target_ids=torch.tensor([[2, 1], [2, 1]]), ground_height=torch.zeros(2, 2),
  )
  robot = SimpleNamespace(
    site_names=["left_foot", "right_foot"],
    data=SimpleNamespace(site_pos_w=torch.zeros(2, 2, 3),
                         site_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]).expand(2, 2, -1).clone()),
  )
  sensor = SimpleNamespace(data=SimpleNamespace(found=torch.ones(2, 2)))
  command = SimpleNamespace(reward_state=state, command=torch.full((2, 14), 99.0))
  env = SimpleNamespace(num_envs=2, device="cpu", scene={"robot": robot, "feet_ground_contact": sensor},
                        command_manager=SimpleNamespace(get_term=lambda name: command))
  return env, state, robot.data, sensor.data


def make_term(name, env):
  config = make_rewards()[name]
  config.params["asset_cfg"] = SimpleNamespace(name="robot", site_ids=[0, 1])
  return FootstepReward(config, env), config


@pytest.mark.parametrize("name", ["footstep_swing_position", "footstep_swing_yaw"])
def test_swing_reward_has_no_ramp_or_actual_contact_gate_and_stops_in_hold(name):
  env, state, data, sensor = make_reward_env()
  reward, config = make_term(name, env)
  data.site_pos_w[:, 0, 0] = 0.1
  state.targets_w[:, 0, 2] = 0.1
  for phase in (-0.39 * math.pi, 0.0, 0.39 * math.pi):
    state.phase.fill_(phase)
    for contact in (0.0, 1.0):
      sensor.found[:, 0] = contact
      torch.testing.assert_close(config.weight * reward(env, **config.params),
                                 torch.full((2,), 5 * math.exp(-1)))
  state.frequency.zero_()
  torch.testing.assert_close(reward(env, **config.params), torch.zeros(2))
  state.frequency.fill_(1.0)
  state.phase.fill_(0.5 * math.pi)
  torch.testing.assert_close(reward(env, **config.params), torch.zeros(2))
  state.phase.fill_(math.pi)
  torch.testing.assert_close(config.weight * reward(env, **config.params), torch.full((2,), 5.0))


def test_swing_yaw_crosses_pi_boundary_using_shortest_angle():
  env, state, data, _ = make_reward_env()
  reward, config = make_term("footstep_swing_yaw", env)
  actual_yaw = math.pi - 0.05
  data.site_quat_w[:, 0] = torch.tensor([math.cos(actual_yaw / 2), 0.0, 0.0, math.sin(actual_yaw / 2)])
  state.targets_w[:, 0, 2] = -math.pi + 0.05
  torch.testing.assert_close(config.weight * reward(env, **config.params), torch.full((2,), 5 * math.exp(-1)))


def test_support_cost_uses_execution_target_without_contact_and_in_hold():
  env, state, data, sensor = make_reward_env()
  reward, config = make_term("footstep_support", env)
  data.site_pos_w[:, 0, 0] = 9.0
  data.site_pos_w[:, 1, 0] = 0.08
  state.targets_w[:, 1, 2] = 0.2
  torch.testing.assert_close(reward(env, **config.params), torch.ones(2))
  sensor.found.zero_()
  torch.testing.assert_close(reward(env, **config.params), torch.ones(2))
  state.frequency.zero_()
  state.targets_w[:, 0, 0] = 9.0
  torch.testing.assert_close(reward(env, **config.params), torch.full((2,), 0.5))


@pytest.mark.parametrize("foot, swing_phase, stance_phase", [(0, 0.0, 0.8 * math.pi), (1, math.pi, 1.8 * math.pi)])
def test_landing_latches_error_and_cannot_be_erased_by_sliding_or_relifting(foot, swing_phase, stance_phase):
  env, state, data, sensor = make_reward_env()
  reward, config = make_term("footstep_landing", env)
  state.phase.fill_(swing_phase)
  sensor.found[:, foot] = 0.0
  reward(env, **config.params)
  data.site_pos_w[:, foot, 0] = 0.08
  state.targets_w[:, foot, 2] = 0.2
  state.phase.fill_(0.5 * math.pi if foot == 0 else 1.5 * math.pi)
  sensor.found[:, foot] = 1.0
  reward(env, **config.params)
  state.phase.fill_(stance_phase)
  torch.testing.assert_close(config.weight * reward(env, **config.params), torch.full((2,), -4.0))
  data.site_pos_w[:, foot, 0] = 0.0
  state.targets_w[:, foot, 2] = 0.0
  sensor.found[:, foot] = 0.0
  torch.testing.assert_close(config.weight * reward(env, **config.params), torch.full((2,), -4.0))
  sensor.found[:, foot] = 1.0
  torch.testing.assert_close(config.weight * reward(env, **config.params), torch.full((2,), -4.0))
  reward.reset(torch.tensor([0]))
  assert not reward.landed[0].any()
  assert reward.landed[1, foot]
  torch.testing.assert_close(reward.landing_cost[1, foot], torch.tensor(1.0))


@pytest.mark.parametrize("touch_phase", [None, 0.1 * math.pi, 0.9 * math.pi])
def test_missing_early_and_late_landings_do_not_escape_cost(touch_phase):
  env, state, data, sensor = make_reward_env()
  reward, config = make_term("footstep_landing", env)
  sensor.found[:, 0] = 0.0
  reward(env, **config.params)
  data.site_pos_w[:, 0, 0] = 0.08
  if touch_phase is not None:
    state.phase.fill_(touch_phase)
    sensor.found[:, 0] = 1.0
    reward(env, **config.params)
  state.phase.fill_(0.95 * math.pi)
  torch.testing.assert_close(config.weight * reward(env, **config.params), torch.full((2,), -6.0))


def test_landing_target_change_and_standing_clear_history():
  env, state, data, sensor = make_reward_env()
  reward, config = make_term("footstep_landing", env)
  sensor.found[:, 0] = 0.0
  reward(env, **config.params)
  state.phase.fill_(0.5 * math.pi)
  sensor.found[:, 0] = 1.0
  reward(env, **config.params)
  assert reward.landed[:, 0].all()
  state.target_ids[0, 0] += 2
  state.phase.fill_(0.8 * math.pi)
  torch.testing.assert_close(reward(env, **config.params), torch.tensor([1.0, 0.0]))
  state.frequency.zero_()
  data.site_pos_w[:, :, 0] = 0.08
  sensor.found.zero_()
  torch.testing.assert_close(config.weight * reward(env, **config.params), torch.full((2,), -2.0))
  assert not reward.landed.any()
  state.frequency.fill_(1.0)
  torch.testing.assert_close(config.weight * reward(env, **config.params), torch.full((2,), -6.0))


def test_default_configuration_uses_only_new_objectives():
  config = make_rewards()
  assert len(config) == 18
  assert "footstep_distance" not in config and "footstep_approach" not in config
  assert config["footstep_landing"].weight == -4.0
  assert config["footstep_support"].weight == -1.0
  assert {name: term.weight for name, term in config.items() if term.weight > 0} == {
    "footstep_swing_position": 5.0, "footstep_swing_yaw": 5.0, "contact_schedule": 2.0, "pelvis_height": 2.0,
  }