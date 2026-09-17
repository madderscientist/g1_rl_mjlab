import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from g1_lower_rl.footsteps import FootstepManager, FootstepManagerCfg, RandomCommandCfg, RandomCommandSource
from g1_lower_rl.footsteps.webui import PreviewSession
from g1_lower_rl.tasks.footstep_tracking.rewards import FilteredPelvisUpright, pelvis_height_reward
from g1_lower_rl.tasks.footstep_tracking.rewards_cfg import make_rewards


def make_fixture(frequencies):
  count = len(frequencies)
  data = SimpleNamespace(
    body_link_pos_w=torch.zeros(count, 2, 3),
    body_link_quat_w=torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(count, 2, 1),
    gravity_vec_w=torch.tensor([0.0, 0.0, -1.0]).repeat(count, 1),
  )
  state = SimpleNamespace(frequency=torch.tensor(frequencies), ground_height=torch.zeros(count, 2))
  env = SimpleNamespace(
    num_envs=count, device="cpu", step_dt=0.02, common_step_counter=0,
    scene={"robot": SimpleNamespace(body_names=["torso_link", "pelvis"], data=data),
           "feet_ground_contact": SimpleNamespace(data=SimpleNamespace(found=torch.ones(count, 2)))},
    termination_manager=SimpleNamespace(terminated=torch.zeros(count, dtype=torch.bool)),
    command_manager=SimpleNamespace(get_term=lambda name: SimpleNamespace(reward_state=state)),
  )
  asset = SimpleNamespace(name="robot", body_ids=[1])
  cfg = make_rewards()["pelvis_upright_filtered"]
  cfg.params["asset_cfg"] = asset
  return env, data, state, asset, FilteredPelvisUpright(cfg, env)


def set_pitch(data, radians):
  angles = torch.as_tensor(radians, dtype=data.body_link_quat_w.dtype)
  data.body_link_quat_w[:, 1, 0] = torch.cos(angles / 2)
  data.body_link_quat_w[:, 1, 2] = torch.sin(angles / 2)


def test_height_monotonic_ground_relative_bounded_and_contact_gated():
  env, data, state, asset, _ = make_fixture([1.0] * 6)
  state.ground_height[:] = 0.3
  data.body_link_pos_w[:, 1, 2] = torch.tensor([0.4, 0.6, 0.78, 1.0, 0.78, 0.78]) + 0.3
  env.scene["feet_ground_contact"].data.found[4] = 0
  env.termination_manager.terminated[5] = True
  actual = pelvis_height_reward(env, asset)
  torch.testing.assert_close(actual, torch.tensor([0.4 / 0.78, 0.6 / 0.78, 1.0, 1.0, 0.0, 0.0]))
  assert actual[0] < actual[1] < actual[2]


def test_filter_tracks_frequency_changes_and_does_not_freeze_when_standing():
  env, data, state, asset, term = make_fixture([0.8, 1.8, 0.0])
  term(env, asset)
  set_pitch(data, [0.3] * 3)
  env.common_step_counter += 1
  actual = term(env, asset)
  cutoffs = torch.tensor([0.2, 0.45, 0.2])
  alpha = -torch.expm1(-2 * math.pi * cutoffs * env.step_dt)
  expected_x = alpha * math.sin(0.3)
  torch.testing.assert_close(actual, expected_x.square())
  previous = term.filtered_gravity.clone()
  torch.testing.assert_close(term(env, asset), actual, rtol=0, atol=0)
  state.frequency[:] = torch.tensor([1.8, 0.8, 0.0])
  env.common_step_counter += 1
  term(env, asset)
  new_alpha = -torch.expm1(-2 * math.pi * torch.tensor([0.45, 0.2, 0.2]) * env.step_dt)
  torch.testing.assert_close(term.filtered_gravity[:, 0], previous[:, 0] + new_alpha * (math.sin(0.3) - previous[:, 0]))


def test_reset_uses_actual_pelvis_and_clears_only_selected_env():
  env, data, state, asset, term = make_fixture([1.0, 1.0])
  set_pitch(data, [0.4, -0.3])
  torch.testing.assert_close(term(env, asset), torch.tensor([math.sin(0.4)**2, math.sin(0.3)**2]))
  old_second = term.filtered_gravity[1].clone()
  term.reset(torch.tensor([0]))
  set_pitch(data, [0.1, 0.8])
  term(env, asset)
  torch.testing.assert_close(term.filtered_gravity[0, 0], torch.tensor(math.sin(0.1)))
  torch.testing.assert_close(term.filtered_gravity[1], old_second, rtol=0, atol=0)


def test_gait_frequency_is_attenuated_but_persistent_tilt_remains():
  env, data, state, asset, term = make_fixture([1.0])
  readings = []
  for step in range(1000):
    env.common_step_counter = step
    set_pitch(data, 0.1 * math.sin(2 * math.pi * step * env.step_dt))
    term(env, asset)
    if step >= 500:
      readings.append(float(term.filtered_gravity[0, 0]))
  assert max(readings) - min(readings) < 0.055
  for step in range(1000, 1300):
    env.common_step_counter = step
    set_pitch(data, 0.3)
    result = term(env, asset)
  assert result.item() == pytest.approx(math.sin(0.3)**2, abs=1e-4)


def test_new_defaults_and_preserved_swing_settings():
  terms = make_rewards()
  assert len(terms) == 18
  assert terms["lower_body_copper_proxy"].weight == -2.0
  assert terms["pelvis_height"].weight == 2.0
  assert terms["pelvis_upright_filtered"].weight == -1.0
  assert terms["torso_upright"].weight == -0.5
  assert terms["swing_clearance"].params["clearance"] == 0.08
  assert RandomCommandCfg().stop_probability == 0.3
  assert RandomCommandCfg().hold_time_s == (2.0, 5.0)
  assert PreviewSession({}).source.cfg.stop_probability == 0.3


def test_stop_sampling_probability_and_automatic_restart_cycle():
  source = RandomCommandSource(seed=42)
  source.reset()
  stops = 0
  for _ in range(5000):
    request = source.advance(source.command_at - source.elapsed + 1e-8, mode="walking", frequency=1.0, random_frequency=False)
    stops += not request.walking
    if not request.walking:
      source.set_request(replace(request, walking=True))
  assert stops / 5000 == pytest.approx(0.3, abs=0.02)

  source = RandomCommandSource(RandomCommandCfg(stop_probability=1.0, command_interval_s=(0.1, 0.1)), seed=42)
  manager = FootstepManager(FootstepManagerCfg(require_contact_confirmation=False), seed=42)
  manager.reset(np.array([[0.0, 0.12, 0.0], [0.0, -0.12, 0.0]]), source.reset())
  modes = [manager.mode]
  standing_started = None
  completed_stands = 0
  for _ in range(1500):
    update = manager.advance()
    manager.apply_request(source.advance(manager.cfg.control_dt, mode=update.command.mode, frequency=update.command.frequency))
    if manager.mode != modes[-1]:
      modes.append(manager.mode)
      if manager.mode == "standing":
        standing_started = manager.elapsed
        assert manager.frequency == 0.0
      elif manager.mode == "starting":
        assert standing_started is not None
        assert 2.0 <= manager.elapsed - standing_started <= 5.02
        completed_stands += 1
  assert completed_stands >= 2
  assert modes[:4] == ["walking", "stopping", "standing", "starting"]


@torch.inference_mode()
def test_real_reward_manager_filter_and_local_reset():
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.utils.lab_api.math import quat_apply_inverse

  from g1_lower_rl.tasks.footstep_tracking.env_cfg import footstep_env_cfg

  cfg = footstep_env_cfg(play=True)
  cfg.scene.num_envs = 2
  cfg.terminations = {"footstep_fault": cfg.terminations["footstep_fault"]}
  cfg.seed = 42
  cfg.rewards = make_rewards()
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env = ManagerBasedRlEnv(cfg, device=device)
  try:
    env.reset()
    term_cfg = env.reward_manager.get_term_cfg("pelvis_upright_filtered")
    term = term_cfg.func
    asset_cfg = term_cfg.params["asset_cfg"]
    filtered_index = env.reward_manager.active_terms.index("pelvis_upright_filtered")
    height_index = env.reward_manager.active_terms.index("pelvis_height")
    swing_indices = [env.reward_manager.active_terms.index(name)
                     for name in ("footstep_swing_position", "footstep_swing_yaw")]
    precision_indices = [env.reward_manager.active_terms.index(name)
                         for name in ("footstep_landing", "footstep_support")]
    landing_term = env.reward_manager.get_term_cfg("footstep_landing").func
    assert len(env.reward_manager.active_terms) == 18
    for _ in range(25):
      _, reward, _, _, _ = env.step(torch.zeros((2, 15), device=device))
      torch.testing.assert_close(env.reward_manager._step_reward[:, filtered_index],
                                 -term.filtered_gravity[:, :2].square().sum(-1))
      assert torch.isfinite(reward).all()
      assert (env.reward_manager._step_reward[:, height_index] >= 0).all()
      assert (env.reward_manager._step_reward[:, height_index] <= 2.0).all()
      assert (env.reward_manager._step_reward[:, swing_indices] >= 0).all()
      assert (env.reward_manager._step_reward[:, swing_indices] <= 5.0).all()
      assert (env.reward_manager._step_reward[:, precision_indices] <= 0).all()
      torch.testing.assert_close(reward, env.reward_manager._step_reward.sum(-1) * env.step_dt)
    previous = term.filtered_gravity[1].clone()
    previous_landing = landing_term.landing_cost[1].clone()
    env.reset(env_ids=torch.tensor([0], device=device))
    assert term.initialized.tolist() == [False, True]
    torch.testing.assert_close(term.filtered_gravity[1], previous, rtol=0, atol=0)
    assert not landing_term.landed[0].any()
    torch.testing.assert_close(landing_term.landing_cost[0], torch.zeros(2, device=device))
    torch.testing.assert_close(landing_term.landing_cost[1], previous_landing, rtol=0, atol=0)
    env.step(torch.zeros((2, 15), device=device))
    data = env.scene["robot"].data
    gravity = quat_apply_inverse(data.body_link_quat_w[:, asset_cfg.body_ids].squeeze(1), data.gravity_vec_w)
    torch.testing.assert_close(term.filtered_gravity[0], gravity[0])
    assert term.initialized.all()
  finally:
    env.close()