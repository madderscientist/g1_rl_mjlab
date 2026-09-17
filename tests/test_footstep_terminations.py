import math
from types import SimpleNamespace

import pytest
import torch

from g1_lower_rl.footstep_phase import FootstepPhaseCfg
from g1_lower_rl.tasks.footstep_tracking.env_cfg import footstep_env_cfg
from g1_lower_rl.tasks.footstep_tracking.rewards import FootstepRewardState
from g1_lower_rl.tasks.footstep_tracking.rewards_cfg import make_rewards
from g1_lower_rl.tasks.footstep_tracking.terminations import footstep_distance_exceeded


def make_env(phase, frequency, positions):
  count = len(phase)
  state = FootstepRewardState(
    phase=torch.tensor(phase, dtype=torch.float64), frequency=torch.tensor(frequency, dtype=torch.float64),
    targets_w=torch.zeros(count, 2, 3, dtype=torch.float64), target_ids=torch.zeros(count, 2, dtype=torch.long),
    ground_height=torch.zeros(count, 2, dtype=torch.float64),
  )
  term = SimpleNamespace(
    reward_state=state, cfg=SimpleNamespace(manager=SimpleNamespace(phase=FootstepPhaseCfg())),
    robot=SimpleNamespace(data=SimpleNamespace(site_pos_w=torch.tensor(positions, dtype=torch.float64))),
    site_ids=[1, 0], managers=[SimpleNamespace(targets=torch.full((2, 3), 100.0)) for _ in phase],
  )
  return SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: term))


def test_strict_xy_threshold_and_per_environment_result():
  env = make_env(
    [0.0] * 4, [1.0] * 4,
    [[[0.999, 0.0, 10.0], [0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
     [[1.001, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.8, 0.8, 0.0], [0.0, 0.0, 0.0]]],
  )
  assert footstep_distance_exceeded(env).tolist() == [False, False, True, True]


def test_swing_ignored_but_stance_and_standing_checked_without_contact_gate():
  positions = [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]] * 5
  env = make_env([0.0, math.pi, math.pi / 2, 0.0, 2 * math.pi], [1.0, 1.0, 1.0, 0.0, 1.0], positions)
  assert footstep_distance_exceeded(env).tolist() == [False, True, True, True, False]


def test_uses_current_reward_snapshot_not_post_advance_future_targets():
  env = make_env([0.0], [1.0], [[[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]]])
  state = env.command_manager.get_term("footsteps").reward_state
  state.targets_w[0, 1, :2] = torch.tensor([3.0, 4.0])
  assert not footstep_distance_exceeded(env).item()
  state.targets_w[0, 1, 0] = 1.99
  assert footstep_distance_exceeded(env).item()


@pytest.mark.parametrize("threshold", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_threshold_rejected(threshold):
  with pytest.raises(ValueError):
    footstep_distance_exceeded(None, max_distance=threshold)


@pytest.mark.parametrize("play", [False, True])
def test_task_defaults_keep_clock_first_and_do_not_change_clearance(play):
  cfg = footstep_env_cfg(play=play)
  assert list(cfg.terminations)[:2] == ["footstep_fault", "footstep_distance"]
  assert cfg.terminations["footstep_distance"].params["max_distance"] == 1.0
  assert not cfg.terminations["footstep_distance"].time_out
  terms = make_rewards()
  assert terms["lower_body_copper_proxy"].weight == -2.0
  assert terms["swing_clearance"].params["clearance"] == 0.08
  assert terms["swing_clearance"].weight == -0.5


@torch.inference_mode()
def test_real_offtrack_termination_penalty_and_isolated_auto_reset():
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper

  cfg = footstep_env_cfg(play=True)
  cfg.scene.num_envs = 2
  cfg.seed = 42
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  wrapped = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device=device), clip_actions=None)
  env = wrapped.unwrapped
  try:
    command = env.command_manager.get_term("footsteps")
    original_finish = command.finish_step
    finish_calls = []

    def inject_current_target_error():
      failed = original_finish()
      finish_calls.append(env.common_step_counter)
      if len(finish_calls) == 1:
        command.reward_state.phase.zero_()
        command.reward_state.targets_w[..., :2] = command.robot.data.site_pos_w[:, command.site_ids, :2]
        command.reward_state.targets_w[1, 1, 0] += 1.01
      return failed

    command.finish_step = inject_current_target_error
    actions = torch.zeros((2, env.action_manager.total_action_dim), device=device)
    observations, _, dones, extras = wrapped.step(actions)
    assert dones.tolist() == [0, 1]
    assert env.termination_manager.get_term("footstep_distance").tolist() == [False, True]
    assert not extras["time_outs"].any()
    fall_index = env.reward_manager.active_terms.index("fall")
    torch.testing.assert_close(env.reward_manager._step_reward[:, fall_index] * env.step_dt,
                               torch.tensor([0.0, -10.0], device=device))
    assert env.episode_length_buf.tolist() == [1, 0]
    assert not command.pending_reset.any()
    assert torch.isfinite(observations["actor"]).all()
    wrapped.step(actions)
    assert not env.termination_manager.get_term("footstep_distance").any()
    assert len(finish_calls) == 2 and finish_calls[1] == finish_calls[0] + 1
  finally:
    wrapped.close()