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


def test_footstep_actor_twin_mirror_equivariance_and_rollout_parity(tmp_path):
  import math
  import torch
  from tensordict import TensorDict
  from rsl_rl.storage import RolloutStorage
  from g1_lower_rl.assets import LOWER_BODY_JOINTS, WHOLE_BODY_JOINTS
  from g1_lower_rl.footstep_deploy import FootstepPolicy
  from g1_lower_rl.rl.footstep_model import FootstepActor, export_footstep_policy
  from g1_lower_rl.rl.mirror import _pair_joints

  obs_td = TensorDict({"actor": torch.zeros(4, 79)}, batch_size=[4])
  dist_cfg = {"class_name": "GaussianDistribution", "init_std": 0.25, "std_type": "scalar"}
  actor = FootstepActor(obs_td, {"actor": ["actor"]}, "actor", 15, distribution_cfg=dist_cfg, twin_mirror=True)
  actor_m = FootstepActor(obs_td, {"actor": ["actor"]}, "actor", 15, distribution_cfg=dist_cfg, twin_mirror=True)
  actor_m.load_state_dict(actor.state_dict(), strict=True)

  q_perm, q_sign = _pair_joints(list(WHOLE_BODY_JOINTS))
  act_perm, act_sign = _pair_joints(list(LOWER_BODY_JOINTS))
  raw_perm = torch.tensor(q_perm + [29 + i for i in q_perm] + list(range(58, 72)) + [75, 76, 77, 72, 73, 74, 78], dtype=torch.long)
  raw_sign = torch.tensor(q_sign + q_sign + [-1., 1., -1., 1., -1., 1., -1., 1., -1., 1., -1., 1., 1., 1., 1., -1., -1., 1., -1., -1., -1.], dtype=torch.float32)
  act_perm_t = torch.tensor(act_perm, dtype=torch.long)
  act_sign_t = torch.tensor(act_sign, dtype=torch.float32)

  actor.reset()
  actor_m.reset()
  for step in range(8):
    raw = torch.randn(4, 79) * 0.2
    raw[:, 70] = torch.rand(4) * 2 * math.pi
    raw[:, 71] = torch.rand(4) * 1.5
    raw_m = raw[:, raw_perm] * raw_sign
    raw_m[:, 70] = torch.remainder(raw[:, 70] + math.pi, 2 * math.pi)
    a1 = actor(TensorDict({"actor": raw}, batch_size=[4]), stochastic_output=False)
    a2 = actor_m(TensorDict({"actor": raw_m}, batch_size=[4]), stochastic_output=False)
    torch.testing.assert_close(a1, a2[:, act_perm_t] * act_sign_t, rtol=1e-5, atol=1e-5)
    if step == 3:
      dones = torch.tensor([1, 0, 1, 0])
      actor.reset(dones)
      actor_m.reset(dones)

  storage = RolloutStorage("rl", 4, 6, obs_td, [15], "cpu")
  actor.reset()
  for step in range(6):
    raw = torch.randn(4, 79) * 0.2
    raw[:, 70] = torch.rand(4) * 2 * math.pi
    td = TensorDict({"actor": raw}, batch_size=[4])
    transition = RolloutStorage.Transition()
    if actor.get_hidden_state() is None:
      actor.rnn.hidden_state = torch.zeros(1, 4, 128)
    transition.hidden_states = (actor.get_hidden_state(), None)
    transition.actions = actor(td, stochastic_output=True).detach()
    transition.values = torch.zeros(4, 1)
    transition.actions_log_prob = actor.get_output_log_prob(transition.actions).detach()
    transition.distribution_params = tuple(p.detach() for p in actor.output_distribution_params)
    transition.observations = td
    transition.rewards = torch.ones(4, 1)
    dones = torch.tensor([[1], [0], [0], [1]]) if step == 2 else torch.zeros(4, 1, dtype=torch.long)
    transition.dones = dones
    storage.add_transition(transition)
    actor.reset(dones.squeeze(-1))

  for batch in storage.recurrent_mini_batch_generator(1, 1):
    actor(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[0], stochastic_output=True)
    torch.testing.assert_close(actor.output_mean, batch.old_distribution_params[0], rtol=1e-5, atol=1e-6)

  onnx_path = tmp_path / "twin.onnx"
  export_footstep_policy(actor.eval(), onnx_path)
  policy = FootstepPolicy(onnx_path)
  assert policy.hidden_state.shape == (1, 1, 128)

