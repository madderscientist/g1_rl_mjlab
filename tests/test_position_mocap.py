from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from g1_lower_rl.tasks.motion_tracking.training.position_collection import PositionPolicyCollector
from g1_lower_rl.tasks.motion_tracking.training.position_policy import (
  HISTORY,
  NATIVE_TERMS,
  OBS_DIM,
  REFERENCE_OFFSETS,
  PositionMlpActor,
  actor_architecture_metadata,
  actor_observations,
  make_actor,
  reference_tokens,
  residual_target,
  unflatten_native_history,
)
from g1_lower_rl.tasks.motion_tracking.training.position_runner import PositionTrainer, PositionTrainingConfig
from g1_lower_rl.tasks.motion_tracking.training.position_task import (
  RECOVERY_XY_TASK,
  PolicyStart,
  PositionReplay,
  PositionTask,
  configure_native_task,
  position_task_contract,
)
from g1_lower_rl.world_model.collection import TransitionBatch
from g1_lower_rl.world_model.model import pack_inputs


def test_history_term_major_conversion_preserves_time_order():
  history = torch.arange(2 * HISTORY * OBS_DIM).float().reshape(2, HISTORY, OBS_DIM)
  chunks, offset = [], 0
  for _, width in NATIVE_TERMS:
    chunks.append(history[..., offset : offset + width].flatten(1))
    offset += width
  torch.testing.assert_close(unflatten_native_history(torch.cat(chunks, -1)), history)
  inputs = actor_observations(history, torch.zeros(2, 798))
  assert inputs["rg_joint_pos"].shape == (2, 290)
  torch.testing.assert_close(inputs["rg_joint_pos"].reshape(2, 10, 29), history[..., :29])
  torch.testing.assert_close(inputs["rg_actions"].reshape(2, 10, 29), history[..., -29:])
  with pytest.raises(ValueError):
    unflatten_native_history(torch.zeros(2, 970))


def test_original_mlp_widths_preserve_current_inputs_scales_and_residual_output():
  actor = make_actor("cpu")
  assert isinstance(actor, PositionMlpActor)
  assert [layer.out_features for layer in actor.encoder if isinstance(layer, torch.nn.Linear)] == [2048, 1024, 512, 256, 32]
  assert [layer.out_features for layer in actor.decoder if isinstance(layer, torch.nn.Linear)] == [2048, 1024, 512, 256, 29]
  assert actor.encoder[0].in_features == 798 and actor.decoder[0].in_features == 962
  assert sum(p.numel() for p in actor.parameters()) == 9132861
  history = torch.randn(3, 10, 93)
  reference = torch.randn(3, 798)
  observations = actor_observations(history, reference)
  term_major = torch.cat([observations["rg_" + name] for name, _ in NATIVE_TERMS], -1)
  latent = torch.tanh(actor.encoder(reference * actor.reference_scale))
  expected = actor.decoder(torch.cat((latent, term_major * actor.history_scale), -1))
  actual = actor(observations, stochastic_output=False)
  torch.testing.assert_close(actual, expected)
  assert actual.shape == (3, 29)
  actual.square().sum().backward()
  assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in actor.parameters())
  assert actor_architecture_metadata()["interface"] == "unchanged_history10_reference21x38_residual29"
  with pytest.raises(ValueError, match="deterministic"):
    actor(observations, stochastic_output=True)


def test_reference_38_and_residual_target_no_gripper_and_encoder_bias():
  state = torch.zeros(2, 21, 71)
  state[..., 3] = 1
  state[..., 7:36] = torch.arange(29)
  state[..., 36:39] = 2
  state[..., 39:42] = 3
  tokens = reference_tokens(state)
  assert tokens.shape == (2, 21, 38)
  torch.testing.assert_close(tokens[..., 9:], state[..., 7:36])
  torch.testing.assert_close(tokens[..., 6:9], torch.tensor([0.0, 0.0, -1.0]).expand(2, 21, 3))
  action = torch.ones(2, 29, requires_grad=True)
  target = residual_target(
    action,
    torch.full((2, 29), 0.2),
    torch.full((29,), 0.5),
    torch.full((2, 29), 0.1),
    torch.tensor([-1.0, 1.0]).expand(29, 2),
    torch.zeros(29, dtype=torch.bool),
  )
  torch.testing.assert_close(target, torch.full((2, 29), 0.6))
  target.sum().backward()
  torch.testing.assert_close(action.grad, torch.full((2, 29), 0.5))


def _record(t: int, n: int, *, reset: bool = False):
  state = torch.zeros(1, n, 71)
  state[..., 3] = 1
  state[..., 2] = 0.8
  state[..., 7] = t
  following = state.clone()
  following[..., 7] += 1
  context = torch.ones(1, n, 48)
  target = torch.full((1, n, 29), float(t))
  batch = TransitionBatch(
    state,
    pack_inputs(state, target, context),
    following,
    torch.zeros(1, n, 2, 3),
    torch.ones(1, n, dtype=torch.bool),
    torch.full((1, n), int(reset), dtype=torch.long),
  )
  return batch, {
    "latest": torch.full((n, 93), float(t)),
    "frame": torch.full((n,), t, dtype=torch.long),
    "bias": torch.zeros(n, 29),
    "remaining": torch.full((n,), 100, dtype=torch.long),
    "clip_end": torch.full((n,), bool(t % 2), dtype=torch.bool),
  }


def test_position_replay_initial_state_and_history_align_and_splits_do_not_leak():
  replay = PositionReplay(16, 4, 1, "cpu")
  for t in range(24):
    replay.append_control(*_record(t, 4, reset=t >= 12))
  torch.manual_seed(4)
  for validation in (False, True):
    start = replay.sample_starts(64, validation=validation)
    torch.testing.assert_close(start.history[:, -1, 0], start.frame.float())
    torch.testing.assert_close(start.state[:, 7], start.frame.float())
    torch.testing.assert_close(start.clip_end, start.frame.remainder(2).bool())
    torch.testing.assert_close(start.history[:, 1:, 0] - start.history[:, :-1, 0], torch.ones(64, 9))
    assert (start.frame >= 21).all()
    sampled = replay.sample(64, 10, validation=validation)
    assert bool((sampled.env == 3).all()) if validation else bool((sampled.env < 3).all())


@pytest.mark.parametrize(
  ("frame", "latched", "remaining", "controls"),
  [
    (99, False, 100, 3),
    (100, False, 100, 2),
    (100, True, 100, 1),
    (90, False, 1, 1),
    (99, False, 2, 2),
  ],
)
def test_imagined_clip_end_matches_native_latch_and_time_limit(frame, latched, remaining, controls):
  trainer = PositionTrainer.__new__(PositionTrainer)
  trainer.cfg = PositionTrainingConfig("wm", "replica", "output", horizon=4)
  trainer.device = torch.device("cpu")
  reference = torch.zeros(1, 5, 71)
  reference[..., 3] = 1
  trainer.collector = SimpleNamespace(
    library=SimpleNamespace(last_frame_of=lambda frames: torch.full_like(frames, 100)),
    reference_trajectory=lambda frames, horizon: (reference, torch.zeros(1, 5, 798)),
  )
  trainer.model = None
  trainer.task = SimpleNamespace(critic_features=lambda model, state, context, ref, window: state[:, :1])
  trainer.critic = lambda feature: torch.full_like(feature, 5)
  trainer._step = lambda state, history, *args: (state, history, torch.ones(len(state)), torch.zeros(len(state)))
  start = PolicyStart(
    torch.zeros(1, 71),
    torch.zeros(1, 10, 93),
    torch.zeros(1, 48),
    torch.tensor([frame]),
    torch.zeros(1, 29),
    torch.tensor([remaining]),
    torch.tensor([latched]),
  )
  loss, critic_batch, _ = trainer.imagine(start, checkpointed=False)
  assert critic_batch[2].tolist() == [1.0] * controls + [0.0] * (4 - controls)
  expected = sum(0.99**step for step in range(controls)) + 0.99**controls * 5
  torch.testing.assert_close(-loss * 4, torch.tensor(expected))


def test_unlimited_training_config_and_invalid_counts():
  cfg = PositionTrainingConfig("wm.pt", "replica", "output")
  cfg.validate()
  assert cfg.iterations == 0
  for updates in (1, 3):
    replace(cfg, iterations=updates).validate()
  for changes in (
    {"iterations": -1},
    {"validation_envs": 1024},
    {"warmup_steps": 8},
    {"learning_rate": float("nan")},
    {"action_noise": -0.1},
    {"policy_target_kl": float("nan")},
    {"native_actor_kl": False},
    {"policy_microbatch": 0},
  ):
    with pytest.raises(ValueError):
      replace(cfg, **changes).validate()


def test_failure_margin_uses_pelvis_and_matches_native_thresholds():
  state = torch.zeros(3, 71)
  state[:, 3] = 1
  reference = state.clone()
  state[:, 2] = torch.tensor([0.0, 0.249, 0.251])
  margin = PositionTask.failure_margin(state, reference)
  assert (margin > 0).tolist() == [False, False, True]


def test_recovery_soft_termination_preserves_failed_state_survival_and_gradients():
  from g1_lower_rl.rl.supermimic import bptt_return

  state = torch.zeros(2, 71)
  state[:, 3] = 1
  reference = state.clone()
  state[:, 2] = torch.tensor([0.30, 0.60])
  state.requires_grad_()
  task = PositionTask.__new__(PositionTask)
  task.contract = position_task_contract(RECOVERY_XY_TASK)
  d = task.termination(state, reference)
  assert 0.0049 < float(d[0].detach()) <= float(d[1].detach()) <= 0.005
  returns, survival = bptt_return(torch.ones(2, 32), d[:, None].expand(2, 32), torch.zeros(2), 0.99)
  assert (survival[:, -1] > 0.85).all()
  returns.sum().backward()
  assert torch.isfinite(state.grad).all() and state.grad[0, 2] < 0


def test_recovery_reward_has_velocity_and_xy_position_with_unsaturated_gradients():
  from pathlib import Path

  import mujoco

  from g1_lower_rl.world_model.adapters.kinematics import StaticKinematicTemplate, TorchKinematics
  from g1_lower_rl.world_model.model import BODY_NAMES

  asset = Path(__file__).resolve().parents[1] / "g1_lower_rl/assets/g1_gloria/xmls/scene_g1_with_dual_gloria_m.xml"
  fk = TorchKinematics(StaticKinematicTemplate.from_mujoco(mujoco.MjModel.from_xml_path(str(asset))))
  reference = torch.zeros(1, 71, dtype=torch.float64)
  reference[:, 2:4] = torch.tensor([0.8, 1.0])
  actual = reference.clone()
  actual[:, 0], actual[:, 36] = 2.5, 0.3
  actual[:, 2] += 0.6
  actual.requires_grad_()
  limits = torch.tensor([-10.0, 10.0], dtype=torch.float64).expand(29, 2)
  new = PositionTask(fk, torch.zeros(29), limits, RECOVERY_XY_TASK)
  action = torch.zeros(1, 29, dtype=torch.float64)
  new_reward = new.reward(actual, reference, action, action)
  from g1_lower_rl.world_model.adapters.tasks import MotionGoal

  signals = new.adapter(actual, MotionGoal(reference)).rewards
  base_reward = 0.02 * (
    0.5 * (signals["anchor_position"] + signals["anchor_orientation"])
    + signals["body_position"]
    + signals["body_orientation"]
    + signals["body_linear_velocity"]
    + signals["body_angular_velocity"]
  )
  anchor = BODY_NAMES.index("torso_link")
  measured, target = fk(actual), fk(reference)
  velocity_error = (measured.body_linear_velocity[:, anchor] - target.body_linear_velocity[:, anchor]).square().sum(-1)
  delta = measured.body_position[:, anchor] - target.body_position[:, anchor]
  position_error = delta[:, :2].square().sum(-1)
  expected = 0.02 * (
    float(new.contract["anchor_velocity_weight"]) * torch.exp(-velocity_error / 0.4**2)
    + float(new.contract["anchor_position_coarse_weight"]) * torch.exp(-position_error / 1.2**2)
  )
  torch.testing.assert_close(new_reward - base_reward, expected)
  new_gradient = torch.autograd.grad(new_reward.sum(), actual)[0]
  assert new_gradient[0, 0] < -1e-6 and torch.isfinite(new_gradient).all()
  torch.testing.assert_close(
    new.reward(reference, reference, action, action),
    torch.tensor([float(new.contract["maximum_step_reward"])], dtype=torch.float64),
  )


def test_xy_position_is_independent_of_z_but_old_position_and_vertical_velocity_remain_tracked():
  from g1_lower_rl.tasks.motion_tracking.mdp.rewards import motion_anchor_pos_xy_error_exp
  from g1_lower_rl.world_model.adapters.kinematics import StaticKinematicTemplate, TorchKinematics
  from g1_lower_rl.world_model.adapters.tasks import MotionGoal

  fk = TorchKinematics(StaticKinematicTemplate.from_xml())
  limits = torch.tensor([-10.0, 10.0], dtype=torch.float64).expand(29, 2)
  task = PositionTask(fk, torch.zeros(29), limits, RECOVERY_XY_TASK)
  reference = torch.zeros(2, 71, dtype=torch.float64)
  reference[:, 2:4] = torch.tensor([0.8, 1.0])
  actual = reference.clone()
  actual[:, 0] = 0.4
  actual[1, 2] += 0.5
  actual[1, 38] = 0.3
  actual.requires_grad_()
  signals = task.adapter(actual, MotionGoal(reference)).rewards
  coarse = signals["anchor_position_coarse"]
  torch.testing.assert_close(coarse[0], coarse[1])
  gradient = torch.autograd.grad(coarse.sum(), actual)[0]
  assert (gradient[:, 0] < 0).all() and (gradient[:, 2] == 0).all()
  assert signals["anchor_position"][1] < signals["anchor_position"][0]
  torch.testing.assert_close(signals["anchor_linear_velocity"][1], torch.exp(actual.new_tensor(-(0.3**2) / 0.4**2)))
  assert signals["body_linear_velocity"][1] < signals["body_linear_velocity"][0]
  anchor = task.adapter.anchor
  command = SimpleNamespace(
    anchor_pos_w=fk(reference).body_position[:, anchor],
    robot_anchor_pos_w=fk(actual.detach()).body_position[:, anchor],
  )
  env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: command))
  torch.testing.assert_close(motion_anchor_pos_xy_error_exp(env, "motion"), coarse)
  assert task.contract["anchor_velocity_weight"] == 0.2 and task.contract["anchor_position_coarse_weight"] == 0.15
  assert task.contract["sampler_g_max"] == pytest.approx(10.7)


def test_native_recovery_uses_one_coin_for_both_failure_predicates_and_preserves_truncation():
  from dataclasses import dataclass

  from mjlab.envs import ManagerBasedRlEnvCfg
  from mjlab.managers.command_manager import CommandTermCfg
  from mjlab.managers.reward_manager import RewardTermCfg
  from mjlab.managers.termination_manager import TerminationTermCfg
  from mjlab.scene import SceneCfg

  @dataclass(kw_only=True)
  class CommandCfg(CommandTermCfg):
    gamma: float = 0.99
    g_max: float = 10.0

    def build(self, env):
      raise RuntimeError("Configuration-only test must not construct a command")

  def failed(env):
    return env.failed

  cfg = ManagerBasedRlEnvCfg(
    decimation=4,
    scene=SceneCfg(num_envs=50000),
    commands={"motion": CommandCfg(resampling_time_range=(0.0, 0.0))},
    rewards={"motion_global_root_pos": RewardTermCfg(func=failed, weight=0.5, params={"command_name": "motion", "std": 0.3})},
    terminations={
      "root_height": TerminationTermCfg(func=failed),
      "root_tilt": TerminationTermCfg(func=failed),
      "clip_end": TerminationTermCfg(func=failed, time_out=True),
    },
  )
  clip_end = cfg.terminations["clip_end"]
  original = cfg.rewards["motion_global_root_pos"]
  contract = configure_native_task(cfg, RECOVERY_XY_TASK)
  assert cfg.episode_length_s == 20 and cfg.terminations["clip_end"] is clip_end
  assert contract["sampler_g_max"] == pytest.approx(10.7)
  assert cfg.rewards["motion_global_root_pos"] is original
  assert cfg.rewards["motion_anchor_lin_vel"].weight == contract["anchor_velocity_weight"]
  assert cfg.rewards["motion_global_root_pos_coarse"].weight == contract["anchor_position_coarse_weight"]
  assert cfg.rewards["motion_global_root_pos_coarse"].params["std"] == 1.2
  env = SimpleNamespace(num_envs=50000, device="cpu", common_step_counter=1, failed=torch.ones(50000, dtype=torch.bool))
  torch.manual_seed(17)
  terms = [cfg.terminations[name] for name in ("root_height", "root_tilt")]
  first, second = (term.func(env, **term.params) for term in terms)
  assert torch.equal(first, second)
  assert 0.004 < float(first.float().mean()) < 0.006
  env.failed.zero_()
  assert not terms[0].func(env, **terms[0].params).any()


def test_online_wm_loss_preserves_pretraining_state_normalization(monkeypatch):
  from g1_lower_rl.tasks.motion_tracking.training import position_runner

  trainer = PositionTrainer.__new__(PositionTrainer)
  trainer.cfg = PositionTrainingConfig(
    "wm.pt",
    "replica",
    "output",
    wm_batches=1,
    wm_batch_size=4,
    wm_microbatch=2,
  )
  trainer.model = torch.nn.Linear(1, 1)
  trainer.model.register_buffer("state_scale", torch.ones(68))
  trainer.model.register_buffer("delta_scale", torch.full((70,), 100.0))
  trainer.model.register_buffer("force_scale", torch.ones(6))
  trainer.optimizers = {"world_model": torch.optim.Adam(trainer.model.parameters())}
  trainer.optimizer_updates = {"world_model": 0}
  trainer.replay = SimpleNamespace(sample=lambda size, horizon: (size, horizon))
  trainer.forward_model = trainer.model
  trainer.loss_cfg = object()
  trainer.progress = lambda *args, **kwargs: None
  scales = []

  def objective(forward, window, state_scale, force_scale, config):
    scales.append(state_scale)
    loss = forward(torch.ones(window[0], 1)).square().mean()
    return loss, {"state_loss": loss.detach()}

  monkeypatch.setattr(position_runner, "autoregressive_loss", objective)
  trainer.fit_model()
  assert len(scales) == 2 and all(scale is trainer.model.state_scale for scale in scales)
  assert trainer.optimizer_updates["world_model"] == 1


def test_nonfinite_native_action_stops_before_execution_and_preserves_diagnostic(tmp_path):
  class InvalidPolicy(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.register_buffer("result", torch.zeros(3, 29))
      self.result[1, 0] = float("nan")

    def forward(self, observations, stochastic_output=False):
      return self.result

  collector = PositionPolicyCollector.__new__(PositionPolicyCollector)
  collector.actor = InvalidPolicy()
  collector.action_failure = None
  collector.policy_action_sampling = False
  collector._observations = {"policy": torch.randn(3, 930)}
  collector._native_command = SimpleNamespace(frame=torch.arange(3), clip_end=torch.zeros(3, dtype=torch.bool))
  collector.env = SimpleNamespace(
    scene={"robot": SimpleNamespace(data=SimpleNamespace(encoder_bias=torch.zeros(3, 29)))},
    max_episode_length=100,
    episode_length_buf=torch.zeros(3, dtype=torch.long),
    step=lambda action: pytest.fail("A nonfinite action must never reach native execution"),
  )
  collector.reference_window = lambda frame: torch.zeros(len(frame), 798)
  collector.metadata = {"policy_collection": "test_only"}
  with pytest.raises(RuntimeError, match="No nonfinite"):
    collector.save_action_failure(tmp_path / "missing.pt")
  with pytest.raises(FloatingPointError, match="Nonfinite current-policy"):
    collector.collect_step()
  path = tmp_path / "diagnostic.pt"
  collector.save_action_failure(path)
  saved = torch.load(path, weights_only=True)
  assert saved["format"] == "native_policy_nonfinite_diagnostic_v1"
  assert saved["inputs"]["env_indices"].tolist() == [1]
  assert saved["inputs"]["history"].shape == (1, 10, 93)
  assert saved["inputs"]["reference"].shape == (1, 798)
  assert saved["inputs"]["history"].isfinite().all() and saved["inputs"]["reference"].isfinite().all()
  assert torch.isnan(saved["inputs"]["mean_action"][0, 0])
  assert torch.isnan(saved["actor_state_dict"]["result"][1, 0])


def test_full_h32_reference_batch_respects_bounded_frame_gathers():
  collector = PositionPolicyCollector.__new__(PositionPolicyCollector)
  collector.device = torch.device("cpu")
  count = 512 * 33
  collector.library = SimpleNamespace(
    frame_clip=torch.zeros(count, dtype=torch.long),
    clip_start=torch.tensor([0]),
    clip_last=torch.tensor([count - 1]),
  )
  gathered = []

  def reference_state(index):
    gathered.append(index.numel())
    assert index.numel() <= 2048 * 21 < 262144
    state = torch.zeros(*index.shape, 71)
    state[..., 3] = 1
    state[..., 7:36] = index[..., None].float()
    return state

  collector.reference_state = reference_state
  frames = torch.arange(count)
  windows = collector.reference_window(frames).reshape(count, 21, 38)
  expected = (frames[:, None] + torch.tensor(REFERENCE_OFFSETS)).clamp(0, count - 1)
  torch.testing.assert_close(windows[..., 9], expected.float())
  assert sum(gathered) == count * 21 and len(gathered) > 1


def test_dense_reference_trajectory_matches_naive_windows_at_both_clip_boundaries():
  collector = PositionPolicyCollector.__new__(PositionPolicyCollector)
  collector.device = torch.device("cpu")
  collector.library = SimpleNamespace(
    frame_clip=torch.arange(100) // 50,
    clip_start=torch.tensor([0, 50]),
    clip_last=torch.tensor([49, 99]),
  )

  def reference_state(index):
    state = torch.zeros(*index.shape, 71)
    state[..., 3] = 1
    state[..., 7:36] = index[..., None].float()
    state[..., 36:39] = index[..., None].float() / 10
    return state

  collector.reference_state = reference_state
  frames = torch.tensor([0, 20, 49, 50, 80, 99])
  last = collector.library.clip_last[collector.library.frame_clip[frames]]
  future = torch.minimum(frames[:, None] + torch.arange(33), last[:, None])
  expected_state = reference_state(future)
  expected_window = collector.reference_window(future.flatten()).reshape(6, 33, 798)
  actual_state, actual_window = collector.reference_trajectory(frames, 32)
  torch.testing.assert_close(actual_state, expected_state)
  torch.testing.assert_close(actual_window, expected_window)


def test_capacity_continuation_does_not_retroactively_expand_collection_clock():
  import math

  trainer = PositionTrainer.__new__(PositionTrainer)
  trainer.cfg = PositionTrainingConfig("wm.pt", "replica", "output", actor_batch_size=8192)
  trainer.replay = SimpleNamespace(train_envs=896)
  trainer.scheduled_actor_starts = 100 * 512
  trainer.scheduled_controls = math.ceil(trainer.scheduled_actor_starts / (896 * 0.4))
  trainer.scheduled_training_transitions = trainer.scheduled_controls * 896
  assert trainer.required_controls() == math.ceil((100 * 512 + 8192) / (896 * 0.4))


def test_native_parallelism_preserves_transition_budget_without_rescaling_history():
  import math

  trainer = PositionTrainer.__new__(PositionTrainer)
  trainer.cfg = PositionTrainingConfig(
    "wm.pt",
    "replica",
    "output",
    num_envs=2048,
    validation_envs=256,
    actor_batch_size=8192,
  )
  trainer.replay = SimpleNamespace(train_envs=1792)
  trainer.scheduled_actor_starts = 100 * 512
  trainer.scheduled_controls = math.ceil(trainer.scheduled_actor_starts / (896 * 0.4))
  trainer.scheduled_training_transitions = trainer.scheduled_controls * 896
  target = math.ceil((trainer.scheduled_actor_starts + 8192) / 0.4)
  count = trainer.required_controls() - trainer.scheduled_controls
  assert count == 12
  assert target <= trainer.scheduled_training_transitions + count * 1792 < target + 1792


def test_rollout_graphs_require_checkpointed_imagination():
  cfg = PositionTrainingConfig(
    world_model="wm.pt",
    replica_source="replica",
    output="output",
    cuda_graphs=True,
    matmul_precision="highest",
  )
  cfg.validate()
  with pytest.raises(ValueError, match="checkpointed imagination"):
    replace(cfg, checkpoint_imagination=False).validate()
  with pytest.raises(ValueError, match="highest matmul precision"):
    replace(cfg, matmul_precision="high").validate()


@pytest.mark.parametrize("fail", [False, True])
def test_wm_precision_is_scoped_and_restored_on_errors(monkeypatch, fail):
  state = {"precision": "highest"}
  monkeypatch.setattr(torch, "get_float32_matmul_precision", lambda: state["precision"])
  monkeypatch.setattr(torch, "set_float32_matmul_precision", lambda value: state.update(precision=value))
  trainer = PositionTrainer.__new__(PositionTrainer)
  trainer.cfg = SimpleNamespace(wm_matmul_precision="high")

  def fit():
    assert state["precision"] == "high"
    if fail:
      raise RuntimeError("expected fitting error")
    return {"loss": 1.0}

  trainer._fit_model = fit
  if fail:
    with pytest.raises(RuntimeError, match="expected fitting error"):
      trainer.fit_model()
  else:
    assert trainer.fit_model() == {"loss": 1.0}
  assert state["precision"] == "highest"


def test_rollout_graph_dispatch_reuses_shapes_with_fresh_inputs(monkeypatch):
  from g1_lower_rl.tasks.motion_tracking.training import position_runner

  captures = []

  class Graph:
    def __init__(self, forward, example, parameters, *, pool):
      assert tuple(example) == (
        "state",
        "history",
        "context",
        "bias",
        "reference",
        "window",
        "observation_noise",
        "action_noise",
      )
      self.forward = forward
      captures.append(len(example["state"]))

    def __call__(self, values):
      return self.forward(values)

  monkeypatch.setattr(position_runner, "CapturedRolloutStep", Graph)
  monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: (1, 2))
  trainer = PositionTrainer.__new__(PositionTrainer)
  trainer._actor_graphs = {}
  trainer.actor = torch.nn.Linear(1, 1)
  trainer.progress = lambda *args, **kwargs: None
  trainer._step = lambda *args: (args[0] + args[2], args[1], args[6], args[7])
  for size, value in ((3, 1.0), (3, 2.0), (1, 5.0)):
    args = tuple(torch.full((size, 1), value + i) for i in range(8))
    result = trainer._captured_step(args)
    torch.testing.assert_close(result[0], args[0] + args[2])
    assert result[1] is args[1] and result[2] is args[6] and result[3] is args[7]
  assert captures == [3, 1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA rollout graphs require CUDA")
def test_rollout_graph_chains_cotangents_and_observes_parameter_updates():
  from g1_lower_rl.tasks.motion_tracking.training.graphs import CapturedRolloutStep

  parameter = torch.nn.Parameter(torch.tensor(0.2, device="cuda"))
  initial = torch.ones(2, 3, device="cuda")
  previous = initial * 0.1

  def forward(data):
    following = (data["state"] * parameter + data["history"] * 0.1).tanh()
    history = data["history"] * 0.5 + data["state"] * 0.25 + parameter
    return following, history, following.square().sum(-1), history.sum(-1).sigmoid() * 0.01

  def rollout(step):
    state = initial.detach().clone().requires_grad_()
    history = previous.detach().clone().requires_grad_()
    current, latest, rewards, survival = state, history, 0, 1
    for _ in range(3):
      current, latest, reward, terminated = step({"state": current, "history": latest})
      rewards = rewards + survival * reward
      survival = survival * (1 - terminated)
    (rewards.sum() + survival.sum() + (current + latest).square().sum()).backward()
    return current.detach().clone(), latest.detach().clone(), state.grad.clone(), history.grad.clone()

  expected = rollout(forward)
  gradient = parameter.grad.clone()
  parameter.grad.zero_()
  graph = CapturedRolloutStep(
    forward,
    {"state": initial, "history": previous},
    [parameter],
    pool=torch.cuda.graph_pool_handle(),
  )
  for index in (1, 2):
    actual = rollout(graph)
    for value, reference in zip(actual, expected, strict=True):
      torch.testing.assert_close(value, reference)
    torch.testing.assert_close(parameter.grad, gradient * index)
  optimizer = torch.optim.SGD([parameter], lr=1e-3)
  optimizer.step()
  optimizer.zero_grad(set_to_none=True)
  updated = rollout(forward)
  updated_gradient = parameter.grad.clone()
  optimizer.zero_grad(set_to_none=True)
  actual = rollout(graph)
  for value, reference in zip(actual, updated, strict=True):
    torch.testing.assert_close(value, reference)
  torch.testing.assert_close(parameter.grad, updated_gradient)
