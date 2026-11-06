from __future__ import annotations

import json
import shutil
import subprocess
import sys
import uuid
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from g1_lower_rl.world_model.checkpoint import atomic_checkpoint, primitive_metadata, read_checkpoint
from g1_lower_rl.world_model.model import (
  CONTROL_DT,
  DynamicsPrediction,
  PositionWorldModel,
  PositionWorldModelCfg,
  integrate_state,
  pack_inputs,
  state_features,
)
from g1_lower_rl.world_model.training import (
  LossConfig,
  ReplayWindow,
  RingReplay,
  TrainingConfig,
  load_world_model,
  run_training,
)
from g1_lower_rl.world_model.training.derivatives import SECANT_CONVENTION, secant_loss
from g1_lower_rl.world_model.training.normalization import fit_normalization
from g1_lower_rl.world_model.training.objectives import autoregressive_loss, finalize_metrics, physical_metric_sums
from g1_lower_rl.world_model.training.provenance import snapshot_sources
from g1_lower_rl.world_model.training.synthetic import SyntheticBatch, SyntheticCollector
from g1_lower_rl.world_model.training.trainer import (
  DistributedContext,
  _capacity_collector_metadata,
  _capacity_fork_origin,
  _fixed_validation_windows,
  _stable_collector_metadata,
  refinement_eligible,
)


@pytest.fixture(scope="module", autouse=True)
def small_cpu_thread_pool():
  original = torch.get_num_threads()
  torch.set_num_threads(1)
  yield
  torch.set_num_threads(original)


@pytest.fixture
def run_dir():
  # Do not use pytest's default /tmp-backed tmp_path in this workspace.
  path = Path("logs") / "wm_tests" / uuid.uuid4().hex
  path.mkdir(parents=True)
  yield path
  shutil.rmtree(path)


def make_batch(steps=8, num_envs=4, offset=0):
  states = torch.zeros(steps, num_envs, 71)
  states[..., 3] = 1
  states[..., 0] = states[..., 2] = torch.arange(offset, offset + steps)[:, None]
  commands = torch.arange(num_envs, dtype=states.dtype)[None, :, None].expand(steps, num_envs, 29).clone()
  contexts = torch.ones(steps, num_envs, 48)
  contexts[..., 45:] = 0
  inputs = pack_inputs(states, commands, contexts)
  targets = states.clone()
  targets[..., 0] += 1
  return SyntheticBatch(
    state=states,
    inputs=inputs,
    next_state=targets,
    foot_force=torch.zeros(steps, num_envs, 2, 3),
    valid=torch.ones(steps, num_envs, dtype=torch.bool),
    segment=torch.zeros(steps, num_envs, dtype=torch.int64),
  )


def tiny_config(run_dir, **kwargs):
  return TrainingConfig(
    output_dir=str(run_dir),
    synthetic=True,
    device="cpu",
    bf16=False,
    num_envs=4,
    validation_envs=1,
    replay_steps=16,
    collect_steps=4,
    normalization_samples=8,
    batch_size=4,
    horizon=2,
    updates_per_collection=1,
    eval_batches=1,
    eval_interval=2,
    save_interval=2,
    eval_horizons_seconds=(0.02, 0.04),
    log_interval=1,
    max_updates=2,
    model=PositionWorldModelCfg(16, 8, 1),
    **kwargs,
  )


def test_replay_isolates_envs_and_excludes_invalid_and_segment_boundaries():
  replay = RingReplay(12, 4, 1)
  batch = make_batch(10)
  batch.valid[3, 0] = False
  # Endpoints agree across 0 -> 1 -> 0: endpoint-only checks would miss this.
  batch.segment[4, 1] = 1
  replay.append(batch)
  generator = torch.Generator().manual_seed(15)
  train = replay.sample(1000, 3, generator=generator)
  validation = replay.sample(100, 3, validation=True, generator=generator)
  assert (train.env < 3).all() and (validation.env == 3).all()
  assert torch.equal(train.sequence[:, 1:] - train.sequence[:, :-1], torch.ones(1000, 2, dtype=torch.int64))
  for row, env in zip(train.sequence.tolist(), train.env.tolist(), strict=True):
    assert bool(batch.valid[row, env].all())
    assert batch.segment[row, env].unique().numel() == 1
  torch.testing.assert_close(train.inputs[..., 0].long(), train.sequence)
  torch.testing.assert_close(train.inputs[..., 68].long(), train.env[:, None].expand(-1, 3))


def test_ring_wrap_and_oversized_append_cannot_cross_overwritten_history():
  replay = RingReplay(5, 4, 1)
  replay.append(make_batch(4))
  old = replay.candidates(3)
  replay.append(make_batch(4, offset=4))
  assert replay.total_steps == 8 and replay.size == 5
  assert replay.candidates(3) is not old
  sampled = replay.sample(100, 3)
  assert int(sampled.sequence.min()) == 3 and int(sampled.sequence.max()) == 7
  torch.testing.assert_close(sampled.inputs[..., 0].long(), sampled.sequence)
  replay.append(make_batch(11, offset=8))
  sampled = replay.sample(100, 5)
  assert int(sampled.sequence.min()) == 14 and int(sampled.sequence.max()) == 18
  torch.testing.assert_close(sampled.inputs[..., 0].long(), sampled.sequence)


def test_replay_checks_valid_inputs_but_allows_invalid_reset_payload():
  replay = RingReplay(8, 4, 1)
  batch = make_batch()
  batch.inputs[3, 0, 5] = float("nan")
  with pytest.raises(ValueError, match="finite"):
    replay.append(batch)
  batch.valid[3, 0] = False
  replay.append(batch)
  assert torch.isfinite(replay.sample(100, 2).inputs).all()
  with pytest.raises(ValueError, match="horizon"):
    replay.candidates(9)
  batch.valid.zero_()
  replay.append(batch)
  with pytest.raises(RuntimeError, match="No contiguous"):
    replay.sample(1, 1)


def test_normalization_uses_training_only_and_freezes_explicit_state():
  model = PositionWorldModel(PositionWorldModelCfg(16, 8, 1))
  replay = RingReplay(8, 4, 1)
  batch = make_batch()
  batch.state[:, 3, :3] = 1e6
  batch.inputs[:, 3, :68] = state_features(batch.state[:, 3])
  batch.next_state[:, 3, :3] = -1e6
  batch.foot_force[:, 3] = 1e6
  batch.valid[0, 0] = False
  replay.append(batch)
  metadata = fit_normalization(model, replay)
  assert metadata["samples"] == 23 and metadata["frozen"]
  assert bool(model.normalization_ready)
  assert float(model.input_mean[0]) < 8
  torch.testing.assert_close(model.force_scale, torch.ones(6))
  frozen = model.input_mean.clone()
  replay.append(make_batch(offset=100))
  torch.testing.assert_close(model.input_mean, frozen)
  other = PositionWorldModel(model.cfg)
  other.load_state_dict(model.state_dict())
  torch.testing.assert_close(other.input_mean, frozen)
  assert bool(other.normalization_ready)


def test_refinement_starts_new_optimizer_and_preserves_normalization(run_dir):
  initial = run_training(tiny_config(run_dir / "initial"))
  cfg = tiny_config(run_dir / "refined", initialize=str(initial), fixed_validation=True, one_step_loss_weight=1)
  result = read_checkpoint(run_training(cfg))
  before = read_checkpoint(initial)
  assert result["update"] == 2
  assert result["normalization"]["source"] == "pretrained_initialization"
  assert result["fixed_validation_sha256"][0]
  for key in ("input_mean", "input_scale", "delta_scale", "state_scale", "force_scale"):
    torch.testing.assert_close(result["model"][key], before["model"][key], rtol=0, atol=0)
  assert all(int(state["step"]) == 2 for state in result["optimizer"]["state"].values())
  resumed = read_checkpoint(
    run_training(replace(cfg, initialize=None, resume=str(run_dir / "refined/final.pt"), max_updates=3))
  )
  assert resumed["update"] == 3
  assert resumed["fixed_validation_sha256"] == result["fixed_validation_sha256"]


def test_fixed_validation_survives_replay_overwrite_and_checks_checksum(run_dir):
  replay = RingReplay(16, 4, 1)
  replay.append(make_batch(16))
  cfg = tiny_config(run_dir, fixed_validation=True)
  context = DistributedContext(torch.device("cpu"), 0, 1)
  generator = torch.Generator().manual_seed(7)
  fixed, checksum = _fixed_validation_windows(run_dir, cfg, replay, context, generator, 2)
  original = fixed[0].inputs.clone()
  replay.append(make_batch(32, offset=100))
  torch.testing.assert_close(fixed[0].inputs, original, rtol=0, atol=0)
  restored, repeated = _fixed_validation_windows(run_dir, cfg, replay, context, generator, 2, checksum)
  assert repeated == checksum
  torch.testing.assert_close(restored[0].inputs, original, rtol=0, atol=0)
  with pytest.raises(ValueError, match="checksum"):
    _fixed_validation_windows(run_dir, cfg, replay, context, generator, 2, "wrong")
  with pytest.raises(FileExistsError):
    _fixed_validation_windows(run_dir, cfg, replay, context, generator, 2)


def test_causal_rollout_discount_uses_fixed_control_dt_and_zero_force_loss():
  replay = RingReplay(3, 4, 1)
  batch = make_batch(3)
  batch.state[..., :3] = 0
  batch.inputs[..., :68] = state_features(batch.state)
  batch.next_state[..., :3] = 0
  batch.inputs[..., 68] = torch.tensor([10, 20, 30])[:, None]
  replay.append(batch)
  window = replay.sample(1, 3)
  observed = []

  def forward(initial, target, context):
    observed.append((initial.detach().clone(), target.detach().clone()))
    state = initial.clone()
    state[:, 0] += 1
    return DynamicsPrediction(state, initial.new_full((initial.shape[0], 2, 3), 2))

  cfg = LossConfig(gamma=0.5)
  loss, parts = autoregressive_loss(forward, window, torch.ones(70), torch.ones(6), cfg)
  expected_state = (1 + 4 * 0.5 + 9 * 0.5**2) / (1 + 0.5 + 0.5**2) / 70
  assert float(parts["state_loss"]) == pytest.approx(expected_state)
  assert float(parts["force_loss"]) == pytest.approx(4)
  assert float(parts["zero_force_loss"]) == pytest.approx(4)
  assert float(loss) == pytest.approx(expected_state + 8)
  assert [float(value[0][0, 0]) for value in observed] == [0, 1, 2]
  assert [float(value[1][0, 0]) for value in observed] == [10, 20, 30]
  metrics = finalize_metrics(physical_metric_sums(forward, window))
  assert metrics["one_step/root_position/rmse"] == pytest.approx(1 / 3**0.5)
  assert metrics["rollout_3/root_position/rmse"] == pytest.approx(3 / 3**0.5)
  assert metrics["rollout_3/seconds/mean"] == pytest.approx(0.06)
  assert metrics["one_step/force/rmse"] == pytest.approx(2)


def test_tiny_synthetic_autoregressive_fit_reduces_loss():
  torch.manual_seed(4)
  replay = RingReplay(16, 4, 1)
  collector = SyntheticCollector(num_envs=4, device=torch.device("cpu"), seed=3, physics_dt=0.005)
  replay.append(collector.collect(4))
  model = PositionWorldModel(PositionWorldModelCfg(32, 16, 1))
  fit_normalization(model, replay)
  window = replay.sample(16, 2, generator=torch.Generator().manual_seed(2))
  forward = lambda state, target, context: model(state, target, context, validate=False)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
  initial = float(autoregressive_loss(forward, window, model.state_scale, model.force_scale, LossConfig())[0].detach())
  for _ in range(50):
    optimizer.zero_grad()
    loss = autoregressive_loss(forward, window, model.state_scale, model.force_scale, LossConfig())[0]
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer.step()
  final_loss = autoregressive_loss(forward, window, model.state_scale, model.force_scale, LossConfig())[0]
  assert float(final_loss.detach()) < initial * 0.4


def test_contact_precision_recall_f1_and_swing_force_are_not_impact_dominated():
  batch = make_batch(1)
  batch.next_state.copy_(batch.state)
  batch.foot_force[0, ..., 2] = torch.tensor([0, 0, 20, 20, 0, 0, 20, 20]).reshape(4, 2)
  predicted_force = torch.zeros(4, 2, 3)
  predicted_force[..., 2] = torch.tensor([0, 20, 20, 0, -4, 9, 10, 30]).reshape(4, 2)
  window = ReplayWindow(
    batch.state.transpose(0, 1),
    batch.inputs.transpose(0, 1),
    batch.next_state.transpose(0, 1),
    batch.foot_force.transpose(0, 1),
    torch.arange(4),
    torch.zeros(4, 1, dtype=torch.int64),
  )
  metrics = finalize_metrics(
    physical_metric_sums(
      lambda state, target, context: DynamicsPrediction(state, predicted_force),
      window,
      contact_threshold=10,
    )
  )
  assert metrics["one_step/contact/precision"] == pytest.approx(0.75)
  assert metrics["one_step/contact/recall"] == pytest.approx(0.75)
  assert metrics["one_step/contact/f1"] == pytest.approx(0.75)
  assert metrics["one_step/contact/false_positive_rate"] == pytest.approx(0.25)
  assert metrics["one_step/contact/support"] == 4
  assert metrics["one_step/swing_force/rmse"] == pytest.approx((497 / 12) ** 0.5)
  assert metrics["one_step/negative_fz/mae"] == pytest.approx(0.5)
  empty_contact = finalize_metrics(
    {
      "one_step/contact/true_positive": torch.tensor(0),
      "one_step/contact/false_positive": torch.tensor(0),
      "one_step/contact/false_negative": torch.tensor(0),
      "one_step/contact/true_negative": torch.tensor(4),
    }
  )
  assert empty_contact["one_step/contact/precision"] is None
  assert empty_contact["one_step/contact/recall"] is None
  assert empty_contact["one_step/contact/f1"] is None


def test_constant_velocity_baseline_advances_position_body_rotation_and_joints_open_loop():
  batch = make_batch(3)
  state = batch.state[0].clone()
  state[:, 36:39] = torch.tensor([1.0, 2.0, 3.0])
  state[:, 39:42] = torch.tensor([0.2, -0.1, 0.4])
  state[:, 42:71] = 0.3
  for step in range(3):
    batch.state[step] = state
    batch.inputs[step, :, :68] = state_features(state)
    state = integrate_state(state, torch.zeros(4, 70))
    batch.next_state[step] = state
  window = ReplayWindow(
    batch.state.transpose(0, 1),
    batch.inputs.transpose(0, 1),
    batch.next_state.transpose(0, 1),
    batch.foot_force.transpose(0, 1),
    torch.arange(4),
    torch.arange(3).expand(4, -1),
  )
  metrics = finalize_metrics(
    physical_metric_sums(
      lambda state, target, context: DynamicsPrediction(state, torch.zeros(4, 2, 3)),
      window,
      delta_scale=torch.ones(70),
    )
  )
  assert metrics["rollout_3/constant_velocity_state_normalized/rmse"] < 1e-6
  assert metrics["rollout_3/state_normalized/rmse"] > 0.001
  assert metrics["rollout_3/constant_velocity/root_rotation/rmse"] < 1e-6
  assert metrics["rollout_3/constant_velocity/joint_position/rmse"] < 1e-6
  assert "rollout_3/state_skill_vs_constant_velocity" in metrics


def test_small_nonnegative_fz_penalty_has_finite_gradient_without_constraining_head():
  batch = make_batch(1)
  batch.next_state.copy_(batch.state)
  replay = RingReplay(1, 4, 1)
  replay.append(batch)
  window = replay.sample(4, 1)
  fz = torch.tensor([-2.0, -4.0], requires_grad=True)

  def forward(state, target, context):
    forces = torch.stack((torch.zeros_like(fz), torch.zeros_like(fz), fz), dim=-1)
    return DynamicsPrediction(state, forces[None].expand(state.shape[0], -1, -1))

  loss, parts = autoregressive_loss(
    forward,
    window,
    torch.ones(70),
    torch.ones(6),
    LossConfig(force_weight=0, zero_force_weight=0, nonnegative_fz_weight=0.01),
  )
  assert float(loss.detach()) == pytest.approx(0.1)
  assert float(parts["nonnegative_fz_loss"]) == pytest.approx(10)
  loss.backward()
  assert fz.grad is not None and torch.isfinite(fz.grad).all() and (fz.grad < 0).all()


def test_unchecked_model_captures_fullgraph_and_backpropagates():
  replay = RingReplay(8, 4, 1)
  replay.append(make_batch())
  model = PositionWorldModel(PositionWorldModelCfg(16, 8, 1))
  fit_normalization(model, replay)
  window = replay.sample(4, 2)
  compiled = torch.compile(model, backend="eager", fullgraph=True)
  args = (window.state[:, 0], window.inputs[:, 0, 68:97], window.inputs[:, 0, 97:145])
  eager = model(*args, validate=False)
  captured = compiled(*args, validate=False)
  torch.testing.assert_close(captured.state, eager.state)
  torch.testing.assert_close(captured.foot_force, eager.foot_force)
  loss = autoregressive_loss(
    lambda state, target, context: compiled(state, target, context, validate=False),
    window,
    model.state_scale,
    model.force_scale,
    LossConfig(),
  )[0]
  loss.backward()
  assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_synthetic_training_resume_optimizer_normalization_and_inference_export(run_dir):
  cfg = tiny_config(run_dir, gradient_accumulation=2)
  final = run_training(cfg)
  first = read_checkpoint(final)
  assert first["update"] == 2 and first["transitions"] > 0
  assert first["normalization"]["split"] == "training_environments_only"
  assert first["config"]["learning_rate"] == 2e-4
  assert len(first["source_provenance"]["sha256"]) == 64
  assert "g1_lower_rl/world_model/model.py" in first["source_provenance"]["files"]
  assert "g1_lower_rl/world_model/checkpoint.py" in first["source_provenance"]["files"]
  assert Path(first["source_provenance"]["snapshot_directory"], "manifest.json").exists()
  assert torch.load(final, weights_only=True)["update"] == 2
  assert first["best_metric"] == "rollout_0.04s/state_normalized/rmse"
  assert (run_dir / "best.pt").exists() and (run_dir / "latest.pt").exists()
  restored = load_world_model(final)
  assert not restored.training and bool(restored.normalization_ready)
  for value in first["optimizer"]["state"].values():
    assert int(value["step"]) == 2
  resumed = run_training(replace(cfg, resume=str(final), max_updates=4))
  second = read_checkpoint(resumed)
  assert second["update"] == 4 and second["transitions"] > first["transitions"]
  torch.testing.assert_close(second["model"]["input_mean"], first["model"]["input_mean"])
  torch.testing.assert_close(second["model"]["delta_scale"], first["model"]["delta_scale"])
  assert not torch.equal(second["model"]["embedding.weight"], first["model"]["embedding.weight"])
  for value in second["optimizer"]["state"].values():
    assert int(value["step"]) == 4
  status = json.loads((run_dir / "status.json").read_text())
  assert status["status"] == "completed" and status["reason"] == "max_updates"
  assert "one_step/force/rmse" in status["metrics"]
  assert "rollout_0.04s/joint_position/rmse" in status["metrics"]
  assert all("timestamp_utc" in json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines())
  assert not list(run_dir.glob("*.pending"))


@pytest.mark.parametrize("base_seed", [0, 100003])
def test_late_resume_bounds_native_seed_and_preserves_optimizer(run_dir, base_seed):
  import numpy as np

  cfg = tiny_config(run_dir)
  first = read_checkpoint(run_training(cfg))
  first["update"] = 5000
  checkpoint = run_dir / "late_resume.pt"
  atomic_checkpoint(checkpoint, first)
  observed_seeds = []

  def factory(config, context, seed):
    np.random.seed(seed)
    observed_seeds.append(seed)
    return SyntheticCollector(num_envs=config.num_envs, device=context.device, seed=seed, physics_dt=config.physics_dt)

  resumed = read_checkpoint(
    run_training(
      replace(cfg, seed=base_seed, resume=str(checkpoint), max_updates=5001),
      collector_factory=factory,
    )
  )
  assert observed_seeds == [(base_seed + 5001 * 1000003) % (2**32)]
  assert resumed["provenance"][0]["seed"] == observed_seeds[0]
  assert resumed["update"] == 5001
  for key in ("input_mean", "input_scale", "delta_scale", "state_scale", "force_scale"):
    torch.testing.assert_close(resumed["model"][key], first["model"][key])
  for state in resumed["optimizer"]["state"].values():
    assert int(state["step"]) == 3


def test_capacity_fork_preserves_adam_fixed_validation_and_strict_resume(run_dir):
  parent_cfg = tiny_config(run_dir / "parent", fixed_validation=True)
  parent_path = run_training(parent_cfg)
  parent = read_checkpoint(parent_path)
  child_root = run_dir / "capacity"
  child_root.mkdir()
  shutil.copy2(parent_path.parent / "fixed_validation_rank_0.pt", child_root / "fixed_validation_rank_0.pt")
  cfg = replace(
    parent_cfg,
    output_dir=str(child_root),
    fork_from=str(parent_path),
    num_envs=8,
    validation_envs=2,
    batch_size=8,
    eval_batch_size=4,
    max_updates=4,
  )
  child_path = run_training(cfg)
  child = read_checkpoint(child_path)
  assert child["update"] == 4
  assert child["fixed_validation_sha256"] == parent["fixed_validation_sha256"]
  assert child["continuation"]["parent_checkpoint"] == str(parent_path.resolve())
  assert child["continuation"]["kind"] == "explicit_capacity_fork_full_optimizer"
  assert child["continuation"]["configuration_changes"]["batch_size"] == {"parent": 4, "current": 8}
  assert all(int(state["step"]) == 4 for state in child["optimizer"]["state"].values())
  for key in ("input_mean", "input_scale", "delta_scale", "state_scale", "force_scale"):
    torch.testing.assert_close(child["model"][key], parent["model"][key], rtol=0, atol=0)
  resumed = read_checkpoint(run_training(replace(cfg, fork_from=None, resume=str(child_path), max_updates=5)))
  assert resumed["update"] == 5
  assert resumed["continuation"] == child["continuation"]
  assert all(int(state["step"]) == 5 for state in resumed["optimizer"]["state"].values())


@pytest.mark.parametrize(
  "changes,message",
  [
    ({"learning_rate": 1e-3}, "non-capacity training configuration"),
    ({"eval_batch_size": 8}, "original evaluation batch"),
    ({"hours": 12}, "extend the cumulative budget"),
    ({"num_envs": 8, "validation_envs": 1}, "held-out environment fraction"),
    ({"seed": 7}, "uint32 identity"),
  ],
)
def test_capacity_fork_rejects_non_capacity_changes(run_dir, changes, message):
  cfg = tiny_config(run_dir / "parent", fixed_validation=True)
  parent = run_training(cfg)
  fork = replace(cfg, output_dir=str(run_dir / "child"), fork_from=str(parent), **changes)
  with pytest.raises(ValueError, match=message):
    run_training(fork)


def test_capacity_fork_does_not_weaken_ordinary_source_resume(run_dir):
  cfg = tiny_config(run_dir, fixed_validation=True)
  final = run_training(cfg)
  saved = read_checkpoint(final)
  saved["source_provenance"]["sha256"] = "0" * 64
  changed = run_dir / "wrong_source.pt"
  atomic_checkpoint(changed, saved)
  with pytest.raises(ValueError, match="immutable checkpoint snapshot"):
    run_training(replace(cfg, resume=str(changed), max_updates=3))


def test_capacity_fork_rejects_changed_model_source_and_mutated_parent(run_dir):
  cfg = tiny_config(run_dir / "parent", fixed_validation=True)
  final = run_training(cfg)
  saved = read_checkpoint(final)
  fork = replace(cfg, output_dir=str(run_dir / "child"), fork_from=str(final))
  source = snapshot_sources(run_dir / "child", write=False)
  source["files"]["g1_lower_rl/world_model/model.py"] = "0" * 64
  with pytest.raises(ValueError, match="cannot change model"):
    _capacity_fork_origin(saved, fork, source)
  source = snapshot_sources(run_dir / "child", write=False)
  snapshot_file = Path(saved["source_provenance"]["snapshot_directory"]) / "g1_lower_rl/world_model/model.py"
  snapshot_file.write_text("changed parent snapshot\n")
  with pytest.raises(ValueError, match="Parent immutable source changed"):
    _capacity_fork_origin(saved, fork, source)


def test_capacity_metadata_changes_only_parallelism_and_known_gripper_projection():
  old = {
    "config": {"num_envs": 512, "seed": 0, "physics_dt": 0.005},
    "storage": {"njmax": 500, "naconmax": 32768},
    "corpus": {"data_sha256": "same", "joint_projection": "legacy31_closed_grippers_to_canonical29"},
  }
  new = {
    "config": {"num_envs": 4096, "seed": 2, "physics_dt": 0.005},
    "storage": {"njmax": 500, "naconmax": 262144},
    "corpus": {
      "data_sha256": "same",
      "joint_projection": "legacy31_ignore_gripper_state_to_canonical29",
      "gripper_state": "excluded_from_reference_policy_and_WM_coordinates_no_clip_filtering",
    },
  }
  assert _capacity_collector_metadata(old) == _capacity_collector_metadata(new)
  new["corpus"]["data_sha256"] = "different"
  assert _capacity_collector_metadata(old) != _capacity_collector_metadata(new)


def test_inference_loader_is_simulator_independent_and_metadata_is_primitive(run_dir):
  final = run_training(tiny_config(run_dir))
  code = """
import importlib.abc
import sys
class BlockSimulator(importlib.abc.MetaPathFinder):
  def find_spec(self, fullname, path=None, target=None):
    blocked = ('mjlab', 'mujoco', 'warp', 'rsl_rl', 'g1_lower_rl.world_model.collection')
    if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
      raise AssertionError('Simulator import attempted: ' + fullname)
sys.meta_path.insert(0, BlockSimulator())
from g1_lower_rl.world_model.training import load_world_model
model = load_world_model(sys.argv[1], device='cpu')
assert bool(model.normalization_ready)
assert not model.training
"""
  result = subprocess.run([sys.executable, "-c", code, str(final)], capture_output=True, text=True, timeout=30, check=False)
  assert result.returncode == 0, result.stdout + result.stderr
  assert primitive_metadata({"path": Path("relative"), "model": PositionWorldModelCfg(16, 8, 1)}) == {
    "path": "relative",
    "model": {"hidden_dim": 16, "interaction_dim": 8, "num_blocks": 1},
  }


def test_checkpoint_finite_guard_preserves_existing_file(run_dir):
  path = run_dir / "snapshot.pt"
  atomic_checkpoint(path, {"finite": torch.ones(1)})
  before = path.read_bytes()
  with pytest.raises(FloatingPointError, match="nonfinite"):
    atomic_checkpoint(path, {"bad": torch.tensor(float("nan"))})
  assert path.read_bytes() == before


def test_shared_initialization_is_not_an_optimizer_resume(run_dir):
  from g1_lower_rl.world_model import export_pretrained_checkpoint

  final = run_training(tiny_config(run_dir))
  initialization = export_pretrained_checkpoint(final, run_dir / "init.pt")
  with pytest.raises(ValueError, match="initialization, not an optimizer resume"):
    run_training(replace(tiny_config(run_dir / "other"), resume=str(initialization)))


def test_source_snapshots_are_content_addressed_and_reject_mutation(run_dir):
  first = snapshot_sources(run_dir, write=True)
  second = snapshot_sources(run_dir, write=True)
  assert first == second
  source = Path(first["snapshot_directory"]) / "g1_lower_rl/world_model/training/__init__.py"
  source.write_text("changed snapshot\n")
  with pytest.raises(RuntimeError, match="Immutable source snapshot was modified"):
    snapshot_sources(run_dir, write=True)


def test_manifest_provenance_is_saved_and_checked_on_resume(run_dir):
  corpus = run_dir / "corpus"
  corpus.mkdir()
  manifest = {"clips": [{"name": "bounded_clip.npz", "sha256": "test-content-hash"}], "version": 1}
  (corpus / "manifest.json").write_text(json.dumps(manifest))
  cfg = replace(tiny_config(run_dir), synthetic=False, checkpoint="test_teacher.pt", motion_dir=str(corpus),
                replica_source="test_native_source")

  def factory(config, context, seed):
    return SyntheticCollector(num_envs=config.num_envs, device=context.device, seed=seed, physics_dt=config.physics_dt)

  final = run_training(cfg, collector_factory=factory)
  stored = read_checkpoint(final)["provenance"][0]["collector"]["corpus_manifest"]
  assert stored["contents"] == manifest and len(stored["sha256"]) == 64
  manifest["version"] = 2
  (corpus / "manifest.json").write_text(json.dumps(manifest))
  with pytest.raises(ValueError, match="Collector provenance differs"):
    run_training(replace(cfg, resume=str(final), max_updates=4), collector_factory=factory)
  assert json.loads((run_dir / "status.json").read_text())["status"] == "failed"


def test_missing_bounded_corpus_fails_before_constructing_collector(run_dir):
  cfg = replace(tiny_config(run_dir), synthetic=False, checkpoint="test_teacher.pt", motion_dir=str(run_dir / "missing"),
                replica_source="test_native_source")

  def factory(*_):
    raise AssertionError("Missing corpus must be rejected before collection")

  with pytest.raises(FileNotFoundError, match="Bounded motion corpus is missing"):
    run_training(cfg, collector_factory=factory)
  assert json.loads((run_dir / "status.json").read_text())["status"] == "failed"


def test_resume_metadata_ignores_local_seed_device_and_teacher_path_but_not_physics():
  first = {
    "teacher_checkpoint": "/old/teacher.pt",
    "teacher_sha256": "same",
    "config": {"checkpoint": "/old/teacher.pt", "seed": 1, "device": "cuda:0", "physics_dt": 0.005},
  }
  second = {
    "teacher_checkpoint": "/new/teacher.pt",
    "teacher_sha256": "same",
    "config": {"checkpoint": "/new/teacher.pt", "seed": 17, "device": "cuda:1", "physics_dt": 0.005},
  }
  assert _stable_collector_metadata(first) == _stable_collector_metadata(second)
  second["config"]["physics_dt"] = 0.0025
  assert _stable_collector_metadata(first) != _stable_collector_metadata(second)


def test_derivative_selection_preserves_local_and_sensitivity_quality():
  baseline = {
    "derivative/0.005/cosine": 0.6,
    "derivative/0.02/cosine": 0.6,
    "one_step/state_normalized/rmse": 1.0,
    "rollout_0.02s/state_normalized/rmse": 2.0,
    "rollout_0.16s/state_normalized/rmse": 4.0,
  }
  assert refinement_eligible(baseline, baseline)
  for key in baseline:
    candidate = dict(baseline)
    candidate[key] += -0.01 if key.endswith("cosine") else 0.01
    assert not refinement_eligible(candidate, baseline)
  missing: dict[str, float | None] = {**baseline, "derivative/0.005/cosine": None}
  with pytest.raises(ValueError, match="finite"):
    refinement_eligible(missing, baseline)


def test_secant_supervision_optimizes_sensitivity_not_state_offset():
  sensitivity = torch.nn.Parameter(torch.tensor(0.2))
  offset = torch.nn.Parameter(torch.tensor(0.7))
  states = torch.zeros(3, 71)
  states[:, 3] = 1
  context = torch.ones(3, 48)
  context[:, 45:] = 0
  inputs = pack_inputs(states, torch.zeros(3, 29), context)
  directions = torch.ones(4, 3, 29) / 29**0.5
  labels = torch.cat((torch.zeros(4, 3, 6), directions * 0.4), -1)
  probe = {
    "state": states,
    "inputs": inputs,
    "directions": directions,
    "labels": {"0.02": labels},
    "secant_convention": SECANT_CONVENTION,
    "label_partition": "training",
    "contract_version": 3,
    "control_dt": CONTROL_DT,
  }

  def forward(initial, target, context):
    state = initial.clone()
    state[:, 42:] = state[:, 42:] + sensitivity * target + offset
    return DynamicsPrediction(state, initial.new_zeros(initial.shape[0], 2, 3))

  loss = secant_loss(forward, probe, torch.ones(70), 8, torch.Generator().manual_seed(7))
  loss.backward()
  assert loss > 0
  assert sensitivity.grad is not None and sensitivity.grad < 0
  assert offset.grad is not None and abs(float(offset.grad)) < 1e-6
  with pytest.raises(ValueError, match="independent training"):
    secant_loss(forward, {**probe, "label_partition": "validation"}, torch.ones(70), 8, torch.Generator())


def test_explicit_failure_status_includes_traceback_and_collector_cleanup(run_dir):
  instances = []

  class BrokenCollector:
    def __init__(self):
      self.metadata = {"source": "broken_test"}
      self.closed = False
      instances.append(self)

    def collect(self, num_control_steps=1):
      raise RuntimeError("intentional collection failure")

    def close(self):
      self.closed = True

  with pytest.raises(RuntimeError, match="intentional"):
    run_training(tiny_config(run_dir), collector_factory=lambda *_: BrokenCollector())
  status = json.loads((run_dir / "status.json").read_text())
  assert status["status"] == "failed" and "Traceback" in status["traceback"]
  assert "intentional collection failure" in status["traceback"]
  assert instances[0].closed


def test_failure_after_baseline_preserves_latest_and_best_checkpoints(run_dir):
  class FailAfterWarmup(SyntheticCollector):
    def collect(self, num_control_steps=1):
      if self.step:
        raise RuntimeError("intentional failure after baseline")
      return super().collect(num_control_steps)

  cfg = tiny_config(run_dir)
  with pytest.raises(RuntimeError, match="after baseline"):
    run_training(
      cfg,
      lambda config, context, seed: FailAfterWarmup(
        num_envs=config.num_envs,
        device=context.device,
        seed=seed,
        physics_dt=config.physics_dt,
      ),
    )
  for name in ("best.pt", "latest.pt"):
    saved = read_checkpoint(run_dir / name)
    assert saved["update"] == 0
    assert saved["metrics"]["rollout_0.04s/joint_position/rmse"] is not None
    assert bool(saved["model"]["normalization_ready"])
  assert json.loads((run_dir / "status.json").read_text())["status"] == "failed"


def test_stop_file_gracefully_exports_after_first_usable_update(run_dir):
  class StopCollector(SyntheticCollector):
    def collect(self, num_control_steps=1):
      result = super().collect(num_control_steps)
      # Create STOP after bootstrap, upon the next collection.
      if self.step >= 8:
        (run_dir / "STOP").touch()
      return result

  cfg = tiny_config(run_dir)
  final = run_training(
    cfg,
    lambda config, context, seed: StopCollector(
      num_envs=config.num_envs,
      device=context.device,
      seed=seed,
      physics_dt=config.physics_dt,
    ),
  )
  assert read_checkpoint(final)["update"] == 1
  status = json.loads((run_dir / "status.json").read_text())
  assert status["status"] == "stopped" and status["reason"] == "stop_file"


def test_two_rank_cpu_torchrun_synthetic_fit_and_normalization(run_dir):
  command = [
    sys.executable,
    "-m",
    "torch.distributed.run",
    "--standalone",
    "--nnodes=1",
    "--nproc_per_node=2",
    "scripts/train_wm.py",
    "--synthetic",
    "--device",
    "cpu",
    "--no-bf16",
    "--output-dir",
    str(run_dir),
    "--num-envs",
    "4",
    "--validation-envs",
    "1",
    "--replay-steps",
    "16",
    "--collect-steps",
    "4",
    "--normalization-samples",
    "8",
    "--batch-size",
    "4",
    "--horizon",
    "2",
    "--updates-per-collection",
    "1",
    "--gradient-accumulation",
    "2",
    "--eval-batches",
    "1",
    "--max-updates",
    "2",
    "--log-interval",
    "1",
    "--eval-horizons-seconds",
    "0.02",
    "0.04",
    "--model.hidden-dim",
    "16",
    "--model.interaction-dim",
    "8",
    "--model.num-blocks",
    "1",
  ]
  result = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
  assert result.returncode == 0, result.stdout + result.stderr
  saved = read_checkpoint(run_dir / "final.pt")
  assert saved["world_size"] == 2 and saved["update"] == 2
  assert saved["normalization"]["samples"] == 24
  assert saved["provenance"][0]["seed"] != saved["provenance"][1]["seed"]
  assert saved["transitions"] == 64
  assert saved["metrics"]["rollout_0.04s/seconds/mean"] == pytest.approx(0.04)
  assert saved["provenance"][0]["collector"]["physics_dt"] == 0.005
  assert saved["provenance"][1]["collector"]["physics_dt"] == 0.005
  assert json.loads((run_dir / "status_rank_1.json").read_text())["status"] == "completed"
  command[command.index("--max-updates") + 1] = "4"
  resumed = subprocess.run(
    [*command, "--resume", str(run_dir / "final.pt")],
    capture_output=True,
    text=True,
    timeout=120,
    check=False,
  )
  assert resumed.returncode == 0, resumed.stdout + resumed.stderr
  recovered = read_checkpoint(run_dir / "final.pt")
  assert recovered["update"] == 4 and recovered["transitions"] > saved["transitions"]
  torch.testing.assert_close(recovered["model"]["input_mean"], saved["model"]["input_mean"])
  torch.testing.assert_close(recovered["model"]["delta_scale"], saved["model"]["delta_scale"])
  assert all(int(value["step"]) == 4 for value in recovered["optimizer"]["state"].values())


def test_configuration_rejects_silent_fallback_and_bad_horizons(run_dir):
  with pytest.raises(ValueError, match="teacher checkpoint"):
    TrainingConfig()
  with pytest.raises(ValueError, match="horizon"):
    replace(tiny_config(run_dir), horizon=17)
