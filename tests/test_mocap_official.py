from __future__ import annotations

import copy
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from mjlab.tasks.registry import list_tasks as list_ppo_tasks

from g1_lower_rl.tasks import MOCAP_TASK, list_tasks
from g1_lower_rl.tasks.motion_tracking.training import TrainingConfig
from g1_lower_rl.tasks.motion_tracking.training.export import OnnxActor, export_onnx
from g1_lower_rl.tasks.motion_tracking.training.official import (
  frozen_runtime,
  load_mlp_checkpoint,
  run_training,
  training_config,
)
from g1_lower_rl.tasks.motion_tracking.training.position_policy import (
  PositionMlpActor,
  actor_architecture_metadata,
  actor_observations,
)
from g1_lower_rl.tasks.motion_tracking.training.position_runner import FORMAT, PositionTrainer, PositionTrainingConfig
from g1_lower_rl.tasks.motion_tracking.training.position_task import position_task_contract


def saved_checkpoint() -> dict:
  cfg = training_config(TrainingConfig(output="new", world_model="wm.pt", replica_source="native"))
  return {
    "format": FORMAT, "rank": 0, "world_size": 2, "iteration": 14160, "config": cfg,
    "origin": {"architecture": actor_architecture_metadata()},
    "collector": {"contract_version": 3}, "source_provenance": {"files": {}},
  }


def test_official_task_is_supermimic_not_a_ppo_registration():
  assert list_tasks().count(MOCAP_TASK) == 1
  assert MOCAP_TASK not in list_ppo_tasks()
  assert not any("LegacyPPO" in name for name in list_tasks())
  cfg = training_config(TrainingConfig(output="new", world_model="wm.pt", replica_source="native"))
  assert cfg["actor_architecture"] == "mlp" and cfg["native_actor_kl"] and "real_ppo" not in cfg
  assert cfg["task_definition"] == "recovery_xy_light_v1" and cfg["horizon"] == 32


def test_fresh_entry_freezes_complete_runtime_before_launch(tmp_path, monkeypatch):
  from g1_lower_rl.tasks.motion_tracking.training import official

  launched = []
  monkeypatch.setattr(official, "launch_worker", lambda cfg, runtime, **_: launched.append((cfg, runtime)))
  args = TrainingConfig(output=str(tmp_path / "new"), world_model="wm.pt", replica_source="native", iterations=1)
  run_training(args)
  assert len(launched) == 1
  cfg, runtime = launched[0]
  assert cfg["actor_architecture"] == "mlp" and cfg["native_actor_kl"] and "real_ppo" not in cfg
  assert (runtime / "READY").is_file()
  assert (runtime / "g1_lower_rl/tasks/motion_tracking/training/position_runner.py").is_file()
  assert (runtime / "scripts/train_mocap.py").is_file()
  with pytest.raises(FileExistsError):
    run_training(args)


@pytest.mark.parametrize("args", [
  TrainingConfig(output="new"),
  TrainingConfig(output="new", world_model="wm.pt"),
  TrainingConfig(output="new", world_model="wm.pt", replica_source="native", preflight=True),
  TrainingConfig(output="new", world_model="wm.pt", replica_source="native", iterations=-1),
])
def test_incomplete_or_invalid_fresh_config_is_rejected(args):
  with pytest.raises(ValueError):
    training_config(args)


def test_continuation_inherits_every_saved_setting_except_output_resume_and_target():
  saved = saved_checkpoint()
  saved["config"].update(actor_microbatch=128, num_envs=2048, validation_envs=256, cuda_graphs=True,
                         matmul_precision="highest")
  cfg = training_config(TrainingConfig(output="new", resume="rank_0.pt", iterations=14160), saved)
  actual = cfg
  for key, value in saved["config"].items():
    if key not in ("output", "resume", "iterations"):
      assert actual[key] == value, key
  assert cfg["iterations"] == 14160
  with pytest.raises(ValueError, match="cannot change"):
    training_config(TrainingConfig(output="new", resume="rank_0.pt", num_envs=4096), saved)


@pytest.mark.parametrize("change", [
  {"actor_architecture": "rgmt"}, {"real_ppo": True}, {"native_actor_kl": False},
  {"task_definition": "supermimic_v1"},
])
def test_official_continuation_refuses_legacy_algorithms(change):
  saved = saved_checkpoint()
  saved["config"].update(change)
  with pytest.raises(ValueError):
    training_config(TrainingConfig(output="new", resume="rank_0.pt"), saved)


def test_checkpoint_type_and_frozen_source_are_checked(tmp_path):
  checkpoint = tmp_path / "training/update_00014160/rank_0.pt"
  checkpoint.parent.mkdir(parents=True)
  saved = saved_checkpoint()
  torch.save(saved, checkpoint)
  assert load_mlp_checkpoint(checkpoint)["iteration"] == 14160
  with pytest.raises(FileNotFoundError, match="runtime"):
    frozen_runtime(checkpoint, saved)
  runner = tmp_path / "runtime/g1_lower_rl/tasks/motion_tracking/training/position_runner.py"
  runner.parent.mkdir(parents=True)
  runner.write_text("unchanged\n")
  assert frozen_runtime(checkpoint, saved) == tmp_path / "runtime"
  saved["source_provenance"]["files"] = {"missing.py": "not-the-right-hash"}
  with pytest.raises(ValueError, match="source changed"):
    frozen_runtime(checkpoint, saved)
  saved["format"] = "actor_only"
  torch.save(saved, checkpoint)
  with pytest.raises(ValueError, match="complete"):
    load_mlp_checkpoint(checkpoint)


def test_resume_refuses_wrong_world_size_before_starting_any_worker(tmp_path):
  checkpoint = tmp_path / "training/rank_0.pt"
  checkpoint.parent.mkdir()
  saved = saved_checkpoint()
  torch.save(saved, checkpoint)
  torch.save({**saved, "rank": 1}, checkpoint.with_name("rank_1.pt"))
  runner = tmp_path / "runtime/g1_lower_rl/tasks/motion_tracking/training/position_runner.py"
  runner.parent.mkdir(parents=True)
  runner.write_text("unchanged\n")
  with pytest.raises(ValueError, match="nproc-per-node=2"):
    run_training(TrainingConfig(output=str(tmp_path / "new"), resume=str(checkpoint)))


def test_clean_trainer_strictly_restores_all_models_adam_sampler_budgets_and_rng(tmp_path, monkeypatch):
  cfg = PositionTrainingConfig("wm", "native", str(tmp_path / "parent"))
  trainer = PositionTrainer.__new__(PositionTrainer)
  trainer.rank, trainer.world_size, trainer.device = 0, 1, torch.device("cpu")
  trainer.sources = {"files": {"current.py": "verified-source"}}
  trainer.actor, trainer.critic, trainer.model = (torch.nn.Linear(2, 2) for _ in range(3))
  trainer.optimizers = {}
  for name, module in (("actor", trainer.actor), ("critic", trainer.critic), ("world_model", trainer.model)):
    optimizer = torch.optim.Adam(module.parameters())
    module(torch.ones(1, 2)).square().sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    trainer.optimizers[name] = optimizer
  contract = position_task_contract(cfg.task_definition, cfg.gamma)
  trainer.collector = SimpleNamespace(
    metadata={"task_contract": contract}, counters={}, _initialization={"reference": 0, "half_squat": 0, "airborne": 0},
    library=SimpleNamespace(visited=torch.zeros(3, dtype=torch.bool)),
    _native_command=SimpleNamespace(bin_return=torch.zeros(2), bin_memory=torch.zeros(2)),
  )
  saved = {
    "format": FORMAT, "rank": 0, "world_size": 1, "config": asdict(cfg),
    "source_provenance": trainer.sources, "collector": trainer.collector.metadata, "task_contract": contract,
    "actor_state_dict": copy.deepcopy(trainer.actor.state_dict()),
    "critic_state_dict": copy.deepcopy(trainer.critic.state_dict()),
    "world_model_state_dict": copy.deepcopy(trainer.model.state_dict()),
    "optimizers": {name: copy.deepcopy(optimizer.state_dict()) for name, optimizer in trainer.optimizers.items()},
    "iteration": 7, "controls": 19, "scheduled_controls": 11,
    "scheduled_actor_starts": 512 * 7, "scheduled_training_transitions": 896 * 11,
    "optimizer_updates": {"actor": 1, "critic": 1, "world_model": 1},
    "collector_counters": {"initial_reference": 17, "initial_half_squat": 3, "initial_airborne": 2},
    "visited_clips": torch.tensor([True, False, True]),
    "sampler": {"bin_return": torch.tensor([2., 3.]), "bin_memory": torch.tensor([.2, .8])},
    "rng": {"torch": torch.get_rng_state(), "cuda": torch.tensor([1, 2], dtype=torch.uint8)},
  }
  checkpoint = tmp_path / "rank_0.pt"
  torch.save(saved, checkpoint)
  trainer.cfg = replace(cfg, output=str(tmp_path / "child"), resume=str(checkpoint))
  cuda_rng = []
  monkeypatch.setattr(torch.cuda, "set_rng_state", lambda value, device: cuda_rng.append(value.clone()))
  with torch.no_grad():
    for module in (trainer.actor, trainer.critic, trainer.model):
      for parameter in module.parameters():
        parameter.zero_()
    for optimizer in trainer.optimizers.values():
      optimizer.state.clear()
  trainer.restore(checkpoint)
  for name, module in (("actor", trainer.actor), ("critic", trainer.critic), ("world_model", trainer.model)):
    torch.testing.assert_close(module.state_dict(), saved[f"{name}_state_dict"], rtol=0, atol=0)
    torch.testing.assert_close(trainer.optimizers[name].state_dict(), saved["optimizers"][name], rtol=0, atol=0)
  for name in ("iteration", "controls", "scheduled_controls", "scheduled_actor_starts",
               "scheduled_training_transitions", "optimizer_updates"):
    assert getattr(trainer, name) == saved[name]
  assert trainer.collector._initialization == {"reference": 17, "half_squat": 3, "airborne": 2}
  torch.testing.assert_close(trainer.collector.library.visited, saved["visited_clips"])
  for name, value in saved["sampler"].items():
    torch.testing.assert_close(getattr(trainer.collector._native_command, name), value, rtol=0, atol=0)
  assert torch.equal(torch.get_rng_state(), saved["rng"]["torch"])
  assert torch.equal(cuda_rng[0], saved["rng"]["cuda"])
  assert trainer.continuation["mode"] == "strict" and not trainer.continuation["changed_sources"]
  for changes, message in (
    ({"source_provenance": {"files": {}}}, "provenance changed"),
    ({"config": {**saved["config"], "learning_rate": 1.}}, "configuration changed"),
    ({"collector": {}}, "provenance changed"),
    ({"world_size": 2}, "world mismatch"),
  ):
    torch.save({**saved, **changes}, checkpoint)
    with pytest.raises(ValueError, match=message):
      trainer.restore(checkpoint)


def test_onnx_wrapper_preserves_time_term_layout_and_all_internal_scales():
  actor = PositionMlpActor().eval()
  history, reference = torch.randn(4, 10, 93), torch.randn(4, 798)
  with torch.no_grad():
    torch.testing.assert_close(
      OnnxActor(actor)(history, reference), actor(actor_observations(history, reference)), rtol=0, atol=0,
    )


def test_dynamic_onnx_export_and_ort_parity(tmp_path, monkeypatch):
  from g1_lower_rl.tasks.motion_tracking.training import export

  saved = saved_checkpoint()
  actor = PositionMlpActor().eval()
  saved.update(actor_state_dict=actor.state_dict(), task_contract={"definition": "recovery_xy_light_v1"})
  checkpoint = tmp_path / "rank_0.pt"
  torch.save(saved, checkpoint)
  monkeypatch.setattr(export, "controller_contract", lambda _: {"joint_names": [str(i) for i in range(29)]})
  native_inputs = tmp_path / "native_inputs.npz"
  np.savez(native_inputs, history=np.zeros((3, 10, 93), dtype="float32"), reference=np.ones((3, 798), dtype="float32"))
  path = export_onnx(checkpoint, tmp_path / "onnx", native_inputs=native_inputs)
  assert path.is_file()
  import json

  contract = json.loads(path.with_name("policy_contract.json").read_text())
  report = json.loads(path.with_name("onnx_validation.json").read_text())
  assert contract["normalization"]["inside_onnx"]
  assert contract["outputs"]["raw_residual"]["shape"] == ["batch", 29]
  assert report["passed"] and report["native_sample_included"]
  assert report["samples"]["native_reset"]["batch"] == 3
  with pytest.raises(FileExistsError):
    export_onnx(checkpoint, path.parent)


def test_native_renderer_requests_render_mode_and_writes_arrays_for_each_clip(tmp_path, monkeypatch):
  import imageio.v2 as imageio

  from g1_lower_rl.tasks.motion_tracking.training import playback

  frames, options, closed = [], [], []

  class Session:
    def __init__(self, *_, **kwargs):
      options.append(kwargs)
      self.names = ["clipA", "clipB"]
      self.env = SimpleNamespace(
        cfg=SimpleNamespace(viewer=SimpleNamespace(env_idx=0)),
        reset=lambda: ({}, {}), render=lambda: np.zeros((24, 32, 3), dtype=np.uint8),
      )

    def step(self):
      return torch.zeros(2), torch.zeros(2, dtype=torch.bool)

    def close(self):
      closed.append(True)

  class Writer:
    def __enter__(self):
      return self

    def __exit__(self, *_):
      return None

    def append_data(self, frame):
      assert isinstance(frame, np.ndarray)
      frames.append(frame)

  monkeypatch.setattr(playback, "NativePlayback", Session)
  monkeypatch.setattr(imageio, "get_writer", lambda *_, **__: Writer())
  playback.render("checkpoint", str(tmp_path / "out.mp4"), seconds=.04)
  assert options[0]["render_mode"] == "rgb_array"
  assert len(frames) == 2 and closed == [True]
