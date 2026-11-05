from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict

import pytest
import torch

from g1_lower_rl.world_model import (
  PositionWorldModel,
  PositionWorldModelCfg,
  export_pretrained_checkpoint,
  load_world_model,
  pretrained_checkpoint_path,
  read_world_model_checkpoint,
)
from g1_lower_rl.world_model import checkpoint as shared


@pytest.fixture
def snapshot(tmp_path):
  model = PositionWorldModel(PositionWorldModelCfg(16, 8, 1))
  mean = torch.zeros(145)
  mean[3] = -1
  model.set_normalization(mean, torch.ones(145), torch.ones(70), torch.ones(6), torch.ones(70))
  source = tmp_path / "best.pt"
  payload = {
    "format_version": shared.CHECKPOINT_VERSION,
    "model_config": asdict(model.cfg),
    "model": model.state_dict(),
    "normalization": {"frozen": True, "samples": 100},
    "optimizer": {"state": torch.ones(3)},
    "rng": {"state": torch.ones(3)},
    "config": {"eval_horizons_seconds": [0.02, 0.16]},
    "timestamp_utc": "2026-10-26T00:00:00Z",
    "update": 123,
    "transitions": 456,
    "best_metric": "test/rmse",
    "best_score": 0.25,
    "metrics": {"test/rmse": 0.25},
    "source_provenance": {
      "format_version": 1,
      "sha256": "source-digest",
      "files": {"g1_lower_rl/world_model/model.py": "model-digest"},
      "snapshot_directory": "/private/source",
    },
    "provenance": [
      {
        "rank": 0,
        "collector": {
          "teacher_checkpoint": "/private/teacher.pt",
          "teacher_sha256": "teacher-digest",
          "motion_dir": "/private/corpus",
          "static_template": {"test": "geometry"},
          "controller_template": {"test": "controller"},
          "physics_dt": 0.005,
          "control_dt": 0.02,
          "fixed_actuator_parameters": {"kp": [10.0], "kd": [1.0]},
          "corpus": {
            "manifest_path": "/private/corpus/manifest.json",
            "manifest_sha256": "full-corpus-digest",
            "clips": 98049,
            "train_clips": 96152,
            "validation_clips": 1897,
          },
          "corpus_manifest": {
            "sha256": "manifest-digest",
            "contents": {
              "version": 1,
              "seed": 1,
              "selected_clips": 1,
              "source_clips": 1,
              "selected_bytes": 100,
              "selection_sha256": "selection-digest",
              "source": "/private/corpus",
              "motions": [{"source": "/private/corpus/clip.npz"}],
            },
          },
        },
      }
    ],
  }
  torch.save(payload, source)
  return source, payload


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_portable_export_preserves_weights_normalization_predictions_and_gradients(snapshot, tmp_path, dtype):
  source, payload = snapshot
  payload["model"] = PositionWorldModel(PositionWorldModelCfg(16, 8, 1)).to(dtype).state_dict()
  payload["model"]["normalization_ready"].fill_(True)
  torch.save(payload, source)
  original_bytes = source.read_bytes()
  destination = tmp_path / "portable" / "init.pt"
  export_pretrained_checkpoint(source, destination)
  first, second = load_world_model(source), load_world_model(destination)
  for key, tensor in first.state_dict().items():
    if isinstance(tensor, torch.Tensor):
      torch.testing.assert_close(tensor, second.state_dict()[key], rtol=0, atol=0)
    else:
      assert tensor == second.state_dict()[key]
  state = torch.randn(2, 71, dtype=dtype)
  state[:, 3:7] = torch.tensor([1, 0, 0, 0], dtype=dtype)
  inputs = torch.randn(2, 29, dtype=dtype, requires_grad=True)
  context = torch.ones(2, 48, dtype=dtype)
  predicted, restored = first(state, inputs, context).as_tensor(), second(state, inputs, context).as_tensor()
  torch.testing.assert_close(predicted, restored, rtol=0, atol=0)
  torch.testing.assert_close(
    torch.autograd.grad(predicted.square().mean(), inputs)[0],
    torch.autograd.grad(restored.square().mean(), inputs)[0],
    rtol=0,
    atol=0,
  )
  assert not second.training and second.input_mean.dtype == dtype
  assert all(parameter.requires_grad for parameter in second.parameters())
  assert source.read_bytes() == original_bytes
  exported = read_world_model_checkpoint(destination)
  assert exported["artifact_kind"] == "pretrained_initialization"
  assert not {"optimizer", "rng", "config", "scaler"} & exported.keys()
  assert "/private" not in repr(exported)
  assert exported["provenance"][0]["collector"]["static_template"] == {"test": "geometry"}
  assert exported["initialization"]["source_update"] == 123
  assert exported["initialization"]["new_training_updates"] == 0
  collector = exported["provenance"][0]["collector"]
  assert collector["fixed_actuator_parameters"] == {"kp": [10.0], "kd": [1.0]}
  assert collector["corpus"]["train_clips"] == 96152
  assert collector["corpus"]["validation_clips"] == 1897
  assert "manifest_path" not in collector["corpus"]
  manifest = json.loads(destination.with_suffix(".json").read_text())
  assert manifest["physical_contract"] == shared.primitive_metadata(exported["model"]["_extra_state"])


def test_full_mmap_export_preserves_provenance_without_local_paths(snapshot, tmp_path):
  source, payload = snapshot
  original = payload["provenance"][0]["collector"]["corpus_manifest"]
  original["contents"] = {
    "version": 1, "dtype": "<f4", "frame_width": 244, "fps": 50,
    "frames": 32440430, "bytes": 31661859680, "data_sha256": "frame-digest",
    "joint_names": ["legacy-joint"], "source_body_names": ["pelvis"], "body_names": ["pelvis"],
    "split_seed": 0, "validation_fraction": 0.02, "limited_to_clips": 0,
    "evaluation_caveat": "Not a zero-shot holdout", "clip_count": 98049,
    "split_counts": {"train": 96152, "validation": 1897},
    "source": "/private/full-corpus",
  }
  torch.save(payload, source)
  destination = tmp_path / "full-corpus-init.pt"
  export_pretrained_checkpoint(source, destination)
  exported = read_world_model_checkpoint(destination)
  manifest = exported["provenance"][0]["collector"]["corpus_manifest"]
  assert manifest["contents"]["split_counts"] == {"train": 96152, "validation": 1897}
  assert "/private" not in repr(exported)


def test_native_collection_helpers_do_not_require_policy_training_entrypoint():
  code = """
import importlib.abc
import sys
class BlockPolicyTraining(importlib.abc.MetaPathFinder):
  def find_spec(self, fullname, path=None, target=None):
    if fullname == 'scripts.train_supermimic29':
      raise AssertionError('WM collection must not import a policy-training CLI')
sys.meta_path.insert(0, BlockPolicyTraining())
from g1_lower_rl.world_model.native import fixed29_runtime, load_weights, tensor_hash
from g1_lower_rl.world_model.baseline_collection import BaselineMotionCollector
assert callable(fixed29_runtime) and callable(load_weights) and callable(tensor_hash)
"""
  result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30, check=False)
  assert result.returncode == 0, result.stdout + result.stderr


def test_export_never_overwrites_source_or_existing_version(snapshot, tmp_path):
  source, _ = snapshot
  with pytest.raises(ValueError, match="training source"):
    export_pretrained_checkpoint(source, source, overwrite=True)
  destination = tmp_path / "init.pt"
  export_pretrained_checkpoint(source, destination)
  with pytest.raises(FileExistsError):
    export_pretrained_checkpoint(source, destination)
  with pytest.raises(ValueError, match="full training snapshot"):
    export_pretrained_checkpoint(destination, tmp_path / "other.pt")


@pytest.mark.parametrize("invalid", ["unfitted", "zero_scale", "nonfinite", "contract"])
def test_loader_refuses_invalid_checkpoint_state(snapshot, invalid):
  source, payload = snapshot
  if invalid == "unfitted":
    payload["model"]["normalization_ready"].fill_(False)
  elif invalid == "zero_scale":
    payload["model"]["input_scale"][0] = 0
  elif invalid == "nonfinite":
    payload["model"]["input_mean"][0] = float("nan")
  else:
    payload["model"]["_extra_state"]["contract_version"] = -1
  torch.save(payload, source)
  with pytest.raises((ValueError, RuntimeError, FloatingPointError)):
    load_world_model(source)


def test_public_loader_has_no_training_simulator_or_cli_imports(snapshot):
  code = """
import importlib.abc
import sys
class BlockDependencies(importlib.abc.MetaPathFinder):
  def find_spec(self, fullname, path=None, target=None):
    blocked = ('mjlab', 'mujoco', 'warp', 'rsl_rl', 'tyro',
               'g1_lower_rl.world_model.training', 'g1_lower_rl.world_model.collection')
    if any(fullname == item or fullname.startswith(item + '.') for item in blocked):
      raise AssertionError('Unexpected import: ' + fullname)
sys.meta_path.insert(0, BlockDependencies())
from g1_lower_rl.world_model import load_world_model
assert bool(load_world_model(sys.argv[1]).normalization_ready)
"""
  result = subprocess.run(
    [sys.executable, "-c", code, str(snapshot[0])],
    capture_output=True,
    text=True,
    timeout=30,
    check=False,
  )
  assert result.returncode == 0, result.stdout + result.stderr


def test_training_api_uses_shared_loader():
  from g1_lower_rl.world_model.training import load_world_model as training_loader

  assert training_loader is load_world_model


def test_legacy_initialization_is_not_silently_reused_for_fixed29(tmp_path):
  path = pretrained_checkpoint_path()
  assert path.name == "g1_wm.pt"
  legacy = tmp_path / "g1_gloria_torque_v1.pt"
  torch.save({"format_version": 1, "model": {"_extra_state": {"contract_version": 1}}}, legacy)
  with pytest.raises(ValueError, match="v3"):
    read_world_model_checkpoint(legacy)
