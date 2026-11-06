from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from g1_lower_rl.rl.supermimic import (
  bptt_return,
  cosine_learning_rate,
  frozen,
  gae_targets,
  optimizer_step,
)
from g1_lower_rl.tasks.motion_tracking.corpus import (
  TRACKED_BODIES,
  MappedMotionCorpus,
  PrepareMocapConfig,
  motion_family,
  prepare_mocap,
)
from g1_lower_rl.tasks.motion_tracking.training.position_runner import PositionTrainingConfig
from g1_lower_rl.world_model.model import BODY_NAMES


def test_survival_return_including_terminal_bootstrap_and_derivatives():
  rewards = torch.tensor([[1.0, 2.0]], requires_grad=True)
  termination = torch.tensor([[0.2, 0.3]], requires_grad=True)
  terminal = torch.tensor([3.0], requires_grad=True)
  result, survival = bptt_return(rewards, termination, terminal, 0.9)
  torch.testing.assert_close(result, torch.tensor([1 + 0.9 * 0.8 * 2 + 0.9**2 * 0.8 * 0.7 * 3]))
  torch.testing.assert_close(survival, torch.tensor([[1.0, 0.8]]))
  result.sum().backward()
  assert rewards.grad is not None and (rewards.grad > 0).all()
  assert termination.grad is not None and (termination.grad < 0).all()
  torch.testing.assert_close(terminal.grad, torch.tensor([0.9**2 * 0.8 * 0.7]))


def test_freezing_restores_preexisting_flags_after_exception():
  layer = nn.Linear(1, 1)
  layer.bias.requires_grad_(False)
  with pytest.raises(RuntimeError, match="deliberate"), frozen(layer):
    raise RuntimeError("deliberate")
  assert layer.weight.requires_grad and not layer.bias.requires_grad


def test_fixed_gae_targets_with_soft_and_hard_termination():
  rewards = torch.tensor([[1.0, 2.0]], requires_grad=True)
  termination = torch.tensor([[0.2, 1.0]], requires_grad=True)
  values = torch.tensor([[0.4, 0.5, 100.0]], requires_grad=True)
  targets = gae_targets(rewards, termination, values, 0.9, 0.95)
  last_advantage = 2 - 0.5
  first_advantage = 1 + 0.9 * 0.8 * 0.5 - 0.4 + 0.9 * 0.8 * 0.95 * last_advantage
  torch.testing.assert_close(targets, torch.tensor([[0.4 + first_advantage, 2.0]]))
  assert not targets.requires_grad
  with torch.no_grad():
    values.add_(10)
  torch.testing.assert_close(targets, torch.tensor([[0.4 + first_advantage, 2.0]]))


def test_time_limit_bootstraps_but_stops_the_gae_trace_and_actor_tail():
  rewards = torch.tensor([[1.0, 1000.0]])
  termination = torch.tensor([[0.2, 0.0]])
  truncation = torch.tensor([[1.0, 0.0]])
  values = torch.tensor([[0.0, 4.0, 900.0]], requires_grad=True)
  result, survival = bptt_return(
    rewards,
    termination,
    values[:, -1],
    0.9,
    truncation=truncation,
    bootstrap_values=values[:, 1:],
  )
  torch.testing.assert_close(result, torch.tensor([1 + 0.9 * 0.8 * 4]))
  torch.testing.assert_close(survival, torch.tensor([[1.0, 0.0]]))
  result.sum().backward()
  torch.testing.assert_close(values.grad, torch.tensor([[0.0, 0.9 * 0.8, 0.0]]))
  targets = gae_targets(rewards, termination, values, 0.9, 0.95, truncation=truncation)
  assert targets[0, 0] == pytest.approx(1 + 0.9 * 0.8 * 4)


def test_actor_clips_norm_then_elements_without_rebinding_ddp_gradients(monkeypatch):
  parameter = nn.Parameter(torch.ones(2))
  parameter.grad = torch.tensor([3.0, 4.0])
  optimizer = torch.optim.SGD([parameter], lr=0.1)
  pointer = parameter.grad.data_ptr()
  monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
  monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
  monkeypatch.setattr(torch.distributed, "all_reduce", lambda value: value.mul_(2))
  assert optimizer_step([parameter], optimizer, actor=True) == pytest.approx(5)
  assert parameter.grad.data_ptr() == pointer
  torch.testing.assert_close(parameter, torch.tensor([0.99, 0.99]))


@pytest.mark.parametrize("actor", [False, True])
def test_optimizer_normal_clipping_is_bitwise_unchanged(actor):
  parameters = [nn.Parameter(torch.ones(7)), nn.Parameter(torch.ones(11))]
  expected = [nn.Parameter(p.detach().clone()) for p in parameters]
  for index, (parameter, reference) in enumerate(zip(parameters, expected, strict=True)):
    parameter.grad = torch.linspace(-2, 3, parameter.numel()) + index
    reference.grad = parameter.grad.clone()
  old = torch.optim.Adam(expected, lr=1e-3)
  optimizer = torch.optim.Adam(parameters, lr=1e-3)
  expected_norm = nn.utils.clip_grad_norm_(expected, 1.0, error_if_nonfinite=True)
  if actor:
    for parameter in expected:
      parameter.grad.clamp_(-0.1, 0.1)
  old.step()
  assert optimizer_step(parameters, optimizer, actor=actor) == float(expected_norm)
  for parameter, reference in zip(parameters, expected, strict=True):
    torch.testing.assert_close(parameter, reference, rtol=0, atol=0)
    torch.testing.assert_close(parameter.grad, reference.grad, rtol=0, atol=0)


def test_optimizer_clips_finite_gradients_despite_fp32_norm_overflow(caplog):
  parameter = nn.Parameter(torch.zeros(3))
  parameter.grad = torch.tensor([1e20, -2e20, 3e20])
  pointer = parameter.grad.data_ptr()
  reference = parameter.grad.double()
  assert not torch.isfinite(parameter.grad.norm())
  optimizer = torch.optim.SGD([parameter], lr=0.1)
  norm = optimizer_step([parameter], optimizer)
  assert norm == pytest.approx(float(reference.norm()))
  assert parameter.grad.dtype == torch.float32 and parameter.grad.data_ptr() == pointer
  torch.testing.assert_close(parameter.grad, (reference / reference.norm()).float())
  assert "FP32 gradient norm overflow" in caplog.text


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_optimizer_still_rejects_nonfinite_gradient_elements(value):
  parameter = nn.Parameter(torch.ones(2))
  parameter.grad = torch.tensor([value, 1])
  optimizer = torch.optim.SGD([parameter], lr=0.1)
  with pytest.raises(RuntimeError, match="non-finite"):
    optimizer_step([parameter], optimizer)
  torch.testing.assert_close(parameter, torch.ones_like(parameter), rtol=0, atol=0)


@pytest.mark.parametrize("iteration,factor", [(0, 1), (250000, 0.03), (320000, 0.03)])
def test_schedule_units_and_two_gpu_scaling(iteration, factor):
  assert cosine_learning_rate(iteration, 2e-4, 2) == pytest.approx(2e-4 * math.sqrt(2) * factor)


def write_npz(path: Path, seed: int):
  rng = np.random.default_rng(seed)
  frames = 55 + seed
  arrays = {
    "fps": np.array([50.0], dtype=np.float32),
    "joint_pos": rng.normal(size=(frames, 29)).astype(np.float32),
    "joint_vel": rng.normal(size=(frames, 29)).astype(np.float32),
  }
  for name in ("pos", "lin_vel", "ang_vel"):
    arrays[f"body_{name}_w"] = rng.normal(size=(frames, 44, 3)).astype(np.float32)
  arrays["body_quat_w"] = np.zeros((frames, 44, 4), dtype=np.float32)
  arrays["body_quat_w"][..., 0] = 1
  np.savez(path, **arrays)


def test_pack_mapped_gather_split_groups_and_corruption(tmp_path):
  source = tmp_path / "source"
  source.mkdir()
  names = [f"family_{i}" for i in range(16)] + ["family_0_mir", "100style_run__000001", "100style_run__000002"]
  for seed, name in enumerate(names):
    write_npz(source / f"{name}.npz", seed)
  shutil.copyfile(source / "family_0.npz", source / "alias.npz")
  output = tmp_path / "packed"
  manifest = prepare_mocap(PrepareMocapConfig(str(source), str(output), validation_fraction=0.25))
  train = [r for r in manifest["clips"] if r["split"] == "train"]
  val = [r for r in manifest["clips"] if r["split"] == "validation"]
  assert train and val and len(train) + len(val) == len(names) + 1
  assert not {r["family"] for r in train} & {r["family"] for r in val}
  assert not {r["sha256"] for r in train} & {r["sha256"] for r in val}
  bodies = torch.tensor([BODY_NAMES.index(name) for name in TRACKED_BODIES])
  corpus = MappedMotionCorpus(str(output / "train"), bodies)
  corpus.verify()
  indices = corpus.start_idx[:2, None] + torch.tensor([[0, 1, 2]])
  assert corpus.body_pos_w[indices, 0].shape == (2, 3, 3)
  assert corpus.joint_pos[indices].shape == (2, 3, 29)
  torch.testing.assert_close(corpus.body_pos_w[corpus.start_idx, 0, :2], torch.zeros(len(train), 2))
  with np.load(source / f"{corpus.names[0]}.npz") as original:
    torch.testing.assert_close(corpus.joint_pos[corpus.start_idx[:1]], torch.from_numpy(original["joint_pos"][:1]))
  with pytest.raises(TypeError, match="materialization"):
    corpus.body_pos_w[:]
  with pytest.raises(IndexError):
    corpus.gather(torch.tensor([-1]))
  merged = MappedMotionCorpus(str(output / "train"), bodies, include_validation=True)
  assert merged.training_motions == len(train) and merged.num_motions == len(train) + len(val)
  with pytest.raises(FileExistsError):
    prepare_mocap(PrepareMocapConfig(str(source), str(output)))
  with (output / "frames.bin").open("r+b") as stream:
    stream.write(b"\x01")
  with pytest.raises(ValueError, match="SHA256"):
    corpus.verify()


def test_family_and_training_units():
  assert motion_family("100style_run__000100_000200_deadbeef_mir") == "100style_run"
  assert motion_family("walk_mir") == "walk"
  cfg = PositionTrainingConfig("wm", "replica", "output")
  cfg.validate()
  assert cfg.iterations == 0
  from dataclasses import replace

  replace(cfg, iterations=2).validate()
  with pytest.raises(ValueError):
    replace(cfg, iterations=-1).validate()
  assert cfg.wm_horizon * 4 == 32 and cfg.horizon * 4 == 128
  with pytest.raises(ValueError, match="H32"):
    replace(cfg, horizon=16).validate()
