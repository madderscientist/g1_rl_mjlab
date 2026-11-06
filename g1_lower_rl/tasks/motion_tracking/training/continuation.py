"""Worker executed with the parent's verified PYTHONPATH, never a relaxed source whitelist."""

from __future__ import annotations

import json
import os
import sys
from datetime import timedelta
from pathlib import Path


def main(request: dict) -> None:
  import numpy as np
  import torch
  import torch.distributed as dist

  from g1_lower_rl.tasks.motion_tracking.training.position_policy import actor_observations, unflatten_native_history
  from g1_lower_rl.tasks.motion_tracking.training.position_runner import (
    PositionTrainer,
    PositionTrainingConfig,
    run_position_training,
  )
  from g1_lower_rl.world_model.checkpoint import atomic_json

  cfg = PositionTrainingConfig(**request["config"])
  if not request["preflight"]:
    run_position_training(cfg)
    return
  rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
  device = f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}" if world > 1 else cfg.device
  torch.cuda.set_device(device)
  trainer = None
  try:
    if world > 1:
      dist.init_process_group("nccl", timeout=timedelta(minutes=30), device_id=torch.device(device))
    trainer = PositionTrainer(cfg, rank, world, device)
    saved = torch.load(Path(cfg.resume) / f"rank_{rank}.pt", map_location="cpu", weights_only=True)

    def equal(actual, expected, name):
      if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or not torch.equal(actual.cpu(), expected):
          raise AssertionError(f"Restored tensor differs: {name}")
        if not torch.isfinite(actual).all():
          raise FloatingPointError(f"Nonfinite restored tensor: {name}")
      elif isinstance(expected, dict):
        if actual.keys() != expected.keys():
          raise AssertionError(f"Restored keys differ: {name}")
        for key in expected:
          equal(actual[key], expected[key], f"{name}.{key}")
      elif isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
          raise AssertionError(f"Restored length differs: {name}")
        for index, value in enumerate(expected):
          equal(actual[index], value, f"{name}.{index}")
      elif actual != expected:
        raise AssertionError(f"Restored value differs: {name}")

    for name in ("actor", "critic", "world_model"):
      module = trainer.model if name == "world_model" else getattr(trainer, name)
      equal(module.state_dict(), saved[f"{name}_state_dict"], name)
      equal(trainer.optimizers[name].state_dict(), saved["optimizers"][name], f"{name}_Adam")
    clocks = [
      "iteration", "controls", "scheduled_controls", "scheduled_actor_starts", "scheduled_training_transitions",
      "optimizer_updates",
    ]
    if "hybrid_updates" in saved:
      clocks.append("hybrid_updates")
    for name in clocks:
      equal(getattr(trainer, name), saved[name], name)
    equal(trainer.collector.library.visited, saved["visited_clips"], "coverage")
    for name in ("bin_return", "bin_memory"):
      equal(getattr(trainer.collector._native_command, name), saved["sampler"][name], name)
    equal(torch.get_rng_state(), saved["rng"]["torch"], "torch_RNG")
    equal(torch.cuda.get_rng_state(device), saved["rng"]["cuda"], "CUDA_RNG")
    if trainer.continuation["mode"] != "strict" or trainer.continuation["changed_sources"]:
      raise AssertionError("Preflight did not use strict, source-identical restoration")
    with torch.no_grad():
      collector = trainer.collector
      history = unflatten_native_history(collector._observations["policy"])[:8]
      reference = collector.reference_window(collector._native_command.frame[:8])
      action = trainer.actor(actor_observations(history, reference), stochastic_output=False)
      if not torch.isfinite(action).all():
        raise FloatingPointError("Restored actor produced nonfinite native reset actions")
      np.savez(
        trainer.output / f"native_inputs_rank_{rank}.npz",
        history=history.cpu().numpy(), reference=reference.cpu().numpy(), action=action.cpu().numpy(),
      )
    atomic_json(trainer.output / f"preflight_rank_{rank}.json", {
      "passed": True, "rank": rank, "world_size": world, "iteration": trainer.iteration,
      "optimizer_updates": trainer.optimizer_updates, "new_optimizer_updates": 0,
      "exact_models_and_three_Adam": True, "exact_sampler_coverage_clocks_and_RNG": True,
      "continuation": trainer.continuation,
      "simulator_and_replay": "reinitialized_as_defined_by_checkpoint_resume_contract",
      "native_controller": {
        "action_scale": trainer.scale.cpu().tolist(),
        "default_joint_pos": trainer.default_position.cpu().tolist(),
        "target_limits": trainer.target_limits.cpu().tolist(),
        "target_limited": trainer.target_limited.cpu().tolist(),
      },
    })
    print(f"[mocap] rank {rank}: strict restore at {trainer.iteration}; all states exact; zero optimizer updates", flush=True)
  finally:
    if trainer is not None:
      trainer.collector.close()
    if dist.is_initialized():
      dist.destroy_process_group()


if __name__ == "__main__":
  main(json.loads(sys.argv[1]))
