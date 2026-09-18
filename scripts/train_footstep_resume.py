"""在两张 GPU 上从零开始或从检查点训练当前脚步任务"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import signal
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as distributed
import torchrunx
import train
from mjlab.rl import MjlabOnPolicyRunner


def assert_same_state(actual, expected):
  if isinstance(expected, torch.Tensor):
    assert torch.isfinite(expected).all()
    torch.testing.assert_close(actual.detach().cpu(), expected.detach().cpu(), rtol=0, atol=0)
  elif isinstance(expected, dict):
    assert actual.keys() == expected.keys()
    for key in expected:
      assert_same_state(actual[key], expected[key])
  elif isinstance(expected, (list, tuple)):
    assert len(actual) == len(expected)
    for actual_item, expected_item in zip(actual, expected, strict=True):
      assert_same_state(actual_item, expected_item)
  else:
    assert actual == expected


def write_json(path, record):
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(record, indent=2) + "\n")
  temporary.replace(path)


class StopRequested(Exception):
  pass


class FootstepResumeRunner(MjlabOnPolicyRunner):
  def load(self, path, load_cfg=None, strict=True, map_location=None):
    reset_optimizer = os.environ.get("FOOTSTEP_RESET_OPTIMIZER") == "1"
    if reset_optimizer:
      if self.alg.optimizer.state or self.alg.rnd is not None:
        raise ValueError("Optimizer reset requires a fresh PPO runner without RND")
      load_cfg = {"actor": True, "critic": True, "iteration": True, "optimizer": False, "rnd": False}
    infos = super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert_same_state(self.alg.actor.state_dict(), saved["actor_state_dict"])
    assert_same_state(self.alg.critic.state_dict(), saved["critic_state_dict"])
    if reset_optimizer:
      assert not self.alg.optimizer.state
      assert self.alg.learning_rate == self.cfg["algorithm"]["learning_rate"]
      assert all(group["lr"] == self.alg.learning_rate for group in self.alg.optimizer.param_groups)
    else:
      assert_same_state(self.alg.optimizer.state_dict(), saved["optimizer_state_dict"])
    self.current_learning_iteration = int(saved["iter"]) + 1
    self.alg.learning_rate = self.alg.optimizer.param_groups[0]["lr"]
    env = self.env.unwrapped
    receipt = {
      "rank": self.gpu_global_rank, "pid": os.getpid(), "checkpoint": str(path),
      "checkpoint_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
      "saved_iteration": int(saved["iter"]), "next_iteration": self.current_learning_iteration,
      "actor_critic_exactly_restored": True, "optimizer_reset": reset_optimizer,
      "actor_critic_optimizer_exactly_restored": not reset_optimizer, "learning_rate": self.alg.learning_rate,
      "optimizer_state_entries": len(self.alg.optimizer.state), "envs_per_rank": env.num_envs,
      "reward_weights": {name: env.reward_manager.get_term_cfg(name).weight
                         for name in env.reward_manager.active_terms},
      "stop_probability": env.command_manager.get_term("footsteps").cfg.source.stop_probability,
      "link_mass_scale_range": list(env.event_manager.get_term_cfg("link_mass").params["scale_range"]),
      "source_root": str(ROOT),
    }
    write_json(Path(self.logger.log_dir) / f"resume_rank_{self.gpu_global_rank}.json", receipt)
    print("RESUME_VERIFIED " + json.dumps(receipt), flush=True)
    return infos

  def learn(self, num_learning_iterations, init_at_random_ep_len=False):
    continuous = os.environ.get("FOOTSTEP_CONTINUOUS") == "1"
    rank = self.gpu_global_rank
    log_dir = Path(self.logger.log_dir)
    first_iteration = self.current_learning_iteration
    if self.cfg.get("resume") is False:
      assert first_iteration == 0 and not self.alg.optimizer.state
      assert self.alg.learning_rate == self.cfg["algorithm"]["learning_rate"]
      initialization = {
        "rank": rank, "pid": os.getpid(), "from_scratch": True, "checkpoint": None,
        "first_iteration": first_iteration, "optimizer_state_entries": len(self.alg.optimizer.state),
        "learning_rate": self.alg.learning_rate, "source_root": str(ROOT),
        "reward_weights": {name: self.env.unwrapped.reward_manager.get_term_cfg(name).weight
                           for name in self.env.unwrapped.reward_manager.active_terms},
        "compile_backend": self.env.unwrapped.command_manager.get_term("footsteps").cfg.compile_backend,
        "envs_per_rank": self.env.unwrapped.num_envs,
      }
      write_json(log_dir / f"initialization_rank_{rank}.json", initialization)
      print("FRESH_INITIALIZATION_VERIFIED " + json.dumps(initialization), flush=True)
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    completed = 0
    stop_signal = None
    stop_flags = torch.zeros(1, dtype=torch.int32, device=self.device)
    original_log = self.logger.log

    def request_stop(signum, frame):
      nonlocal stop_signal
      del frame
      stop_signal = signal.Signals(signum).name

    def write_status(status, **extra):
      elapsed = time.monotonic() - started
      record = {
        "rank": rank, "pid": os.getpid(), "world_size": self.gpu_world_size, "status": status,
        "continuous": continuous, "deadline_utc": None, "started_at_utc": started_at,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(), "elapsed_s": elapsed,
        "first_iteration": first_iteration, "completed_updates": completed,
        "last_completed_iteration": first_iteration + completed - 1,
        "seconds_per_update": elapsed / completed if completed else None, **extra,
      }
      write_json(log_dir / f"status_rank_{rank}.json", record)
      return record

    def log_and_check_stop(*args, **kwargs):
      nonlocal completed
      completed += 1
      original_log(*args, **kwargs)
      if completed == 1 or completed % 10 == 0:
        write_status("running")
      stop_flags[0] = int(stop_signal is not None or (log_dir / "STOP").exists())
      if self.is_distributed:
        distributed.all_reduce(stop_flags, op=distributed.ReduceOp.MAX)
      if stop_flags.item():
        raise StopRequested

    previous_handlers = {signum: signal.signal(signum, request_stop) for signum in (signal.SIGTERM, signal.SIGINT)}
    self.logger.log = log_and_check_stop
    print("TRAINING_STARTED " + json.dumps(write_status("running")), flush=True)
    try:
      reason = "max_updates"
      try:
        while True:
          super().learn(num_learning_iterations, init_at_random_ep_len=init_at_random_ep_len and completed == 0)
          if not continuous:
            break
          for model in (self.alg.actor, self.alg.critic):
            for module in model.modules():
              for name, buffer in module.named_buffers(recurse=False):
                if buffer.is_inference():
                  setattr(module, name, buffer.clone())
          self.current_learning_iteration += 1
          self.logger.writer = None
      except StopRequested:
        reason = "stop_requested"
        if rank == 0:
          self.save(str(log_dir / f"model_{self.current_learning_iteration}.pt"))
          if self.logger.writer is not None:
            self.logger.stop_logging_writer()
      if self.is_distributed:
        distributed.barrier()
      print("TRAINING_FINISHED " + json.dumps(write_status(
        "stopped" if reason == "stop_requested" else "completed", reason=reason,
        final_checkpoint=str(log_dir / f"model_{self.current_learning_iteration}.pt"),
      )), flush=True)
    except BaseException as error:
      write_status("failed", error_type=type(error).__name__, error=str(error))
      raise
    finally:
      self.logger.log = original_log
      for signum, handler in previous_handlers.items():
        signal.signal(signum, handler)


def worker(cfg, log_dir):
  train.MjlabOnPolicyRunner = FootstepResumeRunner
  cfg.env.sim.nan_guard.output_dir = str(log_dir / f"nan_rank_{os.environ.get('RANK', '0')}")
  try:
    train.run_train(train.FOOTSTEP_TASK, cfg, log_dir)
  finally:
    if distributed.is_initialized():
      distributed.destroy_process_group()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("checkpoint", type=Path, nargs="?")
  parser.add_argument("--from-scratch", action="store_true", help="Initialize new models and optimizer; do not load a checkpoint")
  parser.add_argument("--reset-optimizer", action="store_true", help="Restore models but keep the newly initialized optimizer and configured learning rate")
  parser.add_argument("--run-root", type=Path, help="Parent experiment directory for a new run")
  parser.add_argument("--envs-per-rank", type=int, default=128)
  parser.add_argument("--max-updates", type=int, help="Finite validation run; omitted means train until stopped")
  parser.add_argument("--tag", default="swing_linear_continuous")
  args = parser.parse_args()
  if args.from_scratch == (args.checkpoint is not None):
    parser.error("Specify either a checkpoint or --from-scratch, but not both")
  if args.reset_optimizer and args.from_scratch:
    parser.error("--reset-optimizer requires a checkpoint")
  if args.envs_per_rank <= 0 or (args.max_updates is not None and args.max_updates <= 0):
    raise ValueError("Environment and update counts must be positive")
  for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "FOOTSTEP_DEADLINE_UTC", "FOOTSTEP_TRAIN_DURATION_S"):
    os.environ.pop(name, None)
  os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
  os.environ["MUJOCO_GL"] = "egl"
  os.environ["FOOTSTEP_CONTINUOUS"] = "1" if args.max_updates is None else "0"
  os.environ["FOOTSTEP_RESET_OPTIMIZER"] = "1" if args.reset_optimizer else "0"
  checkpoint = None if args.from_scratch else args.checkpoint.resolve(strict=True)
  first_iteration = 0
  if checkpoint is not None:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    first_iteration = int(saved["iter"]) + 1
    del saved
  cfg = replace(train.TrainConfig.from_task(train.FOOTSTEP_TASK), gpu_ids=[0, 1], enable_nan_guard=True)
  cfg.env.scene.num_envs = args.envs_per_rank
  cfg.agent.seed = 42
  cfg.agent.resume = checkpoint is not None
  if checkpoint is not None:
    cfg.agent.load_run = re.escape(checkpoint.parent.name) + "$"
    cfg.agent.load_checkpoint = re.escape(checkpoint.name) + "$"
  cfg.agent.max_iterations = args.max_updates if args.max_updates is not None else 10000
  cfg.agent.save_interval = 100
  cfg.agent.run_name = args.tag
  expected_tracking = {
    "footstep_swing_position": 5.0, "footstep_swing_yaw": 5.0,
    "footstep_landing": -4.0, "footstep_support": -1.0,
  }
  assert len(cfg.env.rewards) == 18
  for name, weight in expected_tracking.items():
    assert cfg.env.rewards[name].weight == weight
  assert "footstep_distance" not in cfg.env.rewards and "footstep_approach" not in cfg.env.rewards
  assert cfg.env.rewards["lower_body_copper_proxy"].weight == -1.0
  assert cfg.env.rewards["head_height"].weight == 1.0
  assert "pelvis_height" not in cfg.env.rewards and "torso_height" not in cfg.env.rewards
  assert cfg.env.commands["footsteps"].source.stop_probability == 0.3
  assert cfg.env.events["link_mass"].params["scale_range"] == (0.95, 1.05)
  run_root = args.run_root or (checkpoint.parent.parent if checkpoint else ROOT / "logs/rsl_rl" / cfg.agent.experiment_name)
  log_dir = run_root.resolve() / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + args.tag)
  log_dir.mkdir(parents=True, exist_ok=False)
  setup = {
    "checkpoint": str(checkpoint) if checkpoint else None, "from_scratch": args.from_scratch,
    "reset_optimizer": args.reset_optimizer, "configured_learning_rate": cfg.agent.algorithm.learning_rate,
    "run_dir": str(log_dir), "source_root": str(ROOT),
    "gpu_ids": [0, 1], "envs_per_rank": args.envs_per_rank, "total_envs": 2 * args.envs_per_rank,
    "rollout_steps": cfg.agent.num_steps_per_env, "samples_per_update": 2 * args.envs_per_rank * cfg.agent.num_steps_per_env,
    "first_iteration": first_iteration, "max_updates": args.max_updates, "continuous": args.max_updates is None,
    "save_interval": cfg.agent.save_interval, "reward_weights": {name: term.weight for name, term in cfg.env.rewards.items()},
  }
  write_json(log_dir / "launch.json", setup)
  print("FOOTSTEP_RESUME_RUN " + json.dumps(setup), flush=True)
  os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")
  pythonpath = os.pathsep.join((str(ROOT), str(ROOT / "scripts")))
  logging.basicConfig(level=logging.INFO)
  torchrunx.Launcher(
    hostnames=["localhost"], workers_per_host=2, backend=None,
    copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*", "OMP_NUM_THREADS", "FOOTSTEP_*", "PYTHONUNBUFFERED"),
    extra_env_vars={"PYTHONPATH": pythonpath},
  ).run(worker, cfg, log_dir)


if __name__ == "__main__":
  main()