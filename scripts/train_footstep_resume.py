"""在两张 GPU 上从零开始或从检查点训练当前脚步任务"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import signal
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
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
  def save(self, path, infos=None):
    if os.environ.get("FOOTSTEP_PROFILE") == "precision":
      params = self.env.unwrapped.reward_manager.get_term_cfg("footstep_landing").params
      infos = {**(infos or {}), "precision_stage": {
        name: params[name] for name in ("landing_start_step", "landing_ramp_steps")
      }}
    return super().save(path, infos)

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
    with torch.inference_mode():
      self.env.reset()
    receipt = {
      "rank": self.gpu_global_rank, "pid": os.getpid(), "checkpoint": str(path),
      "checkpoint_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
      "saved_iteration": int(saved["iter"]), "next_iteration": self.current_learning_iteration,
      "actor_critic_exactly_restored": True, "optimizer_reset": reset_optimizer,
      "actor_critic_optimizer_exactly_restored": not reset_optimizer, "learning_rate": self.alg.learning_rate,
      "optimizer_state_entries": len(self.alg.optimizer.state), "envs_per_rank": env.num_envs,
      "environment_counter": env.common_step_counter,
      "arm_target_scale": env.event_manager.get_term_cfg("reset_arm_pose").params["target_scale"],
      "reward_weights": {name: env.reward_manager.get_term_cfg(name).weight
                         for name in env.reward_manager.active_terms},
      "stop_probability": env.command_manager.get_term("footsteps").cfg.source.stop_probability,
      "link_mass_scale_range": list(env.event_manager.get_term_cfg("link_mass").params["scale_range"]),
      "source_root": str(ROOT),
      "profile": os.environ.get("FOOTSTEP_PROFILE", "tracking"),
    }
    write_json(Path(self.logger.log_dir) / f"resume_rank_{self.gpu_global_rank}.json", receipt)
    print("RESUME_VERIFIED " + json.dumps(receipt), flush=True)
    return infos

  def learn(self, num_learning_iterations, init_at_random_ep_len=False):
    continuous = os.environ.get("FOOTSTEP_CONTINUOUS") == "1"
    duration_s = float(os.environ.get("FOOTSTEP_TRAIN_DURATION_S", "0"))
    if not math.isfinite(duration_s) or duration_s < 0:
      raise ValueError("Training duration must be finite and nonnegative")
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
        "profile": os.environ.get("FOOTSTEP_PROFILE", "tracking"),
        "reward_weights": {name: self.env.unwrapped.reward_manager.get_term_cfg(name).weight
                           for name in self.env.unwrapped.reward_manager.active_terms},
        "compile_backend": self.env.unwrapped.command_manager.get_term("footsteps").cfg.compile_backend,
        "envs_per_rank": self.env.unwrapped.num_envs,
      }
      write_json(log_dir / f"initialization_rank_{rank}.json", initialization)
      print("FRESH_INITIALIZATION_VERIFIED " + json.dumps(initialization), flush=True)
    start_timestamp = datetime.now(timezone.utc).timestamp()
    if duration_s and self.is_distributed:
      timestamp = torch.tensor([start_timestamp], dtype=torch.float64, device=self.device)
      distributed.broadcast(timestamp, src=0)
      start_timestamp = timestamp.item()
    started = time.monotonic()
    started_at = datetime.fromtimestamp(start_timestamp, timezone.utc)
    deadline = (started_at + timedelta(seconds=duration_s)).isoformat() if duration_s else None
    completed = 0
    stop_signal = None
    reason = "max_updates"
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
        "continuous": continuous, "duration_s": duration_s or None,
        "deadline_utc": deadline, "started_at_utc": started_at.isoformat(),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(), "elapsed_s": elapsed,
        "remaining_s": max(0., duration_s - elapsed) if duration_s else None,
        "save_interval": self.cfg.get("save_interval"),
        "first_iteration": first_iteration, "completed_updates": completed,
        "last_completed_iteration": first_iteration + completed - 1,
        "seconds_per_update": elapsed / completed if completed else None, **extra,
      }
      write_json(log_dir / f"status_rank_{rank}.json", record)
      return record

    def log_and_check_stop(*args, **kwargs):
      nonlocal completed, reason
      completed += 1
      original_log(*args, **kwargs)
      if completed == 1 or completed % 10 == 0:
        write_status("running")
      requested = stop_signal is not None or (log_dir / "STOP").exists()
      expired = bool(duration_s and rank == 0 and time.monotonic() - started >= duration_s)
      stop_flags[0] = 2 if requested else int(expired)
      if self.is_distributed:
        distributed.all_reduce(stop_flags, op=distributed.ReduceOp.MAX)
      stop_code = stop_flags.item()
      if stop_code:
        reason = "stop_requested" if stop_code == 2 else "duration_reached"
        raise StopRequested

    previous_handlers = {signum: signal.signal(signum, request_stop) for signum in (signal.SIGTERM, signal.SIGINT)}
    self.logger.log = log_and_check_stop
    print("TRAINING_STARTED " + json.dumps(write_status("running")), flush=True)
    try:
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
  parser.add_argument("--profile", choices=("tracking", "walk-first", "precision"), default="tracking")
  parser.add_argument("--landing-start-step", type=int, help="Precision ramp origin; defaults to the saved stage or checkpoint counter")
  parser.add_argument("--landing-ramp-updates", type=int, help="Precision ramp duration; defaults to the saved stage or 2000 updates")
  budget = parser.add_mutually_exclusive_group()
  budget.add_argument("--max-updates", type=int, help="Maximum PPO updates")
  budget.add_argument("--hours", type=float, help="Wall-clock training hours; stop and save after a complete update")
  parser.add_argument("--save-interval", type=int, default=100, help="Save every N cumulative PPO iterations")
  parser.add_argument("--tag", default="swing_linear_continuous")
  args = parser.parse_args()
  if args.from_scratch == (args.checkpoint is not None):
    parser.error("Specify either a checkpoint or --from-scratch, but not both")
  if args.reset_optimizer and args.from_scratch:
    parser.error("--reset-optimizer requires a checkpoint")
  if args.landing_start_step is not None and args.landing_start_step < 0:
    parser.error("--landing-start-step must be nonnegative")
  if args.landing_ramp_updates is not None and args.landing_ramp_updates <= 0:
    parser.error("--landing-ramp-updates must be positive")
  if args.profile != "precision" and (args.landing_start_step is not None or args.landing_ramp_updates is not None):
    parser.error("Landing ramp options require --profile precision")
  if args.hours is not None and (not math.isfinite(args.hours) or args.hours <= 0 or not math.isfinite(args.hours * 3600)):
    parser.error("--hours must be finite and positive")
  if args.save_interval <= 0:
    parser.error("--save-interval must be positive")
  if args.envs_per_rank <= 0 or (args.max_updates is not None and args.max_updates <= 0):
    raise ValueError("Environment and update counts must be positive")
  for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "FOOTSTEP_DEADLINE_UTC", "FOOTSTEP_TRAIN_DURATION_S"):
    os.environ.pop(name, None)
  os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
  os.environ["MUJOCO_GL"] = "egl"
  continuous = args.max_updates is None and args.hours is None
  os.environ["FOOTSTEP_CONTINUOUS"] = "1" if continuous else "0"
  if args.hours is not None:
    os.environ["FOOTSTEP_TRAIN_DURATION_S"] = str(args.hours * 3600)
  os.environ["FOOTSTEP_RESET_OPTIMIZER"] = "1" if args.reset_optimizer else "0"
  os.environ["FOOTSTEP_PROFILE"] = args.profile
  checkpoint = None if args.from_scratch else args.checkpoint.resolve(strict=True)
  first_iteration = 0
  checkpoint_counter = 0
  stage_config = {}
  if checkpoint is not None:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    first_iteration = int(saved["iter"]) + 1
    checkpoint_infos = saved.get("infos") or {}
    checkpoint_counter = checkpoint_infos.get("env_state", {}).get("common_step_counter", 0)
    if args.profile == "precision":
      stage_config = dict(checkpoint_infos.get("precision_stage", {}))
    del saved
  cfg = replace(train.TrainConfig.from_task(train.FOOTSTEP_TASK), gpu_ids=[0, 1], enable_nan_guard=True)
  if args.profile == "walk-first":
    from g1_lower_rl.tasks.footstep_tracking.walk_first import walk_first_env_cfg

    cfg = replace(cfg, env=walk_first_env_cfg())
  elif args.profile == "precision":
    from g1_lower_rl.tasks.footstep_tracking.walk_first.precision import precision_env_cfg

    stage_config.setdefault("landing_start_step", checkpoint_counter)
    stage_config.setdefault("landing_ramp_steps", 2000 * cfg.agent.num_steps_per_env)
    if args.landing_start_step is not None:
      stage_config["landing_start_step"] = args.landing_start_step
    if args.landing_ramp_updates is not None:
      stage_config["landing_ramp_steps"] = args.landing_ramp_updates * cfg.agent.num_steps_per_env
    cfg = replace(cfg, env=precision_env_cfg(**stage_config))
  cfg.env.scene.num_envs = args.envs_per_rank
  cfg.agent.seed = 42
  cfg.agent.resume = checkpoint is not None
  if checkpoint is not None:
    cfg.agent.load_run = re.escape(checkpoint.parent.name) + "$"
    cfg.agent.load_checkpoint = re.escape(checkpoint.name) + "$"
  cfg.agent.max_iterations = args.max_updates if args.max_updates is not None else (1000000000 if args.hours else 10000)
  cfg.agent.save_interval = args.save_interval
  cfg.agent.run_name = args.tag
  run_root = args.run_root or (checkpoint.parent.parent if checkpoint else ROOT / "logs/rsl_rl" / cfg.agent.experiment_name)
  log_dir = run_root.resolve() / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + args.tag)
  log_dir.mkdir(parents=True, exist_ok=False)
  setup = {
    "profile": args.profile,
    "stage_config": stage_config,
    "checkpoint": str(checkpoint) if checkpoint else None, "from_scratch": args.from_scratch,
    "reset_optimizer": args.reset_optimizer, "configured_learning_rate": cfg.agent.algorithm.learning_rate,
    "run_dir": str(log_dir), "source_root": str(ROOT),
    "gpu_ids": [0, 1], "envs_per_rank": args.envs_per_rank, "total_envs": 2 * args.envs_per_rank,
    "rollout_steps": cfg.agent.num_steps_per_env, "samples_per_update": 2 * args.envs_per_rank * cfg.agent.num_steps_per_env,
    "first_iteration": first_iteration, "max_updates": args.max_updates, "continuous": continuous,
    "hours": args.hours, "duration_s": args.hours * 3600 if args.hours is not None else None,
    "save_interval": cfg.agent.save_interval, "reward_weights": {name: term.weight for name, term in cfg.env.rewards.items()},
    "command_config": {"manager": asdict(cfg.env.commands["footsteps"].manager),
                       "source": asdict(cfg.env.commands["footsteps"].source)},
    "curriculum_config": {name: term.params for name, term in cfg.env.curriculum.items()},
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