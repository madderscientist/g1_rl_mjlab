"""Train one shared footstep actor with task-local walking and step critics."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as distributed
import torchrunx
from rsl_rl.algorithms import PPO

import train
from train_footstep_resume import FootstepResumeRunner, assert_same_state, restore_fixed_exploration, write_json


def validate_task_weights(weights):
  if len(weights) != 2 or any(not math.isfinite(weight) or weight <= 0 for weight in weights):
    raise ValueError("Two finite positive task weights are required")
  if not math.isclose(sum(weights), 1., abs_tol=1e-8):
    raise ValueError("Task weights must sum to one")


def restore_joint_state(algorithm, saved, rank):
  task = ("walking", "single_step")[rank]
  infos = saved["infos"]
  if not infos.get("shared_actor"):
    raise ValueError("Checkpoint is not a compatible shared-actor joint run")
  validate_task_weights(infos["task_weights"])
  algorithm.actor.load_state_dict(saved["actor_state_dict"], strict=True)
  algorithm.critic.load_state_dict(saved["task_critic_state_dicts"][task], strict=True)
  algorithm.optimizer.load_state_dict(saved["task_optimizer_state_dicts"][task])
  assert_same_state(algorithm.actor.state_dict(), saved["actor_state_dict"])
  assert_same_state(algorithm.critic.state_dict(), saved["task_critic_state_dicts"][task])
  assert_same_state(algorithm.optimizer.state_dict(), saved["task_optimizer_state_dicts"][task])
  algorithm.learning_rate = algorithm.optimizer.param_groups[0]["lr"]
  return infos["task_environment_counters"][rank]


class SharedActorPPO(PPO):
  def broadcast_parameters(self):
    for value in self.actor.state_dict().values():
      distributed.broadcast(value, src=0)

  def reduce_parameters(self):
    weight = self.task_weight
    parameters = [parameter for parameter in self.actor.parameters() if parameter.requires_grad]
    gradients = torch.cat([
      (parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)).reshape(-1)
      for parameter in parameters
    ])
    gradients.mul_(weight)
    distributed.all_reduce(gradients)
    offset = 0
    for parameter in parameters:
      count = parameter.numel()
      parameter.grad = gradients[offset:offset + count].view_as(parameter).clone()
      offset += count

  def verify_actor(self):
    for value in self.actor.state_dict().values():
      reference = value.clone()
      distributed.broadcast(reference, src=0)
      if not torch.equal(value, reference):
        raise RuntimeError("Shared actor replicas diverged")

  def update(self):
    losses = super().update()
    self.joint_updates += 1
    self.verify_actor()
    iteration = self.joint_first_iteration + self.joint_updates - 1
    if self.joint_updates % self.joint_save_interval == 0 or iteration == self.joint_evaluation_iteration:
      self.save_joint()
    return losses

  def save_joint(self):
    self.verify_actor()
    local = {
      "critic": {name: value.detach().cpu() for name, value in self.critic.state_dict().items()},
      "optimizer": self.optimizer.state_dict(),
      "counter": self.joint_env.unwrapped.common_step_counter,
    }
    records = [None, None]
    distributed.all_gather_object(records, local)
    iteration = self.joint_first_iteration + self.joint_updates - 1
    if self.gpu_global_rank == 0:
      saved = self.save()
      saved.update(iter=iteration, task_critic_state_dicts={
        task: record["critic"] for task, record in zip(("walking", "single_step"), records)
      }, task_optimizer_state_dicts={
        task: record["optimizer"] for task, record in zip(("walking", "single_step"), records)
      }, infos={"shared_actor": True, "task_weights": self.task_weights,
        "fixed_exploration_std": .25, "actor_normalizer_frozen": True,
        "precision_stage": self.joint_stage,
        "env_state": {"common_step_counter": records[0]["counter"]},
        "task_environment_counters": [record["counter"] for record in records]})
      destination = self.joint_log_dir / f"model_{iteration}.pt"
      temporary = destination.with_suffix(".pt.tmp")
      torch.save(saved, temporary)
      temporary.replace(destination)
      print("JOINT_CHECKPOINT " + str(destination), flush=True)
    distributed.barrier()


class JointRunner(FootstepResumeRunner):
  def _configure_multi_gpu(self):
    if os.environ.get("JOINT_DISTRIBUTED_BACKEND", "nccl") == "nccl":
      return super()._configure_multi_gpu()
    self.gpu_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    self.is_distributed = self.gpu_world_size > 1
    self.gpu_local_rank = int(os.environ.get("LOCAL_RANK", "0")) if self.is_distributed else 0
    self.gpu_global_rank = int(os.environ.get("RANK", "0")) if self.is_distributed else 0
    if not self.is_distributed:
      self.cfg["multi_gpu"] = None
      return
    if self.device != f"cuda:{self.gpu_local_rank}":
      raise ValueError("Device does not match local GPU rank")
    if not 0 <= self.gpu_local_rank < self.gpu_world_size or not 0 <= self.gpu_global_rank < self.gpu_world_size:
      raise ValueError("Invalid distributed rank")
    self.cfg["multi_gpu"] = {"global_rank": self.gpu_global_rank,
      "local_rank": self.gpu_local_rank, "world_size": self.gpu_world_size}
    torch.cuda.set_device(self.gpu_local_rank)
    distributed.init_process_group(backend="gloo", rank=self.gpu_global_rank, world_size=self.gpu_world_size)

  def load(self, path, load_cfg=None, strict=True, map_location=None):
    walking = torch.load(path, map_location="cpu", weights_only=False)
    joint_resume = os.environ.get("JOINT_RESUME") == "1"
    step_path = str(path) if joint_resume else os.environ["JOINT_STEP_CHECKPOINT"]
    local = walking if self.gpu_global_rank == 0 or step_path == str(path) else torch.load(
      step_path, map_location="cpu", weights_only=False)
    actor, critic = self.alg.actor, self.alg.critic
    if joint_resume:
      counter = restore_joint_state(self.alg, walking, self.gpu_global_rank)
    else:
      actor.load_state_dict(walking["actor_state_dict"], strict=True)
      critic.load_state_dict(local["critic_state_dict"], strict=True)
      assert_same_state(actor.state_dict(), walking["actor_state_dict"])
      assert_same_state(critic.state_dict(), local["critic_state_dict"])
      counter = (local.get("infos") or {}).get("env_state", {}).get("common_step_counter", 0)
    restore_fixed_exploration(actor, self.cfg["actor"])
    if joint_resume:
      assert_same_state(actor.state_dict(), walking["actor_state_dict"])
    actor.update_normalization = lambda observations: None
    self.current_learning_iteration = int(walking["iter"]) + 1
    self.env.unwrapped.common_step_counter = counter
    with torch.inference_mode():
      self.env.reset()
    self.alg.task_weights = json.loads(os.environ["JOINT_TASK_WEIGHTS"])
    validate_task_weights(self.alg.task_weights)
    self.alg.task_weight = self.alg.task_weights[self.gpu_global_rank]
    self.alg.joint_updates = 0
    self.alg.joint_first_iteration = self.current_learning_iteration
    self.alg.joint_save_interval = self.cfg["save_interval"]
    self.alg.joint_evaluation_iteration = int(os.environ["JOINT_EVALUATION_ITERATION"])
    self.alg.joint_log_dir = Path(os.environ["JOINT_RUN_DIR"])
    self.alg.joint_env = self.env
    self.alg.joint_stage = (walking.get("infos") or {}).get("precision_stage", {})
    receipt = {"rank": self.gpu_global_rank, "actor_checkpoint": str(path),
      "critic_checkpoint": str(path) if self.gpu_global_rank == 0 else step_path,
      "actor_mean_exact": True, "critic_exact": True, "optimizer_reset": not joint_resume,
      "joint_resume": joint_resume, "optimizer_exact": joint_resume,
      "actor_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
      "fixed_std": .25, "actor_normalizer_frozen": True, "task_weight": self.alg.task_weight,
      "learning_rate": self.alg.learning_rate, "counter": counter}
    write_json(Path(self.logger.log_dir) / "initialization.json", receipt)
    print("JOINT_INITIALIZATION " + json.dumps(receipt), flush=True)
    return local.get("infos")

  def save(self, path, infos=None):
    return None

  def learn(self, num_learning_iterations, init_at_random_ep_len=False):
    super().learn(num_learning_iterations, init_at_random_ep_len=False)
    self.alg.save_joint()
    final = self.alg.joint_log_dir / f"model_{self.current_learning_iteration}.pt"
    status_path = Path(self.logger.log_dir) / f"status_rank_{self.gpu_global_rank}.json"
    status = json.loads(status_path.read_text())
    write_json(status_path, {**status, "final_checkpoint": str(final), "actor_sync_verified": True})


def worker(args, run_dir):
  rank = int(os.environ["RANK"])
  profile = "precision" if rank == 0 else "step-episode"
  os.environ["FOOTSTEP_PROFILE"] = profile
  os.environ["FOOTSTEP_CONTINUOUS"] = "1" if args.continuous else "0"
  os.environ["JOINT_RESUME"] = "1" if args.resume_joint else "0"
  os.environ["JOINT_STEP_CHECKPOINT"] = args.step_checkpoint or args.walking_checkpoint
  os.environ["JOINT_RUN_DIR"] = str(run_dir)
  os.environ["JOINT_TASK_WEIGHTS"] = json.dumps(args.task_weights)
  os.environ["JOINT_EVALUATION_ITERATION"] = str(args.evaluation_iteration or 0)
  os.environ["JOINT_DISTRIBUTED_BACKEND"] = args.distributed_backend
  task = train.FOOTSTEP_TASK if rank == 0 else train.STEP_EPISODE_TASK
  cfg = train.TrainConfig.from_task(task)
  from g1_lower_rl.tasks.footstep_tracking import make_footstep_env_cfg

  stage_config = None
  if rank == 0:
    saved = torch.load(args.walking_checkpoint, map_location="cpu", weights_only=False)
    stage_config = (saved.get("infos") or {}).get("precision_stage", {})
  cfg = replace(cfg, env=make_footstep_env_cfg(profile, precision_stage=stage_config),
    resume_checkpoint=args.walking_checkpoint, enable_nan_guard=True)
  cfg.env.scene.num_envs = args.envs_per_rank
  cfg.agent.resume = True
  cfg.agent.max_iterations = args.max_updates
  cfg.agent.save_interval = args.save_interval
  cfg.agent.algorithm.class_name = "train_footstep_joint:SharedActorPPO"
  cfg.agent.algorithm.learning_rate = args.learning_rate
  cfg.agent.algorithm.schedule = "fixed"
  cfg.agent.algorithm.entropy_coef = 0.
  cfg.agent.actor.distribution_cfg.update(init_std=.25, learn_std=False, std_range=(.25, .25))
  lane = run_dir / ("walking" if rank == 0 else "single_step")
  lane.mkdir(exist_ok=True)
  if args.continuous and not (lane / "STOP").is_symlink() and not (lane / "STOP").exists():
    (lane / "STOP").symlink_to(run_dir / "STOP")
  cfg.env.sim.nan_guard.output_dir = str(lane / "nan_guard")
  train.MjlabOnPolicyRunner = JointRunner
  try:
    train.run_train(task, cfg, lane)
  finally:
    if distributed.is_initialized():
      distributed.destroy_process_group()


def synchronization_test(rank, rendezvous, weights):
  distributed.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
  algorithm = object.__new__(SharedActorPPO)
  algorithm.actor = torch.nn.Linear(2, 1)
  algorithm.critic = torch.nn.Linear(2, 1)
  algorithm.task_weight = weights[rank]
  with torch.no_grad():
    for parameter in algorithm.actor.parameters():
      parameter.fill_(rank + 1.)
    for parameter in algorithm.critic.parameters():
      parameter.fill_(rank + 10.)
  algorithm.broadcast_parameters()
  for parameter in algorithm.actor.parameters():
    assert torch.equal(parameter, torch.ones_like(parameter))
    parameter.grad = torch.full_like(parameter, rank + 1.)
  for parameter in algorithm.critic.parameters():
    parameter.grad = torch.full_like(parameter, rank + 4.)
  algorithm.reduce_parameters()
  for parameter in algorithm.actor.parameters():
    torch.testing.assert_close(parameter.grad, torch.full_like(parameter, weights[0] + 2 * weights[1]))
  for parameter in algorithm.critic.parameters():
    assert torch.equal(parameter, torch.full_like(parameter, rank + 10.))
    assert torch.equal(parameter.grad, torch.full_like(parameter, rank + 4.))
  optimizer = torch.optim.Adam(list(algorithm.actor.parameters()) + list(algorithm.critic.parameters()), lr=1e-5)
  optimizer.step()
  algorithm.verify_actor()
  algorithm.optimizer = optimizer
  saved = {"actor_state_dict": algorithm.actor.state_dict(),
    "task_critic_state_dicts": {"walking": algorithm.critic.state_dict(), "single_step": algorithm.critic.state_dict()},
    "task_optimizer_state_dicts": {"walking": optimizer.state_dict(), "single_step": optimizer.state_dict()},
    "infos": {"shared_actor": True, "task_weights": [.7, .3], "task_environment_counters": [100, 200]}}
  restored = object.__new__(SharedActorPPO)
  restored.actor = torch.nn.Linear(2, 1)
  restored.critic = torch.nn.Linear(2, 1)
  restored.optimizer = torch.optim.Adam(list(restored.actor.parameters()) + list(restored.critic.parameters()), lr=1e-3)
  assert restore_joint_state(restored, saved, rank) == (100, 200)[rank]
  assert restored.learning_rate == 1e-5
  distributed.destroy_process_group()


def evaluate_joint(args, run_dir, check_completion=True):
  from footstep_campaign import aggregate_metrics, regression_reasons

  final = run_dir / f"model_{int(torch.load(args.walking_checkpoint, map_location='cpu', weights_only=False)['iter']) + args.max_updates}.pt"
  if check_completion:
    for rank, lane in enumerate(("walking", "single_step")):
      status = json.loads((run_dir / lane / f"status_rank_{rank}.json").read_text())
      if status["status"] != "completed" or status["completed_updates"] != args.max_updates:
        raise RuntimeError("Joint training did not finish the requested update budget")
      if Path(status["final_checkpoint"]) != final or not status["actor_sync_verified"]:
        raise RuntimeError("Joint final save or actor synchronization is unverified")
  evaluation = run_dir / "eval"
  evaluation.mkdir(exist_ok=True)
  summary = {}
  for rank, profile in enumerate(("precision", "step-episode")):
    metrics = {}
    for label, checkpoint in (("baseline", args.walking_checkpoint), ("trained", str(final))):
      reports = []
      for seed in (42, 43):
        destination = evaluation / f"{profile}_{label}_seed{seed}.json"
        command = [sys.executable, str(ROOT / "run_logs/probe_footstep_reward_balance.py"),
          checkpoint, "--source", str(ROOT), "--output", str(destination),
          "--profile", profile, "--num-envs", "16", "--steps",
          "1500" if rank == 0 else "600", "--seed", str(seed), "--deterministic", "--play", "--joint-diagnostics"]
        with destination.with_suffix(".log").open("w") as output:
          subprocess.run(command, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(rank)},
            stdout=output, stderr=subprocess.STDOUT, check=True, timeout=900)
        report = json.loads(destination.read_text())
        if not report["actor_exactly_loaded"]:
          raise RuntimeError("Evaluation actor was not exactly loaded")
        reports.append(report)
      metrics[label] = aggregate_metrics(reports)
    reasons = regression_reasons(metrics["trained"], metrics["baseline"])
    if rank == 1 and metrics["trained"]["completed"] < metrics["baseline"]["completed"]:
      reasons.append("single_step_completion")
    summary[profile] = {**metrics, "regressions": reasons}
    write_json(run_dir / "evaluation_summary.json", summary)
  print("JOINT_EVALUATION " + json.dumps(summary), flush=True)


def evaluate_when_saved(args, run_dir):
  status_path = run_dir / "evaluation_status.json"
  if status_path.exists() and json.loads(status_path.read_text())["status"] == "completed":
    return
  try:
    iteration = int(torch.load(args.walking_checkpoint, map_location="cpu", weights_only=False)["iter"]) + args.max_updates
    checkpoint = run_dir / f"model_{iteration}.pt"
    status = {"baseline": args.walking_checkpoint, "checkpoint": str(checkpoint), "updates": args.max_updates}
    write_json(status_path, {**status, "status": "waiting"})
    stop = threading.Event()
    while not checkpoint.exists():
      if (run_dir / "STOP").exists():
        write_json(status_path, {**status, "status": "cancelled"})
        return
      stop.wait(15)
    write_json(status_path, {**status, "status": "evaluating"})
    evaluate_joint(args, run_dir, check_completion=False)
    write_json(status_path, {**status, "status": "completed"})
  except Exception as error:
    write_json(status_path, {"status": "failed", "error": str(error)})
    logging.exception("Scheduled joint evaluation failed")


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--walking-checkpoint")
  parser.add_argument("--step-checkpoint", help="Optional pretrained step critic seed; defaults to the walking checkpoint")
  parser.add_argument("--resume-joint", type=Path)
  parser.add_argument("--continuous", action="store_true")
  parser.add_argument("--run-dir", type=Path)
  parser.add_argument("--envs-per-rank", type=int, default=1024)
  parser.add_argument("--max-updates", type=int, default=200)
  parser.add_argument("--save-interval", type=int, default=25)
  parser.add_argument("--learning-rate", type=float, default=1e-5)
  parser.add_argument("--task-weights", type=float, nargs=2, default=[.7, .3], metavar=("WALKING", "SINGLE_STEP"))
  parser.add_argument("--evaluate-at-updates", type=int)
  parser.add_argument("--distributed-backend", choices=("nccl", "gloo"), default="nccl")
  parser.add_argument("--evaluate-after", action="store_true")
  parser.add_argument("--self-test", action="store_true")
  args = parser.parse_args()
  if args.self_test:
    import tempfile
    from unittest.mock import patch
    for weights in ([.7, .3], [.4, .6]):
      with tempfile.TemporaryDirectory() as directory:
        torch.multiprocessing.spawn(synchronization_test, args=("file://" + directory + "/rendezvous", weights), nprocs=2)
    with tempfile.TemporaryDirectory() as directory:
      run_dir = Path(directory)
      baseline = run_dir / "baseline.pt"
      torch.save({"iter": 7}, baseline)
      torch.save({"iter": 257}, run_dir / "model_257.pt")
      evaluation_args = argparse.Namespace(walking_checkpoint=str(baseline), max_updates=250)
      with patch(__name__ + ".evaluate_joint") as evaluate:
        evaluate_when_saved(evaluation_args, run_dir)
        evaluate.assert_called_once_with(evaluation_args, run_dir, check_completion=False)
        assert json.loads((run_dir / "evaluation_status.json").read_text())["status"] == "completed"
        evaluate_when_saved(evaluation_args, run_dir)
        assert evaluate.call_count == 1
      for label, step_options in (("shared_seed", []), ("pretrained_step", ["--step-checkpoint", str(baseline)])):
        with patch.object(sys, "argv", [__file__, "--walking-checkpoint", str(baseline),
            "--run-dir", str(run_dir / label), *step_options]), patch.dict(os.environ), \
            patch.object(torchrunx, "Launcher") as launcher:
          main()
          launched_args = launcher.return_value.run.call_args.args[1]
          assert launched_args.walking_checkpoint == str(baseline)
          assert launched_args.step_checkpoint == (str(baseline) if step_options else None)
    print("SHARED_ACTOR_PRIVATE_CRITICS_TEST_PASSED", flush=True)
    return
  if args.resume_joint:
    if args.walking_checkpoint or args.step_checkpoint:
      parser.error("Joint resume cannot be combined with separate seed checkpoints")
    args.walking_checkpoint = str(args.resume_joint.resolve(strict=True))
  elif not args.walking_checkpoint:
    parser.error("A walking source checkpoint is required; the step source is optional")
  if args.continuous and args.evaluate_after:
    parser.error("Continuous training has no end-of-budget evaluation; use saved checkpoints for evaluation")
  try:
    validate_task_weights(args.task_weights)
  except ValueError as error:
    parser.error(str(error))
  if min(args.envs_per_rank, args.max_updates, args.save_interval) <= 0 or not 0 < args.learning_rate <= 1e-4:
    parser.error("Invalid training budget or learning rate")
  if args.evaluate_at_updates is not None and (not args.continuous or args.evaluate_at_updates <= 0
      or args.evaluate_at_updates % args.save_interval):
    parser.error("Scheduled evaluation requires continuous training and a positive multiple of the save interval")
  args.walking_checkpoint = str(Path(args.walking_checkpoint).resolve(strict=True))
  args.step_checkpoint = str(Path(args.step_checkpoint).resolve(strict=True)) if args.step_checkpoint else None
  args.resume_joint = str(args.resume_joint) if args.resume_joint else None
  run_dir = args.run_dir or ROOT / "logs/rsl_rl" / ("footstep_joint_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
  run_dir = run_dir.resolve()
  if (run_dir / "STOP").exists():
    print("Persistent STOP exists; joint training will not resume", flush=True)
    return
  if run_dir.exists() and args.resume_joint:
    candidates = sorted(run_dir.glob("model_*.pt"), key=lambda path: int(path.stem.split("_")[-1]))
    if candidates:
      args.walking_checkpoint = str(candidates[-1].resolve(strict=True))
  else:
    run_dir.mkdir(parents=True, exist_ok=False)
  args.evaluation_iteration = (int(torch.load(args.resume_joint or args.walking_checkpoint,
    map_location="cpu", weights_only=False)["iter"]) + args.evaluate_at_updates) if args.evaluate_at_updates else None
  write_json(run_dir / "launch.json", {**vars(args), "run_dir": str(run_dir),
    "task_weights": args.task_weights, "shared_actor": True, "separate_critics": True,
    "optimizer_reset": not bool(args.resume_joint), "fixed_std": .25, "source_root": str(ROOT)})
  os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
  os.environ["MUJOCO_GL"] = "egl"
  os.environ["TORCHRUNX_LOG_DIR"] = str(run_dir / "torchrunx")
  logging.basicConfig(level=logging.INFO)
  if args.evaluate_at_updates:
    evaluation_args = argparse.Namespace(**vars(args))
    evaluation_args.walking_checkpoint = args.resume_joint or args.walking_checkpoint
    evaluation_args.max_updates = args.evaluate_at_updates
    threading.Thread(target=evaluate_when_saved, args=(evaluation_args, run_dir), daemon=True).start()
  torchrunx.Launcher(hostnames=["localhost"], workers_per_host=2, backend=None,
    copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*", "OMP_NUM_THREADS", "PYTHONUNBUFFERED"),
    extra_env_vars={"PYTHONPATH": os.pathsep.join((str(ROOT), str(ROOT / "scripts")))},
  ).run(worker, args, run_dir)
  if args.evaluate_after:
    evaluate_joint(args, run_dir)


if __name__ == "__main__":
  main()