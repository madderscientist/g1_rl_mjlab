"""Bounded shared-actor recovery with physical, paired policy evaluation gates."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from footstep_campaign import aggregate_metrics, quality_score
from train_footstep_resume import assert_same_state, write_json

ROOT = Path(__file__).resolve().parents[1]


def recovery_seed(original, stage_checkpoint, learning_rate, yaw_rate_weight):
  if not math.isfinite(learning_rate) or not 0 < learning_rate <= 1e-5:
    raise ValueError("Recovery learning rate must lie in (0,1e-5]")
  if not math.isfinite(yaw_rate_weight) or not -0.01 <= yaw_rate_weight < 0:
    raise ValueError("Recovery yaw-rate weight must lie in [-0.01,0)")
  if not original["infos"].get("shared_actor"):
    raise ValueError("Recovery requires a complete joint checkpoint")
  result = copy.deepcopy(original)
  stage = copy.deepcopy(stage_checkpoint["infos"]["theta_stage"])
  stage["feet_contact_yaw_rate"] = {"weight": yaw_rate_weight, "yaw_rate_scale_deg_s": 1.}
  result["infos"]["theta_stage"] = stage
  for optimizer in [result["optimizer_state_dict"], *result["task_optimizer_state_dicts"].values()]:
    for group in optimizer["param_groups"]:
      group["lr"] = learning_rate
  assert_same_state(result["actor_state_dict"], original["actor_state_dict"])
  assert_same_state(result["task_critic_state_dicts"], original["task_critic_state_dicts"])
  return result


def rejection_reasons(candidate, reference):
  reasons = []
  for key, margin, ratio in (
    ("failures_per_minute", .3, 1.15),
    ("landing_xy_median_cm", 1., 1.15),
    ("landing_yaw_median_deg", 2., 1.15),
    ("torso_tilt_p90_deg", 2., 1.15),
    ("standing_joint_speed_rms_deg_s", 5., 1.2),
  ):
    value, baseline = candidate.get(key), reference.get(key)
    if value is None or not math.isfinite(value):
      reasons.append(key + "_missing")
    elif baseline is not None and value > max(baseline + margin, baseline * ratio):
      reasons.append(key)
  if candidate["landing_events"] < max(1, .8 * reference["landing_events"]):
    reasons.append("landing_events")
  if candidate["moving_fraction"] < max(.02, .8 * reference["moving_fraction"]):
    reasons.append("moving_fraction")
  if reference["completed"] and candidate["completed"] < .85 * reference["completed"]:
    reasons.append("single_step_completion")
  if reference["precision_success"]:
    reference_accuracy = reference["precision_success"] / max(1, reference["episodes"])
    candidate_accuracy = candidate["precision_success"] / max(1, candidate["episodes"])
    if candidate_accuracy + .03 < reference_accuracy:
      reasons.append("precision_success")
  return reasons


def source_digest():
  paths = sorted((ROOT / "g1_lower_rl").rglob("*.py"))
  paths += [ROOT / relative for relative in (
    "scripts/train.py", "scripts/train_footstep_resume.py", "scripts/train_footstep_joint.py",
    "scripts/recover_footstep_joint.py", "run_logs/probe_footstep_reward_balance.py")]
  digest = hashlib.sha256()
  for path in paths:
    digest.update(str(path.relative_to(ROOT)).encode())
    digest.update(path.read_bytes())
  return digest.hexdigest()


class Recovery:
  def __init__(self, args):
    self.args = args
    self.root = args.output.resolve()
    self.root.mkdir(parents=True, exist_ok=args.resume)
    self.lock = (self.root / "recovery.lock").open("w")
    fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    self.resuming = args.resume
    if args.resume:
      self.state = json.loads((self.root / "status.json").read_text())
      identity = {"baseline_source": str(args.checkpoint.resolve()),
        "stage_source": str(args.stage_checkpoint.resolve()), "yaw_rate_weight": args.yaw_rate_weight,
        "learning_rate": args.learning_rate}
      if any(self.state[key] != value for key, value in identity.items()):
        raise ValueError("Resume cannot change recovery source checkpoints or reward settings")
      if not self.state["baseline_metrics"]:
        raise ValueError("Resume requires completed baseline evaluations")
      baseline_envs = self.state.get("eval_envs")
      if baseline_envs is None:
        baseline_envs = json.loads((self.root / "eval/baseline/walking_mixed_seed42.json").read_text())["num_envs"]
      if args.eval_envs != baseline_envs:
        raise ValueError("Resume cannot change evaluation sample sizes")
      self.deadline = datetime.fromisoformat(self.state["deadline_utc"]).timestamp()
      previous = self.state["source_sha256"]
      current = source_digest()
      self.state.setdefault("explicit_source_revisions", []).append({
        "before": previous, "after": current, "time": datetime.now(timezone.utc).isoformat()})
      self.persist(status="resuming", source_sha256=current, pid=os.getpid(), child_pid=None,
        error=None, stop_reason=None)
      return
    self.deadline = time.time() + args.hours * 3600
    self.state = {"status": "initializing", "pid": os.getpid(), "source_sha256": source_digest(),
      "baseline_source": str(args.checkpoint.resolve()), "stage_source": str(args.stage_checkpoint.resolve()),
      "deadline_utc": datetime.fromtimestamp(self.deadline, timezone.utc).isoformat(),
      "yaw_rate_weight": args.yaw_rate_weight, "learning_rate": args.learning_rate,
      "eval_envs": args.eval_envs, "gate_version": 2,
      "continuous_stop_closing_probability": .5, "single_step_aligned_probability": .3,
      "baseline_metrics": {}, "best_metrics": {}, "best_checkpoint": None, "history": []}
    self.persist()

  def persist(self, **changes):
    self.state.update(changes, updated_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(self.root / "status.json", self.state)

  def execute(self, command, logfile, *, phase, timeout, training_run=None):
    if source_digest() != self.state["source_sha256"]:
      raise RuntimeError("Source changed during recovery; refusing mixed-code training")
    with logfile.open("w") as output:
      environment = {**os.environ, "OMP_NUM_THREADS": "1", "MUJOCO_GL": "egl",
        "PYTHONUNBUFFERED": "1", "PYTHONPATH": os.pathsep.join((str(ROOT), str(ROOT / "scripts")))}
      process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
        stdout=output, stderr=subprocess.STDOUT)
      self.persist(status=phase, child_pid=process.pid, active_log=str(logfile))
      limit = min(self.deadline, time.time() + timeout)
      while process.poll() is None:
        if (self.root / "STOP").exists() or time.time() >= limit:
          if training_run is not None:
            for lane in ("walking", "single_step"):
              directory = training_run / lane
              directory.mkdir(exist_ok=True)
              (directory / "STOP").write_text("Recovery supervisor requested a safe stop\n")
            try:
              process.wait(timeout=180)
            except subprocess.TimeoutExpired:
              for rank, lane in enumerate(("walking", "single_step")):
                path = training_run / lane / f"status_rank_{rank}.json"
                if path.exists():
                  status = json.loads(path.read_text())
                  if status["status"] == "running":
                    try:
                      os.kill(status["pid"], signal.SIGTERM)
                    except ProcessLookupError:
                      pass
              process.wait(timeout=120)
          else:
            process.terminate()
            process.wait(timeout=30)
          if (self.root / "STOP").exists() or time.time() >= self.deadline:
            raise InterruptedError("Recovery stopped at its persistent stop request or deadline")
          raise RuntimeError("Recovery command exceeded its bounded execution budget")
        time.sleep(5)
        self.persist()
      self.persist(child_pid=None)
      if process.returncode:
        raise RuntimeError(f"Command exited {process.returncode}; see {logfile}")

  def evaluate(self, checkpoint, label, *, seeds=(42, 43)):
    expected_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    directory = self.root / "eval" / label
    directory.mkdir(parents=True)
    metrics = {}
    cases = (("walking_mixed", "precision", .5), ("walking_open", "precision", 0.),
      ("walking_closed", "precision", 1.), ("single_step", "step-episode", None))
    for name, profile, probability in cases:
      reports = []
      for seed in seeds:
        destination = directory / f"{name}_seed{seed}.json"
        command = [sys.executable, str(ROOT / "run_logs/probe_footstep_reward_balance.py"),
          str(checkpoint), "--source", str(ROOT), "--output", str(destination),
          "--profile", profile, "--num-envs", str(self.args.eval_envs),
          "--steps", "1500" if profile == "precision" else "1000",
          "--seed", str(seed), "--deterministic", "--play", "--joint-diagnostics", "--gpu-trace"]
        if probability is not None:
          command += ["--stop-closing-probability", str(probability)]
        self.execute(command, destination.with_suffix(".log"), phase="evaluating", timeout=900)
        report = json.loads(destination.read_text())
        if not report["actor_exactly_loaded"] or report["actor_raw_obs_dim"] != 79:
          raise RuntimeError("Evaluation did not load the intended theta actor")
        if report["sha256"] != expected_hash:
          raise RuntimeError("Checkpoint changed during evaluation")
        if report["stop_closing_probability"] != (1. if probability is None else probability):
          raise RuntimeError("Evaluation stop distribution does not match its label")
        reports.append(report)
      metrics[name] = aggregate_metrics(reports)
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != expected_hash:
      raise RuntimeError("Checkpoint changed during evaluation")
    write_json(directory / "summary.json", metrics)
    return metrics

  def _keep_if_better(self, checkpoint, metrics):
    score = sum(quality_score(row) for row in metrics.values())
    best_score = sum(quality_score(row) for row in self.state["best_metrics"].values())
    if score < best_score:
      self.persist(best_checkpoint=str(checkpoint), best_metrics=metrics)

  def run(self):
    original = torch.load(self.args.checkpoint, map_location="cpu", weights_only=False)
    template = torch.load(self.args.stage_checkpoint, map_location="cpu", weights_only=False)
    seed_directory = self.root / "seed"
    seed_directory.mkdir(exist_ok=self.resuming)
    seed = seed_directory / f"model_{original['iter']}.pt"
    expected = recovery_seed(original, template, self.args.learning_rate, self.args.yaw_rate_weight)
    if self.resuming:
      assert_same_state(torch.load(seed, map_location="cpu", weights_only=False), expected)
      baseline = self.state["baseline_metrics"]
    else:
      torch.save(expected, seed)
      self.persist(best_checkpoint=str(seed), seed_actor_exact=True,
        seed_critic_exact=True, optimizer_momenta_preserved=True, learning_rate_intentionally_changed=True)
      baseline = self.evaluate(seed, "baseline")
    for name, values in baseline.items():
      if rejection_reasons(values, values):
        raise RuntimeError(f"Baseline lacks required physical evaluation coverage: {name}")
    if not self.resuming:
      self.persist(baseline_metrics=baseline, best_metrics=baseline, status="baseline_verified")
    current, learning_rate, rejected = seed, self.args.learning_rate, 0
    cycle = 0
    if self.resuming:
      self.persist(best_checkpoint=str(seed), best_metrics=baseline)
      reassessments = []
      for record in self.state["history"]:
        values = record["metrics"]
        reasons = {name: rejection_reasons(values[name], baseline[name]) for name in baseline}
        accepted = not any(reasons.values())
        reassessments.append({"cycle": record["cycle"], "accepted": accepted, "rejections": reasons})
        if accepted:
          self._keep_if_better(record["checkpoint"], values)
      current = Path(self.state["best_checkpoint"])
      cycle = max([int(path.name.split("_")[-1]) + 1 for path in self.root.glob("cycle_*")
        if path.is_dir()] + [self.state.get("cycle", 0)])
      self.persist(gate_version=2, gate_reassessments=reassessments, cycle=cycle)
    while time.time() + 3600 < self.deadline and not (self.root / "STOP").exists():
      disk = os.statvfs(self.root)
      if disk.f_bavail * disk.f_frsize < 30 * 1024**3:
        raise RuntimeError("Less than 30 GiB free; stopping without pruning original checkpoints")
      directory = self.root / f"cycle_{cycle:03d}"
      directory.mkdir()
      checkpoint = torch.load(current, map_location="cpu", weights_only=False)
      input_directory = directory / "input"
      input_directory.mkdir()
      source = input_directory / f"model_{checkpoint['iter']}.pt"
      torch.save(recovery_seed(checkpoint, template, learning_rate, self.args.yaw_rate_weight), source)
      run = directory / "training"
      command = [sys.executable, str(ROOT / "scripts/train_footstep_joint.py"),
        "--resume-joint", str(source), "--run-dir", str(run), "--max-updates", str(self.args.updates),
        "--save-interval", str(self.args.updates), "--envs-per-rank", "1024",
        "--task-weights", "0.4", "0.6", "--learning-rate", str(learning_rate)]
      self.execute(command, directory / "train.log", phase="training", timeout=1800, training_run=run)
      candidate = run / f"model_{checkpoint['iter'] + self.args.updates}.pt"
      for rank, lane in enumerate(("walking", "single_step")):
        status = json.loads((run / lane / f"status_rank_{rank}.json").read_text())
        if status["status"] != "completed" or not status["actor_sync_verified"]:
          raise RuntimeError("Both training ranks must finish and verify shared actor synchronization")
        if Path(status["final_checkpoint"]) != candidate or status["completed_updates"] != self.args.updates:
          raise RuntimeError("Training final checkpoint does not match its update budget")
        receipt = json.loads((run / lane / "initialization.json").read_text())
        if not receipt["actor_mean_exact"] or not receipt["critic_exact"] or not receipt["optimizer_exact"]:
          raise RuntimeError("Training did not exactly restore its isolated joint seed")
        if receipt["learning_rate"] != learning_rate:
          raise RuntimeError("Resume unexpectedly discarded the recovery learning rate")
        probability = receipt["command_config"]["manager"]["stop_closing_probability"]
        if probability != (1. if rank else .5):
          raise RuntimeError("Training profile discarded the intended stop configuration")
      trained = torch.load(candidate, map_location="cpu", weights_only=False)
      for name, value in original["actor_state_dict"].items():
        actual = trained["actor_state_dict"][name]
        if not torch.isfinite(actual).all():
          raise RuntimeError(f"Nonfinite trained actor tensor: {name}")
        if not name.startswith(("rnn.", "mlp.")) and not torch.equal(actual, value):
          raise RuntimeError(f"Frozen normalizer or deployment buffer changed: {name}")
      values = self.evaluate(candidate, f"cycle_{cycle:03d}")
      reasons = {name: rejection_reasons(values[name], baseline[name]) +
        rejection_reasons(values[name], self.state["best_metrics"][name]) for name in baseline}
      accepted = not any(reasons.values())
      self.state["history"].append({"cycle": cycle, "checkpoint": str(candidate), "accepted": accepted,
        "rejections": reasons, "metrics": values, "learning_rate": learning_rate})
      if accepted:
        current, rejected = candidate, 0
        self._keep_if_better(candidate, values)
      else:
        current = Path(self.state["best_checkpoint"])
        learning_rate *= .5
        rejected += 1
      cycle += 1
      self.persist(status="between_cycles", current_checkpoint=str(current), cycle=cycle)
      if rejected >= 3:
        self.persist(status="stopped_after_repeated_regression")
        return
    heldout = None
    if not (self.root / "STOP").exists() and time.time() + 1800 < self.deadline:
      reference = self.evaluate(seed, "heldout_baseline", seeds=(314,))
      heldout = self.evaluate(Path(self.state["best_checkpoint"]), "heldout_best", seeds=(314,))
      reasons = {name: rejection_reasons(heldout[name], reference[name]) for name in reference}
      self.persist(heldout_rejections=reasons)
      if any(reasons.values()):
        self.persist(best_checkpoint=str(seed), best_metrics=baseline,
          heldout_candidate_rejected=True)
    self.persist(status="completed" if not (self.root / "STOP").exists() else "stopped",
      heldout_metrics=heldout)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--stage-checkpoint", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--hours", type=float, default=12.)
  parser.add_argument("--updates", type=int, default=50)
  parser.add_argument("--eval-envs", type=int, default=16)
  parser.add_argument("--learning-rate", type=float, default=2e-6)
  parser.add_argument("--yaw-rate-weight", type=float, default=-.005)
  parser.add_argument("--resume", action="store_true", help="Explicitly resume the same directory without extending its deadline")
  args = parser.parse_args()
  if not math.isfinite(args.hours) or not 0 < args.hours <= 12 or not 1 <= args.updates <= 100 or args.eval_envs < 2:
    parser.error("Recovery requires <=12 hours, 1..100 updates per block, and at least two evaluation environments")
  recovery = Recovery(args)
  try:
    recovery.run()
  except InterruptedError as error:
    recovery.persist(status="stopped", stop_reason=str(error), child_pid=None)
  except Exception as error:
    recovery.persist(status="failed", error=f"{type(error).__name__}: {error}")
    raise
  finally:
    recovery.lock.close()


if __name__ == "__main__":
  main()
