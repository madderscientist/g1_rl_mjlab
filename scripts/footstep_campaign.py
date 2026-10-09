"""限期脚步双路线训练的持久监督、数值验收与独立定时巡检"""

import argparse
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import sys
import time
import uuid

import torch

from train_footstep_supervised import valid_checkpoint


def now_utc():
  return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
  temporary.replace(path)


def read_json(path):
  try:
    return json.loads(path.read_text())
  except (OSError, ValueError):
    return None


def drain_pipe(stream, output, seconds=2.):
  deadline = time.monotonic() + seconds
  with selectors.DefaultSelector() as selector:
    selector.register(stream, selectors.EVENT_READ)
    while time.monotonic() < deadline:
      for key, _ in selector.select(timeout=min(.2, max(0., deadline - time.monotonic()))):
        chunk = os.read(key.fileobj.fileno(), 65536)
        if not chunk:
          return
        output.write(chunk)


def aggregate_metrics(reports):
  values = [report["campaign_metrics"] for report in reports]
  seconds = sum(value["environment_seconds"] for value in values)
  result = {key: sum(value[key] for value in values) for key in
    ("environment_seconds", "failures", "episodes", "completed", "precision_success", "landing_events")}
  result["failures_per_minute"] = result["failures"] * 60 / seconds
  for key in ("landing_xy_median_cm", "landing_yaw_median_deg", "torso_tilt_p90_deg",
              "standing_joint_speed_rms_deg_s", "terminal_xy_median_cm", "moving_fraction"):
    finite = [value[key] for value in values if value[key] is not None and math.isfinite(value[key])]
    result[key] = sum(finite) / len(finite) if finite else None
  return result


def quality_score(metrics):
  required = ("landing_xy_median_cm", "landing_yaw_median_deg", "torso_tilt_p90_deg")
  if any(metrics.get(key) is None for key in required):
    return 1000000.
  return (metrics["landing_xy_median_cm"] + .2 * metrics["landing_yaw_median_deg"]
    + .2 * metrics["torso_tilt_p90_deg"] + .05 * (metrics.get("standing_joint_speed_rms_deg_s") or 0.)
    + 20 * metrics["failures_per_minute"])


def regression_reasons(candidate, reference):
  reasons = []
  for key, margin, ratio in (
    ("failures_per_minute", .5, 1.5), ("landing_xy_median_cm", 5., 1.35),
    ("torso_tilt_p90_deg", 5., 1.4), ("standing_joint_speed_rms_deg_s", 10., 1.5),
  ):
    current, old = candidate.get(key), reference.get(key)
    if current is not None and old is not None and current > max(old + margin, old * ratio):
      reasons.append(key)
  if reference["landing_events"] and not candidate["landing_events"]:
    reasons.append("no_landing_events")
  if candidate["moving_fraction"] < .02:
    reasons.append("no_movement")
  return reasons


def prune_owned(root, protected, owned_runs, keep_latest=8):
  root = root.resolve()
  protected = {Path(path).resolve() for path in protected if path}
  files = [path for run in owned_runs for path in Path(run).glob("model_*.pt")
    if not Path(run).is_symlink() and Path(run).resolve().is_relative_to(root / "runs")
    and not path.is_symlink() and path.resolve().is_relative_to(root)]
  files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
  keep = set(files[:keep_latest])
  daily = set()
  for path in files:
    day = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).date()
    if day not in daily:
      keep.add(path)
      daily.add(day)
  removed = []
  for path in files:
    if path not in keep and path.resolve() not in protected and time.time() - path.stat().st_mtime > 3600:
      removed.append({"path": str(path), "bytes": path.stat().st_size})
      path.unlink()
  return removed


def resource_snapshot(gpu, root):
  result = {"free_disk_gb": shutil.disk_usage(root).free / 1024**3}
  result["load_average"] = list(os.getloadavg())
  for line in Path("/proc/meminfo").read_text().splitlines():
    if line.startswith("MemAvailable:"):
      result["available_memory_gb"] = int(line.split()[1]) / 1024**2
  try:
    output = subprocess.run(["nvidia-smi", "-i", str(gpu),
      "--query-gpu=utilization.gpu,memory.used,memory.free,temperature.gpu", "--format=csv,noheader,nounits"],
      capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    result["gpu"] = output
  except (OSError, subprocess.SubprocessError) as error:
    result["gpu_query_error"] = str(error)
  return result


class Campaign:
  def __init__(self, args):
    self.args = args
    self.root = args.root.resolve()
    self.root.mkdir(parents=True, exist_ok=True)
    self.lock = (self.root / "campaign.lock").open("a")
    fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    self.source = Path(__file__).resolve().parents[1]
    self.deadline = datetime.fromisoformat(args.deadline).timestamp()
    self.initial = args.checkpoint.resolve(strict=True)
    self.path = self.root / "state.json"
    self.state = json.loads(self.path.read_text()) if self.path.exists() else None
    identity = {"profile": args.profile, "gpu": args.gpu, "deadline": args.deadline,
      "initial_checkpoint": str(self.initial), "source": str(self.source)}
    if self.state:
      if any(self.state.get(key) != value for key, value in identity.items()):
        raise ValueError("Campaign identity or deadline cannot change on restart")
    else:
      if not valid_checkpoint(self.initial):
        raise ValueError("Initial checkpoint is invalid")
      self.state = {**identity, "phase": "initializing", "cycle": 0, "envs": args.envs,
        "current_checkpoint": str(self.initial), "best_checkpoint": str(self.initial),
        "best_metrics": None, "next_lr_scale": 1., "regressions": 0, "failures": 0,
        "created_at_utc": now_utc(), "eval_history": [], "owned_tags": [], "pending_evaluation": None}
    self.requested = False
    self.process = None
    for signum in (signal.SIGTERM, signal.SIGINT):
      signal.signal(signum, self.request_stop)
    self.persist()

  def request_stop(self, signum, frame):
    self.requested = True

  def persist(self, **fields):
    self.state.update(fields)
    self.state.update(pid=os.getpid(), updated_at_utc=now_utc(),
      remaining_s=max(0., self.deadline - time.time()))
    write_json(self.path, self.state)

  def log(self, message, **fields):
    record = {"time": now_utc(), "message": message, **fields}
    with (self.root / "patrol.jsonl").open("a") as output:
      output.write(json.dumps(record, allow_nan=False) + "\n")
    with (self.root / "RUN_LOG.md").open("a") as output:
      output.write(f"- {record['time']} {message}: {json.dumps(fields, ensure_ascii=False, allow_nan=False)}\n")
    print(json.dumps(record, ensure_ascii=False), flush=True)

  def stop_requested(self):
    return self.requested or (self.root / "STOP").exists() or (self.root.parent / "STOP").exists()

  def run_records(self, tag):
    paths = [path for path in (self.root / "runs").glob(f"*_{tag}/status_rank_0.json")
      if not path.parent.is_symlink()]
    records = [(path, read_json(path)) for path in sorted(paths) if read_json(path)]
    if len(records) > 1:
      raise RuntimeError("Ambiguous attempt identity")
    return records

  def newest_checkpoint(self, tag, fallback):
    candidates = [path for path in (self.root / "runs").glob(f"*_{tag}/model_*.pt")
      if not path.parent.is_symlink() and not path.is_symlink()]
    candidates.sort(key=lambda path: int(path.stem.split("_")[-1]), reverse=True)
    for path in candidates + [Path(fallback)]:
      if valid_checkpoint(path):
        return path.resolve()
    raise RuntimeError("No finite checkpoint remains")

  def check_std(self, checkpoint):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    std = saved["actor_state_dict"]["distribution.std_param"]
    if not torch.isfinite(std).all() or (std <= 0).any():
      raise RuntimeError("Invalid exploration standard deviation")
    if self.args.profile == "step-episode":
      torch.testing.assert_close(std, torch.full_like(std, .25), rtol=0, atol=0)
    elif std.mean() > 2. or std.max() > 4.:
      raise RuntimeError("Exploration standard deviation exceeded safety budget")
    return float(std.mean())

  def stop_child(self, process):
    try:
      os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
      pass
    selector = selectors.DefaultSelector()
    if process.stdout is not None and not process.stdout.closed:
      selector.register(process.stdout, selectors.EVENT_READ)
    try:
      deadline = time.monotonic() + max(120., min(600., 3 * self.state.get("seconds_per_update", 30.) + 60))
      with Path(self.state["active_log"]).open("ab", buffering=0) as output:
        while process.poll() is None and time.monotonic() < deadline:
          for key, _ in selector.select(timeout=1):
            chunk = os.read(key.fileobj.fileno(), 65536)
            if chunk:
              output.write(chunk)
            else:
              selector.unregister(key.fileobj)
        try:
          os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
          pass
        process.wait(timeout=30)
    finally:
      selector.close()

  def execute(self, command, logfile, phase, tag=None, timeout_s=1800):
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(self.args.gpu),
      OMP_NUM_THREADS="1", MUJOCO_GL="egl", PYTHONUNBUFFERED="1", PYTHONPATH=str(self.source))
    process = subprocess.Popen(command, cwd=self.source, env=environment, stdin=subprocess.DEVNULL,
      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    self.process = process
    self.persist(phase=phase, child_pid=process.pid, active_tag=tag, active_log=str(logfile))
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    started = last_progress = time.monotonic()
    next_status = started
    next_patrol = started
    progress = -1
    error = None
    deadline_stop = False
    try:
      with logfile.open("ab", buffering=0) as output:
        while process.poll() is None:
          for key, _ in selector.select(timeout=5):
            chunk = os.read(key.fileobj.fileno(), 65536)
            if chunk:
              output.write(chunk)
            else:
              selector.unregister(key.fileobj)
          current_time = time.monotonic()
          if current_time >= next_status:
            records = self.run_records(tag) if tag else []
            if records:
              status = records[-1][1]
              current = status.get("completed_updates", -1)
              if current != progress:
                progress, last_progress = current, current_time
              if status.get("status") == "failed":
                error = status.get("error", "Training worker failed")
              self.persist(completed_updates=current, last_iteration=status.get("last_completed_iteration"),
                seconds_per_update=status.get("seconds_per_update") or 30.)
            next_status = current_time + 10
          if current_time >= next_patrol:
            resources = resource_snapshot(self.args.gpu, self.root)
            checkpoint = self.newest_checkpoint(tag, self.state["current_checkpoint"]) if tag else Path(self.state["current_checkpoint"])
            try:
              std = self.check_std(checkpoint)
            except (RuntimeError, AssertionError) as exception:
              error = str(exception)
              std = None
            if tag and progress > 2 * self.args.save_interval:
              latest_iteration = int(checkpoint.stem.split("_")[-1])
              if self.state.get("last_iteration", latest_iteration) - latest_iteration > 2 * self.args.save_interval:
                error = "PPO progresses but no new finite checkpoint is being saved"
            self.persist(resources=resources, checkpoint_seen=str(checkpoint))
            self.log("定时巡检", phase=phase, progress=progress, log_bytes=logfile.stat().st_size,
              checkpoint=str(checkpoint), std=std, **resources)
            if resources["free_disk_gb"] < self.args.min_free_gb:
              error = "Low disk space"
            next_patrol = current_time + self.args.patrol_seconds
          if tag and current_time - last_progress > (900 if progress < 1 else 300):
            error = "No PPO progress within watchdog limit"
          if not tag and current_time - started > timeout_s:
            error = "Evaluation timed out"
          if self.stop_requested() or time.time() >= self.deadline or error:
            deadline_stop = self.stop_requested() or time.time() >= self.deadline
            self.stop_child(process)
            break
        drain_pipe(process.stdout, output)
    finally:
      selector.close()
      self.stop_child(process)
      process.stdout.close()
      self.process = None
      self.persist(child_pid=None)
    if error:
      raise RuntimeError(error)
    if deadline_stop:
      return False
    if process.returncode:
      raise RuntimeError(f"Child exited with status {process.returncode}; see {logfile}")
    return True

  def evaluate(self, checkpoint, name):
    directory = self.root / "eval" / name
    directory.mkdir(parents=True, exist_ok=True)
    reports = []
    steps = self.args.eval_steps or (1500 if self.args.profile == "precision" else 600)
    for seed in (42, 43):
      destination = directory / f"seed{seed}.json"
      existing = read_json(destination)
      if (existing and existing.get("checkpoint") == str(checkpoint) and existing.get("steps") == steps
          and existing.get("num_envs") == self.args.eval_envs and existing.get("profile") == self.args.profile
          and existing.get("actor_exactly_loaded")
          and existing.get("sha256") == hashlib.sha256(checkpoint.read_bytes()).hexdigest()):
        reports.append(existing)
        continue
      command = [sys.executable, str(self.source / "run_logs/probe_footstep_reward_balance.py"),
        str(checkpoint), "--source", str(self.source), "--output", str(destination),
        "--profile", self.args.profile, "--num-envs", str(self.args.eval_envs), "--steps", str(steps),
        "--seed", str(seed), "--deterministic", "--play", "--joint-diagnostics"]
      if not self.execute(command, directory / f"seed{seed}.log", "evaluating"):
        return None
      report = read_json(destination)
      if not report or not report.get("actor_exactly_loaded") or report["profile"] != self.args.profile:
        raise RuntimeError("Evaluation identity check failed")
      reports.append(report)
    metrics = aggregate_metrics(reports)
    write_json(directory / "summary.json", metrics)
    self.log("数值验收完成", checkpoint=str(checkpoint), metrics=metrics)
    return metrics

  def cleanup(self):
    owned_runs = [path for tag in self.state.get("owned_tags", []) for path in (self.root / "runs").glob(f"*_{tag}")]
    deleted = prune_owned(self.root, [self.initial, self.state["best_checkpoint"], self.state["current_checkpoint"]], owned_runs)
    if deleted:
      self.log("清理自产中间检查点", removed=deleted)
    directories = sorted((self.root / "eval").glob("cycle*"), key=lambda path: path.name)
    protected = {self.state.get("best_eval"), *[path.name for path in directories[-2:]]}
    for directory in directories:
      if directory.name in protected or directory.is_symlink() or not directory.resolve().is_relative_to(self.root):
        continue
      for path in directory.glob("*.npz"):
        if not path.is_symlink():
          path.unlink()

  def run(self):
    if self.state.get("phase") in {"completed", "blocked"}:
      return
    if self.state.get("phase") == "training" and self.state.get("active_tag"):
      recovered = self.newest_checkpoint(self.state["active_tag"], self.state["current_checkpoint"])
      self.persist(current_checkpoint=str(recovered), next_lr_scale=1., phase="recovering")
      self.log("恢复中断训练的有限检查点", checkpoint=str(recovered))
    try:
      while time.time() < self.deadline and not self.stop_requested():
        if shutil.disk_usage(self.root).free / 1024**3 < self.args.min_free_gb:
          self.cleanup()
          self.persist(phase="waiting_for_disk")
          self.log("磁盘不足，暂缓新作业", **resource_snapshot(self.args.gpu, self.root))
          if self.args.max_cycles:
            raise RuntimeError("Insufficient disk for smoke run")
          with selectors.DefaultSelector() as wait:
            wait.select(timeout=60)
          continue
        if self.state["best_metrics"] is None:
          try:
            metrics = self.evaluate(self.initial, "baseline")
          except (OSError, RuntimeError, ValueError, AssertionError) as error:
            count = self.state["failures"] + 1
            self.log("基准评测失败", error=str(error), consecutive_failures=count)
            self.persist(failures=count, phase="recovering")
            if count >= 3:
              self.persist(phase="blocked", error=str(error))
              return
            continue
          if metrics is None:
            break
          self.persist(best_metrics=metrics, baseline_metrics=metrics, best_eval="baseline")
        cycle = self.state["cycle"]
        pending = self.state.get("pending_evaluation")
        tag = pending["tag"] if pending else f"cycle{cycle:05d}_{uuid.uuid4().hex[:10]}"
        checkpoint = Path(pending["checkpoint"] if pending else self.state["current_checkpoint"])
        runs = self.root / "runs"
        runs.mkdir(exist_ok=True)
        reference = runs / checkpoint.parent.name
        if checkpoint.parent.parent != runs:
          if not reference.exists():
            reference.symlink_to(checkpoint.parent, target_is_directory=True)
          elif reference.resolve() != checkpoint.parent:
            raise RuntimeError("Input checkpoint reference collides with another experiment")
        block_seconds = min(self.args.block_seconds, max(1., self.deadline - time.time()))
        if cycle == 0 or self.state["regressions"]:
          block_seconds = min(block_seconds, 900.)
        command = [sys.executable, str(self.source / "scripts/train_footstep_resume.py"), str(checkpoint),
          "--profile", self.args.profile, "--gpu-ids", str(self.args.gpu), "--envs-per-rank", str(self.state["envs"]),
          "--run-root", str(runs), "--tag", tag, "--save-interval", str(self.args.save_interval),
          "--learning-rate-scale", str(self.state["next_lr_scale"])]
        command += (["--max-updates", str(self.args.smoke_updates)] if self.args.smoke_updates else
          ["--hours", str(block_seconds / 3600)])
        self.log("继续未完成评测" if pending else "启动训练阶段", cycle=cycle, checkpoint=str(checkpoint), envs=self.state["envs"],
          learning_rate_scale=self.state["next_lr_scale"], seconds=block_seconds)
        try:
          if pending:
            candidate = checkpoint
          else:
            self.persist(owned_tags=self.state.get("owned_tags", []) + [tag])
            finished = self.execute(command, self.root / f"train_{tag}.log", "training", tag)
            candidate = self.newest_checkpoint(tag, checkpoint)
            self.persist(current_checkpoint=str(candidate), next_lr_scale=1.)
            if not finished:
              break
            records = self.run_records(tag)
            if not records or records[-1][1].get("status") != "completed":
              raise RuntimeError("Training exited without a completed save")
            final_path = Path(records[-1][1]["final_checkpoint"])
            if final_path.parent != records[-1][0].parent or not valid_checkpoint(final_path):
              raise RuntimeError("Reported final checkpoint is missing, invalid or outside the owned run")
            candidate = final_path.resolve()
            self.persist(current_checkpoint=str(candidate))
            self.persist(pending_evaluation={"tag": tag, "checkpoint": str(candidate)})
          self.check_std(candidate)
          metrics = self.evaluate(candidate, tag)
          if metrics is None:
            break
          reasons = regression_reasons(metrics, self.state["best_metrics"])
          if reasons:
            count = self.state["regressions"] + 1
            self.persist(current_checkpoint=self.state["best_checkpoint"], regressions=count,
              next_lr_scale=max(.125, .5 ** min(count, 3)))
            self.log("质量退化，回退最佳检查点并降学习率", reasons=reasons,
              candidate=str(candidate), retained=self.state["best_checkpoint"], next_lr_scale=self.state["next_lr_scale"])
          else:
            if quality_score(metrics) < quality_score(self.state["best_metrics"]) * .98:
              self.persist(best_checkpoint=str(candidate), best_metrics=metrics, best_eval=tag)
              self.log("更新最佳验收检查点", checkpoint=str(candidate), metrics=metrics)
            self.persist(regressions=0)
          history = self.state["eval_history"] + [{"cycle": cycle, "checkpoint": str(candidate),
            "metrics": metrics, "regression": reasons, "time": now_utc()}]
          self.persist(cycle=cycle + 1, failures=0, eval_history=history, phase="between_stages", pending_evaluation=None)
          self.cleanup()
          logfile = self.root / f"train_{tag}.log"
          if logfile.stat().st_size > 5 * 1024**2:
            with logfile.open("rb") as source, gzip.open(logfile.with_suffix(".log.gz"), "wb") as target:
              shutil.copyfileobj(source, target)
            logfile.unlink()
        except (OSError, RuntimeError, ValueError, AssertionError) as error:
          count = self.state["failures"] + 1
          self.log("作业异常，保守恢复", error=str(error), consecutive_failures=count)
          self.persist(failures=count, cycle=cycle + 1, current_checkpoint=self.state["best_checkpoint"],
            envs=max(128, self.state["envs"] // 2), next_lr_scale=.5, phase="recovering", pending_evaluation=None)
          self.cleanup()
          if count >= 6:
            self.persist(phase="blocked", error=str(error))
            self.log("连续恢复失败，停止本路避免空转", error=str(error))
            return
        if self.args.max_cycles and self.state["cycle"] >= self.args.max_cycles:
          break
      phase = ("paused" if (self.root / "STOP").exists() or (self.root.parent / "STOP").exists()
        else "interrupted" if self.requested else "completed")
      self.persist(phase=phase)
      self.log("本路结束", phase=phase, best_checkpoint=self.state["best_checkpoint"],
        current_checkpoint=self.state["current_checkpoint"], cycles=self.state["cycle"])
    finally:
      if self.process is not None:
        self.stop_child(self.process)
      self.lock.close()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--root", type=Path, required=True)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--profile", choices=("precision", "step-episode"), required=True)
  parser.add_argument("--gpu", type=int, required=True)
  parser.add_argument("--deadline", required=True)
  parser.add_argument("--envs", type=int, default=512)
  parser.add_argument("--block-seconds", type=float, default=3600)
  parser.add_argument("--patrol-seconds", type=float, default=300)
  parser.add_argument("--save-interval", type=int, default=250)
  parser.add_argument("--eval-envs", type=int, default=16)
  parser.add_argument("--eval-steps", type=int)
  parser.add_argument("--min-free-gb", type=float, default=15.)
  parser.add_argument("--max-cycles", type=int, default=0)
  parser.add_argument("--smoke-updates", type=int, default=0)
  args = parser.parse_args()
  if min(args.envs, args.block_seconds, args.patrol_seconds, args.save_interval, args.eval_envs) <= 0:
    parser.error("Resource counts and intervals must be positive")
  if datetime.fromisoformat(args.deadline).tzinfo is None:
    parser.error("Deadline must include a timezone")
  Campaign(args).run()


if __name__ == "__main__":
  main()