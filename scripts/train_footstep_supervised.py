"""Run a bounded footstep continuation with checkpoint-based crash recovery."""

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import pickle
import selectors
import signal
import subprocess
import sys
import time

import torch


def finite_tree(value):
  if isinstance(value, torch.Tensor):
    return bool(torch.isfinite(value).all())
  if isinstance(value, dict):
    return all(finite_tree(item) for item in value.values())
  if isinstance(value, (list, tuple)):
    return all(finite_tree(item) for item in value)
  if isinstance(value, float):
    return math.isfinite(value)
  return True


def valid_checkpoint(path):
  try:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    return (path.stem == f"model_{saved['iter']}"
      and all(saved.get(key) and finite_tree(saved[key]) for key in (
        "actor_state_dict", "critic_state_dict", "optimizer_state_dict"))
      and "env_state" in saved.get("infos", {}) and "precision_stage" in saved["infos"])
  except (OSError, EOFError, RuntimeError, ValueError, KeyError, TypeError, pickle.UnpicklingError):
    return False


def select_checkpoint(root, initial):
  candidates = sorted((path for path in root.glob("*_attempt*/model_*.pt") if not path.parent.is_symlink()),
    key=lambda path: int(path.stem.split("_")[-1]), reverse=True)
  for path in candidates + [initial]:
    if valid_checkpoint(path):
      return path.resolve()
  raise RuntimeError("No complete finite checkpoint available for recovery")


def statuses(root, tag):
  records = []
  for path in root.glob(f"*_{tag}/status_rank_*.json"):
    if path.parent.is_symlink():
      continue
    records.append(json.loads(path.read_text()))
  return records


def completed_checkpoint(records):
  if len(records) != 2 or {entry.get("rank") for entry in records} != {0, 1}:
    return None
  if not all(entry["status"] in {"completed", "stopped"} for entry in records):
    return None
  paths = {entry.get("final_checkpoint") for entry in records}
  if len(paths) != 1 or None in paths:
    return None
  checkpoint = Path(paths.pop())
  return checkpoint.resolve() if valid_checkpoint(checkpoint) else None


def stop_process(process, records):
  for record in records:
    try:
      os.kill(record["pid"], signal.SIGTERM)
    except ProcessLookupError:
      pass
  if not records:
    os.killpg(process.pid, signal.SIGTERM)
  try:
    process.wait(timeout=60)
  except subprocess.TimeoutExpired:
    os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=30)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("checkpoint", type=Path)
  parser.add_argument("--run-root", type=Path, required=True)
  parser.add_argument("--hours", type=float, required=True)
  parser.add_argument("--save-interval", type=int, default=1000)
  parser.add_argument("--max-restarts", type=int, default=3)
  parser.add_argument("--profile", choices=("precision", "step-episode"), default="precision")
  args = parser.parse_args()
  if not math.isfinite(args.hours) or args.hours <= 0 or args.max_restarts < 0 or args.save_interval <= 0:
    parser.error("Hours and save interval must be positive; restarts must be nonnegative")
  source = Path(__file__).resolve().parents[1]
  root = args.run_root.resolve()
  root.mkdir(parents=True, exist_ok=True)
  initial = args.checkpoint.resolve(strict=True)
  if not valid_checkpoint(initial):
    raise RuntimeError("Invalid initial checkpoint")
  state_path = root / "supervisor.json"
  if state_path.exists():
    raise RuntimeError("Supervisor state already exists; refusing to reset its time budget")
  start = time.monotonic()
  duration = args.hours * 3600
  deadline = datetime.now(timezone.utc) + timedelta(seconds=duration)
  requested = False

  def request_stop(signum, frame):
    nonlocal requested
    requested = True

  for signum in (signal.SIGTERM, signal.SIGINT):
    signal.signal(signum, request_stop)

  def record(status, **fields):
    state = {"status": status, "pid": os.getpid(), "deadline_utc": deadline.isoformat(),
      "profile": args.profile,
      "remaining_s": max(0., duration - (time.monotonic() - start)),
      "updated_at_utc": datetime.now(timezone.utc).isoformat(), **fields}
    temporary = state_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(state_path)
    return state

  for attempt in range(args.max_restarts + 1):
    checkpoint = select_checkpoint(root, initial)
    remaining = duration - (time.monotonic() - start)
    if requested or remaining <= 0:
      record("stopped" if requested else "completed", final_checkpoint=str(checkpoint))
      return
    reference = root / checkpoint.parent.name
    if checkpoint.parent.parent != root and not reference.exists():
      reference.symlink_to(checkpoint.parent, target_is_directory=True)
    tag = f"recovered_attempt{attempt}"
    command = [sys.executable, str(source / "scripts/train_footstep_resume.py"), str(checkpoint),
      "--profile", args.profile, "--hours", str(remaining / 3600), "--save-interval", str(args.save_interval),
      "--envs-per-rank", "128", "--run-root", str(root), "--tag", tag]
    process = subprocess.Popen(command, cwd=source, stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT, start_new_session=True)
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    progress = -1
    progress_time = time.monotonic()
    failure = None
    records = []
    record("running", attempt=attempt, launcher_pid=process.pid, checkpoint=str(checkpoint))
    with (root / f"attempt{attempt}.log").open("ab", buffering=0) as output:
      while process.poll() is None:
        for key, _ in selector.select(timeout=5):
          chunk = os.read(key.fileobj.fileno(), 65536)
          if chunk:
            output.write(chunk)
          else:
            selector.unregister(key.fileobj)
        records = statuses(root, tag)
        current = min((entry["completed_updates"] for entry in records), default=-1)
        if current != progress:
          progress, progress_time = current, time.monotonic()
          record("running", attempt=attempt, launcher_pid=process.pid,
            checkpoint=str(checkpoint), completed_updates=progress)
        failure = next((entry.get("error", "worker failed") for entry in records if entry["status"] == "failed"), None)
        if time.monotonic() - progress_time > 300:
          failure = "No PPO update progress for 300 seconds"
        if requested or time.monotonic() - start >= duration or failure:
          stop_process(process, records)
          break
      while True:
        chunk = process.stdout.read(65536)
        if not chunk:
          break
        output.write(chunk)
    selector.close()
    process.stdout.close()
    records = statuses(root, tag)
    checkpoint = select_checkpoint(root, initial)
    final = completed_checkpoint(records)
    complete = final is not None and all(entry["status"] == "completed" and entry.get("reason") == "duration_reached" for entry in records)
    if requested:
      record("stopped", attempt=attempt, final_checkpoint=str(final or checkpoint))
      return
    if complete or time.monotonic() - start >= duration:
      if final is None:
        record("failed", attempt=attempt, error="Deadline reached without a verified final save", final_checkpoint=str(checkpoint))
        raise RuntimeError("Deadline reached without a verified final save")
      record("completed", attempt=attempt, final_checkpoint=str(final))
      return
    record("recovering", attempt=attempt, returncode=process.returncode, error=failure,
      checkpoint=str(checkpoint))
  record("failed", error="Restart budget exhausted", final_checkpoint=str(checkpoint))
  raise RuntimeError("Restart budget exhausted")


if __name__ == "__main__":
  main()