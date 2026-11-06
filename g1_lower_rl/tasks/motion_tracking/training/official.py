"""Public MLP + position-WM SuperMimic training entrypoint."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from g1_lower_rl.tasks.motion_tracking.corpus import file_sha256

from .position_policy import actor_architecture_metadata
from .position_runner import FORMAT, PositionTrainingConfig
from .position_task import RECOVERY_XY_TASK


def load_mlp_checkpoint(path: Path) -> dict:
  if path.is_dir():
    path = path / "rank_0.pt"
  saved = torch.load(path, map_location="cpu", weights_only=True)
  if (
    saved.get("format") != FORMAT
    or saved["config"].get("actor_architecture") != "mlp"
    or saved["origin"]["architecture"] != actor_architecture_metadata()
    or saved["collector"].get("contract_version") != 3
  ):
    raise ValueError("The official mocap task requires a complete position-v3 MLP checkpoint, not PPO/RGMT/actor-only weights")
  return saved


def frozen_runtime(checkpoint: Path, saved: dict) -> Path:
  for parent in checkpoint.resolve().parents:
    runtime = parent / "runtime"
    if (runtime / "g1_lower_rl/tasks/motion_tracking/training/position_runner.py").is_file():
      for name, digest in saved["source_provenance"]["files"].items():
        path = runtime / name
        if not path.is_file() or file_sha256(path) != digest:
          raise ValueError(f"Frozen training source changed or is missing: {path}")
      return runtime
  raise FileNotFoundError("Exact continuation requires the checkpoint's verified runtime/ source snapshot")


@dataclass(frozen=True)
class TrainingConfig:
  output: str
  resume: str | None = None
  """Complete checkpoint directory (all ranks), or a rank checkpoint. Settings are inherited."""
  world_model: str | None = None
  replica_source: str | None = None
  mocap_dir: str | None = None
  iterations: int = 0
  """Total iteration target; 0 means unlimited. No training is started by export or preflight."""
  device: str | None = None
  num_envs: int | None = None
  validation_envs: int | None = None
  actor_batch_size: int | None = None
  actor_microbatch: int | None = None
  wm_batch_size: int | None = None
  wm_microbatch: int | None = None
  preflight: bool = False
  """Strictly restore and check every model/Adam/sampler without any optimizer update."""


def training_config(args: TrainingConfig, saved: dict | None = None) -> dict:
  if args.iterations < 0:
    raise ValueError("iterations must be nonnegative")
  if saved is None:
    if args.resume or args.preflight:
      raise ValueError("Resume/preflight requires a complete parent checkpoint")
    if not args.world_model or not args.replica_source:
      raise ValueError("Fresh mocap training requires --world-model and --replica-source")
    base = PositionTrainingConfig(
      world_model=args.world_model, replica_source=args.replica_source, output=args.output,
    )
    values = asdict(base)
  else:
    values = dict(saved["config"])
    if (values["actor_architecture"] != "mlp" or values.get("real_ppo", False) or not values["native_actor_kl"]
        or values["task_definition"] != RECOVERY_XY_TASK):
      raise ValueError("Official continuation requires MLP + recovery_xy_light_v1 + native actor KL, with PPO disabled")
    # Saved one-shot switches must not execute a second time in the artifact's runtime.
    for flag in values.keys() & {"actor_restart", "performance_resume", "task_resume", "hybrid_resume", "ppo_ablation_resume"}:
      values[flag] = None if flag == "actor_restart" else False
  for name in (
    "world_model", "replica_source", "mocap_dir", "device", "num_envs", "validation_envs",
    "actor_batch_size", "actor_microbatch", "wm_batch_size", "wm_microbatch",
  ):
    value = getattr(args, name)
    if value is not None:
      if saved is not None and value != values[name]:
        raise ValueError(f"Strict continuation cannot change {name}; use the saved configuration")
      values[name] = value
  values.update(output=str(Path(args.output).resolve()), resume=args.resume, iterations=args.iterations)
  if saved is None:
    PositionTrainingConfig(**values).validate()
  return values


def run_training(args: TrainingConfig) -> None:
  if not args.resume:
    cfg = training_config(args)
    runtime = Path(cfg["output"]) / "runtime"
    ready = runtime / "READY"
    if int(os.environ.get("RANK", "0")) == 0:
      if runtime.exists() or (Path(cfg["output"]) / "config.json").exists():
        raise FileExistsError("Fresh training requires a new output directory; use --resume to continue")
      repository = Path(__file__).resolve().parents[4]
      shutil.copytree(
        repository / "g1_lower_rl", runtime / "g1_lower_rl", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
      )
      shutil.copytree(repository / "scripts", runtime / "scripts", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
      ready.write_text("Complete frozen runtime; do not edit.\n")
    else:
      deadline = time.monotonic() + 120
      while not ready.is_file():
        if time.monotonic() > deadline:
          raise TimeoutError("Rank 0 did not finish freezing the mocap runtime")
        time.sleep(.2)
    launch_worker(cfg, runtime, preflight=False)
    return
  checkpoint = Path(args.resume).resolve(strict=True)
  if checkpoint.is_dir():
    checkpoint /= "rank_0.pt"
  saved = load_mlp_checkpoint(checkpoint)
  runtime = frozen_runtime(checkpoint, saved)
  cfg = training_config(args, saved)
  if Path(cfg["output"]) in (checkpoint.parent, runtime.parent):
    raise ValueError("Continuation/preflight must use a new output directory, not overwrite the preserved result")
  if checkpoint.name != "rank_0.pt" or saved["rank"] != 0:
    raise ValueError("Pass the complete checkpoint directory or rank_0.pt, not another rank")
  for rank in range(saved["world_size"]):
    if not (checkpoint.parent / f"rank_{rank}.pt").is_file():
      raise FileNotFoundError(f"Missing continuation rank {rank}")
  if int(os.environ.get("WORLD_SIZE", "1")) != saved["world_size"]:
    raise ValueError(f"This checkpoint needs torchrun --nproc-per-node={saved['world_size']}")
  values = dict(cfg)
  values["resume"] = str(checkpoint.parent)
  launch_worker(values, runtime, preflight=args.preflight)


def launch_worker(cfg: dict, runtime: Path, *, preflight: bool) -> None:
  request = {"config": cfg, "preflight": preflight}
  environment = dict(os.environ, PYTHONPATH=str(runtime))
  worker = Path(__file__).with_name("continuation.py")
  print(f"[mocap] MLP + SuperMimic; verified frozen runtime: {runtime}", flush=True)
  subprocess.run([sys.executable, str(worker), json.dumps(request)], env=environment, check=True)
