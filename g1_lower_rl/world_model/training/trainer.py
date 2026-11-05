"""Bounded single-device/DDP world-model pretraining with local GPU replay."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import random
import signal
import time
import traceback
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from g1_lower_rl.world_model.checkpoint import (
  CHECKPOINT_VERSION,
  atomic_checkpoint,
  atomic_json,
  primitive_metadata,
  read_checkpoint,
)
from g1_lower_rl.world_model.model import (
  CONTRACT_VERSION,
  CONTROL_DT,
  INPUT_DIM,
  STATE_DIM,
  PositionWorldModel,
  PositionWorldModelCfg,
)
from g1_lower_rl.world_model.training.derivatives import SECANT_EPSILONS, derivative_metrics, derivative_sums, secant_loss
from g1_lower_rl.world_model.training.normalization import fit_normalization
from g1_lower_rl.world_model.training.objectives import (
  LossConfig,
  autoregressive_loss,
  finalize_metrics,
  physical_metric_sums,
)
from g1_lower_rl.world_model.training.provenance import snapshot_sources
from g1_lower_rl.world_model.training.replay import ReplayWindow, RingReplay


def utc_now() -> str:
  return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class TrainingConfig:
  checkpoint: str = ""
  """Frozen teacher checkpoint. Required unless --synthetic is explicitly selected."""
  teacher_kind: Literal["rgmt", "supermimic22650"] = "rgmt"
  replica_source: str | None = None
  """Audited third-party source tree required by the frozen native22650 teacher."""
  output_dir: str = "logs/world_model/pretrain"
  resume: str | None = None
  fork_from: str | None = None
  """Explicit capacity/source continuation in a new directory, retaining full optimizer and fixed validation."""
  initialize: str | None = None
  """Start new optimization from pretrained weights and their frozen normalization."""
  fixed_validation: bool = False
  """Persist held-out windows once; reuse identical physical trajectories at every evaluation."""
  one_step_loss_weight: float = 0.0
  """Additional local transition supervision alongside the autoregressive objective."""
  derivative_validation: bool = False
  secant_loss_weight: float = 0.0
  secant_batch_size: int = 128
  device: str = "cuda:0"
  seed: int = 0
  num_envs: int = 512
  validation_envs: int = 64
  motion_dir: str = "run_logs/wm_teacher_corpus"
  """Bounded, manifest-backed teacher corpus; unbounded loader defaults are forbidden."""
  coverage: bool = True
  physics_dt: float = 0.005
  alternate_physics_dt: bool = False
  """Retired option, rejected by v3 rather than mixing unconditioned integration settings."""
  action_noise: float = 0.1
  replay_steps: int = 512
  collect_steps: int = 8
  normalization_samples: int = 32768
  max_bootstrap_collections: int = 256
  batch_size: int = 512
  eval_batch_size: int | None = None
  """Independent evaluation batch; None retains the historical training-batch default."""
  horizon: int = 8
  updates_per_collection: int = 8
  gradient_accumulation: int = 1
  learning_rate: float = 2e-4
  grad_norm: float = 1.0
  bf16: bool = True
  tf32: bool = True
  compile_model: bool = False
  model: PositionWorldModelCfg = field(default_factory=PositionWorldModelCfg)
  loss: LossConfig = field(default_factory=LossConfig)
  input_scale_floor: float = 1e-3
  delta_scale_floor: float = 1e-3
  force_scale_floor: float = 1.0
  hours: float = 11.5
  max_updates: int | None = None
  eval_interval: int = 250
  save_interval: int = 1000
  eval_seconds: float = 600.0
  save_seconds: float = 900.0
  eval_batches: int = 8
  eval_horizons_seconds: tuple[float, ...] = (0.02, 0.16)
  """Physical rollout endpoints, each a whole number of20ms control steps."""
  log_interval: int = 10
  stop_file: str | None = None
  distributed_timeout_seconds: int = 180
  synthetic: bool = False
  """Explicit simulator-free infrastructure smoke; not usable physics training data."""

  def __post_init__(self) -> None:
    if self.alternate_physics_dt or self.physics_dt != 0.005:
      raise ValueError(
        "WM v3 uses fixed5ms native substeps and fixed20ms model/control steps; mixed physical dt is unsupported"
      )
    if sum(value is not None for value in (self.initialize, self.resume, self.fork_from)) > 1:
      raise ValueError("Initialization, optimizer resume and capacity fork are mutually exclusive")
    if self.eval_batch_size is not None and self.eval_batch_size <= 0:
      raise ValueError("eval_batch_size must be positive")
    if not math.isfinite(self.one_step_loss_weight) or self.one_step_loss_weight < 0:
      raise ValueError("one_step_loss_weight must be nonnegative and finite")
    if not math.isfinite(self.secant_loss_weight) or self.secant_loss_weight < 0 or self.secant_batch_size < 1:
      raise ValueError("Invalid secant loss weight/batch size")
    if self.secant_loss_weight and (not self.derivative_validation or self.bf16 or self.tf32):
      raise ValueError("Secant supervision requires independent derivative validation and full FP32 without TF32/BF16")
    positive_ints = (
      "num_envs",
      "validation_envs",
      "replay_steps",
      "collect_steps",
      "normalization_samples",
      "max_bootstrap_collections",
      "batch_size",
      "horizon",
      "updates_per_collection",
      "gradient_accumulation",
      "eval_interval",
      "save_interval",
      "eval_batches",
      "log_interval",
      "distributed_timeout_seconds",
    )
    for name in positive_ints:
      if getattr(self, name) <= 0:
        raise ValueError(f"{name} must be positive")
    for name in (
      "physics_dt",
      "learning_rate",
      "grad_norm",
      "input_scale_floor",
      "delta_scale_floor",
      "force_scale_floor",
      "hours",
      "eval_seconds",
      "save_seconds",
    ):
      if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
        raise ValueError(f"{name} must be positive and finite")
    if not math.isfinite(self.action_noise) or self.action_noise < 0:
      raise ValueError("action_noise must be nonnegative and finite")
    if self.validation_envs >= self.num_envs or self.horizon > self.replay_steps:
      raise ValueError("Need held-out and training environments, and horizon <= replay capacity")
    if self.normalization_samples > self.replay_steps * (self.num_envs - self.validation_envs):
      raise ValueError("normalization_samples exceeds training replay capacity")
    if not self.eval_horizons_seconds or any(not math.isfinite(t) or t <= 0 for t in self.eval_horizons_seconds):
      raise ValueError("eval_horizons_seconds must be nonempty, finite and positive")
    if any(not math.isclose(t / CONTROL_DT, round(t / CONTROL_DT), abs_tol=1e-6) for t in self.eval_horizons_seconds):
      raise ValueError("Evaluation horizons must be whole 20ms control steps")
    if round(max(self.eval_horizons_seconds) / CONTROL_DT) > self.replay_steps:
      raise ValueError("Physical evaluation horizons exceed replay capacity")
    if self.max_updates is not None and self.max_updates <= 0:
      raise ValueError("max_updates must be positive")
    if not self.synthetic and not self.checkpoint:
      raise ValueError("A frozen teacher checkpoint is required (or explicitly use --synthetic)")
    if self.teacher_kind == "supermimic22650" and (not self.replica_source or self.synthetic):
      raise ValueError("Native22650 requires --replica-source and real MuJoCo collection")


@dataclass
class DistributedContext:
  device: torch.device
  rank: int
  world_size: int
  owns_group: bool = False

  @classmethod
  def initialize(cls, cfg: TrainingConfig) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0 or not 0 <= rank < world_size:
      raise ValueError("Invalid torchrun WORLD_SIZE/RANK")
    if world_size > 1:
      missing = [key for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK") if key not in os.environ]
      if missing:
        raise RuntimeError(f"Launch DDP with torchrun; missing rendezvous environment: {missing}")
    pinned = os.environ.get("WM_GPU_PINNED") == "1"
    device = torch.device(
      f"cuda:{0 if pinned else local_rank}" if world_size > 1 and cfg.device.startswith("cuda") else cfg.device
    )
    if device.type == "cuda":
      if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; explicitly select --device cpu for CPU smoke")
      torch.cuda.set_device(device)
      device = torch.device("cuda", torch.cuda.current_device())
      if cfg.bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 requested but unsupported; explicitly select --no-bf16")
    elif cfg.bf16:
      raise ValueError("BF16 training is CUDA-only; explicitly use --no-bf16 for CPU smoke")
    owns = world_size > 1 and not dist.is_initialized()
    if owns:
      dist.init_process_group(
        backend="nccl" if device.type == "cuda" else "gloo",
        init_method="env://",
        timeout=timedelta(seconds=cfg.distributed_timeout_seconds),
      )
    if dist.is_initialized() and (dist.get_world_size() != world_size or dist.get_rank() != rank):
      raise RuntimeError("Existing process group disagrees with torchrun environment")
    return cls(device, rank, world_size, owns)

  def any(self, flag: bool) -> bool:
    value = torch.tensor(int(flag), device=self.device)
    if self.world_size > 1:
      dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return bool(value)

  def sum(self, value: int) -> int:
    tensor = torch.tensor(value, device=self.device, dtype=torch.int64)
    if self.world_size > 1:
      dist.all_reduce(tensor)
    return int(tensor)


def _collector(cfg: TrainingConfig, context: DistributedContext, seed: int):
  if cfg.synthetic:
    from g1_lower_rl.world_model.training.synthetic import SyntheticCollector

    return SyntheticCollector(num_envs=cfg.num_envs, device=context.device, seed=seed, physics_dt=cfg.physics_dt)
  # Keep simulator/task/teacher imports out of inference and replay-only use.
  from g1_lower_rl.world_model.collection import CollectorConfig, MotionCollector

  config = CollectorConfig(
    checkpoint=cfg.checkpoint,
    num_envs=cfg.num_envs,
    device=str(context.device),
    seed=seed,
    motion_dir=cfg.motion_dir,
    coverage=cfg.coverage,
    physics_dt=cfg.physics_dt,
    action_noise=cfg.action_noise,
  )
  if cfg.teacher_kind == "supermimic22650":
    from g1_lower_rl.world_model.baseline_collection import BaselineMotionCollector

    assert cfg.replica_source is not None
    return BaselineMotionCollector(config, replica_source=cfg.replica_source, validation_envs=cfg.validation_envs)
  return MotionCollector(config)


def _autocast(cfg: TrainingConfig):
  return torch.autocast("cuda", dtype=torch.bfloat16) if cfg.bf16 else contextlib.nullcontext()


def _stable_collector_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
  stable = dict(metadata)
  config = dict(stable.get("config", {}))
  for key in ("seed", "device"):
    config.pop(key, None)
  if "teacher_sha256" in stable:
    stable.pop("teacher_checkpoint", None)
    config.pop("checkpoint", None)
  if "config" in stable:
    stable["config"] = config
  return stable


def _capacity_collector_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
  stable = _stable_collector_metadata(metadata)
  num_envs = None
  if "config" in stable:
    num_envs = stable["config"].pop("num_envs", None)
  if "storage" in stable:
    if not isinstance(num_envs, int) or num_envs < 1:
      raise ValueError("Capacity comparison requires the recorded environment count")
    storage = dict(stable["storage"])
    storage["naconmax_per_env"] = storage.pop("naconmax") / num_envs
    stable["storage"] = storage
  if "corpus" in stable:
    corpus = dict(stable["corpus"])
    if corpus.get("joint_projection") in (
      "legacy31_closed_grippers_to_canonical29",
      "legacy31_ignore_gripper_state_to_canonical29",
    ):
      corpus["joint_projection"] = "legacy31_discard_grippers_to_canonical29"
      state = corpus.pop("gripper_state", None)
      if state not in (None, "excluded_from_reference_policy_and_WM_coordinates_no_clip_filtering"):
        raise ValueError("Unrecognized gripper projection policy during capacity fork")
    stable["corpus"] = corpus
  return stable


def _capacity_fork_origin(loaded: dict[str, Any], cfg: TrainingConfig, source: dict[str, Any]) -> dict[str, Any]:
  if not cfg.fixed_validation or cfg.derivative_validation:
    raise ValueError("Capacity forks require fixed validation and do not support derivative probes")
  previous = loaded["config"]
  if not previous.get("fixed_validation") or cfg.hours > previous["hours"]:
    raise ValueError("Capacity fork cannot replace fixed validation or extend the cumulative budget")
  previous_eval = previous.get("eval_batch_size") or previous["batch_size"]
  if (cfg.eval_batch_size or cfg.batch_size) != previous_eval:
    raise ValueError("Capacity fork must retain the original evaluation batch size")
  if cfg.seed % (2**32) != previous["seed"] % (2**32):
    raise ValueError("Capacity fork seed must retain its uint32 identity")
  if cfg.validation_envs * previous["num_envs"] != previous["validation_envs"] * cfg.num_envs:
    raise ValueError("Capacity fork must preserve the held-out environment fraction")
  current = primitive_metadata(asdict(cfg))
  operational = {
    "resume",
    "fork_from",
    "initialize",
    "output_dir",
    "num_envs",
    "validation_envs",
    "batch_size",
    "eval_batch_size",
    "compile_model",
    "hours",
    "max_updates",
    "stop_file",
    "seed",
  }
  changes = {
    key: {"parent": previous.get(key), "current": value} for key, value in current.items() if previous.get(key, None) != value
  }
  forbidden = changes.keys() - operational
  if forbidden:
    raise ValueError(f"Capacity fork changed non-capacity training configuration: {sorted(forbidden)}")
  parent = loaded["source_provenance"]
  directory = Path(parent["snapshot_directory"])
  manifest = json.loads((directory / "manifest.json").read_text())
  digest = hashlib.sha256(json.dumps(manifest["files"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
  if (
    manifest.get("format_version") != 1
    or manifest.get("sha256") != digest
    or digest != parent["sha256"]
    or manifest["files"] != parent["files"]
  ):
    raise ValueError("Parent source manifest does not match the checkpoint")
  for name, expected in parent["files"].items():
    if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
      raise ValueError(f"Parent immutable source changed: {name}")
  source_changes = {
    name: {"parent": parent["files"].get(name), "current": source["files"].get(name)}
    for name in parent["files"].keys() | source["files"].keys()
    if parent["files"].get(name) != source["files"].get(name)
  }
  allowed_sources = {
    "g1_lower_rl/world_model/training/trainer.py",
    "g1_lower_rl/world_model/mocap.py",
    "g1_lower_rl/tasks/motion_tracking/corpus.py",
    "g1_lower_rl/tasks/motion_tracking/mdp/motion_corpus.py",
  }
  if source_changes.keys() - allowed_sources:
    raise ValueError("Capacity fork cannot change model, objective, normalization, physics or other source modules")
  assert cfg.fork_from is not None
  return {
    "kind": "explicit_capacity_fork_full_optimizer",
    "parent_checkpoint": str(Path(cfg.fork_from).resolve()),
    "parent_checkpoint_sha256": hashlib.sha256(Path(cfg.fork_from).read_bytes()).hexdigest(),
    "parent_update": loaded["update"],
    "parent_source_provenance": parent,
    "source_changes": source_changes,
    "configuration_changes": changes,
    "parent_continuation": loaded.get("continuation"),
  }


def _corpus_manifest(cfg: TrainingConfig) -> dict[str, Any] | None:
  if cfg.synthetic:
    return None
  directory = Path(cfg.motion_dir)
  if not directory.is_dir():
    raise FileNotFoundError(f"Bounded motion corpus is missing: {directory}; explicitly prepare it or pass --motion-dir")
  path = directory / "manifest.json"
  if not path.is_file():
    raise FileNotFoundError(f"Bounded motion corpus must contain an exact manifest: {path}")
  data = path.read_bytes()
  contents = json.loads(data)
  if "frame_width" in contents and "data_sha256" in contents and "clips" in contents:
    clips = contents.pop("clips")
    contents["clip_count"] = len(clips)
    contents["split_counts"] = {split: sum(c["split"] == split for c in clips) for split in ("train", "validation")}
  return {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(), "contents": contents}


def _evaluate(
  forward: Callable,
  cfg: TrainingConfig,
  replay: RingReplay,
  context: DistributedContext,
  generator: torch.Generator,
  evaluation_steps: dict[str, int],
  delta_scale: torch.Tensor,
  windows: list[ReplayWindow] | None = None,
) -> dict[str, float | None]:
  sums: dict[str, torch.Tensor] = {}
  with torch.no_grad(), _autocast(cfg):
    for batch in range(cfg.eval_batches):
      window = (
        windows[batch]
        if windows is not None
        else replay.sample(
          cfg.eval_batch_size or cfg.batch_size, max(evaluation_steps.values()), validation=True, generator=generator
        )
      )
      for key, value in physical_metric_sums(
        forward,
        window,
        cfg.loss.zero_force_threshold,
        evaluation_steps=evaluation_steps,
        contact_threshold=cfg.loss.contact_threshold,
        delta_scale=delta_scale,
      ).items():
        sums[key] = sums.get(key, torch.zeros_like(value)) + value
  keys = sorted(sums)
  values = torch.stack([sums[key] for key in keys]).double()
  if context.world_size > 1:
    dist.all_reduce(values)
  if not bool(torch.isfinite(values).all()):
    raise FloatingPointError("Nonfinite held-out rollout metrics")
  return finalize_metrics(dict(zip(keys, values, strict=True)))


def _fixed_validation_windows(
  root: Path,
  cfg: TrainingConfig,
  replay: RingReplay,
  context: DistributedContext,
  generator: torch.Generator,
  horizon: int,
  expected_sha256: str | None = None,
) -> tuple[list[ReplayWindow], str]:
  path = root / f"fixed_validation_rank_{context.rank}.pt"
  names = ("state", "inputs", "next_state", "foot_force", "env", "sequence")
  batch_size = cfg.eval_batch_size or cfg.batch_size
  if expected_sha256 is not None:
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
      raise ValueError("Fixed validation snapshot checksum mismatch")
    saved = torch.load(path, map_location=context.device, weights_only=True)
    if saved.get("contract_version") != CONTRACT_VERSION:
      raise ValueError("Fixed validation snapshot is not position protocol v3")
    windows = [ReplayWindow(**record) for record in saved["windows"]]
    if len(windows) != cfg.eval_batches or any(
      w.inputs.shape != (batch_size, horizon, INPUT_DIM)
      or w.state.shape != (batch_size, horizon, STATE_DIM)
      or w.next_state.shape != w.state.shape
      or w.foot_force.shape != (batch_size, horizon, 2, 3)
      for w in windows
    ):
      raise ValueError("Fixed validation snapshot shape/configuration mismatch")
  else:
    if path.exists():
      raise FileExistsError(f"Refusing to overwrite fixed validation: {path}")
    windows = [replay.sample(batch_size, horizon, validation=True, generator=generator) for _ in range(cfg.eval_batches)]
    atomic_checkpoint(
      path,
      {
        "contract_version": CONTRACT_VERSION,
        "windows": [{name: getattr(window, name).detach().cpu() for name in names} for window in windows],
      },
    )
  return windows, hashlib.sha256(path.read_bytes()).hexdigest()


def refinement_eligible(metrics: Mapping[str, float | None], baseline: Mapping[str, float | None]) -> bool:
  directions = {f"derivative/{epsilon}/cosine": 1 for epsilon in map(str, SECANT_EPSILONS)}
  directions.update({f"{horizon}/state_normalized/rmse": -1 for horizon in ("one_step", "rollout_0.02s", "rollout_0.16s")})
  eligible = True
  for key, direction in directions.items():
    current, initial = metrics[key], baseline[key]
    if current is None or initial is None or not math.isfinite(current) or not math.isfinite(initial):
      raise ValueError(f"Refinement selection requires finite validation: {key}")
    eligible &= direction * (current - initial) >= 0
  return eligible


def run_training(cfg: TrainingConfig, collector_factory: Callable | None = None) -> Path:
  """Run a bounded job; a factory is injectable for synthetic regression tests."""
  root = Path(cfg.output_dir)
  root.mkdir(parents=True, exist_ok=True)
  rank = int(os.environ.get("RANK", "0"))
  status_path = root / ("status.json" if rank == 0 else f"status_rank_{rank}.json")
  start = time.monotonic()
  context: DistributedContext | None = None
  collector = None
  previous_handlers: dict[int, Any] = {}
  requested_stop = False
  update, transitions, prior_wall = 0, 0, 0.0
  normalization: dict[str, Any] = {}
  best_score: float | None = None
  latest_metrics: dict[str, float | None] = {}
  selection_baseline: dict[str, float | None] | None = None
  continuation: dict[str, Any] | None = None
  best_metric = f"rollout_{max(cfg.eval_horizons_seconds):g}s/state_normalized/rmse"

  def status(state: str, **extra: Any) -> None:
    atomic_json(
      status_path,
      {
        "status": state,
        "timestamp_utc": utc_now(),
        "pid": os.getpid(),
        "rank": rank,
        "update": update,
        "transitions": transitions,
        "wall_seconds": prior_wall + time.monotonic() - start,
        **extra,
      },
    )

  def request_stop(_signum, _frame):
    nonlocal requested_stop
    requested_stop = True

  try:
    status("initializing")
    context = DistributedContext.initialize(cfg)
    if not cfg.resume and (root / "latest.pt").exists():
      raise FileExistsError("Output already contains latest.pt; choose another output directory or explicitly resume")
    manifest = _corpus_manifest(cfg)
    parent_checkpoint = cfg.resume or cfg.fork_from
    loaded = read_checkpoint(parent_checkpoint, context.device) if parent_checkpoint else None
    if loaded is not None and loaded.get("artifact_kind") == "pretrained_initialization":
      raise ValueError(
        "A shared pretrained WM is an initialization, not an optimizer resume; "
        "load_world_model() starts a new training run, while --resume requires a full training snapshot"
      )
    source_provenance = snapshot_sources(root, write=rank == 0)
    if context.world_size > 1:
      source_ids: list[Any] = [None] * context.world_size
      dist.all_gather_object(source_ids, source_provenance["sha256"])
      if len(set(source_ids)) != 1:
        raise RuntimeError("DDP ranks are running different model/training sources")
    if (
      loaded is not None
      and cfg.fork_from is None
      and loaded.get("source_provenance", {}).get("sha256", source_provenance["sha256"]) != source_provenance["sha256"]
    ):
      raise ValueError("Resume model/training sources differ from the immutable checkpoint snapshot; start a fresh job")
    if cfg.fork_from:
      if Path(cfg.fork_from).resolve().parent == root.resolve():
        raise ValueError("Capacity fork requires a new output directory")
      assert loaded is not None
      continuation = _capacity_fork_origin(loaded, cfg, source_provenance)
    elif loaded is not None:
      continuation = loaded.get("continuation")
    if loaded is not None:
      for name, default in (
        ("fixed_validation", False),
        ("one_step_loss_weight", 0.0),
        ("tf32", True),
        ("derivative_validation", False),
        ("secant_loss_weight", 0.0),
        ("secant_batch_size", 128),
      ):
        if loaded["config"].get(name, default) != getattr(cfg, name):
          raise ValueError(f"Resume configuration changed: {name}")
      if loaded["model_config"] != asdict(cfg.model):
        raise ValueError("Resume model architecture does not match --model configuration")
      update, transitions, prior_wall = loaded["update"], loaded["transitions"], loaded["wall_seconds"]
      normalization, best_score = loaded["normalization"], loaded["best_score"]
      if loaded.get("best_metric") != best_metric:
        best_score = None
      if loaded["world_size"] != context.world_size:
        raise ValueError("Resume requires the same world size for exact per-rank RNG and optimizer recovery")
      latest_metrics = loaded["metrics"]
      selection_baseline = loaded.get("selection_baseline")
    seed = cfg.seed + 100003 * context.rank
    random.seed(seed)
    torch.manual_seed(seed)
    if context.device.type == "cuda":
      torch.cuda.manual_seed_all(seed)
      torch.backends.cuda.matmul.allow_tf32 = cfg.tf32
    if cfg.initialize:
      from g1_lower_rl.world_model.checkpoint import load_world_model

      model = load_world_model(cfg.initialize, device=context.device)
      if model.cfg != cfg.model:
        raise ValueError("Initialization architecture does not match model configuration")
      normalization = {
        "frozen": True,
        "source": "pretrained_initialization",
        "checkpoint_sha256": hashlib.sha256(Path(cfg.initialize).read_bytes()).hexdigest(),
      }
    else:
      model = PositionWorldModel(cfg.model).to(context.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    train_generator = torch.Generator(device=context.device).manual_seed(seed + 17)
    eval_generator = torch.Generator(device=context.device).manual_seed(seed + 29)
    if loaded is not None:
      model.load_state_dict(loaded["model"])
      optimizer.load_state_dict(loaded["optimizer"])
      if not bool(model.normalization_ready):
        raise RuntimeError("Resume checkpoint is missing frozen normalization")
    for signum in (signal.SIGINT, signal.SIGTERM):
      previous_handlers[signum] = signal.signal(signum, request_stop)
    # A resumed simulator starts fresh, independently seeded trajectories. Its
    # state is intentionally NOT restored; replay is empty until new warmup.
    # Native NumPy/Warp seeders require uint32, including late optimizer resumes.
    collector_seed = (seed + (update + 1) * 1000003 if loaded is not None else seed) % (2**32)
    collector = (collector_factory or _collector)(cfg, context, collector_seed)
    evaluation_steps = {f"rollout_{duration:g}s": round(duration / CONTROL_DT) for duration in cfg.eval_horizons_seconds}
    required_horizon = max(cfg.horizon, *evaluation_steps.values())
    provenance = {
      "collector": primitive_metadata({**collector.metadata, "corpus_manifest": manifest}),
      "rank": rank,
      "seed": collector_seed,
      "validation_env_indices": [cfg.num_envs - cfg.validation_envs, cfg.num_envs],
      "training_env_indices": [0, cfg.num_envs - cfg.validation_envs],
      "resume_replay": "fresh_trajectories_no_environment_restore",
    }
    if loaded is not None:
      previous_collector = loaded["provenance"][rank]["collector"]
      compare_metadata = _capacity_collector_metadata if cfg.fork_from else _stable_collector_metadata
      previous_metadata, current_metadata = compare_metadata(previous_collector), compare_metadata(provenance["collector"])
      if previous_metadata != current_metadata:
        changed = sorted(
          key
          for key in previous_metadata.keys() | current_metadata.keys()
          if previous_metadata.get(key) != current_metadata.get(key)
        )
        raise ValueError(f"Collector provenance differs from resume checkpoint; changed fields: {changed}")
    atomic_json(root / f"provenance_rank_{rank}.json", provenance)
    replay = RingReplay(cfg.replay_steps, cfg.num_envs, cfg.validation_envs, context.device)
    stop_file = Path(cfg.stop_file) if cfg.stop_file else root / "STOP"

    def should_stop() -> bool:
      return context.any(
        requested_stop
        or stop_file.exists()
        or prior_wall + time.monotonic() - start >= cfg.hours * 3600
        or (cfg.max_updates is not None and update >= cfg.max_updates)
      )

    collect_seconds, train_seconds = 0.0, 0.0
    collected_this_run, fitted_samples = 0, 0

    def collect() -> None:
      nonlocal transitions, collect_seconds, collected_this_run, fitted_samples
      before = time.monotonic()
      batch = collector.collect(num_control_steps=cfg.collect_steps)
      count = replay.append(batch)
      fitted_samples = int(replay.valid[:, : replay.train_envs].sum())
      global_count = context.sum(count)
      transitions += global_count
      collected_this_run += global_count
      collect_seconds += time.monotonic() - before

    status("warming_up", world_size=context.world_size, replay_bytes=replay.bytes_allocated, provenance=provenance)
    for _ in range(cfg.max_bootstrap_collections):
      # Readiness must be identical across ranks before any collective stats.
      ready = (
        replay.candidates(cfg.horizon).numel() > 0
        and replay.candidates(required_horizon, validation=True).numel() > 0
        and (loaded is not None or cfg.initialize is not None or fitted_samples >= cfg.normalization_samples)
      )
      if not context.any(not ready):
        break
      if should_stop():
        raise RuntimeError("Stopped before replay/normalization was ready; no usable new checkpoint")
      collect()
    else:
      raise RuntimeError("Replay bootstrap exhausted: insufficient valid contiguous train/validation windows")
    if loaded is None and cfg.initialize is None:
      normalization = fit_normalization(model, replay, cfg.input_scale_floor, cfg.delta_scale_floor, cfg.force_scale_floor)
    derivative_probe, derivative_sha256, training_probe = None, None, None
    if cfg.derivative_validation:
      derivative_path = root / f"fixed_derivatives_rank_{rank}.pt"
      if loaded is not None:
        derivative_sha256 = hashlib.sha256(derivative_path.read_bytes()).hexdigest()
        if derivative_sha256 != loaded["fixed_derivative_sha256"][rank]:
          raise ValueError("Fixed derivative snapshot checksum mismatch")
        derivative_probe = torch.load(derivative_path, map_location=context.device, weights_only=True)
        if cfg.secant_loss_weight:
          training_path = root / f"training_secants_rank_{rank}.pt"
          training_sha = hashlib.sha256(training_path.read_bytes()).hexdigest()
          if training_sha != loaded["training_secant_sha256"][rank]:
            raise ValueError("Training secant checksum mismatch")
          training_probe = torch.load(training_path, map_location=context.device, weights_only=True)
      else:
        if derivative_path.exists():
          raise FileExistsError(f"Refusing to overwrite derivative labels: {derivative_path}")
        probe_factory = getattr(collector, "validation_probe", None)
        if not callable(probe_factory):
          raise TypeError("Derivative validation requires a native probe-capable collector")
        probe_result = probe_factory()
        if (
          not isinstance(probe_result, tuple)
          or len(probe_result) != 3
          or not all(isinstance(x, dict) for x in probe_result[1:])
        ):
          raise TypeError("Native probe must return a transition batch and separate validation/training labels")
        probe_batch, derivative_probe, training_probe = probe_result
        count = context.sum(replay.append(probe_batch))
        transitions += count
        collected_this_run += count
        atomic_checkpoint(derivative_path, derivative_probe)
        derivative_sha256 = hashlib.sha256(derivative_path.read_bytes()).hexdigest()
        atomic_checkpoint(root / f"training_secants_rank_{rank}.pt", training_probe)
    validation_windows, validation_sha256 = None, None
    if cfg.fixed_validation:
      previous_sha = loaded["fixed_validation_sha256"][rank] if loaded is not None else None
      validation_windows, validation_sha256 = _fixed_validation_windows(
        root,
        cfg,
        replay,
        context,
        eval_generator,
        max(evaluation_steps.values()),
        previous_sha,
      )
    if loaded is not None:
      rng = loaded["rng"][rank]
      torch.set_rng_state(rng["torch"].cpu())
      random.setstate(rng["python"])
      train_generator.set_state(rng["train"].cpu())
      eval_generator.set_state(rng["eval"].cpu())
      if context.device.type == "cuda":
        torch.cuda.set_rng_state(rng["cuda"].cpu(), context.device)
    # Compile the inner module, not the DDP wrapper: Dynamo cannot capture DDP's
    # Python reducer bookkeeping with fullgraph=True.
    inference_model: Any = torch.compile(model, fullgraph=True) if cfg.compile_model else model
    train_model: Any = inference_model
    if context.world_size > 1:
      train_model = DistributedDataParallel(
        inference_model,
        device_ids=[context.device.index] if context.device.type == "cuda" else None,
        broadcast_buffers=False,
      )

    # DDP must own the forward so all parameter reductions remain registered.
    # BF16 is restricted to linear operations; states, stats and losses stay FP32.
    def training_forward(state, target, physical_context):
      return train_model(state, target, physical_context, validate=False)

    def evaluation_forward(state, target, physical_context):
      return inference_model(state, target, physical_context, validate=False)

    train_forward, eval_forward = training_forward, evaluation_forward
    last_eval_time = last_save_time = time.monotonic()
    last_log_time, last_log_update, last_log_transitions = time.monotonic(), update, transitions

    def log(record: dict[str, Any]) -> None:
      if rank == 0:
        with (root / "metrics.jsonl").open("a") as stream:
          stream.write(
            json.dumps({"timestamp_utc": utc_now(), "update": update, "transitions": transitions, **record}, allow_nan=False)
            + "\n"
          )

    def save(name: str) -> Path:
      local_rng = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "train": train_generator.get_state(),
        "eval": eval_generator.get_state(),
        "cuda": torch.cuda.get_rng_state(context.device) if context.device.type == "cuda" else None,
      }
      provenance_all: list[Any] = [None] * context.world_size
      rng_all: list[Any] = [None] * context.world_size
      validation_all: list[Any] = [None] * context.world_size
      derivative_all: list[Any] = [None] * context.world_size
      training_all: list[Any] = [None] * context.world_size
      counters_all: list[Any] = [None] * context.world_size
      training_sha = (
        hashlib.sha256((root / f"training_secants_rank_{rank}.pt").read_bytes()).hexdigest()
        if cfg.derivative_validation
        else None
      )
      if context.world_size > 1:
        dist.all_gather_object(provenance_all, provenance)
        dist.all_gather_object(rng_all, local_rng)
        dist.all_gather_object(validation_all, validation_sha256)
        dist.all_gather_object(derivative_all, derivative_sha256)
        dist.all_gather_object(training_all, training_sha)
        dist.all_gather_object(counters_all, dict(collector.counters))
      else:
        provenance_all, rng_all = [provenance], [local_rng]
        validation_all = [validation_sha256]
        derivative_all = [derivative_sha256]
        training_all = [training_sha]
        counters_all = [dict(collector.counters)]
      destination = root / name
      failed = False
      error = None
      if rank == 0:
        try:
          atomic_checkpoint(
            destination,
            {
              "format_version": CHECKPOINT_VERSION,
              "timestamp_utc": utc_now(),
              "model_config": asdict(cfg.model),
              "config": primitive_metadata(asdict(cfg)),
              "model": model.state_dict(),
              "optimizer": optimizer.state_dict(),
              "scaler": None,
              "normalization": primitive_metadata(normalization),
              "fixed_validation_sha256": validation_all,
              "fixed_derivative_sha256": derivative_all,
              "training_secant_sha256": training_all,
              "selection_baseline": selection_baseline,
              "provenance": provenance_all,
              "source_provenance": source_provenance,
              "continuation": continuation,
              "rng": rng_all,
              "update": update,
              "transitions": transitions,
              "wall_seconds": prior_wall + time.monotonic() - start,
              "world_size": context.world_size,
              "best_score": best_score,
              "best_metric": best_metric,
              "metrics": latest_metrics,
              "runtime": {"collection_seconds": collect_seconds, "training_seconds": train_seconds},
              "collector_counters": counters_all,
            },
          )
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
          failed, error = True, exc
      if context.any(failed):
        raise RuntimeError("Rank-zero checkpoint save failed") from error
      return destination

    def evaluate() -> bool:
      nonlocal latest_metrics, best_score, last_eval_time, selection_baseline
      model.eval()
      latest_metrics = _evaluate(
        eval_forward,
        cfg,
        replay,
        context,
        eval_generator,
        evaluation_steps,
        model.state_scale,
        validation_windows,
      )
      if derivative_probe is not None:
        sums = derivative_sums(model, derivative_probe)
        if context.world_size > 1:
          for value in sums.values():
            dist.all_reduce(value)
        latest_metrics.update(derivative_metrics(sums))
      if selection_baseline is None:
        selection_baseline = dict(latest_metrics)
      model.train()
      # Selection uses all tangent-state channels on fixed training scales;
      # physical per-field metrics and the nontrivial baseline remain explicit.
      score = latest_metrics[best_metric]
      assert score is not None
      improved = best_score is None or score < best_score
      if cfg.derivative_validation:
        improved &= refinement_eligible(latest_metrics, selection_baseline)
      if improved:
        best_score = score
      log({"kind": "evaluation", "metrics": latest_metrics, "best_score": best_score})
      if rank == 0:
        print(
          "WM held-out "
          + " | ".join(
            f"{duration:g}s joint_rmse={latest_metrics[f'rollout_{duration:g}s/joint_position/rmse']:.5g} rad "
            f"force_rmse={latest_metrics[f'rollout_{duration:g}s/force/rmse']:.5g} N "
            f"contact_f1={latest_metrics[f'rollout_{duration:g}s/contact/f1']} "
            f"swing_rmse={latest_metrics[f'rollout_{duration:g}s/swing_force/rmse']} N "
            f"state_nrmse={latest_metrics[f'rollout_{duration:g}s/state_normalized/rmse']:.5g} "
            f"cv_nrmse={latest_metrics[f'rollout_{duration:g}s/constant_velocity_state_normalized/rmse']:.5g}"
            for duration in cfg.eval_horizons_seconds
          ),
          flush=True,
        )
      last_eval_time = time.monotonic()
      return improved

    # Baseline held-out metrics also guarantee a best checkpoint for zero-update
    # budget/stop exits after a usable replay has been assembled.
    if evaluate():
      save("best.pt")
    save("latest.pt")
    last_save_time = time.monotonic()
    model.train()
    status("running", world_size=context.world_size, normalization=normalization, metrics=latest_metrics)
    stopped = False
    while not should_stop():
      for _ in range(cfg.updates_per_collection):
        if should_stop():
          stopped = True
          break
        before = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        loss_total = torch.zeros((), device=context.device)
        parts: dict[str, torch.Tensor] = {}
        for microstep in range(cfg.gradient_accumulation):
          window = replay.sample(cfg.batch_size, cfg.horizon, generator=train_generator)
          synchronize = microstep + 1 == cfg.gradient_accumulation
          reduction_context = train_model.no_sync() if context.world_size > 1 and not synchronize else contextlib.nullcontext()
          with reduction_context:
            with _autocast(cfg):
              loss, component = autoregressive_loss(train_forward, window, model.state_scale, model.force_scale, cfg.loss)
              if cfg.one_step_loss_weight:
                one = ReplayWindow(
                  window.state[:, :1],
                  window.inputs[:, :1],
                  window.next_state[:, :1],
                  window.foot_force[:, :1],
                  window.env,
                  window.sequence[:, :1],
                )
                local_loss, _ = autoregressive_loss(train_forward, one, model.state_scale, model.force_scale, cfg.loss)
                loss = loss + cfg.one_step_loss_weight * local_loss
                component["local_loss"] = local_loss.detach()
              if cfg.secant_loss_weight:
                if training_probe is None:
                  raise RuntimeError("Secant supervision has no independent training labels")
                sensitivity = secant_loss(
                  train_forward, training_probe, model.state_scale, cfg.secant_batch_size, train_generator
                )
                loss = loss + cfg.secant_loss_weight * sensitivity
                component["secant_loss"] = sensitivity.detach()
              loss = loss / cfg.gradient_accumulation
            if context.any(not bool(torch.isfinite(loss))):
              raise FloatingPointError("Nonfinite autoregressive training loss")
            loss.backward()
          loss_total += loss.detach()
          for key, value in component.items():
            parts[key] = parts.get(key, torch.zeros_like(value)) + value / cfg.gradient_accumulation
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm)
        if context.any(not bool(torch.isfinite(grad_norm))):
          raise FloatingPointError("Nonfinite gradients; optimizer step refused")
        optimizer.step()
        update += 1
        train_seconds += time.monotonic() - before
        now = time.monotonic()
        eval_due = update % cfg.eval_interval == 0 or now - last_eval_time >= cfg.eval_seconds
        save_due = update % cfg.save_interval == 0 or now - last_save_time >= cfg.save_seconds
        if context.any(eval_due) and evaluate():
          save("best.pt")
        if context.any(save_due):
          save("latest.pt")
          last_save_time = time.monotonic()
        if update % cfg.log_interval == 0 or update == 1 or update == cfg.max_updates:
          values = torch.stack([loss_total, grad_norm.detach(), *parts.values()])
          if context.world_size > 1:
            dist.all_reduce(values)
            values /= context.world_size
          elapsed = time.monotonic() - last_log_time
          runtime = {
            "collector_counters": dict(collector.counters),
            "updates_per_second": (update - last_log_update) / max(elapsed, 1e-9),
            "new_transitions_per_second": (transitions - last_log_transitions) / max(elapsed, 1e-9),
            "collection_transitions_per_second": collected_this_run / max(collect_seconds, 1e-9),
            "optimizer_examples_per_second": (
              (update - (loaded["update"] if loaded else 0))
              * cfg.batch_size
              * cfg.gradient_accumulation
              * cfg.horizon
              * context.world_size
              / max(train_seconds, 1e-9)
            ),
            "collection_seconds": collect_seconds,
            "training_seconds": train_seconds,
            "replay_bytes_per_rank": replay.bytes_allocated,
            "cuda_allocated_bytes": torch.cuda.memory_allocated(context.device) if context.device.type == "cuda" else 0,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(context.device)
            if context.device.type == "cuda"
            else 0,
            "cuda_reserved_bytes": torch.cuda.memory_reserved(context.device) if context.device.type == "cuda" else 0,
          }
          record = {
            "loss": float(values[0]),
            "grad_norm": float(values[1]),
            **dict(zip(parts, values[2:].tolist(), strict=True)),
          }
          log({"kind": "training", **record, "runtime": runtime})
          status("running", world_size=context.world_size, metrics=latest_metrics, runtime=runtime, training=record)
          if rank == 0:
            print(
              f"WM update={update} loss={record['loss']:.5g} steps/s={runtime['optimizer_examples_per_second']:.1f}", flush=True
            )
          last_log_time, last_log_update, last_log_transitions = time.monotonic(), update, transitions
      if stopped or should_stop():
        break
      collect()
      if context.any(
        replay.candidates(cfg.horizon).numel() == 0 or replay.candidates(required_horizon, validation=True).numel() == 0
      ):
        raise RuntimeError("Collection left replay without contiguous train/validation windows")
    if evaluate():
      save("best.pt")
    save("latest.pt")
    final = save("final.pt")
    reason = (
      "max_updates"
      if cfg.max_updates is not None and update >= cfg.max_updates
      else ("stop_file" if stop_file.exists() else "signal" if requested_stop else "wall_clock_budget")
    )
    status(
      "completed" if reason in ("max_updates", "wall_clock_budget") else "stopped",
      reason=reason,
      final_checkpoint=str(final.resolve()),
      metrics=latest_metrics,
      best_score=best_score,
      collector_counters=dict(collector.counters),
    )
    return final
  except BaseException:
    status("failed", traceback=traceback.format_exc())
    raise
  finally:
    for signum, handler in previous_handlers.items():
      signal.signal(signum, handler)
    try:
      if collector is not None:
        collector.close()
    except BaseException:
      status("failed", traceback=traceback.format_exc())
      raise
    finally:
      if context is not None and context.owns_group:
        dist.destroy_process_group()
