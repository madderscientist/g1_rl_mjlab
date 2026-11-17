"""SuperMimic-style online position-WM + H32 MLP BPTT + GAE trainer for velocity tracking."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import signal
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.checkpoint import checkpoint

from g1_lower_rl.rl.supermimic import (
  bptt_return,
  cosine_learning_rate,
  frozen,
  gae_targets,
  optimizer_step,
)
from g1_lower_rl.rl.supermimic import (
  broadcast_module as _broadcast,
)
from g1_lower_rl.rl.supermimic import (
  file_sha256 as _file_sha256,
)
from g1_lower_rl.world_model.adapters.kinematics import TorchKinematics
from g1_lower_rl.world_model.checkpoint import atomic_checkpoint, atomic_json, load_world_model, read_checkpoint
from g1_lower_rl.world_model.collection import CollectorConfig
from g1_lower_rl.world_model.model import CONTROL_DT, PositionWorldModel, PositionWorldModelCfg
from g1_lower_rl.world_model.training.derivatives import derivative_metrics, derivative_sums, native_secants, snapshot
from g1_lower_rl.world_model.training.normalization import fit_normalization
from g1_lower_rl.world_model.training.objectives import (
  LossConfig,
  autoregressive_loss,
  finalize_metrics,
  physical_metric_sums,
)
from g1_lower_rl.world_model.training.replay import ReplayWindow

from .collection import VelocityPolicyCollector
from .graphs import CapturedVelocityRolloutStep, CapturedWmMicrobatch
from .playback import FORMAT, export_onnx
from .policy import (
  ACTION_DIM,
  DEFAULT_HISTORY,
  VelocityPolicyBatch,
  VelocityPolicyRollout,
  actor_architecture_metadata,
  constrain_policy_kl,
  make_actor,
  migrate_actor_state_dict,
  observation_noise_std,
  velocity_observation,
  velocity_policy_kl,
  velocity_target,
)
from .task import (
  CRITIC_DIM,
  DEFAULT_TERMINATION_PROB,
  VelocityPolicyStart,
  VelocityReplay,
  VelocityWmTask,
  advance_arm_ramp,
  advance_height_command,
)

DEFAULT_PRETRAINED_WM = "logs/world_model/full_mocap22650_v3_4h_20261102/capacity8192/best.pt"
DEFAULT_EXPLORATION_STAGES: tuple[tuple[int, float, float], ...] = (
  (0, 0.25, 0.08),
  (6500, 0.08, 0.015),
)
_ROLLOUT_INPUT_NAMES: tuple[str, ...] = (
  "state",
  "history",
  "aux",
  "context",
  "encoder_bias",
  "imu_bias",
  "arm_target",
  "twist",
  "height_cmd",
  "next_height_cmd",
  "next_episode_step",
  "phase_offset",
  "cmd_elapsed",
  "drift_anchor",
  "is_stationary",
  "obs_noise",
  "act_noise",
)


def _snapshot_velocity_sources(output_dir: Path, *, write: bool) -> dict[str, Any]:
  wm_pkg = Path(__file__).resolve().parent
  lower_pkg = wm_pkg.parent
  repo = lower_pkg.parent.parent.parent
  paths = [
    *sorted((repo / "g1_lower_rl/world_model").glob("*.py")),
    *sorted((repo / "g1_lower_rl/world_model/adapters").glob("*.py")),
    *sorted((repo / "g1_lower_rl/world_model/training").glob("*.py")),
    repo / "g1_lower_rl/rl/supermimic.py",
    repo / "g1_lower_rl/scene_graph.py",
    repo / "g1_lower_rl/tasks/domain_randomization.py",
    repo / "g1_lower_rl/assets/g1_gloria/__init__.py",
    *sorted((repo / "g1_lower_rl/assets/g1_gloria/xmls").glob("*.xml")),
    *sorted(lower_pkg.rglob("*.py")),
    *(p for name in ("train.py", "play.py", "export_onnx.py", "train_velocity_wm.py") if (p := repo / "scripts" / name).is_file()),
  ]
  contents = {str(path.relative_to(repo)): path.read_bytes() for path in paths}
  hashes = {name: hashlib.sha256(data).hexdigest() for name, data in sorted(contents.items())}
  digest = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
  directory = output_dir / "sources" / digest
  manifest = {"format_version": 1, "sha256": digest, "files": hashes}
  if write:
    for name, data in contents.items():
      target = directory / name
      target.parent.mkdir(parents=True, exist_ok=True)
      if not target.exists():
        target.write_bytes(data)
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
      atomic_json(manifest_path, manifest)
  return {**manifest, "snapshot_directory": str(directory.resolve())}


@dataclass(frozen=True)
class VelocityWmConfig:
  output: str = "logs/velocity_wm/run"
  world_model: str = DEFAULT_PRETRAINED_WM
  scratch_wm: bool = False
  resume: str | None = None
  preflight: bool = False
  device: str = "cuda:0"
  seed: int = 20261111
  iterations: int = 0
  """0 runs continuously until STOP file or SIGINT/SIGTERM; positive value bounds iterations."""
  history_length: int = DEFAULT_HISTORY
  actor_hidden_dims: tuple[int, ...] = (1024, 512, 256)
  num_envs: int = 4096
  validation_envs: int = 512
  replay_steps: int = 384
  warmup_steps: int = 64
  actor_batch_size: int = 8192
  actor_microbatch: int = 8192
  horizon: int = 32
  wm_batch_size: int = 16384
  wm_microbatch: int = 8192
  wm_horizon: int = 8
  wm_batches: int = 6
  critic_epochs: int = 5
  critic_minibatches: int = 4
  critic_microbatch: int = 32768
  learning_rate: float = 2e-4
  wm_learning_rate: float = 1e-4
  gamma: float = 0.99
  gae_lambda: float = 0.95
  utd: float = 0.5
  action_noise: float | None = None
  """None follows exploration_stages automatically; explicit positive value overrides."""
  policy_target_kl: float | None = None
  """None follows exploration_stages automatically; explicit positive value overrides."""
  exploration_stages: tuple[tuple[int, float, float], ...] = DEFAULT_EXPLORATION_STAGES
  """(start_iteration, action_noise, policy_target_kl) stages from gait acquisition to polish."""
  termination_probability: float = DEFAULT_TERMINATION_PROB
  policy_microbatch: int = 16384
  save_interval: int = 20
  save_seconds: float = 300.0
  validation_interval: int = 10
  validation_batch_size: int = 64
  checkpoint_imagination: bool = True
  cuda_graphs: bool = True
  compile_model: bool = False
  fast_kinematics: bool = True
  scene_cuda_graphs: bool = True
  matmul_precision: Literal["high", "highest"] = "high"
  wm_matmul_precision: Literal["high", "highest"] = "high"

  @property
  def effective_curriculum_steps_per_iter(self) -> float:
    train_envs = self.num_envs - self.validation_envs
    return (self.actor_batch_size / self.utd) / max(1, train_envs)

  def active_exploration(self, iteration: int) -> tuple[float, float]:
    stage_noise, stage_kl = self.exploration_stages[0][1], self.exploration_stages[0][2]
    for start_iter, noise, kl in self.exploration_stages:
      if iteration >= start_iter:
        stage_noise, stage_kl = noise, kl
    noise_out = stage_noise if self.action_noise is None else self.action_noise
    kl_out = stage_kl if self.policy_target_kl is None else self.policy_target_kl
    return float(noise_out), float(kl_out)

  def validate(self) -> None:
    counts = (
      self.history_length,
      self.num_envs,
      self.validation_envs,
      self.replay_steps,
      self.warmup_steps,
      self.actor_batch_size,
      self.actor_microbatch,
      self.horizon,
      self.wm_batch_size,
      self.wm_microbatch,
      self.wm_horizon,
      self.wm_batches,
      self.critic_epochs,
      self.critic_minibatches,
      self.critic_microbatch,
      self.policy_microbatch,
      self.save_interval,
      self.validation_interval,
      self.validation_batch_size,
    )
    if min(counts) < 1 or self.iterations < 0 or not self.actor_hidden_dims or min(self.actor_hidden_dims) < 1:
      raise ValueError("All batch/env/horizon/network counts must be positive and iterations >= 0")
    if self.validation_envs >= self.num_envs:
      raise ValueError("validation_envs must be strictly smaller than num_envs")
    min_window = max(32, self.history_length, self.wm_horizon, self.horizon)
    if self.replay_steps < min_window or self.warmup_steps < min_window:
      raise ValueError(f"replay_steps and warmup_steps must be at least {min_window}")
    if not self.exploration_stages or self.exploration_stages[0][0] != 0:
      raise ValueError("exploration_stages must start at iteration 0")
    for prev_s, next_s in zip(self.exploration_stages, self.exploration_stages[1:], strict=False):
      if next_s[0] <= prev_s[0]:
        raise ValueError("exploration_stages start_iteration must be strictly increasing")
    for _, s_noise, s_kl in self.exploration_stages:
      if not math.isfinite(s_noise) or s_noise <= 0 or not math.isfinite(s_kl) or s_kl <= 0:
        raise ValueError("Invalid exploration_stages noise or target KL")
    positive = [self.learning_rate, self.wm_learning_rate, self.utd, self.save_seconds]
    if self.action_noise is not None:
      positive.append(self.action_noise)
    if self.policy_target_kl is not None:
      positive.append(self.policy_target_kl)
    if any(not math.isfinite(v) or v <= 0 for v in positive) or not 0 < self.gamma < 1 or not 0 <= self.gae_lambda <= 1:
      raise ValueError("Invalid learning rate, discount, GAE lambda, exploration std, or KL limit")
    if not math.isfinite(self.termination_probability) or not 0.0 < self.termination_probability <= 1.0:
      raise ValueError("termination_probability must lie in (0, 1]")
    if self.actor_batch_size / self.utd < self.num_envs - self.validation_envs:
      raise ValueError("actor_batch_size / utd must require at least one fresh native control per policy update")
    if not self.scratch_wm and not self.world_model:
      raise ValueError("Provide --world-model path or pass --scratch-wm True")


class VelocityWmTrainer:
  def __init__(self, cfg: VelocityWmConfig, rank: int, world_size: int, device: str):
    self.cfg = cfg
    self.rank = rank
    self.world_size = world_size
    self.device = torch.device(device)
    torch.set_float32_matmul_precision(cfg.matmul_precision)

    self.output = Path(cfg.output).resolve()
    self.output.mkdir(parents=True, exist_ok=True)
    if not cfg.resume and not cfg.preflight and (self.output / "config.json").exists():
      raise FileExistsError(f"Fresh training must use a new output directory: {self.output}")
    if dist.is_initialized():
      dist.barrier()

    self.iteration = 0
    self.controls = 0
    self.scheduled_controls = 0
    self.scheduled_actor_starts = 0
    self.scheduled_training_transitions = 0
    self.best_validation_reward = -math.inf
    self.continuation: dict[str, object] | None = None
    self.optimizer_updates = {"world_model": 0, "actor": 0, "critic": 0}
    self.started = time.monotonic()
    self.last_save = time.monotonic()
    self.stop_signal = False
    for signum in (signal.SIGTERM, signal.SIGINT):
      signal.signal(signum, self._signal)

    self.sources = _snapshot_velocity_sources(self.output, write=rank == 0)
    torch.manual_seed(cfg.seed)

    self.actor = make_actor(self.device, history_length=cfg.history_length, hidden_dims=cfg.actor_hidden_dims)
    self.critic = nn.Sequential(
      nn.Linear(CRITIC_DIM, 512),
      nn.ELU(),
      nn.Linear(512, 256),
      nn.ELU(),
      nn.Linear(256, 128),
      nn.ELU(),
      nn.Linear(128, 1),
    ).to(self.device)

    if cfg.scratch_wm:
      self.model = PositionWorldModel(PositionWorldModelCfg()).to(self.device).train()
      wm_origin: dict[str, object] = {"world_model_initialization": "scratch_fit_after_warmup"}
    else:
      wm_path = Path(cfg.world_model).resolve(strict=True)
      self.model = load_world_model(wm_path, self.device).train()
      parent = read_checkpoint(wm_path)
      wm_origin = {
        "world_model_initialization": "pretrained_v3_online",
        "world_model_checkpoint": str(wm_path),
        "world_model_sha256": _file_sha256(wm_path),
        "world_model_update": parent.get("update"),
        "world_model_best_score": parent.get("best_score"),
      }
      del parent

    _broadcast(self.actor)
    _broadcast(self.critic)
    _broadcast(self.model)

    self.origin: dict[str, object] = {
      **wm_origin,
      "policy_initialization": "scratch",
      "critic_initialization": "scratch",
      "optimizers": "fresh_Adam",
      "architecture": actor_architecture_metadata(self.actor),
      "actor_parameters": sum(p.numel() for p in self.actor.parameters()),
      "critic_parameters": sum(p.numel() for p in self.critic.parameters()),
      "world_model_parameters": sum(p.numel() for p in self.model.parameters()),
      "unlimited": cfg.iterations == 0,
    }

    init_noise, _ = cfg.active_exploration(0)
    env_seed = (cfg.seed + rank * 9973) % 2**32
    torch.manual_seed(env_seed)
    self.progress("initializing_native_velocity_env")
    self.collector = VelocityPolicyCollector(
      CollectorConfig(
        "",
        num_envs=cfg.num_envs,
        device=device,
        seed=env_seed,
        action_noise=init_noise,
        njmax=300,
      ),
      validation_envs=cfg.validation_envs,
      actor=self.actor,
      history_length=cfg.history_length,
      curriculum_steps_per_iter=cfg.effective_curriculum_steps_per_iter,
      termination_probability=cfg.termination_probability,
      policy_action_sampling=True,
      scene_cuda_graphs=cfg.scene_cuda_graphs,
    )

    self.default_position = self.collector.default_joint_pos_canonical
    self.default_lower = self.default_position[:ACTION_DIM].clone()
    self.scale_lower = self.collector.action_scale.clone()
    waist_kp, waist_kd, waist_effort = self.collector.waist_pd_parameters()

    fk = TorchKinematics(self.collector.template).to(device=self.device, dtype=torch.float32)
    fk.level_batching = cfg.fast_kinematics
    self.task = VelocityWmTask(
      kinematics=fk,
      default_position=self.default_position,
      soft_limits=self.collector.soft_joint_pos_limits_canonical,
      waist_kp=waist_kp,
      waist_kd=waist_kd,
      waist_effort_limit=waist_effort,
      termination_probability=cfg.termination_probability,
    ).to(self.device)

    native_mj = self.collector.env.sim.mj_model
    self.target_limits = torch.tensor(
      native_mj.actuator_ctrlrange[self.collector.indices["actuator"]],
      device=self.device,
      dtype=torch.float32,
    )
    self.target_limited = torch.tensor(
      native_mj.actuator_ctrllimited[self.collector.indices["actuator"]],
      device=self.device,
    ).bool()
    self.obs_noise_std = observation_noise_std(self.device, torch.float32)

    self.controller_contract = {
      "default_joint_pos_29": [float(v) for v in self.default_position.cpu().tolist()],
      "action_scale_15": [float(v) for v in self.scale_lower.cpu().tolist()],
      "target_limits_29": self.target_limits.cpu().tolist(),
      "target_limited_29": self.target_limited.cpu().tolist(),
      "training_action_std": init_noise,
      "inference_action_noise": False,
    }

    self.replay = VelocityReplay(cfg.replay_steps, cfg.num_envs, cfg.validation_envs, self.device)
    self.optimizers = {
      "world_model": torch.optim.Adam(self.model.parameters(), lr=cfg.wm_learning_rate),
      "actor": torch.optim.Adam(self.actor.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.95)),
      "critic": torch.optim.Adam(self.critic.parameters(), lr=cfg.learning_rate),
    }
    self.loss_cfg = LossConfig(gamma=cfg.gamma)
    self.forward_model = torch.compile(self._dynamics, dynamic=True) if cfg.compile_model else self._dynamics
    self._graph_pool = torch.cuda.graph_pool_handle() if cfg.cuda_graphs else None
    self._actor_graphs: dict[int, CapturedVelocityRolloutStep] = {}
    self._wm_graphs: dict[int, CapturedWmMicrobatch] = {}
    self.preflight: dict[str, float] = {}
    # Per split (train=0..6, val=7..13): reward_sum, lin_err_sum, ang_err_sum, height_err_sum, fail_sum, term_sum, count
    self.native_sums = torch.zeros(14, device=self.device, dtype=torch.float64)
    self._native_reward_checked = False

    if cfg.resume:
      self.restore(Path(cfg.resume))
      if self.continuation is None:
        raise RuntimeError("Resume did not establish checkpoint lineage")
      self.origin.update(
        policy_initialization="restored_from_parent",
        world_model_initialization="restored_online_parent",
        critic_initialization="restored_from_parent",
        optimizers="restored_Adam",
      )

    if rank == 0:
      atomic_json(
        self.output / "config.json",
        {
          "format": FORMAT,
          "config": asdict(cfg),
          "origin": self.origin,
          "continuation": self.continuation,
          "collector": self.collector.metadata,
          "controller_contract": self.controller_contract,
          "source_provenance": self.sources,
        },
      )

  def _signal(self, signum: int, frame) -> None:
    del signum, frame
    self.stop_signal = True

  def _dynamics(self, state: torch.Tensor, target: torch.Tensor, context: torch.Tensor):
    return self.model(state, target, context, validate=False)

  def progress(self, phase: str, **details) -> None:
    payload = {
      "phase": phase,
      "iteration": self.iteration,
      "optimizer_updates": self.optimizer_updates,
      "controls_per_rank": self.controls,
      "timestamp_utc": datetime.now(timezone.utc).isoformat(),
      "pid": os.getpid(),
      "rank": self.rank,
      "unlimited": self.cfg.iterations == 0,
      "collector_counters": self.collector.counters if hasattr(self, "collector") else {},
      **details,
    }
    atomic_json(self.output / ("status.json" if self.rank == 0 else f"status_rank_{self.rank}.json"), payload)
    print(json.dumps({"rank": self.rank, "phase": phase, "iteration": self.iteration, **details}), flush=True)

  @torch.no_grad()
  def collect(
    self,
    count: int,
    *,
    audit: bool = False,
    policy_rollout: VelocityPolicyRollout | None = None,
  ) -> None:
    for step in range(count):
      before = {} if audit and step == count - 1 and not self.cfg.scratch_wm else None
      original = self.collector._original_step

      def capture(before=before, original=original):
        if before is not None and not before:
          before.update(snapshot(self.collector))
        original()

      if before is not None:
        self.collector._original_step = capture
      try:
        batch, record = self.collector.collect_step(verify_packing=audit)
      finally:
        self.collector._original_step = original

      self.replay.append_control(batch, record)
      if policy_rollout is not None:
        policy_rollout.append(history=record["history"], mean=record["mean_action"])

      if audit or not self._native_reward_checked:
        executed = batch.inputs[0, :, 68:97]
        expected = velocity_target(
          record["action"],
          record["arm_target"],
          self.default_lower,
          self.scale_lower,
          record["encoder_bias"],
          self.target_limits,
          self.target_limited,
        )
        torch.testing.assert_close(executed, expected, rtol=2e-5, atol=2e-5)

      if not self._native_reward_checked:
        analytic_cont, _, _, _, _ = self.task.continuous_reward_sum(
          prev_state=batch.state[0],
          next_state=batch.next_state[0],
          target29=batch.inputs[0, :, 68:97],
          action=record["action"],
          prev_action=record["latest"][:, -ACTION_DIM:],
          twist=record["twist"],
          height_cmd=record["height_cmd"],
          lin_accum=record["aux"][:, 0:2],
          ang_accum=record["aux"][:, 2],
          cmd_elapsed=record["cmd_elapsed"],
          drift_anchor=record["post_drift_anchor"],
          is_stationary=record["is_stationary"],
          height_std=self.collector.current_height_std,
          exact_l1=True,
          include_qacc=False,
        )
        gap = float((0.02 * (analytic_cont - record["continuous_sum"])).abs().max())
        self.preflight["native_continuous_reward_max_abs_error"] = gap
        if gap > 2e-4:
          raise RuntimeError(f"Analytic/native continuous velocity reward mismatch: {gap}")
        self._native_reward_checked = True

      train = slice(0, self.replay.train_envs)
      val = slice(self.replay.train_envs, None)
      for offset, selected in ((0, train), (7, val)):
        self.native_sums[offset + 0] += record["reward"][selected].double().sum()
        self.native_sums[offset + 1] += record["lin_vel_err"][selected].double().sum()
        self.native_sums[offset + 2] += record["ang_vel_err"][selected].double().sum()
        self.native_sums[offset + 3] += record["height_err"][selected].double().sum()
        self.native_sums[offset + 4] += record["failed"][selected].double().sum()
        self.native_sums[offset + 5] += record["terminated"][selected].double().sum()
        self.native_sums[offset + 6] += float(len(record["reward"][selected]))

      self.controls += 1
      self.task.height_std = self.collector.current_height_std

      if before is not None:
        valid = batch.valid[0].clone()
        valid &= batch.segment[0] == self.collector._segment
        valid[: self.replay.train_envs] = False
        probe = native_secants(
          self.collector,
          before,
          batch.inputs[0],
          valid,
          batch.next_state[0],
          seed=self.cfg.seed + self.rank,
          repeatability_exclusion_limit=0.1,
        )
        metrics = derivative_metrics(derivative_sums(self.model, probe))
        if min(value for key, value in metrics.items() if key.endswith("/cosine")) <= 0:
          raise RuntimeError(f"Native velocity action-response direction is not positively aligned: {metrics}")
        self.preflight.update(metrics)
        self.preflight["native_derivative_states"] = float(probe["valid_states"])
        self.preflight["native_derivative_excluded_nonrepeatable"] = float(probe["excluded_nonrepeatable_states"])
        self.preflight["native_derivative_repeat_max_abs"] = float(probe["native_repeat_max_abs"])

  def policy_mean(self, history: torch.Tensor) -> torch.Tensor:
    return self.actor(history)

  def native_policy_kl(self, batch: VelocityPolicyBatch, action_noise: float) -> torch.Tensor:
    return velocity_policy_kl(self.policy_mean, batch, action_noise, self.cfg.policy_microbatch)

  def fit_model(self) -> dict[str, float]:
    previous = torch.get_float32_matmul_precision()
    try:
      torch.set_float32_matmul_precision(self.cfg.wm_matmul_precision)
      return self._fit_model()
    finally:
      torch.set_float32_matmul_precision(previous)

  def _wm_microbatch_loss(self, inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    window = ReplayWindow(
      state=inputs["state"],
      inputs=inputs["inputs"],
      next_state=inputs["next_state"],
      foot_force=inputs["foot_force"],
      env=inputs["env"],
      sequence=inputs["sequence"],
    )
    loss, metrics = autoregressive_loss(
      self.forward_model,
      window,
      self.model.state_scale,
      self.model.force_scale,
      self.loss_cfg,
    )
    weight = inputs["weight"]
    return loss * weight, metrics["state_loss"].detach() * weight

  def _fit_model(self) -> dict[str, float]:
    cfg = self.cfg
    total = 0.0
    state_loss = 0.0
    norm = 0.0
    use_graph = cfg.cuda_graphs and not cfg.compile_model
    optimizer = self.optimizers["world_model"]
    for _update in range(cfg.wm_batches):
      optimizer.zero_grad(set_to_none=not use_graph)
      for begin in range(0, cfg.wm_batch_size, cfg.wm_microbatch):
        size = min(cfg.wm_microbatch, cfg.wm_batch_size - begin)
        window = self.replay.sample(size, cfg.wm_horizon)
        weight_val = size / cfg.wm_batch_size
        if use_graph:
          inputs = {
            "state": window.state,
            "inputs": window.inputs,
            "next_state": window.next_state,
            "foot_force": window.foot_force,
            "env": window.env,
            "sequence": window.sequence,
            "weight": window.state.new_tensor(weight_val),
          }
          if size not in self._wm_graphs:
            self._wm_graphs[size] = CapturedWmMicrobatch(
              self._wm_microbatch_loss,
              inputs,
              self.model.parameters(),
              pool=self._graph_pool,
            )
            # Capture zeroed grads; re-zero before replay
            optimizer.zero_grad(set_to_none=False)
          loss_det, state_det = self._wm_graphs[size](inputs)
          total += float(loss_det)
          state_loss += float(state_det)
        else:
          loss, metrics = autoregressive_loss(
            self.forward_model,
            window,
            self.model.state_scale,
            self.model.force_scale,
            self.loss_cfg,
          )
          (loss * weight_val).backward()
          total += float(loss.detach()) * weight_val
          state_loss += float(metrics["state_loss"].detach()) * weight_val
      norm = optimizer_step(self.model.parameters(), optimizer)
      self.optimizer_updates["world_model"] += 1
    optimizer.zero_grad(set_to_none=not use_graph)
    return {
      "wm_loss": total / cfg.wm_batches,
      "wm_state_loss": state_loss / cfg.wm_batches,
      "wm_gradient_norm": norm,
    }

  def _rollout_tick(
    self,
    state: torch.Tensor,
    history: torch.Tensor,
    aux: torch.Tensor,
    context: torch.Tensor,
    encoder_bias: torch.Tensor,
    imu_bias: torch.Tensor,
    arm_target: torch.Tensor,
    twist: torch.Tensor,
    height_cmd: torch.Tensor,
    next_height_cmd: torch.Tensor,
    next_episode_step: torch.Tensor,
    phase_offset: torch.Tensor,
    cmd_elapsed: torch.Tensor,
    drift_anchor: torch.Tensor,
    is_stationary: torch.Tensor,
    obs_noise: torch.Tensor,
    act_noise: torch.Tensor,
  ) -> tuple[torch.Tensor, ...]:
    sampled_act = self.actor(history) + act_noise
    target29 = velocity_target(
      sampled_act,
      arm_target,
      self.default_lower,
      self.scale_lower,
      encoder_bias,
      self.target_limits,
      self.target_limited,
    )
    pred = self.forward_model(state, target29, context)
    reward, terminated, next_aux, foot_z = self.task.step(
      prev_state=state,
      next_state=pred.state,
      foot_force=pred.foot_force,
      target29=target29,
      action=sampled_act,
      prev_action=history[:, -1, -ACTION_DIM:],
      aux=aux,
      twist=twist,
      height_cmd=height_cmd,
      next_episode_step=next_episode_step,
      phase_offset=phase_offset,
      cmd_elapsed=cmd_elapsed,
      drift_anchor=drift_anchor,
      is_stationary=is_stationary,
    )
    next_obs = (
      velocity_observation(
        state=pred.state,
        previous_action=sampled_act,
        twist=twist,
        height_cmd=next_height_cmd,
        episode_step=next_episode_step,
        phase_offset=phase_offset,
        encoder_bias_lower=encoder_bias,
        default_lower=self.default_lower,
        imu_bias=imu_bias,
      )
      + obs_noise
    )
    next_history = torch.cat((history[:, 1:], next_obs.unsqueeze(1)), dim=1)
    return pred.state, next_history, next_aux, reward, terminated, foot_z

  def _captured_rollout_tick(self, args: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    values = dict(zip(_ROLLOUT_INPUT_NAMES, args, strict=True))
    size = len(args[0])
    if size not in self._actor_graphs:
      self._actor_graphs[size] = CapturedVelocityRolloutStep(
        lambda inp: self._rollout_tick(*inp.values()),
        values,
        self.actor.parameters(),
        pool=self._graph_pool,
      )
    return self._actor_graphs[size](values)

  def imagine(
    self,
    start: VelocityPolicyStart,
    *,
    action_noise: float,
    checkpointed: bool,
    use_cuda_graphs: bool = False,
  ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], dict[str, float]]:
    cfg = self.cfg
    n = len(start.state)
    state = start.state
    history = start.history
    context = start.context
    aux = start.aux
    arm_target = start.arm_target
    arm_ramp_steps = start.arm_ramp_steps
    arm_ramp_active = start.arm_ramp_active
    height_cmd = start.height_cmd
    episode_step = start.episode_step
    cmd_elapsed = start.cmd_elapsed

    features: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    terminations: list[torch.Tensor] = []
    truncations: list[torch.Tensor] = []

    feat0 = self.task.critic_features(
      self.model,
      state,
      context,
      history[:, -1, -ACTION_DIM:],
      aux,
      arm_target,
      start.twist,
      height_cmd,
      episode_step,
      start.phase_offset,
      cmd_elapsed,
      start.drift_anchor,
      start.is_stationary,
    )
    features.append(feat0.detach())
    values.append(self.critic(feat0).squeeze(-1))

    for step_idx in range(cfg.horizon):
      obs_noise = torch.randn(n, history.shape[-1], device=self.device) * self.obs_noise_std
      act_noise = torch.randn(n, ACTION_DIM, device=self.device) * action_noise
      with torch.no_grad():
        next_height_cmd = advance_height_command(height_cmd, start.height_target)
        next_episode_step = episode_step + 1
      step_args = (
        state,
        history,
        aux,
        context,
        start.encoder_bias,
        start.imu_bias,
        arm_target,
        start.twist,
        height_cmd,
        next_height_cmd,
        next_episode_step,
        start.phase_offset,
        cmd_elapsed,
        start.drift_anchor,
        start.is_stationary,
        obs_noise,
        act_noise,
      )
      if use_cuda_graphs:
        state, history, aux, reward, terminated, foot_z = self._captured_rollout_tick(step_args)
      elif checkpointed:
        state, history, aux, reward, terminated, foot_z = checkpoint(self._rollout_tick, *step_args, use_reentrant=False)
      else:
        state, history, aux, reward, terminated, foot_z = self._rollout_tick(*step_args)

      with torch.no_grad():
        arm_target, arm_ramp_steps, arm_ramp_active = advance_arm_ramp(
          arm_target,
          start.arm_ramp_start,
          start.arm_ramp_target,
          arm_ramp_steps,
          arm_ramp_active,
        )
        height_cmd = next_height_cmd
        episode_step = next_episode_step
        cmd_elapsed = cmd_elapsed + CONTROL_DT

      feat = self.task.critic_features(
        self.model,
        state,
        context,
        history[:, -1, -ACTION_DIM:],
        aux,
        arm_target,
        start.twist,
        height_cmd,
        episode_step,
        start.phase_offset,
        cmd_elapsed,
        start.drift_anchor,
        start.is_stationary,
        foot_z=foot_z,
      )
      features.append(feat.detach())
      values.append(self.critic(feat).squeeze(-1))
      rewards.append(reward)
      terminations.append(terminated)
      truncations.append((step_idx + 1 >= start.remaining).to(state.dtype))

    r = torch.stack(rewards, dim=1)
    d = torch.stack(terminations, dim=1)
    trunc = torch.stack(truncations, dim=1)
    v = torch.stack(values, dim=1)
    returns, survival = bptt_return(r, d, v[:, -1], cfg.gamma, truncation=trunc, bootstrap_values=v[:, 1:])
    targets = gae_targets(r.detach(), d.detach(), v.detach(), cfg.gamma, cfg.gae_lambda, truncation=trunc)
    critic_batch = (
      torch.stack(features[:-1], dim=1).flatten(0, 1),
      targets.flatten().detach(),
      survival.flatten().detach(),
    )
    return (
      -returns.mean() / cfg.horizon,
      critic_batch,
      {
        "imagined_reward": float(r.detach().mean()),
        "imagined_survival": float(survival[:, -1].detach().mean()),
        "imagined_return": float(returns.detach().mean()),
      },
    )

  def improve_policy(self, action_noise: float) -> tuple[dict[str, float], tuple[torch.Tensor, ...]]:
    cfg = self.cfg
    optimizer = self.optimizers["actor"]
    optimizer.zero_grad(set_to_none=True)
    pieces: list[tuple[torch.Tensor, ...]] = []
    metrics: dict[str, float] = {}
    with frozen(self.model, self.critic):
      for begin in range(0, cfg.actor_batch_size, cfg.actor_microbatch):
        size = min(cfg.actor_microbatch, cfg.actor_batch_size - begin)
        start = self.replay.sample_starts(size, cfg.history_length)
        loss, critic_batch, partial = self.imagine(
          start,
          action_noise=action_noise,
          checkpointed=cfg.checkpoint_imagination,
          use_cuda_graphs=cfg.cuda_graphs,
        )
        (loss * (size / cfg.actor_batch_size)).backward()
        pieces.append(critic_batch)
        for key, value in partial.items():
          metrics[key] = metrics.get(key, 0.0) + value * (size / cfg.actor_batch_size)
      metrics["actor_gradient_norm"] = optimizer_step(self.actor.parameters(), optimizer, actor=True)
    if any(p.grad is not None and p.grad.abs().max() > 0 for p in self.model.parameters()):
      raise RuntimeError("Velocity policy BPTT unexpectedly wrote nonzero WM parameter gradients")
    self.optimizer_updates["actor"] += 1
    optimizer.zero_grad(set_to_none=True)
    return metrics, tuple(torch.cat([part[i] for part in pieces]) for i in range(3))

  def audit_imagination(self) -> None:
    audit_noise, _ = self.cfg.active_exploration(self.iteration)
    start_a = self.replay.sample_starts(4, self.cfg.history_length)
    start_b = self.replay.sample_starts(4, self.cfg.history_length)
    gradients: list[torch.Tensor] = []
    with frozen(self.model, self.critic), torch.random.fork_rng(devices=[self.device]):
      if self.cfg.cuda_graphs:
        loss_warmup, _, _ = self.imagine(start_a, action_noise=audit_noise, checkpointed=True, use_cuda_graphs=True)
        _ = torch.autograd.grad(loss_warmup, tuple(self.actor.parameters()))
        del loss_warmup
      for mode in ("eager", "configured"):
        torch.manual_seed(self.cfg.seed + self.rank)
        loss, _, _ = self.imagine(
          start_b,
          action_noise=audit_noise,
          checkpointed=(mode == "configured" and self.cfg.checkpoint_imagination),
          use_cuda_graphs=(mode == "configured" and self.cfg.cuda_graphs),
        )
        grad = torch.cat([value.flatten() for value in torch.autograd.grad(loss, tuple(self.actor.parameters()))])
        if not torch.isfinite(grad).all() or grad.norm() <= 0:
          raise FloatingPointError("H32 velocity policy gradient is nonfinite or zero")
        gradients.append(grad)
        del loss
    self._actor_graphs.pop(4, None)
    relative_error = float((gradients[0] - gradients[1]).norm() / gradients[0].norm())
    if relative_error > 1e-3:
      raise RuntimeError(f"Configured/eager velocity BPTT gradients differ: {relative_error}")
    self.preflight["checkpoint_eager_gradient_relative_error"] = relative_error
    self.preflight["eager_actor_gradient_norm"] = float(gradients[0].norm())

  def fit_critic(self, batch: tuple[torch.Tensor, ...]) -> dict[str, float]:
    features, targets, weights = batch
    if any(t.requires_grad for t in batch):
      raise RuntimeError("Critic GAE targets and features must be detached")
    optimizer = self.optimizers["critic"]
    loss_sum = 0.0
    for _ in range(self.cfg.critic_epochs):
      for indices in torch.randperm(len(targets), device=self.device).tensor_split(self.cfg.critic_minibatches):
        optimizer.zero_grad(set_to_none=True)
        denominator = weights[indices].sum().clamp_min(1e-8)
        if dist.is_initialized():
          dist.all_reduce(denominator)
          denominator.div_(self.world_size)
        for micro in indices.split(self.cfg.critic_microbatch):
          error = self.critic(features[micro]).squeeze(-1) - targets[micro]
          loss = (error.square() * weights[micro]).sum() / denominator
          loss.backward()
          loss_sum += float(loss.detach())
        optimizer_step(self.critic.parameters(), optimizer)
        self.optimizer_updates["critic"] += 1
    optimizer.zero_grad(set_to_none=True)
    return {"critic_loss": loss_sum / (self.cfg.critic_epochs * self.cfg.critic_minibatches)}

  @torch.no_grad()
  def validate_model(self) -> dict[str, float]:
    candidates = self.replay.candidates(32, validation=True)
    eval_horizon = 32 if candidates.numel() > 0 else 8
    eval_steps = {"20ms": 1, "160ms": 8, **({"640ms": 32} if eval_horizon == 32 else {})}
    window = self.replay.sample(self.cfg.validation_batch_size, eval_horizon, validation=True)
    sums = physical_metric_sums(
      self.forward_model,
      window,
      evaluation_steps=eval_steps,
      delta_scale=self.model.state_scale,
    )
    if dist.is_initialized():
      for value in sums.values():
        dist.all_reduce(value)
    return {f"validation_wm/{k}": v for k, v in finalize_metrics(sums).items() if v is not None}

  def stop_requested(self) -> bool:
    value = torch.tensor(int(self.stop_signal or (self.output / "STOP").exists()), device=self.device)
    if dist.is_initialized():
      dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return bool(value)

  def required_controls(self) -> int:
    target = math.ceil((self.scheduled_actor_starts + self.cfg.actor_batch_size) / self.cfg.utd)
    remaining = max(0, target - self.scheduled_training_transitions)
    return self.scheduled_controls + math.ceil(remaining / self.replay.train_envs)

  def _checkpoint_payload(self) -> dict[str, object]:
    return {
      "format": FORMAT,
      "iteration": self.iteration,
      "rank": self.rank,
      "world_size": self.world_size,
      "config": asdict(self.cfg),
      "origin": self.origin,
      "source_provenance": self.sources,
      "collector": self.collector.metadata,
      "controller_contract": self.controller_contract,
      "actor_state_dict": self.actor.state_dict(),
      "critic_state_dict": self.critic.state_dict(),
      "world_model_state_dict": self.model.state_dict(),
      "world_model_config": asdict(self.model.cfg),
      "optimizers": {k: v.state_dict() for k, v in self.optimizers.items()},
      "optimizer_updates": self.optimizer_updates,
      "controls": self.controls,
      "scheduled_controls": self.scheduled_controls,
      "scheduled_actor_starts": self.scheduled_actor_starts,
      "scheduled_training_transitions": self.scheduled_training_transitions,
      "best_validation_reward": (
        self.best_validation_reward if math.isfinite(self.best_validation_reward) else None
      ),
      "curriculum_state": self.collector.curriculum_state_dict(),
      "collector_counters": self.collector.counters,
      "continuation": self.continuation,
      "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(self.device)},
    }

  def save(self, *, is_best: bool = False) -> None:
    directory = self.output / "checkpoints" / f"update_{self.iteration:08d}"
    directory.mkdir(parents=True, exist_ok=True)
    payload = self._checkpoint_payload()
    rank_path = directory / f"rank_{self.rank}.pt"
    atomic_checkpoint(rank_path, payload)
    if is_best:
      best_dir = self.output / "best_native"
      best_dir.mkdir(parents=True, exist_ok=True)
      shutil.copy2(rank_path, best_dir / f"rank_{self.rank}.pt")
    if dist.is_initialized():
      dist.barrier()
    if self.rank == 0:
      atomic_json(
        self.output / "latest.json",
        {
          "iteration": self.iteration,
          "world_size": self.world_size,
          "directory": str(directory.relative_to(self.output)),
          "best_validation_reward": (
            self.best_validation_reward if math.isfinite(self.best_validation_reward) else None
          ),
        },
      )
      export_onnx(payload, self.output / "latest_onnx", overwrite=True)
      if is_best:
        export_onnx(payload, self.output / "best_native", overwrite=True)
      generations = sorted((self.output / "checkpoints").glob("update_*"))
      for previous in generations[:-2]:
        for r in range(self.world_size):
          (previous / f"rank_{r}.pt").unlink(missing_ok=True)
        if not any(previous.iterdir()):
          previous.rmdir()
    self.last_save = time.monotonic()

  def restore(self, path: Path) -> None:
    if path.is_dir():
      if (path / "latest.json").is_file():
        meta = json.loads((path / "latest.json").read_text(encoding="utf-8"))
        path = path / meta["directory"] / f"rank_{self.rank}.pt"
      else:
        path = path / f"rank_{self.rank}.pt"
    saved = torch.load(path, map_location=self.device, weights_only=True)
    if saved.get("format") != FORMAT or saved["rank"] != self.rank or saved["world_size"] != self.world_size:
      raise ValueError("Velocity WM checkpoint format/rank/world_size mismatch")
    ignored = {"resume", "output", "iterations", "preflight"}
    mutable = {"termination_probability", "action_noise", "policy_target_kl", "exploration_stages"}
    expected_cfg = {k: v for k, v in asdict(self.cfg).items() if k not in ignored}
    previous_cfg = {k: v for k, v in saved["config"].items() if k not in ignored}
    expected_cfg["actor_hidden_dims"] = list(expected_cfg["actor_hidden_dims"])
    previous_cfg["actor_hidden_dims"] = list(previous_cfg["actor_hidden_dims"])
    previous_cfg.setdefault("termination_probability", 1.0)
    previous_cfg.setdefault("exploration_stages", [list(s) for s in DEFAULT_EXPLORATION_STAGES])
    expected_cfg["exploration_stages"] = [list(s) for s in expected_cfg["exploration_stages"]]
    previous_cfg["exploration_stages"] = [list(s) for s in previous_cfg["exploration_stages"]]
    changed_cfg = {
      k: {"previous": previous_cfg.get(k), "current": expected_cfg.get(k)}
      for k in mutable
      if expected_cfg.get(k) != previous_cfg.get(k)
    }
    strict_expected = {k: v for k, v in expected_cfg.items() if k not in mutable}
    strict_previous = {k: v for k, v in previous_cfg.items() if k not in mutable}
    if strict_expected != strict_previous:
      raise ValueError(f"Resume configuration changed: {strict_expected} vs {strict_previous}")
    migrated_actor, actor_rescaled = migrate_actor_state_dict(self.actor, saved["actor_state_dict"])
    self.actor.load_state_dict(migrated_actor, strict=True)
    self.critic.load_state_dict(saved["critic_state_dict"], strict=True)
    self.model.load_state_dict(saved["world_model_state_dict"], strict=True)
    for name, opt in self.optimizers.items():
      if name == "actor" and actor_rescaled:
        continue
      opt.load_state_dict(saved["optimizers"][name])
    if actor_rescaled:
      changed_cfg["actor_input_normalization"] = {"previous": "uncentered_height", "current": "centered_height_x10"}
    for field in (
      "iteration",
      "controls",
      "scheduled_controls",
      "scheduled_actor_starts",
      "scheduled_training_transitions",
      "optimizer_updates",
    ):
      setattr(self, field, saved[field])
    if saved.get("best_validation_reward") is not None and not changed_cfg:
      self.best_validation_reward = float(saved["best_validation_reward"])
    if "curriculum_state" in saved:
      self.collector.load_curriculum_state_dict(saved["curriculum_state"])
      self.task.height_std = self.collector.current_height_std
    self.collector.counters.update(saved["collector_counters"])
    cur_noise, _ = self.cfg.active_exploration(self.iteration)
    self.collector.set_action_noise(cur_noise)
    self.controller_contract["training_action_std"] = cur_noise
    self.continuation = {
      "mode": "task_continuation" if changed_cfg else "strict",
      "parent_checkpoint": str(path.resolve()),
      "parent_checkpoint_sha256": _file_sha256(path),
      "parent_iteration": saved["iteration"],
      "parent_optimizer_updates": saved["optimizer_updates"],
      "changed_config": changed_cfg,
    }
    torch.set_rng_state(saved["rng"]["torch"].cpu())
    torch.cuda.set_rng_state(saved["rng"]["cuda"].cpu(), self.device)

  def run(self) -> None:
    cfg = self.cfg
    init_noise, _ = cfg.active_exploration(self.iteration)
    self.collector.set_action_noise(init_noise)
    self.progress("warmup")
    self.collect(cfg.warmup_steps, audit=True)
    post_warmup_controls = self.controls
    if cfg.scratch_wm and not bool(self.model.normalization_ready):
      norm_meta = fit_normalization(self.model, self.replay)
      self.origin["scratch_wm_normalization"] = norm_meta
    self.native_sums.zero_()

    self.progress("validating_dynamics", compilation_enabled=cfg.compile_model)
    self.preflight.update(self.validate_model())
    self.progress("auditing_h32_gradients", compilation_enabled=cfg.compile_model)
    self.audit_imagination()
    if self.rank == 0:
      atomic_json(self.output / "preflight.json", self.preflight)
    self.save()
    if cfg.preflight:
      self.progress("completed", reason="preflight_only")
      return

    while not cfg.iterations or self.iteration < cfg.iterations:
      if self.stop_requested():
        self.save()
        self.progress("stopped", reason="user_stop")
        return

      tick = time.monotonic()
      cur_noise, cur_target_kl = cfg.active_exploration(self.iteration)
      self.collector.set_action_noise(cur_noise)
      self.controller_contract["training_action_std"] = cur_noise

      needed = self.required_controls()
      req_steps = max(1, needed - self.scheduled_controls)
      phase_started = time.monotonic()
      self.progress("collecting", requested_controls=req_steps)
      native_rollout = VelocityPolicyRollout(self.replay.train_envs)
      self.collect(req_steps, policy_rollout=native_rollout)
      self.scheduled_training_transitions += req_steps * self.replay.train_envs
      self.scheduled_controls += req_steps
      self.scheduled_actor_starts += cfg.actor_batch_size
      collect_seconds = time.monotonic() - phase_started

      for name, optimizer in self.optimizers.items():
        base = cfg.wm_learning_rate if name == "world_model" else cfg.learning_rate
        for group in optimizer.param_groups:
          group["lr"] = cosine_learning_rate(self.iteration, base, self.world_size)

      phase_started = time.monotonic()
      self.progress("world_model_start", compilation_enabled=cfg.compile_model)
      metrics = self.fit_model()
      metrics["timing/collect_seconds"] = collect_seconds
      metrics["timing/wm_seconds"] = time.monotonic() - phase_started
      metrics["policy/action_noise"] = cur_noise
      metrics["policy/target_kl"] = cur_target_kl

      policy_batch = native_rollout.finish()
      initial_kl = self.native_policy_kl(policy_batch, cur_noise)
      if not torch.isfinite(initial_kl) or initial_kl * (cur_noise**2) > 1e-5:
        raise RuntimeError(f"Native velocity KL batch is not from behavior policy: KL={float(initial_kl)}")

      phase_started = time.monotonic()
      self.progress("actor_start", compilation_enabled=cfg.compile_model)
      before_weights = [p.detach().clone() for p in self.actor.parameters()]
      adam_before = copy.deepcopy(self.optimizers["actor"].state_dict())
      actor_clock = self.optimizer_updates["actor"]
      policy_metrics, critic_batch = self.improve_policy(cur_noise)
      metrics.update(policy_metrics)
      trust = constrain_policy_kl(
        self.actor,
        self.optimizers["actor"],
        before_weights,
        adam_before,
        lambda batch=policy_batch, std=cur_noise: self.native_policy_kl(batch, std),
        cur_target_kl,
      )
      if trust["rejected"]:
        self.optimizer_updates["actor"] = actor_clock
      metrics.update({f"policy/{key}": value for key, value in trust.items()})
      metrics["timing/actor_seconds"] = time.monotonic() - phase_started

      self.progress("critic")
      phase_started = time.monotonic()
      metrics.update(self.fit_critic(critic_batch))
      metrics["timing/critic_seconds"] = time.monotonic() - phase_started
      del critic_batch, policy_batch, native_rollout, before_weights, adam_before

      self.iteration += 1
      if self.iteration == 1 or self.iteration % cfg.validation_interval == 0:
        self.progress("validation")
        phase_started = time.monotonic()
        metrics.update(self.validate_model())
        metrics["timing/validation_seconds"] = time.monotonic() - phase_started

      sums = self.native_sums.clone()
      if dist.is_initialized():
        dist.all_reduce(sums)
      for offset, label in ((0, "train"), (7, "validation")):
        count = max(float(sums[offset + 6]), 1.0)
        metrics[f"native/{label}_reward"] = float(sums[offset + 0]) / count
        metrics[f"native/{label}_lin_vel_err"] = float(sums[offset + 1]) / count
        metrics[f"native/{label}_ang_vel_err"] = float(sums[offset + 2]) / count
        metrics[f"native/{label}_height_err"] = float(sums[offset + 3]) / count
        metrics[f"native/{label}_failure_rate"] = float(sums[offset + 4]) / count
        metrics[f"native/{label}_termination_rate"] = float(sums[offset + 5]) / count
      self.native_sums.zero_()

      metrics["curriculum/height_std"] = self.collector.current_height_std
      curr_state = self.collector.curriculum_state_dict()
      for k_curr, v_curr in curr_state.items():
        if isinstance(v_curr, dict) and "stage" in v_curr:
          metrics[f"curriculum/{k_curr}_stage"] = float(v_curr["stage"])
          metrics[f"curriculum/{k_curr}_score"] = float(v_curr["score"])

      metrics.update(
        iteration=self.iteration,
        seconds=time.monotonic() - tick,
        peak_allocated_gib=torch.cuda.max_memory_allocated(self.device) / 1024**3,
        **{f"optimizer_updates/{key}": value for key, value in self.optimizer_updates.items()},
        **self.collector.counters,
      )
      if not all(math.isfinite(v) for v in metrics.values()):
        raise FloatingPointError("Nonfinite velocity WM training metric")

      val_rew = metrics["native/validation_reward"]
      is_best = (
        self.controls - post_warmup_controls >= 200
        and self.iteration % cfg.validation_interval == 0
        and val_rew > self.best_validation_reward
      )
      if is_best:
        self.best_validation_reward = val_rew

      checkpoint_started = time.monotonic()
      if (
        is_best
        or self.iteration == 1
        or self.iteration % cfg.save_interval == 0
        or time.monotonic() - self.last_save >= cfg.save_seconds
      ):
        self.save(is_best=is_best)
      metrics["timing/checkpoint_seconds"] = time.monotonic() - checkpoint_started
      metrics["wall_seconds"] = time.monotonic() - tick

      with (self.output / f"metrics_rank_{self.rank}.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(metrics, allow_nan=False) + "\n")
      if self.iteration == 1 or self.iteration % 5 == 0:
        self.progress("training", wall_seconds=metrics["wall_seconds"])
      else:
        print(json.dumps({"rank": self.rank, "iteration": self.iteration, "wall_seconds": metrics["wall_seconds"]}), flush=True)

    self.save()
    self.progress("completed", reason="bounded_iterations")


def run_velocity_wm_training(cfg: VelocityWmConfig) -> None:
  if int(os.environ.get("WORLD_SIZE", "1")) > 1 and os.environ.get("WM_GPU_PINNED") != "1":
    # Ensure EGL device matches local rank when launched directly via torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
  os.environ.setdefault("MUJOCO_GL", "egl")

  cfg.validate()
  if not torch.cuda.is_available():
    raise RuntimeError("Velocity WM training requires CUDA")

  rank = int(os.environ.get("RANK", "0"))
  world = int(os.environ.get("WORLD_SIZE", "1"))
  local = int(os.environ.get("LOCAL_RANK", "0"))
  device = f"cuda:{local}" if world > 1 else cfg.device
  torch.cuda.set_device(device)
  if world > 1 and not dist.is_initialized():
    dist.init_process_group("nccl", timeout=timedelta(minutes=30), device_id=torch.device(device))

  trainer: VelocityWmTrainer | None = None
  try:
    trainer = VelocityWmTrainer(cfg, rank, world, device)
    trainer.run()
  except BaseException as error:
    output = Path(cfg.output)
    output.mkdir(parents=True, exist_ok=True)
    failure = {
      "error": repr(error),
      "traceback": traceback.format_exc(),
      "iteration": trainer.iteration if trainer is not None else 0,
      "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(output / f"failure_rank_{rank}.json", failure)
    if trainer is not None and trainer.collector.action_failure is not None:
      diagnostic = output / f"native_action_failure_rank_{rank}.pt"
      trainer.collector.save_action_failure(diagnostic)
      atomic_json(output / f"failure_rank_{rank}.json", {**failure, "native_action_diagnostic": str(diagnostic)})
    raise
  finally:
    if trainer is not None:
      trainer.collector.close()
    if dist.is_initialized():
      dist.destroy_process_group()
