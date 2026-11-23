"""Online Position WM v3 + H36 Twin-Mirror MLP BPTT + Auto-Curriculum trainer for footstep tracking."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import pickle
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
  prune_update_checkpoints,
)
from g1_lower_rl.rl.supermimic import (
  broadcast_module as _broadcast,
)
from g1_lower_rl.rl.supermimic import (
  file_sha256 as _file_sha256,
)
from g1_lower_rl.tasks.lower_body.wm.graphs import CapturedVelocityRolloutStep, CapturedWmMicrobatch
from g1_lower_rl.tasks.lower_body.wm.task import advance_arm_ramp
from g1_lower_rl.world_model.adapters.kinematics import TorchKinematics
from g1_lower_rl.world_model.checkpoint import atomic_checkpoint, atomic_json, load_world_model, read_checkpoint
from g1_lower_rl.world_model.collection import CollectorConfig
from g1_lower_rl.world_model.model import PositionWorldModel, PositionWorldModelCfg
from g1_lower_rl.world_model.training.derivatives import derivative_metrics, derivative_sums, native_secants, snapshot
from g1_lower_rl.world_model.training.normalization import fit_normalization
from g1_lower_rl.world_model.training.objectives import (
  LossConfig,
  autoregressive_loss,
  finalize_metrics,
  physical_metric_sums,
)
from g1_lower_rl.world_model.training.replay import ReplayWindow

from .collection import FootstepPolicyCollector
from .playback import FORMAT, export_onnx
from .policy import (
  ACTION_DIM,
  DEFAULT_HISTORY,
  PROPRIO_RAW_DIM,
  FootstepPolicyBatch,
  FootstepPolicyRollout,
  FootstepTwinMirrorGruActor,
  FootstepTwinMirrorSkipGruActor,
  actor_architecture_metadata,
  command_from_world_queue,
  constrain_policy_kl,
  encode_proprio_frame,
  footstep_policy_kl,
  footstep_target,
  make_actor,
  proprio_noise_std,
  raw_obs_from_world_queue,
  raw_proprio_from_state,
)
from .task import (
  AUTO_CURRICULUM_STAGES,
  CRITIC_DIM,
  DEFAULT_TERMINATION_PROB,
  POLISH_EXPLORATION,
  CurriculumStage,
  FootstepPolicyStart,
  FootstepReplay,
  FootstepWmTask,
  advance_theta_command,
  stage_qualifies,
)

DEFAULT_PRETRAINED_WM = "logs/world_model/full_mocap22650_v3_4h_20261102/capacity8192/best.pt"
_ROLLOUT_INPUT_NAMES: tuple[str, ...] = (
  "state",
  "history",
  "aux",
  "context",
  "encoder_bias",
  "imu_bias",
  "arm_target",
  "phase",
  "end_phase",
  "frequency",
  "req_freq",
  "targets",
  "supports",
  "queue",
  "future_world",
  "first_side",
  "pending_anchor",
  "double_support",
  "arrived",
  "terminal_feet",
  "theta",
  "proprio_noise",
  "act_noise",
)


def _snapshot_footstep_sources(output_dir: Path, *, write: bool) -> dict[str, Any]:
  wm_pkg = Path(__file__).resolve().parent
  footstep_pkg = wm_pkg.parent
  repo = footstep_pkg.parent.parent.parent
  paths = [
    *sorted((repo / "g1_lower_rl/world_model").glob("*.py")),
    *sorted((repo / "g1_lower_rl/world_model/adapters").glob("*.py")),
    *sorted((repo / "g1_lower_rl/world_model/training").glob("*.py")),
    *sorted((repo / "g1_lower_rl/footsteps").glob("*.py")),
    repo / "g1_lower_rl/footstep_contract.py",
    repo / "g1_lower_rl/footstep_phase.py",
    repo / "g1_lower_rl/footstep_deploy.py",
    repo / "g1_lower_rl/rl/supermimic.py",
    repo / "g1_lower_rl/rl/footstep_model.py",
    repo / "g1_lower_rl/scene_graph.py",
    repo / "g1_lower_rl/tasks/lower_body/wm/graphs.py",
    repo / "g1_lower_rl/tasks/lower_body/wm/task.py",
    repo / "g1_lower_rl/assets/g1_gloria/__init__.py",
    *sorted((repo / "g1_lower_rl/assets/g1_gloria/xmls").glob("*.xml")),
    *sorted(footstep_pkg.rglob("*.py")),
    *(
      p
      for name in ("train.py", "play.py", "export_onnx.py", "train_footstep_wm.py")
      if (p := repo / "scripts" / name).is_file()
    ),
  ]
  contents = {str(path.relative_to(repo)): path.read_bytes() for path in sorted(set(paths))}
  hashes = {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}
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
class FootstepWmConfig:
  output: str = "logs/footstep_wm/run"
  world_model: str = DEFAULT_PRETRAINED_WM
  scratch_wm: bool = False
  resume: str | None = None
  init_ppo_actor: str | None = None
  init_actor_from: str | None = None
  init_wm_critic_from: str | None = None
  preflight: bool = False
  actor_type: Literal["mlp", "gru", "skip_gru"] = "mlp"
  gru_hidden_dim: int = 64
  gru_mlp_hidden_dims: tuple[int, ...] = (256, 256, 128)
  gru_burnin_steps: int = 24
  gru_include_prev_action: bool = False
  gru_skip_connection: bool = False
  gru_skip_hidden_dims: tuple[int, ...] = (256, 128)
  gru_skip_lr_mult: float = 10.0
  auto_curriculum: bool = True
  joint_step_episode: bool = True
  joint_task_weights: tuple[float, float] = (0.7, 0.3)
  initial_stage: int = 0
  final_polish: bool = False
  qualify_streak_required: int = 3
  device: str = "cuda:0"
  seed: int = 20261112
  iterations: int = 0
  history_length: int = DEFAULT_HISTORY
  actor_hidden_dims: tuple[int, ...] = (1024, 512, 256)
  num_envs: int = 4096
  validation_envs: int = 512
  replay_steps: int = 384
  warmup_steps: int = 64
  actor_batch_size: int = 8192
  actor_microbatch: int = 8192
  horizon: int = 36
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
  policy_target_kl: float | None = None
  imag_vel_push_prob: float = 0.0
  imag_vel_push_lin: float = 0.30
  imag_vel_push_ang: float = 0.45
  collector_vel_push_prob: float = 0.0
  collector_vel_push_lin: float = 0.45
  collector_vel_push_ang: float = 0.45
  termination_probability: float = DEFAULT_TERMINATION_PROB
  policy_microbatch: int = 16384
  save_interval: int = 200
  max_retained_checkpoints: int = 12
  critic_warmup_iterations: int = 0
  save_seconds: float = 600.0
  validation_interval: int = 10
  validation_batch_size: int = 64
  checkpoint_imagination: bool = True
  cuda_graphs: bool = True
  compile_model: bool = False
  fast_kinematics: bool = True
  scene_cuda_graphs: bool = True
  matmul_precision: Literal["high", "highest"] = "high"
  wm_matmul_precision: Literal["high", "highest"] = "high"

  def active_exploration(self, stage: CurriculumStage, *, final_polish: bool = False) -> tuple[float, float]:
    base_noise, base_kl = POLISH_EXPLORATION if final_polish else (stage.action_noise, stage.policy_target_kl)
    noise = base_noise if self.action_noise is None else self.action_noise
    kl = base_kl if self.policy_target_kl is None else self.policy_target_kl
    return float(noise), float(kl)

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
      self.qualify_streak_required,
    )
    if (
      min(counts) < 1
      or self.iterations < 0
      or self.critic_warmup_iterations < 0
      or not self.actor_hidden_dims
      or min(self.actor_hidden_dims) < 1
      or self.gru_hidden_dim < 1
      or self.gru_burnin_steps < 1
      or not self.gru_mlp_hidden_dims
      or min(self.gru_mlp_hidden_dims) < 1
    ):
      raise ValueError("All batch/env/horizon/network counts must be positive and iterations >= 0")
    if self.actor_type not in ("mlp", "gru", "skip_gru"):
      raise ValueError(f"actor_type must be 'mlp', 'gru', or 'skip_gru', got {self.actor_type!r}")
    if self.init_ppo_actor and self.actor_type != "gru":
      raise ValueError("--init-ppo-actor requires --actor-type gru")
    if self.init_ppo_actor and self.init_actor_from:
      raise ValueError("Cannot pass both --init-ppo-actor and --init-actor-from")
    if self.resume and (self.init_ppo_actor or self.init_actor_from or self.init_wm_critic_from):
      raise ValueError("--resume cannot be combined with --init-ppo-actor, --init-actor-from, or --init-wm-critic-from")
    if not 0 <= self.initial_stage < len(AUTO_CURRICULUM_STAGES):
      raise ValueError(f"initial_stage must lie in [0, {len(AUTO_CURRICULUM_STAGES) - 1}]")
    if self.validation_envs >= self.num_envs:
      raise ValueError("validation_envs must be strictly smaller than num_envs")
    gru_win = self.gru_burnin_steps if self.actor_type in ("gru", "skip_gru") else 1
    min_window = max(32, self.history_length, gru_win, self.wm_horizon, self.horizon)
    if self.replay_steps < min_window or self.warmup_steps < min_window:
      raise ValueError(f"replay_steps and warmup_steps must be at least {min_window}")
    positive = [self.learning_rate, self.wm_learning_rate, self.utd, self.save_seconds, self.gru_skip_lr_mult]
    if self.action_noise is not None:
      positive.append(self.action_noise)
    if self.policy_target_kl is not None:
      positive.append(self.policy_target_kl)
    if any(not math.isfinite(v) or v <= 0 for v in positive) or not 0 < self.gamma < 1 or not 0 <= self.gae_lambda <= 1:
      raise ValueError("Invalid learning rate, discount, GAE lambda, exploration std, or KL limit")
    if not math.isfinite(self.termination_probability) or not 0.0 < self.termination_probability <= 1.0:
      raise ValueError("termination_probability must lie in (0, 1]")
    if (
      not 0.0 <= self.imag_vel_push_prob <= 1.0
      or self.imag_vel_push_lin < 0.0
      or self.imag_vel_push_ang < 0.0
      or not 0.0 <= self.collector_vel_push_prob <= 1.0
      or self.collector_vel_push_lin < 0.0
      or self.collector_vel_push_ang < 0.0
    ):
      raise ValueError("Invalid velocity push probability or amplitude")
    if self.actor_batch_size / self.utd < self.num_envs - self.validation_envs:
      raise ValueError("actor_batch_size / utd must require at least one fresh native control per policy update")
    if len(self.joint_task_weights) != 2 or any(not math.isfinite(w) or w <= 0 for w in self.joint_task_weights):
      raise ValueError("joint_task_weights must contain two finite positive weights")
    if not math.isclose(sum(self.joint_task_weights), 1.0, abs_tol=1e-6):
      raise ValueError("joint_task_weights must sum to 1.0")
    if not self.scratch_wm and not self.world_model:
      raise ValueError("Provide --world-model path or pass --scratch-wm")


class FootstepWmTrainer:
  def __init__(self, cfg: FootstepWmConfig, rank: int, world_size: int, device: str):
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
    self.stage_index = int(cfg.initial_stage)
    self.stage_entered_iteration = 0
    self.stage_qualified_streak = 0
    self.final_polish_active = bool(cfg.final_polish)
    self.stage_history: list[dict[str, Any]] = []
    self.best_validation_reward = -math.inf
    self.best_validation_score = -math.inf
    self.continuation: dict[str, object] | None = None
    self.optimizer_updates = {"world_model": 0, "actor": 0, "critic": 0}
    self.started = time.monotonic()
    self.last_save = time.monotonic()
    self.stop_signal = False
    for signum in (signal.SIGTERM, signal.SIGINT):
      signal.signal(signum, self._signal)

    self.sources = _snapshot_footstep_sources(self.output, write=rank == 0)
    torch.manual_seed(cfg.seed)

    self.actor = make_actor(
      self.device,
      history_length=cfg.history_length,
      hidden_dims=cfg.actor_hidden_dims,
      actor_type=cfg.actor_type,
      gru_hidden_dim=cfg.gru_hidden_dim,
      gru_mlp_hidden_dims=cfg.gru_mlp_hidden_dims,
      gru_include_prev_action=cfg.gru_include_prev_action,
      gru_skip_connection=cfg.gru_skip_connection,
      gru_skip_hidden_dims=cfg.gru_skip_hidden_dims,
    )
    self.is_gru_actor = isinstance(self.actor, (FootstepTwinMirrorGruActor, FootstepTwinMirrorSkipGruActor))
    self.critic = nn.Sequential(
      nn.Linear(CRITIC_DIM, 512),
      nn.ELU(),
      nn.Linear(512, 256),
      nn.ELU(),
      nn.Linear(256, 128),
      nn.ELU(),
      nn.Linear(128, 1),
    ).to(self.device)

    policy_init_origin: dict[str, object] = {"policy_initialization": "scratch"}
    if cfg.init_ppo_actor:
      assert isinstance(self.actor, FootstepTwinMirrorGruActor)
      ppo_root = Path(cfg.init_ppo_actor).resolve(strict=True)
      if ppo_root.is_dir():
        if (ppo_root / "latest.json").is_file():
          pmeta = json.loads((ppo_root / "latest.json").read_text(encoding="utf-8"))
          ppo_path = (ppo_root / pmeta["directory"] / "rank_0.pt").resolve(strict=True)
        else:
          ppo_path = (ppo_root / "rank_0.pt").resolve(strict=True)
      else:
        ppo_path = ppo_root
      try:
        ppo_saved = torch.load(ppo_path, map_location="cpu", weights_only=True)
      except (pickle.UnpicklingError, RuntimeError):
        ppo_saved = torch.load(ppo_path, map_location="cpu", weights_only=False)
      actor_sd = ppo_saved["actor_state_dict"] if isinstance(ppo_saved, dict) and "actor_state_dict" in ppo_saved else ppo_saved
      self.actor.load_ppo_actor_state_dict(actor_sd)
      self.actor.to(self.device).train()
      policy_init_origin = {
        "policy_initialization": "warmstart_twin_mirror_gru_checkpoint",
        "ppo_actor_checkpoint": str(ppo_path),
        "ppo_actor_sha256": _file_sha256(ppo_path),
        "ppo_actor_iteration": (
          ppo_saved.get("iter", ppo_saved.get("iteration")) if isinstance(ppo_saved, dict) else None
        ),
      }
      del ppo_saved
    elif cfg.init_actor_from:
      act_path = Path(cfg.init_actor_from).resolve(strict=True)
      if act_path.is_dir():
        act_path = (act_path / "rank_0.pt").resolve(strict=True)
      act_saved = torch.load(act_path, map_location="cpu", weights_only=True)
      act_sd = act_saved["actor_state_dict"] if isinstance(act_saved, dict) and "actor_state_dict" in act_saved else act_saved
      self.actor.load_state_dict(act_sd, strict=True)
      self.actor.to(self.device).train()
      policy_init_origin = {
        "policy_initialization": "warmstart_actor_state_dict",
        "actor_init_checkpoint": str(act_path),
        "actor_init_sha256": _file_sha256(act_path),
      }
      del act_saved

    critic_init_origin: dict[str, object] = {"critic_initialization": "scratch"}
    wm_adam_Warmstart: dict[str, Any] | None = None
    if cfg.init_wm_critic_from:
      warm_root = Path(cfg.init_wm_critic_from).resolve(strict=True)
      if warm_root.is_dir():
        if (warm_root / "latest.json").is_file():
          wmeta = json.loads((warm_root / "latest.json").read_text(encoding="utf-8"))
          wdir = warm_root / wmeta["directory"]
        else:
          wdir = warm_root
        warm_file = wdir / f"rank_{self.rank}.pt"
        if not warm_file.is_file():
          warm_file = wdir / "rank_0.pt"
      else:
        warm_file = warm_root
      warm_saved = torch.load(warm_file, map_location=self.device, weights_only=True)
      if warm_saved.get("format") != FORMAT:
        raise ValueError(f"Expected {FORMAT} checkpoint for --init-wm-critic-from, got {warm_saved.get('format')}")
      self.model = PositionWorldModel(PositionWorldModelCfg(**warm_saved["world_model_config"])).to(self.device).train()
      self.model.load_state_dict(warm_saved["world_model_state_dict"], strict=True)
      self.critic.load_state_dict(warm_saved["critic_state_dict"], strict=True)
      wm_adam_Warmstart = warm_saved.get("optimizers", {}).get("world_model")
      wm_origin = {
        "world_model_initialization": "supermimic_online_warmstart",
        "world_model_checkpoint": str(warm_file),
        "world_model_sha256": _file_sha256(warm_file),
        "world_model_parent_iteration": warm_saved.get("iteration"),
      }
      critic_init_origin = {
        "critic_initialization": "supermimic_critic_warmstart",
        "critic_parent_checkpoint": str(warm_file),
        "critic_parent_iteration": warm_saved.get("iteration"),
      }
      del warm_saved
    elif cfg.scratch_wm:
      self.model = PositionWorldModel(PositionWorldModelCfg()).to(self.device).train()
      wm_origin = {"world_model_initialization": "scratch_fit_after_warmup"}
    else:
      wm_root = Path(cfg.world_model).resolve(strict=True)
      if wm_root.is_dir():
        if (wm_root / "latest.json").is_file():
          wmeta = json.loads((wm_root / "latest.json").read_text(encoding="utf-8"))
          wm_path = (wm_root / wmeta["directory"] / "rank_0.pt").resolve(strict=True)
        else:
          wm_path = (wm_root / "rank_0.pt").resolve(strict=True)
      else:
        wm_path = wm_root
      probe_wm = torch.load(wm_path, map_location="cpu", weights_only=True)
      if isinstance(probe_wm, dict) and probe_wm.get("format") == FORMAT:
        self.model = PositionWorldModel(PositionWorldModelCfg(**probe_wm["world_model_config"])).to(self.device).train()
        self.model.load_state_dict(probe_wm["world_model_state_dict"], strict=True)
        wm_origin = {
          "world_model_initialization": "supermimic_online_wm_only",
          "world_model_checkpoint": str(wm_path),
          "world_model_sha256": _file_sha256(wm_path),
          "world_model_parent_iteration": probe_wm.get("iteration"),
        }
        del probe_wm
      else:
        del probe_wm
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
    if not (cfg.init_wm_critic_from and cfg.joint_step_episode and world_size >= 2):
      _broadcast(self.critic)
    _broadcast(self.model)

    self.origin: dict[str, object] = {
      **wm_origin,
      **policy_init_origin,
      **critic_init_origin,
      "optimizers": "warmstarted_wm_Adam" if wm_adam_Warmstart is not None else "fresh_Adam",
      "architecture": actor_architecture_metadata(self.actor),
      "actor_parameters": sum(p.numel() for p in self.actor.parameters()),
      "critic_parameters": sum(p.numel() for p in self.critic.parameters()),
      "world_model_parameters": sum(p.numel() for p in self.model.parameters()),
      "unlimited": cfg.iterations == 0,
    }

    init_stage = AUTO_CURRICULUM_STAGES[self.stage_index]
    init_noise, _ = cfg.active_exploration(init_stage, final_polish=self.final_polish_active)
    env_seed = (cfg.seed + rank * 9973) % 2**32
    torch.manual_seed(env_seed)
    env_profile: Literal["walk-first", "step-episode"] = (
      "step-episode" if (cfg.joint_step_episode and world_size >= 2 and rank == 1) else "walk-first"
    )
    self.progress("initializing_native_footstep_env", env_profile=env_profile)
    self.collector = FootstepPolicyCollector(
      CollectorConfig(
        "",
        num_envs=cfg.num_envs,
        device=device,
        seed=env_seed,
        action_noise=init_noise,
        njmax=400,
      ),
      validation_envs=cfg.validation_envs,
      actor=self.actor,
      history_length=cfg.history_length,
      termination_probability=cfg.termination_probability,
      policy_action_sampling=True,
      scene_cuda_graphs=cfg.scene_cuda_graphs,
      env_profile=env_profile,
      vel_push_prob=cfg.collector_vel_push_prob,
      vel_push_lin=cfg.collector_vel_push_lin,
      vel_push_ang=cfg.collector_vel_push_ang,
    )

    self.default_position = self.collector.default_joint_pos_canonical
    self.default_lower = self.default_position[:ACTION_DIM].clone()
    self.scale_lower = self.collector.action_scale.clone()
    lower_kp, lower_kd, lower_effort = self.collector.lower_pd_parameters()
    if isinstance(self.actor, FootstepTwinMirrorGruActor):
      self.actor.set_actuator_metadata(
        default_joint_pos=self.default_position,
        action_scale=self.scale_lower,
        stiffness=lower_kp,
        damping=lower_kd,
        effort_limit=lower_effort,
      )

    fk = TorchKinematics(self.collector.template).to(device=self.device, dtype=torch.float32)
    fk.level_batching = cfg.fast_kinematics
    self.task = FootstepWmTask(
      kinematics=fk,
      default_position=self.default_position,
      lower_kp=lower_kp,
      lower_kd=lower_kd,
      lower_effort_limit=lower_effort,
      action_scale=self.scale_lower,
      hard_limits=self.collector.hard_joint_pos_limits_canonical,
      is_step_episode=self.collector.is_step_episode,
      phase_cfg=self.collector.phase_cfg,
      termination_probability=cfg.termination_probability,
    ).to(self.device)
    self._apply_stage(init_stage)

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
    self.proprio_noise_std = proprio_noise_std(self.device, torch.float32)

    self.controller_contract = {
      "default_joint_pos_29": [float(v) for v in self.default_position.cpu().tolist()],
      "action_scale_15": [float(v) for v in self.scale_lower.cpu().tolist()],
      "target_limits_29": self.target_limits.cpu().tolist(),
      "target_limited_29": self.target_limited.cpu().tolist(),
      "training_action_std": init_noise,
      "inference_action_noise": False,
    }

    self.replay = FootstepReplay(
      cfg.replay_steps,
      cfg.num_envs,
      cfg.validation_envs,
      self.device,
      include_gru_state=self.is_gru_actor,
      gru_hidden_state_dim=(
        self.actor.hidden_state_dim
        if isinstance(self.actor, (FootstepTwinMirrorGruActor, FootstepTwinMirrorSkipGruActor))
        else 128
      ),
    )
    actor_params: Any = (
      self.actor.optimizer_param_groups(cfg.learning_rate, cfg.gru_skip_lr_mult)
      if isinstance(self.actor, FootstepTwinMirrorGruActor)
      else self.actor.parameters()
    )
    self.optimizers = {
      "world_model": torch.optim.Adam(self.model.parameters(), lr=cfg.wm_learning_rate),
      "actor": torch.optim.Adam(actor_params, lr=cfg.learning_rate, betas=(0.9, 0.95)),
      "critic": torch.optim.Adam(self.critic.parameters(), lr=cfg.learning_rate),
    }
    if wm_adam_Warmstart is not None:
      self.optimizers["world_model"].load_state_dict(wm_adam_Warmstart)
      for group in self.optimizers["world_model"].param_groups:
        group["lr"] = cfg.wm_learning_rate
    self.loss_cfg = LossConfig(gamma=cfg.gamma, outlier_huber_delta=8.0)
    self.forward_model = torch.compile(self._dynamics, dynamic=True) if cfg.compile_model else self._dynamics
    self._graph_pool = torch.cuda.graph_pool_handle() if cfg.cuda_graphs else None
    self._actor_graphs: dict[int, CapturedVelocityRolloutStep] = {}
    self._wm_graphs: dict[int, CapturedWmMicrobatch] = {}
    self.preflight: dict[str, float] = {}
    # Per split (train=0..9, val=10..19) + deterministic Viser survival (20..23) + step-episode success (24..25):
    self.native_sums = torch.zeros(26, device=self.device, dtype=torch.float64)

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

  @property
  def current_stage(self) -> CurriculumStage:
    return AUTO_CURRICULUM_STAGES[self.stage_index]

  def _apply_stage(self, stage: CurriculumStage) -> None:
    self.collector.apply_stage(stage)
    self.task.apply_stage(stage)

  def _signal(self, signum: int, frame) -> None:
    del signum, frame
    self.stop_signal = True

  def _dynamics(self, state: torch.Tensor, target: torch.Tensor, context: torch.Tensor):
    return self.model(state, target, context, validate=False)

  def progress(self, phase: str, **details) -> None:
    payload = {
      "phase": phase,
      "iteration": self.iteration,
      "stage_index": self.stage_index,
      "stage_name": self.current_stage.name,
      "final_polish_active": self.final_polish_active,
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
    print(
      json.dumps(
        {
          "rank": self.rank,
          "phase": phase,
          "iteration": self.iteration,
          "stage": f"{self.stage_index}:{self.current_stage.name}",
          **details,
        }
      ),
      flush=True,
    )

  @torch.no_grad()
  def collect(
    self,
    count: int,
    *,
    audit: bool = False,
    policy_rollout: FootstepPolicyRollout | None = None,
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
        policy_rollout.append(
          history=record["policy_input_a"],
          command=record["policy_input_b"],
          mean=record["mean_action"],
        )

      if audit:
        executed = batch.inputs[0, :, 68:97]
        expected = footstep_target(
          record["action"],
          record["arm_target"],
          self.default_lower,
          self.scale_lower,
          record["encoder_bias"][:, :ACTION_DIM],
          self.target_limits,
          self.target_limited,
        )
        torch.testing.assert_close(executed, expected, rtol=2e-5, atol=2e-5)

      train = slice(0, self.replay.train_envs)
      det = slice(self.collector.det_val_start, None)
      splits = ((0, train),) if self.collector.is_step_episode else ((0, train), (10, det))
      for offset, sel in splits:
        self.native_sums[offset + 0] += record["reward"][sel].double().sum()
        self.native_sums[offset + 1] += record["stance_xy_err"][sel].double().sum()
        self.native_sums[offset + 2] += record["stance_yaw_err"][sel].double().sum()
        self.native_sums[offset + 3] += record["theta_err"][sel].double().sum()
        self.native_sums[offset + 4] += record["failed"][sel].double().sum()
        self.native_sums[offset + 5] += record["terminated"][sel].double().sum()
        self.native_sums[offset + 6] += float(len(record["reward"][sel]))
        self.native_sums[offset + 7] += record["ss_match"][sel].double().sum()
        self.native_sums[offset + 8] += record["swing_lift"][sel].double().sum()
        self.native_sums[offset + 9] += record["ss_active"][sel].double().sum()

      if not self.collector.is_step_episode:
        self.native_sums[20] += record["healthy"][det].double().sum()
        self.native_sums[21] += record["first_fail"][det].double().sum()
        self.native_sums[22] += record["unbroken_s"][det].double().sum()
        self.native_sums[23] += float(len(record["reward"][det]))
      else:
        self.native_sums[24] += record["step_ep_success"][det].double().sum()
        self.native_sums[25] += record["step_ep_done"][det].double().sum()

      self.controls += 1

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
          raise RuntimeError(f"Native footstep action-response direction is not positively aligned: {metrics}")
        self.preflight.update(metrics)
        self.preflight["native_derivative_states"] = float(probe["valid_states"])
        self.preflight["native_derivative_excluded_nonrepeatable"] = float(probe["excluded_nonrepeatable_states"])
        self.preflight["native_derivative_repeat_max_abs"] = float(probe["native_repeat_max_abs"])

  def policy_mean(self, history: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
    if isinstance(self.actor, FootstepTwinMirrorSkipGruActor):
      h_dim = self.actor.hidden_state_dim
      return self.actor.step(history[:, h_dim:], command, history[:, :h_dim])[0]
    if isinstance(self.actor, FootstepTwinMirrorGruActor):
      h_dim = self.actor.hidden_state_dim
      pa = history[:, h_dim:] if history.shape[-1] > h_dim else None
      return self.actor.step(command, history[:, :h_dim], pa)[0]
    return self.actor(history, command)

  def native_policy_kl(self, batch: FootstepPolicyBatch, action_noise: float) -> torch.Tensor:
    return footstep_policy_kl(self.policy_mean, batch, action_noise, self.cfg.policy_microbatch)

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
    wm_skipped = 0
    for _ in range(cfg.wm_batches):
      optimizer.zero_grad(set_to_none=not use_graph)
      batch_loss = 0.0
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
            optimizer.zero_grad(set_to_none=False)
          loss_det, state_det = self._wm_graphs[size](inputs)
          batch_loss += float(loss_det)
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
          batch_loss += float(loss.detach()) * weight_val
          state_loss += float(metrics["state_loss"].detach()) * weight_val
      total += batch_loss
      skip_flag = torch.tensor(
        int(not math.isfinite(batch_loss)),
        device=self.device,
        dtype=torch.int64,
      )
      if dist.is_initialized():
        dist.all_reduce(skip_flag, op=dist.ReduceOp.MAX)
      if int(skip_flag.item()) != 0:
        wm_skipped += 1
        continue
      max_wm_norm = 30.0 if self.iteration > 50 else None
      norm = optimizer_step(self.model.parameters(), optimizer, max_raw_norm=max_wm_norm)
      if max_wm_norm is not None and norm > max_wm_norm:
        wm_skipped += 1
      else:
        self.optimizer_updates["world_model"] += 1
    optimizer.zero_grad(set_to_none=not use_graph)
    return {
      "wm_loss": total / cfg.wm_batches,
      "wm_state_loss": state_loss / cfg.wm_batches,
      "wm_gradient_norm": norm,
      "wm_outlier_skipped": float(wm_skipped),
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
    phase: torch.Tensor,
    end_phase: torch.Tensor,
    frequency: torch.Tensor,
    req_freq: torch.Tensor,
    targets: torch.Tensor,
    supports: torch.Tensor,
    queue: torch.Tensor,
    future_world: torch.Tensor,
    first_side: torch.Tensor,
    pending_anchor: torch.Tensor,
    double_support: torch.Tensor,
    arrived: torch.Tensor,
    terminal_feet: torch.Tensor,
    theta: torch.Tensor,
    proprio_noise: torch.Tensor,
    act_noise: torch.Tensor,
  ) -> tuple[torch.Tensor, ...]:
    if isinstance(self.actor, FootstepTwinMirrorSkipGruActor):
      h_dim = self.actor.hidden_state_dim
      gru_h = history[:, :h_dim]
      cur_frame = history[:, h_dim:]
      prev_action = cur_frame[:, 70:85]
      cmd = command_from_world_queue(phase, frequency, queue, first_side, aux[:, 6:9], theta)
      mean_act, next_gru_h = self.actor.step(cur_frame, cmd, gru_h)
      sampled_act = mean_act + act_noise
      is_skip_gru = True
    elif isinstance(self.actor, FootstepTwinMirrorGruActor):
      h_dim = self.actor.hidden_state_dim
      gru_h = history[:, :h_dim]
      cur_raw_proprio = history[:, h_dim : h_dim + PROPRIO_RAW_DIM]
      prev_action = history[:, h_dim + PROPRIO_RAW_DIM :]
      raw_obs_79 = raw_obs_from_world_queue(
        cur_raw_proprio, phase, frequency, queue, first_side, aux[:, 6:9], theta
      )
      mean_act, next_gru_h = self.actor.step(raw_obs_79, gru_h, prev_action)
      sampled_act = mean_act + act_noise
      is_skip_gru = False
    else:
      next_gru_h = None
      prev_action = history[:, -1, 70:85]
      cmd = command_from_world_queue(phase, frequency, queue, first_side, aux[:, 6:9], theta)
      sampled_act = self.actor(history, cmd) + act_noise
      is_skip_gru = False
    target29 = footstep_target(
      sampled_act,
      arm_target,
      self.default_lower,
      self.scale_lower,
      encoder_bias[:, :ACTION_DIM],
      self.target_limits,
      self.target_limited,
    )
    pred = self.forward_model(state, target29, context)
    reward, terminated, next_aux, frame, _ = self.task.step(
      prev_state=state,
      next_state=pred.state,
      foot_force=pred.foot_force,
      action=sampled_act,
      prev_action=prev_action,
      aux=aux,
      end_phase=end_phase,
      frequency=frequency,
      targets=targets,
      supports=supports,
      future_world=future_world,
      pending_anchor=pending_anchor,
      double_support=double_support,
      theta=theta,
      arrived=arrived,
      terminal_feet=terminal_feet,
      first_side=first_side,
      req_freq=req_freq,
      encoder_bias_lower=encoder_bias[:, :ACTION_DIM],
    )
    torso_ang_w = frame.fk.body_angular_velocity[:, self.task.torso_id, :]
    next_raw_proprio = raw_proprio_from_state(pred.state, encoder_bias, frame.torso_q, torso_ang_w, imu_bias) + proprio_noise
    if is_skip_gru and next_gru_h is not None:
      next_frame = encode_proprio_frame(next_raw_proprio, sampled_act, self.actor.default_joint_pos)
      next_history = torch.cat((next_gru_h, next_frame), dim=-1)
    elif next_gru_h is not None:
      next_history = torch.cat((next_gru_h, next_raw_proprio, sampled_act), dim=-1)
    else:
      next_frame = encode_proprio_frame(next_raw_proprio, sampled_act, self.actor.default_joint_pos)
      next_history = torch.cat((history[:, 1:], next_frame.unsqueeze(1)), dim=1)
    p_fz_anchor = torch.sigmoid((pred.foot_force[:, :, 2].clamp_min(0.0) - 12.0) / 4.0)
    return pred.state, next_history, next_aux, reward, terminated, frame.foot_pos, frame.foot_yaw, p_fz_anchor

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
    start: FootstepPolicyStart,
    *,
    action_noise: float,
    checkpointed: bool,
    use_cuda_graphs: bool = False,
    allow_vel_push: bool = False,
  ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], dict[str, float]]:
    cfg = self.cfg
    n = len(start.state)
    state = start.state
    if isinstance(self.actor, FootstepTwinMirrorSkipGruActor):
      if start.gru_h is None:
        raise RuntimeError("Skip-GRU imagination start requires gru_h")
      h_dim = self.actor.hidden_state_dim
      history = torch.cat((start.gru_h, start.history[:, -1]), dim=-1)
      prev_act0 = history[:, h_dim + PROPRIO_RAW_DIM :]
    elif isinstance(self.actor, FootstepTwinMirrorGruActor):
      if start.gru_h is None or start.raw_proprio_70 is None:
        raise RuntimeError("GRU imagination start requires gru_h and raw_proprio_70")
      h_dim = self.actor.hidden_state_dim
      history = torch.cat((start.gru_h, start.raw_proprio_70, start.history[:, -1, 70:85]), dim=-1)
      prev_act0 = history[:, h_dim + PROPRIO_RAW_DIM :]
    else:
      h_dim = 0
      history = start.history
      prev_act0 = history[:, -1, 70:85]
    context = start.context
    aux = start.aux
    arm_target = start.arm_target
    arm_ramp_steps = start.arm_ramp_steps
    arm_ramp_active = start.arm_ramp_active
    phase = start.phase
    frequency = start.frequency
    mode = start.mode
    elapsed = start.elapsed
    targets = start.targets
    supports = start.supports
    queue = start.queue
    first_side = start.first_side
    pending_anchor = start.pending_anchor
    theta = start.theta
    theta_active = float(self.task.stage_params[6].item())

    features: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    terminations: list[torch.Tensor] = []
    truncations: list[torch.Tensor] = []

    feat0 = self.task.critic_features(
      self.model,
      state,
      context,
      prev_act0,
      aux,
      arm_target,
      phase,
      frequency,
      targets,
      supports,
      queue,
      first_side,
      theta,
    )
    features.append(feat0.detach())
    values.append(self.critic(feat0).squeeze(-1))

    sides = self.task.sides
    push_step = min(6, max(1, cfg.horizon // 5))
    for step_idx in range(cfg.horizon):
      if allow_vel_push and cfg.imag_vel_push_prob > 0.0 and step_idx == push_step:
        with torch.no_grad():
          push_mask = (torch.rand(n, 1, device=self.device) < cfg.imag_vel_push_prob).to(state.dtype)
          dv_xy = (torch.rand(n, 2, device=self.device) * 2.0 - 1.0) * cfg.imag_vel_push_lin
          dv_z = (torch.rand(n, 1, device=self.device) * 2.0 - 1.0) * (0.20 * cfg.imag_vel_push_lin)
          dw = (torch.rand(n, 3, device=self.device) * 2.0 - 1.0) * cfg.imag_vel_push_ang
          delta_state = torch.zeros_like(state)
          delta_state[:, 36:39] = torch.cat((dv_xy, dv_z), dim=-1) * push_mask
          delta_state[:, 39:42] = dw * push_mask
        state = state + delta_state
      proprio_noise = torch.randn(n, PROPRIO_RAW_DIM, device=self.device) * self.proprio_noise_std
      act_noise = torch.randn(n, ACTION_DIM, device=self.device) * action_noise
      with torch.no_grad():
        indices = (sides[None, :] != first_side[:, None]).long()
        future_world = queue.gather(1, indices[:, :, None].expand(-1, -1, 3))
        (
          end_phase,
          next_freq,
          next_mode,
          next_elapsed,
          next_targets,
          next_supports,
          next_queue,
          next_first_side,
          cand_pending,
          double_support,
          arrived_f,
        ) = self.task.advance_queue_schedule(
          phase,
          frequency,
          mode,
          elapsed,
          start.req_freq,
          start.stop_init_freq,
          start.stop_ramp_start,
          start.stop_phase,
          start.start_dur,
          start.stop_dur,
          start.terminal_feet,
          targets,
          supports,
          queue,
          first_side,
          pending_anchor,
        )

      step_args = (
        state,
        history,
        aux,
        context,
        start.encoder_bias,
        start.imu_bias,
        arm_target,
        phase,
        end_phase,
        frequency,
        start.req_freq,
        targets,
        supports,
        queue,
        future_world,
        first_side,
        cand_pending,
        double_support,
        arrived_f,
        start.terminal_feet,
        theta,
        proprio_noise,
        act_noise,
      )
      if use_cuda_graphs:
        state, history, aux, reward, terminated, foot_pos, foot_yaw, p_fz_anchor = self._captured_rollout_tick(step_args)
      elif checkpointed:
        state, history, aux, reward, terminated, foot_pos, foot_yaw, p_fz_anchor = checkpoint(
          self._rollout_tick, *step_args, use_reentrant=False
        )
      else:
        state, history, aux, reward, terminated, foot_pos, foot_yaw, p_fz_anchor = self._rollout_tick(*step_args)

      with torch.no_grad():
        pending_clamped = cand_pending.clamp_min(0)
        pending_p = p_fz_anchor.gather(1, pending_clamped[:, None]).squeeze(1)
        clear_pending = ((cand_pending >= 0) & double_support & (pending_p > 0.90)) | (
          (cand_pending >= 0) & (~double_support) & (pending_p > 0.35)
        )
        pending_anchor = torch.where(clear_pending, torch.full_like(cand_pending, -1), cand_pending)

        phase = end_phase
        frequency = next_freq
        mode = next_mode
        elapsed = next_elapsed
        targets = next_targets
        supports = next_supports
        queue = next_queue
        first_side = next_first_side
        arm_target, arm_ramp_steps, arm_ramp_active = advance_arm_ramp(
          arm_target,
          start.arm_ramp_start,
          start.arm_ramp_target,
          arm_ramp_steps,
          arm_ramp_active,
        )
        theta = advance_theta_command(theta, start.theta_target, start.theta_speed, theta_active)

      cur_prev_act = history[:, h_dim + PROPRIO_RAW_DIM :] if self.is_gru_actor else history[:, -1, 70:85]
      feat = self.task.critic_features(
        self.model,
        state,
        context,
        cur_prev_act,
        aux,
        arm_target,
        phase,
        frequency,
        targets,
        supports,
        queue,
        first_side,
        theta,
        foot_pos=foot_pos,
        foot_yaw=foot_yaw,
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
    targets_gae = gae_targets(r.detach(), d.detach(), v.detach(), cfg.gamma, cfg.gae_lambda, truncation=trunc)
    critic_batch = (
      torch.stack(features[:-1], dim=1).flatten(0, 1),
      targets_gae.flatten().detach(),
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

  def _sample_actor_starts(self, batch_size: int) -> FootstepPolicyStart:
    is_rec = isinstance(self.actor, (FootstepTwinMirrorGruActor, FootstepTwinMirrorSkipGruActor))
    win_len = self.cfg.gru_burnin_steps if is_rec else self.cfg.history_length
    return self.replay.sample_starts(
      batch_size,
      win_len,
      prefer_stepping=self.collector.is_step_episode,
      burnin_gru_actor=(
        self.actor
        if isinstance(self.actor, (FootstepTwinMirrorGruActor, FootstepTwinMirrorSkipGruActor))
        else None
      ),
    )

  def improve_policy(
    self,
    action_noise: float,
    *,
    update_actor: bool = True,
  ) -> tuple[dict[str, float], tuple[torch.Tensor, ...]]:
    cfg = self.cfg
    optimizer = self.optimizers["actor"]
    optimizer.zero_grad(set_to_none=True)
    pieces: list[tuple[torch.Tensor, ...]] = []
    metrics: dict[str, float] = {}
    if cfg.joint_step_episode and self.world_size == 2:
      step_w = min(float(cfg.joint_task_weights[1]), 0.05) if self.stage_index < 3 else float(cfg.joint_task_weights[1])
      task_w = step_w if self.collector.is_step_episode else (1.0 - step_w)
      rank_scale = float(self.world_size) * task_w
    else:
      rank_scale = 1.0
    if not update_actor:
      with torch.no_grad():
        for begin in range(0, cfg.actor_batch_size, cfg.actor_microbatch):
          size = min(cfg.actor_microbatch, cfg.actor_batch_size - begin)
          start = self._sample_actor_starts(size)
          _, critic_batch, partial = self.imagine(
            start,
            action_noise=action_noise,
            checkpointed=False,
            use_cuda_graphs=False,
            allow_vel_push=True,
          )
          pieces.append(critic_batch)
          for key, value in partial.items():
            metrics[key] = metrics.get(key, 0.0) + value * (size / cfg.actor_batch_size)
      metrics["actor_gradient_norm"] = 0.0
      return metrics, tuple(torch.cat([part[i] for part in pieces]) for i in range(3))

    with frozen(self.model, self.critic):
      for begin in range(0, cfg.actor_batch_size, cfg.actor_microbatch):
        size = min(cfg.actor_microbatch, cfg.actor_batch_size - begin)
        start = self._sample_actor_starts(size)
        loss, critic_batch, partial = self.imagine(
          start,
          action_noise=action_noise,
          checkpointed=cfg.checkpoint_imagination,
          use_cuda_graphs=cfg.cuda_graphs,
          allow_vel_push=True,
        )
        (loss * rank_scale * (size / cfg.actor_batch_size)).backward()
        pieces.append(critic_batch)
        for key, value in partial.items():
          metrics[key] = metrics.get(key, 0.0) + value * (size / cfg.actor_batch_size)
      bad_grad = int(
        any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.actor.parameters() if p.requires_grad)
      )
      bad_flag = torch.tensor(bad_grad, device=self.device, dtype=torch.int64)
      if dist.is_initialized():
        dist.all_reduce(bad_flag, op=dist.ReduceOp.MAX)
      if int(bad_flag.item()) > 0:
        optimizer.zero_grad(set_to_none=True)
        metrics["actor_gradient_norm"] = 0.0
        metrics["actor_grad_overflow_skipped"] = 1.0
        return metrics, tuple(torch.cat([part[i] for part in pieces]) for i in range(3))
      metrics["actor_grad_overflow_skipped"] = 0.0
      metrics["actor_gradient_norm"] = optimizer_step(self.actor.parameters(), optimizer, actor=True)
    if any(p.grad is not None and p.grad.abs().max() > 0 for p in self.model.parameters()):
      raise RuntimeError("Footstep policy BPTT unexpectedly wrote nonzero WM parameter gradients")
    self.optimizer_updates["actor"] += 1
    optimizer.zero_grad(set_to_none=True)
    return metrics, tuple(torch.cat([part[i] for part in pieces]) for i in range(3))

  def audit_imagination(self) -> None:
    audit_noise, _ = self.cfg.active_exploration(self.current_stage, final_polish=self.final_polish_active)
    start_a = self._sample_actor_starts(4)
    start_b = self._sample_actor_starts(4)
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
          raise FloatingPointError("H36 footstep policy gradient is nonfinite or zero")
        gradients.append(grad)
        del loss
    self._actor_graphs.pop(4, None)
    relative_error = float((gradients[0] - gradients[1]).norm() / gradients[0].norm())
    if relative_error > 1e-3:
      raise RuntimeError(f"Configured/eager footstep BPTT gradients differ: {relative_error}")
    self.preflight["checkpoint_eager_gradient_relative_error"] = relative_error
    self.preflight["eager_actor_gradient_norm"] = float(gradients[0].norm())

  def fit_critic(self, batch: tuple[torch.Tensor, ...]) -> dict[str, float]:
    features, targets, weights = batch
    if any(t.requires_grad for t in batch):
      raise RuntimeError("Critic GAE targets and features must be detached")
    optimizer = self.optimizers["critic"]
    private_critic = self.cfg.joint_step_episode and self.world_size >= 2
    loss_sum = 0.0
    for _ in range(self.cfg.critic_epochs):
      for indices in torch.randperm(len(targets), device=self.device).tensor_split(self.cfg.critic_minibatches):
        optimizer.zero_grad(set_to_none=True)
        denominator = weights[indices].sum().clamp_min(1e-8)
        if dist.is_initialized() and not private_critic:
          dist.all_reduce(denominator)
          denominator.div_(self.world_size)
        for micro in indices.split(self.cfg.critic_microbatch):
          error = self.critic(features[micro]).squeeze(-1) - targets[micro]
          loss = (
            torch.nn.functional.smooth_l1_loss(error, torch.zeros_like(error), reduction="none", beta=2.0)
            * weights[micro]
          ).sum() / denominator
          loss.backward()
          loss_sum += float(loss.detach())
        if private_critic:
          nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0, error_if_nonfinite=True)
          optimizer.step()
        else:
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
      "stage_index": self.stage_index,
      "stage_name": self.current_stage.name,
      "stage_entered_iteration": self.stage_entered_iteration,
      "stage_qualified_streak": self.stage_qualified_streak,
      "final_polish_active": self.final_polish_active,
      "stage_history": self.stage_history,
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
      "best_validation_score": (
        self.best_validation_score if math.isfinite(self.best_validation_score) else None
      ),
      "collector_counters": self.collector.counters,
      "continuation": self.continuation,
      "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(self.device)},
    }

  def save(self, *, is_best: bool = False, milestone: str | None = None) -> None:
    directory = self.output / "checkpoints" / f"update_{self.iteration:08d}"
    directory.mkdir(parents=True, exist_ok=True)
    payload = self._checkpoint_payload()
    rank_path = directory / f"rank_{self.rank}.pt"
    atomic_checkpoint(rank_path, payload)
    if is_best:
      best_dir = self.output / "best_native"
      best_dir.mkdir(parents=True, exist_ok=True)
      shutil.copy2(rank_path, best_dir / f"rank_{self.rank}.pt")
    if milestone is not None:
      mile_dir = self.output / "checkpoints" / milestone
      mile_dir.mkdir(parents=True, exist_ok=True)
      shutil.copy2(rank_path, mile_dir / f"rank_{self.rank}.pt")
    if dist.is_initialized():
      dist.barrier()
    if self.rank == 0:
      atomic_json(
        self.output / "latest.json",
        {
          "iteration": self.iteration,
          "stage_index": self.stage_index,
          "stage_name": self.current_stage.name,
          "final_polish_active": self.final_polish_active,
          "world_size": self.world_size,
          "directory": str(directory.relative_to(self.output)),
          "best_validation_reward": (
            self.best_validation_reward if math.isfinite(self.best_validation_reward) else None
          ),
          "best_validation_score": (
            self.best_validation_score if math.isfinite(self.best_validation_score) else None
          ),
        },
      )
      if is_best:
        export_onnx(payload, self.output / "best_native", overwrite=True)
      prune_update_checkpoints(
        self.output / "checkpoints",
        save_interval=self.cfg.save_interval,
        max_retained=self.cfg.max_retained_checkpoints,
        world_size=self.world_size,
      )
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
      raise ValueError("Footstep WM checkpoint format/rank/world_size mismatch")
    ignored = {
      "resume",
      "output",
      "iterations",
      "preflight",
      "critic_warmup_iterations",
      "init_ppo_actor",
      "init_actor_from",
      "init_wm_critic_from",
    }
    mutable = {
      "horizon",
      "termination_probability",
      "learning_rate",
      "wm_learning_rate",
      "wm_batches",
      "gru_burnin_steps",
      "gru_skip_lr_mult",
      "action_noise",
      "policy_target_kl",
      "imag_vel_push_prob",
      "imag_vel_push_lin",
      "imag_vel_push_ang",
      "collector_vel_push_prob",
      "collector_vel_push_lin",
      "collector_vel_push_ang",
      "auto_curriculum",
      "joint_step_episode",
      "joint_task_weights",
      "initial_stage",
      "final_polish",
      "save_interval",
      "max_retained_checkpoints",
      "save_seconds",
    }
    expected_cfg = {k: v for k, v in asdict(self.cfg).items() if k not in ignored}
    previous_cfg = {k: v for k, v in saved["config"].items() if k not in ignored}
    previous_cfg.setdefault("actor_type", "mlp")
    previous_cfg.setdefault("gru_hidden_dim", 64)
    previous_cfg.setdefault("gru_mlp_hidden_dims", [256, 256, 128])
    previous_cfg.setdefault("gru_burnin_steps", 24)
    previous_cfg.setdefault("gru_include_prev_action", False)
    previous_cfg.setdefault("gru_skip_connection", False)
    previous_cfg.setdefault("gru_skip_hidden_dims", [256, 128])
    previous_cfg.setdefault("gru_skip_lr_mult", 10.0)
    previous_cfg.setdefault("imag_vel_push_prob", 0.0)
    previous_cfg.setdefault("imag_vel_push_lin", 0.30)
    previous_cfg.setdefault("imag_vel_push_ang", 0.45)
    previous_cfg.setdefault("collector_vel_push_prob", 0.0)
    previous_cfg.setdefault("collector_vel_push_lin", 0.45)
    previous_cfg.setdefault("collector_vel_push_ang", 0.45)
    previous_cfg.setdefault("final_polish", False)
    previous_cfg.setdefault("joint_step_episode", False)
    previous_cfg.setdefault("joint_task_weights", [0.5, 0.5])
    previous_cfg.setdefault("max_retained_checkpoints", 12)
    expected_cfg["actor_hidden_dims"] = list(expected_cfg["actor_hidden_dims"])
    previous_cfg["actor_hidden_dims"] = list(previous_cfg["actor_hidden_dims"])
    expected_cfg["gru_mlp_hidden_dims"] = list(expected_cfg["gru_mlp_hidden_dims"])
    previous_cfg["gru_mlp_hidden_dims"] = list(previous_cfg["gru_mlp_hidden_dims"])
    expected_cfg["gru_skip_hidden_dims"] = list(expected_cfg["gru_skip_hidden_dims"])
    previous_cfg["gru_skip_hidden_dims"] = list(previous_cfg["gru_skip_hidden_dims"])
    expected_cfg["joint_task_weights"] = list(expected_cfg["joint_task_weights"])
    previous_cfg["joint_task_weights"] = list(previous_cfg["joint_task_weights"])
    changed_cfg = {
      k: {"previous": previous_cfg.get(k), "current": expected_cfg.get(k)}
      for k in mutable
      if expected_cfg.get(k) != previous_cfg.get(k)
    }
    strict_expected = {k: v for k, v in expected_cfg.items() if k not in mutable}
    strict_previous = {k: v for k, v in previous_cfg.items() if k not in mutable}
    if strict_expected != strict_previous:
      raise ValueError(f"Resume configuration changed: {strict_expected} vs {strict_previous}")

    self.actor.load_state_dict(saved["actor_state_dict"], strict=True)
    self.critic.load_state_dict(saved["critic_state_dict"], strict=True)
    self.model.load_state_dict(saved["world_model_state_dict"], strict=True)
    for name, opt in self.optimizers.items():
      opt.load_state_dict(saved["optimizers"][name])
    if isinstance(self.actor, FootstepTwinMirrorGruActor) and self.actor.skip_net is not None and len(self.optimizers["actor"].param_groups) > 1:
      self.optimizers["actor"].param_groups[0]["lr_mult"] = 1.0
      self.optimizers["actor"].param_groups[1]["lr_mult"] = float(self.cfg.gru_skip_lr_mult)
    for group in self.optimizers["world_model"].param_groups:
      group["lr"] = self.cfg.wm_learning_rate
    for name in ("actor", "critic"):
      for group in self.optimizers[name].param_groups:
        group["lr"] = self.cfg.learning_rate * float(group.get("lr_mult", 1.0))
    if any(k in changed_cfg for k in ("learning_rate", "action_noise", "policy_target_kl", "gru_skip_lr_mult")):
      self.optimizers["actor"].state.clear()
    if self.cfg.critic_warmup_iterations > 0:
      self.optimizers["critic"].state.clear()

    for field in (
      "iteration",
      "controls",
      "scheduled_controls",
      "scheduled_actor_starts",
      "scheduled_training_transitions",
      "optimizer_updates",
      "stage_index",
      "stage_entered_iteration",
      "stage_qualified_streak",
      "final_polish_active",
      "stage_history",
    ):
      if field in saved:
        setattr(self, field, saved[field])
    if self.cfg.initial_stage > self.stage_index:
      self.stage_history.append(
        {
          "iteration": self.iteration,
          "from_stage": self.stage_index,
          "to_stage": int(self.cfg.initial_stage),
          "reason": "resume_initial_stage_override",
        }
      )
      self.stage_index = int(self.cfg.initial_stage)
      self.stage_entered_iteration = self.iteration
      self.stage_qualified_streak = 0
      self.best_validation_reward = -math.inf
      self.best_validation_score = -math.inf
    self._apply_stage(self.current_stage)
    # Full env reset so `reset_arm_pose(target_scale=...)` and commands take effect across all worlds immediately!
    self.collector.reset_all_envs()

    if saved.get("best_validation_reward") is not None and not changed_cfg:
      self.best_validation_reward = float(saved["best_validation_reward"])
    if saved.get("best_validation_score") is not None and not changed_cfg:
      self.best_validation_score = float(saved["best_validation_score"])
    self.collector.counters.update(saved["collector_counters"])
    cur_noise, _ = self.cfg.active_exploration(self.current_stage, final_polish=self.final_polish_active)
    self.collector.set_action_noise(cur_noise)
    self.controller_contract["training_action_std"] = cur_noise
    self.continuation = {
      "mode": "task_continuation" if changed_cfg else "strict",
      "parent_checkpoint": str(path.resolve()),
      "parent_checkpoint_sha256": _file_sha256(path),
      "parent_iteration": saved["iteration"],
      "parent_stage_index": saved.get("stage_index"),
      "changed_config": changed_cfg,
    }
    torch.set_rng_state(saved["rng"]["torch"].cpu())
    torch.cuda.set_rng_state(saved["rng"]["cuda"].cpu(), self.device)

  def maybe_advance_curriculum(self, metrics: dict[str, float]) -> str | None:
    """Evaluate validation gates; return milestone label if a stage or polish transition occurred."""
    if not self.cfg.auto_curriculum:
      return None
    stage = self.current_stage
    if self.iteration - self.stage_entered_iteration < stage.min_stage_iterations:
      return None
    if stage_qualifies(stage, metrics):
      self.stage_qualified_streak += 1
    else:
      self.stage_qualified_streak = 0
      return None

    if self.stage_qualified_streak < self.cfg.qualify_streak_required:
      return None

    self.stage_qualified_streak = 0
    if self.stage_index < len(AUTO_CURRICULUM_STAGES) - 1:
      old_stage = stage
      milestone = f"milestone_stage_{old_stage.index}_{old_stage.name}_{self.iteration:08d}"
      self.stage_index += 1
      new_stage = self.current_stage
      self.stage_entered_iteration = self.iteration
      self.best_validation_reward = -math.inf
      self.best_validation_score = -math.inf
      self._apply_stage(new_stage)
      record = {
        "iteration": self.iteration,
        "from_stage": old_stage.index,
        "from_name": old_stage.name,
        "to_stage": new_stage.index,
        "to_name": new_stage.name,
        "det_mtbf_s": metrics["native/det_mtbf_s"],
        "det_unbroken_mean_s": metrics["native/det_unbroken_mean_s"],
        "val_stance_xy_err_m": metrics["native/validation_stance_xy_err_m"],
        "val_ss_match": metrics["native/validation_ss_match"],
        "val_failure_rate": metrics["native/validation_failure_rate"],
      }
      self.stage_history.append(record)
      self.progress("curriculum_advanced", **record)
      return milestone

    if not self.final_polish_active:
      milestone = f"milestone_stage_{stage.index}_pre_polish_{self.iteration:08d}"
      self.final_polish_active = True
      self.stage_entered_iteration = self.iteration
      self.best_validation_reward = -math.inf
      self.best_validation_score = -math.inf
      record = {
        "iteration": self.iteration,
        "event": "activated_final_polish",
        "action_noise": POLISH_EXPLORATION[0],
        "policy_target_kl": POLISH_EXPLORATION[1],
      }
      self.stage_history.append(record)
      self.progress("curriculum_polish_activated", **record)
      return milestone
    return None

  def run(self) -> None:
    cfg = self.cfg
    init_noise, _ = cfg.active_exploration(self.current_stage, final_polish=self.final_polish_active)
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
    self.progress("auditing_h36_gradients", compilation_enabled=cfg.compile_model)
    self.audit_imagination()
    if self.rank == 0:
      atomic_json(self.output / "preflight.json", self.preflight)
    self.save()
    start_iteration = self.iteration
    if cfg.preflight:
      self.progress("completed", reason="preflight_only")
      return

    while not cfg.iterations or self.iteration < cfg.iterations:
      if self.stop_requested():
        self.save()
        self.progress("stopped", reason="user_stop")
        return

      tick = time.monotonic()
      cur_noise, cur_target_kl = cfg.active_exploration(self.current_stage, final_polish=self.final_polish_active)
      self.collector.set_action_noise(cur_noise)
      self.controller_contract["training_action_std"] = cur_noise

      needed = self.required_controls()
      req_steps = max(1, needed - self.scheduled_controls)
      phase_started = time.monotonic()
      self.progress("collecting", requested_controls=req_steps)
      native_rollout = FootstepPolicyRollout(self.replay.train_envs)
      self.collect(req_steps, policy_rollout=native_rollout)
      self.scheduled_training_transitions += req_steps * self.replay.train_envs
      self.scheduled_controls += req_steps
      self.scheduled_actor_starts += cfg.actor_batch_size
      collect_seconds = time.monotonic() - phase_started

      for name, optimizer in self.optimizers.items():
        if name == "world_model":
          base = cfg.wm_learning_rate
        elif name == "critic":
          base = max(cfg.learning_rate, 3e-4)
        else:
          base = cfg.learning_rate
        for group in optimizer.param_groups:
          group["lr"] = cosine_learning_rate(self.iteration, base, self.world_size) * float(group.get("lr_mult", 1.0))

      phase_started = time.monotonic()
      self.progress("world_model_start", compilation_enabled=cfg.compile_model)
      metrics = self.fit_model()
      metrics["timing/collect_seconds"] = collect_seconds
      metrics["timing/wm_seconds"] = time.monotonic() - phase_started
      metrics["policy/action_noise"] = cur_noise
      metrics["policy/target_kl"] = cur_target_kl
      metrics["curriculum/stage_index"] = float(self.stage_index)
      metrics["curriculum/final_polish"] = float(self.final_polish_active)

      policy_batch = native_rollout.finish()
      initial_kl = self.native_policy_kl(policy_batch, cur_noise)
      if not torch.isfinite(initial_kl) or initial_kl * (cur_noise**2) > 1e-5:
        raise RuntimeError(f"Native footstep KL batch is not from behavior policy: KL={float(initial_kl)}")

      phase_started = time.monotonic()
      in_critic_warmup = (self.iteration - start_iteration) < cfg.critic_warmup_iterations
      self.progress("actor_start", compilation_enabled=cfg.compile_model, critic_warmup=in_critic_warmup)
      if in_critic_warmup:
        policy_metrics, critic_batch = self.improve_policy(cur_noise, update_actor=False)
        metrics.update(policy_metrics)
        metrics["policy/rejected"] = 0.0
        metrics["policy/kl"] = 0.0
        metrics["timing/actor_seconds"] = time.monotonic() - phase_started
      else:
        before_weights = [p.detach().clone() for p in self.actor.parameters()]
        adam_before = copy.deepcopy(self.optimizers["actor"].state_dict())
        actor_clock = self.optimizer_updates["actor"]
        policy_metrics, critic_batch = self.improve_policy(cur_noise, update_actor=True)
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
        del before_weights, adam_before

      self.progress("critic")
      phase_started = time.monotonic()
      metrics.update(self.fit_critic(critic_batch))
      metrics["timing/critic_seconds"] = time.monotonic() - phase_started
      del critic_batch, policy_batch, native_rollout

      self.iteration += 1
      is_val_iter = self.iteration == 1 or self.iteration % cfg.validation_interval == 0
      if is_val_iter:
        self.progress("validation")
        phase_started = time.monotonic()
        metrics.update(self.validate_model())
        metrics["timing/validation_seconds"] = time.monotonic() - phase_started

        sums = self.native_sums.clone()
        if dist.is_initialized():
          dist.all_reduce(sums)
        for offset, label in ((0, "train"), (10, "validation")):
          count = max(float(sums[offset + 6]), 1.0)
          ss_cnt = max(float(sums[offset + 9]), 1.0)
          metrics[f"native/{label}_reward"] = float(sums[offset + 0]) / count
          metrics[f"native/{label}_stance_xy_err_m"] = float(sums[offset + 1]) / count
          metrics[f"native/{label}_stance_yaw_err_rad"] = float(sums[offset + 2]) / count
          metrics[f"native/{label}_theta_err_rad"] = float(sums[offset + 3]) / count
          metrics[f"native/{label}_failure_rate"] = float(sums[offset + 4]) / count
          metrics[f"native/{label}_termination_rate"] = float(sums[offset + 5]) / count
          metrics[f"native/{label}_ss_match"] = float(sums[offset + 7]) / ss_cnt
          metrics[f"native/{label}_swing_lift_m"] = float(sums[offset + 8]) / ss_cnt
        det_cnt = max(float(sums[23]), 1.0)
        det_healthy_s = float(sums[20]) * 0.02
        det_first_fails = float(sums[21])
        metrics["native/det_mtbf_s"] = det_healthy_s / max(det_first_fails, 0.5)
        metrics["native/det_unbroken_mean_s"] = float(sums[22]) / det_cnt
        metrics["native/det_first_fails"] = det_first_fails
        step_ep_cnt = float(sums[25])
        metrics["native/step_ep_success_rate"] = float(sums[24]) / max(step_ep_cnt, 1.0)
        metrics["native/step_ep_episodes"] = step_ep_cnt
        self.native_sums.zero_()

      milestone: str | None = None
      is_best = False
      if is_val_iter:
        val_rew = metrics["native/validation_reward"]
        val_score = (
          -metrics["native/validation_stance_xy_err_m"]
          - 0.15 * metrics["native/validation_stance_yaw_err_rad"]
          + 0.04 * metrics["native/step_ep_success_rate"]
          - 10.0 * metrics["native/validation_failure_rate"]
        )
        metrics["native/validation_score"] = val_score
        if (not in_critic_warmup) and (self.controls - post_warmup_controls >= 800) and val_score > self.best_validation_score:
          self.best_validation_score = val_score
          self.best_validation_reward = val_rew
          is_best = True
        milestone = self.maybe_advance_curriculum(metrics)
        metrics["curriculum/qualified_streak"] = float(self.stage_qualified_streak)

      metrics.update(
        iteration=self.iteration,
        seconds=time.monotonic() - tick,
        peak_allocated_gib=torch.cuda.max_memory_allocated(self.device) / 1024**3,
        **{f"optimizer_updates/{key}": value for key, value in self.optimizer_updates.items()},
        **self.collector.counters,
      )
      if not all(math.isfinite(v) for v in metrics.values()):
        raise FloatingPointError("Nonfinite footstep WM training metric")

      checkpoint_started = time.monotonic()
      should_save = int(
        is_best
        or milestone is not None
        or self.iteration == 1
        or self.iteration % cfg.save_interval == 0
        or time.monotonic() - self.last_save >= cfg.save_seconds
      )
      if dist.is_initialized():
        save_flag = torch.tensor(should_save, device=self.device, dtype=torch.int64)
        dist.all_reduce(save_flag, op=dist.ReduceOp.MAX)
        should_save = int(save_flag.item())
      if should_save:
        self.save(is_best=is_best, milestone=milestone)
      metrics["timing/checkpoint_seconds"] = time.monotonic() - checkpoint_started
      metrics["wall_seconds"] = time.monotonic() - tick

      with (self.output / f"metrics_rank_{self.rank}.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(metrics, allow_nan=False) + "\n")
      if self.iteration == 1 or self.iteration % 5 == 0:
        self.progress("training", wall_seconds=metrics["wall_seconds"])
      else:
        print(
          json.dumps(
            {
              "rank": self.rank,
              "iteration": self.iteration,
              "stage": self.stage_index,
              "wall_seconds": metrics["wall_seconds"],
            }
          ),
          flush=True,
        )

    self.save()
    self.progress("completed", reason="bounded_iterations")


def run_footstep_wm_training(cfg: FootstepWmConfig) -> None:
  if int(os.environ.get("WORLD_SIZE", "1")) > 1 and os.environ.get("WM_GPU_PINNED") != "1":
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
  os.environ.setdefault("MUJOCO_GL", "egl")

  cfg.validate()
  if not torch.cuda.is_available():
    raise RuntimeError("Footstep WM training requires CUDA")

  rank = int(os.environ.get("RANK", "0"))
  world = int(os.environ.get("WORLD_SIZE", "1"))
  local = int(os.environ.get("LOCAL_RANK", "0"))
  device = f"cuda:{local}" if world > 1 else cfg.device
  torch.cuda.set_device(device)
  if world > 1 and not dist.is_initialized():
    dist.init_process_group("nccl", timeout=timedelta(minutes=30), device_id=torch.device(device))

  trainer: FootstepWmTrainer | None = None
  try:
    trainer = FootstepWmTrainer(cfg, rank, world, device)
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
