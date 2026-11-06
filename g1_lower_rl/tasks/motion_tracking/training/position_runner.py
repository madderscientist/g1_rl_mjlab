"""Full-corpus SuperMimic: native collection, position-WM fitting, MLP BPTT and GAE."""

from __future__ import annotations

import copy
import json
import math
import os
import signal
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.checkpoint import checkpoint

from g1_lower_rl.rl.supermimic import bptt_return, cosine_learning_rate, frozen, gae_targets, optimizer_step
from g1_lower_rl.tasks.motion_tracking.corpus import file_sha256
from g1_lower_rl.world_model.adapters.kinematics import TorchKinematics
from g1_lower_rl.world_model.checkpoint import atomic_checkpoint, atomic_json, load_world_model, read_checkpoint
from g1_lower_rl.world_model.collection import CollectorConfig
from g1_lower_rl.world_model.training.derivatives import derivative_metrics, derivative_sums, native_secants, snapshot
from g1_lower_rl.world_model.training.objectives import LossConfig, autoregressive_loss, finalize_metrics, physical_metric_sums
from g1_lower_rl.world_model.training.provenance import snapshot_sources

from .graphs import CapturedRolloutStep
from .policy_kl import (
  NativePolicyBatch,
  NativePolicyRollout,
  constrain_policy_kl,
  policy_kl,
)
from .position_collection import PositionPolicyCollector
from .position_policy import (
  HISTORY,
  OBS_DIM,
  actor_architecture_metadata,
  actor_observations,
  make_actor,
  proprioception,
  residual_target,
)
from .position_task import (
  RECOVERY_XY_TASK,
  PolicyStart,
  PositionReplay,
  PositionTask,
  TaskDefinition,
  position_task_contract,
)

FORMAT = "full_mocap_position_v3_continuous_rgmt_v1"


@dataclass(frozen=True)
class PositionTrainingConfig:
  world_model: str
  replica_source: str
  output: str
  mocap_dir: str = "motions/mocap_supermimic_v1"
  resume: str | None = None
  device: str = "cuda:0"
  seed: int = 20261102
  iterations: int = 0
  num_envs: int = 1024
  validation_envs: int = 128
  replay_steps: int = 512
  warmup_steps: int = 64
  actor_batch_size: int = 512
  actor_microbatch: int = 64
  horizon: int = 32
  wm_batch_size: int = 2048
  wm_microbatch: int = 256
  wm_horizon: int = 8
  wm_batches: int = 10
  critic_epochs: int = 5
  critic_minibatches: int = 4
  critic_microbatch: int = 4096
  learning_rate: float = 2e-4
  wm_learning_rate: float = 2e-4
  gamma: float = 0.99
  gae_lambda: float = 0.95
  utd: float = 0.4
  save_interval: int = 10
  save_seconds: float = 300
  validation_interval: int = 10
  validation_batch_size: int = 32
  action_noise: float = 0.1
  checkpoint_imagination: bool = True
  compile_model: bool = False
  task_definition: TaskDefinition = RECOVERY_XY_TASK
  actor_architecture: Literal["mlp"] = "mlp"
  native_actor_kl: bool = True
  policy_microbatch: int = 4096
  policy_target_kl: float = 0.01
  fast_kinematics: bool = False
  compile_reward: bool = False
  cuda_graphs: bool = False
  matmul_precision: Literal["high", "highest"] = "high"
  wm_matmul_precision: Literal["high", "highest"] = "high"

  @property
  def sample_policy_actions(self) -> bool:
    return True

  def validate(self) -> None:
    position_task_contract(self.task_definition, self.gamma)
    if self.actor_architecture != "mlp" or not self.native_actor_kl:
      raise ValueError("Mocap training requires the MLP actor and native-policy KL")
    counts = (
      self.num_envs, self.validation_envs, self.replay_steps, self.warmup_steps,
      self.actor_batch_size, self.actor_microbatch, self.horizon, self.wm_batch_size, self.wm_microbatch,
      self.wm_horizon, self.wm_batches, self.critic_epochs, self.critic_minibatches,
      self.critic_microbatch, self.policy_microbatch, self.save_interval, self.validation_interval,
      self.validation_batch_size,
    )
    if min(counts) < 1 or self.iterations < 0:
      raise ValueError("Training counts must be positive; iterations=0 selects unlimited training")
    if self.validation_envs >= self.num_envs or self.replay_steps < max(32, HISTORY, self.wm_horizon):
      raise ValueError("Need disjoint train/validation worlds and sufficient replay history")
    if self.warmup_steps < max(32, HISTORY, self.wm_horizon) or self.horizon != 32:
      raise ValueError("Mocap requires H32 imagination and at least 32 warmup controls")
    positive = (
      self.learning_rate, self.wm_learning_rate, self.utd, self.action_noise,
      self.save_seconds, self.policy_target_kl,
    )
    if any(not math.isfinite(value) or value <= 0 for value in positive) or not 0 <= self.gae_lambda <= 1:
      raise ValueError("Invalid training rate, exploration, KL or GAE configuration")
    if self.actor_batch_size / self.utd < self.num_envs - self.validation_envs:
      raise ValueError("Actor batch/UTD must require at least one fresh native control per policy update")
    if self.cuda_graphs and (not self.checkpoint_imagination or self.matmul_precision != "highest"):
      raise ValueError("CUDA rollout graphs require checkpointed imagination and highest matmul precision")
    if self.matmul_precision not in ("high", "highest") or self.wm_matmul_precision not in ("high", "highest"):
      raise ValueError("Unsupported matmul precision")

def _broadcast(module: nn.Module) -> None:
  if dist.is_initialized():
    for value in (*module.parameters(), *module.buffers()):
      dist.broadcast(value.detach(), 0)


class PositionTrainer:
  def __init__(self, cfg: PositionTrainingConfig, rank: int, world_size: int, device: str):
    self.cfg, self.rank, self.world_size, self.device = cfg, rank, world_size, torch.device(device)
    torch.set_float32_matmul_precision(cfg.matmul_precision)
    self.output = Path(cfg.output).resolve()
    self.output.mkdir(parents=True, exist_ok=True)
    if not cfg.resume and (self.output / "config.json").exists():
      raise FileExistsError("Fresh training must use a new output directory")
    if dist.is_initialized():
      dist.barrier()
    self.iteration, self.controls, self.scheduled_controls, self.scheduled_actor_starts = 0, 0, 0, 0
    self.scheduled_training_transitions = 0
    self.continuation = None
    self.optimizer_updates = {"world_model": 0, "actor": 0, "critic": 0}
    self.started, self.last_save = time.monotonic(), time.monotonic()
    self.stop_signal = False
    for signum in (signal.SIGTERM, signal.SIGINT):
      signal.signal(signum, self._signal)
    self.sources = snapshot_sources(self.output, write=rank == 0)
    torch.manual_seed(cfg.seed)
    self.actor = make_actor(self.device)
    self.critic = nn.Sequential(nn.Linear(918, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(), nn.Linear(256, 1)).to(self.device)
    self.model = load_world_model(cfg.world_model, self.device).train()
    _broadcast(self.actor)
    _broadcast(self.critic)
    _broadcast(self.model)
    parent = read_checkpoint(cfg.world_model)
    self.origin = {
      "world_model_checkpoint": str(Path(cfg.world_model).resolve()), "world_model_sha256": file_sha256(Path(cfg.world_model)),
      "world_model_update": parent.get("update"), "world_model_best_score": parent.get("best_score"),
      "policy_initialization": "scratch", "critic_initialization": "scratch", "optimizers": "fresh_Adam",
      "architecture": actor_architecture_metadata(),
      "actor_parameters": sum(p.numel() for p in self.actor.parameters()),
      "unlimited": cfg.iterations == 0,
    }
    del parent
    torch.manual_seed((cfg.seed + rank * 9973) % 2**32)
    self.progress("initializing_native_full_corpus")
    self.collector = PositionPolicyCollector(
      CollectorConfig("", num_envs=cfg.num_envs, device=device, seed=(cfg.seed + rank * 9973) % 2**32,
                      motion_dir=cfg.mocap_dir, action_noise=cfg.action_noise, njmax=500),
      replica_source=cfg.replica_source, validation_envs=cfg.validation_envs, actor=self.actor,
      task_definition=cfg.task_definition, gamma=cfg.gamma,
      policy_action_sampling=cfg.sample_policy_actions,
    )
    robot = self.collector.env.scene["robot"].data
    self.default_position = robot.default_joint_pos[0].clone()
    self.task = PositionTask(
      TorchKinematics(self.collector.template).to(device=self.device, dtype=torch.float32),
      self.default_position, robot.soft_joint_pos_limits[0], cfg.task_definition, cfg.gamma,
    ).to(self.device)
    self.task.adapter.kinematics.level_batching = cfg.fast_kinematics
    self.scale = self.collector.action_scale.clone()
    native = self.collector.env.sim.mj_model
    self.target_limits = torch.tensor(
      native.actuator_ctrlrange[self.collector.indices["actuator"]], device=self.device, dtype=self.scale.dtype,
    )
    self.target_limited = torch.tensor(native.actuator_ctrllimited[self.collector.indices["actuator"]], device=self.device).bool()
    self.replay = PositionReplay(cfg.replay_steps, cfg.num_envs, cfg.validation_envs, self.device)
    self.optimizers = {
      "world_model": torch.optim.Adam(self.model.parameters(), lr=cfg.wm_learning_rate),
      "actor": torch.optim.Adam(self.actor.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.95)),
      "critic": torch.optim.Adam(self.critic.parameters(), lr=cfg.learning_rate),
    }
    self.loss_cfg = LossConfig(gamma=cfg.gamma)
    self.forward_model = torch.compile(self._dynamics, dynamic=True) if cfg.compile_model else self._dynamics
    self.reward_fn = (
      torch.compile(self.task.reward, dynamic=True, options={"emulate_precision_casts": True})
      if cfg.compile_reward else self.task.reward
    )
    self._actor_graphs: dict[int, CapturedRolloutStep] = {}
    self.preflight: dict[str, float] = {}
    self.native_sums = torch.zeros(8, device=self.device)
    self._native_reward_checked = False
    if cfg.resume:
      self.restore(Path(cfg.resume))
      if self.continuation is None:
        raise RuntimeError("Resume did not establish its checkpoint lineage")
      self.origin.update(
        policy_initialization="restored_from_parent", world_model_initialization="restored_online_parent",
        critic_initialization="restored_from_parent", optimizers="restored_Adam",
      )
    if not rank:
      atomic_json(self.output / "config.json", {
        "format": FORMAT, "config": asdict(cfg), "origin": self.origin, "continuation": self.continuation,
        "collector": self.collector.metadata, "source_provenance": self.sources,
        "task_contract": self.collector.metadata["task_contract"],
        "termination": "native_shared_Bernoulli_on_failure; imagination_probability_times_sigmoid20_margin",
        "resume_semantics": "model/optimizers/RNG/sampler restored; simulator/replay reset, not bitwise trajectories",
      })

  def _signal(self, signum: int, frame) -> None:
    self.stop_signal = True

  def _dynamics(self, state: torch.Tensor, target: torch.Tensor, context: torch.Tensor):
    return self.model(state, target, context, validate=False)

  def progress(self, phase: str, **details) -> None:
    atomic_json(self.output / ("status.json" if self.rank == 0 else f"status_rank_{self.rank}.json"), {
      "phase": phase, "iteration": self.iteration, "optimizer_updates": self.optimizer_updates,
      "controls_per_rank": self.controls, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
      "pid": os.getpid(), "rank": self.rank, "unlimited": self.cfg.iterations == 0,
      "collector_counters": self.collector.counters if hasattr(self, "collector") else {},
      **details,
    })
    print(json.dumps({"rank": self.rank, "phase": phase, "iteration": self.iteration, **details}), flush=True)

  @torch.no_grad()
  def collect(self, count: int, *, audit: bool = False, policy_rollout: NativePolicyRollout | None = None) -> None:
    for step in range(count):
      before = {} if audit and step == count - 1 else None
      original = self.collector._original_step

      def capture(before=before, original=original):
        if before is not None and not before:
          before.update(snapshot(self.collector))
        original()

      if before is not None:
        self.collector._original_step = capture
      try:
        batch, record = self.collector.collect_step()
      finally:
        self.collector._original_step = original
      self.replay.append_control(batch, record)
      if self.cfg.sample_policy_actions:
        torch.testing.assert_close(record["executed_action"], record["action"], rtol=0, atol=0)
      reference = self.collector.reference_state(record["frame"])
      if policy_rollout is not None:
        policy_rollout.append(history=record["history"], reference=record["reference"], mean=record["mean_action"])
      executed = batch.inputs[0, :, 68:97]
      expected = residual_target(
        record["executed_action"],
        reference[:, 7:36], self.scale, record["bias"], self.target_limits, self.target_limited,
      )
      torch.testing.assert_close(executed, expected, rtol=2e-5, atol=2e-5)
      if not self._native_reward_checked:
        analytic = self.task.reward(batch.next_state[0], reference, record["action"], record["latest"][:, -29:])
        gap = (analytic - record["reward"]).abs().max()
        self.preflight["native_reward_max_absolute_error"] = float(gap)
        if gap > 2e-4:
          raise RuntimeError(f"Analytic/native reward mismatch before optimizer updates: {float(gap)}")
        self._native_reward_checked = True
      train = slice(0, self.replay.train_envs)
      val = slice(self.replay.train_envs, None)
      error = (batch.next_state[0, :, 7:36] - reference[:, 7:36]).square().mean(-1)
      for offset, selected in ((0, train), (4, val)):
        self.native_sums[offset] += record["reward"][selected].sum()
        self.native_sums[offset + 1] += error[selected].sum()
        self.native_sums[offset + 2] += record["terminated"][selected].sum()
        self.native_sums[offset + 3] += len(error[selected])
      self.controls += 1
      if before is not None:
        valid = batch.valid[0].clone()
        valid &= batch.segment[0] == self.collector._segment
        valid[:self.replay.train_envs] = False
        probe = native_secants(
          self.collector, before, batch.inputs[0], valid, batch.next_state[0], seed=self.cfg.seed + self.rank,
          repeatability_exclusion_limit=0.1,
        )
        metrics = derivative_metrics(derivative_sums(self.model, probe))
        if min(value for key, value in metrics.items() if key.endswith("/cosine")) <= 0:
          raise RuntimeError(f"Native action-response direction is not positively aligned: {metrics}")
        self.preflight.update(metrics)
        self.preflight["native_derivative_states"] = probe["valid_states"]
        self.preflight["native_derivative_excluded_nonrepeatable"] = probe["excluded_nonrepeatable_states"]
        self.preflight["native_derivative_repeat_max_abs"] = probe["native_repeat_max_abs"]
        atomic_checkpoint(self.output / f"native_derivative_rank_{self.rank}.pt", probe)


  def policy_mean(self, history: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return self.actor(actor_observations(history, reference), stochastic_output=False)

  def native_policy_kl(self, batch: NativePolicyBatch) -> torch.Tensor:
    return policy_kl(self.policy_mean, batch, self.cfg.action_noise, self.cfg.policy_microbatch)


  def fit_model(self) -> dict[str, float]:
    previous = torch.get_float32_matmul_precision()
    try:
      torch.set_float32_matmul_precision(self.cfg.wm_matmul_precision)
      return self._fit_model()
    finally:
      torch.set_float32_matmul_precision(previous)

  def _fit_model(self) -> dict[str, float]:
    cfg, total, state_loss = self.cfg, 0.0, 0.0
    for update in range(cfg.wm_batches):
      optimizer = self.optimizers["world_model"]
      optimizer.zero_grad(set_to_none=True)
      for begin in range(0, cfg.wm_batch_size, cfg.wm_microbatch):
        size = min(cfg.wm_microbatch, cfg.wm_batch_size - begin)
        window = self.replay.sample(size, cfg.wm_horizon)
        loss, metrics = autoregressive_loss(
          self.forward_model, window, self.model.state_scale, self.model.force_scale, self.loss_cfg,
        )
        (loss * (size / cfg.wm_batch_size)).backward()
        total += float(loss.detach()) * size / cfg.wm_batch_size
        state_loss += float(metrics["state_loss"]) * size / cfg.wm_batch_size
      norm = optimizer_step(self.model.parameters(), optimizer)
      self.optimizer_updates["world_model"] += 1
      self.progress("world_model", wm_batch=update + 1)
    self.optimizers["world_model"].zero_grad(set_to_none=True)
    return {"wm_loss": total / cfg.wm_batches, "wm_state_loss": state_loss / cfg.wm_batches, "wm_gradient_norm": norm}

  def _step(
    self, state: torch.Tensor, history: torch.Tensor, context: torch.Tensor, bias: torch.Tensor,
    reference: torch.Tensor, window: torch.Tensor, observation_noise: torch.Tensor, action_noise: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    action = self.actor(actor_observations(history, window), stochastic_output=False)
    sampled_action = action + action_noise
    target = residual_target(
      sampled_action, reference[:, 7:36], self.scale, bias, self.target_limits, self.target_limited,
    )
    following = self.forward_model(state, target, context).state
    recorded_action = sampled_action if self.cfg.sample_policy_actions else action
    reward = self.reward_fn(following, reference, recorded_action, history[:, -1, -29:])
    observation = proprioception(following, recorded_action, bias, self.default_position) + observation_noise
    history = torch.cat((history[:, 1:], observation[:, None]), dim=1)
    termination = self.task.termination(following, reference)
    return following, history, reward, termination

  def _captured_step(self, args: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    names = ("state", "history", "context", "bias", "reference", "window", "observation_noise", "action_noise")
    values = dict(zip(names, args, strict=True))
    size = len(args[0])
    if size not in self._actor_graphs:
      self.progress("capturing_actor_tick", batch_size=size)
      self._actor_graphs[size] = CapturedRolloutStep(
        lambda inputs: self._step(*inputs.values()), values, self.actor.parameters(),
        pool=torch.cuda.graph_pool_handle(),
      )
    return self._actor_graphs[size](values)

  def imagine(self, start: PolicyStart, *, checkpointed: bool) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], dict[str, float]]:
    cfg, c = self.cfg, self.collector
    n = len(start.frame)
    with torch.no_grad():
      last = c.library.last_frame_of(start.frame)
      reference, windows = c.reference_trajectory(start.frame, cfg.horizon)
    state, history, context = start.state, start.history, start.context
    noise_half = state.new_tensor([0.01] * 29 + [0.5] * 29 + [0.2] * 3 + [0.05] * 3 + [0.] * 29)
    features, values, rewards, terminations, truncations = [], [], [], [], []
    feature = self.task.critic_features(self.model, state, context, reference[:, 0], windows[:, 0])
    features.append(feature.detach())
    values.append(self.critic(feature).squeeze(-1))
    for step in range(cfg.horizon):
      obs_noise = (torch.rand(n, OBS_DIM, device=self.device) * 2 - 1) * noise_half
      action_noise = torch.randn(n, 29, device=self.device) * cfg.action_noise
      args = (state, history, context, start.bias, reference[:, step], windows[:, step], obs_noise, action_noise)
      if checkpointed and cfg.cuda_graphs:
        state, history, reward, terminated = self._captured_step(args)
      elif checkpointed:
        state, history, reward, terminated = checkpoint(self._step, *args, use_reentrant=False)
      else:
        state, history, reward, terminated = self._step(*args)
      feature = self.task.critic_features(self.model, state, context, reference[:, step + 1], windows[:, step + 1])
      features.append(feature.detach())
      values.append(self.critic(feature).squeeze(-1))
      rewards.append(reward)
      terminations.append(terminated)
      clip_end = start.clip_end | (start.frame + step > last)
      truncations.append((clip_end | (step + 1 >= start.remaining)).to(state.dtype))
    r, d, trunc, v = (torch.stack(items, dim=1) for items in (rewards, terminations, truncations, values))
    returns, survival = bptt_return(r, d, v[:, -1], cfg.gamma, truncation=trunc, bootstrap_values=v[:, 1:])
    targets = gae_targets(r.detach(), d.detach(), v.detach(), cfg.gamma, cfg.gae_lambda, truncation=trunc)
    critic_batch = (
      torch.stack(features[:-1], dim=1).flatten(0, 1),
      targets.flatten().detach(), survival.flatten().detach(),
    )
    return -returns.mean() / cfg.horizon, critic_batch, {
      "imagined_reward": float(r.detach().mean()), "imagined_survival": float(survival[:, -1].detach().mean()),
      "imagined_return": float(returns.detach().mean()),
    }

  def improve_policy(self) -> tuple[dict[str, float], tuple[torch.Tensor, ...]]:
    cfg = self.cfg
    optimizer = self.optimizers["actor"]
    optimizer.zero_grad(set_to_none=True)
    pieces, metrics = [], {}
    with frozen(self.model, self.critic):
      for begin in range(0, cfg.actor_batch_size, cfg.actor_microbatch):
        size = min(cfg.actor_microbatch, cfg.actor_batch_size - begin)
        start = self.replay.sample_starts(size)
        loss, critic_batch, partial = self.imagine(start, checkpointed=cfg.checkpoint_imagination)
        (loss * size / cfg.actor_batch_size).backward()
        pieces.append(critic_batch)
        for key, value in partial.items():
          metrics[key] = metrics.get(key, 0.0) + value * size / cfg.actor_batch_size
        self.progress("actor", actor_samples=begin + size)
      metrics["actor_gradient_norm"] = optimizer_step(self.actor.parameters(), optimizer, actor=True)
    if any(p.grad is not None for p in self.model.parameters()):
      raise RuntimeError("Policy BPTT unexpectedly wrote WM parameter gradients")
    self.optimizer_updates["actor"] += 1
    optimizer.zero_grad(set_to_none=True)
    return metrics, tuple(torch.cat([part[i] for part in pieces]) for i in range(3))

  def audit_imagination(self) -> None:
    start = self.replay.sample_starts(2)
    gradients = []
    reward_fn = self.reward_fn
    with frozen(self.model, self.critic), torch.random.fork_rng(devices=[self.device]):
      try:
        for checkpointed in (False, True):
          self.reward_fn = reward_fn if checkpointed else self.task.reward
          torch.manual_seed(self.cfg.seed + self.rank)
          loss, _, _ = self.imagine(start, checkpointed=checkpointed)
          gradient = torch.cat([
            value.flatten() for value in torch.autograd.grad(loss, tuple(self.actor.parameters()))
          ])
          if not torch.isfinite(gradient).all() or gradient.norm() <= 0:
            raise FloatingPointError("H32 policy gradient is nonfinite or zero")
          gradients.append(gradient)
          # Do not keep eager AccumulateGrad nodes alive during side-stream capture.
          del loss
      finally:
        self.reward_fn = reward_fn
    relative_error = (gradients[0] - gradients[1]).norm() / gradients[0].norm()
    if relative_error > 2e-4:
      raise RuntimeError(f"Checkpointed/eager policy gradients differ: {float(relative_error)}")
    self.preflight["checkpoint_eager_gradient_relative_error"] = float(relative_error)
    self.preflight["eager_actor_gradient_norm"] = float(gradients[0].norm())

  def fit_critic(self, batch: tuple[torch.Tensor, ...]) -> dict[str, float]:
    features, targets, weights = batch
    if any(t.requires_grad for t in batch):
      raise RuntimeError("Critic GAE targets and features must be detached")
    optimizer, loss_sum = self.optimizers["critic"], 0.0
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
    window = self.replay.sample(self.cfg.validation_batch_size, 32, validation=True)
    sums = physical_metric_sums(
      self.forward_model, window, evaluation_steps={"20ms": 1, "160ms": 8, "640ms": 32},
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

  def save(self) -> None:
    directory = self.output / "checkpoints" / f"update_{self.iteration:08d}"
    directory.mkdir(parents=True, exist_ok=True)
    command = self.collector._native_command
    payload = {
      "format": FORMAT, "iteration": self.iteration, "rank": self.rank, "world_size": self.world_size,
      "config": asdict(self.cfg), "origin": self.origin, "source_provenance": self.sources,
      "task_contract": self.collector.metadata["task_contract"],
      "collector": self.collector.metadata, "actor_state_dict": self.actor.state_dict(),
      "critic_state_dict": self.critic.state_dict(), "world_model_state_dict": self.model.state_dict(),
      "world_model_config": asdict(self.model.cfg),
      "optimizers": {k: v.state_dict() for k, v in self.optimizers.items()},
      "optimizer_updates": self.optimizer_updates, "controls": self.controls,
      "scheduled_controls": self.scheduled_controls, "collector_counters": self.collector.counters,
      "scheduled_actor_starts": self.scheduled_actor_starts, "continuation": self.continuation,
      "scheduled_training_transitions": self.scheduled_training_transitions,
      "visited_clips": self.collector.library.visited,
      "sampler": {"bin_return": command.bin_return, "bin_memory": command.bin_memory},
      "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(self.device)},
    }
    atomic_checkpoint(directory / f"rank_{self.rank}.pt", payload)
    if dist.is_initialized():
      dist.barrier()
    if not self.rank:
      atomic_json(self.output / "latest.json", {
        "iteration": self.iteration, "world_size": self.world_size, "directory": str(directory.relative_to(self.output)),
      })
      generations = sorted((self.output / "checkpoints").glob("update_*"))
      for previous in generations[:-2]:
        for rank in range(self.world_size):
          (previous / f"rank_{rank}.pt").unlink()
        previous.rmdir()
    self.last_save = time.monotonic()

  def restore(self, path: Path) -> None:
    if path.is_dir():
      path /= f"rank_{self.rank}.pt"
    saved = torch.load(path, map_location=self.device, weights_only=True)
    if saved.get("format") != FORMAT or saved["rank"] != self.rank or saved["world_size"] != self.world_size:
      raise ValueError("Position training checkpoint format/rank/world mismatch")
    expected_config = {key: value for key, value in asdict(self.cfg).items() if key not in ("resume", "output", "iterations")}
    previous_config = {key: value for key, value in saved["config"].items() if key not in ("resume", "output", "iterations")}
    if expected_config != previous_config:
      raise ValueError("Strict resume configuration changed")
    if saved["source_provenance"]["files"] != self.sources["files"] or saved["collector"] != self.collector.metadata:
      raise ValueError("Resume sources/corpus/physics provenance changed")
    if saved["task_contract"] != position_task_contract(self.cfg.task_definition, self.cfg.gamma):
      raise ValueError("Resume task contract changed")
    self.actor.load_state_dict(saved["actor_state_dict"], strict=True)
    self.critic.load_state_dict(saved["critic_state_dict"], strict=True)
    self.model.load_state_dict(saved["world_model_state_dict"], strict=True)
    for name, optimizer in self.optimizers.items():
      optimizer.load_state_dict(saved["optimizers"][name])
    for name in (
      "iteration", "controls", "scheduled_controls", "scheduled_actor_starts",
      "scheduled_training_transitions", "optimizer_updates",
    ):
      setattr(self, name, saved[name])
    self.collector.counters.update(saved["collector_counters"])
    self.collector.library.visited.copy_(saved["visited_clips"])
    for name in self.collector._initialization:
      self.collector._initialization[name] = self.collector.counters[f"initial_{name}"]
    for name, value in saved["sampler"].items():
      getattr(self.collector._native_command, name).copy_(value)
    self.continuation = {
      "mode": "strict", "parent_checkpoint": str(path.resolve()), "parent_checkpoint_sha256": file_sha256(path),
      "parent_iteration": saved["iteration"], "parent_optimizer_updates": saved["optimizer_updates"],
      "changed_config": {}, "changed_sources": {}, "changed_collector": {},
    }
    torch.set_rng_state(saved["rng"]["torch"].cpu())
    torch.cuda.set_rng_state(saved["rng"]["cuda"].cpu(), self.device)

  def run(self) -> None:
    cfg = self.cfg
    self.progress("warmup")
    self.collect(cfg.warmup_steps, audit=True)
    if not self.replay.candidates(32, validation=True).numel():
      raise RuntimeError("No reset-safe640ms held-out WM validation trajectories after warmup")
    self.progress("validating_dynamics", compilation_enabled=cfg.compile_model)
    self.preflight.update(self.validate_model())
    self.progress("auditing_h32_gradients", compilation_enabled=cfg.compile_model)
    self.audit_imagination()
    if not self.rank:
      atomic_json(self.output / "preflight.json", self.preflight)
    self.save()
    while not cfg.iterations or self.iteration < cfg.iterations:
      if self.stop_requested():
        self.save()
        self.progress("stopped", reason="user_stop")
        return
      tick = time.monotonic()
      needed = self.required_controls()
      phase_started = time.monotonic()
      self.progress("collecting", requested_controls=max(0, needed - self.scheduled_controls))
      native_rollout = NativePolicyRollout(self.replay.train_envs)
      self.collect(max(0, needed - self.scheduled_controls), policy_rollout=native_rollout)
      self.scheduled_training_transitions += (needed - self.scheduled_controls) * self.replay.train_envs
      self.scheduled_controls = needed
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
      policy_batch = native_rollout.finish_policy()
      initial_kl = self.native_policy_kl(policy_batch)
      if not torch.isfinite(initial_kl) or initial_kl > 1e-6:
        raise RuntimeError(f"Native KL batch is not from the current behavior policy: KL={float(initial_kl)}")
      phase_started = time.monotonic()
      self.progress("actor_start", compilation_enabled=cfg.compile_model)
      before = [parameter.detach().clone() for parameter in self.actor.parameters()]
      adam_before = copy.deepcopy(self.optimizers["actor"].state_dict())
      actor_clock = self.optimizer_updates["actor"]
      policy_metrics, critic_batch = self.improve_policy()
      metrics.update(policy_metrics)
      trust = constrain_policy_kl(
        self.actor, self.optimizers["actor"], before, adam_before,
        lambda batch=policy_batch: self.native_policy_kl(batch), cfg.policy_target_kl,
      )
      if trust["rejected"]:
        self.optimizer_updates["actor"] = actor_clock
      metrics.update({f"policy/{key}": value for key, value in trust.items()})
      metrics["timing/actor_seconds"] = time.monotonic() - phase_started
      self.progress("critic")
      phase_started = time.monotonic()
      metrics.update(self.fit_critic(critic_batch))
      metrics["timing/critic_seconds"] = time.monotonic() - phase_started
      del critic_batch
      del policy_batch, native_rollout, before, adam_before
      self.iteration += 1
      if self.iteration == 1 or self.iteration % cfg.validation_interval == 0:
        self.progress("validation")
        phase_started = time.monotonic()
        metrics.update(self.validate_model())
        metrics["timing/validation_seconds"] = time.monotonic() - phase_started
      sums = self.native_sums.clone()
      if dist.is_initialized():
        dist.all_reduce(sums)
      for offset, label in ((0, "train"), (4, "validation")):
        count = max(float(sums[offset + 3]), 1)
        metrics[f"native/{label}_reward"] = float(sums[offset]) / count
        metrics[f"native/{label}_joint_rmse"] = math.sqrt(float(sums[offset + 1]) / count)
        metrics[f"native/{label}_termination_rate"] = float(sums[offset + 2]) / count
      self.native_sums.zero_()
      metrics.update(
        iteration=self.iteration, seconds=time.monotonic() - tick,
        peak_allocated_gib=torch.cuda.max_memory_allocated(self.device) / 1024**3,
        **{f"optimizer_updates/{key}": value for key, value in self.optimizer_updates.items()},
        **self.collector.counters,
      )
      if not all(math.isfinite(v) for v in metrics.values()):
        raise FloatingPointError("Nonfinite training metric")
      checkpoint_started = time.monotonic()
      if (self.iteration == 1 or self.iteration % cfg.save_interval == 0
          or time.monotonic() - self.last_save >= cfg.save_seconds):
        self.save()
      metrics["timing/checkpoint_seconds"] = time.monotonic() - checkpoint_started
      metrics["wall_seconds"] = time.monotonic() - tick
      with (self.output / f"metrics_rank_{self.rank}.jsonl").open("a") as stream:
        stream.write(json.dumps(metrics, allow_nan=False) + "\n")
        stream.flush()
      print(json.dumps({"rank": self.rank, **metrics}), flush=True)
      self.progress("training", metrics=metrics)
    self.save()
    self.progress("completed", reason="bounded_preflight")


def run_position_training(cfg: PositionTrainingConfig) -> None:
  cfg.validate()
  if not torch.cuda.is_available():
    raise RuntimeError("Native full-corpus SuperMimic training requires CUDA")
  rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
  local = int(os.environ.get("LOCAL_RANK", "0"))
  device = f"cuda:{local}" if world > 1 else cfg.device
  torch.cuda.set_device(device)
  if world > 1:
    dist.init_process_group("nccl", timeout=timedelta(minutes=30), device_id=torch.device(device))
  trainer = None
  try:
    trainer = PositionTrainer(cfg, rank, world, device)
    trainer.run()
  except BaseException as error:
    output = Path(cfg.output)
    output.mkdir(parents=True, exist_ok=True)
    failure = {
      "error": repr(error), "traceback": traceback.format_exc(),
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
