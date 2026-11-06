"""Current-policy collection with the v3 WM's native position-control contract."""

from __future__ import annotations

import importlib
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import torch

from g1_lower_rl.tasks.motion_tracking.corpus import file_sha256
from g1_lower_rl.world_model.adapters.kinematics import StaticKinematicTemplate
from g1_lower_rl.world_model.baseline_collection import FullCorpusPositionCollector, NativeMotionCommand
from g1_lower_rl.world_model.collection import CollectorConfig, TransitionBatch
from g1_lower_rl.world_model.mocap import MmapMotionLibrary
from g1_lower_rl.world_model.model import CONTROL_DT, pack_inputs, world_to_root
from g1_lower_rl.world_model.native import fixed29_runtime

from .position_policy import (
  REFERENCE_OFFSETS,
  actor_observations,
  reference_tokens,
  unflatten_native_history,
)
from .position_task import RECOVERY_XY_TASK, TaskDefinition, configure_native_task


class PositionPolicyCollector(FullCorpusPositionCollector):
  def __init__(
    self, config: CollectorConfig, *, replica_source: str, validation_envs: int, actor,
    task_definition: TaskDefinition = RECOVERY_XY_TASK, gamma: float = 0.99,
    policy_action_sampling: bool = False,
    render_mode: Literal["rgb_array"] | None = None,
  ):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.sensor import ContactMatch, ContactSensorCfg

    if not 0 < validation_envs < config.num_envs or config.recover_nonfinite or config.checkpoint:
      raise ValueError("Current-policy collection needs disjoint partitions, no teacher and fail-fast physics")
    source = Path(replica_source).resolve()
    if not (source / "src/supermimic/envs/tracking_env_cfg.py").is_file():
      raise FileNotFoundError(source)
    sys.path.insert(0, str(source / "src"))
    package = importlib.import_module("supermimic")
    if package.__file__ is None or Path(package.__file__).resolve().parents[2] != source:
      raise RuntimeError("A different supermimic package is already imported")
    self.config, self.num_envs, self.actor = config, config.num_envs, actor
    self.action_failure: dict[str, torch.Tensor] | None = None
    self.policy_action_sampling = policy_action_sampling
    if policy_action_sampling and config.action_noise <= 0:
      raise ValueError("Policy sampling requires positive fixed Gaussian exploration")
    self.device = torch.device(config.device)
    self._runtime = ExitStack()
    self._initialization = {"reference": 0, "half_squat": 0, "airborne": 0}
    try:
      self._runtime.enter_context(fixed29_runtime({"observed_initialization_counts": self._initialization}))
      native_cfg = importlib.import_module("supermimic.envs.tracking_env_cfg")
      commands = importlib.import_module("supermimic.envs.mdp.commands")

      def library_factory(npz_files, body_names, default_joint_pos, device, bin_frames, **_kwargs):
        if npz_files or not config.motion_dir:
          raise ValueError("Only the complete manifest-backed mocap corpus is supported")
        return MmapMotionLibrary(Path(config.motion_dir), body_names, default_joint_pos, device, bin_frames)

      args = native_cfg.SuperMimicEnvCfgArgs(
        npz_files=[], num_envs=config.num_envs, action_noise_std=0.0 if policy_action_sampling else config.action_noise,
        record_transitions=False, seed=config.seed, sampling_mode="return",
      )
      cfg = native_cfg.make_env_cfg(args)
      failure_terms = tuple(cfg.terminations[name] for name in ("root_height", "root_tilt"))
      task_contract = configure_native_task(cfg, task_definition, gamma)
      cfg.observations.pop("goal")
      cfg.actions["joint_pos"].use_default_offset = False
      cfg.sim.njmax = config.njmax or 500
      cfg.scene.sensors = (*cfg.scene.sensors, ContactSensorCfg(
        name="wm_feet",
        primary=ContactMatch(mode="body", pattern=r"^(left|right)_ankle_roll_link$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found", "force"), reduce="netforce",
      ))
      with patch.object(commands, "MotionLibrary", library_factory):
        self.env = ManagerBasedRlEnv(cfg, device=config.device, render_mode=render_mode)
      if self.env.step_dt != CONTROL_DT:
        raise ValueError("Position policy requires native20ms control steps")
      self._initialize_recording()
      self.counters.update(failed_control_steps=0, failed_continued_control_steps=0, failed_terminated_control_steps=0)
      self._last_failure = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
      command = self.env.command_manager.get_term("motion")
      if not isinstance(command, NativeMotionCommand):
        raise TypeError("Expected the native mapped motion command")
      self._native_command = command
      self.library = command.library
      self._validation = torch.arange(self.num_envs, device=self.device) >= self.num_envs - validation_envs
      self._sampling_validation = False
      self._original_native_resample = command._resample_command
      command._resample_command = self._resample
      command._sample_frames = self._sample_frames
      self.train_bins = int((self.library.bin_clip < self.library.corpus.training_motions).sum())
      self._action = self.env.action_manager.get_term("joint_pos")
      original_process = self._action.process_actions

      def process(actions):
        original_process(actions)
        self._action._processed_actions += command.joint_pos
        self.executed_action = self._action.executed_raw.clone()

      self._action.process_actions = process
      reward_compute = self.env.reward_manager.compute

      def compute_reward(dt):
        command.update_relative_body_poses()
        self._last_failure = torch.stack([term.func(self.env, **term.params) for term in failure_terms]).any(0)
        return reward_compute(dt)

      self.env.reward_manager.compute = compute_reward
      self._observations, _ = self.env.reset()
      self.template = StaticKinematicTemplate.from_mujoco(self.env.sim.mj_model, "robot/")
      self.source_manifest = {
        str(p.relative_to(source)): file_sha256(p) for p in sorted((source / "src").rglob("*.py"))
      }
      self.metadata = {
        "contract_version": 3,
        "policy_collection": "current_trainable_MLP",
        "corpus": self.library.metadata, "replica_source_manifest": self.source_manifest,
        "static_template": self.template.to_dict(), "physics_dt": 0.005, "control_dt": CONTROL_DT,
        "controller": "reference_q_plus_scaled_noisy_residual_minus_encoder_bias_no_GC",
        "sampling": "train50pct_uniform_clip_40pct_current_return_10pct_memory_validation_uniform_disjoint",
        "gripper_state": False, "njmax": self.env.sim.wp_data.njmax,
        "naconmax": self.env.sim.wp_data.naconmax,
        "endpoint_kinematics": "native_forward_after_integration_before_reward_and_reanchor",
        "task_contract": task_contract,
        "clip_boundary": "native_clip_end_latch_v1",
      }
      if policy_action_sampling:
        self.metadata["action_sampling"] = "policy_Gaussian_fixed_std_raw_action_history_and_penalty_no_execution_noise"
    except BaseException:
      if hasattr(self, "env"):
        self.env.close()
      self._runtime.close()
      raise

  @property
  def action_scale(self) -> torch.Tensor:
    value = self._action._scale
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[-1] != 29:
      raise TypeError("Native residual controller requires canonical29 per-world action scales")
    return value[0]

  def _sample_frames(self, count: int) -> torch.Tensor:
    if self._sampling_validation:
      return self.library.sample(count, validation=True)
    command, lib = self._native_command, self.library
    hard = torch.softmax(-25 * command.bin_return[:self.train_bins] / command.cfg.g_max, dim=0)
    memory = command.bin_memory[:self.train_bins]
    memory.lerp_(hard, command.cfg.memory_alpha)
    bins = torch.multinomial(0.8 * hard + 0.2 * memory / memory.sum(), count, replacement=True)
    adaptive = lib.bin_start[bins] + (torch.rand(count, device=self.device) * (lib.bin_end[bins] - lib.bin_start[bins])).long()
    uniform = torch.rand(count, device=self.device) < 0.5
    frames = adaptive
    frames[uniform] = lib.sample(int(uniform.sum()), validation=False)
    lib.visited[lib.frame_clip[frames]] = True
    return frames

  def _step(self) -> None:
    super()._step()
    if self._substep == 0:
      # MuJoCo's integrated qpos is newer than its last-substep link kinematics.
      self.env.sim.forward()

  def reference_state(self, frames: torch.Tensor) -> torch.Tensor:
    r = self.library.ref(frames)
    return torch.cat((
      r.root_pos_w, r.root_quat_w, r.joint_pos, r.root_lin_vel_w,
      world_to_root(r.root_quat_w, r.root_ang_vel_w), r.joint_vel,
    ), dim=-1)

  def reference_window(self, frames: torch.Tensor) -> torch.Tensor:
    lib = self.library
    offsets = torch.tensor(REFERENCE_OFFSETS, device=self.device)
    windows = []
    for block in frames.split(2048):
      clip = lib.frame_clip[block]
      index = (block[:, None] + offsets).clamp(min=lib.clip_start[clip, None], max=lib.clip_last[clip, None])
      windows.append(reference_tokens(self.reference_state(index)).flatten(1))
    return torch.cat(windows)

  def reference_trajectory(self, frames: torch.Tensor, horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = torch.tensor(REFERENCE_OFFSETS, device=self.device)
    first, last = min(REFERENCE_OFFSETS), horizon + max(REFERENCE_OFFSETS)
    dense = torch.arange(first, last + 1, device=self.device)
    steps = torch.arange(horizon + 1, device=self.device)
    states, windows = [], []
    for block in frames.split(max(1, 43008 // len(dense))):
      clip = self.library.frame_clip[block]
      index = (block[:, None] + dense).clamp(
        min=self.library.clip_start[clip, None], max=self.library.clip_last[clip, None],
      )
      reference = self.reference_state(index)
      states.append(reference[:, -first:-first + horizon + 1])
      centers = torch.minimum(steps[None], self.library.clip_last[clip, None] - block[:, None])
      lookup = centers[..., None] + offsets - first
      rows = torch.arange(len(block), device=self.device)[:, None, None]
      windows.append(reference_tokens(reference)[rows, lookup].flatten(-2))
    return torch.cat(states), torch.cat(windows)

  @torch.no_grad()
  def collect_step(self) -> tuple[TransitionBatch, dict[str, torch.Tensor]]:
    history = unflatten_native_history(self._observations["policy"])
    frame = self._native_command.frame.clone()
    clip_end = self._native_command.clip_end.clone()
    bias = self.env.scene["robot"].data.encoder_bias.clone()
    remaining = self.env.max_episode_length - self.env.episode_length_buf.clone()
    reference = self.reference_window(frame)
    mean = self.actor(actor_observations(history, reference), stochastic_output=False)
    action = mean + self.config.action_noise * torch.randn_like(mean) if self.policy_action_sampling else mean
    if not torch.isfinite(action).all():
      indices = torch.nonzero(~torch.isfinite(action).all(-1)).flatten()[:16]
      self.action_failure = {
        "env_indices": indices.cpu(), "history": history[indices].cpu(), "reference": reference[indices].cpu(),
        "frame": frame[indices].cpu(), "mean_action": mean[indices].cpu(), "action": action[indices].cpu(),
      }
      raise FloatingPointError("Nonfinite current-policy native action")
    self._records.clear()
    self._collecting = True
    self.nonfinite_envs.zero_()
    try:
      self._observations, reward, terminated, truncated, _ = self.env.step(action)
    finally:
      self._collecting = False
    if self._substep or len(self._records) != 1 or self.nonfinite_envs.any():
      raise FloatingPointError("Invalid/overflowed native control transition; training stopped")
    batch = TransitionBatch(*(torch.stack([record[i] for record in self._records]) for i in range(6)))
    self._records.clear()
    failed = self._last_failure
    self.counters["failed_control_steps"] += int(failed.sum())
    self.counters["failed_continued_control_steps"] += int((failed & ~terminated & ~truncated).sum())
    self.counters["failed_terminated_control_steps"] += int((failed & terminated).sum())
    self.counters.update(self.library.coverage())
    self.counters.update({f"initial_{k}": v for k, v in self._initialization.items()})
    returns = self._native_command.bin_return[:self.train_bins]
    self.counters.update(
      train_bins_with_return_updates=int((returns != self._native_command.cfg.g_max).sum()),
      train_bin_return_min=float(returns.min()),
      train_bin_return_max=float(returns.max()),
    )
    expected = pack_inputs(batch.state[0], batch.inputs[0, :, 68:97], batch.inputs[0, :, 97:])
    torch.testing.assert_close(batch.inputs[0], expected)
    return batch, {
      "history": history, "reference": reference, "mean_action": mean,
      "latest": history[:, -1], "frame": frame, "bias": bias, "remaining": remaining,
      "clip_end": clip_end,
      "action": action, "executed_action": self.executed_action,
      "reward": reward, "terminated": terminated, "truncated": truncated,
    }

  def save_action_failure(self, path: Path) -> None:
    if self.action_failure is None:
      raise RuntimeError("No nonfinite native action diagnostic is available")
    # Diagnostics intentionally preserve nonfinite tensors and cannot be resumed as checkpoints.
    torch.save({
      "format": "native_policy_nonfinite_diagnostic_v1", "inputs": self.action_failure,
      "actor_state_dict": {name: value.detach().cpu() for name, value in self.actor.state_dict().items()},
      "policy_collection": self.metadata["policy_collection"],
    }, path)

  def close(self) -> None:
    try:
      self.env.close()
    finally:
      self._runtime.close()
