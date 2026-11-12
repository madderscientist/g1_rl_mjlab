"""Native 50 Hz position-control collection for lower-body velocity WM training."""

from __future__ import annotations

import math
from typing import Literal

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import ContactSensor

from g1_lower_rl.assets import ARM_JOINTS, LOWER_BODY_JOINTS
from g1_lower_rl.scene_graph import enable_scene_cuda_graphs
from g1_lower_rl.tasks.lower_body.mdp.commands import BaseHeightCommand, ScenarioVelocityCommand
from g1_lower_rl.tasks.lower_body.mdp.events import gait_phase_offset, hold_arm_pose
from g1_lower_rl.tasks.lower_body.mdp.observations import height_above_feet
from g1_lower_rl.tasks.lower_body.mdp.rewards.tracking import (
  stationary_drift,
  track_angular_velocity_avg,
  track_linear_velocity_avg,
)
from g1_lower_rl.world_model.adapters.kinematics import StaticKinematicTemplate
from g1_lower_rl.world_model.collection import CollectorConfig, MotionCollector, TransitionBatch
from g1_lower_rl.world_model.model import CONTROL_DT, JOINT_NAMES, pack_inputs

from .policy import (
  ACTION_DIM,
  DEFAULT_HISTORY,
  VelocityMlpActor,
  unflatten_native_history,
)
from .task import DEFAULT_TERMINATION_PROB, flat_env_cfg_wm, pack_task_aux

CONTINUOUS_PARITY_TERMS: tuple[str, ...] = (
  "track_linear_velocity",
  "track_linear_velocity_avg",
  "track_angular_velocity",
  "track_angular_velocity_avg",
  "stationary_drift",
  "velocity_shortfall",
  "track_base_height",
  "straight_knee",
  "body_orientation_l2",
  "pelvis_backward_lean",
  "waist_deviation",
  "waist_deviation_still",
  "waist_effort_still",
  "hip_yaw_still",
  "pose",
  "body_ang_vel",
  "action_rate_l2",
  "action_rate_still",
)


class VelocityPolicyCollector(MotionCollector):
  """Collect transitions and Markovian task states from native velocity-tracking environments."""

  def __init__(
    self,
    config: CollectorConfig,
    *,
    validation_envs: int,
    actor: VelocityMlpActor,
    history_length: int = DEFAULT_HISTORY,
    curriculum_steps_per_iter: float = 2.0,
    termination_probability: float = DEFAULT_TERMINATION_PROB,
    policy_action_sampling: bool = True,
    scene_cuda_graphs: bool = True,
    play: bool = False,
    enable_corruption: bool | None = None,
    render_mode: Literal["rgb_array"] | None = None,
  ):
    if not 0 <= validation_envs < config.num_envs or config.recover_nonfinite:
      raise ValueError("Velocity WM collection requires 0 <= validation_envs < num_envs and fail-fast physics")
    if policy_action_sampling and config.action_noise <= 0:
      raise ValueError("Policy sampling requires positive Gaussian action_noise")
    if tuple(LOWER_BODY_JOINTS) != JOINT_NAMES[:ACTION_DIM] or tuple(ARM_JOINTS) != JOINT_NAMES[ACTION_DIM:]:
      raise AssertionError("Canonical 29-joint order must place LOWER_BODY_JOINTS[:15] before ARM_JOINTS[15:29]")

    self.config = config
    self.num_envs = config.num_envs
    self.validation_envs = validation_envs
    self.actor = actor
    self.history_length = history_length
    self.termination_probability = termination_probability
    self.policy_action_sampling = policy_action_sampling
    self.device = torch.device(config.device)
    self.action_failure: dict[str, torch.Tensor] | None = None

    env_cfg = flat_env_cfg_wm(
      history_length=history_length,
      curriculum_steps_per_iter=curriculum_steps_per_iter,
      termination_probability=termination_probability,
      play=play,
    )
    env_cfg.scene.num_envs = config.num_envs
    env_cfg.seed = config.seed
    if config.njmax is not None:
      env_cfg.sim.njmax = config.njmax
    if enable_corruption is not None:
      env_cfg.observations["actor"].enable_corruption = enable_corruption

    self.env = ManagerBasedRlEnv(env_cfg, device=config.device, render_mode=render_mode)
    try:
      if self.env.step_dt != CONTROL_DT:
        raise ValueError("Velocity WM requires native 20ms control steps")
      if scene_cuda_graphs:
        enable_scene_cuda_graphs(self.env)
      self._initialize_recording()

      robot = self.env.scene["robot"]
      lower_ids, lower_names = robot.find_joints(LOWER_BODY_JOINTS, preserve_order=True)
      arm_ids, arm_names = robot.find_joints(ARM_JOINTS, preserve_order=True)
      if tuple(lower_names) != tuple(LOWER_BODY_JOINTS) or tuple(arm_names) != tuple(ARM_JOINTS):
        raise RuntimeError("Failed to resolve canonical lower-body or arm joint order on entity")
      self._lower_joint_ids = torch.tensor(lower_ids, device=self.device, dtype=torch.long)
      self._arm_joint_ids = torch.tensor(arm_ids, device=self.device, dtype=torch.long)
      self._foot_site_ids, _ = robot.find_sites(["left_foot", "right_foot"], preserve_order=True)

      action_term = self.env.action_manager.get_term("joint_pos")
      scale_tensor = action_term._scale
      if isinstance(scale_tensor, torch.Tensor):
        self._action_scale = scale_tensor[0].to(device=self.device, dtype=torch.float32).clone()
      else:
        self._action_scale = torch.full((ACTION_DIM,), float(scale_tensor), device=self.device, dtype=torch.float32)

      twist_cmd = self.env.command_manager.get_term("twist")
      height_cmd = self.env.command_manager.get_term("height")
      if not isinstance(twist_cmd, ScenarioVelocityCommand) or not isinstance(height_cmd, BaseHeightCommand):
        raise TypeError("Expected ScenarioVelocityCommand('twist') and BaseHeightCommand('height')")
      self._twist_cmd = twist_cmd
      self._height_cmd = height_cmd

      arm_ramp_fn = self.env.event_manager.get_term_cfg("reset_arm_pose").func
      if not isinstance(arm_ramp_fn, hold_arm_pose):
        raise TypeError("Expected hold_arm_pose on reset_arm_pose")
      self._arm_ramp_fn = arm_ramp_fn

      lin_avg_fn = self.env.reward_manager.get_term_cfg("track_linear_velocity_avg").func
      ang_avg_fn = self.env.reward_manager.get_term_cfg("track_angular_velocity_avg").func
      drift_fn = self.env.reward_manager.get_term_cfg("stationary_drift").func
      if (
        not isinstance(lin_avg_fn, track_linear_velocity_avg)
        or not isinstance(ang_avg_fn, track_angular_velocity_avg)
        or not isinstance(drift_fn, stationary_drift)
      ):
        raise TypeError("Unexpected stateful tracking reward term classes")
      self._lin_avg_fn = lin_avg_fn
      self._ang_avg_fn = ang_avg_fn
      self._drift_fn = drift_fn

      feet_sensor = self.env.scene["feet_ground_contact"]
      if not isinstance(feet_sensor, ContactSensor):
        raise TypeError("Expected ContactSensor for feet_ground_contact")
      self._feet_sensor = feet_sensor

      self.counters.update(
        failed_control_steps=0,
        failed_continued_control_steps=0,
        failed_terminated_control_steps=0,
      )
      self._last_failure = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
      self._last_continuous_sum = torch.zeros(self.num_envs, device=self.device)
      self._last_drift_anchor = torch.zeros(self.num_envs, 2, device=self.device)
      self._last_lin_vel_err = torch.zeros(self.num_envs, device=self.device)
      self._last_ang_vel_err = torch.zeros(self.num_envs, device=self.device)
      self._last_height_err = torch.zeros(self.num_envs, device=self.device)
      cos_tilt_limit = math.cos(math.radians(70.0))

      reward_manager = self.env.reward_manager
      parity_indices = [reward_manager._term_names.index(name) for name in CONTINUOUS_PARITY_TERMS]
      self._parity_term_indices = torch.tensor(parity_indices, device=self.device, dtype=torch.long)
      original_compute = reward_manager.compute

      def compute_reward(dt: float) -> torch.Tensor:
        total = original_compute(dt)
        # _step_reward holds [num_envs, num_terms] unweighted * weight (i.e. reward / dt)
        step_rew = reward_manager._step_reward
        self._last_continuous_sum = step_rew.index_select(1, self._parity_term_indices).sum(dim=1).clone()
        self._last_drift_anchor = (self._drift_fn._anchor - self.env.scene.env_origins[:, :2]).clone()
        cmd = self._twist_cmd.command
        h_cmd = self._height_cmd.command[:, 0]
        cur_height = height_above_feet(robot, self._foot_site_ids)
        self._last_lin_vel_err = torch.linalg.vector_norm(cmd[:, :2] - robot.data.root_link_lin_vel_b[:, :2], dim=1)
        self._last_ang_vel_err = (cmd[:, 2] - robot.data.root_link_ang_vel_b[:, 2]).abs()
        self._last_height_err = (h_cmd - cur_height).abs()
        fell_over = (-robot.data.projected_gravity_b[:, 2]) < cos_tilt_limit
        collapsed = cur_height < 0.35
        self._last_failure = (fell_over | collapsed).clone()
        return total

      reward_manager.compute = compute_reward

      self._observations, _ = self.env.reset()
      self.template = StaticKinematicTemplate.from_mujoco(self.env.sim.mj_model, "robot/")
      self.metadata = {
        "contract_version": 3,
        "policy_collection": "current_trainable_velocity_MLP",
        "history_length": history_length,
        "curriculum_steps_per_iter": curriculum_steps_per_iter,
        "termination_probability": termination_probability,
        "static_template": self.template.to_dict(),
        "physics_dt": 0.005,
        "control_dt": CONTROL_DT,
        "controller": "default_lower15_plus_scaled_action_and_arm_pd_target14_no_GC",
        "shared_foot_friction": True,
        "unmodeled_external_wrenches": False,
        "njmax": int(self.env.sim.wp_data.njmax),
        "naconmax": int(self.env.sim.wp_data.naconmax),
      }
    except BaseException:
      self.env.close()
      raise

  @property
  def action_scale(self) -> torch.Tensor:
    return self._action_scale

  @property
  def default_joint_pos_canonical(self) -> torch.Tensor:
    robot = self.env.scene["robot"]
    canonical_ids = torch.cat((self._lower_joint_ids, self._arm_joint_ids))
    return robot.data.default_joint_pos[0, canonical_ids].clone()

  @property
  def soft_joint_pos_limits_canonical(self) -> torch.Tensor:
    robot = self.env.scene["robot"]
    canonical_ids = torch.cat((self._lower_joint_ids, self._arm_joint_ids))
    return robot.data.soft_joint_pos_limits[0, canonical_ids].clone()

  @property
  def current_height_std(self) -> float:
    return float(self.env.reward_manager.get_term_cfg("track_base_height").params["std"])

  def waist_pd_parameters(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    waist_indices = [JOINT_NAMES.index(f"waist_{axis}_joint") for axis in ("yaw", "roll", "pitch")]
    kp = self._kp[0, waist_indices].clone()
    kd = self._kd[0, waist_indices].clone()
    effort = self._effort[0, waist_indices, 1].clone()
    return kp, kd, effort

  def _step(self) -> None:
    super()._step()
    if self._substep == 0:
      # Refresh kinematics after final-substep integration before native reward/termination compute.
      self.env.sim.forward()

  def _current_imu_bias(self) -> torch.Tensor:
    obs_mods = self.env.observation_manager._group_obs_class_instances.get("actor", {})
    if "base_ang_vel" in obs_mods and "projected_gravity" in obs_mods:
      ang_b = obs_mods["base_ang_vel"]._bias.expand(self.num_envs, 3)
      grav_b = obs_mods["projected_gravity"]._bias.expand(self.num_envs, 3)
      return torch.cat((ang_b, grav_b), dim=-1).clone()
    return torch.zeros(self.num_envs, 6, device=self.device)

  def set_action_noise(self, action_noise: float) -> None:
    if self.policy_action_sampling and action_noise <= 0:
      raise ValueError("Policy sampling requires positive Gaussian action_noise")
    self._action_noise = float(action_noise)

  @torch.no_grad()
  def collect_step(self, *, verify_packing: bool = False) -> tuple[TransitionBatch, dict[str, torch.Tensor]]:
    robot = self.env.scene["robot"]
    history = unflatten_native_history(self._observations["actor"], self.history_length)
    encoder_bias = robot.data.encoder_bias[:, self._lower_joint_ids].clone()
    imu_bias = self._current_imu_bias()

    arm_target = robot.data.joint_pos_target[:, self._arm_joint_ids].clone()
    arm_ramp_start = self._arm_ramp_fn.ramp_starts.clone()
    arm_ramp_target = self._arm_ramp_fn.ramp_targets.clone()
    arm_ramp_steps = self._arm_ramp_fn.ramp_steps.clone()
    arm_ramp_active = self._arm_ramp_fn.ramp_active.clone()

    twist = self._twist_cmd.command.clone()
    height_cmd = self._height_cmd.command[:, 0].clone()
    height_target = self._height_cmd.height_target[:, 0].clone()
    episode_step = self.env.episode_length_buf.clone()
    phase_offset = gait_phase_offset(self.env).clone()
    remaining = self.env.max_episode_length - episode_step

    left = self._twist_cmd.time_left
    assert left is not None
    prev_left = left if self._lin_avg_fn._prev_left is None else self._lin_avg_fn._prev_left
    fresh = (left > prev_left + 1e-6) | (episode_step == 0)
    lin_accum = torch.where(fresh.unsqueeze(-1), torch.zeros_like(self._lin_avg_fn._accum), self._lin_avg_fn._accum)
    ang_accum = torch.where(fresh, torch.zeros_like(self._ang_avg_fn._accum), self._ang_avg_fn._accum)
    cmd_elapsed = torch.where(fresh, torch.zeros_like(self._lin_avg_fn._elapsed), self._lin_avg_fn._elapsed).clone()
    current_xy = robot.data.root_link_pos_w[:, :2] - self.env.scene.env_origins[:, :2]
    stored_anchor = self._drift_fn._anchor - self.env.scene.env_origins[:, :2]
    drift_anchor = torch.where(fresh.unsqueeze(-1), current_xy, stored_anchor).clone()
    is_stationary = (self._twist_cmd.is_turning_env | self._twist_cmd.is_standing_env).clone()

    found = self._feet_sensor.data.found
    air_time = self._feet_sensor.data.current_air_time
    contact_time = self._feet_sensor.data.current_contact_time
    assert found is not None and air_time is not None and contact_time is not None
    aux = pack_task_aux(lin_accum, ang_accum, (found > 0).float(), air_time, contact_time).clone()

    mean = self.actor(history)
    noise_std = getattr(self, "_action_noise", self.config.action_noise)
    action = mean + noise_std * torch.randn_like(mean) if self.policy_action_sampling else mean
    if not torch.isfinite(action).all():
      indices = torch.nonzero(~torch.isfinite(action).all(-1)).flatten()[:16]
      self.action_failure = {
        "env_indices": indices.cpu(),
        "history": history[indices].cpu(),
        "mean_action": mean[indices].cpu(),
        "action": action[indices].cpu(),
      }
      raise FloatingPointError("Nonfinite current-policy native velocity action")

    self._records.clear()
    self._collecting = True
    self.nonfinite_envs.zero_()
    try:
      self._observations, reward, terminated, truncated, _ = self.env.step(action)
    finally:
      self._collecting = False

    if self._substep or len(self._records) != 1 or self.nonfinite_envs.any():
      raise FloatingPointError("Invalid/overflowed native velocity control transition; training stopped")

    batch = TransitionBatch(*(torch.stack([record[i] for record in self._records]) for i in range(6)))
    self._records.clear()

    failed = self._last_failure
    self.counters["failed_control_steps"] += int(failed.sum())
    self.counters["failed_continued_control_steps"] += int((failed & ~terminated & ~truncated).sum())
    self.counters["failed_terminated_control_steps"] += int((failed & terminated).sum())

    if verify_packing:
      expected = pack_inputs(batch.state[0], batch.inputs[0, :, 68:97], batch.inputs[0, :, 97:])
      torch.testing.assert_close(batch.inputs[0], expected)

    return batch, {
      "history": history,
      "mean_action": mean,
      "latest": history[:, -1],
      "aux": aux,
      "action": action,
      "encoder_bias": encoder_bias,
      "imu_bias": imu_bias,
      "arm_target": arm_target,
      "arm_ramp_start": arm_ramp_start,
      "arm_ramp_target": arm_ramp_target,
      "arm_ramp_steps": arm_ramp_steps,
      "arm_ramp_active": arm_ramp_active,
      "twist": twist,
      "height_cmd": height_cmd,
      "height_target": height_target,
      "episode_step": episode_step,
      "phase_offset": phase_offset,
      "remaining": remaining,
      "cmd_elapsed": cmd_elapsed,
      "drift_anchor": drift_anchor,
      "post_drift_anchor": self._last_drift_anchor,
      "is_stationary": is_stationary,
      "continuous_sum": self._last_continuous_sum,
      "lin_vel_err": self._last_lin_vel_err,
      "ang_vel_err": self._last_ang_vel_err,
      "height_err": self._last_height_err,
      "failed": failed,
      "reward": reward,
      "terminated": terminated,
      "truncated": truncated,
    }

  def curriculum_state_dict(self) -> dict[str, object]:
    manager = self.env.curriculum_manager
    states: dict[str, object] = {"common_step_counter": int(self.env.common_step_counter)}
    for name in manager.active_terms:
      fn = manager.get_term_cfg(name).func
      if hasattr(fn, "state_dict"):
        states[name] = fn.state_dict()
    return states

  def load_curriculum_state_dict(self, states: dict[str, object]) -> None:
    if "common_step_counter" in states:
      self.env.common_step_counter = int(states["common_step_counter"])  # type: ignore[arg-type]
    all_envs = torch.arange(self.num_envs, device=self.device)
    manager = self.env.curriculum_manager
    for name in manager.active_terms:
      cfg = manager.get_term_cfg(name)
      if name in states and hasattr(cfg.func, "load_state_dict"):
        cfg.func.load_state_dict(states[name])  # type: ignore[arg-type]
      cfg.func(self.env, all_envs, **cfg.params)
    # Refresh commands and observations under the restored curriculum ranges before warmup collection.
    self.env.command_manager.reset(env_ids=all_envs)
    self._observations = self.env.observation_manager.compute(update_history=True)

  def save_action_failure(self, path) -> None:
    if self.action_failure is None:
      raise RuntimeError("No nonfinite native velocity action diagnostic is available")
    torch.save(
      {
        "format": "native_velocity_policy_nonfinite_diagnostic_v1",
        "inputs": self.action_failure,
        "actor_state_dict": {name: value.detach().cpu() for name, value in self.actor.state_dict().items()},
      },
      path,
    )

  def close(self) -> None:
    self.env.close()
