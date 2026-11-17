"""Native 50 Hz position-control collector and in-place stage manager for footstep WM."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Literal

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import ContactSensor

from g1_lower_rl.assets import ARM_JOINTS, LOWER_BODY_JOINTS, WHOLE_BODY_JOINTS
from g1_lower_rl.footstep_phase import FootstepPhaseCfg
from g1_lower_rl.footsteps.tensor_math import wrap
from g1_lower_rl.scene_graph import enable_scene_cuda_graphs
from g1_lower_rl.tasks.footstep_tracking.commands import FootstepCommand, TorsoHeadingCommand, foot_poses
from g1_lower_rl.tasks.footstep_tracking.reward_math import phase_windows
from g1_lower_rl.tasks.footstep_tracking.rewards import _torso_heading_offset
from g1_lower_rl.tasks.footstep_tracking.terminations import footstep_distance_exceeded
from g1_lower_rl.tasks.lower_body.mdp.events import hold_arm_pose
from g1_lower_rl.tasks.lower_body.mdp.observations import height_above_feet
from g1_lower_rl.world_model.adapters.kinematics import StaticKinematicTemplate
from g1_lower_rl.world_model.collection import CollectorConfig, MotionCollector, TransitionBatch
from g1_lower_rl.world_model.model import CONTROL_DT, JOINT_NAMES, pack_inputs

from .policy import (
  ACTION_DIM,
  DEFAULT_HISTORY,
  FRAME_DIM,
  FootstepTwinMirrorMlpActor,
  command_from_world_queue,
  encode_command,
  encode_proprio_frame,
)
from .task import (
  AUTO_CURRICULUM_STAGES,
  DEFAULT_TERMINATION_PROB,
  CurriculumStage,
  footstep_env_cfg_wm,
  pack_footstep_aux,
)


class FootstepPolicyCollector(MotionCollector):
  """Collect WM transitions, Markovian footstep queue states, and validation metrics at 50 Hz."""

  def __init__(
    self,
    config: CollectorConfig,
    *,
    validation_envs: int,
    actor: FootstepTwinMirrorMlpActor,
    history_length: int = DEFAULT_HISTORY,
    termination_probability: float = DEFAULT_TERMINATION_PROB,
    policy_action_sampling: bool = True,
    scene_cuda_graphs: bool = True,
    env_profile: Literal["walk-first", "step-episode"] = "walk-first",
    play: bool = False,
    enable_corruption: bool | None = None,
    render_mode: Literal["rgb_array"] | None = None,
  ):
    if not 0 <= validation_envs < config.num_envs or config.recover_nonfinite:
      raise ValueError("Footstep WM collection requires 0 <= validation_envs < num_envs and fail-fast physics")
    if policy_action_sampling and config.action_noise <= 0:
      raise ValueError("Policy sampling requires positive Gaussian action_noise")
    if tuple(WHOLE_BODY_JOINTS) != JOINT_NAMES:
      raise AssertionError("WHOLE_BODY_JOINTS must match canonical JOINT_NAMES")

    self.config = config
    self.num_envs = config.num_envs
    self.validation_envs = validation_envs
    self.actor = actor
    self.history_length = history_length
    self.termination_probability = termination_probability
    self.policy_action_sampling = policy_action_sampling
    self.is_step_episode = env_profile == "step-episode"
    self.device = torch.device(config.device)
    self._action_noise = float(config.action_noise)
    self.action_failure: dict[str, torch.Tensor] | None = None
    self.current_stage: CurriculumStage = AUTO_CURRICULUM_STAGES[-1 if play else 0]

    env_cfg = footstep_env_cfg_wm(
      termination_probability=termination_probability,
      play=play,
      profile="step-episode" if self.is_step_episode else "precision",
    )
    env_cfg.scene.num_envs = config.num_envs
    env_cfg.seed = config.seed
    if config.njmax is not None:
      env_cfg.sim.njmax = config.njmax
    if enable_corruption is not None:
      env_cfg.observations["actor"].enable_corruption = enable_corruption

    self.env = ManagerBasedRlEnv(env_cfg, device=config.device, render_mode=render_mode)
    try:
      if not math.isclose(self.env.step_dt, CONTROL_DT, abs_tol=1e-9):
        raise ValueError(f"Footstep WM requires 20ms step_dt, got {self.env.step_dt}")
      if scene_cuda_graphs:
        enable_scene_cuda_graphs(self.env)
      self._initialize_recording()

      robot = self.env.scene["robot"]
      all_ids, all_names = robot.find_joints(WHOLE_BODY_JOINTS, preserve_order=True)
      lower_ids, lower_names = robot.find_joints(LOWER_BODY_JOINTS, preserve_order=True)
      arm_ids, arm_names = robot.find_joints(ARM_JOINTS, preserve_order=True)
      if (
        tuple(all_names) != tuple(WHOLE_BODY_JOINTS)
        or tuple(lower_names) != tuple(LOWER_BODY_JOINTS)
        or tuple(arm_names) != tuple(ARM_JOINTS)
      ):
        raise RuntimeError("Failed to resolve canonical joint ordering on robot entity")
      self._all_joint_ids = torch.tensor(all_ids, device=self.device, dtype=torch.long)
      self._lower_joint_ids = torch.tensor(lower_ids, device=self.device, dtype=torch.long)
      self._arm_joint_ids = torch.tensor(arm_ids, device=self.device, dtype=torch.long)
      self._torso_body_id = robot.find_bodies(("torso_link",), preserve_order=True)[0][0]
      self._foot_site_ids = robot.find_sites(("left_foot", "right_foot"), preserve_order=True)[0]

      action_term = self.env.action_manager.get_term("joint_pos")
      scale_tensor = action_term._scale
      if isinstance(scale_tensor, torch.Tensor):
        self._action_scale = scale_tensor[0].to(device=self.device, dtype=torch.float32).clone()
      else:
        self._action_scale = torch.full((ACTION_DIM,), float(scale_tensor), device=self.device, dtype=torch.float32)

      foot_cmd = self.env.command_manager.get_term("footsteps")
      theta_cmd = self.env.command_manager.get_term("torso_heading")
      if not isinstance(foot_cmd, FootstepCommand) or not isinstance(theta_cmd, TorsoHeadingCommand):
        raise TypeError("Expected FootstepCommand('footsteps') and TorsoHeadingCommand('torso_heading')")
      self._foot_cmd = foot_cmd
      self._theta_cmd = theta_cmd
      self.phase_cfg: FootstepPhaseCfg = foot_cmd.cfg.manager.phase

      arm_ramp_fn = self.env.event_manager.get_term_cfg("reset_arm_pose").func
      if not isinstance(arm_ramp_fn, hold_arm_pose):
        raise TypeError("Expected hold_arm_pose on reset_arm_pose")
      self._arm_ramp_fn = arm_ramp_fn

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
      self._last_stance_xy_err = torch.zeros(self.num_envs, device=self.device)
      self._last_stance_yaw_err = torch.zeros(self.num_envs, device=self.device)
      self._last_ss_match = torch.zeros(self.num_envs, device=self.device)
      self._last_ss_active = torch.zeros(self.num_envs, device=self.device)
      self._last_swing_lift = torch.zeros(self.num_envs, device=self.device)
      self._last_theta_err = torch.zeros(self.num_envs, device=self.device)
      self._last_step_completed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
      self._last_step_accurate = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
      cos_tilt_limit = math.cos(math.radians(70.0))

      reward_manager = self.env.reward_manager
      original_compute = reward_manager.compute

      def compute_reward(dt: float) -> torch.Tensor:
        total = original_compute(dt)
        rs = self._foot_cmd.reward_state
        stance_mask, _ = phase_windows(rs.phase, phase_cfg=self.phase_cfg)
        moving = rs.frequency > 1e-5
        req_stance = stance_mask | ~moving[:, None]

        found = self._feet_sensor.data.found
        assert found is not None
        actual_contact = found > 0
        planted = (stance_mask & actual_contact) | ~moving[:, None]
        eval_mask = torch.where(planted.any(dim=-1, keepdim=True), planted, req_stance).float()

        feet = foot_poses(robot.data, self._foot_site_ids)
        dist_xy = torch.linalg.vector_norm(feet[:, :, :2] - rs.targets_w[:, :, :2], dim=-1)
        yaw_err = wrap(feet[:, :, 2] - rs.targets_w[:, :, 2]).abs()
        denom = eval_mask.sum(dim=-1).clamp_min(1.0)
        self._last_stance_xy_err = (dist_xy * eval_mask).sum(dim=-1) / denom
        self._last_stance_yaw_err = (yaw_err * eval_mask).sum(dim=-1) / denom

        single_sup = (stance_mask.sum(dim=-1) == 1) & moving
        self._last_ss_active = single_sup.float()
        self._last_ss_match = ((stance_mask == actual_contact).all(dim=-1) & single_sup).float()
        feet_z = robot.data.site_pos_w[:, self._foot_site_ids, 2]
        self._last_swing_lift = (feet_z[:, 0] - feet_z[:, 1]).abs() * single_sup.float()

        orientations = torch.cat(
          (
            robot.data.site_quat_w[:, self._foot_site_ids],
            robot.data.body_link_quat_w[:, [self._torso_body_id]],
          ),
          dim=1,
        )
        actual_offset = _torso_heading_offset(orientations)
        desired_theta = (
          self._theta_cmd.command[:, 0]
          if self.current_stage.theta_active
          else torch.zeros_like(actual_offset)
        )
        self._last_theta_err = wrap(actual_offset - desired_theta).abs()

        cur_height = height_above_feet(robot, self._foot_site_ids)
        fell_over = (-robot.data.projected_gravity_b[:, 2]) < cos_tilt_limit
        collapsed = cur_height < 0.35
        dist_fail = footstep_distance_exceeded(self.env)
        self._last_failure = (fell_over | collapsed | self._foot_cmd.failed | dist_fail).clone()
        if self.is_step_episode:
          comp = self._foot_cmd.completed
          f_dist = self._foot_cmd.final_distance
          f_yaw = self._foot_cmd.final_yaw
          self._last_step_completed = comp.clone()
          self._last_step_accurate = (comp & (f_dist <= 0.05) & (f_yaw <= math.radians(10.0))).clone()
        return total

      reward_manager.compute = compute_reward

      # Last half of validation_envs executes zero-noise deterministic actions (exact Viser closed-loop mode!)
      det_count = max(1, self.validation_envs // 2) if self.validation_envs > 0 else 0
      self.det_val_start = self.num_envs - det_count
      self._noise_mask = torch.ones(self.num_envs, 1, device=self.device)
      if det_count > 0:
        self._noise_mask[self.det_val_start :] = 0.0
      self.apply_stage(self.current_stage)
      self.reset_all_envs()

      self.template = StaticKinematicTemplate.from_mujoco(self.env.sim.mj_model, "robot/")
      self.metadata = {
        "contract_version": 3,
        "policy_collection": "current_trainable_footstep_TwinMirror_MLP",
        "history_length": history_length,
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
    return robot.data.default_joint_pos[0, self._all_joint_ids].clone()

  @property
  def hard_joint_pos_limits_canonical(self) -> torch.Tensor:
    robot = self.env.scene["robot"]
    return robot.data.joint_pos_limits[0, self._all_joint_ids].clone()

  def lower_pd_parameters(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kp = self._kp[0, :ACTION_DIM].clone()
    kd = self._kd[0, :ACTION_DIM].clone()
    effort = self._effort[0, :ACTION_DIM, 1].clone()
    return kp, kd, effort

  def set_action_noise(self, action_noise: float) -> None:
    if self.policy_action_sampling and action_noise <= 0:
      raise ValueError("Policy sampling requires positive Gaussian action_noise")
    self._action_noise = float(action_noise)

  def apply_stage(self, stage: CurriculumStage) -> None:
    """Update footstep spatial/cadence/stop/arm/theta distributions in-place."""
    self.current_stage = stage
    batch = self._foot_cmd.batch
    if self.is_step_episode:
      batch.set_sampling_ranges(
        (0.12, 0.35),
        (0.12, 0.30),
        yaw_noise=(-stage.foot_yaw_half_range, stage.foot_yaw_half_range),
      )
    else:
      mgr_init_freq = 1.2 if stage.initial_frequency is None else stage.initial_frequency
      batch.cfg = replace(
        batch.cfg,
        frequency_range=stage.frequency_range,
        initial_frequency=mgr_init_freq,
      )
      batch.source_cfg = replace(
        batch.source_cfg,
        frequency_range=stage.frequency_range,
        initial_frequency=stage.initial_frequency,
        initial_standing_probability=stage.initial_standing_prob,
        stop_probability=stage.stop_prob,
        hold_time_s=(2.0, 5.0),
      )
      batch.initial_direction_range.copy_(
        batch.initial_direction_range.new_tensor((-stage.direction_half_range, stage.direction_half_range))
      )
      batch.change_ranges[0].copy_(
        batch.change_ranges.new_tensor((-stage.direction_change_half_range, stage.direction_change_half_range))
      )
      batch.change_ranges[1].copy_(
        batch.change_ranges.new_tensor((-stage.frequency_rate_half_range, stage.frequency_rate_half_range))
      )
      batch.set_sampling_ranges(
        stage.distance_range,
        stage.width_range,
        yaw_noise=(-stage.foot_yaw_half_range, stage.foot_yaw_half_range),
      )
    for event_name in ("reset_arm_pose", "arm_pose_drift"):
      if event_name in self.env.cfg.events:
        self.env.cfg.events[event_name].params["target_scale"] = stage.arm_target_scale
      try:
        self.env.event_manager.get_term_cfg(event_name).params["target_scale"] = stage.arm_target_scale
      except (KeyError, ValueError):
        pass
    self._theta_cmd.angle_limit = float(stage.theta_limit_rad)

  def reset_all_envs(self) -> None:
    """Full environment reset applying current stage events (`reset_arm_pose` target_scale, etc.) across all worlds."""
    self._observations, _ = self.env.reset()
    self._history = torch.zeros(self.num_envs, self.history_length, FRAME_DIM, device=self.device)
    self._prev_action = torch.zeros(self.num_envs, ACTION_DIM, device=self.device)
    self._episode_fresh = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
    self._was_failing = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self._unbroken_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

  def _step(self) -> None:
    super()._step()
    if self._substep == 0:
      self.env.sim.forward()

  def _current_imu_bias(self) -> torch.Tensor:
    obs_mods = self.env.observation_manager._group_obs_class_instances.get("actor", {})
    keys = ("pelvis_angular", "pelvis_gravity", "torso_angular", "torso_gravity")
    if all(k in obs_mods for k in keys):
      return torch.cat([obs_mods[k]._bias.expand(self.num_envs, 3) for k in keys], dim=-1).clone()
    return torch.zeros(self.num_envs, 12, device=self.device)

  @torch.no_grad()
  def collect_step(self, *, verify_packing: bool = False) -> tuple[TransitionBatch, dict[str, torch.Tensor]]:
    robot = self.env.scene["robot"]
    raw_obs = self._observations["actor"]
    default_pos = self.actor.default_joint_pos

    prev_act = torch.where(self._episode_fresh.unsqueeze(-1), torch.zeros_like(self._prev_action), self._prev_action)
    frame = encode_proprio_frame(raw_obs[:, :70], prev_act, default_pos)
    repeated = frame.unsqueeze(1).expand(-1, self.history_length, -1)
    shifted = torch.cat((self._history[:, 1:], frame.unsqueeze(1)), dim=1)
    self._history = torch.where(self._episode_fresh[:, None, None], repeated, shifted)

    # Extract Markovian planner state in env-local world coordinates (env_origins subtracted on XY)
    env_xy = self.env.scene.env_origins[:, :2]
    st = self._foot_cmd.batch.state
    phase = st["phase"].to(torch.float32).clone()
    frequency = st["frequency"].to(torch.float32).clone()

    def _local_world_poses(poses: torch.Tensor) -> torch.Tensor:
      out = poses.to(torch.float32).clone()
      out[..., :2] = out[..., :2] - env_xy.reshape(self.num_envs, *([1] * (out.ndim - 2)), 2)
      return out

    targets = _local_world_poses(st["targets"])
    supports = _local_world_poses(st["supports"])
    queue = _local_world_poses(st["queue"])
    anchor = _local_world_poses(st["anchor"])
    terminal_feet = _local_world_poses(st["terminal_feet"])
    first_side = st["first_side"].clone()
    pending_anchor = st["pending_anchor"].clone()
    mode = st["mode"].clone()
    elapsed = st["elapsed"].to(torch.float32).clone()
    req_src = st["motion_frequency"] if self.is_step_episode else st["request_frequency"]
    stop_src = st["motion_frequency"] if self.is_step_episode else st["stop_initial_frequency"]
    req_freq = req_src.to(torch.float32).clone()
    stop_init_freq = stop_src.to(torch.float32).clone()
    stop_ramp_start = st["stop_ramp_start"].to(torch.float32).clone()
    stop_phase = st["stop_phase"].to(torch.float32).clone()
    start_dur = torch.full((self.num_envs,), float(self._foot_cmd.batch.cfg.start_duration_s), device=self.device)
    stop_dur = torch.full((self.num_envs,), float(self._foot_cmd.batch.cfg.stop_duration_s), device=self.device)

    if self.current_stage.theta_active:
      theta = self._theta_cmd.command[:, 0].to(torch.float32).clone()
      theta_target = self._theta_cmd.target.to(torch.float32).clone()
      theta_speed = self._theta_cmd.speed.to(torch.float32).clone()
    else:
      theta = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
      theta_target = torch.zeros_like(theta)
      theta_speed = torch.zeros_like(theta)

    # Encode command from native raw_obs and optionally verify exact parity with queue-derived command
    raw_phase = raw_obs[:, 70]
    raw_freq = raw_obs[:, 71]
    raw_fp = raw_obs[:, 72:78].unflatten(-1, (2, 3))
    command = encode_command(raw_phase, raw_freq, raw_fp, theta)
    if verify_packing:
      queue_cmd = command_from_world_queue(phase, frequency, queue, first_side, anchor, theta)
      # Wrap phase modulo 2*pi induces identical sin/cos
      torch.testing.assert_close(command, queue_cmd, rtol=1e-4, atol=1e-4)

    encoder_bias = robot.data.encoder_bias[:, self._all_joint_ids].clone()
    imu_bias = self._current_imu_bias()
    arm_target = robot.data.joint_pos_target[:, self._arm_joint_ids].clone()
    arm_ramp_start = self._arm_ramp_fn.ramp_starts.clone()
    arm_ramp_target = self._arm_ramp_fn.ramp_targets.clone()
    arm_ramp_steps = self._arm_ramp_fn.ramp_steps.clone()
    arm_ramp_active = self._arm_ramp_fn.ramp_active.clone()
    remaining = self.env.max_episode_length - self.env.episode_length_buf.clone()

    found = self._feet_sensor.data.found
    air_time = self._feet_sensor.data.current_air_time
    contact_time = self._feet_sensor.data.current_contact_time
    assert found is not None and air_time is not None and contact_time is not None
    aux = pack_footstep_aux((found > 0).float(), air_time, contact_time, anchor).clone()

    history_snapshot = self._history.clone()
    mean = self.actor(history_snapshot, command)
    if self.policy_action_sampling:
      action = mean + (self._action_noise * self._noise_mask) * torch.randn_like(mean)
    else:
      action = mean
    if not torch.isfinite(action).all():
      indices = torch.nonzero(~torch.isfinite(action).all(-1)).flatten()[:16]
      self.action_failure = {
        "env_indices": indices.cpu(),
        "history": history_snapshot[indices].cpu(),
        "command": command[indices].cpu(),
        "mean_action": mean[indices].cpu(),
        "action": action[indices].cpu(),
      }
      raise FloatingPointError("Nonfinite current-policy native footstep action")

    self._records.clear()
    self._collecting = True
    self.nonfinite_envs.zero_()
    try:
      self._observations, reward, terminated, truncated, _ = self.env.step(action)
    finally:
      self._collecting = False

    if self._substep or len(self._records) != 1 or self.nonfinite_envs.any():
      raise FloatingPointError("Invalid/overflowed native footstep control transition; training stopped")

    batch = TransitionBatch(*(torch.stack([rec[i] for rec in self._records]) for i in range(6)))
    self._records.clear()

    failed = self._last_failure
    first_fail = failed & ~self._was_failing
    healthy = ~failed
    self._unbroken_steps = torch.where(healthy, self._unbroken_steps + 1, torch.zeros_like(self._unbroken_steps))
    unbroken_s = self._unbroken_steps.to(torch.float32) * CONTROL_DT
    done = terminated | truncated
    step_ep_done = (done & self.is_step_episode) if self.is_step_episode else torch.zeros_like(done)
    step_ep_success = self._last_step_accurate.clone() if self.is_step_episode else torch.zeros_like(done)
    self._unbroken_steps = torch.where(done, torch.zeros_like(self._unbroken_steps), self._unbroken_steps)
    self._was_failing = failed & ~done

    self._prev_action.copy_(action)
    self._episode_fresh = done.clone()

    self.counters["failed_control_steps"] += int(failed.sum())
    self.counters["failed_continued_control_steps"] += int((failed & ~terminated & ~truncated).sum())
    self.counters["failed_terminated_control_steps"] += int((failed & terminated).sum())

    if verify_packing:
      expected = pack_inputs(batch.state[0], batch.inputs[0, :, 68:97], batch.inputs[0, :, 97:])
      torch.testing.assert_close(batch.inputs[0], expected)

    return batch, {
      "history": history_snapshot,
      "command": command,
      "mean_action": mean,
      "latest": frame.clone(),
      "aux": aux,
      "action": action,
      "encoder_bias": encoder_bias,
      "imu_bias": imu_bias,
      "arm_target": arm_target,
      "arm_ramp_start": arm_ramp_start,
      "arm_ramp_target": arm_ramp_target,
      "arm_ramp_steps": arm_ramp_steps,
      "arm_ramp_active": arm_ramp_active,
      "phase": phase,
      "frequency": frequency,
      "mode": mode,
      "elapsed": elapsed,
      "req_freq": req_freq,
      "stop_init_freq": stop_init_freq,
      "stop_ramp_start": stop_ramp_start,
      "stop_phase": stop_phase,
      "start_dur": start_dur,
      "stop_dur": stop_dur,
      "terminal_feet": terminal_feet,
      "targets": targets,
      "supports": supports,
      "queue": queue,
      "first_side": first_side,
      "pending_anchor": pending_anchor,
      "theta": theta,
      "theta_target": theta_target,
      "theta_speed": theta_speed,
      "remaining": remaining,
      "stance_xy_err": self._last_stance_xy_err.clone(),
      "stance_yaw_err": self._last_stance_yaw_err.clone(),
      "ss_match": self._last_ss_match.clone(),
      "ss_active": self._last_ss_active.clone(),
      "swing_lift": self._last_swing_lift.clone(),
      "theta_err": self._last_theta_err.clone(),
      "failed": failed,
      "first_fail": first_fail,
      "healthy": healthy,
      "unbroken_s": unbroken_s,
      "step_ep_done": step_ep_done,
      "step_ep_success": step_ep_success,
      "reward": reward,
      "terminated": terminated,
      "truncated": truncated,
    }

  def save_action_failure(self, path) -> None:
    if self.action_failure is None:
      raise RuntimeError("No nonfinite native footstep action diagnostic is available")
    torch.save(
      {
        "format": "native_footstep_policy_nonfinite_diagnostic_v1",
        "inputs": self.action_failure,
        "actor_state_dict": {name: value.detach().cpu() for name, value in self.actor.state_dict().items()},
      },
      path,
    )

  def close(self) -> None:
    self.env.close()
