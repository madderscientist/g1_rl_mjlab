"""WM-v3 aligned velocity environment config, schema-driven replay, and fused differentiable task."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple

import torch
import torch.nn.functional as F
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from torch import nn

from g1_lower_rl.tasks.lower_body.cfg.constants import FOOT_GEOMS, HEIGHT_STD_STAGES, ITER
from g1_lower_rl.tasks.lower_body.cfg.env_cfg import _rescale_curriculum_steps, flat_env_cfg
from g1_lower_rl.world_model.adapters.kinematics import KinematicState, TorchKinematics
from g1_lower_rl.world_model.adapters.math import conjugate, rotate, unit
from g1_lower_rl.world_model.model import BODY_NAMES, CONTROL_DT, JOINT_NAMES, PositionWorldModel, state_features
from g1_lower_rl.world_model.training.replay import RingReplay

from .policy import ACTION_DIM, ARM_DIM, DEFAULT_HISTORY, GAIT_PERIOD, OBS_DIM

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg

CRITIC_DIM = 167
AUX_DIM = 9
ARM_RAMP_DURATION_S = 5.0
HEIGHT_MAX_RATE = 0.15
DEFAULT_TERMINATION_PROB = 0.05


def _probabilistic_termination(
  env: ManagerBasedRlEnv,
  base_func: Callable[..., torch.Tensor],
  p_term: float,
  **params,
) -> torch.Tensor:
  """Share one per-tick Bernoulli coin across failure predicates so union hazard equals p_term."""
  step = int(env.common_step_counter)
  cache = getattr(env, "_vel_wm_coin_cache", None)
  if cache is None or cache[0] != step or cache[1].shape[0] != env.num_envs:
    cache = (step, torch.rand(env.num_envs, device=env.device) < p_term)
    env._vel_wm_coin_cache = cache  # type: ignore[attr-defined]
  return base_func(env, **params) & cache[1]


def flat_env_cfg_wm(
  history_length: int = DEFAULT_HISTORY,
  curriculum_steps_per_iter: float = 2.0,
  termination_probability: float = DEFAULT_TERMINATION_PROB,
  *,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Construct lower-body velocity task aligned with Position WM v3 physical contract."""
  if history_length < 1 or curriculum_steps_per_iter <= 0:
    raise ValueError("history_length must be >= 1 and curriculum_steps_per_iter > 0")
  if not math.isfinite(termination_probability) or not 0.0 < termination_probability <= 1.0:
    raise ValueError("termination_probability must lie in (0, 1]")
  cfg = flat_env_cfg(play=play)

  for event_name in ("arm_torque", "body_impulse", "push_robot", "foot_friction_left", "foot_friction_right"):
    cfg.events.pop(event_name, None)
  cfg.events["foot_friction"] = EventTermCfg(
    mode="startup",
    func=dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOMS),
      "operation": "abs",
      "ranges": (0.5, 1.6),
      "shared_random": True,
    },
  )
  for curr_name in ("arm_torque_level", "body_impulse_level"):
    cfg.curriculum.pop(curr_name, None)

  if not play:
    _rescale_curriculum_steps(cfg, curriculum_steps_per_iter / ITER)
    if termination_probability < 1.0:
      for name in ("fell_over", "collapsed"):
        if name in cfg.terminations:
          term = cfg.terminations[name]
          cfg.terminations[name] = TerminationTermCfg(
            func=_probabilistic_termination,
            params={"base_func": term.func, "p_term": termination_probability, **term.params},
          )

  cfg.observations["actor"].history_length = history_length
  for term in cfg.observations["actor"].terms.values():
    term.delay_min_lag = 0
    term.delay_max_lag = 0
    term.delay_hold_prob = 0.0

  cfg.scene.sensors = (
    *cfg.scene.sensors,
    ContactSensorCfg(
      name="wm_feet",
      primary=ContactMatch(mode="body", pattern=r"^(left|right)_ankle_roll_link$", entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="netforce",
    ),
  )
  return cfg


def pack_task_aux(
  lin_accum: torch.Tensor,
  ang_accum: torch.Tensor,
  contact_prob: torch.Tensor,
  air_time: torch.Tensor,
  contact_time: torch.Tensor,
) -> torch.Tensor:
  """Pack Markovian reward/contact state into [B, 9]: [lin(2), ang(1), p_c(2), t_air(2), t_c(2)]."""
  return torch.cat((lin_accum, ang_accum.unsqueeze(-1), contact_prob, air_time, contact_time), dim=-1)


class VelocityPolicyStart(NamedTuple):
  state: torch.Tensor
  history: torch.Tensor
  context: torch.Tensor
  aux: torch.Tensor
  encoder_bias: torch.Tensor
  imu_bias: torch.Tensor
  arm_target: torch.Tensor
  arm_ramp_start: torch.Tensor
  arm_ramp_target: torch.Tensor
  arm_ramp_steps: torch.Tensor
  arm_ramp_active: torch.Tensor
  twist: torch.Tensor
  height_cmd: torch.Tensor
  height_target: torch.Tensor
  episode_step: torch.Tensor
  phase_offset: torch.Tensor
  remaining: torch.Tensor
  cmd_elapsed: torch.Tensor
  drift_anchor: torch.Tensor
  is_stationary: torch.Tensor


_REPLAY_SPECS: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
  "latest": ((OBS_DIM,), torch.float32),
  "aux": ((AUX_DIM,), torch.float32),
  "encoder_bias": ((ACTION_DIM,), torch.float32),
  "imu_bias": ((6,), torch.float32),
  "arm_target": ((ARM_DIM,), torch.float32),
  "arm_ramp_start": ((ARM_DIM,), torch.float32),
  "arm_ramp_target": ((ARM_DIM,), torch.float32),
  "arm_ramp_steps": ((), torch.long),
  "arm_ramp_active": ((), torch.bool),
  "twist": ((3,), torch.float32),
  "height_cmd": ((), torch.float32),
  "height_target": ((), torch.float32),
  "episode_step": ((), torch.long),
  "phase_offset": ((), torch.float32),
  "remaining": ((), torch.long),
  "cmd_elapsed": ((), torch.float32),
  "drift_anchor": ((2,), torch.float32),
  "is_stationary": ((), torch.bool),
}
_START_TAIL_FIELDS: tuple[str, ...] = tuple(k for k in _REPLAY_SPECS if k != "latest")


class VelocityReplay(RingReplay):
  """Ring replay storing reset-safe WM transitions and compact Markovian velocity-task states."""

  def __init__(self, capacity: int, num_envs: int, validation_envs: int, device: str | torch.device):
    super().__init__(capacity, num_envs, validation_envs, device)
    self._fields = {
      name: torch.empty(capacity, num_envs, *shape, dtype=dtype, device=self.device)
      for name, (shape, dtype) in _REPLAY_SPECS.items()
    }

  def append_control(self, batch, record: dict[str, torch.Tensor]) -> None:
    if batch.state.shape[0] != 1:
      raise ValueError("Velocity replay appends one control tick at a time")
    row = self.total_steps % self.capacity
    for key, buffer in self._fields.items():
      destination = buffer[row]
      source = record[key]
      if source.shape != destination.shape or source.dtype != destination.dtype or source.device != self.device:
        raise ValueError(
          f"Incorrect velocity replay field {key}: got {source.shape}/{source.dtype}, expected {destination.shape}/{destination.dtype}"
        )
      if source.is_floating_point() and not torch.isfinite(source).all():
        raise FloatingPointError(f"Nonfinite velocity replay field: {key}")
      destination.copy_(source)
    self.append(batch)

  def sample_starts(
    self,
    batch_size: int,
    history_length: int = DEFAULT_HISTORY,
    *,
    validation: bool = False,
  ) -> VelocityPolicyStart:
    pool_size = batch_size * 2 if not validation else batch_size
    window = self.sample(pool_size, history_length, validation=validation)
    if not validation and len(window.env) > batch_size:
      s = window.state[:, -1]
      upright_cos = 1.0 - 2.0 * (s[:, 4].square() + s[:, 5].square())
      upright = (s[:, 2] > 0.40) & (upright_cos > 0.35)
      good_idx = torch.nonzero(upright, as_tuple=False).squeeze(-1)
      if len(good_idx) >= batch_size:
        sel = good_idx[:batch_size]
      else:
        bad_idx = torch.nonzero(~upright, as_tuple=False).squeeze(-1)
        sel = torch.cat((good_idx, bad_idx[: batch_size - len(good_idx)]))
      rows = (window.sequence % self.capacity)[sel]
      env = window.env[sel]
      state_start = window.state[sel, -1]
      context_start = window.inputs[sel, -1, 97:]
    else:
      rows = window.sequence % self.capacity
      env = window.env
      state_start = window.state[:, -1]
      context_start = window.inputs[:, -1, 97:]
    last = rows[:, -1]
    return VelocityPolicyStart(
      state=state_start,
      history=self._fields["latest"][rows, env[:, None]],
      context=context_start,
      **{key: self._fields[key][last, env] for key in _START_TAIL_FIELDS},
    )


def advance_arm_ramp(
  arm_target: torch.Tensor,
  ramp_start: torch.Tensor,
  ramp_target: torch.Tensor,
  ramp_steps: torch.Tensor,
  ramp_active: torch.Tensor,
  dt: float = CONTROL_DT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Mirror hold_arm_pose.advance() at the end of a control tick."""
  next_steps = ramp_steps + ramp_active.long()
  fraction = ((next_steps.to(arm_target.dtype) * dt) / ARM_RAMP_DURATION_S).clamp(0.0, 1.0)
  interpolated = torch.lerp(ramp_start, ramp_target, fraction.unsqueeze(-1))
  next_target = torch.where(ramp_active.unsqueeze(-1), interpolated, arm_target)
  next_active = ramp_active & (fraction < 1.0)
  return next_target, next_steps, next_active


def advance_height_command(
  height_cmd: torch.Tensor,
  height_target: torch.Tensor,
  dt: float = CONTROL_DT,
  max_rate: float = HEIGHT_MAX_RATE,
) -> torch.Tensor:
  max_step = max_rate * dt
  return height_cmd + (height_target - height_cmd).clamp(-max_step, max_step)


class _KinematicFrame(NamedTuple):
  fk: KinematicState
  q_quat: torch.Tensor
  v_b: torch.Tensor
  omega_b: torch.Tensor
  g_pelvis: torch.Tensor
  g_torso: torch.Tensor
  foot_z: torch.Tensor
  min_foot_z: torch.Tensor
  height: torch.Tensor
  relative_z: torch.Tensor
  cmd_xy_speed: torch.Tensor
  cmd_strength: torch.Tensor
  is_moving: torch.Tensor
  is_still: torch.Tensor


class VelocityWmTask(nn.Module):
  """Fused differentiable lower-body velocity rewards, soft terminations, and critic features."""

  default_position: torch.Tensor
  soft_limits: torch.Tensor
  waist_kp: torch.Tensor
  waist_kd: torch.Tensor
  waist_effort_limit: torch.Tensor
  waist_l2_weights: torch.Tensor
  waist_l1_weights: torch.Tensor
  lateral_std_standing: torch.Tensor
  lateral_std_walking: torch.Tensor
  lateral_std_running: torch.Tensor
  knee_index_tensor: torch.Tensor
  hip_yaw_index_tensor: torch.Tensor
  lateral_index_tensor: torch.Tensor
  default_hip_yaw: torch.Tensor
  default_lateral: torch.Tensor
  gait_offsets: torch.Tensor

  def __init__(
    self,
    kinematics: TorchKinematics,
    default_position: torch.Tensor,
    soft_limits: torch.Tensor,
    waist_kp: torch.Tensor,
    waist_kd: torch.Tensor,
    waist_effort_limit: torch.Tensor,
    *,
    termination_probability: float = DEFAULT_TERMINATION_PROB,
  ):
    super().__init__()
    if not math.isfinite(termination_probability) or not 0.0 < termination_probability <= 1.0:
      raise ValueError("termination_probability must lie in (0, 1]")
    self.termination_probability = float(termination_probability)
    self.kinematics = kinematics
    self.torso_id = BODY_NAMES.index("torso_link")
    self.knee_ids = (JOINT_NAMES.index("left_knee_joint"), JOINT_NAMES.index("right_knee_joint"))
    self.waist_ids = tuple(JOINT_NAMES.index(f"waist_{axis}_joint") for axis in ("yaw", "roll", "pitch"))
    if self.waist_ids != (12, 13, 14):
      raise AssertionError(f"Expected contiguous waist slice 12:15, got {self.waist_ids}")
    self.hip_yaw_ids = (JOINT_NAMES.index("left_hip_yaw_joint"), JOINT_NAMES.index("right_hip_yaw_joint"))
    lateral_names = tuple(
      f"{side}_{part}_joint"
      for side in ("left", "right")
      for part in ("hip_roll", "hip_yaw", "ankle_roll")
    )
    self.lateral_ids = tuple(JOINT_NAMES.index(name) for name in lateral_names)

    def_pos = default_position.detach().clone()
    self.register_buffer("default_position", def_pos)
    self.register_buffer("soft_limits", soft_limits.detach().clone())
    self.register_buffer("waist_kp", waist_kp.detach().clone())
    self.register_buffer("waist_kd", waist_kd.detach().clone())
    self.register_buffer("waist_effort_limit", waist_effort_limit.detach().clone())
    self.register_buffer("waist_l2_weights", torch.tensor([0.4, 0.85, 1.0], dtype=torch.float32))
    self.register_buffer("waist_l1_weights", torch.tensor([1.0, 1.0, 0.5], dtype=torch.float32))
    self.register_buffer("lateral_std_standing", torch.full((6,), 0.1, dtype=torch.float32))
    self.register_buffer("lateral_std_walking", torch.tensor([0.35, 0.3, 0.2, 0.35, 0.3, 0.2], dtype=torch.float32))
    self.register_buffer("lateral_std_running", torch.tensor([0.5, 0.4, 0.25, 0.5, 0.4, 0.25], dtype=torch.float32))
    self.register_buffer("knee_index_tensor", torch.tensor(self.knee_ids, dtype=torch.long), persistent=False)
    self.register_buffer("hip_yaw_index_tensor", torch.tensor(self.hip_yaw_ids, dtype=torch.long), persistent=False)
    self.register_buffer("lateral_index_tensor", torch.tensor(self.lateral_ids, dtype=torch.long), persistent=False)
    self.register_buffer("default_hip_yaw", def_pos[list(self.hip_yaw_ids)].clone(), persistent=False)
    self.register_buffer("default_lateral", def_pos[list(self.lateral_ids)].clone(), persistent=False)
    self.register_buffer("gait_offsets", torch.tensor([[0.0, 0.5]], dtype=torch.float32), persistent=False)

    self.height_std: float = float(HEIGHT_STD_STAGES[0][1])
    self.cos_tilt_limit: float = math.cos(math.radians(70.0))
    self.back_deadband: float = math.sin(math.radians(3.0))
    self.fwd_deadband: float = math.sin(math.radians(6.0))

  def _kinematic_frame(self, next_state: torch.Tensor, twist: torch.Tensor) -> _KinematicFrame:
    fk = self.kinematics(next_state, validate=False)
    dtype = next_state.dtype
    q_quat = unit(next_state[:, 3:7])
    v_b = rotate(conjugate(q_quat), next_state[:, 36:39])
    omega_b = next_state[:, 39:42]
    gravity = torch.zeros_like(next_state[:, :3])
    gravity[:, 2] = -1.0
    g_pelvis = rotate(conjugate(q_quat), gravity)
    g_torso = rotate(conjugate(fk.body_quaternion[:, self.torso_id, :]), gravity)

    foot_z = fk.foot_position[:, :, 2]
    min_foot_z = foot_z.min(dim=-1).values
    height = next_state[:, 2] - min_foot_z
    relative_z = foot_z - min_foot_z.unsqueeze(-1)

    cmd_xy_speed = torch.linalg.vector_norm(twist[:, :2], dim=-1)
    cmd_strength = cmd_xy_speed + twist[:, 2].abs()
    is_moving = (cmd_strength > 0.1).to(dtype)
    return _KinematicFrame(
      fk=fk,
      q_quat=q_quat,
      v_b=v_b,
      omega_b=omega_b,
      g_pelvis=g_pelvis,
      g_torso=g_torso,
      foot_z=foot_z,
      min_foot_z=min_foot_z,
      height=height,
      relative_z=relative_z,
      cmd_xy_speed=cmd_xy_speed,
      cmd_strength=cmd_strength,
      is_moving=is_moving,
      is_still=1.0 - is_moving,
    )

  def continuous_reward_sum(
    self,
    prev_state: torch.Tensor,
    next_state: torch.Tensor,
    target29: torch.Tensor,
    action: torch.Tensor,
    prev_action: torch.Tensor,
    twist: torch.Tensor,
    height_cmd: torch.Tensor,
    lin_accum: torch.Tensor,
    ang_accum: torch.Tensor,
    cmd_elapsed: torch.Tensor,
    drift_anchor: torch.Tensor,
    is_stationary: torch.Tensor,
    *,
    height_std: float | None = None,
    exact_l1: bool = False,
    include_qacc: bool = True,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, _KinematicFrame]:
    """Compute unweighted continuous reward sum (dt not yet multiplied) + updated accumulators & FK."""
    frame = self._kinematic_frame(next_state, twist)
    dtype = next_state.dtype
    h_std = self.height_std if height_std is None else height_std

    q = next_state[:, 7:36]
    qd = next_state[:, 42:71]
    qd_prev = prev_state[:, 42:71]

    lin_err = (twist[:, :2] - frame.v_b[:, :2]).square().sum(-1) + 1.5 * frame.v_b[:, 2].square()
    lin_std = math.sqrt(0.2) * (1.0 + (frame.cmd_xy_speed - 0.8).clamp_min(0.0))
    r_lin = 2.0 * torch.exp(-lin_err / lin_std.square())

    next_lin_accum = lin_accum + frame.v_b[:, :2] * CONTROL_DT
    next_elapsed = cmd_elapsed + CONTROL_DT
    avg_lin = next_lin_accum / next_elapsed.clamp_min(CONTROL_DT).unsqueeze(-1)
    ramp = (next_elapsed / 1.5).clamp(max=1.0)
    r_lin_avg = 0.8 * torch.exp(-(twist[:, :2] - avg_lin).square().sum(-1) / 0.30**2) * ramp

    ang_err = (twist[:, 2] - frame.omega_b[:, 2]).square() + 0.05 * frame.omega_b[:, :2].square().sum(-1)
    r_ang = 0.5 * torch.exp(-ang_err / 0.7**2)

    next_ang_accum = ang_accum + frame.omega_b[:, 2] * CONTROL_DT
    avg_ang = next_ang_accum / next_elapsed.clamp_min(CONTROL_DT)
    r_ang_avg = 1.6 * torch.exp(-(twist[:, 2] - avg_ang).square() / 0.5**2) * frame.is_moving

    drift_sq = (next_state[:, :2] - drift_anchor).square().sum(-1)
    r_drift = -0.5 * (1.0 - torch.exp(-drift_sq / 0.2**2)) * is_stationary.to(dtype)

    dir_xy = twist[:, :2] / frame.cmd_xy_speed.clamp_min(1e-6).unsqueeze(-1)
    achieved = (frame.v_b[:, :2] * dir_xy).sum(-1)
    shortfall = (0.5 * frame.cmd_xy_speed - achieved).clamp_min(0.0) * (frame.cmd_xy_speed > 0.1).to(dtype)
    r_shortfall = -0.08 * shortfall

    r_height = 0.8 * torch.exp(-(height_cmd - frame.height).square() / (h_std * h_std))

    active_knee = (height_cmd >= 0.78).to(dtype) * frame.is_still
    r_knee = 0.5 * torch.exp(-q.index_select(-1, self.knee_index_tensor).square().mean(-1) / 0.5**2) * active_knee

    r_torso_ori = -1.0 * (frame.g_torso[:, 0].square() + 2.0 * frame.g_torso[:, 1].square())

    lean_cost = (-frame.g_pelvis[:, 0] - self.back_deadband).clamp_min(0.0) + 0.2 * (
      frame.g_pelvis[:, 0] - self.fwd_deadband
    ).clamp_min(0.0).square()
    r_lean = -2.0 * lean_cost

    waist_diff = q[:, 12:15] - self.default_position[12:15]
    r_waist_l2 = -1.0 * (waist_diff.square() * self.waist_l2_weights).sum(-1)

    waist_abs = waist_diff.abs() if exact_l1 else ((waist_diff.square() + 1e-6).sqrt() - 1e-3)
    r_waist_l1 = -0.3 * (waist_abs * self.waist_l1_weights).sum(-1) * frame.is_still

    waist_tau = (
      self.waist_kp * (target29[:, 12:15] - q[:, 12:15])
      - self.waist_kd * qd[:, 12:15]
    ).clamp(-self.waist_effort_limit, self.waist_effort_limit)
    r_waist_effort = -1.0 * (waist_tau / self.waist_effort_limit).square().sum(-1) * frame.is_still

    hip_yaw_diff = q.index_select(-1, self.hip_yaw_index_tensor) - self.default_hip_yaw
    r_hip_yaw = -5.0 * hip_yaw_diff.square().sum(-1) * frame.is_still

    standing_mask = (frame.cmd_strength < 0.1).to(dtype).unsqueeze(-1)
    walking_mask = ((frame.cmd_strength >= 0.1) & (frame.cmd_strength < 1.4)).to(dtype).unsqueeze(-1)
    running_mask = (frame.cmd_strength >= 1.4).to(dtype).unsqueeze(-1)
    pose_std = (
      self.lateral_std_standing * standing_mask
      + self.lateral_std_walking * walking_mask
      + self.lateral_std_running * running_mask
    )
    lat_diff = q.index_select(-1, self.lateral_index_tensor) - self.default_lateral
    r_pose = 0.5 * torch.exp(-(lat_diff.square() / pose_std.square()).mean(-1))

    torso_ang_xy = frame.fk.body_angular_velocity[:, self.torso_id, :2]
    r_body_ang = -0.05 * torso_ang_xy.square().sum(-1)

    qacc_lower = (qd[:, :ACTION_DIM] - qd_prev[:, :ACTION_DIM]) / CONTROL_DT
    r_qacc = (-2.5e-7 * qacc_lower.square().sum(-1)) if include_qacc else torch.zeros_like(r_lin)

    act_diff_sq = (action - prev_action).square().sum(-1)
    r_act_rate = -0.02 * act_diff_sq
    r_act_still = -0.08 * act_diff_sq * frame.is_still

    total_continuous = (
      r_lin
      + r_lin_avg
      + r_ang
      + r_ang_avg
      + r_drift
      + r_shortfall
      + r_height
      + r_knee
      + r_torso_ori
      + r_lean
      + r_waist_l2
      + r_waist_l1
      + r_waist_effort
      + r_hip_yaw
      + r_pose
      + r_body_ang
      + r_qacc
      + r_act_rate
      + r_act_still
    )
    return total_continuous, next_lin_accum, next_ang_accum, next_elapsed, frame

  def step(
    self,
    prev_state: torch.Tensor,
    next_state: torch.Tensor,
    foot_force: torch.Tensor,
    target29: torch.Tensor,
    action: torch.Tensor,
    prev_action: torch.Tensor,
    aux: torch.Tensor,
    twist: torch.Tensor,
    height_cmd: torch.Tensor,
    next_episode_step: torch.Tensor,
    phase_offset: torch.Tensor,
    cmd_elapsed: torch.Tensor,
    drift_anchor: torch.Tensor,
    is_stationary: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance one imagined 20ms step and return (step_reward, termination_prob, next_aux, foot_z)."""
    dtype = next_state.dtype
    continuous, next_lin_accum, next_ang_accum, _, frame = self.continuous_reward_sum(
      prev_state=prev_state,
      next_state=next_state,
      target29=target29,
      action=action,
      prev_action=prev_action,
      twist=twist,
      height_cmd=height_cmd,
      lin_accum=aux[:, 0:2],
      ang_accum=aux[:, 2],
      cmd_elapsed=cmd_elapsed,
      drift_anchor=drift_anchor,
      is_stationary=is_stationary,
    )

    fz = foot_force[:, :, 2].clamp_min(0.0)
    fz_total = fz.sum(dim=-1, keepdim=True) + 10.0
    # Load-share unloading + foot clearance prevents sigmoid underflow in double support (fz ~= 175N).
    unloaded_share = (1.0 - fz / fz_total).square() * torch.exp(-fz / 45.0)
    p_lifted = torch.sigmoid((frame.relative_z - 0.020) / 0.012)
    p_airborne = (unloaded_share * p_lifted).clamp(0.0, 1.0)
    p_contact = 1.0 - p_airborne
    p_slip_contact = torch.sigmoid((fz - 20.0) / 5.0)

    next_air_time = p_airborne * (aux[:, 5:7] + CONTROL_DT)
    next_contact_time = p_contact * (aux[:, 7:9] + CONTROL_DT)

    clock = (next_episode_step.to(dtype) * CONTROL_DT) / GAIT_PERIOD + phase_offset
    leg_phase = torch.remainder(clock.unsqueeze(-1) + self.gait_offsets.to(dtype), 1.0)
    in_stance_window = (leg_phase < 0.56).to(dtype)
    in_swing_window = 1.0 - in_stance_window

    stance_load_share = fz / fz_total
    r_gait = 1.5 * (in_stance_window * stance_load_share + in_swing_window * p_airborne).sum(dim=-1) * frame.is_moving

    single_stance = p_contact[:, 0] * p_airborne[:, 1] + p_airborne[:, 0] * p_contact[:, 1]
    in_mode_time = p_contact * next_contact_time + p_airborne * next_air_time
    mode_time = torch.minimum(in_mode_time[:, 0], in_mode_time[:, 1]) * single_stance
    r_air_time = 3.2 * (0.4 - (mode_time - 0.4).abs()).clamp_min(0.0) * frame.is_moving

    swing_u = ((leg_phase - 0.56) / 0.44).clamp(0.0, 1.0)
    swing_sin = torch.sin(math.pi * swing_u)
    swing_peak = (0.09 * ((height_cmd - 0.35) / 0.39).clamp(0.55, 1.0)).unsqueeze(-1)
    lift_shortfall = F.smooth_l1_loss(frame.relative_z, swing_peak * swing_sin, reduction="none", beta=0.015)
    r_swing_arch = (
      ((1.8 - 18.0 * lift_shortfall - 1.2 * stance_load_share * swing_sin) * in_swing_window).sum(dim=-1)
      * frame.is_moving
    )

    height_l1 = F.smooth_l1_loss(frame.height, height_cmd, reduction="none", beta=0.020)
    r_height_pull = (-25.0 - 15.0 * frame.is_still) * height_l1 + 1.5 * torch.exp(
      -(frame.height - height_cmd).square() / 0.035**2
    )

    vel_l1 = F.smooth_l1_loss(frame.v_b[:, :2], twist[:, :2], reduction="none", beta=0.12).sum(dim=-1)
    ang_l1 = F.smooth_l1_loss(frame.omega_b[:, 2], twist[:, 2], reduction="none", beta=0.15)
    r_vel_pull = (-3.2 * vel_l1 - 1.2 * ang_l1) * frame.is_moving

    cmd_3d = torch.cat((twist[:, :2], torch.zeros_like(twist[:, :1])), dim=-1)
    v_cmd_w_xy = rotate(frame.q_quat, cmd_3d)[:, :2]
    v_swing_target = (math.pi / (2.0 * 0.44)) * swing_sin.unsqueeze(-1) * v_cmd_w_xy.unsqueeze(1)
    swing_vel_err_sq = (frame.fk.foot_linear_velocity[:, :, :2] - v_swing_target).square().sum(dim=-1)
    r_swing_vel = (
      1.5
      * (in_swing_window * p_lifted * torch.exp(-swing_vel_err_sq / 0.6**2)).sum(dim=-1)
      * (frame.cmd_xy_speed > 0.1).to(dtype)
    )

    foot_vel_xy_sq = frame.fk.foot_linear_velocity[:, :, :2].square().sum(-1)
    foot_speed_xy = (foot_vel_xy_sq + 1e-6).sqrt() - 1e-3

    r_slip = -2.0 * (foot_vel_xy_sq * p_slip_contact).sum(-1) * frame.is_moving
    r_slip_still = -4.0 * (foot_speed_xy * p_slip_contact).sum(-1) * frame.is_still
    r_stand_feet = -0.5 * p_airborne.sum(-1) * frame.is_still

    onset_prob = p_contact * (1.0 - aux[:, 3:5])
    force_mag = (foot_force.square().sum(-1) + 1e-6).sqrt()
    r_landing = -2e-3 * (force_mag * onset_prob).sum(-1) * frame.is_moving

    q_lower = next_state[:, 7 : 7 + ACTION_DIM]
    lim_lower = self.soft_limits[:ACTION_DIM]
    limit_excess = (lim_lower[:, 0] - q_lower).clamp_min(0.0) + (q_lower - lim_lower[:, 1]).clamp_min(0.0)
    r_limits = -2.0 * limit_excess.sum(-1)

    termination = self._termination_from_pose(frame.g_pelvis, frame.height)
    r_terminated = -120.0 * termination

    reward = CONTROL_DT * (
      continuous
      + r_height_pull
      + r_vel_pull
      + r_gait
      + r_air_time
      + r_swing_arch
      + r_swing_vel
      + r_slip
      + r_slip_still
      + r_stand_feet
      + r_landing
      + r_limits
      + r_terminated
    )
    next_aux = pack_task_aux(next_lin_accum, next_ang_accum, p_contact, next_air_time, next_contact_time)
    return reward, termination, next_aux, frame.foot_z

  def _termination_from_pose(self, g_pelvis: torch.Tensor, height: torch.Tensor) -> torch.Tensor:
    tilt_margin = (self.cos_tilt_limit - (-g_pelvis[:, 2])) / 0.15
    height_margin = (0.35 - height) / 0.10
    smooth_margin = 0.1 * torch.logsumexp(10.0 * torch.stack((tilt_margin, height_margin), dim=-1), dim=-1)
    return self.termination_probability * torch.sigmoid(15.0 * smooth_margin)

  def termination(self, state: torch.Tensor, min_foot_z: torch.Tensor | None = None) -> torch.Tensor:
    if min_foot_z is None:
      min_foot_z = self.kinematics(state, validate=False).foot_position[:, :, 2].min(dim=-1).values
    q_quat = unit(state[:, 3:7])
    gravity = torch.zeros_like(state[:, :3])
    gravity[:, 2] = -1.0
    g_b = rotate(conjugate(q_quat), gravity)
    return self._termination_from_pose(g_b, state[:, 2] - min_foot_z)

  def critic_features(
    self,
    model: PositionWorldModel,
    state: torch.Tensor,
    context: torch.Tensor,
    prev_action: torch.Tensor,
    aux: torch.Tensor,
    arm_target: torch.Tensor,
    twist: torch.Tensor,
    height_cmd: torch.Tensor,
    episode_step: torch.Tensor,
    phase_offset: torch.Tensor,
    cmd_elapsed: torch.Tensor,
    drift_anchor: torch.Tensor,
    is_stationary: torch.Tensor,
    foot_z: torch.Tensor | None = None,
  ) -> torch.Tensor:
    if foot_z is None:
      foot_z = self.kinematics(state, validate=False).foot_position[:, :, 2]
    dtype = state.dtype
    feature = (state_features(state, validate=False) - model.input_mean[:68]) / model.input_scale[:68]
    physical = (context - model.input_mean[97:]) / model.input_scale[97:]

    min_foot_z = foot_z.min(dim=-1).values
    height = state[:, 2] - min_foot_z
    rel_foot_z = foot_z - min_foot_z.unsqueeze(-1)

    clock = (episode_step.to(dtype) * CONTROL_DT) / GAIT_PERIOD + phase_offset
    angle = torch.remainder(clock, 1.0) * (2.0 * torch.pi)
    clock_sc = torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)

    cmd_strength = torch.linalg.vector_norm(twist[:, :2], dim=-1) + twist[:, 2].abs()
    is_moving = (cmd_strength > 0.1).to(dtype)

    elapsed = cmd_elapsed.clamp_min(CONTROL_DT)
    avg_lin_err = aux[:, 0:2] / elapsed.unsqueeze(-1) - twist[:, :2]
    ramp = (cmd_elapsed / 1.5).clamp(0.0, 1.0).unsqueeze(-1)
    avg_ang_err = ((aux[:, 2] / elapsed - twist[:, 2]) * is_moving).unsqueeze(-1)
    drift_xy = (state[:, :2] - drift_anchor) * is_stationary.to(dtype).unsqueeze(-1)
    arm_err = arm_target - state[:, 7 + ACTION_DIM : 36]

    out = torch.cat(
      (
        feature,
        physical,
        twist,
        height_cmd.unsqueeze(-1),
        (height - height_cmd).unsqueeze(-1),
        clock_sc,
        is_moving.unsqueeze(-1),
        avg_lin_err,
        ramp,
        avg_ang_err,
        drift_xy,
        aux[:, 3:9],
        rel_foot_z,
        prev_action,
        arm_err,
      ),
      dim=-1,
    )
    if out.shape[-1] != CRITIC_DIM:
      raise AssertionError(f"Expected {CRITIC_DIM} critic features, got {out.shape[-1]}")
    return out

