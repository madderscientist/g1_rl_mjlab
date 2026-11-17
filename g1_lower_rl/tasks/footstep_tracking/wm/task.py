"""WM-v3 aligned footstep environment, auto-curriculum stages, replay, and differentiable task."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, NamedTuple

import torch
import torch.nn.functional as F
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from torch import nn

from g1_lower_rl.footstep_phase import FootstepPhaseCfg
from g1_lower_rl.footsteps.tensor_math import to_local, wrap
from g1_lower_rl.tasks.footstep_tracking.commands import FootstepCommandCfg, TorsoHeadingCommandCfg
from g1_lower_rl.tasks.footstep_tracking.reward_math import joint_edge_cost, phase_windows
from g1_lower_rl.tasks.footstep_tracking.walking import walk_first_env_cfg
from g1_lower_rl.tasks.lower_body.cfg.constants import FOOT_GEOMS
from g1_lower_rl.tasks.lower_body.wm.task import _probabilistic_termination
from g1_lower_rl.world_model.adapters.kinematics import KinematicState, TorchKinematics
from g1_lower_rl.world_model.adapters.math import conjugate, rotate, unit
from g1_lower_rl.world_model.model import BODY_NAMES, CONTROL_DT, JOINT_DIM, PositionWorldModel, state_features
from g1_lower_rl.world_model.training.replay import RingReplay

from .policy import ACTION_DIM, ARM_DIM, DEFAULT_HISTORY, FRAME_DIM, command_from_world_queue

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnvCfg

CRITIC_DIM = 191
AUX_DIM = 9  # [contact_prob(2), air_time(2), contact_time(2), anchor(3)]
ARM_RAMP_DURATION_S = 5.0
DEFAULT_TERMINATION_PROB = 0.05
SWING_FRACTION = 0.40  # 1.0 - stance_fraction (0.60) for standard FootstepPhaseCfg


@dataclass(frozen=True)
class CurriculumStage:
  index: int
  name: str
  profile: str = "precision"
  direction_half_range: float = math.pi
  distance_range: tuple[float, float] = (0.12, 0.48)
  width_range: tuple[float, float] = (0.12, 0.36)
  direction_change_half_range: float = math.pi
  foot_yaw_half_range: float = math.radians(30.0)
  frequency_range: tuple[float, float] = (0.8, 1.8)
  initial_frequency: float | None = None
  frequency_rate_half_range: float = 0.30
  initial_standing_prob: float = 0.22
  stop_prob: float = 0.16
  arm_target_scale: float = 1.0
  theta_active: bool = False
  theta_limit_rad: float = math.radians(20.0)
  swing_pos_w: float = 5.0
  swing_yaw_w: float = 5.0
  swing_pos_std: float = 0.20
  swing_yaw_std: float = 0.10
  swing_power: float = 2.0
  landing_w: float = -0.10
  action_noise: float = 0.16
  policy_target_kl: float = 0.04
  max_failure_rate: float = 0.006
  min_det_mtbf_s: float = 60.0
  min_det_unbroken_s: float = 20.0
  min_ss_match: float = 0.70
  min_swing_lift_m: float = 0.035
  max_stance_xy_err_m: float = 0.096
  max_stance_yaw_err_rad: float = 0.12
  max_theta_err_rad: float = 1.0
  min_stage_iterations: int = 90


AUTO_CURRICULUM_STAGES: tuple[CurriculumStage, ...] = (
  CurriculumStage(
    0, "walk_straight_init", profile="walk-first",
    direction_half_range=0.0, distance_range=(0.27, 0.27), width_range=(0.24, 0.24),
    direction_change_half_range=0.0, foot_yaw_half_range=0.0,
    frequency_range=(1.2, 1.2), initial_frequency=1.2, frequency_rate_half_range=0.0,
    initial_standing_prob=0.18, stop_prob=0.12, arm_target_scale=0.2,
    swing_pos_w=2.0, swing_yaw_w=1.5, swing_pos_std=0.30, swing_yaw_std=0.25, swing_power=1.0, landing_w=-0.03,
    action_noise=0.24, policy_target_kl=0.07, max_failure_rate=0.008,
    min_det_mtbf_s=45.0, min_det_unbroken_s=18.0, min_ss_match=0.58, min_swing_lift_m=0.038,
    max_stance_xy_err_m=0.125, max_stance_yaw_err_rad=0.18, min_stage_iterations=100,
  ),
  CurriculumStage(
    1, "walk_omni_60deg", profile="walk-first",
    direction_half_range=math.pi / 3.0, distance_range=(0.24, 0.30), width_range=(0.20, 0.28),
    direction_change_half_range=math.pi / 12.0, foot_yaw_half_range=0.0,
    frequency_range=(1.2, 1.2), initial_frequency=1.2, frequency_rate_half_range=0.0,
    initial_standing_prob=0.18, stop_prob=0.12, arm_target_scale=0.35,
    swing_pos_w=2.5, swing_yaw_w=2.0, swing_pos_std=0.28, swing_yaw_std=0.22, swing_power=1.0, landing_w=-0.04,
    action_noise=0.24, policy_target_kl=0.07, max_failure_rate=0.008,
    min_det_mtbf_s=45.0, min_det_unbroken_s=18.0, min_ss_match=0.62, min_swing_lift_m=0.038,
    max_stance_xy_err_m=0.115, max_stance_yaw_err_rad=0.16, min_stage_iterations=80,
  ),
  CurriculumStage(
    2, "walk_omni_180deg", profile="walk-first",
    distance_range=(0.18, 0.38), width_range=(0.15, 0.33),
    direction_change_half_range=math.pi / 3.0, foot_yaw_half_range=math.radians(12.0),
    frequency_range=(1.2, 1.2), initial_frequency=1.2, frequency_rate_half_range=0.0,
    initial_standing_prob=0.20, stop_prob=0.15, arm_target_scale=0.60,
    swing_pos_w=3.2, swing_yaw_w=2.8, swing_pos_std=0.24, swing_yaw_std=0.18, swing_power=1.5, landing_w=-0.06,
    action_noise=0.22, policy_target_kl=0.06, max_failure_rate=0.008,
    min_det_mtbf_s=50.0, min_det_unbroken_s=18.0, min_ss_match=0.65, min_swing_lift_m=0.038,
    max_stance_xy_err_m=0.110, max_stance_yaw_err_rad=0.15, min_stage_iterations=80,
  ),
  CurriculumStage(
    3, "walk_full_spatial", profile="walk-first",
    frequency_range=(1.2, 1.2), initial_frequency=1.2, frequency_rate_half_range=0.0,
    initial_standing_prob=0.20, stop_prob=0.15,
    swing_pos_w=4.0, swing_yaw_w=3.8, swing_pos_std=0.22, swing_yaw_std=0.14, swing_power=1.5, landing_w=-0.08,
    action_noise=0.20, policy_target_kl=0.05, max_failure_rate=0.008,
    min_det_mtbf_s=55.0, min_det_unbroken_s=19.0, min_ss_match=0.68, min_swing_lift_m=0.038,
    max_stance_xy_err_m=0.105, max_stance_yaw_err_rad=0.15,
  ),
  CurriculumStage(
    4, "precision_var_cadence",
    min_swing_lift_m=0.036, max_stance_xy_err_m=0.094, min_stage_iterations=100,
  ),
  CurriculumStage(
    5, "theta_30deg",
    theta_active=True, theta_limit_rad=math.radians(30.0), landing_w=-0.25,
    action_noise=0.15, policy_target_kl=0.035, min_det_mtbf_s=65.0, max_stance_xy_err_m=0.095, max_theta_err_rad=0.15,
  ),
  CurriculumStage(
    6, "theta_60deg",
    theta_active=True, theta_limit_rad=math.radians(60.0), landing_w=-0.35,
    action_noise=0.14, policy_target_kl=0.03, min_det_mtbf_s=70.0, max_theta_err_rad=0.18,
  ),
  CurriculumStage(
    7, "theta_90deg_full",
    initial_frequency=1.2, initial_standing_prob=0.50, stop_prob=0.30,
    theta_active=True, theta_limit_rad=math.radians(90.0), landing_w=-0.50,
    action_noise=0.13, policy_target_kl=0.025, max_failure_rate=0.005,
    min_det_mtbf_s=75.0, min_det_unbroken_s=21.0, min_ss_match=0.72,
    max_stance_yaw_err_rad=0.11, max_theta_err_rad=0.21, min_stage_iterations=120,
  ),
)
POLISH_EXPLORATION: tuple[float, float] = (0.08, 0.015)


def stage_qualifies(stage: CurriculumStage, val_metrics: dict[str, float]) -> bool:
  """Check whether held-out validation metrics (including deterministic Viser survival time) satisfy all gates."""
  if val_metrics["native/validation_failure_rate"] > stage.max_failure_rate:
    return False
  if val_metrics.get("native/det_mtbf_s", 0.0) < stage.min_det_mtbf_s:
    return False
  if val_metrics.get("native/det_unbroken_mean_s", 0.0) < stage.min_det_unbroken_s:
    return False
  if val_metrics["native/validation_ss_match"] < stage.min_ss_match:
    return False
  if val_metrics["native/validation_swing_lift_m"] < stage.min_swing_lift_m:
    return False
  if val_metrics["native/validation_stance_xy_err_m"] > stage.max_stance_xy_err_m:
    return False
  if val_metrics["native/validation_stance_yaw_err_rad"] > stage.max_stance_yaw_err_rad:
    return False
  return not (stage.theta_active and val_metrics["native/validation_theta_err_rad"] > stage.max_theta_err_rad)


def _apply_stage_to_command_cfg(
  foot_cmd: FootstepCommandCfg,
  stage: CurriculumStage,
  *,
  set_standing_probs: bool,
) -> None:
  foot_cmd.manager = replace(
    foot_cmd.manager,
    frequency_range=stage.frequency_range,
    sampler=replace(
      foot_cmd.manager.sampler,
      distance_range=stage.distance_range,
      min_width=stage.width_range[0],
      max_width=stage.width_range[1],
      yaw_noise=(-stage.foot_yaw_half_range, stage.foot_yaw_half_range),
    ),
  )
  src_updates: dict[str, object] = {
    "direction_range": (-stage.direction_half_range, stage.direction_half_range),
    "direction_change_range": (-stage.direction_change_half_range, stage.direction_change_half_range),
    "frequency_range": stage.frequency_range,
    "frequency_rate_range": (-stage.frequency_rate_half_range, stage.frequency_rate_half_range),
  }
  if set_standing_probs:
    src_updates.update(
      initial_frequency=stage.initial_frequency,
      initial_standing_probability=stage.initial_standing_prob,
      stop_probability=stage.stop_prob,
    )
  foot_cmd.source = replace(foot_cmd.source, **src_updates)


def footstep_env_cfg_wm(
  termination_probability: float = DEFAULT_TERMINATION_PROB,
  *,
  play: bool = False,
  profile: str = "precision",
  theta_limit_rad: float = math.radians(90.0),
) -> ManagerBasedRlEnvCfg:
  """Construct footstep environment aligned with Position WM v3 20ms (4 x 5ms) physical contract."""
  if not math.isfinite(termination_probability) or not 0.0 < termination_probability <= 1.0:
    raise ValueError("termination_probability must lie in (0, 1]")
  if profile == "step-episode":
    from g1_lower_rl.tasks.footstep_tracking.step_episode import step_episode_env_cfg

    cfg = step_episode_env_cfg(play=play)
  elif play:
    from g1_lower_rl.tasks.footstep_tracking.walking import precision_env_cfg

    cfg = precision_env_cfg(play=True)
    full_stage = AUTO_CURRICULUM_STAGES[-1]
    foot_cmd = cfg.commands["footsteps"]
    assert isinstance(foot_cmd, FootstepCommandCfg)
    _apply_stage_to_command_cfg(foot_cmd, full_stage, set_standing_probs=False)
    for ev_name in ("reset_arm_pose", "arm_pose_drift"):
      if ev_name in cfg.events:
        cfg.events[ev_name].params["target_scale"] = full_stage.arm_target_scale
  else:
    cfg = walk_first_env_cfg(play=False)

  cfg.sim.mujoco.timestep = 0.005
  cfg.decimation = 4
  if play:
    cfg.sim.nconmax = 256
    cfg.sim.njmax = 1024
  else:
    cfg.sim.njmax = 400
  cfg.curriculum = {}

  # Always attach TorsoHeadingCommand so theta is ready in-place without recreating the env.
  cfg.commands["torso_heading"] = TorsoHeadingCommandCfg(
    angle_limit=theta_limit_rad if play else AUTO_CURRICULUM_STAGES[0].theta_limit_rad
  )
  if not play and profile != "step-episode":
    foot_cmd = cfg.commands["footsteps"]
    assert isinstance(foot_cmd, FootstepCommandCfg)
    _apply_stage_to_command_cfg(foot_cmd, AUTO_CURRICULUM_STAGES[0], set_standing_probs=True)
    cfg.episode_length_s = 26.0

  # Remove unmodeled external wrenches/teleports and mid-substep CPU-sync NumericalSafety.
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
  if not play and "reset_initialization" in cfg.events:
    cfg.events["reset_initialization"].params.update(
      probabilities=(0.75, 0.25, 0.0),
      pelvis_height_range=(0.62, 0.78),
    )

  cfg.terminations.pop("numerical_safety", None)
  if play:
    from g1_lower_rl.tasks.lower_body.cfg.terminations import make_terminations as make_hard_terminations

    hard_terms = make_hard_terminations()
    for name in ("fell_over", "collapsed"):
      if name in cfg.terminations and name in hard_terms:
        cfg.terminations[name] = hard_terms[name]
  else:
    for name in ("fell_over", "collapsed"):
      if name in cfg.terminations:
        base_term = cfg.terminations[name]
        base_fn = base_term.params.get("base_func", base_term.func)
        base_params = {k: v for k, v in base_term.params.items() if k not in ("base_func", "p_term")}
        cfg.terminations[name] = TerminationTermCfg(
          func=_probabilistic_termination,
          params={"base_func": base_fn, "p_term": termination_probability, **base_params},
        )

  for term in cfg.observations["actor"].terms.values():
    term.delay_min_lag = 0
    term.delay_max_lag = 0
    term.delay_hold_prob = 0.0

  if not any(sensor.name == "wm_feet" for sensor in cfg.scene.sensors):
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


def pack_footstep_aux(
  contact_prob: torch.Tensor,
  air_time: torch.Tensor,
  contact_time: torch.Tensor,
  anchor: torch.Tensor,
) -> torch.Tensor:
  """Pack differentiable Markovian contact & anchor state into [B, 9]."""
  return torch.cat((contact_prob, air_time, contact_time, anchor), dim=-1)


class FootstepPolicyStart(NamedTuple):
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
  phase: torch.Tensor
  frequency: torch.Tensor
  mode: torch.Tensor
  elapsed: torch.Tensor
  req_freq: torch.Tensor
  stop_init_freq: torch.Tensor
  stop_ramp_start: torch.Tensor
  stop_phase: torch.Tensor
  start_dur: torch.Tensor
  stop_dur: torch.Tensor
  terminal_feet: torch.Tensor
  targets: torch.Tensor
  supports: torch.Tensor
  queue: torch.Tensor
  first_side: torch.Tensor
  pending_anchor: torch.Tensor
  theta: torch.Tensor
  theta_target: torch.Tensor
  theta_speed: torch.Tensor
  remaining: torch.Tensor


_REPLAY_SPECS: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
  "latest": ((FRAME_DIM,), torch.float32),
  "aux": ((AUX_DIM,), torch.float32),
  "encoder_bias": ((JOINT_DIM,), torch.float32),
  "imu_bias": ((12,), torch.float32),
  "arm_target": ((ARM_DIM,), torch.float32),
  "arm_ramp_start": ((ARM_DIM,), torch.float32),
  "arm_ramp_target": ((ARM_DIM,), torch.float32),
  "arm_ramp_steps": ((), torch.long),
  "arm_ramp_active": ((), torch.bool),
  "phase": ((), torch.float32),
  "frequency": ((), torch.float32),
  "mode": ((), torch.long),
  "elapsed": ((), torch.float32),
  "req_freq": ((), torch.float32),
  "stop_init_freq": ((), torch.float32),
  "stop_ramp_start": ((), torch.float32),
  "stop_phase": ((), torch.float32),
  "start_dur": ((), torch.float32),
  "stop_dur": ((), torch.float32),
  "terminal_feet": ((2, 3), torch.float32),
  "targets": ((2, 3), torch.float32),
  "supports": ((2, 3), torch.float32),
  "queue": ((4, 3), torch.float32),
  "first_side": ((), torch.long),
  "pending_anchor": ((), torch.long),
  "theta": ((), torch.float32),
  "theta_target": ((), torch.float32),
  "theta_speed": ((), torch.float32),
  "remaining": ((), torch.long),
}
_START_TAIL_FIELDS: tuple[str, ...] = tuple(k for k in _REPLAY_SPECS if k != "latest")


class FootstepReplay(RingReplay):
  """Ring replay with upright start filtering and Markovian footstep queue/anchor snapshots."""

  def __init__(self, capacity: int, num_envs: int, validation_envs: int, device: str | torch.device):
    super().__init__(capacity, num_envs, validation_envs, device)
    self._fields = {
      name: torch.empty(capacity, num_envs, *shape, dtype=dtype, device=self.device)
      for name, (shape, dtype) in _REPLAY_SPECS.items()
    }

  def append_control(self, batch, record: dict[str, torch.Tensor]) -> None:
    if batch.state.shape[0] != 1:
      raise ValueError("Footstep replay appends one control tick at a time")
    row = self.total_steps % self.capacity
    for key, buffer in self._fields.items():
      destination = buffer[row]
      source = record[key]
      if source.shape != destination.shape or source.dtype != destination.dtype or source.device != self.device:
        raise ValueError(
          f"Incorrect footstep replay field {key}: got {source.shape}/{source.dtype}, expected {destination.shape}/{destination.dtype}"
        )
      if source.is_floating_point() and not torch.isfinite(source).all():
        raise FloatingPointError(f"Nonfinite footstep replay field: {key}")
      destination.copy_(source)
    self.append(batch)

  def sample_starts(
    self,
    batch_size: int,
    history_length: int = DEFAULT_HISTORY,
    *,
    validation: bool = False,
    prefer_stepping: bool = False,
  ) -> FootstepPolicyStart:
    mult = 8 if (not validation and prefer_stepping) else (2 if not validation else 1)
    pool_size = batch_size * mult
    window = self.sample(pool_size, history_length, validation=validation)
    if not validation and len(window.env) > batch_size:
      s = window.state[:, -1]
      upright_cos = 1.0 - 2.0 * (s[:, 4].square() + s[:, 5].square())
      upright = (s[:, 2] > 0.42) & (upright_cos > 0.35)
      cand_rows = (window.sequence % self.capacity)[:, -1]
      cand_env = window.env
      if prefer_stepping:
        cand_mode = self._fields["mode"][cand_rows, cand_env]
        cand_freq = self._fields["frequency"][cand_rows, cand_env]
        launch_step = upright & (cand_mode == 1)
        mid_step = upright & ~launch_step & ((cand_freq > 1e-5) | (cand_mode == 0) | (cand_mode == 2))
        stand_ok = upright & ~(launch_step | mid_step)
        launch_idx = torch.nonzero(launch_step, as_tuple=False).squeeze(-1)
        mid_idx = torch.nonzero(mid_step, as_tuple=False).squeeze(-1)
        stand_idx = torch.nonzero(stand_ok, as_tuple=False).squeeze(-1)
        n_launch = min(len(launch_idx), (batch_size * 11) // 20)
        chosen = launch_idx[:n_launch]
        n_mid = min(len(mid_idx), batch_size // 4)
        chosen = torch.cat((chosen, mid_idx[:n_mid]))
        rem = batch_size - len(chosen)
        if rem > 0:
          chosen = torch.cat((chosen, stand_idx[:rem]))
        rem = batch_size - len(chosen)
        if rem > 0:
          rest_good = torch.cat((launch_idx[n_launch:], mid_idx[n_mid:]))
          chosen = torch.cat((chosen, rest_good[:rem]))
        rem = batch_size - len(chosen)
        if rem > 0:
          bad_idx = torch.nonzero(~upright, as_tuple=False).squeeze(-1)
          chosen = torch.cat((chosen, bad_idx[:rem]))
        sel = chosen
      else:
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

    # Build history padding across episode boundaries (first frame of a new episode repeats its own proprio).
    raw_hist = self._fields["latest"][rows, env[:, None]]
    seg = self.segment[rows, env[:, None]]
    same_ep = (seg == seg[:, -1:]).unsqueeze(-1)
    # Walk backwards from newest frame so cross-reset prefix frames repeat the episode's initial frame.
    clean_hist = raw_hist.clone()
    for h_idx in range(history_length - 2, -1, -1):
      clean_hist[:, h_idx] = torch.where(same_ep[:, h_idx], clean_hist[:, h_idx], clean_hist[:, h_idx + 1])

    return FootstepPolicyStart(
      state=state_start,
      history=clean_hist,
      context=context_start,
      **{key: self._fields[key][last, env] for key in _START_TAIL_FIELDS},
    )


def advance_theta_command(
  theta: torch.Tensor,
  theta_target: torch.Tensor,
  theta_speed: torch.Tensor,
  theta_active: float,
  dt: float = CONTROL_DT,
) -> torch.Tensor:
  if theta_active <= 0.0:
    return torch.zeros_like(theta)
  diff = theta_target - theta
  inc = torch.minimum(diff.abs(), theta_speed * dt)
  return theta + diff.sign() * inc


def _quat_yaw(q: torch.Tensor) -> torch.Tensor:
  w, x, y, z = q.unbind(-1)
  return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))


class _KinematicFrame(NamedTuple):
  fk: KinematicState
  pelvis_q: torch.Tensor
  torso_q: torch.Tensor
  v_w: torch.Tensor
  v_b: torch.Tensor
  omega_b: torch.Tensor
  g_pelvis: torch.Tensor
  g_torso: torch.Tensor
  foot_pos: torch.Tensor
  foot_vel: torch.Tensor
  foot_quat: torch.Tensor
  foot_yaw: torch.Tensor
  feet_pose: torch.Tensor
  foot_z: torch.Tensor
  min_foot_z: torch.Tensor
  height: torch.Tensor
  head_z: torch.Tensor
  relative_z: torch.Tensor


class FootstepWmTask(nn.Module):
  """Fused differentiable footstep rewards aligned with PPO, differentiable re-anchoring, and SE(2)-invariant Critic."""

  default_position: torch.Tensor
  hard_limits: torch.Tensor
  lower_kp: torch.Tensor
  lower_kd: torch.Tensor
  lower_effort_limit: torch.Tensor
  action_scale: torch.Tensor
  head_local_offset: torch.Tensor
  centers: torch.Tensor
  liftoff: torch.Tensor
  contact_start: torch.Tensor
  sides: torch.Tensor
  stage_params: torch.Tensor

  def __init__(
    self,
    kinematics: TorchKinematics,
    default_position: torch.Tensor,
    soft_limits: torch.Tensor | None = None,
    lower_kp: torch.Tensor | None = None,
    lower_kd: torch.Tensor | None = None,
    lower_effort_limit: torch.Tensor | None = None,
    action_scale: torch.Tensor | None = None,
    *,
    hard_limits: torch.Tensor | None = None,
    is_step_episode: bool = False,
    phase_cfg: FootstepPhaseCfg | None = None,
    termination_probability: float = DEFAULT_TERMINATION_PROB,
  ):
    super().__init__()
    if not math.isfinite(termination_probability) or not 0.0 < termination_probability <= 1.0:
      raise ValueError("termination_probability must lie in (0, 1]")
    resolved_hard = hard_limits if hard_limits is not None else soft_limits
    if resolved_hard is None or lower_kp is None or lower_kd is None or lower_effort_limit is None or action_scale is None:
      raise ValueError("FootstepWmTask requires joint limits, lower PD gains/effort, and action_scale")
    self.termination_probability = float(termination_probability)
    self.is_step_episode = bool(is_step_episode)
    self.kinematics = kinematics
    self.phase_cfg = phase_cfg or FootstepPhaseCfg()
    self.torso_id = BODY_NAMES.index("torso_link")
    self.cos_tilt_limit = math.cos(math.radians(70.0))

    self.register_buffer("default_position", default_position.detach().clone())
    self.register_buffer("hard_limits", resolved_hard.detach().clone())
    self.register_buffer("lower_kp", lower_kp.detach().clone())
    self.register_buffer("lower_kd", lower_kd.detach().clone())
    self.register_buffer("lower_effort_limit", lower_effort_limit.detach().clone())
    self.register_buffer("action_scale", action_scale.detach().clone())
    self.register_buffer("head_local_offset", torch.tensor([0.0, 0.0, 0.43], dtype=torch.float32), persistent=False)

    self.register_buffer(
      "centers",
      torch.tensor([self.phase_cfg.left_stance_phase, self.phase_cfg.right_stance_phase], dtype=torch.float32),
      persistent=False,
    )
    self.register_buffer("liftoff", torch.tensor(self.phase_cfg.liftoff_rad, dtype=torch.float32), persistent=False)
    self.register_buffer(
      "contact_start",
      torch.tensor(self.phase_cfg.contact_start_rad, dtype=torch.float32).remainder(2.0 * math.pi),
      persistent=False,
    )
    self.register_buffer("sides", torch.arange(2, dtype=torch.long), persistent=False)

    s0 = AUTO_CURRICULUM_STAGES[0]
    self.register_buffer(
      "stage_params",
      torch.tensor(
        [
          s0.swing_pos_w,
          s0.swing_yaw_w,
          s0.swing_pos_std,
          s0.swing_yaw_std,
          s0.swing_power,
          s0.landing_w,
          1.0 if s0.theta_active else 0.0,
        ],
        dtype=torch.float32,
      ),
    )

  def apply_stage(self, stage: CurriculumStage) -> None:
    """Update reward shaping weights in-place so captured CUDA graphs stay valid across stages."""
    updated = torch.tensor(
      [
        stage.swing_pos_w,
        stage.swing_yaw_w,
        stage.swing_pos_std,
        stage.swing_yaw_std,
        stage.swing_power,
        stage.landing_w,
        1.0 if stage.theta_active else 0.0,
      ],
      device=self.stage_params.device,
      dtype=self.stage_params.dtype,
    )
    self.stage_params.copy_(updated)

  def _kinematic_frame(self, next_state: torch.Tensor) -> _KinematicFrame:
    fk = self.kinematics(next_state, validate=False)
    pelvis_q = unit(next_state[:, 3:7])
    torso_q = unit(fk.body_quaternion[:, self.torso_id, :])
    v_w = next_state[:, 36:39]
    v_b = rotate(conjugate(pelvis_q), v_w)
    omega_b = next_state[:, 39:42]
    gravity = torch.zeros_like(v_w)
    gravity[:, 2] = -1.0
    g_pelvis = rotate(conjugate(pelvis_q), gravity)
    g_torso = rotate(conjugate(torso_q), gravity)

    foot_pos = fk.foot_position
    foot_vel = fk.foot_linear_velocity
    foot_quat = unit(fk.foot_quaternion)
    foot_yaw = _quat_yaw(foot_quat)
    feet_pose = torch.cat((foot_pos[:, :, :2], foot_yaw.unsqueeze(-1)), dim=-1)
    foot_z = foot_pos[:, :, 2]
    min_foot_z = foot_z.min(dim=-1).values
    height = next_state[:, 2] - min_foot_z
    torso_pos = fk.body_position[:, self.torso_id, :]
    head_pos = torso_pos + rotate(torso_q, self.head_local_offset.to(torso_pos).expand_as(torso_pos))
    head_z = head_pos[:, 2] - min_foot_z
    relative_z = foot_z - min_foot_z.unsqueeze(-1)
    return _KinematicFrame(
      fk=fk,
      pelvis_q=pelvis_q,
      torso_q=torso_q,
      v_w=v_w,
      v_b=v_b,
      omega_b=omega_b,
      g_pelvis=g_pelvis,
      g_torso=g_torso,
      foot_pos=foot_pos,
      foot_vel=foot_vel,
      foot_quat=foot_quat,
      foot_yaw=foot_yaw,
      feet_pose=feet_pose,
      foot_z=foot_z,
      min_foot_z=min_foot_z,
      height=height,
      head_z=head_z,
      relative_z=relative_z,
    )

  @torch.no_grad()
  def advance_queue_schedule(
    self,
    phase: torch.Tensor,
    frequency: torch.Tensor,
    mode: torch.Tensor,
    elapsed: torch.Tensor,
    req_freq: torch.Tensor,
    stop_init_freq: torch.Tensor,
    stop_ramp_start: torch.Tensor,
    stop_phase: torch.Tensor,
    start_dur: torch.Tensor,
    stop_dur: torch.Tensor,
    terminal_feet: torch.Tensor,
    targets: torch.Tensor,
    supports: torch.Tensor,
    queue: torch.Tensor,
    first_side: torch.Tensor,
    pending_anchor: torch.Tensor,
  ) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
  ]:
    """Advance phase, dynamic start/stop/single-step frequency ramps, and queue/targets across one 20ms tick."""
    # WALKING (0), STARTING (1), STOPPING (2), STANDING (3), SETTLING (5)
    next_elapsed = elapsed + CONTROL_DT
    arrived = (mode == 2) & (next_elapsed >= stop_ramp_start + stop_dur - 1e-6)
    raw_end = phase + 2.0 * math.pi * frequency * CONTROL_DT
    end_phase = torch.where(arrived, stop_phase, raw_end)
    moving = frequency > 1e-5

    next_touch = self.centers[None, :] + ((phase[:, None] - self.centers[None, :]).div(2.0 * math.pi).floor() + 1.0) * (2.0 * math.pi)
    touched = (next_touch <= end_phase[:, None]) & moving[:, None]
    touchdown = touched.any(-1)
    side = touched.to(torch.long).argmax(-1)

    next_supports = torch.where(touched[:, :, None], queue[:, 0, None], supports)
    next_pending = torch.where(touchdown, side, pending_anchor)
    cycle_step = queue[:, 3] - queue[:, 1]
    extrapolated = torch.stack(
      (
        queue[:, 2, 0] + cycle_step[:, 0],
        queue[:, 2, 1] + cycle_step[:, 1],
        wrap(queue[:, 2, 2] + wrap(queue[:, 3, 2] - queue[:, 1, 2])),
      ),
      dim=-1,
    )
    shifted_queue = torch.cat((queue[:, 1:], extrapolated.unsqueeze(1)), dim=1)
    next_queue = torch.where(touchdown[:, None, None], shifted_queue, queue)
    next_first_side = torch.where(touchdown, 1 - first_side, first_side)

    next_lift = self.liftoff[None, :] + ((phase[:, None] - self.liftoff[None, :]).div(2.0 * math.pi).floor() + 1.0) * (2.0 * math.pi)
    lifted = (next_lift <= end_phase[:, None]) & moving[:, None]
    indices = (self.sides[None, :] != next_first_side[:, None]).long()
    cand_targets = next_queue.gather(1, indices[:, :, None].expand(-1, -1, 3))
    next_targets = torch.where(lifted[:, :, None], cand_targets, targets)
    next_targets = torch.where(arrived[:, None, None], terminal_feet, next_targets)
    next_pending = torch.where(arrived, torch.full_like(next_pending, -1), next_pending)

    # Update mode & frequency for tick t+1 (exact match to TensorFootstepManager & StepEpisodeManager)
    next_mode = mode.clone()
    next_freq = frequency.clone()

    starting = mode == 1
    ramped_up = (frequency + req_freq * (CONTROL_DT / start_dur.clamp_min(1e-3))).clamp_max(req_freq)
    next_freq = torch.where(starting, ramped_up, next_freq)
    next_mode = torch.where(starting & (ramped_up >= req_freq - 1e-5), torch.zeros_like(next_mode), next_mode)

    enter_stop = (
      ((next_mode == 0) | (next_mode == 1))
      & (stop_phase > 0.0)
      & (elapsed < stop_ramp_start - 1e-6)
      & (next_elapsed >= stop_ramp_start - 1e-6)
    )
    next_mode = torch.where(enter_stop, torch.full_like(next_mode, 2), next_mode)

    stopping = (next_mode == 2) & ~arrived
    sd = stop_dur.clamp_min(1e-3)
    rb = (next_elapsed - stop_ramp_start).clamp(torch.zeros_like(sd), sd)
    ra = (next_elapsed + CONTROL_DT - stop_ramp_start).clamp(torch.zeros_like(sd), sd)
    coast = (stop_ramp_start - next_elapsed).clamp(0.0, CONTROL_DT)
    ramp_area = (ra - rb) * (1.0 - (ra + rb) / (2.0 * sd))
    stop_freq_next = stop_init_freq * (coast + ramp_area) / CONTROL_DT
    next_freq = torch.where(stopping, stop_freq_next, next_freq)

    next_mode = torch.where(arrived, torch.full_like(next_mode, 5), next_mode)
    next_freq = torch.where(arrived, torch.zeros_like(next_freq), next_freq)

    wrapped_end = end_phase[:, None].remainder(2.0 * math.pi)
    end_win = self.liftoff[None, :].remainder(2.0 * math.pi)
    start_win = self.contact_start[None, :]
    planned_contacts = torch.where(
      end_win > start_win,
      (wrapped_end >= start_win) & (wrapped_end < end_win),
      (wrapped_end >= start_win) | (wrapped_end < end_win),
    )
    double_support = planned_contacts.all(-1) & moving
    return (
      end_phase,
      next_freq,
      next_mode,
      next_elapsed,
      next_targets,
      next_supports,
      next_queue,
      next_first_side,
      next_pending,
      double_support,
      arrived.to(phase.dtype),
    )

  def step(
    self,
    prev_state: torch.Tensor,
    next_state: torch.Tensor,
    foot_force: torch.Tensor,
    action: torch.Tensor,
    prev_action: torch.Tensor,
    aux: torch.Tensor,
    end_phase: torch.Tensor,
    frequency: torch.Tensor,
    targets: torch.Tensor,
    supports: torch.Tensor,
    future_world: torch.Tensor,
    pending_anchor: torch.Tensor,
    double_support: torch.Tensor,
    theta: torch.Tensor,
    arrived: torch.Tensor | None = None,
    terminal_feet: torch.Tensor | None = None,
    first_side: torch.Tensor | None = None,
    req_freq: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, _KinematicFrame, torch.Tensor]:
    """Advance one 20ms imagined footstep tick with differentiable contact and differentiable re-anchoring."""
    frame = self._kinematic_frame(next_state)
    dtype = next_state.dtype
    swing_pos_w = self.stage_params[0]
    swing_yaw_w = self.stage_params[1]
    swing_pos_std = self.stage_params[2]
    swing_yaw_std = self.stage_params[3]
    swing_power = self.stage_params[4]
    landing_w = self.stage_params[5]
    theta_active = self.stage_params[6]

    moving = frequency > 1e-5
    is_moving = moving.to(dtype)
    is_standing = 1.0 - is_moving
    stance_mask, swing_progress = phase_windows(end_phase, phase_cfg=self.phase_cfg)
    stance_mask = stance_mask | ~moving.unsqueeze(-1)
    in_stance = stance_mask.to(dtype)
    in_swing = 1.0 - in_stance

    # 1. Non-underflowing differentiable contact & soft timers
    fz = foot_force[:, :, 2].clamp_min(0.0)
    fz_total = fz.sum(dim=-1, keepdim=True) + 10.0
    stance_load_share = fz / fz_total
    unloaded_share = (1.0 - stance_load_share).square() * torch.exp(-fz / 45.0)
    p_lifted = torch.sigmoid((frame.relative_z - 0.020) / 0.012)
    p_airborne = (unloaded_share * p_lifted).clamp(0.0, 1.0)
    p_contact = 1.0 - p_airborne
    p_slip_contact = torch.sigmoid((fz - 20.0) / 5.0)

    prev_contact_prob = aux[:, 0:2]
    next_air_time = p_airborne * (aux[:, 2:4] + CONTROL_DT)
    next_contact_time = p_contact * (aux[:, 4:6] + CONTROL_DT)

    # 2. Differentiable re-anchoring: propagates Step-2 local_footprint gradients back to Step-1 landing pose!
    cur_anchor = aux[:, 6:9]
    pending_clamped = pending_anchor.clamp_min(0)
    pending_p_contact = p_contact.gather(1, pending_clamped[:, None]).squeeze(1)
    reanchor_cond = (pending_anchor >= 0) & double_support & (pending_p_contact > 0.35)
    # Use smooth weight when reanchoring so foot pose & contact force both remain differentiable.
    w_reanchor = reanchor_cond.to(dtype) * pending_p_contact
    chosen_foot = frame.feet_pose.gather(1, pending_clamped[:, None, None].expand(-1, 1, 3)).squeeze(1)
    next_anchor_xy = cur_anchor[:, :2] + w_reanchor.unsqueeze(-1) * (chosen_foot[:, :2] - cur_anchor[:, :2])
    next_anchor_yaw = wrap(cur_anchor[:, 2] + w_reanchor * wrap(chosen_foot[:, 2] - cur_anchor[:, 2]))
    next_anchor = torch.cat((next_anchor_xy, next_anchor_yaw.unsqueeze(-1)), dim=-1)
    next_pending = torch.where(reanchor_cond, torch.full_like(pending_anchor, -1), pending_anchor)

    # 3. PPO contact_schedule (+1.0), foot_air_time (+3.2), swing_clearance (-0.5), and swing_contact (-0.5)
    p_foot_match = in_stance * p_contact + in_swing * p_airborne
    p_matched = p_foot_match[:, 0] * p_foot_match[:, 1]
    r_schedule = (1.0 * p_matched - 1.5 * (in_stance * p_airborne).sum(-1)) * is_moving

    single_sup = (in_stance.sum(-1) == 1).to(dtype)
    held_time = (in_stance * next_contact_time + in_swing * next_air_time).min(dim=-1).values
    norm_held = (held_time * frequency / SWING_FRACTION).clamp(0.0, 1.0)
    air_credit = torch.minimum(norm_held.unsqueeze(-1), swing_progress)
    r_air_time = 3.2 * 0.4 * (air_credit * in_swing).sum(-1) * p_matched * single_sup * is_moving

    swing_sin = torch.sin(math.pi * swing_progress)
    sin_sq = swing_sin.square()
    clearance_target = 0.11 * sin_sq * in_swing * is_moving.unsqueeze(-1)
    r_clearance = -0.5 * (F.smooth_l1_loss(frame.relative_z, clearance_target, reduction="none", beta=0.005) / 0.11).sum(-1)
    r_swing_contact = -0.5 * (p_contact * in_swing * sin_sq).sum(-1) * is_moving

    # 4. PPO footstep_swing_position (+5.0), footstep_swing_yaw (+5.0), footstep_landing (-0.5/event), and step_landing_bonus (+10)
    delta_xy = frame.foot_pos[:, :, :2] - targets[:, :, :2]
    dist_xy = (delta_xy.square().sum(-1) + 1e-8).sqrt() - 1e-4
    yaw_err = wrap(frame.foot_yaw - targets[:, :, 2])
    yaw_abs = (yaw_err.square() + 1e-6).sqrt() - 1e-3

    stance_supported = 1.0 - (1.0 - in_stance[:, 0] * p_contact[:, 0]) * (1.0 - in_stance[:, 1] * p_contact[:, 1])
    eff_power = torch.ones_like(swing_power) if self.is_step_episode else swing_power
    ramp = swing_progress.clamp(0.0, 1.0).pow(eff_power) * in_swing
    r_swing_pos = swing_pos_w * stance_supported * (torch.exp(-(dist_xy / swing_pos_std).square()) * ramp).sum(-1) * is_moving
    r_swing_yaw = swing_yaw_w * stance_supported * (torch.exp(-(yaw_err / swing_yaw_std).square()) * ramp).sum(-1) * is_moving

    # Phase-normalized late-swing integral (\int 4*s^3*(f/SWING_FRACTION) dt == 1 per step) matches PPO's once-per-landing event weight
    accuracy_cost = 0.5 * dist_xy / 0.05 + 0.5 * yaw_abs / 0.20
    landing_density = 4.0 * swing_progress.clamp(0.0, 1.0).pow(3) * in_swing * (frequency.unsqueeze(-1) / SWING_FRACTION)
    r_landing = landing_w * (accuracy_cost * landing_density).sum(-1) * is_moving

    # Smooth moving-only BPTT spatial bridge (strictly 0 when standing at f==0)
    xy_l1 = F.smooth_l1_loss(frame.foot_pos[:, :, :2], targets[:, :, :2], reduction="none", beta=0.015).sum(-1)
    yaw_l1 = F.smooth_l1_loss(yaw_err, torch.zeros_like(yaw_err), reduction="none", beta=0.04)
    move_pull_gate = 0.60 * swing_progress.clamp(0.0, 1.0).square() * in_swing + 0.40 * in_stance
    r_foot_pull = -((2.5 * xy_l1 + 0.8 * yaw_l1) * move_pull_gate).sum(-1) * is_moving

    if self.is_step_episode and arrived is not None and terminal_feet is not None:
      arr_delta_xy = frame.foot_pos[:, :, :2] - terminal_feet[:, :, :2]
      arr_dist_xy = (arr_delta_xy.square().sum(-1) + 1e-8).sqrt() - 1e-4
      arr_yaw_err = wrap(frame.foot_yaw - terminal_feet[:, :, 2])
      worst_dist = arr_dist_xy.amax(-1)
      worst_yaw = arr_yaw_err.abs().amax(-1)
      both_planted = p_contact[:, 0] * p_contact[:, 1]
      yaw_gate = torch.exp(-(worst_yaw / 0.20).square())
      arr_bullseye = both_planted * yaw_gate * (
        4.0 * torch.exp(-(worst_dist / 0.10).square())
        + 6.0 * torch.exp(-(worst_dist / 0.042).square())
      )
      worst_l1 = F.smooth_l1_loss(worst_dist, torch.zeros_like(worst_dist), reduction="none", beta=0.015)
      worst_yaw_l1 = F.smooth_l1_loss(worst_yaw, torch.zeros_like(worst_yaw), reduction="none", beta=0.04)
      r_arrival = (arrived.to(dtype) / CONTROL_DT) * (arr_bullseye - 4.0 * worst_l1 - 1.0 * worst_yaw_l1)
    else:
      r_arrival = torch.zeros_like(r_schedule)

    # 5. Moving-only BPTT horizon CoM stabilizer (strictly inactive during normal quiet standing so f==0 equilibrium matches PPO)
    m_sup = 0.5 * (supports[:, 0, :2] + supports[:, 1, :2])
    if first_side is not None:
      anchor_idx = (1 - first_side)[:, None, None].expand(-1, 1, 2)
      swing_idx = first_side[:, None, None].expand(-1, 1, 2)
      anchor_sup = supports[:, :, :2].gather(1, anchor_idx).squeeze(1)
      next_swing_tgt = future_world[:, :, :2].gather(1, swing_idx).squeeze(1)
      m_next_step = 0.5 * (anchor_sup + next_swing_tgt)
    else:
      m_next_step = 0.5 * (targets[:, 0, :2] + targets[:, 1, :2])
    m_actual_feet = 0.5 * (frame.foot_pos[:, 0, :2] + frame.foot_pos[:, 1, :2]).detach()
    eff_freq = (
      torch.where(req_freq > frequency, 0.5 * (frequency + req_freq), frequency)
      if req_freq is not None
      else frequency
    )
    v_com_star = (2.0 * eff_freq.unsqueeze(-1) * (m_next_step - m_sup)).clamp(-1.5, 1.5) * is_moving.unsqueeze(-1)
    com_vel_l1 = F.smooth_l1_loss(frame.v_w[:, :2], v_com_star, reduction="none", beta=0.05).sum(-1)

    step_alpha = torch.where(
      (in_swing.sum(-1) > 0.5),
      0.25 + 0.75 * swing_progress.amax(-1),
      torch.full_like(frequency, 0.25),
    )
    cp_moving = torch.lerp(m_sup, m_next_step, step_alpha.unsqueeze(-1))
    cp_center = is_moving.unsqueeze(-1) * cp_moving + is_standing.unsqueeze(-1) * m_actual_feet
    capture_point = next_state[:, :2] + 0.22 * frame.v_w[:, :2]
    cp_dist = ((capture_point - cp_center).square().sum(-1) + 1e-6).sqrt() - 1e-3
    # At f==0, 9 cm support-polygon margin leaves the natural straight-leg ankle/hip axis (-4 cm behind foot_site) 100% penalty-free.
    cp_margin = 0.050 * is_moving + 0.090 * is_standing
    cp_excess = (cp_dist - cp_margin).clamp_min(0.0)
    r_com_guidance = -1.4 * is_moving * com_vel_l1 - 3.5 * cp_excess

    # 6. PPO slip, soft_landing, standing velocity/hold, flatness, contact yaw rate, posture, height, knee straightness, and theta
    foot_vel_xy_sq = frame.foot_vel[:, :, :2].square().sum(-1)
    foot_speed_3d = (frame.foot_vel.square().sum(-1) + 1e-6).sqrt() - 1e-3
    r_slip = -2.0 * (foot_vel_xy_sq * p_slip_contact).sum(-1) * is_moving
    r_slip_still = -1.0 * foot_speed_3d.sum(-1) * is_standing
    r_stand_feet = -0.5 * p_airborne.sum(-1) * is_standing

    onset_prob = (p_contact * (1.0 - prev_contact_prob)).detach()
    force_mag = (foot_force.square().sum(-1) + 1e-6).sqrt() - 1e-3
    excess_spike = (force_mag - 300.0).clamp_min(0.0)
    near_ground_gate = torch.sigmoid((0.055 - frame.relative_z) / 0.015) * in_swing
    flare_vz_excess = (-frame.foot_vel[:, :, 2] - 0.55).clamp_min(0.0)
    r_soft_landing = (
      (-3e-3 * force_mag - 6e-3 * excess_spike) * onset_prob
      - 3.0 * flare_vz_excess.square() * near_ground_gate
    ).sum(-1) * is_moving

    pos_score = (1.0 + dist_xy.square().mean(-1) / 0.10**2).reciprocal()
    r_stand_lin = (
      2.0
      * torch.exp(-(frame.v_b[:, :2].square().sum(-1) + 1.5 * frame.v_b[:, 2].square()) / 0.20)
      * pos_score
      * is_standing
    )
    r_stand_ang = (
      0.5
      * torch.exp(-(frame.omega_b[:, 2].square() + 0.05 * frame.omega_b[:, :2].square().sum(-1)) / 0.70**2)
      * pos_score
      * is_standing
    )
    hold_deadband = 0.0 if self.is_step_episode else 0.03
    r_hold_pos = -0.2 * (((dist_xy - hold_deadband).clamp_min(0.0) / 0.10).square().mean(-1)) * is_standing

    omega_foot_z = frame.fk.foot_angular_velocity[:, :, 2]
    omega_foot_z_abs = (omega_foot_z.square() + 1e-6).sqrt() - 1e-3
    r_foot_yaw_rate = -0.005 * (p_contact * omega_foot_z_abs).sum(-1) / math.radians(1.0)

    gravity_foot = torch.zeros_like(frame.foot_pos)
    gravity_foot[:, :, 2] = -1.0
    foot_grav = rotate(conjugate(frame.foot_quat), gravity_foot)
    foot_tilt = torch.atan2(
      (foot_grav[:, :, :2].square().sum(-1) + 1e-8).sqrt(),
      (-foot_grav[:, :, 2]).clamp_min(1e-4),
    )
    flat_active = torch.maximum(torch.maximum(in_stance, p_contact), is_standing.unsqueeze(-1))
    r_flatness = -0.2 * ((foot_tilt / math.radians(15.0)).square() * flat_active).sum(-1) / flat_active.sum(-1).clamp_min(1.0)

    torso_tilt_rad = torch.atan2(
      (frame.g_torso[:, :2].square().sum(-1) + 1e-8).sqrt(),
      (-frame.g_torso[:, 2]).clamp_min(1e-4),
    )
    pelvis_tilt_rad = torch.atan2(
      (frame.g_pelvis[:, :2].square().sum(-1) + 1e-8).sqrt(),
      (-frame.g_pelvis[:, 2]).clamp_min(1e-4),
    )
    r_torso_ori = -2.0 * torso_tilt_rad.square() - 3.0 * is_standing * F.smooth_l1_loss(
      torso_tilt_rad, torch.zeros_like(torso_tilt_rad), reduction="none", beta=0.04
    )
    r_pelvis_ori = (
      -1.0 * frame.g_pelvis[:, :2].square().sum(-1)
      - 1.0 * is_standing * pelvis_tilt_rad.square()
      - 3.0 * is_standing * F.smooth_l1_loss(
        pelvis_tilt_rad, torch.zeros_like(pelvis_tilt_rad), reduction="none", beta=0.04
      )
    )
    r_torso_ang = -0.05 * frame.fk.body_angular_velocity[:, self.torso_id, :2].square().sum(-1)

    # PPO head_height (+0.4 capped at straight-leg height, zero crouch-inducing Gaussian target) + head_height_low (-1.0)
    any_supported = 1.0 - p_airborne[:, 0] * p_airborne[:, 1]
    r_head_height = 0.4 * (frame.head_z / 1.265).clamp(0.0, 1.0) * any_supported
    r_head_low = -1.0 * (1.161 - frame.head_z).clamp_min(0.0) / 0.20

    # PPO torso_feet_heading (-12.0 in theta stages, balance-gated under extreme tilt) or waist_yaw_zero (-0.8)
    sine_f, cosine_f = frame.foot_yaw.sin().sum(-1), frame.foot_yaw.cos().sum(-1)
    mean_foot_yaw = torch.where(torch.hypot(sine_f, cosine_f) > 1e-6, torch.atan2(sine_f, cosine_f), frame.foot_yaw[:, 0])
    torso_yaw = _quat_yaw(frame.torso_q)
    actual_theta = wrap(torso_yaw - mean_foot_yaw)
    theta_err = wrap(actual_theta - theta)
    upright_gate = (1.0 - 1.5 * pelvis_tilt_rad.detach()).clamp(0.25, 1.0)
    r_theta = -12.0 * theta_err.square() * upright_gate
    waist_yaw_q = next_state[:, 7 + 12]
    r_heading = theta_active * r_theta + (1.0 - theta_active) * (-0.8 * waist_yaw_q.square())

    # 7. PPO stance_knee_bend (-0.2), lower_body_copper_proxy (-0.25, standing_scale=2.0, hard limits), hip_yaw_zero (-0.1), waist edges
    q_lower = next_state[:, 7 : 7 + ACTION_DIM]
    qd_lower = next_state[:, 42 : 42 + ACTION_DIM]
    qd_prev = prev_state[:, 42 : 42 + ACTION_DIM]
    qacc_lower = (qd_lower - qd_prev) / CONTROL_DT

    knee_q = torch.stack((q_lower[:, 3], q_lower[:, 9]), dim=-1)
    walk_knee_excess = ((knee_q - (math.pi / 4.0)).clamp_min(0.0) / (math.pi / 4.0)).square()
    walk_knee_cost = (walk_knee_excess * in_stance).sum(-1) / in_stance.sum(-1).clamp_min(1.0)
    stand_knee_pos = knee_q.clamp_min(0.0)
    r_stand_straight = (
      0.8 * torch.exp(-stand_knee_pos.square().mean(-1) / 0.16**2)
      - 3.5 * F.smooth_l1_loss(stand_knee_pos, torch.zeros_like(stand_knee_pos), reduction="none", beta=0.03).mean(-1)
    )
    r_stance_knee = -0.2 * is_moving * walk_knee_cost + is_standing * r_stand_straight

    r_hip_yaw = -0.1 * (q_lower[:, 2].square() + q_lower[:, 8].square())

    q_des = self.default_position[:ACTION_DIM] + action * self.action_scale
    tau = (self.lower_kp * (q_des - q_lower) - self.lower_kd * qd_lower).clamp(
      -self.lower_effort_limit, self.lower_effort_limit
    )
    hard_lower = self.hard_limits[:ACTION_DIM]
    limit_margin = math.radians(0.1)
    pushing_lim = (
      ((q_lower <= hard_lower[:, 0] + limit_margin) & (tau < 0))
      | ((q_lower >= hard_lower[:, 1] - limit_margin) & (tau > 0))
    )
    copper_w = 1.0 + 49.0 * pushing_lim.to(dtype)
    stand_copper_scale = 1.0 + is_standing  # standing_scale = 2.0 when f == 0
    r_copper = -0.25 * stand_copper_scale * ((tau / 100.0).square() * copper_w).sum(-1)

    waist_limits = hard_lower[13:15].unsqueeze(0).expand(len(next_state), 2, 2)
    r_waist_edge = -2.0 * joint_edge_cost(q_lower[:, 13:15], waist_limits, 0.15)

    act_diff_sq = (action - prev_action).square().sum(-1)
    if self.is_step_episode:
      r_act_rate = -0.02 * act_diff_sq - 0.04 * is_standing * act_diff_sq.clamp_max(25.0)
      r_stand_qd = -0.2 * qd_lower.square().mean(-1).clamp_max(4.0) * is_standing
    else:
      r_act_rate = -0.02 * act_diff_sq
      r_stand_qd = torch.zeros_like(r_act_rate)
    r_qacc = -2.5e-7 * qacc_lower.square().sum(-1)

    # 8. Differentiable soft termination (tilt > 70deg, height < 0.35m, or stance foot > 1.0m from target)
    stance_dist = (dist_xy * in_stance).amax(-1)
    termination = self._termination_from_frame(frame.g_pelvis, frame.height, stance_dist)
    term_weight = -250.0 if self.is_step_episode else -120.0
    r_terminated = term_weight * termination

    reward = CONTROL_DT * (
      r_schedule
      + r_air_time
      + r_clearance
      + r_swing_contact
      + r_swing_pos
      + r_swing_yaw
      + r_landing
      + r_foot_pull
      + r_arrival
      + r_com_guidance
      + r_slip
      + r_slip_still
      + r_stand_feet
      + r_soft_landing
      + r_stand_lin
      + r_stand_ang
      + r_hold_pos
      + r_foot_yaw_rate
      + r_flatness
      + r_torso_ori
      + r_pelvis_ori
      + r_torso_ang
      + r_head_height
      + r_head_low
      + r_heading
      + r_stance_knee
      + r_hip_yaw
      + r_copper
      + r_waist_edge
      + r_act_rate
      + r_stand_qd
      + r_qacc
      + r_terminated
    )
    next_aux = pack_footstep_aux(p_contact, next_air_time, next_contact_time, next_anchor)
    return reward, termination, next_aux, frame, next_pending

  def _termination_from_frame(
    self,
    g_pelvis: torch.Tensor,
    height: torch.Tensor,
    stance_dist: torch.Tensor,
  ) -> torch.Tensor:
    tilt_margin = (self.cos_tilt_limit - (-g_pelvis[:, 2])) / 0.15
    height_margin = (0.35 - height) / 0.10
    dist_margin = (stance_dist - 1.0) / 0.25
    smooth_margin = 0.1 * torch.logsumexp(
      10.0 * torch.stack((tilt_margin, height_margin, dist_margin), dim=-1),
      dim=-1,
    )
    return self.termination_probability * torch.sigmoid(15.0 * smooth_margin)

  def critic_features(
    self,
    model: PositionWorldModel,
    state: torch.Tensor,
    context: torch.Tensor,
    prev_action: torch.Tensor,
    aux: torch.Tensor,
    arm_target: torch.Tensor,
    phase: torch.Tensor,
    frequency: torch.Tensor,
    targets: torch.Tensor,
    supports: torch.Tensor,
    queue: torch.Tensor,
    first_side: torch.Tensor,
    theta: torch.Tensor,
    *,
    foot_pos: torch.Tensor | None = None,
    foot_yaw: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Construct 191D SE(2) translation/yaw-invariant Critic features."""
    if foot_pos is None or foot_yaw is None:
      fk = self.kinematics(state, validate=False)
      foot_pos = fk.foot_position
      foot_yaw = _quat_yaw(unit(fk.foot_quaternion))

    feature = (state_features(state, validate=False) - model.input_mean[:68]) / model.input_scale[:68]
    physical = (context - model.input_mean[97:]) / model.input_scale[97:]
    cmd_enc = command_from_world_queue(phase, frequency, queue, first_side, aux[:, 6:9], theta)

    pelvis_q = unit(state[:, 3:7])
    pelvis_yaw = _quat_yaw(pelvis_q)
    pelvis_pose_2d = torch.cat((state[:, :2], pelvis_yaw.unsqueeze(-1)), dim=-1)

    sup_local = to_local(supports, pelvis_pose_2d[:, None, :])
    sup_enc = torch.cat(
      (sup_local[..., :2], sup_local[..., 2:3].sin(), sup_local[..., 2:3].cos() - 1.0),
      dim=-1,
    ).flatten(1)

    tgt_local = to_local(targets, pelvis_pose_2d[:, None, :])
    tgt_enc = torch.cat(
      (tgt_local[..., :2], tgt_local[..., 2:3].sin(), tgt_local[..., 2:3].cos() - 1.0),
      dim=-1,
    ).flatten(1)

    cos_p, sin_p = pelvis_yaw.cos(), pelvis_yaw.sin()
    d_foot_w = foot_pos[:, :, :2] - targets[:, :, :2]
    d_foot_bx = cos_p[:, None] * d_foot_w[:, :, 0] + sin_p[:, None] * d_foot_w[:, :, 1]
    d_foot_by = -sin_p[:, None] * d_foot_w[:, :, 0] + cos_p[:, None] * d_foot_w[:, :, 1]
    d_foot_yaw = wrap(foot_yaw - targets[:, :, 2])
    foot_to_tgt = torch.cat((d_foot_bx, d_foot_by, d_foot_yaw), dim=-1)

    indices = (self.sides[None, :] != first_side[:, None]).long()
    future_world = queue.gather(1, indices[:, :, None].expand(-1, -1, 3))
    m_sup = 0.5 * (supports[:, 0, :2] + supports[:, 1, :2])
    anchor_sup = supports[:, :, :2].gather(1, (1 - first_side)[:, None, None].expand(-1, 1, 2)).squeeze(1)
    next_swing_tgt = future_world[:, :, :2].gather(1, first_side[:, None, None].expand(-1, 1, 2)).squeeze(1)
    m_next_step = 0.5 * (anchor_sup + next_swing_tgt)
    is_moving = (frequency > 1e-5).to(state.dtype).unsqueeze(-1)
    v_star_w = (2.0 * frequency.unsqueeze(-1) * (m_next_step - m_sup)).clamp(-1.5, 1.5) * is_moving
    dv_w = state[:, 36:38] - v_star_w
    dv_b = torch.stack((cos_p * dv_w[:, 0] + sin_p * dv_w[:, 1], -sin_p * dv_w[:, 0] + cos_p * dv_w[:, 1]), dim=-1)

    foot_z = foot_pos[:, :, 2]
    min_foot_z = foot_z.min(dim=-1).values
    height_rel = (state[:, 2] - min_foot_z - 0.73).unsqueeze(-1)
    rel_foot_z = foot_z - min_foot_z.unsqueeze(-1)
    arm_err = arm_target - state[:, 7 + ACTION_DIM : 36]

    out = torch.cat(
      (
        feature,
        physical,
        cmd_enc,
        sup_enc,
        tgt_enc,
        foot_to_tgt,
        dv_b,
        aux[:, 0:6],
        rel_foot_z,
        height_rel,
        prev_action,
        arm_err,
      ),
      dim=-1,
    )
    if out.shape[-1] != CRITIC_DIM:
      raise AssertionError(f"Expected {CRITIC_DIM} critic features, got {out.shape[-1]}")
    return out
