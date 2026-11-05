"""Task-facing observations and unweighted signals for position-world-model rollouts.

These are explicit tensor interfaces, not replacements for mjlab's task managers.
Hard events and smooth planning surrogates are returned under distinct names.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, NamedTuple

import torch
from torch import nn

from g1_lower_rl.footstep_phase import DEFAULT_PHASE_CFG, FootstepPhaseCfg
from g1_lower_rl.world_model.model import BODY_NAMES, STATE_FIELDS, TORQUE_DIM

from .contact import ContactUpdate
from .kinematics import KinematicState, TorchKinematics
from .math import conjugate, heading, multiply, orientation_error_squared, rotate, rotation_6d, unit, wrap, yaw

TRACKED_BODIES = (
  "pelvis",
  "left_hip_roll_link",
  "left_knee_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_knee_link",
  "right_ankle_roll_link",
  "torso_link",
  "left_shoulder_roll_link",
  "left_elbow_link",
  "left_wrist_yaw_link",
  "right_shoulder_roll_link",
  "right_elbow_link",
  "right_wrist_yaw_link",
)
POLICY_JOINT_INDICES = tuple(range(TORQUE_DIM))


class TaskSignals(NamedTuple):
  observations: dict[str, torch.Tensor]
  rewards: dict[str, torch.Tensor]
  costs: dict[str, torch.Tensor]
  events: dict[str, torch.Tensor]


def _positive(**values: float) -> None:
  if any(not math.isfinite(value) or value <= 0 for value in values.values()):
    raise ValueError(f"Expected finite positive scales: {values}")


def _phase_windows(phase: torch.Tensor, cfg: FootstepPhaseCfg) -> tuple[torch.Tensor, torch.Tensor]:
  # Same [start, liftoff) intervals as tasks/footstep_tracking/reward_math.py,
  # generalized to arbitrary batch axes and without importing task registration.
  start = phase.new_tensor(cfg.contact_start_rad)
  duration = phase.new_tensor(cfg.stance_fractions)
  progress = torch.remainder(phase.unsqueeze(-1) - start, 2 * math.pi) / (2 * math.pi)
  wrapped_start = torch.remainder(start, 2 * math.pi)
  liftoff = torch.remainder(phase.new_tensor(cfg.liftoff_rad), 2 * math.pi)
  wrapped = torch.remainder(phase, 2 * math.pi).unsqueeze(-1)
  stance = torch.where(
    liftoff > wrapped_start, (wrapped >= wrapped_start) & (wrapped < liftoff), (wrapped >= wrapped_start) | (wrapped < liftoff)
  )
  swing = ((progress - duration) / (1 - duration)).clamp(0, 1)
  return stance, swing


class TaskAdapter(nn.Module):
  default_joint_position: torch.Tensor

  def __init__(self, kinematics: TorchKinematics, default_joint_position: torch.Tensor | None = None):
    super().__init__()
    self.kinematics = kinematics
    default = torch.zeros(TORQUE_DIM, dtype=torch.float64) if default_joint_position is None else default_joint_position
    if default.shape != (TORQUE_DIM,) or not torch.isfinite(default).all():
      raise ValueError(f"default_joint_position must contain {TORQUE_DIM} finite canonical joint positions")
    self.register_buffer("default_joint_position", default.detach().clone())

  def proprioception(self, state: torch.Tensor, last_action: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    q = unit(state[..., 3:7])
    gravity = torch.zeros_like(state[..., :3])
    gravity[..., 2] = -1
    result = {
      "base_ang_vel": state[..., STATE_FIELDS["root_angular_velocity"]],
      "base_lin_vel": rotate(conjugate(q), state[..., STATE_FIELDS["root_linear_velocity"]]),
      "projected_gravity": rotate(conjugate(q), gravity),
      "joint_pos": state[..., STATE_FIELDS["joint_position"]] - self.default_joint_position.to(state),
      "joint_vel": state[..., STATE_FIELDS["joint_velocity"]],
    }
    if last_action is not None:
      if last_action.shape[:-1] != state.shape[:-1]:
        raise ValueError("last_action must share the state batch shape")
      result["actions"] = last_action
    return result

  @staticmethod
  def contact_observations(contact: ContactUpdate) -> dict[str, torch.Tensor]:
    return {
      "foot_contact": contact.memory.contact.to(contact.soft),
      "foot_contact_soft": contact.soft,
      "foot_air_time": contact.memory.air_time,
      "foot_contact_time": contact.memory.contact_time,
      "foot_contact_forces": (torch.sign(contact.force) * torch.log1p(contact.force.abs())).flatten(-2),
    }


@dataclass(frozen=True)
class MotionConfig:
  body_names: tuple[str, ...] = TRACKED_BODIES
  anchor_body: str = "torso_link"
  body_position_std: float = 0.3
  body_orientation_std: float = 0.4
  body_velocity_std: float = 1.0
  body_angular_std: float = 3.14
  anchor_position_std: float = 0.3
  anchor_position_coarse_std: float = 1.2
  anchor_position_coarse_axes: Literal["xyz", "xy"] = "xyz"
  anchor_orientation_std: float = 0.4
  anchor_velocity_std: float = 0.4
  swing_ratio_std: float = 0.3
  minimum_lift: float = 0.05
  anchor_height_limit: float = 0.25
  anchor_tilt_limit: float = 0.8
  foot_height_limit: float = 0.25

  def __post_init__(self) -> None:
    _positive(**{name: value for name, value in vars(self).items() if isinstance(value, (int, float))})
    if self.anchor_position_coarse_axes not in ("xyz", "xy"):
      raise ValueError("Coarse anchor position axes must be xyz or xy")
    if not self.body_names or len(set(self.body_names)) != len(self.body_names):
      raise ValueError("Motion bodies must be nonempty and unique")
    if any(name not in BODY_NAMES for name in (*self.body_names, self.anchor_body)):
      raise ValueError("Unknown motion body name")


@dataclass(frozen=True)
class MotionGoal:
  """Reference state/window in canonical joint order; optional archive body data.

  kinematics, when supplied, must have all 44 canonical bodies and world velocities.
  This permits mocap references whose measured body poses differ from robot FK.
  window is [..., time, 71]; its body-frame 38D RGMT tokens are exposed without flattening.
  """

  state: torch.Tensor
  kinematics: KinematicState | None = None
  window: torch.Tensor | None = None


class MotionTrackingAdapter(TaskAdapter):
  def __init__(
    self, kinematics: TorchKinematics, config: MotionConfig | None = None, default_joint_position: torch.Tensor | None = None
  ):
    super().__init__(kinematics, default_joint_position)
    self.config = config = MotionConfig() if config is None else config
    self.body_indices = tuple(BODY_NAMES.index(name) for name in config.body_names)
    self.anchor = BODY_NAMES.index(config.anchor_body)
    self.feet = tuple(BODY_NAMES.index(f"{side}_ankle_roll_link") for side in ("left", "right"))
    for name, indices in (
      ("body_index_tensor", self.body_indices), ("foot_index_tensor", self.feet),
      ("policy_index_tensor", POLICY_JOINT_INDICES),
    ):
      self.register_buffer(name, torch.tensor(indices, dtype=torch.long), persistent=False)

  @staticmethod
  def reference_tokens(window: torch.Tensor) -> torch.Tensor:
    q = unit(window[..., 3:7])
    gravity = torch.zeros_like(window[..., :3])
    gravity[..., 2] = -1
    return torch.cat(
      (
        rotate(conjugate(q), window[..., STATE_FIELDS["root_linear_velocity"]]),
        window[..., STATE_FIELDS["root_angular_velocity"]],
        rotate(conjugate(q), gravity),
        window[..., STATE_FIELDS["joint_position"]],
      ),
      dim=-1,
    )

  def forward(self, state: torch.Tensor, goal: MotionGoal, last_action: torch.Tensor | None = None) -> TaskSignals:
    actual = self.kinematics(state)
    reference = self.kinematics(goal.state) if goal.kinematics is None else goal.kinematics
    if reference.body_position.shape != actual.body_position.shape:
      raise ValueError("Motion reference must have the same batch and 44 canonical bodies")
    c, a, ids = self.config, self.anchor, self.body_index_tensor.to(device=state.device)
    ap, aq = actual.body_position[..., a, :], actual.body_quaternion[..., a, :]
    rp, rq = reference.body_position[..., a, :], reference.body_quaternion[..., a, :]
    # Preserve the production task's yaw(relative quaternion), including tilted anchors.
    alignment = heading(multiply(aq, conjugate(rq)))
    aligned_anchor = torch.cat((ap[..., :2], rp[..., 2:3]), dim=-1)
    aligned_position = aligned_anchor.unsqueeze(-2) + rotate(
      alignment.unsqueeze(-2).expand_as(reference.body_quaternion), reference.body_position - rp.unsqueeze(-2)
    )
    aligned_quaternion = multiply(alignment.unsqueeze(-2), reference.body_quaternion)
    pos_error = (actual.body_position[..., ids, :] - aligned_position[..., ids, :]).square().sum(-1).mean(-1)
    ori_error = orientation_error_squared(actual.body_quaternion[..., ids, :], aligned_quaternion[..., ids, :]).mean(-1)
    vel_error = (
      (actual.body_linear_velocity[..., ids, :] - reference.body_linear_velocity[..., ids, :]).square().sum(-1).mean(-1)
    )
    ang_error = (
      (actual.body_angular_velocity[..., ids, :] - reference.body_angular_velocity[..., ids, :]).square().sum(-1).mean(-1)
    )
    anchor_orientation = orientation_error_squared(aq, rq)
    anchor_position = (ap - rp).square().sum(-1)
    coarse_position = (ap[..., :2] - rp[..., :2]).square().sum(-1) if c.anchor_position_coarse_axes == "xy" else anchor_position
    anchor_velocity = (actual.body_linear_velocity[..., a, :] - reference.body_linear_velocity[..., a, :]).square().sum(-1)
    foot_ids = self.foot_index_tensor.to(device=state.device)
    actual_z = actual.body_position[..., foot_ids, 2]
    target_z = aligned_position[..., foot_ids, 2]
    lift = actual_z.max(-1).values - actual_z.min(-1).values
    target_lift = target_z.max(-1).values - target_z.min(-1).values
    ratio_reward = torch.exp(-(lift / target_lift.clamp_min(c.minimum_lift) - 1).square() / c.swing_ratio_std**2)
    inv_anchor = conjugate(aq).unsqueeze(-2).expand_as(actual.body_quaternion)
    obs = self.proprioception(state, last_action)
    obs.update(
      {
        "motion_joint_position": goal.state[..., STATE_FIELDS["joint_position"]],
        "motion_joint_velocity": goal.state[..., STATE_FIELDS["joint_velocity"]],
        "policy_joint_pos": obs["joint_pos"].index_select(-1, self.policy_index_tensor.to(device=state.device)),
        "policy_joint_vel": obs["joint_vel"].index_select(-1, self.policy_index_tensor.to(device=state.device)),
        "body_position_anchor": rotate(inv_anchor, actual.body_position - ap.unsqueeze(-2))[..., ids, :],
        "reference_position_anchor": rotate(inv_anchor, aligned_position - ap.unsqueeze(-2))[..., ids, :],
        "reference_orientation_anchor": rotation_6d(multiply(inv_anchor, aligned_quaternion))[..., ids, :],
        "reference_tokens": self.reference_tokens(goal.state.unsqueeze(-2) if goal.window is None else goal.window),
      }
    )
    gravity = torch.zeros_like(ap)
    gravity[..., 2] = -1
    gravity_error = (rotate(conjugate(aq), gravity)[..., 2] - rotate(conjugate(rq), gravity)[..., 2]).abs()
    return TaskSignals(
      obs,
      {
        "body_position": torch.exp(-pos_error / c.body_position_std**2),
        "body_orientation": torch.exp(-ori_error / c.body_orientation_std**2),
        "body_linear_velocity": torch.exp(-vel_error / c.body_velocity_std**2),
        "body_angular_velocity": torch.exp(-ang_error / c.body_angular_std**2),
        "anchor_position": torch.exp(-anchor_position / c.anchor_position_std**2),
        "anchor_position_coarse": torch.exp(-coarse_position / c.anchor_position_coarse_std**2),
        "anchor_orientation": torch.exp(-anchor_orientation / c.anchor_orientation_std**2),
        "anchor_linear_velocity": torch.exp(-anchor_velocity / c.anchor_velocity_std**2),
        "swing_lift_ratio": torch.where(target_lift > c.minimum_lift, ratio_reward, torch.ones_like(ratio_reward)),
      },
      {"body_position_squared": pos_error, "body_orientation_squared": ori_error},
      {
        "anchor_height_failure": (ap[..., 2] - rp[..., 2]).abs() > c.anchor_height_limit,
        "anchor_gravity_failure": gravity_error > c.anchor_tilt_limit,
        "foot_height_failure": (actual_z - target_z).abs().amax(-1) > c.foot_height_limit,
      },
    )


@dataclass(frozen=True)
class VelocityConfig:
  linear_std: float = 0.5
  angular_std: float = 0.5
  height_std: float = 0.05
  z_penalty: float = 2.0
  std_scale: float = 0.0
  std_knee: float = 0.8
  cap_fraction: float = 0.5
  command_threshold: float = 0.1

  def __post_init__(self) -> None:
    _positive(linear_std=self.linear_std, angular_std=self.angular_std, height_std=self.height_std)
    if any(
      not math.isfinite(x) or x < 0
      for x in (self.z_penalty, self.std_scale, self.std_knee, self.cap_fraction, self.command_threshold)
    ):
      raise ValueError("Velocity penalty/scaling parameters must be finite and nonnegative")


@dataclass(frozen=True)
class VelocityGoal:
  twist: torch.Tensor
  height: torch.Tensor
  phase: torch.Tensor | None = None


class LowerBodyVelocityAdapter(TaskAdapter):
  """Body-frame vx/vy/yaw-rate commands; lower-body joint selection is explicit."""

  def __init__(
    self, kinematics: TorchKinematics, config: VelocityConfig | None = None, default_joint_position: torch.Tensor | None = None
  ):
    super().__init__(kinematics, default_joint_position)
    self.config = VelocityConfig() if config is None else config

  def forward(
    self, state: torch.Tensor, goal: VelocityGoal, contact: ContactUpdate | None = None, last_action: torch.Tensor | None = None
  ) -> TaskSignals:
    k = self.kinematics(state)
    obs = self.proprioception(state, last_action)
    c, command = self.config, goal.twist
    if command.shape != (*state.shape[:-1], 3):
      raise ValueError("Velocity twist must match state batch [..., 3]")
    linear, angular = obs["base_lin_vel"], obs["base_ang_vel"]
    speed = torch.linalg.vector_norm(command[..., :2], dim=-1)
    std = c.linear_std * (1 + c.std_scale * (speed - c.std_knee).clamp_min(0))
    linear_error = (linear[..., :2] - command[..., :2]).square().sum(-1) + c.z_penalty * linear[..., 2].square()
    angular_error = (angular[..., 2] - command[..., 2]).square() + 0.05 * angular[..., :2].square().sum(-1)
    height = state[..., 2] - k.foot_position[..., 2].min(-1).values
    height_target = torch.broadcast_to(goal.height, state.shape[:-1])
    direction = command[..., :2] / speed.clamp_min(1e-6).unsqueeze(-1)
    achieved = (linear[..., :2] * direction).sum(-1)
    shortfall = (speed * c.cap_fraction - achieved).clamp_min(0) * (speed > c.command_threshold)
    obs.update(
      {
        "command_twist": command,
        "command_height": height_target.unsqueeze(-1),
        "base_height": height.unsqueeze(-1),
        "foot_height": k.foot_position[..., 2],
        "lower_joint_pos": obs["joint_pos"][..., :15],
        "lower_joint_vel": obs["joint_vel"][..., :15],
      }
    )
    if goal.phase is not None:
      clock = torch.stack((goal.phase.sin(), goal.phase.cos()), dim=-1)
      standing = torch.linalg.vector_norm(command, dim=-1) < c.command_threshold
      obs["phase"] = torch.where(standing.unsqueeze(-1), torch.zeros_like(clock), clock)
    costs = {"velocity_shortfall": shortfall, "tilt": obs["projected_gravity"][..., :2].square().sum(-1)}
    events = {}
    if contact is not None:
      obs.update(self.contact_observations(contact))
      speed_sq = k.foot_linear_velocity[..., :2].square().sum(-1)
      costs["slip"] = (speed_sq * contact.memory.contact).sum(-1)
      costs["soft_slip"] = (speed_sq * contact.soft).sum(-1)
      events = {"contact_onset": contact.onset, "liftoff": contact.liftoff}
    return TaskSignals(
      obs,
      {
        "linear_velocity": torch.exp(-linear_error / std.square()),
        "angular_velocity": torch.exp(-angular_error / c.angular_std**2),
        "height": torch.exp(-(height - height_target).square() / c.height_std**2),
      },
      costs,
      events,
    )


@dataclass(frozen=True)
class FootstepConfig:
  phase: FootstepPhaseCfg = DEFAULT_PHASE_CFG
  position_scale: float = 0.05
  yaw_scale: float = 0.2
  swing_position_std: float = 0.2
  swing_yaw_std: float = 0.1
  swing_progress_power: float = 2.0
  clearance: float = 0.11
  air_time_scale: float = 0.4
  slip_linear: bool = False
  yaw_rate_scale_deg_s: float = 1.0

  def __post_init__(self) -> None:
    _positive(**{name: value for name, value in vars(self).items() if type(value) is float})
    if self.swing_progress_power < 1:
      raise ValueError("Swing progress power must be >= 1")


@dataclass(frozen=True)
class FootstepGoal:
  targets: torch.Tensor
  target_ids: torch.Tensor
  phase: torch.Tensor
  frequency: torch.Tensor
  ground_height: torch.Tensor


class FootstepMemory(NamedTuple):
  target_ids: torch.Tensor
  saw_air: torch.Tensor
  landed: torch.Tensor
  previous_contact: torch.Tensor

  @classmethod
  def zeros(cls, reference: torch.Tensor) -> FootstepMemory:
    shape = reference.shape[:-1] if reference.shape[-2:] == (2, 3) else reference.shape
    ids = torch.full(shape, -1, dtype=torch.long, device=reference.device)
    zero = torch.zeros_like(ids, dtype=torch.bool)
    return cls(ids, zero, zero.clone(), torch.ones_like(zero))

  def reset(self, mask: torch.Tensor) -> FootstepMemory:
    mask = mask.unsqueeze(-1)
    return FootstepMemory(
      torch.where(mask, -torch.ones_like(self.target_ids), self.target_ids),
      torch.where(mask, False, self.saw_air),
      torch.where(mask, False, self.landed),
      torch.where(mask, True, self.previous_contact),
    )


class FootstepOutput(NamedTuple):
  signals: TaskSignals
  memory: FootstepMemory


class FootstepTrackingAdapter(TaskAdapter):
  def __init__(
    self, kinematics: TorchKinematics, config: FootstepConfig | None = None, default_joint_position: torch.Tensor | None = None
  ):
    super().__init__(kinematics, default_joint_position)
    self.config = FootstepConfig() if config is None else config

  def forward(
    self,
    state: torch.Tensor,
    goal: FootstepGoal,
    contact: ContactUpdate,
    previous: FootstepMemory,
    dt: float | torch.Tensor,
    last_action: torch.Tensor | None = None,
  ) -> FootstepOutput:
    k, c = self.kinematics(state), self.config
    if goal.targets.shape != (*state.shape[:-1], 2, 3):
      raise ValueError("Footstep targets must be [..., 2, 3] world XY/yaw in left/right order")
    if goal.target_ids.shape != goal.targets.shape[:-1] or goal.target_ids.dtype != torch.long or (goal.target_ids < 0).any():
      raise ValueError("target_ids must be nonnegative int64 [..., 2]")
    if previous.target_ids.shape != goal.target_ids.shape or contact.memory.contact.shape != goal.target_ids.shape:
      raise ValueError("Footstep/contact memories must match goal batch")
    dt = torch.as_tensor(dt, dtype=state.dtype, device=state.device)
    if not torch.isfinite(dt).all() or (dt <= 0).any():
      raise ValueError("dt must be finite and positive")
    dt = torch.broadcast_to(dt, state.shape[:-1])
    phase = torch.broadcast_to(goal.phase, state.shape[:-1])
    frequency = torch.broadcast_to(goal.frequency, state.shape[:-1])
    if not torch.isfinite(frequency).all() or (frequency < 0).any():
      raise ValueError("frequency must be finite and nonnegative")
    stance, progress = _phase_windows(phase, c.phase)
    moving = frequency > 0
    stance = stance | ~moving.unsqueeze(-1)
    hard, soft = contact.memory.contact, contact.soft
    distance = torch.linalg.vector_norm(k.foot_position[..., :2] - goal.targets[..., :2], dim=-1)
    yaw_error = wrap(yaw(k.foot_quaternion) - goal.targets[..., 2])
    accuracy = 0.5 * distance / c.position_scale + 0.5 * yaw_error.abs() / c.yaw_scale
    changed = (previous.target_ids != goal.target_ids) | ~moving.unsqueeze(-1)
    saw_air = (previous.saw_air & ~changed) | (~hard & ~stance)
    landed = previous.landed & ~changed
    first_contact = hard & ~previous.previous_contact & saw_air & ~landed
    memory = FootstepMemory(goal.target_ids.clone(), saw_air, landed | first_contact, hard.clone())
    lift = torch.sin(math.pi * progress).square()
    height = k.foot_position[..., 2] - torch.broadcast_to(goal.ground_height, goal.targets.shape[:-1])
    speed = torch.linalg.vector_norm(k.foot_linear_velocity[..., :2], dim=-1)
    slip = speed if c.slip_linear else speed.square()
    matched = (stance == hard).all(-1) & (stance == (contact.memory.contact_time > 0)).all(-1)
    held = torch.where(stance, contact.memory.contact_time, contact.memory.air_time).min(-1).values
    swing_fraction = phase.new_tensor([1 - f for f in c.phase.stance_fractions])
    credit = torch.minimum((held.unsqueeze(-1) * frequency.unsqueeze(-1) / swing_fraction).clamp(0, 1), progress)
    ramp = progress.pow(c.swing_progress_power) * ~stance
    support = (hard & stance).any(-1)
    soft_support = 1 - (1 - soft * stance).prod(-1)
    inv = conjugate(heading(unit(state[..., 3:7]))).unsqueeze(-2).expand_as(k.foot_quaternion)
    target_xyz = torch.cat((goal.targets[..., :2], k.foot_position[..., 2:3]), dim=-1)
    local_target = rotate(inv, target_xyz - state[..., :3].unsqueeze(-2))
    target_heading = wrap(goal.targets[..., 2] - yaw(unit(state[..., 3:7])).unsqueeze(-1))
    obs = self.proprioception(state, last_action)
    obs.update(self.contact_observations(contact))
    obs.update(
      {
        "footstep_targets_local": torch.cat((local_target[..., :2], target_heading.unsqueeze(-1)), dim=-1),
        "phase": torch.stack((phase.sin(), phase.cos()), dim=-1),
        "frequency": frequency.unsqueeze(-1),
        "stance": stance.to(state),
        "swing_progress": progress,
        "foot_height": height,
        "foot_position_error": distance,
        "foot_yaw_error": yaw_error,
        "joint_pos_absolute": state[..., STATE_FIELDS["joint_position"]],
      }
    )
    position_score = (torch.exp(-(distance / c.swing_position_std).square()) * ramp).sum(-1)
    yaw_score = (torch.exp(-(yaw_error / c.swing_yaw_std).square()) * ramp).sum(-1)
    rewards = {
      "swing_position": position_score * support,
      "swing_yaw": yaw_score * support,
      "soft_swing_position": position_score * soft_support,
      "soft_swing_yaw": yaw_score * soft_support,
      "schedule": matched.to(state) * moving,
      "soft_schedule": torch.where(stance, soft, 1 - soft).prod(-1) * moving,
      "air_time": c.air_time_scale * (credit * ~stance).sum(-1) * matched * (stance.sum(-1) == 1) * moving,
    }
    costs = {
      "landing": (accuracy * first_contact).sum(-1) / dt,
      "footprint_accuracy": accuracy.mean(-1),
      "clearance": ((height - c.clearance * lift * ~stance).abs() / c.clearance).sum(-1),
      "swing_contact": (hard * ~stance * lift).sum(-1),
      "soft_swing_contact": (soft * ~stance * lift).sum(-1),
      "slip": (slip * hard).sum(-1) * moving,
      "soft_slip": (slip * soft).sum(-1) * moving,
      "feet_contact_yaw_rate": (k.foot_angular_velocity[..., 2].abs() * hard).sum(-1) / math.radians(c.yaw_rate_scale_deg_s),
      "soft_feet_contact_yaw_rate": (k.foot_angular_velocity[..., 2].abs() * soft).sum(-1)
      / math.radians(c.yaw_rate_scale_deg_s),
      "slip_still": torch.linalg.vector_norm(k.foot_linear_velocity, dim=-1).sum(-1) * ~moving,
      "stationary": (~(contact.memory.contact_time > 0)).to(state).sum(-1) * ~moving,
    }
    return FootstepOutput(
      TaskSignals(
        obs,
        rewards,
        costs,
        {
          "landing": first_contact,
          "contact_onset": contact.onset,
          "liftoff": contact.liftoff,
          "schedule_matched": matched,
        },
      ),
      memory,
    )
