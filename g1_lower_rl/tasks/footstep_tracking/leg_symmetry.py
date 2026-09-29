"""Low-frequency mirror bias of the six left/right leg joint pairs."""

import math

import torch

from g1_lower_rl.footsteps.tensor_manager import STANDING, WALKING

LEG_JOINTS = tuple(f"{side}_{joint}_joint" for side in ("left", "right")
  for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"))


class LegSymmetryEMA:
  def __init__(self, cfg, env):
    asset_cfg = cfg.params["asset_cfg"]
    asset = env.scene[asset_cfg.name]
    names = [asset.joint_names[index] for index in asset_cfg.joint_ids]
    if len(names) != 12 or set(names) != set(LEG_JOINTS):
      raise ValueError("Leg symmetry must select exactly the twelve leg joints")
    self.joint_ids = [asset.joint_names.index(name) for name in LEG_JOINTS]
    self.signs = torch.tensor([1., -1., -1., 1., 1., -1.], device=env.device)
    self.filtered_error = torch.zeros(env.num_envs, 6, device=env.device)
    self.eligible = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.elapsed = torch.zeros(env.num_envs, device=env.device)
    self.last_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    self.last_direction = torch.zeros(env.num_envs, device=env.device)
    self.last_heading = torch.zeros(env.num_envs, device=env.device)
    self.last_mode = torch.full_like(self.last_step, -1)

  def reset(self, env_ids=None):
    selected = slice(None) if env_ids is None else env_ids
    self.filtered_error[selected] = 0.
    self.eligible[selected] = False
    self.elapsed[selected] = 0.
    self.last_step[selected] = -1
    self.last_direction[selected] = 0.
    self.last_heading[selected] = 0.
    self.last_mode[selected] = -1

  def __call__(self, env, asset_cfg, command_name="footsteps", cycles=1., standing_tau_s=1.,
               max_tau_s=2., deadband=math.radians(3), direction_tolerance=math.radians(15),
               target_yaw_tolerance=math.radians(30), standing_fore_aft_tolerance=.05,
               ramp_s=1.):
    if not all(math.isfinite(value) and value > 0 for value in (
        cycles, standing_tau_s, max_tau_s, env.step_dt, direction_tolerance,
        target_yaw_tolerance, standing_fore_aft_tolerance, ramp_s)):
      raise ValueError("Symmetry time constants and gate tolerances must be finite and positive")
    if not math.isfinite(deadband) or deadband < 0:
      raise ValueError("Symmetry deadband must be finite and nonnegative")
    command = env.command_manager.get_term(command_name)
    frequency = command.reward_state.frequency
    targets = command.reward_state.targets_w
    intent = command.symmetry_state
    direction, heading, mode = intent["direction"], intent["heading"], intent["mode"]
    wrap = lambda angle: torch.atan2(angle.sin(), angle.cos())
    forward = wrap(direction - heading).abs() <= direction_tolerance
    yaw_symmetric = wrap(targets[..., 2].sum(-1) - 2 * heading).abs() <= target_yaw_tolerance
    displacement = targets[:, 0, :2] - targets[:, 1, :2]
    fore_aft = displacement[:, 0] * heading.cos() + displacement[:, 1] * heading.sin()
    standing = (frequency == 0) & (mode == STANDING) & (fore_aft.abs() <= standing_fore_aft_tolerance)
    walking = (frequency > 0) & (mode == WALKING) & forward
    eligible = (standing | walking) & yaw_symmetric
    positions = env.scene[asset_cfg.name].data.joint_pos[:, self.joint_ids]
    error = positions[:, :6] - self.signs * positions[:, 6:]
    tau = (cycles / frequency.clamp_min(1e-6)).clamp_max(max_tau_s)
    tau = torch.where(frequency > 0, tau, standing_tau_s)
    alpha = -torch.expm1(-env.step_dt / tau)
    update = self.last_step != env.common_step_counter
    changed = ((wrap(direction - self.last_direction).abs() > 1e-5)
               | (wrap(heading - self.last_heading).abs() > 1e-5) | (mode != self.last_mode))
    fresh = (~self.eligible | changed | (self.last_step < 0)) & eligible
    filtered = self.filtered_error + alpha[:, None] * (error - self.filtered_error)
    filtered = torch.where(fresh[:, None], error, filtered)
    filtered = torch.where(eligible[:, None], filtered, 0.)
    elapsed = torch.where(fresh | ~eligible, 0., self.elapsed + env.step_dt)
    self.filtered_error.copy_(torch.where(update[:, None], filtered, self.filtered_error))
    self.elapsed.copy_(torch.where(update, elapsed, self.elapsed))
    self.eligible.copy_(torch.where(update, eligible, self.eligible))
    self.last_direction.copy_(torch.where(update, direction, self.last_direction))
    self.last_heading.copy_(torch.where(update, heading, self.last_heading))
    self.last_mode.copy_(torch.where(update, mode, self.last_mode))
    self.last_step.masked_fill_(update, env.common_step_counter)
    cost = (self.filtered_error.abs() - deadband).clamp_min(0.).square().mean(-1)
    return cost * (self.elapsed / ramp_s).clamp_max(1.) * self.eligible