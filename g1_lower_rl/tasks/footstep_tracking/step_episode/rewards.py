"""Single-step failure, landing bonus, and standing penalties."""

import torch

from g1_lower_rl.tasks.footstep_tracking.commands import foot_poses
from g1_lower_rl.tasks.footstep_tracking.reward_math import phase_windows


def step_failure_cost(env):
  completed = env.command_manager.get_term("footsteps").completed
  failed = torch.zeros_like(completed)
  manager = env.termination_manager
  for name in manager.active_terms:
    if name == "step_complete":
      continue
    active = manager.get_term(name)
    if manager.get_term_cfg(name).time_out:
      active = active & ~completed
    failed |= active
  return failed.to(torch.float32) / env.step_dt


class StepLandingBonus:
  """Pay each planned foot's first actual landing within the episode budget."""

  def __init__(self, cfg, env):
    self.saw_air = torch.zeros((env.num_envs, 2), dtype=torch.bool, device=env.device)
    self.previous_contact = torch.ones_like(self.saw_air)
    self.claimed = torch.zeros_like(self.saw_air)
    self.value = torch.zeros(env.num_envs, device=env.device)
    self.last_step = -1

  def reset(self, env_ids=None):
    selected = slice(None) if env_ids is None else env_ids
    self.saw_air[selected] = False
    self.previous_contact[selected] = True
    self.claimed[selected] = False
    self.value[selected] = 0.

  def __call__(self, env, asset_cfg, command_name="footsteps", sensor_name="feet_ground_contact",
               position_std=.1, yaw_std=.2, minimum_lift=.02):
    if self.last_step == env.common_step_counter:
      return self.value
    command = env.command_manager.get_term(command_name)
    state = command.reward_state
    episode = command.batch.state
    data = env.scene[asset_cfg.name].data
    contact = env.scene[sensor_name].data.found > 0
    stance, _ = phase_windows(state.phase, phase_cfg=command.cfg.manager.phase)
    sides = torch.arange(2, device=env.device)
    eligible = episode["episode_started"][:, None] & (
      episode["episode_aligned"][:, None] | (sides == episode["episode_side"][:, None]))
    height = data.site_pos_w[:, asset_cfg.site_ids, 2] - state.ground_height
    self.saw_air |= eligible & (state.frequency > 0)[:, None] & ~stance & ~contact & (height >= minimum_lift)
    first_contact = eligible & contact & ~self.previous_contact & self.saw_air & ~self.claimed
    actual = foot_poses(data, asset_cfg.site_ids)
    distance = (actual[..., :2] - state.targets_w[..., :2]).norm(dim=-1)
    yaw = actual[..., 2] - state.targets_w[..., 2]
    yaw = torch.atan2(yaw.sin(), yaw.cos())
    score = torch.exp(-(distance / position_std).square() - (yaw / yaw_std).square())
    count = 1 + episode["episode_aligned"].to(score.dtype)
    self.value.copy_((score * first_contact * contact.flip(-1)).sum(-1) / count / env.step_dt)
    self.claimed |= first_contact
    self.previous_contact.copy_(contact)
    self.last_step = env.common_step_counter
    return self.value


def standing_penalty_gate(env, command_name="footsteps", ramp_s=.3):
  command = env.command_manager.get_term(command_name)
  age = torch.where(command.batch.state["episode_started"], command.standing_time,
    env.episode_length_buf * env.step_dt)
  return (command.reward_state.frequency == 0) * (age / ramp_s).clamp(0., 1.)


def standing_joint_velocity_cost(env, asset_cfg, command_name="footsteps"):
  velocity = env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]
  return velocity.square().mean(-1).clamp_max(4.) * standing_penalty_gate(env, command_name)


def standing_action_change_cost(env, command_name="footsteps"):
  delta = env.action_manager.action - env.action_manager.prev_action
  return delta.square().sum(-1).clamp_max(25.) * standing_penalty_gate(env, command_name)