"""Single-step completion, holding time, and episode metrics."""

from dataclasses import dataclass
import math

import torch

from g1_lower_rl.footsteps.step_episode import StepEpisodeManager
from g1_lower_rl.footsteps.tensor_manager import STANDING
from g1_lower_rl.tasks.footstep_tracking.commands import FootstepCommand, FootstepCommandCfg, foot_poses


def update_standing_time(previous, stable, dt):
  return torch.where(stable, previous + dt, torch.zeros_like(previous))


class StepEpisodeCommand(FootstepCommand):
  cfg: "StepEpisodeCommandCfg"
  manager_type = StepEpisodeManager

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.standing_time = torch.zeros(self.num_envs, device=self.device, dtype=torch.float64)
    self.hold_started_at = torch.full_like(self.standing_time, -1.)
    self.stable_time = torch.zeros_like(self.standing_time)
    self.longest_hold = torch.zeros_like(self.standing_time)
    self.completed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
    self.final_distance = torch.zeros(self.num_envs, device=self.device)
    self.final_yaw = torch.zeros_like(self.final_distance)

  def finish_step(self):
    if self.last_step == self._env.common_step_counter:
      return self.failed
    failed = super().finish_step()
    data = self.robot.data
    contact = self._env.scene[self.cfg.sensor_name].data.found > 0
    foot_speed = data.site_lin_vel_w[:, self.site_ids].norm(dim=-1).amax(dim=-1)
    state = self.batch.state
    holding = state["episode_started"] & (state["mode"] == STANDING) & ~failed
    self.hold_started_at.copy_(torch.where(holding & (self.hold_started_at < 0),
      state["elapsed"], self.hold_started_at))
    self.standing_time.copy_(torch.where(self.hold_started_at >= 0,
      state["elapsed"] - self.hold_started_at, 0.))
    stable = holding.clone()
    stable &= contact.all(-1) & (foot_speed < .05)
    stable &= (data.root_link_lin_vel_w.norm(dim=-1) < .1) & (data.root_link_ang_vel_w.norm(dim=-1) < .3)
    self.stable_time.copy_(update_standing_time(self.stable_time, stable, self._env.step_dt))
    self.longest_hold.copy_(torch.maximum(self.longest_hold, self.stable_time))
    self.completed.copy_((self.standing_time >= self.required_hold_s - 1e-8) & ~failed)
    actual = foot_poses(data, self.site_ids)
    target = self.batch.state["terminal_feet"]
    self.final_distance.copy_((actual[..., :2] - target[..., :2]).norm(dim=-1).amax(-1))
    yaw = actual[..., 2] - target[..., 2]
    self.final_yaw.copy_(torch.atan2(yaw.sin(), yaw.cos()).abs().amax(-1))
    return failed

  @property
  def required_hold_s(self):
    if self.cfg.sample_hold:
      return self.batch.state["hold_duration"]
    return torch.full_like(self.standing_time, self.cfg.final_stand_s)

  def reset(self, env_ids):
    selected = slice(None) if env_ids is None else env_ids
    if isinstance(env_ids, torch.Tensor) and env_ids.numel() == 0:
      return {}
    state = self.batch.state
    distance, yaw = self.final_distance, self.final_yaw
    accurate = (distance <= .05) & (yaw <= math.radians(10))
    metrics = (("hold_complete", self.completed), ("success", self.completed & accurate),
      ("max_foot_error_cm", distance * 100), ("longest_hold_s", self.longest_hold))
    result = {}
    for side, name in ((0, "left"), (1, "right")):
      for aligned, ending in ((False, "single"), (True, "aligned")):
        mask = ((state["episode_side"] == side) & (state["episode_aligned"] == aligned)
          & (self._env.episode_length_buf > 0))[selected]
        episode_count = mask.sum()
        count = episode_count.clamp_min(1)
        prefix = f"step/{name}_{ending}"
        result[f"{prefix}/episodes"] = episode_count.item()
        for metric, value in metrics:
          result[f"{prefix}/{metric}"] = ((value[selected] * mask).sum() / count).item()
    result.update(super().reset(env_ids))
    self.standing_time[selected] = 0.
    self.hold_started_at[selected] = -1.
    self.stable_time[selected] = 0.
    self.longest_hold[selected] = 0.
    self.completed[selected] = False
    return result


@dataclass(kw_only=True)
class StepEpisodeCommandCfg(FootstepCommandCfg):
  final_stand_s: float = 5.0
  sample_hold: bool = False

  def build(self, env):
    return StepEpisodeCommand(self, env)


def step_episode_complete(env):
  return env.command_manager.get_term("footsteps").completed