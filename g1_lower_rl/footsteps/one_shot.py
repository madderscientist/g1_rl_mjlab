"""单环境预览用的一次或两次落脚计划，保持四槽观测接口"""

from dataclasses import replace
import math

import torch

from g1_lower_rl.footsteps.tensor_manager import TensorFootstepManager, STARTING, WALKING, STOPPING, STANDING


class OneShotFootstepManager(TensorFootstepManager):
  def __init__(self, cfg, source, num_envs, device, seed=0, *, compiled=False):
    if num_envs != 1:
      raise ValueError("Manual preview requires one environment")
    if source.automatic_commands or source.automatic_restart or source.initial_standing_probability != 1.:
      raise ValueError("Manual preview requires stationary reset and disabled automatic commands")
    self.motion_start = 0.
    self.motion_frequency = 1.2
    self.cruise_duration = 0.
    self.executions = 0
    self.base_cfg = cfg
    super().__init__(cfg, source, num_envs, device, seed, compiled=False)

  def _sample(self, previous, side, uniform):
    return self._choose(self.state["terminal_feet"], side)

  def _source_advance(self, uniform):
    pass

  def _apply_request(self, uniform):
    pass

  def start_forward(self, distance, frequency, first_side, *, close_stance=True):
    """从冻结的并列双脚目标前移，行走过程中不修改实际机器人状态"""
    if first_side not in (0, 1):
      raise ValueError("First foot must be left or right")
    if not math.isfinite(distance) or not 0. <= distance <= .35:
      raise ValueError("Distance must be between 0 and 0.35 m")
    if not math.isfinite(frequency) or not self.cfg.frequency_range[0] <= frequency <= self.cfg.frequency_range[1]:
      raise ValueError("Frequency is outside the configured range")
    state = self.state
    if state["mode"].item() != STANDING or state["failed"].item():
      raise ValueError("Both feet must be standing before starting")
    feet = state["targets"].clone()
    heading = torch.atan2(feet[..., 2].sin().sum(-1), feet[..., 2].cos().sum(-1))
    cosine, sine = heading.cos(), heading.sin()
    delta = feet[:, 0, :2] - feet[:, 1, :2]
    fore_aft = delta[:, 0] * cosine + delta[:, 1] * sine
    width = -delta[:, 0] * sine + delta[:, 1] * cosine
    yaw_delta = feet[:, 0, 2] - feet[:, 1, 2]
    if (fore_aft.abs() > .03).any() or (width < .12).any() or (width > .36).any():
      raise ValueError("Reset to an aligned stance before starting")
    if torch.atan2(yaw_delta.sin(), yaw_delta.cos()).abs().max() > math.radians(10):
      raise ValueError("Initial foot headings must be aligned")
    if torch.hypot(width, width.new_tensor(distance)).max() > .48:
      raise ValueError("Requested footprint span exceeds 0.48 m")
    phase = self.cfg.phase
    start_phase = (phase.left_stance_phase, phase.right_stance_phase)[1 - first_side]
    first_touch = (phase.left_stance_phase, phase.right_stance_phase)[first_side]
    while first_touch <= start_phase:
      first_touch += 2 * math.pi
    second_touch = (phase.left_stance_phase, phase.right_stance_phase)[1 - first_side]
    while second_touch <= first_touch:
      second_touch += 2 * math.pi
    stop_phase = (second_touch if close_stance else first_touch) + .5 * phase.contact_half_width
    duration = (stop_phase - start_phase) / (2 * math.pi * frequency)
    ramp_time = .5 * (self.base_cfg.start_duration_s + self.base_cfg.stop_duration_s)
    if close_stance and duration < ramp_time:
      raise ValueError("Frequency is too high for the start/stop ramps")
    ramp_scale = min(1., duration / ramp_time)
    motion_cfg = replace(self.base_cfg, start_duration_s=self.base_cfg.start_duration_s * ramp_scale,
      stop_duration_s=self.base_cfg.stop_duration_s * ramp_scale)
    cruise = max(0., duration - ramp_time * ramp_scale)
    center = feet[:, :, :2].mean(1) + distance * torch.stack((cosine, sine), dim=-1)
    lateral = torch.stack((-sine, cosine), dim=-1) * width[:, None] / 2
    terminal = feet.clone()
    if close_stance:
      terminal[:, 0, :2], terminal[:, 1, :2] = center + lateral, center - lateral
      terminal[:, :, 2] = heading[:, None]
    else:
      terminal[:, first_side, :2] += distance * torch.stack((cosine, sine), dim=-1)
    self.cfg = motion_cfg
    self.motion_start = state["elapsed"].item()
    self.motion_frequency = frequency
    self.cruise_duration = cruise
    next_id = state["next_id"].item()
    sides = [(first_side + slot) % 2 for slot in range(4)]
    state["terminal_feet"].copy_(terminal)
    state["supports"].copy_(feet)
    state["queue"].copy_(terminal[:, sides])
    state["queue_ids"].copy_(torch.arange(next_id, next_id + 4, device=self.device)[None])
    state["next_id"].fill_(next_id + 4)
    state["first_side"].fill_(first_side)
    state["phase"].fill_(start_phase)
    state["heading"].copy_(heading)
    state["direction"].copy_(heading)
    state["anchor"].copy_(feet[:, 1 - first_side])
    state["pending_anchor"].fill_(-1)
    state["stop_target_id"].fill_(next_id + int(close_stance))
    state["stop_phase"].fill_(stop_phase)
    state["stop_ramp_start"].fill_(self.motion_start + self.cfg.start_duration_s + cruise)
    state["stop_initial_frequency"].fill_(frequency)
    state["stop_wait"].fill_(-1)
    state["start_progress"].zero_()
    state["request_frequency"].fill_(frequency)
    state["request_walking"].fill_(False)
    state["mode"].fill_(STARTING)
    state["frequency"].copy_(self._stop_frequency())
    self.executions += 1
    self._publish()

  def _stop_frequency(self):
    """对起步、恒频、收步曲线积分，使用下一控制拍的平均频率"""
    start, stop = self.cfg.start_duration_s, self.cfg.stop_duration_s

    def integral(time):
      rising = time.clamp(0., start)
      cruise = (time - start).clamp(0., self.cruise_duration)
      falling = (time - start - self.cruise_duration).clamp(0., stop)
      return self.motion_frequency * (rising.square() / (2 * start) + cruise + falling - falling.square() / (2 * stop))

    elapsed = self.state["elapsed"] - self.motion_start
    return (integral(elapsed + self.cfg.control_dt) - integral(elapsed)) / self.cfg.control_dt

  def _advance(self, feet, contact):
    super()._advance(feet, contact)
    state = self.state
    moving = (state["mode"] == STARTING) | (state["mode"] == WALKING) | (state["mode"] == STOPPING)
    stopping = moving & (state["elapsed"] >= state["stop_ramp_start"])
    self._put("mode", stopping, STOPPING)
    self._put("frequency", moving, self._stop_frequency())
    self._publish()