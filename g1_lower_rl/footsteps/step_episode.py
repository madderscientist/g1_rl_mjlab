"""批量站立、一次迈步、站立任务的设备驻留脚步规划"""

import math

import torch

from g1_lower_rl.footsteps.tensor_manager import TensorFootstepManager, STARTING, WALKING, STOPPING, STANDING


class StepEpisodeManager(TensorFootstepManager):
  initial_stand_s = 1.0
  aligned_probability = 0.3

  def __init__(self, cfg, source, num_envs, device, seed=0, *, compiled=False):
    if source.initial_standing_probability != 1. or source.automatic_commands or source.automatic_restart:
      raise ValueError("Step episodes require standing resets without automatic commands")
    if .5 * (cfg.start_duration_s + cfg.stop_duration_s) > .5 / source.frequency_range[1]:
      raise ValueError("Start/stop ramps must fit one footstep at the maximum frequency")
    super().__init__(cfg, source, num_envs, device, seed, compiled=compiled)
    for name, shape, dtype in (
      ("episode_feet", (2, 3), torch.float64), ("episode_side", (), torch.long),
      ("episode_aligned", (), torch.bool), ("episode_started", (), torch.bool),
      ("motion_frequency", (), torch.float64), ("motion_start", (), torch.float64),
      ("hold_duration", (), torch.float64),
      ("cruise_duration", (), torch.float64), ("planned_landings", (), torch.long),
    ):
      self.state[name] = torch.zeros((num_envs, *shape), device=device, dtype=dtype)

  def _reset(self, feet, mask):
    super()._reset(feet, mask)
    uniform = self._uniform(mask)
    lower, upper = self.source_cfg.frequency_range
    for name, value in (
      ("episode_feet", feet), ("episode_side", (uniform[:, 0] < .5).long()),
      ("episode_aligned", uniform[:, 1] < self.aligned_probability), ("episode_started", False),
      ("motion_frequency", lower + uniform[:, 2] * (upper - lower)),
      ("hold_duration", torch.where(uniform[:, 3] < .2, 5., 1. + uniform[:, 4])),
      ("motion_start", 0.), ("cruise_duration", 0.), ("planned_landings", 0),
    ):
      self._put(name, mask, value)

  def _sample(self, previous, side, uniform):
    return self._choose(self.state["terminal_feet"], side)

  def _source_advance(self, uniform):
    pass

  def _apply_request(self, uniform):
    pass

  def _start(self, mask):
    state, cfg = self.state, self.cfg
    uniform = self._uniform(mask)
    feet, side = state["episode_feet"], state["episode_side"]
    support = self._choose(feet, 1 - side)
    target = self.sampler.sample(support, side, state["direction"], state["heading"], uniform[:, :3],
      self.sampler_parameters, self.yaw_noise_range)
    selected = self.sides[None, :] == side[:, None]
    terminal = torch.where(selected[:, :, None], target[:, None, :], feet)
    forward = torch.stack((state["heading"].cos(), state["heading"].sin()), dim=-1)
    shift = ((target[:, :2] - support[:, :2]) * forward).sum(-1, keepdim=True) * forward
    closing = support.clone()
    closing[:, :2] += shift
    closing[:, 2] = state["heading"]
    terminal = torch.where((~selected & state["episode_aligned"][:, None])[:, :, None], closing[:, None, :], terminal)
    start_phase = self.centers[1 - side]
    first_touch = self.centers[side]
    first_touch += torch.ceil((start_phase - first_touch) / (2 * math.pi)).clamp_min(0.) * (2 * math.pi)
    stop_phase = first_touch + state["episode_aligned"] * math.pi + .5 * cfg.phase.contact_half_width
    duration = (stop_phase - start_phase) / (2 * math.pi * state["motion_frequency"])
    cruise = duration - .5 * (cfg.start_duration_s + cfg.stop_duration_s)
    queue_sides = (side[:, None] + torch.arange(4, device=self.device)) % 2
    next_id = state["next_id"].clone()
    for name, value in (
      ("terminal_feet", terminal), ("queue", terminal.gather(1, queue_sides[:, :, None].expand(-1, -1, 3))),
      ("queue_ids", next_id[:, None] + torch.arange(4, device=self.device)), ("next_id", next_id + 4),
      ("first_side", side), ("phase", start_phase), ("anchor", support), ("pending_anchor", -1),
      ("stop_target_id", next_id + state["episode_aligned"].long()), ("stop_phase", stop_phase),
      ("motion_start", state["elapsed"]), ("cruise_duration", cruise),
      ("stop_ramp_start", state["elapsed"] + cfg.start_duration_s + cruise),
      ("stop_initial_frequency", state["motion_frequency"]), ("stop_wait", -1.),
      ("start_progress", 0.), ("request_frequency", state["motion_frequency"]),
      ("request_walking", False), ("mode", STARTING), ("episode_started", True),
    ):
      self._put(name, mask, value)
    self._put("frequency", mask, self._stop_frequency())

  def _stop_frequency(self):
    state, cfg = self.state, self.cfg

    def integral(time):
      rising = time.clamp(0., cfg.start_duration_s)
      cruise = torch.minimum(time.sub(cfg.start_duration_s).clamp_min(0.), state["cruise_duration"])
      falling = (time - cfg.start_duration_s - state["cruise_duration"]).clamp(0., cfg.stop_duration_s)
      return state["motion_frequency"] * (rising.square() / (2 * cfg.start_duration_s) + cruise
        + falling - falling.square() / (2 * cfg.stop_duration_s))

    elapsed = state["elapsed"] - state["motion_start"]
    return (integral(elapsed + cfg.control_dt) - integral(elapsed)) / cfg.control_dt

  def _advance(self, feet, contact):
    previous_id = self.state["queue_ids"][:, 0].clone()
    super()._advance(feet, contact)
    state = self.state
    state["planned_landings"].add_((state["queue_ids"][:, 0] != previous_id).long())
    moving = (state["mode"] == STARTING) | (state["mode"] == WALKING) | (state["mode"] == STOPPING)
    self._put("mode", moving & (state["elapsed"] >= state["stop_ramp_start"]), STOPPING)
    self._put("frequency", moving, self._stop_frequency())
    start = ~state["episode_started"] & ~state["failed"] & (state["mode"] == STANDING)
    self._start(start & (state["elapsed"] >= self.initial_stand_s - 1e-10))
    self._publish()