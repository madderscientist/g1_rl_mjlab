"""供并行训练和策略回放使用的设备驻留脚步规划器

合并 NumPy 版 FootstepManager 与 RandomCommandSource 的职责，调度行走意图、推进步态时钟并维护落点
批量状态保留在仿真设备上，避免逐环境 Python 更新和逐拍 CPU/GPU 传输，NumPy 版仍作为独立参考实现

FootstepCommand 在正向运动学后调用 reset，在物理步之后、奖励之前调用 advance
本模块只发布目标，不生成关节动作或奖励；策略读取14维 command，奖励适配器读取 completed_* 快照
"""

from __future__ import annotations

import math

import torch

from g1_lower_rl.footsteps.config import FootstepManagerCfg, RandomCommandCfg
from g1_lower_rl.footsteps.tensor_math import TensorFootstepSampler, from_local, to_local, wrap


WALKING, STARTING, STOPPING, STANDING, FAULT, SETTLING = range(6)


class TensorFootstepManager:
  """以固定形状状态和掩码更新替代逐环境 Python 循环

  所有状态张量首维都是环境维，脚侧0为左、1为右
  位姿采用世界系 (x, y, yaw)，只有 command 输出转换到冻结参考系
  相位使用未取模弧度，频率表示每秒完整左右周期数
  时钟和几何使用 float64 保持事件时序精度，策略命令使用 float32
  """

  def __init__(self, cfg: FootstepManagerCfg, source: RandomCommandCfg, num_envs: int,
               device: str | torch.device, seed: int = 0, *, compiled: bool = False):
    self.cfg, self.source_cfg = cfg, source
    self.device = torch.device(device)
    self.num_envs = num_envs
    self.sampler = TensorFootstepSampler(cfg.sampler)
    if not cfg.frequency_range[0] <= source.frequency_range[0] <= source.frequency_range[1] <= cfg.frequency_range[1]:
      raise ValueError("Source frequency range must fit the manager execution range")
    self.state = {}
    for name, shape, dtype in (
      # supports 保存上次计划落点，targets 保存当前双脚执行目标
      ("phase", (), torch.float64), ("frequency", (), torch.float64), ("mode", (), torch.long),
      ("elapsed", (), torch.float64), ("supports", (2, 3), torch.float64), ("targets", (2, 3), torch.float64),
      # queue 从 first_side 开始，按时间顺序左右交替排列
      ("target_ids", (2,), torch.long), ("queue", (4, 3), torch.float64), ("queue_ids", (4,), torch.long),
      ("first_side", (), torch.long), ("next_id", (), torch.long), ("anchor", (3,), torch.float64),
      ("pending_anchor", (), torch.long), ("contact_count", (), torch.long),
      ("terminal_feet", (2, 3), torch.float64), ("stop_target_id", (), torch.long),
      ("stop_wait", (), torch.float64), ("start_progress", (), torch.float64),
      ("stop_phase", (), torch.float64), ("stop_ramp_start", (), torch.float64), ("stop_initial_frequency", (), torch.float64),
      # 随机源的意图与执行器的实际频率及模式分开保存
      ("direction", (), torch.float64), ("heading", (), torch.float64),
      ("request_frequency", (), torch.float64), ("request_walking", (), torch.bool),
      ("heading_origin", (), torch.float64), ("rate", (), torch.float64), ("rate_remaining", (), torch.float64),
      ("command_at", (), torch.float64), ("restart_at", (), torch.float64),
      # 奖励快照描述刚执行完的一拍，而不是下一拍命令
      ("completed_phase", (), torch.float64), ("completed_frequency", (), torch.float64),
      ("completed_targets", (2, 3), torch.float64), ("completed_ids", (2,), torch.long),
      ("command", (14,), torch.float32), ("future_world", (2, 3), torch.float64),
      ("future_ids", (2,), torch.long), ("future_sides", (2,), torch.long),
      ("failed", (), torch.bool), ("random_counter", (), torch.long),
    ):
      self.state[name] = torch.zeros((num_envs, *shape), device=device, dtype=dtype)
    self.sides = torch.arange(2, device=device)
    self.state["future_sides"].copy_(self.sides)
    self.reset_queue_ids = torch.arange(2, 6, device=device)
    self.centers = torch.tensor([cfg.phase.left_stance_phase, cfg.phase.right_stance_phase], device=device, dtype=torch.float64)
    self.liftoff = torch.tensor(cfg.phase.liftoff_rad, device=device, dtype=torch.float64)
    self.contact_start = torch.tensor(cfg.phase.contact_start_rad, device=device, dtype=torch.float64).remainder(2 * math.pi)
    self.random_keys = (torch.arange(num_envs, device=device, dtype=torch.long) * 0x9E3779B9 + int(seed)) & 0xFFFFFFFF
    self.random_keys = ((self.random_keys >> 16) ^ self.random_keys) * 0x45D9F3B & 0xFFFFFFFF
    self.random_keys = ((self.random_keys >> 16) ^ self.random_keys) * 0x45D9F3B & 0xFFFFFFFF
    self.random_keys = (self.random_keys >> 16) ^ self.random_keys
    self.random_columns = torch.arange(32, device=device, dtype=torch.long) * 0x85EBCA6B
    # 快照捕获必须单独编译，避免状态更新覆盖完成拍奖励需要的旧频率和旧目标
    self._capture = torch.compile(self._capture_completed, fullgraph=True) if compiled else self._capture_completed
    self._step = torch.compile(self._advance, fullgraph=True) if compiled else self._advance
    self._reset_step = torch.compile(self._reset, fullgraph=True) if compiled else self._reset

  def _put(self, name, mask, value):
    """原地更新选中环境，保持对外暴露的张量引用不变"""
    target = self.state[name]
    expanded = mask.reshape((self.num_envs,) + (1,) * (target.ndim - 1))
    target.copy_(torch.where(expanded, value, target))

  def _choose(self, poses, side):
    """为每个环境选取一只脚的位姿，不在 CPU 上读取索引"""
    return poses.gather(1, side[:, None, None].expand(-1, 1, poses.shape[-1])).squeeze(1)

  def _uniform(self, mask):
    """采样32个 (0, 1) 内的值，仅推进掩码选中环境的计数器

    局部重置不会扰动其他环境的随机流；本后端可确定性复现，但不与 NumPy 同种子序列逐值相同
    """
    counter = self.state["random_counter"]
    values = (self.random_keys[:, None] + counter[:, None] * 0x9E3779B9 + self.random_columns) & 0xFFFFFFFF
    values = ((values >> 16) ^ values) * 0x45D9F3B & 0xFFFFFFFF
    values = ((values >> 16) ^ values) * 0x45D9F3B & 0xFFFFFFFF
    values = (values >> 16) ^ values
    counter.add_(mask.to(torch.long))
    return (values.to(torch.float64) + 0.5) / 4294967296.0

  def _sample(self, previous, side, uniform):
    """追加行走落点，停止计划确定后则重复双脚终止落点"""
    state = self.state
    sampled = self.sampler.sample(previous, side, state["direction"], state["heading"], uniform)
    terminal = self._choose(state["terminal_feet"], side)
    return torch.where(((state["mode"] == STOPPING) | (state["mode"] == SETTLING) | (state["mode"] == STANDING))[:, None], terminal, sampled)

  def _fill_queue(self, mask, first_side, hold_first, uniform):
    """构造四个左右交替目标，hold_first 保留已经承诺的首个支撑位置"""
    state = self.state
    previous = self._choose(state["supports"], 1 - first_side)
    queue, ids = [], []
    for slot in range(4):
      side = (first_side + slot) % 2
      pose = self._sample(previous, side, uniform[:, slot * 3:slot * 3 + 3])
      if slot == 0:
        pose = torch.where(hold_first[:, None], self._choose(state["supports"], first_side), pose)
      queue.append(pose)
      ids.append(state["next_id"] + slot)
      previous = pose
    self._put("queue", mask, torch.stack(queue, dim=1))
    self._put("queue_ids", mask, torch.stack(ids, dim=1))
    self._put("next_id", mask, state["next_id"] + 4)
    self._put("first_side", mask, first_side)

  def _directions(self, uniform):
    """相对重置时的双脚平均航向采样移动方向和脚掌朝向"""
    cfg = self.source_cfg
    origin = self.state["heading_origin"]
    return (wrap(origin + cfg.direction_range[0] + uniform[:, 0] * (cfg.direction_range[1] - cfg.direction_range[0])),
            wrap(origin + cfg.foot_heading_range[0] + uniform[:, 1] * (cfg.foot_heading_range[1] - cfg.foot_heading_range[0])))

  def _contacts(self):
    """返回相位计划的支撑窗口，不代表实测接地状态"""
    phase = self.state["phase"][:, None].remainder(2 * math.pi)
    end = self.liftoff.remainder(2 * math.pi)
    start = self.contact_start
    return torch.where(end > start, (phase >= start) & (phase < end), (phase >= start) | (phase < end))

  def _publish(self):
    """在共同参考系中发布左支撑基准、右支撑基准、左下一落点和右下一落点"""
    state = self.state
    indices = (self.sides[None, :] != state["first_side"][:, None]).long()
    future = state["queue"].gather(1, indices[:, :, None].expand(-1, -1, 3))
    state["future_world"].copy_(future)
    state["future_ids"].copy_(state["queue_ids"].gather(1, indices))
    world = torch.cat((state["supports"], future), dim=1)
    local = to_local(world, state["anchor"][:, None, :])
    state["command"].copy_(torch.cat((state["phase"].remainder(2 * math.pi)[:, None],
                                     state["frequency"][:, None], local.flatten(1)), dim=1))
    state["command"].masked_fill_(state["failed"][:, None], 0.0)

  def reset(self, feet, mask):
    """在 sim.forward() 后用形状为 (N, 2, 3) 的实测脚位重置选中环境"""
    self._reset_step(feet.to(torch.float64), mask)

  def _reset(self, feet, mask):
    state = self.state
    uniform = self._uniform(mask)
    heading_origin = torch.atan2(feet[:, :, 2].sin().sum(-1), feet[:, :, 2].cos().sum(-1))
    self._put("heading_origin", mask, heading_origin)
    direction, heading = self._directions(uniform)
    standing = self.source_cfg.initial_standing
    # 用尚未占用的随机数选择两个双支撑中心之一，不改变其他采样的取值位置
    first_side = (uniform[:, 31] < 0.5).long() if standing else torch.zeros_like(state["first_side"])
    for name, value in (
      ("direction", direction), ("heading", heading), ("frequency", 0.0 if standing else self.source_cfg.initial_frequency),
      ("request_frequency", self.source_cfg.initial_frequency), ("request_walking", not standing),
      ("phase", self.centers[1 - first_side] if standing else self.cfg.phase.liftoff_rad[0]),
      ("mode", STANDING if standing else WALKING), ("elapsed", 0.0),
      ("supports", feet), ("targets", feet), ("target_ids", self.sides), ("next_id", 6 if standing else 2),
      ("terminal_feet", feet), ("anchor", self._choose(feet, 1 - first_side)), ("pending_anchor", -1), ("contact_count", 0),
      ("stop_target_id", -1), ("stop_wait", -1.0), ("start_progress", 0.0),
      ("stop_phase", 0.0), ("stop_ramp_start", 0.0), ("stop_initial_frequency", 0.0),
      ("rate", 0.0), ("rate_remaining", 0.0), ("restart_at", -1.0), ("failed", False),
      ("command_at", self.source_cfg.command_interval_s[0] + uniform[:, 2] * (self.source_cfg.command_interval_s[1] - self.source_cfg.command_interval_s[0])),
    ):
      self._put(name, mask, value)
    if standing:
      # 站立直接重复实测脚位，不生成随后会被丢弃的随机落点
      sides = (first_side[:, None] + self.reset_queue_ids) % 2
      self._put("queue", mask, feet.gather(1, sides[:, :, None].expand(-1, -1, 3)))
      self._put("queue_ids", mask, self.reset_queue_ids)
      self._put("first_side", mask, first_side)
    else:
      self._fill_queue(mask, first_side, torch.zeros_like(mask), uniform[:, 3:15])
      self._put("targets", mask, torch.stack((state["queue"][:, 0], feet[:, 1]), dim=1))
      self._put("target_ids", mask, torch.stack((state["queue_ids"][:, 0], torch.ones_like(state["next_id"])), dim=1))
    for name, source in (("completed_phase", "phase"), ("completed_frequency", "frequency"),
                         ("completed_targets", "targets"), ("completed_ids", "target_ids")):
      self._put(name, mask, state[source])
    self._publish()

  def _apply_request(self, uniform):
    """将行走意图转换为起停状态切换，不重置相位"""
    state, cfg = self.state, self.cfg
    stopping = ~state["request_walking"] & ((state["mode"] == WALKING) | (state["mode"] == STARTING))
    # 保留队列中的四步，再让异侧脚在队尾落点旁完成收步
    last = state["queue"][:, -1]
    closing_side = state["first_side"]
    offsets = torch.stack((torch.zeros_like(state["phase"]), (1 - 2 * closing_side).to(state["phase"].dtype) * cfg.hold_width,
                           torch.zeros_like(state["phase"])), dim=-1)
    closing = from_local(offsets, last)
    terminal = torch.where((self.sides[None, :] == closing_side[:, None])[:, :, None], closing[:, None], last[:, None])
    self._put("terminal_feet", stopping, terminal)
    self._put("stop_target_id", stopping, state["next_id"])
    self._put("stop_wait", stopping, -1.0)
    center = self.centers[closing_side]
    closing_touchdown = center + ((state["phase"] - center).div(2 * math.pi).floor() + 3) * (2 * math.pi)
    # 在收步双支撑窗口内、再次抬脚前结束；线性减速的相位增量为同时间恒频的一半，因此提前 T/2 开始
    stop_phase = closing_touchdown + 0.5 * cfg.phase.contact_half_width
    self._put("stop_phase", stopping, stop_phase)
    self._put("stop_initial_frequency", stopping, state["frequency"])
    self._put("stop_ramp_start", stopping, state["elapsed"] + (stop_phase - state["phase"]) / (2 * math.pi * state["frequency"].clamp_min(1e-12)) - 0.5 * cfg.stop_duration_s)
    self._put("mode", stopping, STOPPING)
    self._put("frequency", stopping, self._stop_frequency())
    # 从冻结相位继续；若下一次落地先于抬脚，先保持该脚原地支撑
    starting = state["request_walking"] & (state["mode"] == STANDING)
    touchdown = (self.centers[None, :] - state["phase"][:, None]).remainder(2 * math.pi)
    touchdown = torch.where(touchdown > 1e-10, touchdown, 2 * math.pi)
    first = touchdown.argmin(-1)
    distance = touchdown.gather(1, first[:, None]).squeeze(1)
    liftoff_distance = (self.liftoff[first] - state["phase"]).remainder(2 * math.pi)
    self._put("mode", starting, STARTING)
    self._put("start_progress", starting, min(cfg.control_dt / cfg.start_duration_s, 1.0))
    self._put("frequency", starting, state["request_frequency"] * state["start_progress"])
    self._put("stop_target_id", starting, -1)
    self._put("stop_wait", starting, -1.0)
    self._fill_queue(starting, first, distance < liftoff_distance, uniform[:, :12])

  def _stop_frequency(self):
    """计算下一控制拍内恒频或线性减速阶段的平均频率

    对减速曲线积分而非只取端点频率，保证离散相位准确到达 stop_phase，也覆盖跨越减速边界的控制拍
    """
    state, duration, dt = self.state, self.cfg.stop_duration_s, self.cfg.control_dt
    ramp_before = (state["elapsed"] - state["stop_ramp_start"]).clamp(0.0, duration)
    ramp_after = (state["elapsed"] + dt - state["stop_ramp_start"]).clamp(0.0, duration)
    coast = (state["stop_ramp_start"] - state["elapsed"]).clamp(0.0, dt)
    ramp = (ramp_after - ramp_before) * (1.0 - (ramp_after + ramp_before) / (2 * duration))
    return state["stop_initial_frequency"] * (coast + ramp) / dt

  def _source_advance(self, uniform):
    """调度站立保持与再起步、触边反向的频率游走和随机行走意图"""
    state, cfg, dt = self.state, self.source_cfg, self.cfg.control_dt
    # SETTLING 的频率也为零，但尚不能启动站立保持计时
    standing = state["mode"] == STANDING
    active = standing & ~state["request_walking"] & cfg.automatic_restart
    self._put("restart_at", active & (state["restart_at"] < 0), state["elapsed"] + cfg.hold_time_s[0] + uniform[:, 0] * (cfg.hold_time_s[1] - cfg.hold_time_s[0]))
    restart = active & (state["elapsed"] + 1e-10 >= state["restart_at"])
    direction, heading = self._directions(uniform[:, 1:3])
    self._put("direction", restart, direction)
    self._put("heading", restart, heading)
    self._put("request_walking", restart, True)
    self._put("request_frequency", restart, cfg.initial_frequency)
    interval = cfg.command_interval_s[0] + uniform[:, 3] * (cfg.command_interval_s[1] - cfg.command_interval_s[0])
    self._put("command_at", restart, state["elapsed"] + interval)
    self._put("restart_at", ~standing, -1.0)
    walking = (state["mode"] == WALKING) & state["request_walking"]
    new_rate = walking & (state["rate_remaining"] <= 1e-10)
    self._put("rate", new_rate, cfg.frequency_rate_range[0] + uniform[:, 4] * (cfg.frequency_rate_range[1] - cfg.frequency_rate_range[0]))
    self._put("rate_remaining", new_rate, cfg.frequency_rate_interval_s)
    lower, upper = cfg.frequency_range
    if upper == lower:
      target = torch.full_like(state["frequency"], lower)
    else:
      width = upper - lower
      remainder = (state["frequency"] + state["rate"] * dt - lower).remainder(2 * width)
      target = lower + torch.where(remainder <= width, remainder, 2 * width - remainder)
      rate = torch.where(remainder > width, -state["rate"], state["rate"])
      rate = torch.where(remainder == 0, state["rate"].abs(), rate)
      rate = torch.where(remainder == width, -state["rate"].abs(), rate)
      self._put("rate", walking, rate)
    self._put("request_frequency", walking, target)
    self._put("rate_remaining", walking, state["rate_remaining"] - dt)
    command = walking & cfg.automatic_commands & (state["elapsed"] >= state["command_at"])
    self._put("command_at", command, state["elapsed"] + interval)
    stop = uniform[:, 5] < cfg.stop_probability
    self._put("request_walking", command & stop, False)
    self._put("direction", command & ~stop, direction)
    self._put("heading", command & ~stop, heading)

  def advance(self, feet, contact):
    """用 (N, 2, 3) 实测位姿和 (N, 2) 接触状态推进一个物理控制拍"""
    self._capture()
    self._step(feet.to(torch.float64), contact)

  def _capture_completed(self):
    """在队列和模式变化前保存步末相位及旧频率、旧目标

    停止终拍对齐计划相位以避免数值越界，奖励仍评价刚执行一拍所使用的目标
    """
    state = self.state
    arrived = (state["mode"] == STOPPING) & (state["elapsed"] + self.cfg.control_dt >= state["stop_ramp_start"] + self.cfg.stop_duration_s - 1e-10)
    end_phase = state["phase"] + 2 * math.pi * state["frequency"] * self.cfg.control_dt
    state["completed_phase"].copy_(torch.where(arrived, state["stop_phase"], end_phase))
    state["completed_frequency"].copy_(state["frequency"])
    state["completed_targets"].copy_(state["targets"])
    state["completed_ids"].copy_(state["target_ids"])

  def _advance(self, feet, contact):
    """处理相位事件、确认支撑并更新意图，最后发布下一拍命令"""
    state, cfg = self.state, self.cfg
    alive = ~state["failed"]
    uniform = self._uniform(alive)
    old_phase = state["phase"].clone()
    end_phase = state["completed_phase"]
    state["contact_count"].copy_(torch.where(contact.all(-1), state["contact_count"] + 1, 0))
    # 计划落地消费队列，传感器接触不驱动队列时钟
    next_touch = self.centers + ((old_phase[:, None] - self.centers).div(2 * math.pi).floor() + 1) * (2 * math.pi)
    touched = (next_touch <= end_phase[:, None]) & alive[:, None]
    touchdown = touched.any(-1)
    side = touched.to(torch.long).argmax(-1)
    mismatch = touchdown & (side != state["first_side"])
    self._put("failed", mismatch, True)
    self._put("mode", mismatch, FAULT)
    supports = torch.where(touched[:, :, None], state["queue"][:, 0, None], state["supports"])
    state["supports"].copy_(supports)
    self._put("pending_anchor", touchdown, side)
    appended = self._sample(state["queue"][:, -1], state["first_side"], uniform[:, :3])
    self._put("queue", touchdown, torch.cat((state["queue"][:, 1:], appended[:, None]), dim=1))
    self._put("queue_ids", touchdown, torch.cat((state["queue_ids"][:, 1:], state["next_id"][:, None]), dim=1))
    self._put("next_id", touchdown, state["next_id"] + 1)
    self._put("first_side", touchdown, 1 - state["first_side"])
    # 每只脚的执行目标在抬脚时切换，不随预览队列推进而切换
    next_lift = self.liftoff + ((old_phase[:, None] - self.liftoff).div(2 * math.pi).floor() + 1) * (2 * math.pi)
    lifted = (next_lift <= end_phase[:, None]) & alive[:, None]
    indices = (self.sides[None, :] != state["first_side"][:, None]).long()
    targets = state["queue"].gather(1, indices[:, :, None].expand(-1, -1, 3))
    state["targets"].copy_(torch.where(lifted[:, :, None], targets, state["targets"]))
    state["target_ids"].copy_(torch.where(lifted, state["queue_ids"].gather(1, indices), state["target_ids"]))
    self._put("phase", alive, end_phase)
    self._put("elapsed", alive, state["elapsed"] + cfg.control_dt)
    double_support = self._contacts().all(-1)
    # 两次确认落地之间冻结局部参考系，确认后再按实测脚位重建
    pending = state["pending_anchor"].clamp_min(0)
    reanchor = (state["mode"] != STANDING) & (state["mode"] != SETTLING) & (state["pending_anchor"] >= 0) & double_support & contact.gather(1, pending[:, None]).squeeze(1)
    self._put("anchor", reanchor, self._choose(feet, pending))
    self._put("pending_anchor", reanchor, -1)
    stopping = state["mode"] == STOPPING
    self._put("frequency", stopping, self._stop_frequency())
    arrived = stopping & (state["elapsed"] >= state["stop_ramp_start"] + cfg.stop_duration_s - 1e-10)
    # 零频率在再次抬脚前冻结相位和队列，但不代表实际可靠支撑
    # SETTLING 等待连续接触确认，超时则进入故障状态
    self._put("mode", arrived, SETTLING)
    self._put("frequency", arrived, 0.0)
    self._put("targets", arrived, state["terminal_feet"])
    self._put("pending_anchor", arrived, -1)
    self._put("stop_wait", arrived, state["elapsed"])
    confirmed = (state["contact_count"] >= cfg.contact_confirm_steps) | (not cfg.require_contact_confirmation)
    standing = (state["mode"] == SETTLING) & confirmed
    self._put("mode", standing, STANDING)
    failed = (state["mode"] == SETTLING) & (state["elapsed"] - state["stop_wait"] >= cfg.landing_timeout_s - 1e-10)
    self._put("failed", failed, True)
    self._put("mode", failed, FAULT)
    starting = state["mode"] == STARTING
    self._put("start_progress", starting, (state["start_progress"] + cfg.control_dt / cfg.start_duration_s).clamp_max(1.0))
    self._put("frequency", starting, state["request_frequency"] * state["start_progress"])
    walking = state["mode"] == WALKING
    limit = cfg.frequency_slew_rate * cfg.control_dt
    self._put("frequency", walking, state["frequency"] + (state["request_frequency"] - state["frequency"]).clamp(-limit, limit))
    self._put("mode", starting & (state["start_progress"] >= 1.0), WALKING)
    # 新意图只影响下一拍发布的命令，不改变 completed_* 完成拍快照
    self._source_advance(uniform[:, 3:9])
    self._apply_request(uniform[:, 9:21])
    self._publish()