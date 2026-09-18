"""单机器人四步内部计划、双脚支撑基准加各一步目标及停走状态管理"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike

from g1_lower_rl.footsteps.command_source import GaitRequest
from g1_lower_rl.footsteps.config import FootstepManagerCfg
from g1_lower_rl.footsteps.sampler import FootstepSampler, from_local, pose_array, to_local, wrap_angle


@dataclass(frozen=True)
class Footstep:
  """一个带脚侧、世界位姿和唯一目标编号的计划落点"""

  side: int
  pose_w: np.ndarray
  target_id: int


@dataclass(frozen=True)
class ExecutionSnapshot:
  """刚结束控制拍的执行快照，保留命令切换前目标，不依赖训练奖励"""

  phase: float
  frequency: float
  targets_w: np.ndarray
  target_ids: np.ndarray


@dataclass(frozen=True)
class FootstepCommand:
  """四槽位固定为左支撑基准、右支撑基准、左下一落点、右下一落点"""

  phase: float
  frequency: float
  footsteps: np.ndarray
  footsteps_w: np.ndarray
  future_ids: np.ndarray
  future_sides: np.ndarray
  anchor_w: np.ndarray
  required_contact: np.ndarray
  mode: str


@dataclass(frozen=True)
class FootstepUpdate:
  """一次推进产生的新命令、已完成拍的执行快照和理论落地脚侧"""

  command: FootstepCommand
  completed: ExecutionSnapshot
  landed_sides: tuple[int, ...]


class FootstepManager:
  """先 reset 再逐拍 advance，公开输出数组为副本，不允许外部改写内部队列"""

  def __init__(self, cfg: FootstepManagerCfg | None = None, seed: int | None = None):
    """初始化脚印随机流，时钟和执行状态不再包含随机调度"""
    self.cfg = cfg or FootstepManagerCfg()
    sampler_seed = np.random.SeedSequence(seed).spawn(2)[1]
    self.sampler = FootstepSampler(self.cfg.sampler, sampler_seed)
    self.initialized = False

  def _require_ready(self) -> None:
    """拒绝尚未初始化或已故障的管理器操作"""
    if not self.initialized:
      raise RuntimeError("Call reset(feet_w) before using the manager")
    if self.mode == "fault":
      raise RuntimeError("Manager faulted; recover safely and reset before continuing")

  def _contacts_at(self, phase: float) -> np.ndarray:
    """按累计弧度相位计算左右计划接触掩码，不代表实际接地"""
    start = np.remainder(self.cfg.phase.contact_start_rad, 2 * math.pi)
    end = np.remainder(self.cfg.phase.liftoff_rad, 2 * math.pi)
    wrapped = phase % (2 * math.pi)
    return np.where(end > start, (wrapped >= start) & (wrapped < end), (wrapped >= start) | (wrapped < end))

  def _new_step(self, side: int, previous: np.ndarray) -> Footstep:
    """分配新目标编号，行走时采样新落点，停止或站立时复用终止落点"""
    if self.mode in ("stopping", "settling", "standing"):
      target = self.terminal_feet[side].copy()
    else:
      target = self.sampler.sample(previous, side)
    step = Footstep(side, target, self.next_id)
    self.next_id += 1
    return step

  def _fill_queue(self, first_side: int, *, hold_first: bool = False) -> None:
    """从指定脚侧交替填充四步，必要时让首个目标保持当前支撑位置"""
    self.queue: list[Footstep] = []
    previous = self.supports[1 - first_side]
    for offset in range(4):
      if hold_first and offset == 0:
        step = Footstep(first_side, self.supports[first_side].copy(), self.next_id)
        self.next_id += 1
      else:
        step = self._new_step((first_side + offset) % 2, previous)
      self.queue.append(step)
      previous = step.pose_w

  def reset(
    self,
    feet_w: ArrayLike,
    request: GaitRequest | None = None,
  ) -> FootstepCommand:
    """用实测脚位重建站立目标，默认频率为零；显式意图可覆盖初态"""
    feet = pose_array(feet_w, (2, 3))
    feet[:, 2] = wrap_angle(feet[:, 2])
    heading = math.atan2(np.sin(feet[:, 2]).sum(), np.cos(feet[:, 2]).sum())
    request = request or GaitRequest(heading, heading, self.cfg.initial_frequency, walking=False)
    self._validate_request(request)
    self.request = request
    self.sampler.set_direction(request.movement_direction, request.foot_heading)
    standing = not request.walking
    # 两个双支撑中心等概率覆盖，下一抬脚侧是该落地中心的异侧
    stance_side = int(self.sampler.rng.random() >= 0.5) if standing else 1
    self.mode = "standing" if standing else "walking"
    self.frequency = 0.0 if standing else request.frequency
    self.elapsed = 0.0
    self.phase = (self.cfg.phase.left_stance_phase, self.cfg.phase.right_stance_phase)[stance_side] if standing else self.cfg.phase.liftoff_rad[0]
    self.supports = feet
    self.targets = feet.copy()
    self.target_ids = np.array([0, 1], dtype=np.int64)
    self.next_id = 2
    self.terminal_feet = feet.copy()
    self.anchor = feet[stance_side].copy()
    self.pending_anchor_side = None
    self.contact_count = 0
    self.stop_target_id = None
    self.stop_wait_started = None
    self.stop_phase = 0.0
    self.stop_ramp_start = 0.0
    self.stop_initial_frequency = 0.0
    self.start_progress = 0.0
    self._fill_queue(1 - stance_side)
    if not standing:
      self.targets[0] = self.queue[0].pose_w
      self.target_ids[0] = self.queue[0].target_id
    self.initialized = True
    return self.command()

  def _validate_request(self, request: GaitRequest) -> None:
    """在改变任何状态之前校验意图类型和执行频率范围"""
    if not isinstance(request, GaitRequest):
      raise TypeError("Expected GaitRequest")
    if not self.cfg.frequency_range[0] <= request.frequency <= self.cfg.frequency_range[1]:
      raise ValueError("Request frequency outside manager execution range")

  def apply_request(self, request: GaitRequest) -> None:
    """接受外部意图而不推进相位，已开始的收步不会中途取消"""
    self._require_ready()
    self._validate_request(request)
    self.request = request
    self.sampler.set_direction(request.movement_direction, request.foot_heading)
    if not request.walking:
      self.request_stop()
    elif self.mode == "standing":
      self.request_start()

  def set_direction(self, movement_direction: float, foot_heading: float) -> None:
    """设置后续追加脚印的移动方向和脚掌朝向，不改已发布四步"""
    self._require_ready()
    self.apply_request(replace(self.request, movement_direction=movement_direction, foot_heading=foot_heading))

  def request_stop(self) -> None:
    """保留承诺四步，末段减速对齐收脚相位，再冻结并等待双接触确认"""
    self._require_ready()
    self.request = replace(self.request, walking=False)
    if self.mode in ("stopping", "settling", "standing"):
      return
    last = self.queue[-1]
    self.terminal_feet = np.repeat(last.pose_w[None, :], 2, axis=0)
    closing_side = 1 - last.side
    sign = 1 if closing_side == 0 else -1
    self.terminal_feet[closing_side] = from_local([0.0, sign * self.cfg.hold_width, 0.0], last.pose_w)
    # 首个新追加目标负责收步，已经发布的四步仍按原计划执行
    self.stop_target_id = self.next_id
    self.stop_wait_started = None
    center = (self.cfg.phase.left_stance_phase, self.cfg.phase.right_stance_phase)[closing_side]
    closing_touchdown = center + (math.floor((self.phase - center) / (2 * math.pi)) + 3) * (2 * math.pi)
    self.stop_phase = closing_touchdown + 0.5 * self.cfg.phase.contact_half_width
    self.stop_initial_frequency = self.frequency
    self.stop_ramp_start = self.elapsed + (self.stop_phase - self.phase) / (2 * math.pi * self.frequency) - 0.5 * self.cfg.stop_duration_s
    self.mode = "stopping"
    self.frequency = self._stop_frequency()

  def _stop_frequency(self) -> float:
    """计算下一控制拍内连续末段减速曲线的平均频率"""
    duration, dt = self.cfg.stop_duration_s, self.cfg.control_dt
    ramp_before = min(max(self.elapsed - self.stop_ramp_start, 0.0), duration)
    ramp_after = min(max(self.elapsed + dt - self.stop_ramp_start, 0.0), duration)
    coast = min(max(self.stop_ramp_start - self.elapsed, 0.0), dt)
    ramp = (ramp_after - ramp_before) * (1.0 - (ramp_after + ramp_before) / (2 * duration))
    return self.stop_initial_frequency * (coast + ramp) / dt

  def set_frequency(self, frequency: float) -> None:
    """设置正频率执行目标，随机与手动模式由外部指令源决定"""
    self._require_ready()
    self.apply_request(replace(self.request, frequency=frequency))

  def request_start(self) -> None:
    """从站立按固定时长升至请求频率，重建预览并保留相位及起脚前支撑目标"""
    self._require_ready()
    self.request = replace(self.request, walking=True)
    if self.mode != "standing":
      return
    candidates = [
      ((center - self.phase) % (2 * math.pi), side)
      for side, center in enumerate((self.cfg.phase.left_stance_phase, self.cfg.phase.right_stance_phase))
    ]
    touchdown_distance, first_side = min((distance if distance > 1e-10 else 2 * math.pi, side) for distance, side in candidates)
    liftoff_distance = (self.cfg.phase.liftoff_rad[first_side] - self.phase) % (2 * math.pi)
    self.mode = "starting"
    self.start_progress = min(self.cfg.control_dt / self.cfg.start_duration_s, 1.0)
    self.frequency = self.request.frequency * self.start_progress
    self.stop_target_id = None
    self.stop_wait_started = None
    # 若落地中心先于下一次起脚到来，必须保留原地目标而不是提前消费新步
    self._fill_queue(first_side, hold_first=touchdown_distance < liftoff_distance)

  def command(self) -> FootstepCommand:
    """返回当前命令副本，将世界队列统一转换到冻结参考系"""
    self._require_ready()
    ordered = [next(step for step in self.queue if step.side == side) for side in (0, 1)]
    world = np.concatenate((self.supports, np.stack([step.pose_w for step in ordered])), axis=0)
    return FootstepCommand(
      phase=self.phase % (2 * math.pi),
      frequency=self.frequency,
      footsteps=to_local(world, self.anchor).astype(np.float32),
      footsteps_w=world,
      future_ids=np.array([step.target_id for step in ordered], dtype=np.int64),
      future_sides=np.array([step.side for step in ordered], dtype=np.int64),
      anchor_w=self.anchor.copy(),
      required_contact=np.ones(2, dtype=bool) if self.mode in ("settling", "standing") else self._contacts_at(self.phase),
      mode=self.mode,
    )

  def _events(self, end_phase: float) -> list[tuple[float, str, int]]:
    """列出当前相位到步末相位之间跨越的理论落地和起脚事件"""
    events = []
    for kind, boundaries in (
      ("touchdown", (self.cfg.phase.left_stance_phase, self.cfg.phase.right_stance_phase)),
      ("liftoff", self.cfg.phase.liftoff_rad),
    ):
      for side, boundary in enumerate(boundaries):
        next_phase = boundary + (math.floor((self.phase - boundary) / (2 * math.pi)) + 1) * (2 * math.pi)
        if next_phase <= end_phase:
          events.append((next_phase, kind, side))
    return sorted(events)

  def advance(self, *, feet_w: ArrayLike | None = None, contacts: ArrayLike | None = None) -> FootstepUpdate:
    """物理控制拍结束后推进一次，先保存执行快照再更新目标与停走状态

    feet_w 为可选的左右足端世界 XY/yaw，contacts 为可选的左右实测接触
    始终使用上一拍发布的频率积分，外部适配器将执行快照转换为奖励输入
    """
    self._require_ready()
    measured = None if feet_w is None else pose_array(feet_w, (2, 3))
    contact = None if contacts is None else np.asarray(contacts)
    if contact is not None and (contact.shape != (2,) or not np.isin(contact, [0, 1]).all()):
      raise ValueError("contacts must contain two boolean left/right contact estimates")
    dt = self.cfg.control_dt
    self.contact_count = self.contact_count + 1 if contact is not None and contact.all() else 0
    old_frequency = self.frequency
    end_phase = self.phase + 2 * math.pi * old_frequency * dt
    if self.mode == "stopping" and self.elapsed + dt >= self.stop_ramp_start + self.cfg.stop_duration_s - 1e-10:
      end_phase = self.stop_phase
    # 奖励属于刚结束的物理拍，快照必须早于落地或起脚引起的目标切换
    completed = ExecutionSnapshot(end_phase, old_frequency, self.targets.copy(), self.target_ids.copy())
    landed = []
    for _, kind, side in self._events(end_phase):
      if kind == "touchdown":
        step = self.queue.pop(0)
        if step.side != side:
          self.mode = "fault"
          raise RuntimeError("Footstep queue and phase clock disagree")
        self.supports[side] = step.pose_w
        self.pending_anchor_side = side
        landed.append(side)
        self.queue.append(self._new_step(1 - self.queue[-1].side, self.queue[-1].pose_w))
      else:
        step = next(step for step in self.queue if step.side == side)
        self.targets[side] = step.pose_w
        self.target_ids[side] = step.target_id
    self.phase = end_phase
    self.elapsed += dt
    if (
      self.mode not in ("settling", "standing")
      and self.pending_anchor_side is not None
      and measured is not None
      and contact is not None
      and contact[self.pending_anchor_side]
      and self.cfg.phase.in_double_support(self.phase)
    ):
      self.anchor = measured[self.pending_anchor_side].copy()
      self.pending_anchor_side = None

    if self.mode == "standing":
      if self.request.walking:
        self.request_start()
    elif self.mode in ("stopping", "settling"):
      if self.mode == "stopping":
        if self.elapsed >= self.stop_ramp_start + self.cfg.stop_duration_s - 1e-10:
          self.mode = "settling"
          self.frequency = 0.0
          self.targets[:] = self.terminal_feet
          self.pending_anchor_side = None
          self.stop_wait_started = self.elapsed
        else:
          self.frequency = self._stop_frequency()
      if self.mode == "settling":
        confirmed = not self.cfg.require_contact_confirmation or self.contact_count >= self.cfg.contact_confirm_steps
        if confirmed:
          self.mode = "standing"
      if (
        self.mode == "settling"
        and self.stop_wait_started is not None
        and self.elapsed - self.stop_wait_started >= self.cfg.landing_timeout_s - 1e-10
      ):
        self.mode = "fault"
        raise TimeoutError("Double support was not confirmed; stop hardware safely before resetting")
    elif self.mode == "starting":
      target = self.request.frequency
      self.start_progress = min(1.0, self.start_progress + dt / self.cfg.start_duration_s)
      self.frequency = target * self.start_progress
      if self.start_progress >= 1.0:
        self.mode = "walking"
    else:
      limit = self.cfg.frequency_slew_rate * dt
      self.frequency += float(np.clip(self.request.frequency - self.frequency, -limit, limit))
    return FootstepUpdate(self.command(), completed, tuple(landed))

  def state_dict(self) -> dict:
    """深拷贝完整 Python/NumPy 状态及随机数状态，仅供可信本地检查点使用"""
    self._require_ready()
    return copy.deepcopy(self.__dict__)

  def load_state_dict(self, state: dict) -> None:
    """恢复相同配置下已初始化的完整快照，拒绝配置不一致的状态"""
    if state.get("cfg") != self.cfg or not state.get("initialized"):
      raise ValueError("Manager checkpoint must be initialized and use the same configuration")
    self.__dict__.update(copy.deepcopy(state))
