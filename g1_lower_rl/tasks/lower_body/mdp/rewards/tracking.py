"""跟随类奖励：线速度、角速度、原地位移、骨盆高度。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from g1_lower_rl.tasks.lower_body.mdp.observations import height_above_feet
from g1_lower_rl.tasks.lower_body.mdp.rewards._common import (
  ROBOT,
  command_restarted,
  moving,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  z_penalty: float = 2.0,
  std_scale: float = 0.0,
  std_knee: float = 0.8,
  asset_cfg: SceneEntityCfg = ROBOT,
) -> torch.Tensor:
  """线速度跟随的指数奖励。

  指令的 z 速度视为 0。``z_penalty`` 是垂直速度误差的相对权重：它和 xy 误差共用同一个
  指数核，所以调大等于把核在 z 方向收窄。

  指令超过 ``std_knee`` 之后，核宽按 ``std * (1 + std_scale * (|指令xy| - std_knee))``
  放宽。固定核宽的问题是：指令一超过约 2*std，“站着不动”和“尽力走”的奖励都是数值零，
  这一项对行为完全没有梯度，而走路要付的动作/打滑/加速度代价照收不误，于是最优解
  变成“大指令就别动”。

  **拐点不能省。** 不带拐点地线性放宽会把低速段的精度一起丢掉，而低速精度正是
  “蹭地走”那个坑的防线。拐点取 2 倍原核宽附近，刚好是原核还有梯度的边界。
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  error = xy_error + z_penalty * z_error
  std_eff: torch.Tensor | float = std
  if std_scale:
    excess = (torch.norm(command[:, :2], dim=1) - std_knee).clamp(min=0.0)
    std_eff = std * (1.0 + std_scale * excess)
  return torch.exp(-error / std_eff**2)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = ROBOT,
) -> torch.Tensor:
  """瞬时角速度跟随。指令的 xy 角速度视为 0。"""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  return torch.exp(-(z_error + 0.05 * xy_error) / std**2)


def velocity_shortfall(
  env: ManagerBasedRlEnv,
  command_name: str,
  cap_frac: float = 0.5,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = ROBOT,
) -> torch.Tensor:
  """惩罚“叫你走却没走起来”。单边：只罚不足，不罚超速。

  ``track_linear_velocity`` 是个指数核，只在误差进入捕获域之后才有梯度；指令一大，
  “原地不动”就落在数值上的平地里。本项补的就是那一段：线性，所以梯度恒等于权重。
  （``track_linear_velocity`` 的 ``std_scale`` 拓宽了捕获域，但拓不掉尾部。）

  四条设计约束：

  * **投影而不是模长**：沿指令方向取分量，横着走满速也不算数。
  * **单边**：超速交给指数核管，两项不打架。
  * **``cap_frac`` 封顶的是目标，不是惩罚**：只要求“动起来”到指令的 ``cap_frac``
    倍。要求全速的话，高速指令下它会变成一个压不下去的常数惩罚。
  * **目标取比例而不是固定值**：固定值下，小指令和大指令站着不动挨的罚一样重，
    而死区只在大指令处，压力加错了地方。正比后不用读指令量程、不用跟着课程走。

  取值：站着不动时恰好等于 ``cap_frac * |指令|``；**往反方向走时没有上界**（投影为负，
  罚得比站着更重）。这是故意的：历史上出现过 6:1 偏向后退的失败训练，把这段也 clamp 掉
  等于在最需要信号的地方重新造一个死区。

  ``|水平指令| < command_threshold`` 时恒为零，所以站立环境和纯原地转弯完全不受影响。
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  command_xy = command[:, :2]
  speed_command = torch.norm(command_xy, dim=1)
  direction = command_xy / speed_command.clamp(min=1e-6).unsqueeze(-1)
  achieved = torch.sum(asset.data.root_link_lin_vel_b[:, :2] * direction, dim=1)
  shortfall = (speed_command * cap_frac - achieved).clamp(min=0.0)
  return shortfall * (speed_command > command_threshold).float()


def track_base_height(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = ROBOT,
) -> torch.Tensor:
  """骨盆高度跟随指令的指数奖励。``std`` 由 ``tightening_height_std`` 课程接管。"""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  error = command[:, 0] - height_above_feet(asset, asset_cfg.site_ids)
  return torch.exp(-torch.square(error) / std**2)


class track_angular_velocity_avg:
  """转向跟随，但用**指令重采后累计的平均角速度**而不是瞬时值。

  瞬时版 ``track_angular_velocity`` 有一个实测到的问题：步态本身产生的 yaw 率振荡
  σ = 0.23~0.77，和核宽 std=0.5 同量级，直流跟踪误差被它稀释。实测同一批场景：

  | 原地转 +0.8 | 实测均值 | 瞬时奖励 | 平均后奖励 |
  |---|---|---|---|
  | 标杆 16-36-09 | 0.68 | 0.700 | **0.934** |
  | 一个不转的策略 | 0.03 | 0.147 | **0.103** |
  | **区分度** | | **4.8×** | **9.1×** |

  平均之后，转对了的不再为振荡付钱，不转的再也躲不掉——两个方向相反，所以区分度
  翻倍。ω=1.5 时从 163× 到 840×。

  **为什么是“重采后累计”而不是滑动窗口。** 指令是每 3~8 s 阶跃一次的，滑窗在阶跃
  后会有整整一个窗长的数据来自旧指令，那段时间的奖励是错的。从重采时刻重新起算就
  没有这个污染期，而且累计越久噪声抑制越好。

  **为什么积分角速度而不是取朝向差。** ``heading_w`` 是 wrap 到 ±π 的，ω=1.5 跑满
  8 s 会转 1.9 圈，直接做差会被绕回毁掉。

  **不要拿它替换瞬时项。** 平均会把振荡一起滤掉：实测一个直行时 yaw 率 σ=0.77
  （标杆是 0.47）的策略，换成平均后奖励从 0.365 涨到 0.842，过度晃动完全隐形。
  两项并列：瞬时项放宽 std 管振荡，本项收紧管直流。

  线速度的同款见 ``track_linear_velocity_avg``。它的收益比这一项小得多（收紧 1.19×
  对 1.9×），因为线速度是质心运动、被整机质量平滑过，振荡本来就不大。
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    self._accum = torch.zeros(env.num_envs, device=env.device)
    self._elapsed = torch.zeros(env.num_envs, device=env.device)
    self._prev_left: torch.Tensor | None = None

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = ROBOT,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    term = env.command_manager.get_term(command_name)
    command = env.command_manager.get_command(command_name)
    assert term is not None and command is not None

    fresh, self._prev_left = command_restarted(env, term, self._prev_left)
    zero = torch.zeros_like(self._accum)
    self._accum = torch.where(fresh, zero, self._accum)
    self._elapsed = torch.where(fresh, zero, self._elapsed)

    dt = env.step_dt
    self._accum = self._accum + asset.data.root_link_ang_vel_b[:, 2] * dt
    self._elapsed = self._elapsed + dt
    avg = self._accum / self._elapsed.clamp(min=dt)

    is_moving = (
      torch.linalg.vector_norm(command[:, :2], dim=1) + command[:, 2].abs()
      > command_threshold
    )
    error = torch.abs(command[:, 2] - avg)
    env.extras.setdefault("log", {})["Metrics/twist/avg_yaw_rate_error"] = (
      error[is_moving].mean()
      if is_moving.any()
      else torch.zeros((), device=error.device)
    )
    return torch.exp(-torch.square(error) / std**2) * is_moving.float()


class track_linear_velocity_avg:
  """线速度跟随，用**指令重采后累计的平均速度**——即“这段窗口内实际走了多远 / 窗口时长”。

  **前提是指令在窗口内恒定。** 以前指令带 OU 游走，目标本身每一拍都在动，“平均速度”
  没有可比的对象；游走删掉之后这一项才成立。

  实测（满档扰动、512 环境、跳过阶跃后 1 s 暂态、窗口再攒 >1 s）：

  | | 均值 | p50 | p90 |
  |---|---|---|---|
  | 瞬时误差 \\|cmd - v\\| | 0.348 | 0.300 | 0.646 |
  | 平均误差 \\|cmd - avg\\| | **0.293** | 0.240 | 0.582 |

  收紧 1.19×，振荡幅度 \\|v - avg\\| 均值 0.249（是瞬时项核宽 0.447 的 0.56 倍）。
  比转向那一项的 1.9× 小得多——线速度是质心运动，被整机质量平滑过，本来就没多少
  振荡可滤。所以它的权重也该按比例给小：转向是 avg 占 76%，这里只占 32%。

  **``ramp_time``：窗口刚开始时权重按线性爬坡。** 不加坡的话这一项在阶跃后是有害的：
  实测窗口前 1.5 s 内平均误差反而比瞬时误差**差** 19~24%，因为加速暂态整段被记进了
  平均。爬坡让“刚换指令”几乎不收费、“攒够时间”才全额收费，正好对上这个物理。
  代价是本项的时间平均降到约 0.86 倍（窗长均值 5.5 s，坡长 1.5 s）。

  **不按 ``moving`` 抓闸**，和转向那一项相反。这里是把原来 ``track_linear_velocity``
  的权重拆开，两项相加必须还等于原来的尺度——抓闸会让站立环境只剩瞬时项那一半。
  站立环境本项算的是“净位移/时长”，本身就是一道额外的抗漂移信号。
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    self._accum = torch.zeros(env.num_envs, 2, device=env.device)
    self._elapsed = torch.zeros(env.num_envs, device=env.device)
    self._prev_left: torch.Tensor | None = None

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    ramp_time: float = 1.5,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = ROBOT,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    term = env.command_manager.get_term(command_name)
    command = env.command_manager.get_command(command_name)
    assert term is not None and command is not None

    fresh, self._prev_left = command_restarted(env, term, self._prev_left)
    self._accum = torch.where(
      fresh.unsqueeze(-1), torch.zeros_like(self._accum), self._accum
    )
    self._elapsed = torch.where(fresh, torch.zeros_like(self._elapsed), self._elapsed)

    dt = env.step_dt
    self._accum = self._accum + asset.data.root_link_lin_vel_b[:, :2] * dt
    self._elapsed = self._elapsed + dt
    avg = self._accum / self._elapsed.clamp(min=dt).unsqueeze(-1)

    error = torch.linalg.vector_norm(command[:, :2] - avg, dim=1)
    ramp = (
      (self._elapsed / ramp_time).clamp(max=1.0)
      if ramp_time > 0.0
      else torch.ones_like(self._elapsed)
    )
    # 门槛只影响日志口径，奖励本身对所有环境生效。
    is_moving = moving(env, command_name, command_threshold) > 0.5
    env.extras.setdefault("log", {})["Metrics/twist/avg_lin_vel_error"] = (
      error[is_moving].mean()
      if is_moving.any()
      else torch.zeros((), device=error.device)
    )
    return torch.exp(-torch.square(error) / std**2) * ramp


class stationary_drift:
  """xy 指令恒为零的环境（原地转弯 + 站立），罚骨盆相对指令下达时刻的**位移**。

  ``track_linear_velocity`` 在 vx=vy=0 时罚的是速度，不是位置，而慢漂几乎不要钱：
  0.05 m/s 对应 ``exp(-0.05²/0.25)`` = 0.990，只丢 1% 的那一项。实测这个漏洞是
  真的被利用了（8 s 回放，位移从第 1 s 起算）：

  | 场景 | 当前策略末位移 | 标杆 16-36-09 |
  |---|---|---|
  | 站立 | **17.3 cm** | 1.4 cm |
  | 原地转 +0.4 | **46.5 cm** | 24.1 cm |

  46 cm / 7 s = 0.066 m/s，速度项只收 1.7% 的费——机器人是在“绕着圈走”而不是
  “原地转”，而奖励看不出区别。

  **为什么是有界核而不是 l2。** 扰动课程会给机身冲量，一脚踹出去 30 cm 之后 l2 会
  持续放大到淹没其他所有项；``1 - exp(-d²/std²)`` 封顶在 1，被踹飞的代价有限。

  **为什么锚点跟着指令重采而不是整局固定。** 指令每 3~8 s 阶跃一次，站立/转弯与
  行走会交替出现；整局固定的话，一段行走指令走出去 5 m，回到原地转弯时锚点还在
  五米外，惩罚就变成了“回原点”这个完全无关的目标。
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    self._anchor = torch.zeros(env.num_envs, 2, device=env.device)
    self._prev_left: torch.Tensor | None = None

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = ROBOT,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    term = env.command_manager.get_term(command_name)
    assert term is not None
    xy = asset.data.root_link_pos_w[:, :2]

    fresh, self._prev_left = command_restarted(env, term, self._prev_left)
    self._anchor = torch.where(fresh.unsqueeze(-1), xy, self._anchor)

    # 用指令类的标志位而不是“指令模长≈0”：行走环境偶尔也会采到接近零的 vx/vy，
    # 那时候罚位移是错的（它确实被要求走，只是走得慢）。
    stationary = term.is_turning_env | term.is_standing_env
    dist = torch.linalg.vector_norm(xy - self._anchor, dim=1)
    env.extras.setdefault("log", {})["Metrics/twist/stationary_drift"] = (
      dist[stationary].mean()
      if stationary.any()
      else torch.zeros((), device=dist.device)
    )
    return (1.0 - torch.exp(-torch.square(dist) / std**2)) * stationary.float()
