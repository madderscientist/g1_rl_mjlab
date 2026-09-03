"""GMT 式通用动作跟踪命令。

与 mjlab 自带的单动作 ``MotionCommand`` 的两点关键差别，也正是「换一段没训过的动作
还能跟住」的来源：

1. **多动作 + bin 级自适应采样**。语料里的每条动作都参与训练，且整份语料被切成固定时长的
   bin，复位时按每个 bin 的失败率加权挑一个、再在 bin 内取起点（SONIC 的 bin-based
   adaptive sampling）。按整条动作加权不够：难点往往只占一条长动作里的几秒，把整条上
   调之后起点仍然撒满全篇，额外算力大部分花在已经练熟的段落上。

2. **偏航/平移不变的前瞻观测**。策略看到的不是「参考轨迹在世界里的绝对位姿」，而是
   未来若干帧在**根坐标系**下的：离地高度、重力方向、线/角速度、关节角。绝对位置和
   朝向被彻底剔除，于是同一段动作平移或转向之后对策略而言完全相同——这是泛化到新动作
   的前提。用投影重力而不是 roll/pitch 角，是为了避开角度回绕。

参考：GMT: General Motion Tracking for Humanoid Whole-Body Control (arXiv:2506.14770)，
采样策略改成了 SONIC (arXiv:2511.07820) 的 bin 级版本。
奖励/终止项沿用 mjlab tracking 的实现，所以这里刻意保持了和 ``MotionCommand`` 一致的
属性名。
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_error_magnitude,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
  yaw_quat,
)

from g1_lower_rl.tasks.motion_tracking.mdp.motion_corpus import MotionCorpus

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  """把世界系向量转到 q 所定义的局部系。"""
  return quat_apply(quat_inv(q), v)


# 复位时写进仿真的参考速度上限。语料里的重定向假象（欧拉角万向节翻转）会造出
# 单帧 π 级跳变，折合 50 Hz 就是 157 rad/s——直接写进去会让物理当场发散成 NaN，
# 而多卡下这个 NaN 会伪装成 SIGSEGV + NCCL 超时（见 eval-gotchas 第 15/16 条）。
# 取值参照 LAFAN1 真实动捕的实测上限（关节速度最大 37 rad/s）。
RESET_JOINT_VEL_LIMIT = 40.0
"""rad/s。"""
RESET_ROOT_LIN_VEL_LIMIT = 10.0
"""m/s。"""
RESET_ROOT_ANG_VEL_LIMIT = 20.0
"""rad/s。"""


class GeneralMotionCommand(CommandTerm):
  cfg: GeneralMotionCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: GeneralMotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self.robot_anchor_body_index = self.robot.body_names.index(cfg.anchor_body_name)
    self.motion_anchor_body_index = cfg.body_names.index(cfg.anchor_body_name)
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )
    # 前瞻观测只喂策略实际驱动的那些关节，维度才对得上动作空间。
    self.policy_joint_indexes = torch.tensor(
      self.robot.find_joints(cfg.policy_joint_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )
    self.reference_key_indexes = torch.tensor(
      [cfg.body_names.index(n) for n in cfg.reference_key_bodies],
      dtype=torch.long,
      device=self.device,
    )

    self.motion = MotionCorpus(cfg.motion_dir, self.body_indexes, device=self.device)
    self.bins = self.motion.make_bins(cfg.bin_frames)
    print(
      f"[GMT] 载入语料: {self.motion.describe()}，切成 {self.bins.num_bins} 个采样 bin"
    )

    self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    # 当前回合从哪个 bin 起步。-1 = 还没经自适应采样派发过，其成败不计入统计。
    self.bin_ids = torch.full(
      (self.num_envs,), -1, dtype=torch.long, device=self.device
    )
    # 钉死到某一条动作（逐条渲染/调试用），None = 正常采样。
    self.forced_motion_id: int | None = None
    # 相位用浮点：每拍推进 self.speed 帧，取参考帧时再取整。
    self.phase = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
    self.speed = torch.ones(self.num_envs, dtype=torch.float, device=self.device)

    self.lookahead = torch.tensor(
      cfg.lookahead_steps, dtype=torch.long, device=self.device
    )

    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0

    # 每个 bin 的失败率，值域 [0,1]。乐观初始化成上限：没练过的 bin 先当成最难，
    # 保证冷启动时先把全语料扫一遍，而不是卡在最早出现失败的那几个 bin 上。
    self.bin_failed = torch.full(
      (self.bins.num_bins,), cfg.adaptive_failure_cap, device=self.device
    )
    self.bin_speed_weight = self._speed_balance_weights()
    self._offsets_cache: torch.Tensor | None = None

    self._ghost_model = None

    for key in (
      "error_anchor_pos",
      "error_anchor_rot",
      "error_body_pos",
      "error_body_rot",
      "error_joint_pos",
      "error_joint_vel",
      "sampling_entropy",
      "sampling_failure_rate",
    ):
      self.metrics[key] = torch.zeros(self.num_envs, device=self.device)

  ##
  # 帧寻址
  ##

  @property
  def time_steps(self) -> torch.Tensor:
    """当前参考帧在拼接语料里的全局下标。"""
    return self.motion.start_idx[self.motion_ids] + self.phase.long()

  def _lookahead_indexes(self) -> torch.Tensor:
    """(num_envs, K) 前瞻帧的全局下标，超出本条动作末尾的部分钳到末帧。

    前瞻步长也乘速度：放慢时该看到的是同一段「未来多少秒」的动作，而不是同样帧数。
    """
    phase = self.phase.unsqueeze(1) + self.lookahead.unsqueeze(0) * self.speed.unsqueeze(1)
    last = (self.motion.num_frames[self.motion_ids] - 1).unsqueeze(1).float()
    phase = torch.minimum(phase, last)
    return self.motion.start_idx[self.motion_ids].unsqueeze(1) + phase.long()

  ##
  # 参考量（属性名与 mjlab MotionCommand 对齐，奖励/终止项可直接复用）
  ##

  @property
  def command(self) -> torch.Tensor:
    """偏航与平移不变的前瞻观测，(num_envs, K * 39)。"""
    idx = self._lookahead_indexes()
    root_quat = self.motion.body_quat_w[idx, 0]  # (N, K, 4)

    height = self.motion.body_pos_w[idx, 0, 2:3]
    gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(
      root_quat.shape[0], root_quat.shape[1], 3
    )
    proj_gravity = quat_rotate_inverse(root_quat, gravity)
    lin_vel = quat_rotate_inverse(root_quat, self.motion.body_lin_vel_w[idx, 0])
    ang_vel = quat_rotate_inverse(root_quat, self.motion.body_ang_vel_w[idx, 0])
    # 参考速度要跟着播放倍率缩放：放慢后位置推进变慢，速度也必须变慢，
    # 否则“该到哪”和“该多快”互相矛盾，策略无法同时满足。
    scale = self.speed[:, None, None]
    lin_vel = lin_vel * scale
    ang_vel = ang_vel * scale
    joint_pos = self.motion.joint_pos[idx][:, :, self.policy_joint_indexes]

    return torch.cat(
      [height, proj_gravity, lin_vel, ang_vel, joint_pos], dim=-1
    ).flatten(1)

  def _window_indexes(self) -> torch.Tensor:
    """(num_envs, 2L+1) 参考窗口的全局帧下标，含过去与未来各 L 个 token。

    与 ``_lookahead_indexes`` 的区别：那个是稀疏远前瞻（最远 1.9 s）给 MLP 拍扁用的；
    这个是 RGMT 的局部窗口，要保住 token 结构喂 cross-attention。
    两端都按本条动作的首末帧钳住，避免跨条串帧。
    """
    offsets = self._window_offsets()
    phase = self.phase.unsqueeze(1) + offsets.unsqueeze(0) * self.speed.unsqueeze(1)
    last = (self.motion.num_frames[self.motion_ids] - 1).unsqueeze(1).float()
    phase = phase.clamp(min=0.0).minimum(last)
    return self.motion.start_idx[self.motion_ids].unsqueeze(1) + phase.long()

  def _window_offsets(self) -> torch.Tensor:
    """token 的帧偏移，跨度与曲率都是照着实测注意力分布定的。

    在 ±0.6 s 窗口上钩出 cross-attn 的权重后，形状非常明确：

      偏移(s)  -0.60  -0.30   0.00  +0.16  +0.22  +0.30  +0.40  +0.60
      权重      0.4%   0.7%  11.0%  10.2%  19.0%  13.5%   3.2%   0.65%

    未来占 70.7%（策略确实在看前瞻），但峰值在 +0.22 s，0.4 s 以外总共不到 5%。
    也就是说 ±0.6 s 里有三分之一的 token 只分到 7% 注意力，纯属浪费；而最早的
    ±0.2 s 窗口边界恰好卡在峰值前面，这才是当初「前瞻不足」的真实缺口——只差
    0.1 s，不是 0.4 s。多给的远端还会反过来拖累高动态：把可视未来从 0.6 s 砍到
    0.04 s，jumps/fight/dance 各涨 7~8%（追一个已经跟丢的远期目标只会诱发激进动作）。

    所以跨度收到 ±0.3 s 刚好罩住峰值，省下的 token 全部加密到关键区间。
    指数取 1.5 而非 2：跨度缩小后曲率必须跟着变缓，否则近处会算出重复偏移。
    """
    L = self.cfg.reference_window
    s = self.cfg.reference_stride
    if s == 1:
      return torch.arange(-L, L + 1, device=self.device, dtype=torch.float)
    if self._offsets_cache is None:
      i = torch.arange(1, L + 1, device=self.device, dtype=torch.float)
      span = round(L * s)
      far = torch.round(((i / L) ** 1.5) * (span - 1) + 1)
      self._offsets_cache = torch.cat([-far.flip(0), torch.zeros(1, device=self.device), far])
    return self._offsets_cache

  @property
  def reference_tokens(self) -> torch.Tensor:
    """RGMT 的参考窗口观测，(num_envs, (2L+1) * token_dim)，模型内部再 reshape 回 token。

    基础 token 是 Extreme-RGMT 式 (2)：``[v_ref, ω_ref, g_ref, q_ref]``（38 维）。
    前三项都转到该帧参考根节点的自身坐标系，因此对世界偏航与平移不变。

    **但只有这 38 维时 actor 是开环的**：整个 token 对世界平移不变，策略无法知道自己
    离参考漂了多远，位置误差没有任何输入通道，漂移必然随时间无界累积；手臂也只有关节角
    目标，没有笛卡尔目标可供补偿重力/惯性负载下的稳态误差。

    key body 的位置和速度都表达在机器人当前 anchor 的 yaw 局部系下；位置依赖里程计，
    速度只依赖 yaw 姿态。
    """
    idx = self._window_indexes()
    root_quat = self.motion.body_quat_w[idx, 0]  # (N, T, 4)

    gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(
      root_quat.shape[0], root_quat.shape[1], 3
    )
    proj_gravity = quat_rotate_inverse(root_quat, gravity)
    scale = self.speed[:, None, None]
    lin_vel = quat_rotate_inverse(root_quat, self.motion.body_lin_vel_w[idx, 0]) * scale
    ang_vel = quat_rotate_inverse(root_quat, self.motion.body_ang_vel_w[idx, 0]) * scale
    joint_pos = self.motion.joint_pos[idx][:, :, self.policy_joint_indexes]

    tokens = torch.cat([lin_vel, ang_vel, proj_gravity, joint_pos], dim=-1)
    k = len(self.reference_key_indexes)
    if k == 0:
      return tokens.flatten(1)

    n, t = tokens.shape[0], tokens.shape[1]
    inv = quat_inv(yaw_quat(self.robot_anchor_quat_w))
    inv = inv[:, None, None, :].expand(n, t, k, 4).reshape(-1, 4)
    parts = [tokens]

    if self.cfg.reference_key_body_pos:
      ref_pos_w = (
        self.motion.body_pos_w[idx][:, :, self.reference_key_indexes]
        + self._env.scene.env_origins[:, None, None, :]
      )
      rel = ref_pos_w - self.robot_anchor_pos_w[:, None, None, :]
      parts.append(quat_apply(inv, rel.reshape(-1, 3)).view(n, t, k * 3))

    if self.cfg.reference_key_body_vel:
      ref_vel_w = (
        self.motion.body_lin_vel_w[idx][:, :, self.reference_key_indexes]
        * self.speed[:, None, None, None]
      )
      parts.append(quat_apply(inv, ref_vel_w.reshape(-1, 3)).view(n, t, k * 3))

    return torch.cat(parts, dim=-1).flatten(1)

  @property
  def joint_pos(self) -> torch.Tensor:
    return self.motion.joint_pos[self.time_steps]

  @property
  def joint_vel(self) -> torch.Tensor:
    return self.motion.joint_vel[self.time_steps] * self.speed[:, None]

  @property
  def body_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]
    )

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self.time_steps]

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[self.time_steps] * self.speed[:, None, None]

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[self.time_steps] * self.speed[:, None, None]

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index]
      + self._env.scene.env_origins
    )

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

  ##
  # 机器人实测量
  ##

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.body_indexes]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.body_indexes]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

  ##
  # CommandTerm 接口
  ##

  def _update_metrics(self) -> None:
    self.metrics["error_anchor_pos"] = torch.norm(
      self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
    )
    self.metrics["error_anchor_rot"] = quat_error_magnitude(
      self.anchor_quat_w, self.robot_anchor_quat_w
    )
    self.metrics["error_body_pos"] = torch.norm(
      self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_rot"] = quat_error_magnitude(
      self.body_quat_relative_w, self.robot_body_quat_w
    ).mean(dim=-1)
    self.metrics["error_joint_pos"] = torch.norm(
      self.joint_pos - self.robot_joint_pos, dim=-1
    )
    self.metrics["error_joint_vel"] = torch.norm(
      self.joint_vel - self.robot_joint_vel, dim=-1
    )

  def _record_outcome(self, env_ids: torch.Tensor) -> None:
    """把这批回合的成败记到它们各自的**起点 bin** 上。

    采样单位和统计单位必须是同一个，否则加权依据和实际观测对不上：这里估的正是
    P(失败 | 从该 bin 起步)，而这恰好就是下次采样要用的量。
    """
    launched = self.bin_ids[env_ids] >= 0
    if not torch.any(launched):
      return
    bins = self.bin_ids[env_ids][launched]
    failed = self._env.termination_manager.terminated[env_ids][launched].float()

    n_bins = self.bins.num_bins
    visits = torch.bincount(bins, minlength=n_bins).float()
    fails = torch.bincount(bins, weights=failed, minlength=n_bins)
    # EMA 按**访问次数**推进而非仿真步：一个 bin 平均几百步才被抽中一次，按步衰减的话
    # 绝大多数 bin 会一直停在初值。同一 bin 在这批里出现 k 次就复合 k 步；没被访问到的
    # 衰减因子为 1，原样不动。
    decay = (1.0 - self.cfg.adaptive_alpha) ** visits
    self.bin_failed.mul_(decay).add_((1.0 - decay) * fails / visits.clamp(min=1.0))

  def _speed_balance_weights(self) -> torch.Tensor:
    """按**整条动作**的速度档做逆频率加权，纠正语料里高动态动作的稀缺。

    语料速度分布极端偏斜：按 clip 的 p90 速度归档，跑步及以上（>2 m/s）只占 4.4%
    的采样 bin，而这正好对上评测里 walk 109 s、run 11 s 的落差——策略不是学不会跑，
    是几乎没见过。

    档位必须按整条 clip 定、clip 内所有 bin 继承，不能按单个 bin 的均速定。
    走过的弯路：先按 bin 均速分档，实测把 sprint 压到 0.70x、walk 0.76x，只有
    run 拿到 1.06x —— 因为每条 clip 都含大量低速过渡帧（sprint 也有 61% 的帧
    <0.5 m/s，那是助跑准备和冲刺后减速），压低速档等于把所有动作的过渡段一起压了。
    原意是「多练跑步这类**动作**」，不是「多练跑步的**瞬间**」，高速帧脱离低速上下文
    反而有害。改成按 clip 归档后：sprint 3.90x、run 2.69x，而 walk/dance/jumps
    保持在 0.93~0.96x 基本不受影响。

    用 p90 而非均值刻画 clip：一条 sprint 的均速会被大段准备动作拉低，
    p90 才抓得住「这条到底跑没跑起来」。

    为什么不靠已有的失败率自适应：那套按 bin 估 P(失败)，而当前语料 24 h 切出
    84302 个 bin 配 1536 环境（比例 55，设计点是 4.3），EMA 时间常数 50 次访问，
    5000 iter 下每个 bin 平均只被访问约 15 次，估计量还没走出初值。日志里
    sampling_entropy 恒为 1.0000 就是这么来的。

    ``alpha=0`` 时权重恒为 1，行为与改动前完全一致，可无痛回退。
    """
    alpha = self.cfg.speed_balance_alpha
    if alpha <= 0.0:
      return torch.ones(self.bins.num_bins, device=self.device)

    vel = self.motion.body_lin_vel_w[:, 0, :2].norm(dim=-1)
    starts = self.motion.start_idx
    ends = starts + self.motion.num_frames
    p90 = torch.stack(
      [torch.quantile(vel[s:e], 0.9) for s, e in zip(starts.tolist(), ends.tolist())]
    )

    edges = torch.tensor([0.5, 1.0, 1.5, 2.0, 3.0], device=self.device)
    tier = torch.bucketize(p90, edges)[self.bins.motion_id]
    cnt = torch.bincount(tier, minlength=len(edges) + 1).float()
    freq = (cnt / cnt.sum()).clamp(min=1e-6)
    w = freq.pow(-alpha)
    w = w / (w * freq).sum()

    share = (w * freq) * 100
    print(
      f"[GMT] 速度重加权 alpha={alpha}（按 clip p90 归档）：各档份额 "
      + " / ".join(f"{s:.1f}%" for s in share.tolist())
    )
    return w[tier]

  def _sample_bins(self, num: int) -> torch.Tensor:
    """按「带上限的失败率 + 均匀分布」混合采样一批 bin。

    上限是关键：物理上做不到的片段（本机手臂在 25 N·m 下的冲刺、跳跃）失败率恒为 1，
    不封顶的话它们会把采样预算吃光，而再练也不会变好。
    """
    n_bins = self.bins.num_bins
    uniform = 1.0 / n_bins
    fail = self.bin_failed.clamp(max=self.cfg.adaptive_failure_cap)
    total = fail.sum()

    if self.cfg.sampling_mode == "uniform" or n_bins == 1 or total <= 0:
      probs = torch.full((n_bins,), uniform, device=self.device)
    else:
      blend = self.cfg.adaptive_uniform_ratio
      probs = (1.0 - blend) * (fail / total) + blend * uniform

    # 速度重加权独立于失败率：前者纠正语料的先天偏斜，后者追踪学习进度
    probs = probs * self.bin_speed_weight
    probs = probs / probs.sum()

    self.metrics["sampling_entropy"][:] = -(probs * (probs + 1e-12).log()).sum() / max(
      math.log(n_bins), 1e-6
    )
    self.metrics["sampling_failure_rate"][:] = self.bin_failed.mean()
    return torch.multinomial(probs, num, replacement=True)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self._record_outcome(env_ids)

    if self.forced_motion_id is not None:
      self.motion_ids[env_ids] = self.forced_motion_id
      self.bin_ids[env_ids] = -1
      self.phase[env_ids] = 0.0
    elif self.cfg.sampling_mode == "start":
      # 回放：每条动作等概率，从头播。bin_ids 保持 -1，不污染采样统计。
      self.motion_ids[env_ids] = torch.randint(
        self.motion.num_motions, (len(env_ids),), device=self.device
      )
      self.phase[env_ids] = 0.0
    else:
      bins = self._sample_bins(len(env_ids))
      self.bin_ids[env_ids] = bins
      self.motion_ids[env_ids] = self.bins.motion_id[bins]
      # bin 内均匀取起点，免得策略把 1 秒网格上的固定起始状态背下来。
      # span 按本条动作末帧截断，末尾那个 bin 会连并入的残帧一起覆盖到。
      start = self.bins.phase_start[bins].float()
      last = (self.motion.num_frames[self.motion_ids[env_ids]] - 1).float()
      span = torch.minimum(start + self.bins.frames, last) - start
      frac = sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)
      self.phase[env_ids] = start + frac * span

    lo, hi = self.cfg.speed_range
    self.speed[env_ids] = sample_uniform(lo, hi, (len(env_ids),), device=self.device)

    self._write_state_from_motion(env_ids)

  def _write_state_from_motion(self, env_ids: torch.Tensor) -> None:
    """把机器人摆到参考帧的位姿上，并加一点随机扰动。"""
    root_pos = self.body_pos_w[:, 0].clone()
    root_ori = self.body_quat_w[:, 0].clone()
    root_lin_vel = self.body_lin_vel_w[:, 0].clone()
    root_ang_vel = self.body_ang_vel_w[:, 0].clone()

    ranges = torch.tensor(
      [
        self.cfg.pose_range.get(k, (0.0, 0.0))
        for k in ("x", "y", "z", "roll", "pitch", "yaw")
      ],
      device=self.device,
    )
    samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_pos[env_ids] += samples[:, 0:3]
    root_ori[env_ids] = quat_mul(
      quat_from_euler_xyz(samples[:, 3], samples[:, 4], samples[:, 5]),
      root_ori[env_ids],
    )

    ranges = torch.tensor(
      [
        self.cfg.velocity_range.get(k, (0.0, 0.0))
        for k in ("x", "y", "z", "roll", "pitch", "yaw")
      ],
      device=self.device,
    )
    samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_lin_vel[env_ids] += samples[:, :3]
    root_ang_vel[env_ids] += samples[:, 3:]

    joint_pos = self.joint_pos.clone()
    joint_vel = self.joint_vel.clone()
    joint_pos += sample_uniform(
      self.cfg.joint_position_range[0],
      self.cfg.joint_position_range[1],
      joint_pos.shape,
      device=self.device,
    )
    limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos[env_ids] = torch.clip(joint_pos[env_ids], limits[:, :, 0], limits[:, :, 1])

    # 速度也要钐。关节角已经被 clip 到限位内，但速度一直是原样写入的，
    # 语料里一个 157 rad/s 的重定向假象就能把整个训练炸成 NaN。
    joint_vel.clamp_(-RESET_JOINT_VEL_LIMIT, RESET_JOINT_VEL_LIMIT)
    root_lin_vel.clamp_(-RESET_ROOT_LIN_VEL_LIMIT, RESET_ROOT_LIN_VEL_LIMIT)
    root_ang_vel.clamp_(-RESET_ROOT_ANG_VEL_LIMIT, RESET_ROOT_ANG_VEL_LIMIT)

    self.robot.write_joint_state_to_sim(
      joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids
    )
    self.robot.write_root_state_to_sim(
      torch.cat(
        [
          root_pos[env_ids],
          root_ori[env_ids],
          root_lin_vel[env_ids],
          root_ang_vel[env_ids],
        ],
        dim=-1,
      ),
      env_ids=env_ids,
    )
    self.robot.clear_state(env_ids=env_ids)

  def _update_command(self) -> None:
    self.phase += self.speed
    done = torch.where(self.phase >= self.motion.num_frames[self.motion_ids])[0]
    if done.numel() > 0:
      self._resample_command(done)

    n_bodies = len(self.cfg.body_names)
    anchor_pos = self.anchor_pos_w[:, None, :].repeat(1, n_bodies, 1)
    anchor_quat = self.anchor_quat_w[:, None, :].repeat(1, n_bodies, 1)
    robot_anchor_pos = self.robot_anchor_pos_w[:, None, :].repeat(1, n_bodies, 1)
    robot_anchor_quat = self.robot_anchor_quat_w[:, None, :].repeat(1, n_bodies, 1)

    # 参考姿态整体对齐到机器人当前的水平位置与偏航，只保留高度差。
    # 这样「站在哪里、朝哪边」不进入奖励，策略学的是相对运动。
    delta_pos = robot_anchor_pos
    delta_pos[..., 2] = anchor_pos[..., 2]
    delta_ori = yaw_quat(quat_mul(robot_anchor_quat, quat_inv(anchor_quat)))

    self.body_quat_relative_w = quat_mul(delta_ori, self.body_quat_w)
    self.body_pos_relative_w = delta_pos + quat_apply(
      delta_ori, self.body_pos_w - anchor_pos
    )

  def _debug_vis_impl(self, visualizer) -> None:
    """把参考动作画成半透明 ghost，叠在实际机器人上做对照。"""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    if self._ghost_model is None:
      # 碰撞几何设成全透明，只留视觉几何，否则 ghost 会糊成一团。
      self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
      for gi in range(self._ghost_model.ngeom):
        if (
          self._ghost_model.geom_contype[gi] != 0
          or self._ghost_model.geom_conaffinity[gi] != 0
        ):
          self._ghost_model.geom_rgba[gi, 3] = 0
        else:
          self._ghost_model.geom_rgba[gi] = (0.2, 0.8, 1.0, 0.45)

    indexing = self._env.scene[self.cfg.entity_name].indexing
    free_q = indexing.free_joint_q_adr.cpu().numpy()
    joint_q = indexing.joint_q_adr.cpu().numpy()

    for batch in env_indices:
      qpos = np.zeros(self._env.sim.mj_model.nq)
      qpos[free_q[0:3]] = self.body_pos_w[batch, 0].cpu().numpy()
      qpos[free_q[3:7]] = self.body_quat_w[batch, 0].cpu().numpy()
      qpos[joint_q] = self.joint_pos[batch].cpu().numpy()
      visualizer.add_ghost_mesh(qpos, model=self._ghost_model, label=f"ghost_{batch}")


@dataclass(kw_only=True)
class GeneralMotionCommandCfg(CommandTermCfg):
  motion_dir: str
  """存放 NPZ 动作的目录，目录里所有动作一起参与训练。"""
  anchor_body_name: str
  body_names: tuple[str, ...]
  """参与跟踪的刚体；**第一个必须是根节点**（前瞻观测按它取根位姿）。"""
  policy_joint_names: tuple[str, ...]
  entity_name: str
  lookahead_steps: tuple[int, ...] = (1, 5, 10, 15, 20, 30, 40, 55, 75, 95)
  """前瞻帧偏移（控制步）。50 Hz 下最远看到约 1.9 秒后。"""
  reference_window: int = 10
  """RGMT 局部参考窗口的单边半径 L，产出 2L+1 个 token。"""
  reference_stride: float = 1.5
  """窗口跨度系数：最远 token 落在 ``L*stride`` 帧外。1 = 稠密均匀，覆盖 ±L/50 秒。
  论文用 21 token 稠密窗口（±0.2 s），比本仓原来的稀疏前瞻（最远 1.9 s）短得多——
  RGMT 的设计是让 cross-attention 在局部窗口里按当前状态挑相关帧，靠的不是看得远。

  取 1.5（±0.3 s）是钩出注意力权重后定的：未来占 70.7%，峰值在 +0.22 s，
  而 0.4 s 以外加起来不到 5%。±0.3 s 刚好罩住峰值，再远就是白送 token。

  走过的弯路记在这：先按「起跳蓄力中位提前 0.500 s」把它设成 3（±0.6 s），
  结果训练 eplen 124→167 但评测 41.6→38.9；改成近密远疏后评测回到 42.1，
  仍未超基线。蓄力提前量的统计没错，错在假定策略能用上那么远的信息。"""
  reference_key_bodies: tuple[str, ...] = ()
  """参考 token 里额外给出笛卡尔位置的刚体，表达在**机器人当前** anchor 的 yaw 局部系下。

  空元组 = 关闭（token 维度退回 38，兼容旧 checkpoint）。"""
  reference_key_body_pos: bool = False
  """是否加入 key body 位置（每个 body 3 维）；该通道依赖里程计。"""
  reference_key_body_vel: bool = False
  """是否加入 key body 线速度（每个 body 3 维）；该通道只依赖 yaw 姿态。"""
  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  joint_position_range: tuple[float, float] = (-0.1, 0.1)
  bin_frames: int = 50
  """自适应采样的 bin 长度（帧）。50 Hz 下 = 1 秒，与 SONIC 对齐。

  bin 数要和并行环境数同量级：太粗则难点被同条动作里的简单段稀释，太细则单个 bin
  攒不够样本估失败率。当前语料 4.9 h → 约 17.6k 个 bin，配 4096 环境（SONIC 是
  2.2M bin 配 524k 环境，比例几乎相同）。
  """
  adaptive_uniform_ratio: float = 0.1
  """均匀分布在采样概率里占的比重，其余按失败率分配。"""
  adaptive_alpha: float = 0.02
  """失败率 EMA 的步长，按 bin 的**访问次数**计而非仿真步；0.02 ≈ 50 次访问的时间常数。"""
  adaptive_failure_cap: float = 0.5
  """失败率上限。超过它的 bin 一律等同看待。

  没有这个上限，物理上不可达的片段（本机手臂 25 N·m 撑不住的冲刺/跳跃）失败率恒为 1，
  会把采样预算全部吸走，而它们再练也不会变好。它同时是 ``bin_failed`` 的初值。
  """
  sampling_mode: Literal["adaptive", "uniform", "start"] = "adaptive"
  speed_balance_alpha: float = 0.5
  """按 clip 速度档做逆频率加权的强度。0 = 不加权（原行为），1 = 各档完全等份。

  按 clip p90 归档后各档 bin 份额是 41.6/25.1/23.2/5.7/3.0/1.4%，跑步及以上占 4.4%。
  取 0.5 把它抬到 13.5%（3 倍），bench 上 sprint 拿到 3.90x、run 2.69x，
  而 walk/dance/jumps 保持 0.93~0.96x 基本不动。不取 1.0 是因为 >3 m/s 档只有
  19 条 clip，等份会把它抬到 16.7%，几乎必然过拟合到那几条。
  """

  speed_range: tuple[float, float] = (1.0, 1.0)
  """播放倍率的采样区间，每回合一个。<1 即放慢。

  原速的 LAFAN1 里有大量动作在这台机器上物理不可达（肩关节 1.4 Hz 以上就力矩饱和）。
  早期对照：整体放慢 1.5 倍后，iter 12000 的回合长度从 12.91 涨到 50.44。
  这里改成每回合随机，让同一段动作能以不同速度反复练到，而不用把语料扩容好几倍。
  """

  def build(self, env: ManagerBasedRlEnv) -> GeneralMotionCommand:
    return GeneralMotionCommand(self, env)
