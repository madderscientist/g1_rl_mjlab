"""Stage II 的 AMP（Adversarial Motion Priors, arXiv:2104.02180）。

判别器区分「策略的状态转移」和「参考语料的状态转移」，其输出作为风格奖励并入跟踪奖励。

为什么用它：上游跟踪奖励是逐帧 ``exp(-‖e‖²/σ²)``，对**系统性**偏差几乎无感——
拖着一只脚（差 0.107 m）摊到 14 个 body 上只让奖励掉 0.9%。判别器看的是状态转移的
**分布**，「每步都少抬 40%」这种系统性偏移对它是极强信号，恰恰是平方误差最钝的那类。
（同一思路的手写特例 ``motion_swing_lift_ratio`` 已验证有效：抬脚比 0.61 -> 0.79x。
AMP 的价值在于不必为每种风格缺陷手写一个奖励。）

只在 ``amp_coef > 0`` 时激活，否则行为与父类 PPO 完全一致。
"""

from __future__ import annotations

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_inv, yaw_quat
from rsl_rl.algorithms import PPO
from torch import nn


def _local_features(
  joint_pos: torch.Tensor,
  body_pos_w: torch.Tensor,
  anchor_pos_w: torch.Tensor,
  anchor_quat_w: torch.Tensor,
  root_lin_vel_w: torch.Tensor,
) -> torch.Tensor:
  """判别器观测：关节角 + 各 body 相对锚点的局部位置 + 根部线速度。

  **必须显式含末端相对位置**——判别器看不见的东西就约束不了，而抬脚正是要治的问题。
  全部转到锚点的 yaw 局部系，消掉全局位置与朝向，只留下姿态和步态本身。
  """
  inv = quat_inv(yaw_quat(anchor_quat_w))
  n_body = body_pos_w.shape[1]
  rel = body_pos_w - anchor_pos_w.unsqueeze(1)
  rel = quat_apply(inv.unsqueeze(1).expand(-1, n_body, -1), rel)
  vel = quat_apply(inv, root_lin_vel_w)
  return torch.cat([joint_pos, rel.flatten(1), vel], dim=-1)


class MotionExpertSampler:
  """从参考语料随机采相邻帧对，特征定义与策略侧共用 ``_local_features``。"""

  def __init__(self, motion, body_indexes: list[int], anchor_index: int) -> None:
    self._m = motion
    self._bi = body_indexes
    self._ai = anchor_index

  def feature_dim(self) -> int:
    return self._m.joint_pos.shape[-1] + len(self._bi) * 3 + 3

  def _at(self, t: torch.Tensor) -> torch.Tensor:
    m = self._m
    return _local_features(
      m.joint_pos[t],
      m.body_pos_w[t][:, self._bi],
      m.body_pos_w[t][:, self._ai],
      m.body_quat_w[t][:, self._ai],
      m.body_lin_vel_w[t][:, self._ai],
    )

  def __call__(self, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    # 少量跨 clip 边界的样本可忽略：80 个边界 / 88 万帧。
    hi = self._m.joint_pos.shape[0] - 1
    t = torch.randint(0, hi, (batch,), device=self._m.joint_pos.device)
    return self._at(t), self._at(t + 1)


class AmpDiscriminator(nn.Module):
  """最小二乘 GAN 判别器：专家 -> +1，策略 -> -1。"""

  def __init__(self, obs_dim: int, hidden: tuple[int, ...] = (1024, 512)) -> None:
    super().__init__()
    layers: list[nn.Module] = []
    last = obs_dim * 2  # (s, s') 拼接
    for h in hidden:
      layers += [nn.Linear(last, h), nn.ReLU()]
      last = h
    self.trunk = nn.Sequential(*layers)
    self.head = nn.Linear(last, 1)
    # 输出层小初始化：开局判别器接近 0，风格奖励约 0.75，避免一上来就压过跟踪奖励。
    nn.init.uniform_(self.head.weight, -0.01, 0.01)
    nn.init.zeros_(self.head.bias)

  def forward(self, s: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
    return self.head(self.trunk(torch.cat([s, s_next], dim=-1))).squeeze(-1)

  def style_reward(self, s: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
    """AMP 原文式 (7)：r = max(0, 1 - 0.25(D-1)²)，专家侧 D=1 时取满分。"""
    d = self.forward(s, s_next)
    return torch.clamp(1.0 - 0.25 * (d - 1.0).square(), min=0.0)

  def loss(
    self,
    policy_s: torch.Tensor,
    policy_s_next: torch.Tensor,
    expert_s: torch.Tensor,
    expert_s_next: torch.Tensor,
    grad_penalty_coef: float = 10.0,
  ) -> tuple[torch.Tensor, dict[str, float]]:
    d_policy = self.forward(policy_s, policy_s_next)
    expert_s = expert_s.detach().requires_grad_(True)
    expert_s_next = expert_s_next.detach().requires_grad_(True)
    d_expert = self.forward(expert_s, expert_s_next)

    loss = 0.5 * ((d_expert - 1.0).square().mean() + (d_policy + 1.0).square().mean())

    # 梯度惩罚不是可选项：去掉它判别器会迅速变得过强，风格奖励塌到 0，策略拿不到梯度。
    grad = torch.autograd.grad(
      d_expert.sum(), [expert_s, expert_s_next], create_graph=True
    )
    gp = sum(g.square().sum(dim=-1).mean() for g in grad)
    loss = loss + grad_penalty_coef * gp

    return loss, {
      "amp/d_expert": d_expert.mean().item(),
      "amp/d_policy": d_policy.mean().item(),
      "amp/grad_penalty": gp.detach().item(),
    }


class AmpPPO(PPO):
  """PPO + AMP。``amp_coef=0``（默认）时与父类行为一致。"""

  def __init__(self, *args, **kwargs) -> None:
    self.amp_coef: float = kwargs.pop("amp_coef", 0.0)
    self.amp_lr: float = kwargs.pop("amp_lr", 1e-4)
    self.amp_grad_penalty: float = kwargs.pop("amp_grad_penalty", 10.0)
    self.amp_epochs: int = kwargs.pop("amp_epochs", 1)
    super().__init__(*args, **kwargs)
    self.discriminator: AmpDiscriminator | None = None
    self.expert_sampler = None  # 可调用对象：batch_size -> (s, s')
    self._amp_optim: torch.optim.Optimizer | None = None
    self._amp_buf: list[tuple[torch.Tensor, torch.Tensor]] = []
    self._amp_stats: dict[str, float] = {}

  @property
  def amp_enabled(self) -> bool:
    return self.discriminator is not None and self.amp_coef > 0.0

  def attach_amp(self, obs_dim: int, expert_sampler, device: str) -> None:
    self.discriminator = AmpDiscriminator(obs_dim).to(device)
    self.expert_sampler = expert_sampler
    self._amp_optim = torch.optim.Adam(
      self.discriminator.parameters(), lr=self.amp_lr
    )

  def record_amp_pair(self, s: torch.Tensor, s_next: torch.Tensor) -> None:
    self._amp_buf.append((s.detach(), s_next.detach()))

  def style_reward(self, s: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
    if not self.amp_enabled:
      return torch.zeros(s.shape[0], device=s.device)
    assert self.discriminator is not None
    with torch.no_grad():
      return self.amp_coef * self.discriminator.style_reward(s, s_next)

  def update(self, *args, **kwargs):
    if self.amp_enabled and self._amp_buf:
      self._update_discriminator()
    out = super().update(*args, **kwargs)
    if isinstance(out, dict):
      out.update(self._amp_stats)
    return out

  def _update_discriminator(self) -> None:
    assert self.discriminator is not None and self._amp_optim is not None
    ps = torch.cat([p[0] for p in self._amp_buf])
    pn = torch.cat([p[1] for p in self._amp_buf])
    self._amp_buf.clear()

    n = ps.shape[0]
    for _ in range(self.amp_epochs):
      idx = torch.randperm(n, device=ps.device)[: min(n, 4096)]
      es, en = self.expert_sampler(idx.shape[0])
      loss, stats = self.discriminator.loss(
        ps[idx], pn[idx], es, en, self.amp_grad_penalty
      )
      self._amp_optim.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
      self._amp_optim.step()
    stats["amp/loss"] = loss.detach().item()
    self._amp_stats = stats
