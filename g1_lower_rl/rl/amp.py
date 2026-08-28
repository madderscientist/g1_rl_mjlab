"""Stage II 的 AMP（Adversarial Motion Priors, arXiv:2104.02180）。

判别器区分「策略的状态转移」和「参考语料的状态转移」，其输出作为风格奖励并入跟踪奖励。

为什么用它：上游跟踪奖励是逐帧 ``exp(-‖e‖²/σ²)``，对**系统性**偏差几乎无感——
拖着一只脚（差 0.107 m）摊到 14 个 body 上只让奖励掉 0.9%。判别器看的是状态转移的
**分布**，「每步都少抬 40%」这种系统性偏移对它是极强信号。

实现对齐官方 ``nv-tlabs/ASE``（AMP 原作者）。第一版自研漏了四处，导致判别器学不动、
风格奖励塌成常数（分离度 0.045/-0.040），反而稀释跟踪奖励让抬脚倒退 13%：

1. **观测归一化**。关节角 ~1 rad、位置 ~0.1 m、速度 ~1 m/s 差一个数量级，
   不归一化判别器光适应尺度就耗光容量。
2. **replay buffer**。只拿当前策略当负样本，判别器会追着策略跑、过拟合瞬时分布。
3. **风格奖励用 ``-log(1-D)``** 而非有界的最小二乘式——后者接近饱和就没梯度。
4. **logit 正则**，抑制输出层权重爆炸。

官方源码另点出一个陷阱：判别器可能只学会「抖动」——策略带 σ≈0.29 探索噪声、参考数据
平滑，光检测抖动就能分开，根本不看抬脚。对策是用部分确定性动作产生平滑轨迹训判别器。

只在 ``amp_coef > 0`` 时激活，否则行为与父类 PPO 完全一致。
"""

from __future__ import annotations

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul, yaw_quat
from rsl_rl.algorithms import PPO
from torch import nn

DISC_LOGIT_INIT_SCALE = 1.0


def _tan_norm(q: torch.Tensor) -> torch.Tensor:
  """官方 ``quat_to_tan_norm``：用旋转后的 x 轴与 z 轴表示姿态，避免四元数双覆盖。"""
  tan = torch.zeros_like(q[..., :3])
  tan[..., 0] = 1.0
  nrm = torch.zeros_like(q[..., :3])
  nrm[..., 2] = 1.0
  return torch.cat([quat_apply(q, tan), quat_apply(q, nrm)], dim=-1)


def _local_features(
  joint_pos: torch.Tensor,
  joint_vel: torch.Tensor,
  body_pos_w: torch.Tensor,
  anchor_pos_w: torch.Tensor,
  anchor_quat_w: torch.Tensor,
  root_lin_vel_w: torch.Tensor,
  root_ang_vel_w: torch.Tensor,
) -> torch.Tensor:
  """判别器观测，逐项对齐官方 ``build_amp_observations``。

  官方顺序：``[root_h, root_rot_6d, local_root_vel, local_root_ang_vel,
  dof_pos, dof_vel, local_key_body_pos]``。全部转到锚点的 yaw 局部系，消掉全局位置
  与朝向，只留姿态与步态本身。

  **关节速度不能省**：第一版漏了它，判别器只能看静态姿态、看不到运动动态，
  而 AMP 的全部价值就在动态风格。G1 全是 1 自由度转动关节，官方的 ``dof_to_obs``
  在这里退化为恒等，所以关节角直接用原值。
  """
  inv = quat_inv(yaw_quat(anchor_quat_w))
  n_body = body_pos_w.shape[1]
  rel = body_pos_w - anchor_pos_w.unsqueeze(1)
  rel = quat_apply(inv.unsqueeze(1).expand(-1, n_body, -1), rel)
  return torch.cat(
    [
      anchor_pos_w[:, 2:3],
      _tan_norm(quat_mul(inv, anchor_quat_w)),
      quat_apply(inv, root_lin_vel_w),
      quat_apply(inv, root_ang_vel_w),
      joint_pos,
      joint_vel,
      rel.flatten(1),
    ],
    dim=-1,
  )


class MotionExpertSampler:
  """从参考语料随机采相邻帧对，特征定义与策略侧共用 ``_local_features``。"""

  def __init__(self, motion, body_indexes: list[int], anchor_index: int) -> None:
    self._m = motion
    self._bi = body_indexes
    self._ai = anchor_index

  def feature_dim(self) -> int:
    n_joint = self._m.joint_pos.shape[-1]
    return 1 + 6 + 3 + 3 + n_joint * 2 + len(self._bi) * 3

  def _at(self, t: torch.Tensor) -> torch.Tensor:
    m = self._m
    return _local_features(
      m.joint_pos[t],
      m.joint_vel[t],
      m.body_pos_w[t][:, self._bi],
      m.body_pos_w[t][:, self._ai],
      m.body_quat_w[t][:, self._ai],
      m.body_lin_vel_w[t][:, self._ai],
      m.body_ang_vel_w[t][:, self._ai],
    )

  def __call__(self, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    # 少量跨 clip 边界的样本可忽略：80 个边界 / 88 万帧。
    hi = self._m.joint_pos.shape[0] - 1
    t = torch.randint(0, hi, (batch,), device=self._m.joint_pos.device)
    return self._at(t), self._at(t + 1)


class _RunningMeanStd(nn.Module):
  """判别器观测的在线标准化，策略侧与专家侧共用同一份统计量。"""

  def __init__(self, dim: int, eps: float = 1e-4) -> None:
    super().__init__()
    self.register_buffer("mean", torch.zeros(dim))
    self.register_buffer("var", torch.ones(dim))
    self.register_buffer("count", torch.tensor(eps))

  @torch.no_grad()
  def update(self, x: torch.Tensor) -> None:
    bm, bv, bc = x.mean(0), x.var(0, unbiased=False), x.shape[0]
    delta = bm - self.mean
    tot = self.count + bc
    self.mean += delta * bc / tot
    self.var = (
      self.var * self.count + bv * bc + delta.square() * self.count * bc / tot
    ) / tot
    self.count = tot

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return (x - self.mean) / torch.sqrt(self.var + 1e-5)


class AmpDiscriminator(nn.Module):
  """BCE 判别器：专家 -> logit>0，策略 -> logit<0（与官方一致）。"""

  def __init__(self, obs_dim: int, hidden: tuple[int, ...] = (1024, 512)) -> None:
    super().__init__()
    self.norm = _RunningMeanStd(obs_dim)
    layers: list[nn.Module] = []
    last = obs_dim * 2  # (s, s') 拼接
    for h in hidden:
      layers += [nn.Linear(last, h), nn.ReLU()]
      last = h
    self.trunk = nn.Sequential(*layers)
    self.logits = nn.Linear(last, 1)
    nn.init.uniform_(self.logits.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
    nn.init.zeros_(self.logits.bias)

  def forward(self, s: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
    x = torch.cat([self.norm(s), self.norm(s_next)], dim=-1)
    return self.logits(self.trunk(x)).squeeze(-1)

  def style_reward(self, s: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
    """官方式：r = -log(1 - sigmoid(D))。**无上界**，越像专家梯度越持续。

    有界形式（如 ``1-0.25(D-1)²``）一接近饱和就没梯度，实测会让风格奖励塌成常数、
    退化成生存奖励并稀释跟踪信号。
    """
    prob = torch.sigmoid(self.forward(s, s_next))
    return -torch.log(torch.clamp(1.0 - prob, min=1e-4))


class AmpPPO(PPO):
  """PPO + AMP。``amp_coef=0``（默认）时与父类行为一致。"""

  def __init__(self, *args, **kwargs) -> None:
    self.amp_coef: float = kwargs.pop("amp_coef", 0.0)
    self.amp_task_w: float = kwargs.pop("amp_task_w", 0.5)
    self.amp_lr: float = kwargs.pop("amp_lr", 1e-4)
    self.amp_grad_penalty: float = kwargs.pop("amp_grad_penalty", 5.0)
    self.amp_logit_reg: float = kwargs.pop("amp_logit_reg", 0.05)
    self.amp_epochs: int = kwargs.pop("amp_epochs", 2)
    self.amp_replay_size: int = kwargs.pop("amp_replay_size", 200_000)
    super().__init__(*args, **kwargs)
    self.discriminator: AmpDiscriminator | None = None
    self.expert_sampler = None
    self._amp_optim: torch.optim.Optimizer | None = None
    self._amp_buf: list[tuple[torch.Tensor, torch.Tensor]] = []
    self._replay: tuple[torch.Tensor, torch.Tensor] | None = None
    self._amp_stats: dict[str, float] = {}

  @property
  def amp_enabled(self) -> bool:
    return self.discriminator is not None and self.amp_coef > 0.0

  def attach_amp(self, obs_dim: int, expert_sampler, device: str) -> None:
    self.discriminator = AmpDiscriminator(obs_dim).to(device)
    self.expert_sampler = expert_sampler
    self._amp_optim = torch.optim.Adam(self.discriminator.parameters(), lr=self.amp_lr)

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

  def _push_replay(self, s: torch.Tensor, sn: torch.Tensor) -> None:
    if self._replay is not None:
      s = torch.cat([self._replay[0], s])
      sn = torch.cat([self._replay[1], sn])
    if s.shape[0] > self.amp_replay_size:
      keep = torch.randperm(s.shape[0], device=s.device)[: self.amp_replay_size]
      s, sn = s[keep], sn[keep]
    self._replay = (s, sn)

  def _update_discriminator(self) -> None:
    assert self.discriminator is not None and self._amp_optim is not None
    d = self.discriminator
    ps = torch.cat([p[0] for p in self._amp_buf])
    pn = torch.cat([p[1] for p in self._amp_buf])
    self._amp_buf.clear()
    d.norm.update(ps)

    bs = min(ps.shape[0], 4096)
    bce = nn.BCEWithLogitsLoss()
    for _ in range(self.amp_epochs):
      i = torch.randperm(ps.shape[0], device=ps.device)[:bs]
      agent_s, agent_n = ps[i], pn[i]
      # 掺入历史策略样本，否则判别器会追着当前策略跑、过拟合瞬时分布。
      if self._replay is not None:
        j = torch.randperm(self._replay[0].shape[0], device=ps.device)[:bs]
        agent_s = torch.cat([agent_s, self._replay[0][j]])
        agent_n = torch.cat([agent_n, self._replay[1][j]])

      es, en = self.expert_sampler(bs)
      es = es.detach().requires_grad_(True)
      en = en.detach().requires_grad_(True)

      d_agent = d(agent_s, agent_n)
      d_expert = d(es, en)
      loss = 0.5 * (
        bce(d_agent, torch.zeros_like(d_agent))
        + bce(d_expert, torch.ones_like(d_expert))
      )

      # 梯度惩罚只加在专家侧，与官方一致。
      grad = torch.autograd.grad(
        d_expert, [es, en], grad_outputs=torch.ones_like(d_expert), create_graph=True
      )
      gp = sum(g.square().sum(dim=-1).mean() for g in grad)
      loss = loss + self.amp_grad_penalty * gp + self.amp_logit_reg * d.logits.weight.square().sum()

      self._amp_optim.zero_grad()
      loss.backward()
      self._amp_optim.step()

    self._push_replay(ps, pn)
    with torch.no_grad():
      self._amp_stats = {
        "amp/d_expert": d_expert.mean().item(),
        "amp/d_agent": d_agent.mean().item(),
        # 准确率才是判别器还在工作的直接证据：塌到 0.5 就等于在给白噪声加分。
        "amp/acc_expert": (d_expert > 0).float().mean().item(),
        "amp/acc_agent": (d_agent < 0).float().mean().item(),
        "amp/grad_penalty": gp.detach().item(),
        "amp/loss": loss.detach().item(),
      }
