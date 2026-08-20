"""Stage II 的 PACE + STAR（Extreme-RGMT 第 V 节）。

PACE：并行环境按角色非对称分配——acquisition 用 PPO 攻 challenging，consolidation 用
对冻结 base policy 的行为克隆约束 mastered 上的漂移；约束强度随 acquisition 的有效
样本占比自适应上调。

STAR：把 bin 级难度先验下放到 transition 级，对高难组单独做 advantage 归一化，再在每个
高难 bin 内部挑高 advantage 的连续片段重采样进 mini-batch。

两者都只在显式提供 ``reference_actor`` / ``bin_weight_fn`` 时才生效，Stage I 不受影响。
"""

from __future__ import annotations

import copy
from typing import Protocol

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.modules.distribution import GaussianDistribution
from rsl_rl.storage.rollout_storage import RolloutStorage
from torch import nn


class BinWeightSource(Protocol):
  """STAR 的难度权重来源：调用得到 (T, num_envs) 的权重，``bins()`` 给对应的 bin 下标。"""

  def bins(self) -> torch.Tensor | None: ...

  def __call__(self) -> torch.Tensor | None: ...


class PaceStarPPO(PPO):
  """PPO + PACE + STAR。未配置 Stage II 组件时行为与父类一致。"""

  def __init__(self, *args, **kwargs) -> None:
    self.acquisition_fraction: float = kwargs.pop("acquisition_fraction", 0.8)
    self.lambda_base: float = kwargs.pop("lambda_base", 0.3)
    self.lambda_gain: float = kwargs.pop("lambda_gain", 5.0)
    self.rho_ref: float = kwargs.pop("rho_ref", 0.6)
    self.rho_beta: float = kwargs.pop("rho_beta", 0.99)
    self.star_topk_ratio: float = kwargs.pop("star_topk_ratio", 0.05)
    self.star_ratio: float = kwargs.pop("star_ratio", 0.25)
    super().__init__(*args, **kwargs)
    self.reference_actor: nn.Module | None = None
    self.bin_weight_fn: BinWeightSource | None = None  # None = 关闭 STAR
    self.pace_requested = False
    self._rho_bar = self.rho_ref
    self._lambda_con = self.lambda_base

  def freeze_reference(self) -> None:
    """把当前 actor 冻结成 π_ref。Stage II 开始时调一次。"""
    self.reference_actor = copy.deepcopy(self.actor)
    for p in self.reference_actor.parameters():
      p.requires_grad_(False)
    self.reference_actor.eval()

  @property
  def pace_enabled(self) -> bool:
    return self.reference_actor is not None

  def _env_is_acquisition(self, num_envs: int) -> torch.Tensor:
    n_acq = int(torch.ceil(torch.tensor(self.acquisition_fraction * num_envs)).item())
    m = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
    m[:n_acq] = True
    return m

  def compute_returns(self, obs) -> None:
    super().compute_returns(obs)
    if self.bin_weight_fn is None:
      return
    st = self.storage
    w = self.bin_weight_fn()
    if w is None or w.shape != (st.num_transitions_per_env, st.num_envs):
      return
    # STAR §V-B1：高难组与其余组各自归一化。共享统计量会把高难区的相对学习价值抹平。
    self._star_high = (w > 1.0).flatten(0, 1)
    adv = st.advantages.flatten(0, 1)
    self._star_raw_adv = adv.clone()
    for mask in (self._star_high, ~self._star_high):
      if mask.sum() > 1:
        sel = adv[mask]
        adv[mask] = (sel - sel.mean()) / (sel.std() + 1e-8)
    st.advantages = adv.view_as(st.advantages)

  def _star_pool(self) -> tuple[torch.Tensor, torch.Tensor] | None:
    """挑出高 advantage 的连续片段，返回 (可采样下标, 采样权重)。"""
    if self.bin_weight_fn is None or not hasattr(self, "_star_high"):
      return None
    st = self.storage
    T, N = st.num_transitions_per_env, st.num_envs
    w = self.bin_weight_fn()
    if w is None:
      return None
    dones = st.dones.view(T, N).bool()
    # fragment = 同一环境内被终止切开的连续段
    prev_done = torch.zeros_like(dones)
    prev_done[1:] = dones[:-1]
    frag = prev_done.cumsum(0)
    key = (torch.arange(N, device=self.device)[None, :] * (int(frag.max()) + 1) + frag).flatten()

    high = self._star_high
    if high.sum() == 0:
      return None
    bins = self.bin_weight_fn.bins()
    if bins is None:
      return None
    bins = bins.flatten()
    raw = self._star_raw_adv.squeeze(-1) if self._star_raw_adv.dim() > 1 else self._star_raw_adv

    idx = high.nonzero(as_tuple=False).squeeze(-1)
    # 按 (bin, fragment) 聚合平均 raw advantage，再在每个 bin 内取前 top-k
    pair = bins[idx] * (key.max() + 1) + key[idx]
    uniq, inv = torch.unique(pair, return_inverse=True)
    s = torch.zeros(len(uniq), device=self.device).index_add_(0, inv, raw[idx])
    c = torch.zeros(len(uniq), device=self.device).index_add_(0, inv, torch.ones_like(raw[idx]))
    q = s / c.clamp(min=1)
    ubin = torch.zeros(len(uniq), dtype=torch.long, device=self.device)
    ubin.scatter_(0, inv, bins[idx])

    keep = torch.zeros(len(uniq), dtype=torch.bool, device=self.device)
    for b in torch.unique(ubin):
      m = (ubin == b).nonzero(as_tuple=False).squeeze(-1)
      k = max(int(self.star_topk_ratio * len(m) + 0.999), 1)
      keep[m[q[m].topk(k).indices]] = True
    if not keep.any():
      return None

    sel_pairs = uniq[keep]
    in_pool = torch.isin(bins * (key.max() + 1) + key, sel_pairs)
    pool = in_pool.nonzero(as_tuple=False).squeeze(-1)
    weight = w.flatten(0, 1)[pool].clamp(min=1e-6)
    return pool, weight / weight.sum()

  def _generator(self):
    """与 rsl_rl 的 mini_batch_generator 等价，但额外交出扁平下标（角色/难度都靠它反推）。"""
    st = self.storage
    total = st.num_envs * st.num_transitions_per_env
    mb = total // self.num_mini_batches
    obs = st.observations.flatten(0, 1)
    actions = st.actions.flatten(0, 1)
    values = st.values.flatten(0, 1)
    returns = st.returns.flatten(0, 1)
    adv = st.advantages.flatten(0, 1)
    logp = st.actions_log_prob.flatten(0, 1)
    dparams = tuple(p.flatten(0, 1) for p in st.distribution_params)
    star = self._star_pool()

    for _ in range(self.num_learning_epochs):
      perm = torch.randperm(total, device=self.device)
      for i in range(self.num_mini_batches):
        idx = perm[i * mb : (i + 1) * mb]
        if star is not None and self.star_ratio > 0:
          pool, wgt = star
          n_star = min(mb, max(int(self.star_ratio * mb), 1))
          extra = pool[torch.multinomial(wgt, n_star, replacement=True)]
          idx = torch.cat([idx[: mb - n_star], extra])
        yield (
          RolloutStorage.Batch(
            observations=obs[idx],
            actions=actions[idx],
            values=values[idx],
            advantages=adv[idx],
            returns=returns[idx],
            old_actions_log_prob=logp[idx],
            old_distribution_params=tuple(p[idx] for p in dparams),
          ),
          idx,
        )

  def update(self) -> dict[str, float]:
    # 惰性冻结：第一次更新时权重必然已加载完毕，比挂在某个具体的 load 入口上更稳。
    if self.pace_requested and self.reference_actor is None:
      self.freeze_reference()
      print("[INFO]: PACE 已冻结参考策略 π_ref")
    if not self.pace_enabled and self.bin_weight_fn is None:
      return super().update()
    assert not (self.actor.is_recurrent or self.critic.is_recurrent), (
      "PACE/STAR 的分组依赖扁平下标反推环境，循环网络走的是按轨迹切分的生成器，不兼容"
    )
    assert self.rnd is None and self.symmetry is None, "PACE/STAR 未适配 RND / 镜像增广"
    # BC 项直接拿分布参数的第 0 项当均值用（省一次前向），这只对高斯成立。
    # Beta 分布的 params[0] 是 alpha，换了不会报错，只会静默把 BC 目标训歪。
    assert isinstance(self.actor.distribution, GaussianDistribution), "PACE 的行为克隆项依赖高斯分布的 params[0] 就是均值"

    st = self.storage
    num_envs = st.num_envs
    is_acq_env = self._env_is_acquisition(num_envs)
    self._update_lambda(is_acq_env)

    sums = {"value": 0.0, "surrogate": 0.0, "entropy": 0.0, "consolidation": 0.0}
    n_updates = 0
    for batch, idx in self._generator():
      env_of = idx % num_envs
      acq = is_acq_env[env_of]

      self.actor(batch.observations, stochastic_output=True)
      logp = self.actor.get_output_log_prob(batch.actions)
      values = self.critic(batch.observations)
      dparams = tuple(p for p in self.actor.output_distribution_params)
      entropy = self.actor.output_entropy
      self._adapt_lr(batch, dparams)

      ratio = torch.exp(logp - torch.squeeze(batch.old_actions_log_prob))
      a = torch.squeeze(batch.advantages)
      surr = torch.max(-a * ratio, -a * ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param))
      # acquisition 样本才进 PPO 目标；consolidation 样本只受行为克隆约束。
      surrogate_loss = _masked_mean(surr, acq)

      if self.use_clipped_value_loss:
        vc = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
        value_loss = torch.max((values - batch.returns).pow(2), (vc - batch.returns).pow(2)).mean()
      else:
        value_loss = (batch.returns - values).pow(2).mean()

      loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * _masked_mean(entropy, acq)

      con_loss = torch.zeros((), device=self.device)
      if self.pace_enabled and (~acq).any():
        with torch.no_grad():
          a_ref = self.reference_actor(batch.observations)
        # 复用上面那次前向的均值，别再前向一遍：高斯分布下 forward(obs) 返回的就是
        # mlp_output，而 update(mlp_output) 里 mean = mlp_output，两者同值同图。
        a_cur = dparams[0]
        con_loss = _masked_mean((a_cur - a_ref).pow(2).sum(-1), ~acq)
        loss = loss + self._lambda_con * con_loss

      self.optimizer.zero_grad()
      loss.backward()
      if self.is_multi_gpu:
        self.reduce_parameters()
      nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
      self.optimizer.step()

      sums["value"] += value_loss.item()
      sums["surrogate"] += surrogate_loss.item()
      sums["entropy"] += entropy.mean().item()
      sums["consolidation"] += con_loss.item()
      n_updates += 1

    st.clear()
    out = {k: v / max(n_updates, 1) for k, v in sums.items()}
    out["lambda_con"] = self._lambda_con
    out["rho_acquisition"] = self._rho_bar
    return out

  def _update_lambda(self, is_acq_env: torch.Tensor) -> None:
    """进度自适应权重：acquisition 有效样本越多，越该加大约束防遗忘（式 17-19）。"""
    if not self.pace_enabled:
      return
    st = self.storage
    valid = ~st.dones.view(st.num_transitions_per_env, st.num_envs).bool()
    n_a = float(valid[:, is_acq_env].sum())
    n_c = float(valid[:, ~is_acq_env].sum())
    rho = n_a / max(n_a + n_c, 1.0)
    self._rho_bar = self.rho_beta * self._rho_bar + (1.0 - self.rho_beta) * rho
    self._lambda_con = min(1.0, self.lambda_base + self.lambda_gain * max(0.0, self._rho_bar - self.rho_ref))

  def _adapt_lr(self, batch, dparams) -> None:
    if self.desired_kl is None or self.schedule != "adaptive":
      return
    with torch.inference_mode():
      kl = self.actor.get_kl_divergence(batch.old_distribution_params, dparams).mean()
      if self.is_multi_gpu:
        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
        kl /= self.gpu_world_size
      if self.gpu_global_rank == 0:
        if kl > self.desired_kl * 2.0:
          self.learning_rate = max(1e-5, self.learning_rate / 1.5)
        elif 0.0 < kl < self.desired_kl / 2.0:
          self.learning_rate = min(1e-2, self.learning_rate * 1.5)
      if self.is_multi_gpu:
        t = torch.tensor(self.learning_rate, device=self.device)
        torch.distributed.broadcast(t, src=0)
        self.learning_rate = t.item()
      for g in self.optimizer.param_groups:
        g["lr"] = self.learning_rate


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
  """按掩码求均值。全掩掉时返回 0 而不是 NaN——某个 mini-batch 恰好没有该角色是正常的。"""
  if not mask.any():
    return torch.zeros((), device=x.device)
  return x[mask].mean()
