"""RGMT 风格的 actor：三分支编码 + 因果历史编码器 + cross-attention + FSQ 瓶颈。

复现 Extreme-RGMT (arXiv:2607.20110) 第 IV-A 节。要点与"为什么"：

* **三分支各自 LayerNorm，而不是经验归一化。** 本体量、动作、参考指令的物理量纲和时间
  尺度都不同；用 running statistics 做整体归一化时，一旦语料分布变了（Stage II 专攻
  高动态）统计量就漂，而 LayerNorm 是逐样本的，不受分布迁移影响。
* **cross-attention 用历史当 query、参考窗口当 K/V。** 高动态动作里相位偏一点点，该看
  的参考帧就完全不同；固定前瞻偏移是"盲选"，注意力是按当前状态"挑"。
* **FSQ 压在聚合后的 command 特征上，而不是原始参考输入上。** 目的不是压缩输入，是把
  送进 actor 的指令表征限制成离散有界的，从而对参考轨迹的局部毛刺不敏感。

注意力和量化都用基础算子手写，保证 ``torch.onnx.export`` 能整图导出。
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
from mjlab.rl import RslRlModelCfg
from rsl_rl.models import MLPModel
from tensordict import TensorDict
from torch import nn


def _encoder(in_dim: int, hidden: int, out_dim: int, activation: str) -> nn.Module:
  act = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation]
  return nn.Sequential(nn.Linear(in_dim, hidden), act(), nn.Linear(hidden, out_dim), nn.LayerNorm(out_dim))


class _MultiHeadAttention(nn.Module):
  """手写多头注意力。用基础算子是为了整图能进 ONNX。"""

  def __init__(self, dim: int, num_heads: int) -> None:
    super().__init__()
    assert dim % num_heads == 0
    self.num_heads = num_heads
    self.head_dim = dim // num_heads
    self.q_proj = nn.Linear(dim, dim)
    self.k_proj = nn.Linear(dim, dim)
    self.v_proj = nn.Linear(dim, dim)
    self.out_proj = nn.Linear(dim, dim)

  def _split(self, x: torch.Tensor) -> torch.Tensor:
    n, t, _ = x.shape
    return x.view(n, t, self.num_heads, self.head_dim).transpose(1, 2)

  def forward(self, q: torch.Tensor, kv: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    qh = self._split(self.q_proj(q))
    kh = self._split(self.k_proj(kv))
    vh = self._split(self.v_proj(kv))
    scores = qh @ kh.transpose(-2, -1) / math.sqrt(self.head_dim)
    if mask is not None:
      scores = scores + mask
    out = torch.softmax(scores, dim=-1) @ vh
    n, _, t, _ = out.shape
    return self.out_proj(out.transpose(1, 2).reshape(n, t, -1))


class _CausalBlock(nn.Module):
  def __init__(self, dim: int, num_heads: int) -> None:
    super().__init__()
    self.attn = _MultiHeadAttention(dim, num_heads)
    self.norm1 = nn.LayerNorm(dim)
    self.norm2 = nn.LayerNorm(dim)
    self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

  def forward(self, x: torch.Tensor, mask: torch.Tensor, last_only: bool = False) -> torch.Tensor:
    h = self.norm1(x)
    if last_only:
      # 只有最后一个位置的输出会被用到。因果掩码下它本来就能看到全部历史，
      # 所以取单 query 与算完整层再取最后一格**数学等价**，省掉其余位置的
      # 注意力输出投影和整个 FFN。只对最后一层有效：前面的层其输出会被
      # 下一层的 query 全部读到，不能省。
      x = x[:, -1:] + self.attn(h[:, -1:], h)
    else:
      x = x + self.attn(h, h, mask)
    return x + self.ff(self.norm2(x))


def _fsq(x: torch.Tensor, levels: int) -> torch.Tensor:
  """有限标量量化，直通梯度。输出规范化到 [-1, 1]。"""
  half = (levels - 1) / 2.0
  z = torch.tanh(x) * half
  zq = torch.round(z)
  return (z + (zq - z).detach()) / half


class RgmtActor(MLPModel):
  """按 RGMT 结构消费多个命名观测组。

  观测保持 2D 扁平（rsl_rl 的硬性要求），token 结构在这里 reshape 还原：
  每个本体分支单独成组，``组维度 / history_len`` 就是它的每拍维度。
  """

  is_recurrent: bool = False

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (1024, 1024, 512, 256),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict | None = None,
    history_len: int = 10,
    reference_group: str = "rg_reference",
    action_group: str = "rg_actions",
    reference_dim: int = 38,
    token_dim: int = 64,
    num_heads: int = 4,
    history_layers: int = 2,
    fsq_levels: int = 5,
    fast_last_layer: bool = True,
  ) -> None:
    if obs_normalization:
      raise ValueError(
        "RGMT 用逐分支 LayerNorm 取代经验归一化：两者叠加会让 LayerNorm 白做，且经验统计量在 Stage II 换语料后会漂。"
      )
    self._history_len = history_len
    self._reference_group = reference_group
    self._action_group = action_group
    self._reference_dim = reference_dim
    self._token_dim = token_dim

    groups = list(obs_groups[obs_set])
    for name in (reference_group, action_group):
      if name not in groups:
        raise ValueError(f"观测组 {name!r} 不在 obs_groups[{obs_set!r}] = {groups}")
    self._prop_groups = [g for g in groups if g not in (reference_group, action_group)]

    self._prop_dims = [obs[g].shape[-1] // history_len for g in self._prop_groups]
    self._prop_dim = sum(self._prop_dims)
    self._action_dim_obs = obs[action_group].shape[-1] // history_len
    self._num_ref_tokens = obs[reference_group].shape[-1] // reference_dim

    super().__init__(
      obs=obs,
      obs_groups=obs_groups,
      obs_set=obs_set,
      output_dim=output_dim,
      hidden_dims=hidden_dims,
      activation=activation,
      obs_normalization=False,
      distribution_cfg=distribution_cfg,
    )

    self.state_encoder = _encoder(self._prop_dim, 128, token_dim, activation)
    self.action_encoder = _encoder(self._action_dim_obs, 64, token_dim, activation)
    self.command_encoder = _encoder(reference_dim, 128, token_dim, activation)
    self.pos_embedding = nn.Parameter(torch.zeros(1, self._num_ref_tokens, token_dim))
    nn.init.normal_(self.pos_embedding, std=0.02)

    self.history_blocks = nn.ModuleList([_CausalBlock(token_dim, num_heads) for _ in range(history_layers)])
    self.history_norm = nn.LayerNorm(token_dim)
    self.cross_attn = _MultiHeadAttention(token_dim, num_heads)
    self.fsq_levels = fsq_levels
    self.fast_last_layer = fast_last_layer

    seq = 2 * history_len
    mask = torch.full((seq, seq), float("-inf")).triu(1)
    self.register_buffer("causal_mask", mask.view(1, 1, seq, seq), persistent=False)

  def _get_latent_dim(self) -> int:
    # actor 头吃 [o_prop_t, a_{t-1}, û_t]，对应论文式 (11)。
    return self._prop_dim + self._action_dim_obs + self._token_dim

  def _split_history(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
    n = obs[self._prop_groups[0]].shape[0]
    prop = torch.cat(
      [obs[g].view(n, self._history_len, d) for g, d in zip(self._prop_groups, self._prop_dims)],
      dim=-1,
    )
    act = obs[self._action_group].view(n, self._history_len, self._action_dim_obs)
    return prop, act

  def get_latent(self, obs: TensorDict, masks=None, hidden_state=None) -> torch.Tensor:
    prop, act = self._split_history(obs)
    n = prop.shape[0]

    z_o = self.state_encoder(prop)
    z_a = self.action_encoder(act)
    # 交错成 [a_{τ-1}, o_τ, ...]：动作在前，因为它是产生该观测的原因。
    seq = torch.stack([z_a, z_o], dim=2).reshape(n, 2 * self._history_len, -1)
    last = len(self.history_blocks) - 1
    for i, block in enumerate(self.history_blocks):
      seq = block(seq, self.causal_mask, last_only=self.fast_last_layer and i == last)
    h = self.history_norm(seq[:, -1:, :])

    ref = obs[self._reference_group].view(n, self._num_ref_tokens, self._reference_dim)
    z_g = self.command_encoder(ref) + self.pos_embedding
    u = self.cross_attn(h, z_g).squeeze(1)
    u_hat = _fsq(u, self.fsq_levels)

    return torch.cat([prop[:, -1], act[:, -1], u_hat], dim=-1)

  def update_normalization(self, obs: TensorDict) -> None:
    pass

  def as_onnx(self, verbose: bool) -> nn.Module:
    return _OnnxRgmtActor(self, verbose)


class _OnnxRgmtActor(nn.Module):
  """把每个观测组做成一个独立的 ONNX 输入，部署端按名字喂。"""

  is_recurrent: bool = False

  def __init__(self, model: RgmtActor, verbose: bool) -> None:
    super().__init__()
    self.verbose = verbose
    # 必须 deepcopy：rsl_rl 导出时会对本模块 ``.to("cpu")``，共享引用会把训练中的
    # 模型一起搬下 GPU。而直接 deepcopy 会挂在 ``GaussianDistribution._distribution``
    # 缓存的 ``Normal`` 上——它持有上一次 forward 的非叶子张量。先摘掉再复原。
    dist = model.distribution
    cached = getattr(dist, "_distribution", None) if dist is not None else None
    if cached is not None:
      dist._distribution = None
    try:
      self.model = copy.deepcopy(model)
    finally:
      if cached is not None:
        dist._distribution = cached
    self.model.eval()
    if model.distribution is not None:
      self.deterministic_output = model.distribution.as_deterministic_output_module()
    else:
      self.deterministic_output = nn.Identity()
    self._groups = list(model._prop_groups) + [model._action_group, model._reference_group]
    self._sizes = [
      *(d * model._history_len for d in model._prop_dims),
      model._action_dim_obs * model._history_len,
      model._num_ref_tokens * model._reference_dim,
    ]

  def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
    obs = TensorDict(dict(zip(self._groups, inputs)), batch_size=inputs[0].shape[:1])
    return self.deterministic_output(self.model.mlp(self.model.get_latent(obs)))

  def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
    return tuple(torch.zeros(1, s) for s in self._sizes)

  @property
  def input_names(self) -> list[str]:
    return list(self._groups)

  @property
  def output_names(self) -> list[str]:
    return ["actions"]


@dataclass
class RgmtModelCfg(RslRlModelCfg):
  """``RslRlModelCfg`` 加上 RGMT 专属字段。

  rsl_rl 最后是 ``**cfg["actor"]`` splat 进模型构造函数，所以字段名必须和
  ``RgmtActor.__init__`` 的形参对得上。
  """

  history_len: int = 10
  reference_group: str = "rg_reference"
  action_group: str = "rg_actions"
  reference_dim: int = 38
  token_dim: int = 64
  num_heads: int = 4
  history_layers: int = 2
  fsq_levels: int = 5
  fast_last_layer: bool = True
