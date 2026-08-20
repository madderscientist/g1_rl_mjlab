"""GMT 任务的终止项。"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _shared_termination_coin(env: "ManagerBasedRlEnv", p_term: float) -> torch.Tensor:
  """同一控制步内所有判据共用一次伯努利采样。

  三个失败判据若各自独立抛硬币，实际终止概率会变成 1-(1-p)^3，期望恢复窗口
  从 200 步缩到 67 步（1.3 s），根本不够爬起来。论文式 (2) 的语义是
  「任一条件满足则以 p_term 终止」，所以必须共享同一次采样。
  """
  step = int(env.common_step_counter)
  cache = getattr(env, "_pt_coin_cache", None)
  if cache is not None and cache[0] == step and cache[1].shape[0] == env.num_envs:
    return cache[1]
  coin = torch.rand(env.num_envs, device=env.device) < p_term
  env._pt_coin_cache = (step, coin)
  return coin


def with_probabilistic_termination(
  env: "ManagerBasedRlEnv",
  base_func: Callable[..., torch.Tensor],
  p_term: float,
  **params,
) -> torch.Tensor:
  """跟踪失败后不立即复位，而是每步以 ``p_term`` 概率终止（Stubborn 式软终止）。

  硬终止让策略永远见不到「已经摔了」之后的状态，恢复行为无从学起。改成概率终止后，
  失败态会持续若干步进入 rollout，跟踪奖励自然把机器人往参考姿态上拉，
  恢复行为作为副产物涌现——不需要任何恢复专用奖励或独立的起身策略。

  ``p_term`` 由物理恢复时间反推而非拍脑袋：存活步数服从几何分布 Geo(p_term)，
  令其期望等于恢复所需时间，即 1/p_term = f_ctrl * t_rec。50 Hz 下取 t_rec=4 s
  得 p_term=0.005（期望窗口 200 步）。窗口太短会退化成硬终止，太长则回合被
  垃圾样本淹没。

  用几何分布而非固定延迟是关键：无记忆性让 TD 更新保持一致，
  固定延迟会在截断处引入价值估计偏差。

  每步重新判断而不锁定：机器人若真爬回阈值内就不再面临终止——这正是要学的东西。

  出处：Stubborn (arXiv:2606.12814) 3.1 节，式 (2)(3)。
  """
  return base_func(env, **params) & _shared_termination_coin(env, p_term)
