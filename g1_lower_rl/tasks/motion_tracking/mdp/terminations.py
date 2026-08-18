"""GMT 任务的终止项。"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def with_grace(
  env: "ManagerBasedRlEnv",
  base_func: Callable[..., torch.Tensor],
  grace_steps: int,
  **params,
) -> torch.Tensor:
  """复位后的前 ``grace_steps`` 步屏蔽 ``base_func``。

  只该用在"跟丢"类判据上。加宽初始随机化后，偏离姿态本身就意味着末端位置误差超阈值，
  复位当拍即终止——那等于把"从偏离姿态追回参考"这件事定义成了失败，策略永远学不到。
  宽限期给策略留出收敛时间（0.3 s 量级，够 PD 把 0.3 rad 的关节偏差拉回来）。

  倒地判据不要用它包：真摔了就该立刻结束，宽限只会制造无意义的样本。
  """
  return base_func(env, **params) & (env.episode_length_buf >= grace_steps)
