"""GMT 任务自己的观测项。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def motion_reference_window(env: "ManagerBasedRlEnv", command_name: str = "motion") -> torch.Tensor:
  """RGMT 局部参考窗口，(num_envs, (2L+1) * 38)。

  扁平化只是为了走 rsl_rl 的 1D 观测通道，token 结构由模型内部 reshape 还原。
  """
  return env.command_manager.get_term(command_name).reference_tokens
