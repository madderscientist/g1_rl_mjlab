"""下肢任务的每步指标与分布式课程同步钩子。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.metrics import mean_action_acc

from g1_lower_rl.tasks.lower_body.mdp.curriculums import (
  synchronize_distributed_curriculum,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def synchronized_mean_action_acc(env: ManagerBasedRlEnv) -> torch.Tensor:
  """同步多卡课程，再计算原有的动作加速度指标。"""
  synchronize_distributed_curriculum(env)
  return mean_action_acc(env)