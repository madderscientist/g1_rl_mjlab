"""奖励项之间共用的小工具。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

ROBOT = SceneEntityCfg("robot")


def moving(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  """“有没有叫它动”的判别码。指令强度 = |水平速度| + |转向速率|，超过阈值返回 1。"""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  total = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  return (total > threshold).float()


def command_restarted(
  env: ManagerBasedRlEnv, term, prev_left: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
  """指令刚重采（``time_left`` 变大）或刚 reset 的环境掩码，以及新的 ``time_left`` 快照。

  靠重采划窗口的累计类奖励都要用它归零。
  """
  left = term.time_left
  if prev_left is None:
    prev_left = left
  fresh = (left > prev_left + 1e-6) | (env.episode_length_buf <= 1)
  return fresh, left.clone()
