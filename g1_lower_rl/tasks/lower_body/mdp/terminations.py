"""下肢任务的终止条件。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from g1_lower_rl.tasks.lower_body.mdp.observations import height_above_feet

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def base_height_below_minimum(
  env: ManagerBasedRlEnv,
  minimum_height: float,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """机器人塌坐到膝盖上时终止。

  和 ``mdp.root_height_below_minimum`` 不同，这里以脚为基准，不会因为机器人走下坡就误触发。
  """
  asset: Entity = env.scene[asset_cfg.name]
  return height_above_feet(asset, asset_cfg.site_ids) < minimum_height
