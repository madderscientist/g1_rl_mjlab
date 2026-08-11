"""终止条件配置。"""

from __future__ import annotations

import math

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from g1_lower_rl.tasks.lower_body import mdp


def make_terminations() -> dict[str, TerminationTermCfg]:
  return {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
    # 以脚为基准而不是世界 z，走下坡时不会误触发。
    "collapsed": TerminationTermCfg(
      func=mdp.base_height_below_minimum,
      params={
        "minimum_height": 0.35,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # 按机器人设置。
      },
    ),
  }
