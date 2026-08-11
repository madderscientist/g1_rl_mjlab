"""站立任务的自定义奖励项与课程项。

这个任务只有一个真正的目标：**用尽量小的力矩站住，并且把重心压在两脚之间**。
其余各项都是为了让「站住」这件事有唯一解——否则最省力的策略是直接坐到地上。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.string import resolve_matching_names_values

from g1_lower_rl.tasks.lower_body import mdp as lower_body_mdp

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.reward_manager import RewardTermCfg


class gated_event_params(lower_body_mdp.gated_event_params):
  """速度任务那套逐档抬扰动，闸门改看「还站着吗」。

  父类的闸门量的是速度跟随误差，本任务没有速度指令，直接复用会在 ``command_manager``
  上取空。判据换成骨盆倾角：``projected_gravity_b`` 的水平分量模长就是 sin(倾角)，
  终止阈值 0.8 rad 对应 0.717，所以 ``max_tilt`` 取 0.25（约 14 度）留足余量——
  能稳在这个范围里才说明上一档真的扛住了。
  """

  def _local_counts(
    self,
    env: ManagerBasedRlEnv,
    event_name: str,
    stages: list,
    log_key: str,
    gate: dict[str, Any],
  ) -> tuple[torch.Tensor, torch.Tensor]:
    del event_name, stages, log_key
    gravity = env.scene["robot"].data.projected_gravity_b
    tilt = torch.norm(gravity[:, :2], dim=1)
    return (tilt < gate["max_tilt"]).sum(), tilt.new_tensor(tilt.numel())


class effort_cost:
  """按各关节力矩上限归一化后的平均力矩平方。

  不归一化的话这一项会被膝盖（139 N·m）完全主导，脚踝（50 N·m）怎么动都看不出来，
  而站立时真正决定省不省力的恰恰是踝和髋的配平。归一化之后各关节比的是「用掉了自己
  额定力矩的百分之多少」，这才是「轻松」该有的含义。
  """

  def __init__(self, cfg: "RewardTermCfg", env: "ManagerBasedRlEnv"):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)
    _, _, limits = resolve_matching_names_values(
      data=cfg.params["effort_limits"], list_of_strings=joint_names
    )
    self.inv_limit = 1.0 / torch.tensor(
      limits, device=env.device, dtype=torch.float32
    ).clamp(min=1e-3)

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg,
    effort_limits: dict[str, float],
  ) -> torch.Tensor:
    del effort_limits  # 只在构造时用到
    asset: Entity = env.scene[asset_cfg.name]
    frac = asset.data.actuator_force[:, asset_cfg.joint_ids] * self.inv_limit
    return torch.mean(torch.square(frac), dim=-1)


def com_over_feet(
  env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg, std: float
) -> torch.Tensor:
  """重心水平投影落在两脚中点附近的程度，exp 形。

  用两脚中点而不是严格的支撑多边形：多边形内的点都「合法」，但只有靠近中心才对
  左右两侧的扰动都有余量。写成 exp 而不是罚项，是为了让它和省力项能直接争夺权重——
  纯罚项在策略摔倒后反而变小，会奖励放弃。
  """
  asset: Entity = env.scene[asset_cfg.name]
  com_xy = asset.data.root_com_pos_w[:, :2]
  feet_xy = asset.data.site_pos_w[:, asset_cfg.site_ids, :2]
  offset = torch.norm(com_xy - feet_xy.mean(dim=1), dim=-1)
  return torch.exp(-torch.square(offset / std))


def com_inside_feet_margin(
  env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """重心越过两脚连线中点向外的距离（超出脚间距一半的部分才计）。

  ``com_over_feet`` 在重心刚要出界时已经很平，光靠它压不住缓慢的侧向漂移；
  这一项只在真的快出支撑面时才起作用，专门负责最后那段。
  """
  asset: Entity = env.scene[asset_cfg.name]
  feet_xy = asset.data.site_pos_w[:, asset_cfg.site_ids, :2]
  mid = feet_xy.mean(dim=1)
  half_span = 0.5 * torch.norm(feet_xy[:, 0] - feet_xy[:, 1], dim=-1)
  offset = torch.norm(asset.data.root_com_pos_w[:, :2] - mid, dim=-1)
  return torch.clamp(offset - half_span, min=0.0)


def base_stillness(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """躯干平动 + 转动速度的平方和。站着不动才是目标，方向不重要。"""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_lin_vel_w), dim=-1) + torch.sum(
    torch.square(asset.data.root_link_ang_vel_w), dim=-1
  )


def feet_both_grounded(env: "ManagerBasedRlEnv", sensor_name: str) -> torch.Tensor:
  """双脚同时着地才给分。单脚站立也能省力，但那不是要的东西。"""
  sensor = env.scene.sensors[sensor_name]
  found = sensor.data.found.reshape(env.num_envs, -1) > 0
  return (found.sum(dim=-1) >= 2).float()


def upright_height(
  env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg, target: float, std: float
) -> torch.Tensor:
  """躯干相对双脚的高度贴近目标值的程度。

  用相对高度而不是世界系高度：地形高度或脚下摩擦让机器人整体下沉时，世界系高度会误判。
  """
  asset: Entity = env.scene[asset_cfg.name]
  feet_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2].mean(dim=1)
  height = asset.data.root_link_pos_w[:, 2] - feet_z
  return torch.exp(-torch.square((height - target) / std))
