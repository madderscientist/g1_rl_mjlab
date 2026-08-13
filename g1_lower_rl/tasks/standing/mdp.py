"""站立任务的自定义奖励项与课程项。

目标是**站着不摔的前提下下肢耗电最小**，姿势不限定，所以这里只剩「别摔」类的单边约束；
真正的目标函数直接复用速度任务的 ``normalized_joint_effort_l2``。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from g1_lower_rl.tasks.lower_body import mdp as lower_body_mdp

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


class gated_event_params(lower_body_mdp.gated_event_params):
  """速度任务那套逐档抬扰动，闸门改看「还站着吗」。

  父类的闸门量的是速度跟随误差，本任务没有速度指令，直接复用会在 ``command_manager``
  上取空。判据换成骨盆倾角：``projected_gravity_b`` 的水平分量模长就是 sin(倾角)，
  终止阈值 0.8 rad 对应 0.717，而 ``max_tilt`` 取 0.45（约 27 度）——它问的是「还明显站着吗」，
  不是「站得笔直吗」；后者是姿态先验，而本任务故意不约束姿态，取 0.25 会让闸门永远开不了。

  **量的是全体环境的当前倾角，不是只量刚复位的那几个。** 复位瞬间骨盆是正的（实测达标率 100%），
  所以摔得越勤、刚复位的环境占比越高，分数反而越好看——这个反向激励是真实存在的，
  换成按存活率度量才能根治。
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


def com_over_feet(
  env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg, std: float
) -> torch.Tensor:
  """重心水平投影落在两脚中点附近的程度，exp 形。是奖励，如果没有奖励会让模型直接选择终止训练

  用两脚中点而不是严格的支撑多边形：多边形内的点都「合法」，但只有靠近中心才对左右两侧的扰动都有余量。
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

  这是「别摔」的硬约束，不是姿态先验：只在重心真的快出支撑面时才非零，
  中间那一大片区域完全不收费，站成什么样随便。
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


def height_floor(
  env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg, minimum: float
) -> torch.Tensor:
  """骨盆相对双脚低于 ``minimum`` 的那部分，单位米；站高不罚。

  不能写成打靶：上一版是 ``exp(-((h-0.72)/0.06)^2)`` 配权重 2.0，而实测自然站姿高 0.78，
  站直反而单拍亏 1.26，于是策略蹲到 0.72 去硬扛力矩——实机上的弯腿和腰部过热就是这么来的。
  改成单边下限后，「站多高」交给省电项：直腿时重力线贴着膝轴过、膝力矩趋零。

  取 L1 而不是平方：平方在刚跌破下限时梯度为零，拦不住缓慢下沉。
  """
  asset: Entity = env.scene[asset_cfg.name]
  feet_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2].mean(dim=1)
  height = asset.data.root_link_pos_w[:, 2] - feet_z
  return torch.clamp(minimum - height, min=0.0)
