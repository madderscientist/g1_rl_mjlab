"""姿态与关节相关的奖励项。"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Sequence

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from g1_lower_rl.tasks.lower_body.mdp.rewards._common import ROBOT, moving

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def body_orientation_l2(
  env: ManagerBasedRlEnv,
  lateral_scale: float = 1.0,
  asset_cfg: SceneEntityCfg = ROBOT,
) -> torch.Tensor:
  """惩罚指定刚体偏离竖直，罚的是重力在体系里 xy 分量的平方和（= sin²(倾角)）。

  ``lateral_scale`` 单独放大左右分量 ``g_b[1]``。两个方向必须分开调：前后分量
  另有 ``backward_lean`` 单边管着，而左右除了这一项没人管。

  ``asset_cfg.body_ids`` 为空时退化成根刚体。mjlab 1.5.x 把自带的同名项换成了
  ``upright`` 类，语义不同，这里保留原实现。
  """
  asset: Entity = env.scene[asset_cfg.name]
  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
    g_b = quat_apply_inverse(body_quat_w, asset.data.gravity_vec_w)
  else:
    g_b = asset.data.projected_gravity_b
  return torch.square(g_b[:, 0]) + lateral_scale * torch.square(g_b[:, 1])


def backward_lean(
  env: ManagerBasedRlEnv,
  deadband_deg: float = 4.0,
  forward_scale: float = 0.0,
  forward_deadband_deg: float | None = None,
  asset_cfg: SceneEntityCfg = ROBOT,
) -> torch.Tensor:
  """罚骨盆偏离竖直，前后不对称，两边各留一段死区。

  ``body_orientation_l2`` 量的是 torso_link 且前后对称，而骨盆整体后坐时腰关节会把
  偏差吸收掉，那一项看不见。本项补的就是这个真空，且必须是单边的——正常步态本来
  就需要前倾，对称项加重会把好姿态一起罚上。

  **两侧的死区和函数形式都不同，两者都是必须的：**

  * 后仰侧 ``deadband_deg``，**线性**。平方形式在死区附近是二阶小量，推不动姿态；
    线性的梯度是常数，不随角度衰减。死区不能取 0：合理站姿本来就带几度后仰。
  * 前倾侧 ``forward_deadband_deg``（不给则沿用后仰侧）× ``forward_scale``，**平方**。
    正常步态的骨盆前倾随速度递增，改成线性会把那个好姿态一起罚上。

  两侧共用死区会变成“对前进需要的姿态单向收费、对后退需要的姿态免费”——这是
  整套奖励里唯一前后不对称的项，方向弄反会直接造出一个恒定后退偏置。

  用重力在刚体系里的投影而不是关节角：量的是对重力的**绝对**倾角。``g_b[0] > 0``
  是前倾。
  """
  asset: Entity = env.scene[asset_cfg.name]
  quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
  g_b = quat_apply_inverse(quat_w, asset.data.gravity_vec_w)
  deadband = math.sin(math.radians(deadband_deg))
  cost = torch.clamp(-g_b[:, 0] - deadband, min=0.0)
  if forward_scale:
    if forward_deadband_deg is not None:
      deadband = math.sin(math.radians(forward_deadband_deg))
    forward = torch.clamp(g_b[:, 0] - deadband, min=0.0)
    cost = cost + forward_scale * torch.square(forward)
  return cost


def straight_knee(
  env: ManagerBasedRlEnv,
  command_name: str,
  movement_command_name: str,
  height_threshold: float,
  command_threshold: float,
  std: float,
  asset_cfg: SceneEntityCfg = ROBOT,
) -> torch.Tensor:
  """高度指令够高且没有运动指令时，奖励膝关节接近伸直。

  注意这不是“把膝盖伸到 0”：对称蹲姿下高度已经把膝角确定下来了（实测 0.78 m 需要
  0.387 rad，0.79 m 需要 0.153 rad，而 0.7919 m 是硬上限）。它真正的作用是在给定高度下
  选出最直的那个姿态——同一个骨盆高度可以用“腿直”或“前倾 + 多屈膝”凑出来，后者在
  实机上就是那副蹲着走的样子。

  量的是膝角本身而不是相对默认位姿的偏差：默认位姿的膝角就是 0.3 rad，以它为基准等于
  把屈膝当成目标。
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  movement_command = env.command_manager.get_command(movement_command_name)
  assert movement_command is not None
  command_strength = torch.norm(movement_command[:, :2], dim=1) + torch.abs(
    movement_command[:, 2]
  )
  active = (command[:, 0] >= height_threshold).float() * (
    command_strength < command_threshold
  ).float()
  knee = asset.data.joint_pos[:, asset_cfg.joint_ids]
  return torch.exp(-torch.mean(torch.square(knee), dim=1) / std**2) * active


def action_rate_still(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """只在“没叫它动”时生效的动作变化率惩罚，与全程生效的 ``action_rate_l2`` 叠加。

  站着不动时才需要把动作压平；走起来之后再压就是在压步态本身。

  **不要把它换成“偏离自身慢速均值”那类对直流免疫的形式。** 试过：站立分支下
  “把膝关节顶死在限位上保持不动”是一个纯直流动作，代价精确为零，于是“坐下装死”
  变成最优解：实测 400 迭代内膝关节动作从 1.1 顶到 5.3，目标角 2.15 rad（限位 2.88）。
  差分形式虽然对低频不敏感，但它至少不会把饱和变成免费的。
  """
  delta = env.action_manager.action - env.action_manager.prev_action
  return torch.sum(torch.square(delta), dim=1) * (
    1.0 - moving(env, command_name, command_threshold)
  )


def joint_deviation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = ROBOT,
  weights: Sequence[float] | None = None,
  command_name: str | None = None,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """惩罚关节偏离默认位姿。``weights`` 按 ``asset_cfg`` 解析出的关节顺序逐项加权。

  给了 ``command_name`` 就只在“没叫它动”时收费——用于那些走路时必须能动、站着时不
  该动的关节。不给则全程生效。

  腰部用逐关节权重是必要的：``waist_yaw`` 是唯一能与摆动腿反向旋转、
  抵消其角动量的关节，把它和 roll/pitch 同等对待地拉回 0 实测会把 20 s 直行
  累计航向从 16.5° 推到 99.8°——扭腰被压住，角动量没处去，只能让整机绕立轴转。
  """
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  diff = (
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - default_joint_pos[:, asset_cfg.joint_ids]
  )
  cost = torch.square(diff)
  if weights is not None:
    cost = cost * torch.as_tensor(weights, device=cost.device, dtype=cost.dtype)
  cost = torch.sum(cost, dim=1)
  if command_name is not None:
    cost = cost * (1.0 - moving(env, command_name, command_threshold))
  return cost
