"""下肢任务的观测项。

这里定义的项**只给 critic 用**（特权信息），actor 侧的项全部来自 ``mjlab.envs.mdp``，
唯一的例外是 ``shifted_phase``——它是相位时钟，真机上也能重建。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

from g1_lower_rl.tasks.lower_body.mdp.events import gait_phase_offset

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


def height_above_feet(asset: Entity, site_ids: Sequence[int] | slice) -> torch.Tensor:
  """相对最低那个足底 site 测得的骨盆高度，形状 (num_envs,)。

  用脚而不是世界 z 作基准，在斜坡和台阶上依然有意义，也刚好是真机光靠腿部运动学就
  能重建出来的量。实测过：足底 site 正好落在鞋底，所以平地上它就等于卷尺量的骨盆离地高度。
  """
  foot_z = asset.data.site_pos_w[:, site_ids, 2]
  return asset.data.root_link_pos_w[:, 2] - foot_z.min(dim=1).values


def shifted_phase(
  env: ManagerBasedRlEnv, period: float, command_name: str
) -> torch.Tensor:
  """步态相位 ``[sin, cos]``，起点逐环境随机。

  相位时钟是 ``episode_length_buf * dt``，每局必从 0 开始，而相位 0 对应右脚先迈，
  于是绝大多数带运动指令的 reset 都是右脚起步——这是实测到的唯一左右破缺来源。
  加一个逐环境偏移（见 ``resample_gait_phase``）就消掉了。

  指令接近零时置零，与部署端的 ``gait_phase()`` 一致。注意时钟本身不停，只是观测被遮住，
  所以从站立切到行走那一刻的相位是任意的——部署端也是同样的行为。
  """
  clock = (env.episode_length_buf * env.step_dt) / period + gait_phase_offset(env)
  angle = (clock % 1.0) * 2.0 * torch.pi
  out = torch.stack((torch.sin(angle), torch.cos(angle)), dim=1)
  command = env.command_manager.get_command(command_name)
  assert command is not None
  standing = torch.linalg.norm(command, dim=1) < 0.1
  return torch.where(standing.unsqueeze(1), torch.zeros_like(out), out)


def base_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """骨盆相对最低足底 site 的高度，形状 (num_envs, 1)。"""
  asset: Entity = env.scene[asset_cfg.name]
  return height_above_feet(asset, asset_cfg.site_ids).unsqueeze(-1)


def foot_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  """两个足底 site 的世界 z，形状 (num_envs, num_sites)。"""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  air_time = sensor.data.current_air_time
  assert air_time is not None
  return air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None
  return (found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """接触力做对数压缩：量程跨越三个数量级，线性输入会让归一化层被落地冲击主导。"""
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  assert force is not None
  flat = force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(flat) * torch.log1p(torch.abs(flat))
