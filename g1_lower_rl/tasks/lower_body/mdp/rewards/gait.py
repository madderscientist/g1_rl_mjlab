"""步态与足部相关的奖励项。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

from g1_lower_rl.tasks.lower_body.mdp.events import gait_phase_offset
from g1_lower_rl.tasks.lower_body.mdp.rewards._common import ROBOT, moving

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float = 0.4,
  command_name: str | None = None,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """奖励单脚支撑期的时长接近 ``threshold``。

  唯一一个双脚支撑时恰好为零、只在单脚支撑时为正的项。没有它策略会收敛成雕像：
  站着不动就能拿满高度奖励加 56% 的 ``shifted_feet_gait``，而所有运动惩罚都是零，
  迈出第一步严格更差。

  注意 mjlab 1.5.x 自带的 ``feet_air_time`` 换成了“腾空时长落在区间内就计一分”的
  形式，与这里的单脚支撑判据不同，不能替换。
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  in_contact = data.current_contact_time > 0.0
  in_mode_time = torch.where(in_contact, data.current_contact_time, data.current_air_time)
  single_stance = torch.mean(in_contact.float(), dim=1) == 0.5
  mode_time = torch.min(
    torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1
  )[0]
  reward = torch.clamp(threshold - torch.abs(mode_time - threshold), min=0.0)
  if command_name is not None:
    reward = reward * moving(env, command_name, command_threshold)
  return reward


def shifted_feet_gait(
  env: ManagerBasedRlEnv,
  period: float,
  offset: list[float],
  threshold: float,
  command_threshold: float,
  command_name: str,
  sensor_name: str,
) -> torch.Tensor:
  """步态相位奖励，起点逐环境随机。

  参数含义：``period`` 是一个完整步态周期的秒数（左脚落地到再次落地）；``offset``
  是每只脚在周期里的相位偏移，按传感器里的脚顺序（0=左，1=右），``[0.0, 0.5]`` 就是
  交替步态，``[0.0, 0.0]`` 是双脚同步的兔跳；``threshold`` 是周期里应当着地的比例。

  相位偏移必须与 ``shifted_phase`` 观测用同一个——否则策略看到的时钟和被要求的
  节拍对不上。
  """
  sensor: ContactSensor = env.scene[sensor_name]
  in_contact = sensor.data.current_contact_time > 0  # type: ignore[operator]
  clock = (env.episode_length_buf * env.step_dt) / period + gait_phase_offset(env)
  offsets = torch.as_tensor(offset, device=env.device, dtype=clock.dtype).view(1, -1)
  leg_phase = (clock.unsqueeze(1) + offsets) % 1.0
  reward = ((leg_phase < threshold) == in_contact).float().mean(dim=1)
  return reward * moving(env, command_name, command_threshold)


def feet_clearance_relative(
  env: ManagerBasedRlEnv,
  target_height: float,
  command_name: str,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = ROBOT,
) -> torch.Tensor:
  """摆动脚相对支撑脚的离地高度，按该脚水平速度加权。

  原版 ``feet_clearance`` 用的是足底 site 的**世界 z**，在平地上没问题，一上起伏地形就把
  地面高度也算进去了：起伏 15 cm 时，10 cm 的目标能被地面本身带偏 ±7.5 cm。

  以两只脚里较低的那只为基准就与坡度无关：平滑地形上两脚间的地面高差实测中位数只有
  0.6 cm。支撑脚自己的相对高度是 0，但它水平速度也接近 0，所以不贡献代价——和原版一样。
  """
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  relative_z = foot_z - foot_z.min(dim=1, keepdim=True).values
  speed = torch.norm(asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2], dim=-1)
  cost = torch.sum(torch.abs(relative_z - target_height) * speed, dim=1)
  return cost * moving(env, command_name, command_threshold)


def feet_stationary(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """指令要求站住时，惩罚抬脚。

  ``feet_air_time`` 和 ``shifted_feet_gait`` 奖励的是“迈步”而不是“位移”，所以原地
  踏步能把它们满额拿走。这一项让没叫你动的时候踏步徒劳无益，同时保持足够便宜，
  使得被推后迈一步仍然划算。
  """
  sensor: ContactSensor = env.scene[sensor_name]
  airborne = (~(sensor.data.current_contact_time > 0)).float().sum(dim=1)
  return airborne * (1.0 - moving(env, command_name, command_threshold))
