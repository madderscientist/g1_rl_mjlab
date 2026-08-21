"""GMT 任务的奖励项。"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.tracking.mdp.commands import MotionCommand


def motion_swing_lift_ratio(
  env: ManagerBasedRlEnv,
  command_name: str,
  body_names: tuple[str, ...],
  std: float = 0.3,
  min_lift: float = 0.05,
) -> torch.Tensor:
  """摆动腿抬脚完成度，按**比值**而非绝对误差评分。

  上游的 ``exp(-‖e‖²/σ²)`` 对「系统性抬不够」几乎无感：拖着一只脚（差 0.107 m）
  摊到 14 个 body 上只让奖励掉 **0.9%**，比噪声还小。给末端单独立项、把区分度
  提高 22 倍也只把抬脚比从 0.64 推到 0.67——**权重救不了形式上的缺陷**。

  换成比值后同一偏差的信号强 87 倍：实测抬脚比 0.63，σ=0.3 下得分 0.213，
  与满分差 **0.787**。

  用「双脚高度差」而不是单脚绝对高度：``ankle_roll_link`` 的原点在踝关节不在脚底
  （复位瞬间参考 0.0 m、机器人 0.033 m），绝对高度两侧基准就对不齐；高度差还能
  同时消掉地面高度和躯干下沉的影响——策略塌腰时单脚绝对高度会假性变化。

  参考未要求抬脚时返回满分，避免在支撑相压制策略。
  """
  command = cast("MotionCommand", env.command_manager.get_term(command_name))
  idx = [i for i, n in enumerate(command.cfg.body_names) if n in body_names]
  rob = command.robot_body_pos_w[:, idx, 2]
  ref = command.body_pos_relative_w[:, idx, 2]

  rob_lift = rob.max(dim=-1).values - rob.min(dim=-1).values
  ref_lift = ref.max(dim=-1).values - ref.min(dim=-1).values

  ratio = rob_lift / ref_lift.clamp(min=min_lift)
  reward = torch.exp(-(ratio - 1.0).square() / std**2)
  return torch.where(ref_lift > min_lift, reward, torch.ones_like(reward))
