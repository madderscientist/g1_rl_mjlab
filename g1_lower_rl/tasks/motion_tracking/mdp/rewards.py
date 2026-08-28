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
  one_sided: bool = False,
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

  ``one_sided`` 对抬过头不扣分。双边形式在 ratio→1 时「往上推」和「别超过」两个
  梯度互相抵消，而实测偏差全在抬不够一侧；抬过头自有 ``motion_body_pos`` 兜底。
  """
  command = cast("MotionCommand", env.command_manager.get_term(command_name))
  idx = [i for i, n in enumerate(command.cfg.body_names) if n in body_names]
  rob = command.robot_body_pos_w[:, idx, 2]
  ref = command.body_pos_relative_w[:, idx, 2]

  rob_lift = rob.max(dim=-1).values - rob.min(dim=-1).values
  ref_lift = ref.max(dim=-1).values - ref.min(dim=-1).values

  ratio = rob_lift / ref_lift.clamp(min=min_lift)
  if one_sided:
    ratio = ratio.clamp(max=1.0)
  reward = torch.exp(-(ratio - 1.0).square() / std**2)
  return torch.where(ref_lift > min_lift, reward, torch.ones_like(reward))


def motion_anchor_lin_vel_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float = 0.4,
) -> torch.Tensor:
  """anchor 线速度跟踪，单独立项。

  已有的 ``motion_body_lin_vel`` 是 14 个 body 平均且 σ=1.0 极松：0.3 m/s 的系统性
  速度亏欠只掉该项 8.6%，而 10 秒就累积 3 m 漂移。**位置误差是速度误差的积分**，
  只靠位置项（且 σ=0.3 的 exp 核在 0.6 m 外已饱和）永远慢一拍，速度项才锁得住漂移速率。

  GMT 把 ``tracking root pose`` 与 ``tracking root vel`` 列为两个独立项，
  DeepMimic 的 root 复合误差里也含速度分量。
  """
  command = cast("MotionCommand", env.command_manager.get_term(command_name))
  i = command.motion_anchor_body_index
  err = torch.sum(
    (command.body_lin_vel_w[:, i] - command.robot_body_lin_vel_w[:, i]).square(), dim=-1
  )
  return torch.exp(-err / std**2)


def motion_ee_pos_torso_relative_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  body_names: tuple[str, ...],
  std: float = 0.07,
  still_std: float = 0.03,
  still_speed: float = 0.5,
) -> torch.Tensor:
  """末端**相对躯干**的位形跟踪，参考末端慢时自动收紧 σ。

  与 ``motion_hand_pos`` 的区别：那一项比的是重锚定后的世界位置，里面混着整个身体
  的平移与朝向；精细操作真正要求的是「手相对身体在哪」，所以这里把两侧都减去各自的
  躯干位置，只留位形。两项互补，不是替代。

  σ 随参考末端速度切换：慢速段多是取放/对准这类精细动作，容差本来就该更小；
  快速挥臂受带宽限制（实测参考手速 3~5 m/s 时误差 0.118 m vs 慢速段 0.056 m，
  见 `run_logs/probe_arm_action_headroom.py`），用同一个严 σ 只会制造无法兑现的梯度。
  """
  command = cast("MotionCommand", env.command_manager.get_term(command_name))
  idx = [i for i, n in enumerate(command.cfg.body_names) if n in body_names]
  a = command.motion_anchor_body_index

  ref = command.body_pos_relative_w[:, idx] - command.body_pos_relative_w[:, a : a + 1]
  rob = command.robot_body_pos_w[:, idx] - command.robot_body_pos_w[:, a : a + 1]
  err = (ref - rob).square().sum(-1).mean(-1)

  speed = command.body_lin_vel_w[:, idx].norm(dim=-1).mean(-1)
  sigma = torch.where(
    speed < still_speed,
    torch.full_like(speed, still_std),
    torch.full_like(speed, std),
  )
  return torch.exp(-err / sigma.square())
