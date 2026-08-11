"""左右镜像数据增强（DUP）

**为什么需要。** 环境是左右对称的（实测：腿部质心 y 精确为 0、执行器与关节限位
全对称、整机质心 y 偏 0.074 mm），但**目标函数里没有任何一项要求策略对称**：
``exp(-(指令ω - 实测ω)²/std²)`` 只看误差绝对值，一个“右转好左转差”的策略和一个
两边都中等的策略拿到的奖励相同。于是随机初始化带来的微小不对称被 PPO 逐步放大。

实测证据（同一批检查点，8 s 回放，达成率 = 实测ω/指令ω）：

* 破缺方向**不固定**，且会在同一条血统内翻转：07-09-08 偏左 +20.7%，它的直接
  续训 10-13-12 偏右 -44.2%；term100 偏右 -35.2%，从它同一个存档派生的 v2c
  偏左 +21.5%、abl_no_standstill 偏右 -29.4%。
* 一条链上单调放大：-2.7% -> -23.5% -> -41.1% -> -79.2%。
* 镜像测试：把观测镜像、动作反镜像回来之后，左右达成率**恰好对调**
  （左转 36.8% -> 118.9%，右转 120.8% -> 29.3%），而已经对称的标杆策略在同一
  变换下几乎不变（<3%）——后者同时验证了镜像变换本身的正确性。

**做法。** 在 PPO 更新时把每个小批扩成 ``[原始; 镜像]``（DUP，Abdolhosseini et al. 2019），
同一个 ``theta`` 必须同时解释 ``(s, a)`` 和 ``(M(s), M(a))``。
接的是 rsl_rl 自带的 ``symmetry_cfg["use_data_augmentation"]``。

**镜像变换**（关于矢状面 y -> -y）：

* 极矢量（重力、线速度）：(x, y, z) -> (x, -y, z)
* 轴矢量（角速度）：a -> det(M)·M·a = (-x, y, -z)
* 关节：左右互换；绕 x(roll) 和 z(yaw) 的取反，绕 y(pitch) 的不变
* 相位：左右脚互换 = 半周期相移，(sin, cos) -> (-sin, -cos)
* 脚相关的成对量：两只脚互换
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

# 非关节观测项的逐分量符号。**这张表必须覆盖所有观测项**，
# 漏一项会静默地把不对称重新引进来，所以下面用未知项直接抛异常。
_VECTOR_SIGNS: dict[str, tuple[float, ...]] = {
  "base_ang_vel": (-1.0, 1.0, -1.0),  # 轴矢量
  "projected_gravity": (1.0, -1.0, 1.0),  # 极矢量
  "base_lin_vel": (1.0, -1.0, 1.0),  # 极矢量
  "command_twist": (1.0, -1.0, -1.0),  # vx, vy, ωz
  "command_height": (1.0,),
  "base_height": (1.0,),
  "phase": (-1.0, -1.0),  # 半周期相移
}
# 两只脚的成对量，镜像 = 交换（顺序为 左, 右）
_FOOT_PAIR_TERMS = ("foot_height", "foot_air_time", "foot_contact")
# 每只脚三个力分量，交换两只脚并把 y 分量取反
_FOOT_FORCE_TERMS = ("foot_contact_forces",)
# 用关节顺序做镜像的项
_JOINT_TERMS = ("joint_pos", "joint_vel", "arm_joint_pos", "arm_joint_vel")


def _pair_joints(names: list[str]) -> tuple[list[int], list[float]]:
  """给定有序关节名，返回镜像置换下标和逐关节符号。

  符号按关节轴判定：roll(绕 x) 与 yaw(绕 z) 取反，pitch(绕 y) 与 knee/elbow 不变。
  轴向从名字推断，并要求左右成对，配不上就抛异常——宁可训练起不来，也不要
  静默地漏掉一个关节。
  """
  index = {n: i for i, n in enumerate(names)}
  perm: list[int] = []
  sign: list[float] = []
  for n in names:
    if n.startswith("left_"):
      mate = "right_" + n[len("left_") :]
    elif n.startswith("right_"):
      mate = "left_" + n[len("right_") :]
    else:
      mate = n  # 腰部等中线关节，镜像到自己
    if mate not in index:
      raise ValueError(f"关节 {n!r} 找不到镜像对应 {mate!r}，无法构建镜像映射")
    perm.append(index[mate])
    sign.append(-1.0 if ("roll" in n or "yaw" in n) else 1.0)
  return perm, sign


def _joint_names_of(term_cfg, all_names: list[str]) -> list[str]:
  cfg = term_cfg.params.get("asset_cfg")
  ids = getattr(cfg, "joint_ids", None) if cfg is not None else None
  if ids is None:
    raise ValueError("关节类观测项缺少 asset_cfg.joint_ids")
  if isinstance(ids, slice):
    ids = list(range(len(all_names)))
  return [all_names[i] for i in ids]


def _action_joint_names(env) -> list[str]:
  return [
    n.replace("_actuator", "")
    for n in env.action_manager.get_term("joint_pos").target_names
  ]


def build_obs_mirror(env, group: str) -> tuple[torch.Tensor, torch.Tensor]:
  """为某个观测组构建 (置换下标, 符号) 两个一维张量，长度等于该组维度。"""
  om = env.observation_manager
  all_names = list(env.scene["robot"].joint_names)
  perm: list[int] = []
  sign: list[float] = []
  base = 0
  term_cfgs = om._group_obs_term_cfgs[group]  # noqa: SLF001
  for name, dim, tcfg in zip(
    om.active_terms[group], om.group_obs_term_dim[group], term_cfgs, strict=True
  ):
    width = int(dim[0])
    if name in _VECTOR_SIGNS:
      s = _VECTOR_SIGNS[name]
      if len(s) != width:
        raise ValueError(f"观测项 {name} 维度 {width} 与符号表 {len(s)} 不符")
      perm += [base + i for i in range(width)]
      sign += list(s)
    elif name in _JOINT_TERMS or name == "actions":
      jn = _action_joint_names(env) if name == "actions" else _joint_names_of(
        tcfg, all_names
      )
      if len(jn) != width:
        raise ValueError(f"观测项 {name} 维度 {width} 与关节数 {len(jn)} 不符")
      p, g = _pair_joints(jn)
      perm += [base + i for i in p]
      sign += g
    elif name in _FOOT_PAIR_TERMS:
      if width != 2:
        raise ValueError(f"{name} 维度应为 2，实际 {width}")
      perm += [base + 1, base + 0]
      sign += [1.0, 1.0]
    elif name in _FOOT_FORCE_TERMS:
      if width != 6:
        raise ValueError(f"{name} 维度应为 6，实际 {width}")
      perm += [base + 3, base + 4, base + 5, base + 0, base + 1, base + 2]
      sign += [1.0, -1.0, 1.0, 1.0, -1.0, 1.0]
    else:
      raise ValueError(
        f"观测项 {name!r} 没有镜像规则。必须显式补上——漏掉一项等于把左右不对称重新引进来，而且不会报错。"
      )
    base += width
  return (
    torch.tensor(perm, dtype=torch.long, device=env.device),
    torch.tensor(sign, dtype=torch.float32, device=env.device),
  )


def build_action_mirror(env) -> tuple[torch.Tensor, torch.Tensor]:
  perm, sign = _pair_joints(_action_joint_names(env))
  return (
    torch.tensor(perm, dtype=torch.long, device=env.device),
    torch.tensor(sign, dtype=torch.float32, device=env.device),
  )


class MirrorAugmentation:
  """rsl_rl PPO 的 ``data_augmentation_func``：把每个小批扩成 ``[原始; 镜像]``。

  **``prob`` 的含义。** 逐样本以 ``prob`` 的概率把第二份换成镜像，否则原样复制。
  批一律扩成两倍，于是 rsl_rl 里 ``num_aug`` 恒为 2、形状永远对得上，而 ``prob``
  仍是一个连续可调的强度旋钮。复制出来的那些行在对称损失里贡献恰好 0，不会污染日志。

  档位进度由 ``common_step_counter`` 推算，而它已随存档持久化，续训自动落回正确档位。

  **一个已知的近似。** 镜像样本沿用原样本的 ``old_actions_log_prob``，等价于假设
  ``log pi_old(M(a)|M(s)) = log pi_old(a|s)``。策略还不对称时这不成立，但 rsl_rl 和
  原论文都是这么做的，重要性比会被 PPO 的 clip 挡住。
  """

  def __init__(
    self,
    schedule: tuple[tuple[int, float], ...] = ((0, 1.0),),
    steps_per_iter: int = 24,
  ):
    self._schedule = tuple(sorted(schedule))
    self._steps_per_iter = max(int(steps_per_iter), 1)
    self._obs_maps: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None
    self._act_perm: torch.Tensor | None = None
    self._act_sign: torch.Tensor | None = None
    self._prob = -1.0
    self._mask: torch.Tensor | None = None

  def _build(self, env) -> None:
    u = env.unwrapped
    self._obs_maps = {
      g: build_obs_mirror(u, g) for g in u.observation_manager.active_terms
    }
    self._act_perm, self._act_sign = build_action_mirror(u)
    stages = " -> ".join(f"{it}轮:{p:.0%}" for it, p in self._schedule)
    print(f"[INFO]: 镜像 DUP 已启用，档位 {stages}，覆盖观测组 {list(self._obs_maps)}")

  def _prob_now(self, env) -> float:
    iteration = env.unwrapped.common_step_counter // self._steps_per_iter
    prob = 0.0
    for start, value in self._schedule:
      if iteration >= start:
        prob = value
    if prob != self._prob:
      if self._prob >= 0.0:
        print(f"[INFO]: 镜像 DUP 比例 {self._prob:.0%} -> {prob:.0%}")
      self._prob = prob
    return prob

  def __call__(self, env, obs=None, actions=None):
    if self._obs_maps is None:
      self._build(env)
    assert self._obs_maps is not None

    # obs 非空 = 本小批的第一次调用，此时重掷；对称损失那次传 obs=None，必须复用同
    # 一张掩码，否则比较的是错位的两行。
    if obs is not None:
      self._mask = (
        torch.rand(obs.batch_size[0], 1, device=obs.device) < self._prob_now(env)
      )

    obs_aug = None
    if obs is not None:
      assert self._mask is not None
      items = {}
      for key in obs.keys():
        x = obs[key]
        m = self._obs_maps.get(key)
        flipped = x if m is None else x[:, m[0]] * m[1]
        items[key] = torch.cat([x, torch.where(self._mask, flipped, x)], dim=0)
      obs_aug = TensorDict(items, batch_size=[obs.batch_size[0] * 2], device=obs.device)

    act_aug = None
    if actions is not None:
      assert self._mask is not None and self._act_perm is not None
      flipped = actions[:, self._act_perm] * self._act_sign
      act_aug = torch.cat([actions, torch.where(self._mask, flipped, actions)], dim=0)

    return obs_aug, act_aug
