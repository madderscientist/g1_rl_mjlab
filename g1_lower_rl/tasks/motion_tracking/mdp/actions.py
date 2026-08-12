"""带手臂重力补偿的关节位置动作项。

真机上手臂有独立的重力补偿器：它算出所需力矩 ``tau_g``，再按 ``tau_g / kp`` 折算成
位置偏移叠加到目标位置上（走位置通道是为了避开力矩通道的延迟）。代入 PD 律：

    tau = kp * (q_des + tau_g/kp - q) - kd * qd = kp * (q_des - q) - kd * qd + tau_g

即与直接前馈 ``tau_g`` 等价。这里按同样的方式建模，所以仿真与真机的控制律逐项对应。

**为什么不用 MuJoCo 原生的 body_gravcomp**：那个是按**刚体**补偿的，补偿力矩会出现在
该刚体所有祖先 DoF 上（腰、腿、浮动基座）。这些 DoF 没开 ``jnt_actgravcomp`` 时会被
当成被动力加进去，等于让腿感觉不到手臂的重量。真机只在手臂关节上补，必须自己算。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg
from mjlab.utils.lab_api.math import quat_apply, sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class GravityCompensatedJointPositionAction(JointPositionAction):
  """在指定关节上叠加 ``tau_g / kp`` 的位置偏移。"""

  cfg: "GravityCompensatedJointPositionActionCfg"

  def __init__(
    self, cfg: "GravityCompensatedJointPositionActionCfg", env: "ManagerBasedRlEnv"
  ):
    super().__init__(cfg=cfg, env=env)
    model = env.sim.mj_model
    entity = self._entity

    names = cfg.compensated_joint_names
    missing = [n for n in names if n not in self._target_names]
    if missing:
      raise ValueError(f"补偿关节不在动作关节里: {missing}")

    # 补偿关节在**动作向量**里的位置，用于把偏移加回去。
    self._comp_slots = torch.tensor(
      [self._target_names.index(n) for n in names], device=self.device
    )

    jnt_ids, body_ids, axes, anchors, kps, subtrees = [], [], [], [], [], []
    for name in names:
      # sim.mj_model 里的名字带实体前缀，走 indexing 拿全局下标更稳妥。
      jid = int(entity.indexing.joint_ids[entity.joint_names.index(name)])
      if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
        raise ValueError(f"{name} 不是转动关节，重力补偿只支持 hinge")
      act = [i for i in range(model.nu) if model.actuator_trnid[i, 0] == jid]
      if not act:
        raise ValueError(f"{name} 没有执行器，无法按 kp 折算位置偏移")
      jnt_ids.append(jid)
      body_ids.append(model.jnt_bodyid[jid])
      axes.append(model.jnt_axis[jid].copy())
      anchors.append(model.jnt_pos[jid].copy())
      kps.append(abs(model.actuator_biasprm[act[0], 1]))
      subtrees.append(_subtree_bodies(model, model.jnt_bodyid[jid]))

    self._axis_local = torch.tensor(axes, device=self.device, dtype=torch.float32)
    self._anchor_local = torch.tensor(anchors, device=self.device, dtype=torch.float32)
    self._kp = torch.tensor(kps, device=self.device, dtype=torch.float32)

    # data.body_* 按 entity 内部顺序索引，模型里是全局下标，这里做一次映射。
    global_body_ids = entity.indexing.body_ids.tolist()
    to_local = {g: i for i, g in enumerate(global_body_ids)}
    self._jnt_body_local = torch.tensor(
      [to_local[int(b)] for b in body_ids], device=self.device, dtype=torch.long
    )
    # 读逐环境质量用的列下标。
    self._body_global = entity.indexing.body_ids.to(self.device)

    # (J, B_entity) 的 0/1 掩码：关节 j 的远端刚体集合。质量在运行时才取，因为
    # 负载随机化会逐环境改 body_mass——真机上补偿器会自动检测负载，这里对应地让它看到。
    mask = torch.zeros(len(names), len(global_body_ids), device=self.device)
    for j, bodies in enumerate(subtrees):
      for b in bodies:
        if b in to_local:
          mask[j, to_local[b]] = 1.0
    self._distal_mask = mask

    self._gravity = torch.tensor(
      model.opt.gravity, device=self.device, dtype=torch.float32
    )
    self._model = env.sim.model

    # 补偿器误差在同一条臂内是**相关**的：共用一套动力学模型参数和一个负载估计器，
    # 不会每个关节各犯各的错。按臂共享一个系数，而不是 14 个独立采样——后者会让误差
    # 在合力上互相抵消，把训练难度调得比真机低。
    groups = sorted({n.split("_", 1)[0] for n in names})
    self._gain_group = torch.tensor(
      [groups.index(n.split("_", 1)[0]) for n in names], device=self.device
    )
    self._num_groups = len(groups)
    # 每个环境一个系数，代表补偿器的模型误差与负载估计误差。
    self._gain = torch.ones(self.num_envs, len(names), device=self.device)
    self._resample_gain(slice(None))

  def _resample_gain(self, env_ids) -> None:
    lo, hi = self.cfg.gain_range
    if lo == hi == 1.0:
      return
    n = self.num_envs if isinstance(env_ids, slice) else len(env_ids)
    per_group = sample_uniform(
      lo, hi, (n, self._num_groups), device=self.device
    )
    self._gain[env_ids] = per_group[:, self._gain_group]

  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    self._resample_gain(slice(None) if env_ids is None else env_ids)

  def _gravity_torque(self) -> torch.Tensor:
    """各补偿关节上重力产生的力矩，(N, J)。"""
    data = self._entity.data
    body_pos = data.body_link_pos_w  # (N, B, 3)
    body_quat = data.body_link_quat_w
    com = data.body_com_pos_w

    # 关节世界轴与锚点：由所属刚体的位姿把局部量转到世界。
    idx = self._jnt_body_local  # (J,) entity 内下标
    q = body_quat[:, idx]  # (N, J, 4)
    axis_w = quat_apply(q, self._axis_local.expand(q.shape[0], -1, -1))
    anchor_w = body_pos[:, idx] + quat_apply(
      q, self._anchor_local.expand(q.shape[0], -1, -1)
    )

    # 标量三重积恒等式：Σ_b m_b[(c_b-p_j)×g]·a_j = [Σ_b m_b(c_b-p_j)]·(g×a_j)。
    # 左式要 (N,J,B,3) 的中间量，右式只要 (N,J,3)。
    body_mass = self._model.body_mass
    mass = body_mass[:, self._body_global]  # (nworld, B)
    if mass.shape[0] == 1:
      mass = mass.expand(com.shape[0], -1)
    mw = mass[:, None, :] * self._distal_mask[None]  # (N, J, B)
    total = mw.sum(-1)  # (N, J)
    weighted = torch.einsum("njb,nbc->njc", mw, com)  # (N, J, 3)
    lever = weighted - total[..., None] * anchor_w
    return (lever * torch.linalg.cross(self._gravity.expand_as(axis_w), axis_w)).sum(-1)

  def apply_actions(self) -> None:
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    target = self._processed_actions - encoder_bias
    # 补偿器要抵消重力力矩，故取负；再按 kp 折算成位置偏移。
    offset = -self._gravity_torque() * self._gain / self._kp
    target = target.index_add(1, self._comp_slots, offset)
    self._entity.set_joint_position_target(target, joint_ids=self._target_ids)


def _subtree_bodies(model, root: int) -> list[int]:
  """``root`` 及其所有后代刚体。"""
  out, stack = [], [int(root)]
  while stack:
    b = stack.pop()
    out.append(b)
    stack.extend(i for i in range(model.nbody) if model.body_parentid[i] == b and i != b)
  return out


@dataclass(kw_only=True)
class GravityCompensatedJointPositionActionCfg(JointPositionActionCfg):
  compensated_joint_names: tuple[str, ...] = field(default_factory=tuple)
  """开启重力补偿的关节，必须是动作关节的子集。"""

  gain_range: tuple[float, float] = (1.0, 1.0)
  """补偿系数的采样区间，**按臂共享**。取 (1,1) 是理想补偿器，不该单独用。"""

  def build(self, env: "ManagerBasedRlEnv") -> GravityCompensatedJointPositionAction:
    return GravityCompensatedJointPositionAction(self, env)
