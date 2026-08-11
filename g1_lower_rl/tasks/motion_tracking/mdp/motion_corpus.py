"""多动作语料库：把一个目录下的所有 NPZ 拼成扁平张量，用全局帧下标寻址。

单动作跟踪（BeyondMimic / mjlab 自带的 tracking）里，策略只见过一条轨迹，换一段动作
基本跟不住。要让一个策略泛化到没见过的动作，训练时就得同时见到成百上千条，
所以这里把整个语料拼成一条长张量：``全局帧号 = start_idx[动作号] + 相位``。
拼接之后所有按帧取值的操作都退化成一次 gather，和单动作实现的开销一样。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class MotionCorpus:
  """一批 NPZ 动作的扁平拼接视图。

  NPZ 的字段与 mjlab tracking 的 ``MotionLoader`` 一致：``joint_pos`` / ``joint_vel``
  形状 (T, nj)，``body_*_w`` 形状 (T, nb, ...)。
  """

  def __init__(
    self, motion_dir: str, body_indexes: torch.Tensor, device: str = "cpu"
  ) -> None:
    files = sorted(Path(motion_dir).expanduser().glob("*.npz"))
    if not files:
      raise FileNotFoundError(f"目录下没有 .npz 动作文件: {motion_dir}")

    joint_pos, joint_vel = [], []
    body_pos, body_quat, body_lin, body_ang = [], [], [], []
    num_frames, names = [], []

    for f in files:
      data = np.load(f)
      bp = torch.tensor(data["body_pos_w"], dtype=torch.float32)
      # 把首帧根节点的水平位置挪到原点。各条动作原本带着自己的世界坐标，
      # 不归一化的话它们会散落在地图各处，复位时机器人被扔到很远的地方。
      bp[..., :2] -= bp[0, 0, :2].clone()

      # 刚体维先按 body_indexes 切了再存：整段语料只用其中 14 个刚体，
      # 而 NPZ 里是全部 44 个。不先切的话 80 段语料光刚体数组就要几个 GB。
      joint_pos.append(torch.tensor(data["joint_pos"], dtype=torch.float32))
      joint_vel.append(torch.tensor(data["joint_vel"], dtype=torch.float32))
      body_pos.append(bp[:, body_indexes.cpu()])
      body_quat.append(
        torch.tensor(data["body_quat_w"], dtype=torch.float32)[:, body_indexes.cpu()]
      )
      body_lin.append(
        torch.tensor(data["body_lin_vel_w"], dtype=torch.float32)[:, body_indexes.cpu()]
      )
      body_ang.append(
        torch.tensor(data["body_ang_vel_w"], dtype=torch.float32)[:, body_indexes.cpu()]
      )
      num_frames.append(int(joint_pos[-1].shape[0]))
      names.append(f.stem)

    self.names: list[str] = names
    self.joint_pos = torch.cat(joint_pos).to(device)
    self.joint_vel = torch.cat(joint_vel).to(device)
    self.body_pos_w = torch.cat(body_pos).to(device)
    self.body_quat_w = torch.cat(body_quat).to(device)
    self.body_lin_vel_w = torch.cat(body_lin).to(device)
    self.body_ang_vel_w = torch.cat(body_ang).to(device)

    self.num_frames = torch.tensor(num_frames, dtype=torch.long, device=device)
    start = torch.zeros_like(self.num_frames)
    start[1:] = self.num_frames.cumsum(0)[:-1]
    self.start_idx = start
    self.num_motions = len(files)
    self.time_step_total = int(self.num_frames.sum().item())

  def describe(self, fps: float = 50.0) -> str:
    total_s = self.time_step_total / fps
    return f"{self.num_motions} 条动作，共 {self.time_step_total} 帧 / {total_s:.1f} 秒"
