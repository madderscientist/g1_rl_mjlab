"""Preserve a motion policy's reference interface when loading it for inference."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import yaml
from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnvCfg
  from mjlab.rl import RslRlBaseRunnerCfg


def anchor_pose_features(
  position: torch.Tensor,
  quaternion: torch.Tensor,
  target_position: torch.Tensor,
  target_quaternion: torch.Tensor,
) -> torch.Tensor:
  inverse = quat_inv(quaternion)
  offset = quat_apply(inverse, target_position - position)
  relative = quat_mul(inverse, target_quaternion)
  axes = relative.new_tensor(((1, 0, 0), (0, 1, 0))).expand(*relative.shape[:-1], 2, 3)
  rotation = quat_apply(relative.unsqueeze(-2).expand(*relative.shape[:-1], 2, 4), axes)
  return torch.cat((offset, rotation.flatten(-2)), dim=-1)


def motion_reference_layout(env_cfg: ManagerBasedRlEnvCfg) -> dict[str, Any] | None:
  from g1_lower_rl.tasks.motion_tracking.mdp.commands import GeneralMotionCommandCfg

  command = env_cfg.commands.get("motion")
  if not isinstance(command, GeneralMotionCommandCfg):
    return None
  return {
    "reference_window": command.reference_window,
    "reference_stride": command.reference_stride,
    "reference_key_bodies": list(command.reference_key_bodies),
    "reference_key_body_pos": command.reference_key_body_pos,
    "reference_key_body_vel": command.reference_key_body_vel,
    "reference_anchor_pose": getattr(command, "reference_anchor_pose", False),
  }


def restore_motion_reference_layout(
  env_cfg: ManagerBasedRlEnvCfg,
  agent_cfg: RslRlBaseRunnerCfg,
  checkpoint: Path,
) -> None:
  """Restore legacy policy inputs, not its physics, rewards or randomization."""
  from mjlab.rl import RslRlOnPolicyRunnerCfg

  from g1_lower_rl.rl.rgmt_model import RgmtModelCfg
  from g1_lower_rl.tasks.motion_tracking.mdp.commands import GeneralMotionCommandCfg

  command = env_cfg.commands.get("motion")
  if not isinstance(agent_cfg, RslRlOnPolicyRunnerCfg):
    return
  actor = agent_cfg.actor
  if not isinstance(command, GeneralMotionCommandCfg) or not isinstance(actor, RgmtModelCfg):
    return

  saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
  raw = (saved.get("infos") or {}).get("motion_reference_layout")
  if raw is None:
    params_path = checkpoint.parent / "params" / "env.yaml"
    if not params_path.is_file():
      raise ValueError(
        f"{checkpoint} 缺少参考布局元数据；请保留原params/env.yaml，不能仅凭53维猜测位置/速度通道"
      )
    # BaseLoader reads tagged training YAML as plain containers/strings, never Python objects.
    with params_path.open(encoding="utf-8") as stream:
      raw = yaml.load(stream, Loader=yaml.BaseLoader)["commands"]["motion"]

  flags = []
  for name in ("reference_key_body_pos", "reference_key_body_vel"):
    value = raw[name]
    if isinstance(value, bool):
      flags.append(value)
    elif isinstance(value, str) and value in ("true", "false"):
      flags.append(value == "true")
    else:
      raise ValueError(f"无效的参考布局 {name}: {value!r}")
  position, velocity = flags
  anchor_pose = raw.get("reference_anchor_pose", False)
  if isinstance(anchor_pose, str) and anchor_pose in ("true", "false"):
    anchor_pose = anchor_pose == "true"
  if not isinstance(anchor_pose, bool):
    raise TypeError(f"Invalid reference_anchor_pose: {anchor_pose!r}")
  if anchor_pose and not hasattr(command, "reference_anchor_pose"):
    raise ValueError("This command implementation does not support reference_anchor_pose")
  bodies = raw["reference_key_bodies"]
  if not isinstance(bodies, (list, tuple)) or not all(isinstance(name, str) for name in bodies):
    raise ValueError("reference_key_bodies必须是刚体名称列表")
  if len(set(bodies)) != len(bodies) or any(name not in command.body_names for name in bodies):
    raise ValueError(f"参考关键刚体与当前机器人不符: {bodies}")
  window = int(raw["reference_window"])
  stride = float(raw["reference_stride"])
  if window < 0 or not math.isfinite(stride) or stride <= 0:
    raise ValueError(f"无效的参考窗口: window={window}, stride={stride}")
  feature_dim = 38 + len(bodies) * 3 * (position + velocity) + 9 * anchor_pose
  state = saved["actor_state_dict"]
  if state["command_encoder.0.weight"].shape[1] != feature_dim:
    raise ValueError("存档参考布局与command_encoder权重维度不一致")
  if state["pos_embedding"].shape[1] != 2 * window + 1:
    raise ValueError("存档参考窗口与pos_embedding长度不一致")

  command.reference_key_bodies = tuple(bodies) if position or velocity else ()
  command.reference_key_body_pos = position
  command.reference_key_body_vel = velocity
  if hasattr(command, "reference_anchor_pose"):
    command.reference_anchor_pose = anchor_pose
  command.reference_window = window
  command.reference_stride = stride
  actor.reference_dim = feature_dim
  print(f"[INFO]: 按存档恢复参考输入: {feature_dim}维, key_pos={position}, key_vel={velocity}；不裁剪旧策略")
