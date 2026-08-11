"""把训练好的 GMT 策略导出成 ONNX，附上部署端需要的全部契约。

部署端不再抄一份关节顺序/默认位姿/动作缩放——凡是会和权重一起变的东西，都写进
ONNX 的 metadata，读的时候对不上就拒绝启动。

**参考动作不打包进 ONNX**。mjlab 自带的 tracking 导出会把整条轨迹塞成模型 buffer，
换一段动作就得重新导出一次；而这个任务的目的恰恰是「准备很多动作在机器人上放」，
所以参考轨迹留在外部 NPZ，部署端自己按时间索引取帧。
"""

from __future__ import annotations

import json
from pathlib import Path

import tyro

import g1_lower_rl.tasks  # noqa: F401  注册任务
from g1_lower_rl.rl import deployment_metadata, load_trained_runner


def _resolve_task_id(checkpoint: Path, task_id: str | None) -> str:
  """按日志目录中的 experiment_name 推断任务，必要时允许显式指定。"""
  if task_id is not None:
    return task_id

  from mjlab.tasks.registry import list_tasks, load_rl_cfg

  experiment_name = checkpoint.parent.parent.name
  matches = [
    candidate
    for candidate in list_tasks()
    if load_rl_cfg(candidate).experiment_name == experiment_name
  ]
  if len(matches) != 1:
    choices = ", ".join(matches) if matches else "无"
    raise ValueError(
      f"无法从实验目录 {experiment_name!r} 唯一推断任务（匹配项: {choices}），"
      "请传入 --task-id。"
    )
  return matches[0]


def main(
  checkpoint: str,
  output_dir: str | None = None,
  device: str = "cpu",
  task_id: str | None = None,
) -> None:
  """导出 ONNX 到 ``output_dir``（默认与检查点同目录）。"""
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  checkpoint_path = Path(checkpoint)
  if not checkpoint_path.is_file():
    raise FileNotFoundError(f"检查点不存在: {checkpoint_path}")
  resolved_task_id = _resolve_task_id(checkpoint_path, task_id)
  print(f"[导出] 任务: {resolved_task_id}")

  cfg = load_env_cfg(resolved_task_id, play=True)
  agent_cfg = load_rl_cfg(resolved_task_id)
  cfg.scene.num_envs = 1

  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = load_trained_runner(
    resolved_task_id, wrapped, agent_cfg, checkpoint_path, device
  )

  out = Path(output_dir) if output_dir else checkpoint_path.parent
  out.mkdir(parents=True, exist_ok=True)
  onnx_name = checkpoint_path.with_suffix(".onnx").name
  contract_name = f"{checkpoint_path.stem}_contract.json"
  onnx_path = out / onnx_name
  contract_path = out / contract_name
  runner.export_policy_to_onnx(str(out), onnx_name)

  robot = env.scene["robot"]

  metadata = get_base_metadata(env.unwrapped, "local")
  metadata.update(deployment_metadata(env.unwrapped))
  metadata.update(
    {
      "control_dt": env.step_dt,
      "task_id": resolved_task_id,
    }
  )
  if resolved_task_id == "G1-Gloria-MotionTracking":
    motion_cmd = env.command_manager.get_term("motion")
    metadata.update(
      {
        "lookahead_steps": list(motion_cmd.cfg.lookahead_steps),
        "lookahead_feature_dim": 39,
        "lookahead_layout": "height1,proj_gravity3,lin_vel_local3,ang_vel_local3,joint_pos29",
        "anchor_body_name": motion_cmd.cfg.anchor_body_name,
        "tracked_body_names": list(motion_cmd.cfg.body_names),
        "all_body_names": list(robot.body_names),
      }
    )
  attach_metadata_to_onnx(str(onnx_path), metadata)

  # 同时落一份可读的契约，便于人工核对与写部署测试。
  contract_path.write_text(
    json.dumps({k: v for k, v in metadata.items()}, ensure_ascii=False, indent=2),
    encoding="utf-8",
  )
  print(f"[导出] {onnx_path}")
  print(f"[导出] {contract_path}")
  print(f"  观测 {env.observation_manager.group_obs_dim['actor']} -> 动作 {env.action_manager.total_action_dim}")


if __name__ == "__main__":
  tyro.cli(main)
