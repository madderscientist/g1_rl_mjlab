"""把训练好的 GMT 策略导出成 ONNX，附上部署端需要的全部契约。

部署端不再抄一份关节顺序/默认位姿/动作缩放——凡是会和权重一起变的东西，都写进
ONNX 的 metadata，读的时候对不上就拒绝启动。

**参考动作不打包进 ONNX**。mjlab 自带的 tracking 导出会把整条轨迹塞成模型 buffer，
换一段动作就得重新导出一次；而这个任务的目的恰恰是「准备很多动作在机器人上放」，
所以参考轨迹留在外部 NPZ，部署端自己按时间索引取帧。
"""

from __future__ import annotations

import copy
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


def _restore_actor_group(cfg) -> None:
  """训练时 ``actor`` 组被删掉（死代码，每步白算，删掉提速 12.9%），导出得补回来。

  mjlab 的 ``get_base_metadata`` 和本仓的 ``deployment_metadata`` 都按
  ``active_terms["actor"]`` 取观测规格来生成部署契约。没有它会 KeyError，
  ONNX 照样导得出来但契约是空的，部署端不知道观测该怎么喂——是个静默的坑。

  用 ``rg_*`` 各组的并集重建，而不是拿 critic 当模板：critic 含 command、body_pos
  这些特权信息，写进部署契约会误导部署端去准备它拿不到的量。rg_* 的并集才是
  策略真正消费的本体观测。
  """
  from mjlab.managers.observation_manager import ObservationGroupCfg

  if "actor" in cfg.observations:
    return
  terms = {}
  for group in ("rg_projected_gravity", "rg_base_ang_vel", "rg_joint_pos", "rg_joint_vel", "rg_actions"):
    if group in cfg.observations:
      for name, term in cfg.observations[group].terms.items():
        terms[name] = copy.deepcopy(term)
  if not terms:
    raise KeyError("没有可用于重建 actor 组的 rg_* 观测组")
  cfg.observations["actor"] = ObservationGroupCfg(
    terms=terms, concatenate_terms=True, enable_corruption=False
  )


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
  _restore_actor_group(cfg)

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
    # 窗口由 _window_offsets() 实算（含历史帧，非均匀），cfg.lookahead_steps 是遗留字段，对不上。
    offsets = [int(x) for x in motion_cmd._window_offsets().tolist()]
    ref_dim = int(motion_cmd.reference_tokens.shape[-1])
    feature_dim, rem = divmod(ref_dim, len(offsets))
    assert rem == 0, f"参考窗口契约与实际不符: {ref_dim} 不能被 {len(offsets)} 整除"
    key_bodies = list(motion_cmd.cfg.reference_key_bodies)
    layout = "lin_vel_local3,ang_vel_local3,proj_gravity3,joint_pos29"
    if key_bodies:
      # 这几维是**机器人当前 anchor 的 yaw 局部系**下的参考 key body 位置，
      # 部署端必须用机器人自身位姿去算，用参考位姿算等于把误差信号抹掉。
      layout += f",key_body_pos_in_robot_anchor_yaw{len(key_bodies) * 3}"
      if motion_cmd.cfg.reference_key_body_vel:
        layout += f",key_body_vel_in_robot_anchor_yaw{len(key_bodies) * 3}"
    expect = 38 + len(key_bodies) * (6 if motion_cmd.cfg.reference_key_body_vel else 3)
    assert feature_dim == expect, (
      f"契约 layout 与实际维度不符: layout 描述 {expect}，实际 {feature_dim}。"
      "部署端会按 layout 排布输入，对不上就是静默错位。"
    )
    metadata.update(
      {
        "lookahead_steps": offsets,
        "lookahead_feature_dim": feature_dim,
        "lookahead_layout": layout,
        "reference_key_bodies": key_bodies,
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
