"""带课程状态持久化的 PPO runner。

mjlab 的 ``MjlabOnPolicyRunner`` 只把 ``common_step_counter`` 存进 checkpoint，课程项
自己的档位和闸门分数没存。这两者缺一不可：续训时步数条件全部满足、而分数从 0 重来，
档位能不能爬回去完全取决于分数能不能重新越过 ``min_fraction``。

**实测代价：** 闸门取 ``err<0.4 比例>=0.5`` 时分数封顶 0.43，三条上肢扰动课程在整整
3000 迭代里一次都没升过档——力矩/外力/漂移全是字面意义的 0，而同期对照组是满扰动
（4.0 Nm / 20 N / drift 1.0）。两者的所有指标对比因此全部作废。高度 std 课程同理，
续训会把它从 0.05 退回 0.08。

这个坑踩过两次：早先分数初值是 1.0，续训时闸门一开就是开的、二十迭代顶满；改成 0.0
之后变成另一个极端。**根因不是初值，是状态没跟着 checkpoint 走。**
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx
from mjlab.tasks.registry import load_runner_cls
from mjlab.tasks.velocity.rl.runner import VelocityOnPolicyRunner


def action_joint_names(env) -> list[str]:
  """动作项实际驱动的关节，按动作向量的顺序。"""
  return list(env.action_manager.get_term("joint_pos").target_names)


def deployment_metadata(env) -> dict[str, str]:
  """mjlab 的 ``get_base_metadata`` 漏掉、而部署端必需的那几个关节名单。

  ``joint_names`` 写的是模型**全部**关节，``action_scale`` 只覆盖动作关节，两者长度不同。
  靠 ``joint_names[:n]`` 截断去猜在别的任务上会错位（29 自由度那版里 ``left_eccentric_joint``
  夹在左右臂中间，右臂整体差一格）。观测里的关节子集也不一定等于动作关节：站立任务
  观测全身 29 轴、只驱动下肢 15 轴。两份名单都得显式写出去。
  """
  metadata = {"action_joint_names": ",".join(action_joint_names(env))}
  all_joints = env.scene["robot"].joint_names
  for term_name in ("joint_pos", "joint_vel"):
    asset_cfg = env.observation_manager.get_term_cfg("actor", term_name).params.get(
      "asset_cfg"
    )
    # 没有 asset_cfg 的观测项取的是全部关节（动作跟踪任务就是这样，含两个夹爪轴）。
    ids = asset_cfg.joint_ids if asset_cfg is not None else slice(None)
    selected = (
      all_joints[ids] if isinstance(ids, slice) else [all_joints[i] for i in ids]
    )
    metadata[f"obs_{term_name}_joint_names"] = ",".join(selected)
  return metadata


def latest_checkpoint(root: Path) -> Path:
  """``{log_root}/{experiment_name}`` 下 mtime 最新的检查点。"""
  ckpts = sorted(root.glob("*/model_*.pt"), key=lambda p: p.stat().st_mtime)
  if not ckpts:
    raise FileNotFoundError(f"{root} 下没有找到检查点")
  return ckpts[-1]


def load_trained_runner(task_id: str, env, agent_cfg, checkpoint: Path, device: str):
  """建 runner 并载入 actor 权重。

  必须走任务注册表拿 runner 类：本包的 ``GloriaOnPolicyRunner`` 才认得存档里的
  课程状态，用基类会静默丢掉。
  """
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
  )
  return runner


class GloriaOnPolicyRunner(VelocityOnPolicyRunner):
  """保存课程状态，并将基类每次导出的 ONNX 固定命名为 policy.onnx。

  与任务无关：无课程的任务存空字典，关节名单从环境里读。所有任务都该用它，
  否则落回基类就只存 .pt、没有 ONNX 和部署元数据。
  """

  env: RslRlVecEnvWrapper

  @staticmethod
  def _get_export_paths(checkpoint_path: str) -> tuple[Path, str, Path]:
    export_dir = Path(checkpoint_path).parent
    filename = "policy.onnx"
    return export_dir, filename, export_dir / filename

  def _stateful_curriculum_terms(self):
    """返回 (名字, 课程实例)——只挑实现了 state_dict 的。"""
    manager = getattr(self.env.unwrapped, "curriculum_manager", None)
    for name in getattr(manager, "active_terms", ()):
      func = manager.get_term_cfg(name).func
      if hasattr(func, "state_dict") and hasattr(func, "load_state_dict"):
        yield name, func

  def save(self, path: str, infos=None):
    curriculum_state = {
      name: term.state_dict() for name, term in self._stateful_curriculum_terms()
    }
    # 父类会覆盖 infos["env_state"]，但其余键原样保留，所以另起一个键。
    super().save(path, {**(infos or {}), "curriculum_state": curriculum_state})

    # 父类导出时不写这几项，不补的话只有手动跑 export_onnx.py 的产物才带。
    _, _, onnx_path = self._get_export_paths(path)
    if not onnx_path.exists():
      return  # 父类的导出是 try/except 包着的，失败时不该连累存档。
    try:
      attach_metadata_to_onnx(str(onnx_path), deployment_metadata(self.env.unwrapped))
    except Exception as e:
      print(f"[WARN] 写入部署元数据失败（训练继续）: {e}")

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    infos = super().load(path, load_cfg, strict, map_location)
    saved = (infos or {}).get("curriculum_state") or {}
    restored, missing = [], []
    for name, term in self._stateful_curriculum_terms():
      if name in saved:
        term.load_state_dict(saved[name])
        restored.append(f"{name}={saved[name]['stage']}")
      else:
        missing.append(name)
    if restored:
      print(f"[INFO]: 已恢复课程档位: {', '.join(restored)}")
    if missing:
      # 必须刺眼。静默回退到第 0 档正是这个 bug 当初没被发现的原因。
      print(
        "\n"
        + "!" * 78
        + "\n[警告]: 以下课程没有存档状态，将从第 0 档、分数 0 重新开始：\n"
        f"        {', '.join(missing)}\n"
        "        存档早于课程状态持久化功能。续训后这些课程可能永远升不回原来的档位，\n"
        "        指标会因为“扰动变小”而虚高，与其它运行不可比。\n"
        "        训练开始后请立刻确认 Curriculum/<名字> 是否回到了预期值。\n"
        + "!" * 78
        + "\n"
      )
    return infos
