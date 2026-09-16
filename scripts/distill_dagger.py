"""G1/RGMT 师生训练入口；通用 DAgger 实现在 rl/distillation/。

本文件只选择老师、学生和环境，不实现采样、梯度同步或检查点恢复循环。
默认使用归档版 RGMT；学生尺寸可由显式 JSON 配置指定。
可用 JSON 模型配置替换默认尺寸，或使用带配置的学生检查点初始化/续训。
第一轮只优化老师监督损失，学生执行动作不等于 PPO 探索；第二轮独立启动。
"""

import copy
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

from g1_lower_rl.rl.distillation import DaggerConfig, ModelSpec, run_distillation
from g1_lower_rl.rl.distillation.models import resolve_model_config
from g1_lower_rl.rl.rgmt_model import RgmtActor
from g1_lower_rl.tasks.motion_tracking.env_cfg import motion_tracking_env_cfg
from g1_lower_rl.tasks.motion_tracking.rl_cfg import motion_tracking_ppo_runner_cfg

DEFAULT_TEACHER = "logs/rsl_rl/g1_gloria_gmt/2026-08-26_14-14-19_stage2_ctrl/model_215787.pt"


def initialize_student(teacher, student) -> None:
  """迁移 RGMT 的兼容编码器，保留学生 MLP 初始化，不修改老师。"""
  source = teacher.state_dict()
  target = student.state_dict()
  for name, value in target.items():
    if name.startswith("mlp."):
      continue
    if name not in source or source[name].shape != value.shape:
      raise ValueError(f"Incompatible teacher/student branch: {name}")
    target[name] = source[name]
  student.load_state_dict(target, strict=True)


def build_actor(observations, config, groups, action_dim):
  """RGMT 项目适配器：输入配置描述结构，版本标签不参与实例化。

  构造函数仅返回确定性 actor；探索分布参数保存在权重中，但第一轮冻结。
  通用训练包不要求这个类，其他模型可自行提供相同接口的 factory。
  """
  options = {name: config[name] for name in (
    "hidden_dims", "activation", "obs_normalization", "distribution_cfg", "history_len",
    "reference_dim", "token_dim", "num_heads", "history_layers", "fsq_levels", "fast_last_layer",
    "reference_group", "action_group",
  ) if name in config}
  model = RgmtActor(observations, groups, "actor", action_dim, **copy.deepcopy(options))
  if model.distribution is not None:
    model.distribution.requires_grad_(False)
  return model


def model_config(saved, config_file, fallback):
  """读取应用层 JSON，配置优先级由通用包统一解释。"""
  config = json.loads(Path(config_file).read_text()) if config_file is not None else None
  return resolve_model_config(config, saved, fallback)


def main(
  teacher_checkpoint: str = DEFAULT_TEACHER,
  teacher_model_config: str | None = None,
  student_model_config: str | None = None,
  student_checkpoint: str | None = None,
  motion_dir: str = "motions/lafan1",
  output_dir: str | None = None,
  resume_checkpoint: str | None = None,
  device: str = "cuda:0",
  num_envs: int = 256,
  iterations: int = 1000,
  rollout_steps: int = 24,
  teacher_steps: int = 200,
  replay_capacity: int = 65536,
  batch_size: int = 2048,
  updates_per_iteration: int = 12,
  learning_rate: float = 3e-4,
  save_interval: int = 100,
  seed: int = 42,
):
  """组装 G1 配方，运行第一轮监督蒸馏。

  teacher_model_config/student_model_config: 模型配置 JSON 路径，不是版本名称。
  student_checkpoint: 只初始化学生权重；resume_checkpoint: 还恢复 Adam 和轮次。
  iterations 为新增轮数；num_envs、batch_size、replay_capacity 均为每卡预算。
  teacher_steps 按累计轮次退火；已超过该轮数的续训不会重新启用老师接管。
  环境随机化保持归档配置。双卡必须显式传 output_dir。
  """
  if student_checkpoint is not None and resume_checkpoint is not None:
    raise ValueError("Choose student_checkpoint for initialization or resume_checkpoint for continuation, not both")
  torch.set_num_threads(4)
  teacher_default = motion_tracking_ppo_runner_cfg()
  student_default = motion_tracking_ppo_runner_cfg()
  groups = student_default.obs_groups
  saved_teacher = torch.load(teacher_checkpoint, map_location="cpu", weights_only=False)
  teacher_cfg = model_config(saved_teacher, teacher_model_config, asdict(teacher_default.actor))
  extras = {"critic_state_dict": saved_teacher["critic_state_dict"]} if "critic_state_dict" in saved_teacher else {}
  student_path = student_checkpoint or resume_checkpoint
  saved_student = torch.load(student_path, map_location="cpu", weights_only=False) if student_path else None
  student_cfg = model_config(saved_student, student_model_config, asdict(student_default.actor))
  del saved_teacher, saved_student
  cfg = motion_tracking_env_cfg(motion_dir=motion_dir)
  action_dim = len(cfg.commands["motion"].policy_joint_names)
  for role, actor_cfg in (("teacher", teacher_cfg), ("student", student_cfg)):
    if actor_cfg.get("reference_dim") != student_default.actor.reference_dim:
      raise ValueError(f"{role} reference dimension does not match configured environment")

  def factory(observations, actor_cfg):
    return build_actor(observations, actor_cfg, groups, action_dim)

  def environment_factory(actual_device, actual_seed):
    cfg.seed = actual_seed
    cfg.scene.num_envs = num_envs
    return RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=actual_device), clip_actions=student_default.clip_actions)

  def metrics(env):
    command = env.unwrapped.command_manager.get_term("motion")
    return {"body_error_m": torch.linalg.vector_norm(command.body_pos_relative_w - command.robot_body_pos_w, dim=-1).mean()}

  def environment_metadata(env):
    command = env.unwrapped.command_manager.get_term("motion")
    return {"corpus_clips": command.motion.num_motions, "corpus_frames": command.motion.time_step_total}

  run_config = DaggerConfig(
    iterations=iterations, rollout_steps=rollout_steps, teacher_steps=teacher_steps,
    replay_capacity=replay_capacity, batch_size=batch_size, updates_per_iteration=updates_per_iteration,
    learning_rate=learning_rate, save_interval=save_interval, seed=seed, device=device,
    observation_keys=tuple(groups["actor"]),
  )
  if output_dir is None:
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
      raise ValueError("Distributed runs require an explicit shared output directory")
    output_dir = str(Path("logs/dagger") / datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_rgmt_phase1"))
  return run_distillation(
    environment_factory, ModelSpec(factory, teacher_cfg, teacher_checkpoint),
    ModelSpec(factory, student_cfg, student_checkpoint), run_config, output_dir,
    resume_checkpoint=resume_checkpoint, initialize=initialize_student,
    metrics_fn=metrics, metadata_fn=environment_metadata, checkpoint_extras=extras,
    metadata={"motion_dir": motion_dir, "obs_groups": groups,
              "teacher_hidden_dims": teacher_cfg["hidden_dims"], "student_hidden_dims": student_cfg["hidden_dims"],
              "critic_origin": "teacher initialization only; no DAgger value training"},
  )


if __name__ == "__main__":
  tyro.cli(main)