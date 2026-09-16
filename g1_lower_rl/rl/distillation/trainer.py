"""独立于任务的 DAgger：实际状态采样、老师标注、监督更新与检查点交接。"""

import json
import os
import time
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Protocol

import torch
from tensordict import TensorDict
from torch import distributed, nn
from torch.utils.tensorboard import SummaryWriter

from .models import ModelSpec
from .replay import DaggerBuffer, average_gradients, teacher_probability


class DistillationEnv(Protocol):
  """环境适配接口；历史动作必须来自实际执行动作，而非老师的假想输出。"""

  num_envs: int

  def reset(self) -> tuple[TensorDict, Any]: ...

  def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, Any]: ...

  def close(self) -> None: ...


@dataclass
class DaggerConfig:
  """本次新增轮数和每卡采样/优化预算；teacher_steps 按累计轮次退火。"""

  iterations: int = 1000
  rollout_steps: int = 24
  teacher_steps: int = 200
  replay_capacity: int = 65536
  batch_size: int = 2048
  updates_per_iteration: int = 12
  learning_rate: float = 3e-4
  save_interval: int = 100
  seed: int = 42
  device: str = "cuda:0"
  observation_keys: tuple[str, ...] | None = None

  def validate(self) -> None:
    if min(self.iterations, self.rollout_steps, self.replay_capacity,
           self.batch_size, self.updates_per_iteration, self.save_interval) <= 0:
      raise ValueError("Training sizes must be positive")
    if self.learning_rate <= 0:
      raise ValueError("Learning rate must be positive")


def run_distillation(
  environment_factory: Callable[[str, int], DistillationEnv],
  teacher_spec: ModelSpec,
  student_spec: ModelSpec,
  config: DaggerConfig,
  output_dir: str | Path,
  resume_checkpoint: str | Path | None = None,
  initialize: Callable[[nn.Module, nn.Module], None] | None = None,
  metrics_fn: Callable[[DistillationEnv], dict[str, torch.Tensor]] | None = None,
  metadata_fn: Callable[[DistillationEnv], dict[str, Any]] | None = None,
  metadata: dict[str, Any] | None = None,
  checkpoint_extras: dict[str, Any] | None = None,
) -> Path:
  """运行一轮独立的监督蒸馏，不执行 PPO，也不自动切换到自主探索阶段。

  environment_factory(device, seed) 负责环境及任务配置；模型构造由 ModelSpec
  提供。师生输出必须具有相同动作语义和 [num_envs, action_dim] 形状。
  initialize 只用于新学生的可选参数迁移，不覆盖学生检查点或续训权重。
  metrics_fn 返回标量指标，仅用于日志，不能影响 loss。

  每卡独立缓存最近样本，以同步梯度更新同一个学生。缓存是有界 FIFO，不是
  全历史数据集。续训只恢复权重、Adam 和累计轮次；仿真、缓存及 RNG 重新开始。
  """
  config.validate()
  output = Path(output_dir)
  env = None
  resources = ExitStack()
  rank_output = None
  info: dict[str, Any] = {}
  try:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = config.device
    if world_size > 1 and not distributed.is_initialized():
      device = f"cuda:{int(os.environ['LOCAL_RANK'])}"
      torch.cuda.set_device(device)
      distributed.init_process_group("nccl", timeout=timedelta(minutes=15))
      resources.callback(distributed.destroy_process_group)
    if distributed.is_initialized():
      world_size = distributed.get_world_size()
      rank = distributed.get_rank()
      if distributed.get_backend() == "nccl":
        device = f"cuda:{int(os.environ['LOCAL_RANK'])}"
        torch.cuda.set_device(device)
    else:
      rank = 0
    torch.manual_seed(config.seed + rank)
    if rank == 0:
      output.mkdir(parents=True, exist_ok=False)
    if world_size > 1:
      distributed.barrier()
    rank_output = output if rank == 0 else output / f"rank_{rank}"
    rank_output.mkdir(parents=True, exist_ok=True)
    info = {**(metadata or {}), **asdict(config), "rank": rank, "world_size": world_size,
            "teacher_checkpoint": str(teacher_spec.checkpoint) if teacher_spec.checkpoint else None,
            "method": "bounded_online_DAgger", "training_phase": "phase1_dagger_supervised",
            "ppo_updates": False, "exploration_noise": False, "automatic_phase2": False,
            "status": "initializing"}
    (rank_output / "run.json").write_text(json.dumps(info, indent=2))

    # 顺序建环境降低大动作库的主机内存峰值；完成后各卡独立并行采样。
    for loading_rank in range(world_size):
      if rank == loading_rank:
        env = environment_factory(device, config.seed + rank)
        resources.callback(env.close)
      if world_size > 1:
        distributed.barrier()
    assert env is not None
    writer = SummaryWriter(str(rank_output))
    resources.callback(writer.close)
    with torch.no_grad():
      observations, _ = env.reset()
    keys = config.observation_keys or tuple(observations.keys())

    def actor_observations(values):
      return TensorDict({name: values[name] for name in keys}, batch_size=[env.num_envs])

    template = actor_observations(observations)
    teacher, teacher_config = teacher_spec.build(template, device)
    student, student_config = student_spec.build(template, device)
    teacher.requires_grad_(False)
    teacher.eval()
    if world_size > 1:
      for value in teacher.state_dict().values():
        distributed.broadcast(value, src=0)
    teacher_state = {name: value.detach().cpu().clone() for name, value in teacher.state_dict().items()}
    if initialize is not None and student_spec.checkpoint is None and resume_checkpoint is None:
      initialize(teacher, student)
    if world_size > 1:
      for value in student.state_dict().values():
        distributed.broadcast(value, src=0)
    parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
    start_iteration = 0
    if resume_checkpoint is not None:
      saved = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
      student.load_state_dict(saved["actor_state_dict"], strict=True)
      optimizer.load_state_dict(saved["dagger_optimizer_state_dict"])
      start_iteration = int(saved["dagger_iteration"])
      info.update(resume_checkpoint=str(Path(resume_checkpoint).resolve()), optimizer_restored=True,
                  replay_restored=False, environment_restored=False)
    student.eval()
    with torch.no_grad():
      labels = teacher(template)
      prediction = student(template)
    if labels.ndim != 2 or labels.shape[0] != env.num_envs or labels.shape != prediction.shape:
      raise ValueError("Teacher/student outputs must match [num_envs, action_dim]")
    action_dim = labels.shape[1]
    replay = DaggerBuffer(template, action_dim, config.replay_capacity)
    end_iteration = start_iteration + config.iterations
    info.update(metadata_fn(env) if metadata_fn else {})
    info.update(teacher_config=teacher_config, actor_config=student_config,
                action_dim=action_dim, num_envs=env.num_envs, total_envs=env.num_envs * world_size,
                start_iteration=start_iteration, target_iteration=end_iteration,
                learning_rate=optimizer.param_groups[0]["lr"], status="running")
    (rank_output / "run.json").write_text(json.dumps(info, indent=2))
    started = time.monotonic()

    def save(iteration):
      # 老师权重必须保持不变；所有进程参与同步检查，只有 rank 0 写模型。
      for name, value in teacher.state_dict().items():
        torch.testing.assert_close(value.cpu(), teacher_state[name], rtol=0, atol=0)
      target = output / f"model_dagger_{iteration:06d}.pt"
      if world_size > 1:
        flat = torch.nn.utils.parameters_to_vector(student.parameters()).detach()
        expected = flat.clone()
        distributed.broadcast(expected, src=0)
        difference = (flat - expected).abs().max()
        distributed.all_reduce(difference, op=distributed.ReduceOp.MAX)
        if difference.item() != 0:
          raise RuntimeError("Student parameters diverged between ranks")
      if rank == 0:
        payload = {"actor_state_dict": student.state_dict(), "model_config": student_config,
                   "iter": 0, "dagger_iteration": iteration,
                   "dagger_optimizer_state_dict": optimizer.state_dict(),
                   "infos": {"distillation": {**info, "dagger_iteration": iteration}}}
        if set(payload) & set(checkpoint_extras or {}):
          raise ValueError("checkpoint_extras cannot overwrite trainer state")
        payload.update(checkpoint_extras or {})
        temporary = target.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(target)
      return target

    def require_finite(*values):
      valid = torch.stack([torch.isfinite(value).all() for value in values]).all().int()
      if world_size > 1:
        distributed.all_reduce(valid, op=distributed.ReduceOp.MIN)
      if not valid.item():
        raise FloatingPointError("Nonfinite distillation values on at least one rank")

    save(start_iteration)
    with (rank_output / "metrics.jsonl").open("a") as log:
      for iteration in range(start_iteration, end_iteration):
        beta = teacher_probability(iteration, config.teacher_steps)
        totals: dict[str, torch.Tensor] = {}
        student.eval()
        with torch.no_grad():
          for _ in range(config.rollout_steps):
            current = actor_observations(observations)
            labels = teacher(current)
            predicted = student(current)
            require_finite(labels, predicted)
            replay.add(current, labels)
            use_teacher = torch.rand((env.num_envs, 1), device=device) < beta
            observations, rewards, dones, _ = env.step(torch.where(use_teacher, labels, predicted))
            values = {"online_label_mse": (predicted - labels).square().mean(),
                      "teacher_fraction": use_teacher.float().mean(),
                      "reward": rewards.mean(), "done_fraction": dones.float().mean()}
            if metrics_fn:
              values.update(metrics_fn(env))
            for name, value in values.items():
              totals[name] = totals.get(name, torch.zeros((), device=device)) + value.detach().mean()
        # 监督标签始终来自老师，奖励仅记录；平均梯度之后再裁剪并更新 Adam。
        student.train()
        loss_total = torch.zeros((), device=device)
        for _ in range(config.updates_per_iteration):
          batch, labels = replay.sample(config.batch_size)
          loss = (student(batch) - labels).square().mean()
          require_finite(loss)
          optimizer.zero_grad(set_to_none=True)
          loss.backward()
          average_gradients(student)
          torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
          optimizer.step()
          loss_total += loss.detach()
        record = {}
        for name, value in totals.items():
          if world_size > 1:
            distributed.all_reduce(value)
          record[name] = value.item() / world_size / config.rollout_steps
        if world_size > 1:
          distributed.all_reduce(loss_total)
        record.update(iteration=iteration + 1, beta=beta, replay_size=replay.size,
                      train_mse=loss_total.item() / world_size / config.updates_per_iteration,
                      elapsed_s=time.monotonic() - started)
        log.write(json.dumps(record) + "\n")
        log.flush()
        for name, value in record.items():
          writer.add_scalar(name, value, iteration + 1)
        if rank == 0 and (iteration == start_iteration or (iteration + 1) % 10 == 0):
          print(json.dumps(record), flush=True)
        if (iteration + 1) % config.save_interval == 0 and iteration + 1 < end_iteration:
          save(iteration + 1)
    final_path = save(end_iteration)
    resources.close()
    info.update(status="completed", final_checkpoint=str(final_path), elapsed_s=time.monotonic() - started)
    (rank_output / "run.json").write_text(json.dumps(info, indent=2))
    if rank == 0:
      print(f"DAGGER_COMPLETE={final_path}", flush=True)
    return final_path
  except BaseException as error:
    if rank_output is not None:
      info.update(status="failed", error_type=type(error).__name__, error=str(error))
      (rank_output / "run.json").write_text(json.dumps(info, indent=2))
    raise
  finally:
    resources.close()