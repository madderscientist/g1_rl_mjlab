"""训练脚本。基于 mjlab 自带的 train，额外接上左右镜像数据增强（DUP）。"""

from __future__ import annotations

import faulthandler
import logging
import os
import signal
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import mjlab
import tyro
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder

from g1_lower_rl.tasks.lower_body.cfg.constants import MIRROR_STAGES


@dataclass(frozen=True)
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  agent: RslRlBaseRunnerCfg
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  enable_nan_guard: bool = False
  log_root: str = "logs/rsl_rl"
  torchrunx_log_dir: str | None = None
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])
  mirror_schedule: tuple[tuple[int, float], ...] = MIRROR_STAGES
  """左右镜像数据增强（DUP）的强度课程，一串 ``(起始迭代, 逐样本镜像概率)``。

  每个小批都扩成两倍，第二份里按该概率逐样本选“镜像”或“原样复制”。缺省值来自任务配置
  里的 ``MIRROR_STAGES``，要改档位去那里改；命令行也能覆盖，如
  ``--mirror-schedule '((0,0.0),(800,1.0))'``，关闭写 ``'()'``。字面量语法是 mjlab 的
  ``TYRO_FLAGS`` 里 ``UsePythonSyntaxForLiteralCollections`` 要求的。

  环境是左右对称的，但目标函数里没有任何一项要求策略对称，于是随机初始化
  带来的微小不对称会被 PPO 逐步放大。详见 ``g1_lower_rl/rl/mirror.py``。"""

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    return TrainConfig(env=load_env_cfg(task_id), agent=load_rl_cfg(task_id))


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  # 多卡训练挂起时唯一能拿到栈的手段：ptrace_scope=1 下 py-spy/gdb 都够不到 worker
  # （它们是兄弟进程不是祖先），只能让进程自己转储。`kill -USR1 <worker pid>`。
  faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)

  if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "":
    device, seed, rank = "cpu", cfg.agent.seed, 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)  # EGL 设备要跟 CUDA 设备一致
    device = f"cuda:{local_rank}"
    seed = cfg.agent.seed + rank  # 各进程种子不同，保证多样性

  configure_torch_backends()
  cfg.agent.seed = seed
  cfg.env.seed = seed
  print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")

  env = ManagerBasedRlEnv(
    cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
  )

  resume_path: Path | None = None
  if cfg.agent.resume:
    resume_path = get_checkpoint_path(
      log_dir.parent, cfg.agent.load_run, cfg.agent.load_checkpoint
    )

  # 只在 rank 0 录像，避免多个 worker 写同一批文件。
  if cfg.video and rank == 0:
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "train",
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")

  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

  agent_cfg = asdict(cfg.agent)
  env_cfg = asdict(cfg.env)

  if cfg.mirror_schedule:
    from g1_lower_rl.rl import MirrorAugmentation

    agent_cfg["algorithm"]["symmetry_cfg"] = {
      "use_data_augmentation": True,
      "use_mirror_loss": False,
      "mirror_loss_coeff": 0.0,
      "data_augmentation_func": MirrorAugmentation(
        cfg.mirror_schedule, cfg.agent.num_steps_per_env
      ),
    }

  # 存档写在 runner 创建之前：runner 会就地改 agent_cfg（塞进不可序列化的对象）。
  if rank == 0:
    serializable = dict(agent_cfg)
    if "symmetry_cfg" in serializable["algorithm"]:
      # 增强函数和它持有的 env 引用序列化不了，存档位就够了。
      serializable["algorithm"] = {
        **serializable["algorithm"],
        "symmetry_cfg": {"mirror_stages": [list(s) for s in cfg.mirror_schedule]},
      }
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", serializable)

  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, agent_cfg, str(log_dir), device)
  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path))

  runner.learn(
    num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
  )
  env.close()


def launch_training(task_id: str, args: TrainConfig | None = None) -> None:
  args = args or TrainConfig.from_task(task_id)

  log_root_path = (Path(args.log_root) / args.agent.experiment_name).resolve()
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name

  selected_gpus, num_gpus = select_gpus(args.gpu_ids)
  os.environ["CUDA_VISIBLE_DEVICES"] = (
    "" if selected_gpus is None else ",".join(map(str, selected_gpus))
  )
  os.environ["MUJOCO_GL"] = "egl"

  if num_gpus <= 1:
    run_train(task_id, args, log_dir)
    return

  import torchrunx

  logging.basicConfig(level=logging.INFO)  # torchrunx 把 stdout 重定向到 logging
  if "TORCHRUNX_LOG_DIR" not in os.environ:
    os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir or str(
      log_dir / "torchrunx"
    )
  print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
  torchrunx.Launcher(
    hostnames=["localhost"],
    workers_per_host=num_gpus,
    backend=None,  # 让 rsl_rl 自己初始化进程组
    copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
  ).run(run_train, task_id, args, log_dir)


def main() -> None:
  import g1_lower_rl.tasks  # noqa: F401  注册任务

  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(list_tasks()),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    prog=f"{sys.argv[0]} {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()
