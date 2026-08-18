"""回放脚本：加载检查点（或 zero/random 假策略）并可视化。"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mjlab
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from g1_lower_rl.rl import latest_checkpoint, load_trained_runner


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  checkpoint_file: str | None = None
  """检查点路径。不给则取 ``{log_root}/{experiment_name}`` 下最新的一个。"""
  log_root: str = "logs/rsl_rl"
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  no_terminations: bool = False
  """关掉所有终止条件（用假策略看运动时有用）。"""
  motion_dir: str | None = None
  """覆盖动作语料目录，比如用 ``motions/bench_lafan1`` 只看评测基准那 80 条。"""


def _dummy_policy(kind: str, env: RslRlVecEnvWrapper):
  shape: tuple[int, ...] = env.unwrapped.action_space.shape
  device = env.unwrapped.device
  if kind == "zero":
    return lambda obs: torch.zeros(shape, device=device)
  return lambda obs: 2 * torch.rand(shape, device=device) - 1


def run_play(task_id: str, cfg: PlayConfig) -> None:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)
  trained = cfg.agent == "trained"

  if cfg.motion_dir is not None:
    env_cfg.commands["motion"].motion_dir = cfg.motion_dir

  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  resume_path: Path | None = None
  if trained:
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
    else:
      resume_path = latest_checkpoint(
        (Path(cfg.log_root) / agent_cfg.experiment_name).resolve()
      )
    print(f"[INFO]: Loading checkpoint: {resume_path}")

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  record = cfg.video and trained
  if cfg.video and not record:
    print("[WARN] 假策略没有 log_dir，录像已禁用。")
  env = ManagerBasedRlEnv(
    cfg=env_cfg, device=device, render_mode="rgb_array" if record else None
  )

  if record:
    assert resume_path is not None
    env = VideoRecorder(
      env,
      video_folder=resume_path.parent / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during play")

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  if trained:
    assert resume_path is not None
    runner = load_trained_runner(task_id, env, agent_cfg, resume_path, device)
    policy = runner.get_inference_policy(device=device)
  else:
    policy = _dummy_policy(cfg.agent, env)

  resolved = cfg.viewer
  if resolved == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved = "native" if has_display else "viser"

  if resolved == "native":
    NativeMujocoViewer(env, policy).run()
  else:
    ViserPlayViewer(env, policy).run()
  env.close()


def main() -> None:
  import g1_lower_rl.tasks  # noqa: F401  注册任务

  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(list_tasks()),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=f"{sys.argv[0]} {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
