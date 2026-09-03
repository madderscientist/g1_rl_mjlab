"""把训练好的跟踪策略在指定动作上的表现渲成视频。

与 ``render_motion.py`` 的区别：那个只做参考轨迹的运动学回放，这个跑完整物理 + 策略，
看的是「机器人到底跟不跟得住」。片段之间会打上动作名和存活时长。

    python scripts/render_policy.py \
        --checkpoint artifacts/final_model_215787/model_215787.pt \
        --motions walk1_subject1,jumps1_subject1 --output logs/render/policy.mp4
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import numpy as np
import torch
import tyro
from PIL import Image, ImageDraw, ImageFont

import g1_lower_rl.tasks  # noqa: F401  注册任务

TASK = "G1-Gloria-MotionTracking"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def _label(frame: np.ndarray, lines: list[str]) -> np.ndarray:
  img = Image.fromarray(frame)
  draw = ImageDraw.Draw(img)
  try:
    font = ImageFont.truetype(FONT_PATH, 22)
  except OSError:
    font = ImageFont.load_default()
  for i, text in enumerate(lines):
    xy = (12, 10 + i * 26)
    draw.text((xy[0] + 1, xy[1] + 1), text, fill=(0, 0, 0), font=font)
    draw.text(xy, text, fill=(255, 255, 255), font=font)
  return np.asarray(img)


def main(
  checkpoint: str,
  motions: str = "",
  output: str = "logs/render/policy.mp4",
  seconds: float = 12.0,
  width: int = 960,
  height: int = 720,
  device: str = "cuda:0",
  fps: int = 25,
) -> None:
  """``motions`` 为逗号分隔的动作名，留空则取语料里的前 6 条。"""
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  from g1_lower_rl.rl import load_trained_runner

  cfg = load_env_cfg(TASK, play=True)
  agent_cfg = load_rl_cfg(TASK)
  cfg.scene.num_envs = 1
  cfg.viewer.width, cfg.viewer.height = width, height

  env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode="rgb_array")
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = load_trained_runner(TASK, wrapped, agent_cfg, Path(checkpoint), device)
  policy = runner.get_inference_policy(device=device)

  cmd = env.command_manager.get_term("motion")
  names = list(cmd.motion.names)
  wanted = [m.strip() for m in motions.split(",") if m.strip()] or names[:6]
  missing = [m for m in wanted if m not in names]
  if missing:
    raise SystemExit(f"语料里没有这些动作: {missing}\n可选: {names[:10]} ...")

  # 控制 50 Hz，按 fps 抽帧。
  stride = max(1, round(50 / fps))
  max_steps = int(seconds * 50)

  out = Path(output)
  out.parent.mkdir(parents=True, exist_ok=True)
  writer = imageio.get_writer(out, fps=fps, macro_block_size=1)

  print(f"[渲染] {len(wanted)} 段动作 -> {out}")
  for name in wanted:
    idx = names.index(name)
    # 固定到这一条，绕开自适应采样。
    cmd.forced_motion_id = idx
    # reset 与 step 必须在同一个 inference_mode 里：跨界就地改张量会报错。
    with torch.inference_mode():
      env.reset()
      obs = wrapped.get_observations()
      alive = 0
      for step in range(max_steps):
        obs, _, dones, _ = wrapped.step(policy(obs))
        alive += 1
        if step % stride == 0:
          writer.append_data(
            _label(env.render(), [name, f"{alive / 50:.1f}s"])  # type: ignore[arg-type]
          )
        if bool(dones[0]):
          break
    print(f"  {name:<28} 存活 {alive / 50:>5.1f}s{'  (跟满)' if alive >= max_steps else ''}")

  writer.close()
  env.close()
  print(f"[渲染] 完成: {out}")


if __name__ == "__main__":
  tyro.cli(main)
