"""满档扰动下的训练环境实况，6 宫格每格跟拍一个环境。

    python scripts/disturb_video.py <checkpoint.pt> <out.mp4> [seconds] [device]

用的是**训练**配置：观测噪声、延时、终止条件、push_robot 全部保留，并把所有课程档位
直接拨到最后一档。play 配置做不到这件事——它关掉了观测噪声和终止，画面会比真实训练干净。

每一格是一个独立环境的跟拍视角（同一台离屏渲染器逐格换 env_idx），指令、扰动强度系数、
复位都各自独立，所以能同时看到站立/直行/转弯三类环境在满档扰动下的表现。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402
from mjlab.viewer import ViewerConfig  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import g1_lower_rl.tasks  # noqa: F401,E402  注册任务
from g1_lower_rl.rl import load_trained_runner  # noqa: E402
from g1_lower_rl.tasks import DEFAULT_TASK  # noqa: E402
from g1_lower_rl.tasks.lower_body.cfg import max_out_curriculum  # noqa: E402
from g1_lower_rl.tasks.lower_body.mdp import disturbance_level  # noqa: E402

TASK = DEFAULT_TASK
COLS, ROWS = 3, 2
NUM_ENVS = COLS * ROWS
PANEL_W, PANEL_H = 480, 360
BANNER_H = 26
FPS = 25
STRIDE = 2  # 50 Hz 控制 -> 25 fps
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
RESET_FLASH_FRAMES = 8


def _font(size: int):
  if Path(FONT_PATH).exists():
    return ImageFont.truetype(FONT_PATH, size)
  return ImageFont.load_default()


FONT = _font(15)
BANNER_FONT = _font(16)


def mode_of(vx: float, vy: float, wz: float) -> str:
  """按指令幅值反推这个环境属于哪一类（比读命令项内部标志更稳）。"""
  moving = abs(vx) > 0.1 or abs(vy) > 0.1
  turning = abs(wz) > 0.1
  if not moving and not turning:
    return "STAND"
  if not moving:
    return "TURN"
  return "WALK+TURN" if turning else "STRAIGHT"


def annotate(
  frame: np.ndarray, env_id: int, cmd, height: float, level: float, reset: bool
) -> np.ndarray:
  vx, vy, wz = float(cmd[0]), float(cmd[1]), float(cmd[2])
  img = Image.fromarray(frame)
  draw = ImageDraw.Draw(img)
  lines = [
    f"#{env_id}  {mode_of(vx, vy, wz)}",
    f"v {vx:+.2f} {vy:+.2f}  w {wz:+.2f}",
    f"h {height:.2f}   disturb {level:.2f}",
  ]
  for i, text in enumerate(lines):
    xy = (8, 6 + i * 17)
    draw.text(xy, text, font=FONT, fill=(0, 0, 0))  # 描边，浅色地面上也读得清
    draw.text((xy[0] - 1, xy[1] - 1), text, font=FONT, fill=(255, 255, 255))
  if reset:
    draw.text((8, PANEL_H - 24), "RESET", font=FONT, fill=(255, 90, 90))
  draw.rectangle([0, 0, PANEL_W - 1, PANEL_H - 1], outline=(60, 60, 60))
  return np.asarray(img)


def main() -> None:
  if len(sys.argv) < 3:
    print(__doc__)
    raise SystemExit(1)
  checkpoint = Path(sys.argv[1])
  out = Path(sys.argv[2])
  seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0
  device = (
    sys.argv[4]
    if len(sys.argv) > 4
    else ("cuda:0" if torch.cuda.is_available() else "cpu")
  )

  configure_torch_backends()

  env_cfg = load_env_cfg(TASK, play=False)
  banner = max_out_curriculum(env_cfg)
  env_cfg.scene.num_envs = NUM_ENVS
  env_cfg.viewer = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY,
    entity_name="robot",
    body_name="torso_link",
    distance=2.9,
    elevation=-20.0,
    azimuth=110.0,
    width=PANEL_W,
    height=PANEL_H,
    env_idx=0,
    max_extra_envs=0,  # 每格只画自己，邻居会串进画面
  )

  agent_cfg = load_rl_cfg(TASK)
  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)

  runner = load_trained_runner(TASK, env, agent_cfg, checkpoint, device)
  policy = runner.get_inference_policy(device=device)

  # 离屏渲染器持有的就是这个 ViewerConfig 实例，逐格改 env_idx 即可换视角。
  view_cfg = raw_env._offline_renderer._cfg  # noqa: SLF001

  steps = int(seconds / raw_env.step_dt)
  print(f"[INFO] {banner}")
  print(f"[INFO] {NUM_ENVS} 环境 x {steps} 步 ({seconds:.0f}s)，输出 {out}")

  out.parent.mkdir(parents=True, exist_ok=True)
  writer = imageio.get_writer(str(out), fps=FPS, macro_block_size=1)
  reset_flash = np.zeros(NUM_ENVS, dtype=int)
  obs = env.get_observations()

  try:
    for step in range(steps):
      with torch.inference_mode():
        obs, _, dones, _ = env.step(policy(obs))
      reset_flash[dones.bool().cpu().numpy()] = RESET_FLASH_FRAMES

      if step % STRIDE:
        reset_flash = np.maximum(reset_flash - 1, 0)
        continue

      twist = raw_env.command_manager.get_command("twist").cpu().numpy()
      height = raw_env.command_manager.get_command("height").cpu().numpy()
      level = disturbance_level(raw_env).cpu().numpy()
      panels = []
      for i in range(NUM_ENVS):
        view_cfg.env_idx = i
        panels.append(
          annotate(
            raw_env.render(),
            i,
            twist[i],
            float(height[i, 0]),
            float(level[i]),
            reset_flash[i] > 0,
          )
        )
      grid = np.vstack(
        [np.hstack(panels[r * COLS : (r + 1) * COLS]) for r in range(ROWS)]
      )

      canvas = Image.new(
        "RGB", (grid.shape[1], grid.shape[0] + BANNER_H), (18, 18, 18)
      )
      canvas.paste(Image.fromarray(grid), (0, BANNER_H))
      ImageDraw.Draw(canvas).text(
        (10, 5), banner, font=BANNER_FONT, fill=(235, 235, 235)
      )
      writer.append_data(np.asarray(canvas))

      reset_flash = np.maximum(reset_flash - 1, 0)
      if step % 200 == 0:
        print(f"  {step}/{steps}")
  finally:
    writer.close()
    env.close()
  print(f"[OK] {out}")


if __name__ == "__main__":
  main()
