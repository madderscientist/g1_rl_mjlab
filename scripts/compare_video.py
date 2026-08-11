"""多个策略的纵向对比视频。

    python scripts/compare_video.py A.onnx B.onnx C.onnx out.mp4

两遍式：先只跑物理、把 qpos 轨迹存下来，再按所有轨迹的合并范围定一个机位，最后重放
渲染。这样所有画面用的是完全同一台相机——跟拍相机会各自跟各自的机器人，谁走得远反而
看不出来。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from g1_lower_rl.deploy import (  # noqa: E402
  Index,
  build_model,
  load_policy,
  reset,
  rollout,
  set_arm_mode,
)

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
WIDTH, HEIGHT = 900, 400
FPS = 25
STRIDE = 2  # 50 Hz 控制 -> 25 fps
PANEL_COLOURS = (
  (120, 255, 140),
  (255, 200, 120),
  (120, 210, 255),
  (255, 140, 210),
  (255, 245, 120),
  (200, 170, 255),
)


@dataclass(frozen=True)
class Scenario:
  label: str
  command: tuple[float, float, float, float]
  seconds: float
  arm_target: tuple[float, ...] | None = None
  arm_kp: float = 0.1
  arm_kd: float = 0.0


# arm_target 顺序见 assets.ARM_JOINTS：先左臂再右臂，每侧依次为
# shoulder pitch/roll/yaw、elbow、wrist roll/pitch/yaw。None 表示手臂 limp。
SCENARIOS = (
  Scenario(
    "stand h=0.72 arms forward",
    (0.0, 0.0, 0.0, 0.72),
    7.0,
    arm_target=(
      -1.6, 0.25, 0.0, 1.5708, 0.0, 0.0, 0.0,
      -1.6, -0.25, 0.0, 1.5708, 0.0, 0.0, 0.0,
    ),
    arm_kp=30.0,
    arm_kd=2.0,
  ),
  Scenario("forward -0.5 m/s", (-0.5, 0.0, 0.0, 0.74), 6.0),
  Scenario("forward 1.0 m/s", (1.0, 0.0, 0.0, 0.7), 6.0, arm_kp=5.0, arm_kd=1.0),
  Scenario("strafe 0.3 m/s", (0.0, 0.3, 0.0, 0.8), 6.0, arm_kp=10.0, arm_kd=4.0),
  Scenario("squat-walk 0.3 m/s  h=0.62", (0.3, 0.0, 0.0, 0.62), 6.0),
  Scenario("squat in place  h=0.6", (0.0, 0.0, 0.0, 0.4), 6.0),
  Scenario("turn 1.5 rad/s", (0.0, 0.0, 1.5, 0.52), 6.0),
  Scenario("turn -1.5 rad/s", (0.0, 0.0, -1.5, 0.52), 6.0),
  Scenario("turn 0.5 rad/s", (0.0, 0.0, 0.5, 0.8), 6.0),
)


def configure_arms(model, data, index: Index, scenario: Scenario) -> None:
  """应用场景的手臂初始位姿和位置执行器 PD 参数。"""
  actuators = np.asarray(index.arm_actuator)
  model.actuator_gainprm[actuators, 0] = scenario.arm_kp
  model.actuator_biasprm[actuators, 1] = -scenario.arm_kp
  model.actuator_biasprm[actuators, 2] = -scenario.arm_kd

  if scenario.arm_target is None:
    data.ctrl[actuators] = 0.0
    return

  target = np.asarray(scenario.arm_target, dtype=np.float64)
  if target.shape != actuators.shape:
    raise ValueError(f"arm_target 应有 {len(actuators)} 个值，实际为 {len(target)} 个")
  data.ctrl[actuators] = target
  joints = model.actuator_trnid[actuators, 0]
  data.qpos[model.jnt_qposadr[joints]] = target
  data.qvel[model.jnt_dofadr[joints]] = 0.0
  mujoco.mj_forward(model, data)


def simulate(model, policy_path: str, tag: str):
  """只跑物理，返回每个场景的 qpos 轨迹（已按 STRIDE 抽到 25 Hz）。"""
  data = mujoco.MjData(model)
  policy = load_policy(policy_path)
  index = Index(model, policy.action_joint_names)
  tracks = []
  for scenario in SCENARIOS:
    reset(model, data, policy.joint_names, policy.default_pos)
    configure_arms(model, data, index, scenario)
    log, _ = rollout(
      model,
      data,
      policy.session,
      index,
      policy.action_default_pos,
      policy.action_scale,
      scenario.command,
      scenario.seconds,
      None,
    )
    qpos = np.asarray(log["qpos"])[::STRIDE]
    travel = float(np.linalg.norm(qpos[-1, :2] - qpos[0, :2]))
    tracks.append((scenario, travel, qpos))
    print(f"  {tag:<12s} {scenario.label:<28s} traveled {travel:.2f} m")
  return tracks


def frame_camera(paths) -> mujoco.MjvCamera:
  """一个覆盖所有给定轨迹的固定机位。45° 斜视角；就地场景给一个近景。"""
  xy = np.concatenate([p[:, :2] for p in paths])
  lo, hi = xy.min(axis=0), xy.max(axis=0)
  centre = (lo + hi) / 2
  extent = float(np.max(hi - lo))
  camera = mujoco.MjvCamera()
  camera.type = mujoco.mjtCamera.mjCAMERA_FREE
  camera.lookat[:] = [centre[0], centre[1], 0.65]
  camera.distance = float(np.clip(1.15 * extent + 2.2, 3.2, 9.0))
  camera.azimuth = 45.0
  camera.elevation = -10.0
  return camera


def annotate(frame, lines, colour, font):
  image = Image.fromarray(frame)
  draw = ImageDraw.Draw(image)
  draw.rectangle([0, 0, image.width, 22 * len(lines) + 6], fill=(0, 0, 0))
  for row, text in enumerate(lines):
    draw.text((10, 4 + 22 * row), text, font=font, fill=colour)
  return np.asarray(image)


def main() -> None:
  if len(sys.argv) < 3:
    raise SystemExit(
      f"用法: {Path(sys.argv[0]).name} POLICY.onnx [POLICY.onnx ...] OUT.mp4"
    )

  policy_paths = sys.argv[1:-1]
  out = Path(sys.argv[-1])
  out.parent.mkdir(parents=True, exist_ok=True)
  font = (
    ImageFont.truetype(FONT_PATH, 17)
    if Path(FONT_PATH).exists()
    else ImageFont.load_default()
  )

  model = build_model()
  set_arm_mode(model, "limp")
  print("跑物理…")
  runs = []
  for i, policy_path in enumerate(policy_paths):
    tag = f"{i + 1}: {Path(policy_path).name}"
    runs.append((tag, PANEL_COLOURS[i % len(PANEL_COLOURS)], simulate(model, policy_path, tag)))

  print("渲染…")
  data = mujoco.MjData(model)
  renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
  frame_count = 0
  writer_context = cast(Any, imageio.get_writer(str(out), fps=FPS, macro_block_size=1))
  with writer_context as writer:
    for scenario_runs in zip(*(tracks for _, _, tracks in runs)):
      scenario = scenario_runs[0][0]
      camera = frame_camera([qpos for _, _, qpos in scenario_runs])
      scene_frames = max(len(qpos) for _, _, qpos in scenario_runs)
      frame = None
      for frame_index in range(scene_frames):
        panels = []
        for (tag, colour, _), (_, travel, qpos) in zip(runs, scenario_runs):
          data.qpos[:] = qpos[min(frame_index, len(qpos) - 1)]
          mujoco.mj_forward(model, data)
          renderer.update_scene(data, camera=camera)
          panels.append(
            annotate(
              renderer.render(),
              [tag, f"{scenario.label}   traveled {travel:.2f} m"],
              colour,
              font,
            )
          )
        frame = np.concatenate(panels, axis=0)
        writer.append_data(frame)
        frame_count += 1

      for _ in range(12):  # 场景之间停一下
        writer.append_data(frame)
        frame_count += 1
  del renderer

  print(f"\n视频: {out}  ({frame_count} 帧, {frame_count / FPS:.0f} s)")


if __name__ == "__main__":
  main()
