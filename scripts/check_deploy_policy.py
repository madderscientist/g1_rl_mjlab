"""上机前的最后一道检查：在 CPU 上闭环重跑部署链路。

    python scripts/check_deploy_policy.py policy.onnx [--video out.mp4] [--arms limp]

输出：
1. 每个指令场景的速度/高度跟随、腾空率、是否摔倒；
2. 目标位置相对关节行程和 ctrlrange 的越界率——部署侧的裁剪边界就是按这个定的。
"""

from __future__ import annotations

import argparse

import mujoco
import numpy as np

from g1_lower_rl.deploy import (
  Index,
  build_model,
  load_policy,
  reset,
  rollout,
  set_arm_mode,
)

SCENARIOS = (
  ("站立 h=0.74", (0.0, 0.0, 0.0, 0.74), 6.0),
  ("前进 0.5 m/s", (0.5, 0.0, 0.0, 0.74), 10.0),
  ("前进 1.0 m/s", (1.0, 0.0, 0.0, 0.74), 10.0),
  ("后退 0.3 m/s", (-0.3, 0.0, 0.0, 0.74), 10.0),
  ("侧移 0.3 m/s", (0.0, 0.3, 0.0, 0.74), 10.0),
  ("蹲行 0.3 m/s h=0.62", (0.3, 0.0, 0.0, 0.62), 10.0),
  ("原地蹲 h=0.62", (0.0, 0.0, 0.0, 0.62), 8.0),
  # 左右各一个：实测转向增益左右差异能到 2 倍，只量单边会把差的那侧漏掉。
  ("原地转 +1.0 rad/s", (0.0, 0.0, 1.0, 0.74), 10.0),
  ("原地转 -1.0 rad/s", (0.0, 0.0, -1.0, 0.74), 10.0),
)


def open_video(model, path):
  if not path:
    return None
  camera = mujoco.MjvCamera()
  camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
  camera.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
  camera.distance, camera.azimuth, camera.elevation = 3.2, 135.0, -12.0
  return {
    "camera": camera,
    "renderer": mujoco.Renderer(model, height=480, width=640),
    "images": [],
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("policy", help="导出的 policy.onnx")
  parser.add_argument("--video", help="渲染 mp4 到这个路径（需要 MUJOCO_GL=egl）")
  parser.add_argument(
    "--arms",
    choices=("limp", "hold"),
    default="limp",
    help="limp: 不给手臂加力矩，自然下垂（默认）；hold: 目标写 0，对应部署侧 passive_targets",
  )
  args = parser.parse_args()

  model = build_model()
  data = mujoco.MjData(model)
  policy = load_policy(args.policy)
  index = Index(model, policy)
  set_arm_mode(model, args.arms)
  video = open_video(model, args.video)

  hard_lo, hard_hi = index.joint_range[:, 0], index.joint_range[:, 1]
  ctrl_lo, ctrl_hi = index.ctrl_range[:, 0], index.ctrl_range[:, 1]

  arm_note = "（不加力矩，自然下垂）" if args.arms == "limp" else "（目标写 0）"
  print(f"策略: {args.policy}\n手臂: {args.arms}{arm_note}\n")
  print(
    f"{'场景':<22}{'结果':<6}{'速度跟随':>20}{'高度误差':>10}"
    f"{'摆动占比':>10}{'越行程':>8}{'越ctrl':>8}"
  )
  print("-" * 84)

  worst_joints: dict[str, float] = {}
  failed = False
  for label, command, seconds in SCENARIOS:
    reset(model, data, policy.joint_names, policy.default_pos)
    data.ctrl[index.arm_actuator] = 0.0  # limp 模式下增益已清零，这一行不起作用
    log, fell = rollout(
      model,
      data,
      policy.session,
      index,
      policy.action_default_pos,
      policy.action_scale,
      command,
      seconds,
      video,
    )
    settled = slice(len(log["target"]) // 4, None)  # 丢掉起步的前 1/4

    targets = log["target"]
    over_hard = (targets < hard_lo) | (targets > hard_hi)
    over_ctrl = ((targets < ctrl_lo) | (targets > ctrl_hi)).any(axis=1).mean()

    if abs(command[2]) > 0.1:
      tracking = f"{log['yaw_rate'][settled].mean():+.2f}/{command[2]:+.2f} rad/s"
    else:
      axis = 0 if abs(command[0]) >= abs(command[1]) else 1
      achieved, wanted = log["vel_b"][settled, axis].mean(), command[axis]
      tracking = (
        f"{achieved:+.2f}/{wanted:+.2f} m/s"
        if abs(wanted) > 1e-6
        else f"{achieved:+.2f} m/s (指令 0)"
      )

    print(
      f"{label:<22}{'摔倒' if fell else '站住':<6}{tracking:>20}"
      f"{np.abs(log['height'][settled] - command[3]).mean():9.3f}m"
      f"{log['swing'][settled].mean() * 100:9.0f}%"
      f"{over_hard.any(axis=1).mean() * 100:7.1f}%{over_ctrl * 100:7.1f}%"
    )
    failed |= fell or over_ctrl > 0.0
    for joint, rate in zip(policy.action_joint_names, over_hard.mean(axis=0)):
      worst_joints[joint] = max(worst_joints.get(joint, 0.0), rate)

  print("\n目标位置越出关节行程最多的关节（这正是不能按行程裁剪的原因）：")
  for joint, rate in sorted(worst_joints.items(), key=lambda item: -item[1])[:5]:
    if rate > 0.0:
      print(f"  {joint:<26}{rate * 100:5.1f}% 的拍")

  if video is not None:
    import imageio.v2 as imageio

    imageio.mimsave(args.video, video["images"], fps=25, macro_block_size=1)
    print(f"\n视频: {args.video}  ({len(video['images'])} 帧)")

  print(
    "\n"
    + (
      "有场景摔倒或目标越出 ctrlrange，不要上机"
      if failed
      else "全部场景站得住，且没有一拍越过 ctrlrange"
    )
  )
  return 1 if failed else 0


if __name__ == "__main__":
  raise SystemExit(main())
