"""把一段参考动作 NPZ 回放成视频，用来肉眼检查动作本身对不对。

只做运动学回放：直接把 NPZ 里的位姿写进 qpos 后 ``mj_forward``，不跑物理。
所以看到的是「参考轨迹长什么样」，不是「机器人能不能跟住」——后者要用
``scripts/eval_corpus.py``。
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import tyro


def main(
  motion: str,
  output: str = 'motion.mp4',
  width: int = 960,
  height: int = 720,
  camera_distance: float = 3.0,
  elevation: float = -10.0,
  azimuth: float = 135.0,
) -> None:
  from g1_lower_rl.assets.g1_gloria import get_spec

  data_npz = np.load(Path(motion))
  joint_pos = data_npz['joint_pos']
  root_pos = data_npz['body_pos_w'][:, 0]
  root_quat = data_npz['body_quat_w'][:, 0]  # wxyz，与 MuJoCo qpos 一致
  fps = float(data_npz['fps'][0]) if 'fps' in data_npz.files else 50.0
  n_frames, n_joints = joint_pos.shape

  spec = get_spec()
  # 离屏帧缓冲是编译期尺寸，默认 640x480，不调大渲不出更高分辨率。
  spec.visual.global_.offwidth = width
  spec.visual.global_.offheight = height
  model = spec.compile()
  data = mujoco.MjData(model)
  if model.nq != 7 + n_joints:
    raise ValueError(f'模型 nq={model.nq} 与动作的 7+{n_joints} 对不上')

  renderer = mujoco.Renderer(model, height=height, width=width)
  camera = mujoco.MjvCamera()
  camera.distance = camera_distance
  camera.elevation = elevation
  camera.azimuth = azimuth

  out = Path(output)
  out.parent.mkdir(parents=True, exist_ok=True)
  writer = imageio.get_writer(out, fps=fps)
  for i in range(n_frames):
    data.qpos[0:3] = root_pos[i]
    data.qpos[3:7] = root_quat[i]
    data.qpos[7:] = joint_pos[i]
    mujoco.mj_forward(model, data)
    camera.lookat[:] = root_pos[i]  # 跟拍根节点，否则走动的动作会出画
    renderer.update_scene(data, camera)
    writer.append_data(renderer.render())
  writer.close()

  print(f'[渲染] {out.resolve()}')
  print(f'  {n_frames} 帧 @ {fps:.0f} fps = {n_frames / fps:.1f} 秒')
  print(f'  盆骨高度 {root_pos[:, 2].min():.3f} ~ {root_pos[:, 2].max():.3f} m')


if __name__ == '__main__':
  tyro.cli(main)
