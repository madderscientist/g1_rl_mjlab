"""把训练格式的动作 NPZ 瘦身成部署格式。

训练格式为每一帧存全部 44 个刚体的位姿与速度，而部署端只读两个：根刚体（前瞻特征的
参考系）和锚刚体（``motion_anchor_ori_b``）。其余 42 个刚体是训练时算奖励用的，
上机一个字节都不会被读到——单条 6 分钟的行走因此从 47 MB 降到 4 MB 左右。

瘦身后的字段::

    fps           (1,)
    joint_pos     (T, 31)
    root_pos      (T, 3)
    root_quat     (T, 4)   wxyz
    root_lin_vel  (T, 3)
    root_ang_vel  (T, 3)
    anchor_quat   (T, 4)   wxyz

``MotionClip`` 两种格式都认，所以训练产物仍可直接丢进部署包，只是文件大很多。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tyro


def slim_one(src: Path, dst: Path, root_index: int, anchor_index: int) -> tuple[int, float, float]:
    data = np.load(src)
    joint_pos = np.asarray(data['joint_pos'], dtype=np.float32)
    np.savez_compressed(
        dst,
        fps=np.asarray(data['fps'] if 'fps' in data.files else [50.0], dtype=np.float32),
        joint_pos=joint_pos,
        root_pos=np.asarray(data['body_pos_w'][:, root_index], dtype=np.float32),
        root_quat=np.asarray(data['body_quat_w'][:, root_index], dtype=np.float32),
        root_lin_vel=np.asarray(data['body_lin_vel_w'][:, root_index], dtype=np.float32),
        root_ang_vel=np.asarray(data['body_ang_vel_w'][:, root_index], dtype=np.float32),
        anchor_quat=np.asarray(data['body_quat_w'][:, anchor_index], dtype=np.float32),
    )
    return len(joint_pos), src.stat().st_size / 1e6, dst.stat().st_size / 1e6


def main(
    input_dir: str,
    output_dir: str,
    contract: str = 'export/policy_contract.json',
    root_body: str = 'pelvis',
) -> None:
    """按 ONNX 契约里的刚体名单定位下标，逐个瘦身。"""
    meta = json.loads(Path(contract).read_text(encoding='utf-8'))
    bodies = list(meta['all_body_names'])
    root_index = bodies.index(root_body)
    anchor_index = bodies.index(meta['anchor_body_name'])

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    total_src = total_dst = 0.0
    for src in sorted(Path(input_dir).glob('*.npz')):
        frames, mb_src, mb_dst = slim_one(src, out / src.name, root_index, anchor_index)
        total_src += mb_src
        total_dst += mb_dst
        print(f'  {src.stem:<26} {frames:>6} 帧  {mb_src:6.1f} -> {mb_dst:5.1f} MB')
    print(f'\n合计 {total_src:.0f} MB -> {total_dst:.0f} MB '
          f'(缩小 {total_src / max(total_dst, 1e-9):.0f} 倍)')
    print(f'[瘦身] 已写入 {out.resolve()}')


if __name__ == '__main__':
    tyro.cli(main)
