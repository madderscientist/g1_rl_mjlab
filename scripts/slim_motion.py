"""把训练格式的动作 NPZ 瘦身成部署格式。

训练格式为每一帧存全部 44 个刚体的位姿与速度，而部署端只读其中几个：根刚体（参考系）、
锚刚体（对齐与姿态差）以及参考 token 里的 key body。其余刚体是训练时算奖励用的，
上机一个字节都不会被读到——单条 6 分钟的行走因此从 47 MB 降到 4 MB 左右。

瘦身后的字段::

    fps           (1,)
    joint_pos     (T, 31)
    root_pos      (T, 3)
    root_quat     (T, 4)   wxyz
    root_lin_vel  (T, 3)
    root_ang_vel  (T, 3)
    anchor_pos    (T, 3)
    anchor_quat   (T, 4)   wxyz
    key_pos       (T, K, 3)
    key_lin_vel   (T, K, 3)

``anchor_pos`` 和两个 ``key_*`` 是 RGMT 策略新增的：前者用于启动时把参考按偏航+平移
对齐到机器人，后两者就是参考窗口后 30 维的来源。旧的 GMT 部署包不读它们，兼容。

``MotionClip`` 两种格式都认，所以训练产物仍可直接丢进部署包，只是文件大很多。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tyro


def slim_one(src: Path, dst: Path, root_index: int, anchor_index: int,
             key_indexes: np.ndarray, start: int = 0,
             max_frames: int | None = None) -> tuple[int, float, float]:
    data = np.load(src)
    stop = None if max_frames is None else start + max_frames
    cut = slice(start, stop)
    joint_pos = np.asarray(data['joint_pos'], dtype=np.float32)[cut]
    pos_w = data['body_pos_w'][cut]
    quat_w = data['body_quat_w'][cut]
    lin_w = data['body_lin_vel_w'][cut]
    np.savez_compressed(
        dst,
        fps=np.asarray(data['fps'] if 'fps' in data.files else [50.0], dtype=np.float32),
        joint_pos=joint_pos,
        root_pos=np.asarray(pos_w[:, root_index], dtype=np.float32),
        root_quat=np.asarray(quat_w[:, root_index], dtype=np.float32),
        root_lin_vel=np.asarray(lin_w[:, root_index], dtype=np.float32),
        root_ang_vel=np.asarray(data['body_ang_vel_w'][cut][:, root_index], dtype=np.float32),
        anchor_pos=np.asarray(pos_w[:, anchor_index], dtype=np.float32),
        anchor_quat=np.asarray(quat_w[:, anchor_index], dtype=np.float32),
        key_pos=np.asarray(pos_w[:, key_indexes], dtype=np.float32),
        key_lin_vel=np.asarray(lin_w[:, key_indexes], dtype=np.float32),
    )
    return len(joint_pos), src.stat().st_size / 1e6, dst.stat().st_size / 1e6


def main(
    input_dir: str,
    output_dir: str,
    contract: str = 'export/policy_contract.json',
    root_body: str = 'pelvis',
    start_seconds: float = 0.0,
    max_seconds: float | None = None,
) -> None:
    """按 ONNX 契约里的刚体名单定位下标，逐个瘦身。

    ``start_seconds`` / ``max_seconds`` 摸取片段。首次上机建议用短片段：里程计漂移
    随时长累积，而参考窗口里那 3 维漂移量是直接喂给策略的。
    摩取中段时注意首帧最好是站立位形，STAND 阶段要从实测位形插值到它。
    """
    meta = json.loads(Path(contract).read_text(encoding='utf-8'))
    bodies = list(meta['all_body_names'])
    root_index = bodies.index(root_body)
    anchor_index = bodies.index(meta['anchor_body_name'])
    # 顺序必须跟契约走：部署端按下标取 key_pos，重排了就是静默错位。
    key_names = list(meta.get('reference_key_bodies', ()))
    key_indexes = np.array([bodies.index(n) for n in key_names], dtype=np.intp)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    window = '' if max_seconds is None else f'，取 {start_seconds:.0f}~{start_seconds + max_seconds:.0f}s'
    print(f'[瘦身] key body {key_names or "无"}{window}')
    total_src = total_dst = 0.0
    for src in sorted(Path(input_dir).glob('*.npz')):
        with np.load(src) as probe:
            fps = float(probe['fps'][0]) if 'fps' in probe.files else 50.0
        start = int(start_seconds * fps)
        limit = None if max_seconds is None else int(max_seconds * fps)
        frames, mb_src, mb_dst = slim_one(src, out / src.name, root_index, anchor_index,
                                          key_indexes, start, limit)
        total_src += mb_src
        total_dst += mb_dst
        print(f'  {src.stem:<26} {frames:>6} 帧  {mb_src:6.1f} -> {mb_dst:5.1f} MB')
    print(f'\n合计 {total_src:.0f} MB -> {total_dst:.0f} MB '
          f'(缩小 {total_src / max(total_dst, 1e-9):.0f} 倍)')
    print(f'[瘦身] 已写入 {out.resolve()}')


if __name__ == '__main__':
    tyro.cli(main)
