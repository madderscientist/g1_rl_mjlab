"""导入已重定向到 G1 的极限动作（空翻/侧手翻类），转成 build_corpus 吃的 CSV。

来源 `elijahgalahad/any4hdmi-g1-extreme-motions`：29 关节名与本仓 `JOINT_NAMES`
逐一对齐、fps 50、qpos = [root_pos(3), root_quat_wxyz(4), dof_pos(29)]。

我们已有的 AMASS 通道里**没有翻跟头**——17714 个文件中 flip/cartwheel/somersault
全为 0 条，所以这批是唯一的极限动作来源。

⚠️ 四元数：本仓 CSV 约定是 **xyzw**，而该数据集 manifest 写明已做过 `xyzw -> wxyz`
转换，所以这里要转回去。判据见仓库记忆第 12 条：站立帧按正确约定解释时，
根的局部 +z 应指向世界 +z。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tyro

REPO = "elijahgalahad/any4hdmi-g1-extreme-motions"
TAKES = (143, 144, 146, 148, 149)


def main(output_dir: str = "motions/csv_extreme", check_only: bool = False) -> None:
  from huggingface_hub import hf_hub_download

  out = Path(output_dir)
  out.mkdir(parents=True, exist_ok=True)

  for t in TAKES:
    f = hf_hub_download(REPO, f"motions/Take_{t}_Skeleton0.npz", repo_type="dataset")
    q = np.asarray(np.load(f)["qpos"], dtype=np.float64)
    if q.shape[1] != 36:
      raise ValueError(f"Take_{t} 期望 36 列，实得 {q.shape[1]}")

    root_pos, quat_wxyz, dof = q[:, :3], q[:, 3:7], q[:, 7:]
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]

    # 站立帧自检：根局部 +z 在世界系的分量应接近 +1，否则约定弄反了
    x, y = quat_wxyz[:, 1], quat_wxyz[:, 2]
    up_z = 1 - 2 * (x * x + y * y)
    upright = up_z[up_z > 0.9]
    if len(upright) == 0:
      raise ValueError(f"Take_{t} 找不到直立帧，四元数约定可能不对")

    inv_ratio = (up_z < 0).mean() * 100
    print(
      f"  Take_{t}: {len(q)} 帧 / {len(q) / 50:.1f}s  "
      f"倒置 {inv_ratio:.1f}%  根高 [{root_pos[:, 2].min():.2f}, {root_pos[:, 2].max():.2f}]"
    )
    if check_only:
      continue

    np.savetxt(
      out / f"extreme_take{t}.csv",
      np.hstack([root_pos, quat_xyzw, dof]),
      delimiter=",",
      fmt="%.6f",
    )

  if not check_only:
    print(f"\n已写入 {out}/  （{len(TAKES)} 条）")
    print("下一步：")
    print(
      f"  python scripts/build_corpus.py --input-dir {out} "
      f"--output-dir motions/extreme --input-fps 50"
    )


if __name__ == "__main__":
  tyro.cli(main)
