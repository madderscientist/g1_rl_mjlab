"""把 AMASS 的 G1 重定向版拉下来，转成 ``build_corpus.py`` 吃的 CSV。

数据源 https://huggingface.co/datasets/ember-lab-berkeley/AMASS_Retargeted_for_G1
（CC-BY-4.0，免授权），17714 段、100% 的 AMASS 重定向到 29 轴 G1。它的 ``dof_names``
顺序与 ``build_corpus.JOINT_NAMES`` **完全一致**，``fps`` 也同为 30，所以转换只是取根位姿
+ 关节角、把四元数从 wxyz 换成 CSV 用的 xyzw。（wxyz 是实测定的：站立片段里按 wxyz 解释
时根的局部 +z 指向世界 [0,0,0.999]，按 xyzw 则是侧躺。）

**做了物理可行性筛选。** AMASS 里有大量这台机器执行不了的片段（重定向漂浮、脚够不到地、
速度尖峰），SONIC 也把 700 h 筛到 611 h 才训。不筛的话它们会永远占着采样预算却学不会。

用法::

    python scripts/import_amass.py --max-hours 6
    python scripts/build_corpus.py --input-dir motions/csv_amass \\
        --output-dir motions/lafan1 --input-fps 30
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import numpy as np
import tyro

from build_corpus import JOINT_NAMES

REPO = "ember-lab-berkeley/AMASS_Retargeted_for_G1"
API = f"https://huggingface.co/api/datasets/{REPO}?full=true"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/"


@dataclass
class Filters:
  """物理可行性判据。全部按**这台机器**能不能做来定，不是按动作好不好看。"""

  min_seconds: float = 2.0
  """太短的片段学不到东西，还会让 bin 级采样切出一堆退化 bin。"""
  max_seconds: float = 60.0
  min_foot_clearance: float = 0.12
  """整段里最低的那只脚必须至少一次接近地面；否则是重定向漂浮或上下楼梯。"""
  pelvis_height_range: tuple[float, float] = (0.15, 1.05)
  """中位骨盆高度。上界排掉漂浮，下界保留趴地/爬行但排掉穿地。"""
  max_joint_speed: float = 30.0
  """rad/s，按**位置有限差分**算。

  不能用数据自带的 ``dof_velocities``：``build_corpus.py`` 重采样到 50 Hz 后会自己重算
  速度，两者对不上。AMASS→G1 重定向里的欧拉角万向节翻转会造出单帧 π 级跳变（实测源
  CSV 单帧最大 5.24 rad），而 ``dof_velocities`` 里看不出来——漏进来的片段会在复位瞬间
  把物理炸成 NaN。``np.unwrap`` 修不了，因为跳变不是 2π 的整数倍。"""

  require_dynamic: bool = False
  """只收高动态片段。按子库均匀采样时 AMASS 里 33% 是近乎静止、跑/冲刺只占 0.2%，
  而失败恰恰集中在 sprint/run/jumps/fight——均匀扩充只会把高动态占比越稀释。"""
  dyn_speed_p95: float = 1.5
  """m/s，水平速度。抓快走/跑/冲刺。"""
  dyn_root_z_range: float = 0.35
  """m，根高度摆幅。抓跳跃/蹲起/摔倒起身——这些水平速度不高但竖向很狠。"""
  dyn_joint_speed_p95: float = 4.0
  """rad/s。抓格斗/舞蹈——原地不动但四肢快。"""


def list_clips() -> list[str]:
  meta = json.load(urllib.request.urlopen(API, timeout=60))
  return [
    s["rfilename"]
    for s in meta["siblings"]
    if s["rfilename"].startswith("g1/") and s["rfilename"].endswith(".npz")
  ]


def interleave_by_subset(paths: list[str], seed: int) -> list[str]:
  """按 AMASS 子库轮转排序，保证预算用完时各子库都取到，而不是全给了字母序靠前的。"""
  groups: dict[str, list[str]] = defaultdict(list)
  for p in paths:
    groups[p.split("/")[1]].append(p)
  rng = np.random.default_rng(seed)
  for g in groups.values():
    rng.shuffle(g)
  order = sorted(groups)
  out: list[str] = []
  for i in range(max(len(g) for g in groups.values())):
    for k in order:
      if i < len(groups[k]):
        out.append(groups[k][i])
  return out


def check(z, f: Filters) -> str | None:
  """返回剔除原因；None 表示通过。"""
  dof, bp = z["dof_positions"], z["body_positions"]
  fps = float(z["fps"][0])
  if len(dof) / fps < f.min_seconds:
    return "太短"
  if len(dof) / fps > f.max_seconds:
    return "太长"
  if not (np.isfinite(dof).all() and np.isfinite(bp).all()):
    return "含 NaN/Inf"
  names = [str(n) for n in z["body_names"]]
  feet = [names.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
  if bp[:, feet, 2].min() > f.min_foot_clearance:
    return "脚够不到地"
  pelvis = float(np.median(bp[:, names.index("pelvis"), 2]))
  lo, hi = f.pelvis_height_range
  if not lo <= pelvis <= hi:
    return f"骨盆高度异常({pelvis:.2f}m)"
  if np.abs(np.diff(dof, axis=0)).max() * fps > f.max_joint_speed:
    return "关节速度尖峰"

  if f.require_dynamic:
    root = bp[:, names.index("pelvis")]
    speed = np.linalg.norm(np.diff(root[:, :2], axis=0) * fps, axis=1)
    # 三个门槛取**或**：快走/跳跃/格斗是三种不同的高动态，各有各的特征量。
    if not (
      float(np.quantile(speed, 0.95)) >= f.dyn_speed_p95
      or float(root[:, 2].max() - root[:, 2].min()) >= f.dyn_root_z_range
      or float(np.quantile(np.abs(z["dof_velocities"]), 0.95)) >= f.dyn_joint_speed_p95
    ):
      return "动态性不足"
  return None


def to_csv(z, dof_perm: np.ndarray) -> np.ndarray:
  root_pos = z["body_positions"][:, 0]
  quat_wxyz = z["body_rotations"][:, 0]
  root_quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
  return np.concatenate(
    [root_pos, root_quat_xyzw, z["dof_positions"][:, dof_perm]], axis=1
  ).astype(np.float32)


def main(
  output_dir: str = "motions/csv_amass",
  max_hours: float = 6.0,
  seed: int = 0,
  workers: int = 16,
  filters: Filters = Filters(),  # noqa: B008
) -> None:
  """下载并转换 AMASS-G1，直到累计时长达到 ``max_hours``。

  ``max_hours`` 是**源时长**。语料会被重采样到 50 Hz 并做左右镜像，所以显存里的帧数是
  ``max_hours * 3600 * 50 * 2``；按每帧约 960 B 估，6 h 约 2.1 GB。
  """
  out = Path(output_dir)
  out.mkdir(parents=True, exist_ok=True)

  paths = interleave_by_subset(list_clips(), seed)
  print(f"[AMASS] 源仓库 {len(paths)} 段，目标 {max_hours} h")

  budget = max_hours * 3600.0
  lock = Lock()
  state = {"sec": 0.0, "kept": 0}
  dropped: dict[str, int] = defaultdict(int)
  perm: list[np.ndarray] = []

  def work(path: str) -> bool:
    """返回是否还要继续。"""
    with lock:
      if state["sec"] >= budget:
        return False
    name = "amass_" + path[len("g1/") : -len("_jpos.npz")].replace("/", "_")
    name = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
    # 增量补充：已导入过的直接跳过，预算只花在新片段上。
    if (out / f"{name}.csv").exists():
      with lock:
        dropped["已存在"] += 1
      return True
    try:
      raw = urllib.request.urlopen(BASE + urllib.parse.quote(path), timeout=120).read()
      z = np.load(__import__("io").BytesIO(raw))
    except Exception as e:  # noqa: BLE001  网络抖动跳过即可，不值得中断整批
      with lock:
        dropped[f"下载失败({type(e).__name__})"] += 1
      return True

    why = check(z, filters)
    if why is not None:
      with lock:
        dropped[why] += 1
      return True

    if not perm:
      src = [str(n) for n in z["dof_names"]]
      perm.append(np.array([src.index(n) for n in JOINT_NAMES]))

    dur = len(z["dof_positions"]) / float(z["fps"][0])
    with lock:
      if state["sec"] >= budget:
        return False
      state["sec"] += dur
      state["kept"] += 1
      n = state["kept"]
    np.savetxt(out / f"{name}.csv", to_csv(z, perm[0]), delimiter=",", fmt="%.6f")
    if n % 100 == 0:
      print(f"  已保留 {n} 段 / {state['sec'] / 3600:.2f} h", flush=True)
    return True

  # 分块提交：``ThreadPoolExecutor.map`` 会把整个列表一次性排队，预算用完也停不下来，
  # 结果是把 17714 段全下完。
  chunk = max(workers * 4, 32)
  with ThreadPoolExecutor(max_workers=workers) as ex:
    for i in range(0, len(paths), chunk):
      if not all(ex.map(work, paths[i : i + chunk])):
        break

  print(f"\n[AMASS] 保留 {state['kept']} 段，共 {state['sec'] / 3600:.2f} h -> {out}")
  print("剔除统计:")
  for k, v in sorted(dropped.items(), key=lambda x: -x[1]):
    print(f"  {k:<28}{v:>6}")


if __name__ == "__main__":
  tyro.cli(main)
