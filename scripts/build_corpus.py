"""构建动作跟踪的训练语料。

输入是 Unitree 重定向格式的 CSV（每行 ``[root_pos(3), root_quat_xyzw(4), dof(29)]``），
输出是 mjlab tracking 用的 NPZ（含各刚体的位姿与速度）。

只用**真实动捕**。早期版本曾用程序化生成的站立/下蹲/摆臂来凑语料，那些动作虽然跟得准，
但看上去不自然，也教不会策略真实人体运动的动力学，已删。

增强只保留**左右镜像**：环境本身左右对称，镜像后仍是一段物理上合法的真实动作，等于免费翻倍。
变速由 ``--input-fps`` 控制（CSV 原生 30fps，填 20 就是放慢 1.5 倍）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import tyro

# 与 CSV 列顺序一致的 29 个 G1 关节。
JOINT_NAMES: list[str] = [
  f"{side}_{j}_joint"
  for side in ("left", "right")
  for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
] + [
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
] + [
  f"{side}_{j}_joint"
  for side in ("left", "right")
  for j in (
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
  )
]

# 矢状面镜像下需要变号的关节（绕 x / z 轴的自由度）。绕 y 轴的 pitch 类保持原样。
_FLIP_SIGN_KEYS = ("_roll", "_yaw")


def _mirror_index_and_sign() -> tuple[np.ndarray, np.ndarray]:
  idx, sign = [], []
  for name in JOINT_NAMES:
    if name.startswith("left_"):
      partner = "right_" + name[len("left_") :]
    elif name.startswith("right_"):
      partner = "left_" + name[len("right_") :]
    else:
      partner = name  # waist_*，居中关节没有对侧
    idx.append(JOINT_NAMES.index(partner))
    stem = name.replace("_joint", "")
    sign.append(-1.0 if any(stem.endswith(k) for k in _FLIP_SIGN_KEYS) else 1.0)
  return np.asarray(idx), np.asarray(sign, dtype=np.float32)


MIRROR_IDX, MIRROR_SIGN = _mirror_index_and_sign()


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """读 CSV，返回 (root_pos, root_quat_wxyz, dof)。"""
  raw = np.loadtxt(path, delimiter=",", dtype=np.float32)
  root_pos = raw[:, :3]
  root_quat = raw[:, 3:7][:, [3, 0, 1, 2]]  # xyzw -> wxyz
  dof = raw[:, 7:]
  if dof.shape[1] != len(JOINT_NAMES):
    raise ValueError(f"{path.name}: 期望 {len(JOINT_NAMES)} 个关节列，实际 {dof.shape[1]}")
  return root_pos, root_quat, dof


def mirror(
  root_pos: np.ndarray, root_quat: np.ndarray, dof: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """关于矢状面（XZ 平面）做左右镜像。"""
  pos = root_pos.copy()
  pos[:, 1] *= -1.0
  quat = root_quat.copy()
  quat[:, 1] *= -1.0  # x
  quat[:, 3] *= -1.0  # z
  return pos, quat, dof[:, MIRROR_IDX] * MIRROR_SIGN


def time_scale(
  root_pos: np.ndarray, root_quat: np.ndarray, dof: np.ndarray, factor: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """把整段动作按 ``factor`` 重采样（>1 变慢）。四元数用最近邻避免插值出非单位四元数。"""
  n_out = max(int(round(len(dof) * factor)), 2)
  src = np.linspace(0, len(dof) - 1, n_out)
  lo, hi = np.floor(src).astype(int), np.ceil(src).astype(int)
  w = (src - lo)[:, None]
  return (
    root_pos[lo] * (1 - w) + root_pos[hi] * w,
    root_quat[np.round(src).astype(int)],
    dof[lo] * (1 - w) + dof[hi] * w,
  )


def _velocities(x: np.ndarray, fps: float) -> np.ndarray:
  v = np.zeros_like(x)
  v[:-1] = (x[1:] - x[:-1]) * fps
  v[-1] = v[-2] if len(v) > 1 else 0.0
  return v


def _angular_velocities(quat: np.ndarray, fps: float) -> np.ndarray:
  """由相邻帧四元数之差求世界系角速度，quat 形状 (T, B, 4)，wxyz。"""
  q0, q1 = quat[:-1], quat[1:]
  w0, x0, y0, z0 = q0[..., 0], -q0[..., 1], -q0[..., 2], -q0[..., 3]  # 共轭
  w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
  w = w1 * w0 - x1 * x0 - y1 * y0 - z1 * z0
  x = w1 * x0 + x1 * w0 + y1 * z0 - z1 * y0
  y = w1 * y0 - x1 * z0 + y1 * w0 + z1 * x0
  z = w1 * z0 + x1 * y0 - y1 * x0 + z1 * w0
  # 取 w>=0 的那一半球，否则同一个旋转会解出 2π-θ 的巨大角速度。
  sign = np.where(w < 0.0, -1.0, 1.0)
  w, x, y, z = w * sign, x * sign, y * sign, z * sign
  norm = np.sqrt(x * x + y * y + z * z)
  angle = 2.0 * np.arctan2(norm, np.clip(w, -1.0, 1.0))
  scale = np.where(norm > 1e-8, angle / np.maximum(norm, 1e-8), 0.0)
  out = np.zeros(quat.shape[:-1] + (3,), dtype=np.float32)
  out[:-1] = np.stack([x * scale, y * scale, z * scale], axis=-1) * fps
  if len(out) > 1:
    out[-1] = out[-2]
  return out


def main(
  input_dir: str = "motions/csv",
  output_dir: str = "motions/g1_gloria",
  fps: float = 50.0,
  input_fps: float = 30.0,
  augment: bool = True,
  device: str = "cuda:0",
) -> None:
  """把 CSV 动作转成 NPZ 语料。"""
  from mjlab.scene import Scene
  from mjlab.sim.sim import Simulation, SimulationCfg

  from g1_lower_rl.tasks.motion_tracking import motion_tracking_env_cfg

  out = Path(output_dir)
  out.mkdir(parents=True, exist_ok=True)

  # 收集所有待转换的轨迹。
  clips: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
  for csv in sorted(Path(input_dir).glob("*.csv")):
    rp, rq, dof = load_csv(csv)
    # 先统一到目标帧率。整体快慢由 --input-fps 控制，不在这里再叠一层变速。
    rp, rq, dof = time_scale(rp, rq, dof, fps / input_fps)
    clips[csv.stem] = (rp, rq, dof)
    if augment:
      clips[f"{csv.stem}_mir"] = mirror(rp, rq, dof)

  print(f"[语料] 待转换 {len(clips)} 段")

  # 用与训练完全相同的场景做正运动学，保证刚体顺序一致。
  cfg = motion_tracking_env_cfg()
  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / fps
  # fallAndGetUp 这类动作有整段躺在地上的帧，接触点数远超训练配置里按站立姿态定的
  # nconmax=35。溢出在 warp 里不是报错而是**越界写**，表现为 CUDA illegal memory
  # access，且是异步报出来的，栈指向的往往是无辜的下一行。只做正运动学，放宽不费什么。
  sim_cfg.nconmax = 2000
  sim_cfg.njmax = 4000
  scene = Scene(cfg.scene, device=device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  robot = scene["robot"]
  joint_idx = robot.find_joints(JOINT_NAMES, preserve_order=True)[0]
  scene.reset()

  for name, (rp, rq, dof) in sorted(clips.items()):
    # 断点续传：一段几分钟的动作要转半分钟，中途挂了不该从头再来。
    if (out / f"{name}.npz").exists():
      print(f"  [跳过] {name}（已存在）", flush=True)
      continue
    rp, rq, dof = (np.ascontiguousarray(a, dtype=np.float32) for a in (rp, rq, dof))
    n = len(dof)
    rp_t = torch.tensor(rp, device=device, dtype=torch.float32)
    rq_t = torch.tensor(rq, device=device, dtype=torch.float32)
    dof_t = torch.tensor(dof, device=device, dtype=torch.float32)
    lin = torch.tensor(_velocities(rp, fps), device=device, dtype=torch.float32)
    dof_vel = torch.tensor(_velocities(dof, fps), device=device, dtype=torch.float32)

    log = {k: [] for k in ("body_pos_w", "body_quat_w", "joint_pos", "joint_vel")}
    for i in range(n):
      root = robot.data.default_root_state.clone()
      root[:, 0:3] = rp_t[i]
      root[:, 3:7] = rq_t[i]
      root[:, 7:10] = lin[i]
      root[:, 10:] = 0.0
      robot.write_root_state_to_sim(root)

      qpos = robot.data.default_joint_pos.clone()
      qvel = robot.data.default_joint_vel.clone()
      qpos[:, joint_idx] = dof_t[i]
      qvel[:, joint_idx] = dof_vel[i]
      robot.write_joint_state_to_sim(qpos, qvel)

      sim.forward()
      scene.update(1.0 / fps)
      log["body_pos_w"].append(robot.data.body_link_pos_w[0].cpu().numpy().copy())
      log["body_quat_w"].append(robot.data.body_link_quat_w[0].cpu().numpy().copy())
      log["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
      log["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())

    body_pos = np.stack(log["body_pos_w"])
    body_quat = np.stack(log["body_quat_w"])
    # 刚体速度由位姿差分得到，避免逐帧写入速度状态带来的数值噪声。
    body_lin = _velocities(body_pos, fps)
    body_ang = _angular_velocities(body_quat, fps)
    np.savez(
      out / f"{name}.npz",
      fps=np.array([fps]),
      joint_pos=np.stack(log["joint_pos"]),
      joint_vel=np.stack(log["joint_vel"]),
      body_pos_w=body_pos,
      body_quat_w=body_quat,
      body_lin_vel_w=body_lin,
      body_ang_vel_w=body_ang,
    )
    print(f"  [完成] {name}: {n} 帧 / {n / fps:.1f}s")

  print(f"[语料] 已写入 {out.resolve()}")


if __name__ == "__main__":
  tyro.cli(main)
