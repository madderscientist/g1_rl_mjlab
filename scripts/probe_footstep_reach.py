"""扫描 G1 模型的平足静态可达范围，不验证动态行走能力"""

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from g1_lower_rl import PKG_PATH


def load_kinematic_model(path: Path) -> mujoco.MjModel:
  """去掉可视网格依赖后加载模型，保留运动学、惯量和碰撞定义"""
  root = ET.parse(path).getroot()
  for parent in root.iter():
    for child in list(parent):
      if child.tag == "mesh" or (child.tag == "geom" and "mesh" in child.attrib):
        parent.remove(child)
  return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def scan(model: mujoco.MjModel, height: float, width: float, increment: float, margin: float) -> dict:
  """在指定骨盆高度和站距下递增前后间距，用有界逆运动学搜索可达候选"""
  data = mujoco.MjData(model)
  names = [
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
  ]
  joint_ids = [model.joint(name).id for name in names]
  addresses = model.jnt_qposadr[joint_ids]
  site_ids = [model.site(f"{side}_foot").id for side in ("left", "right")]
  lower = np.r_[-0.2, model.jnt_range[joint_ids, 0] + margin]
  upper = np.r_[0.2, model.jnt_range[joint_ids, 1] - margin]
  guess = np.r_[0.0, [-0.3, 0, 0, 0.6, -0.3, 0] * 2]
  guess = np.clip(guess, lower + 1e-6, upper - 1e-6)
  data.qpos[:7] = [0, 0, height, 1, 0, 0, 0]

  def residual(values, targets):
    """根据骨盆横移和双腿关节值，计算足端位置及平足朝向的加权残差"""
    data.qpos[0] = values[0]
    data.qpos[addresses] = values[1:]
    mujoco.mj_kinematics(model, data)
    orientation = Rotation.from_matrix(data.site_xmat[site_ids].reshape(2, 3, 3)).as_rotvec()
    return np.r_[(data.site_xpos[site_ids] - targets).ravel(), (0.2 * orientation).ravel()]

  accepted = None
  rejected = None
  for forward in np.arange(0, 0.80001, increment):
    targets = np.array([[forward / 2, width / 2, 0], [-forward / 2, -width / 2, 0]])
    result = least_squares(
      residual, guess, args=(targets,), bounds=(lower, upper), max_nfev=200, ftol=1e-10, xtol=1e-10, gtol=1e-10
    )
    errors = residual(result.x, targets)
    position_error = np.linalg.norm(errors[:6].reshape(2, 3), axis=1).max()
    angle_error = np.linalg.norm(errors[6:].reshape(2, 3), axis=1).max() / 0.2
    if position_error > 0.001 or angle_error > math.radians(0.5):
      rejected = {
        "forward_m": float(forward),
        "position_error_m": float(position_error),
        "angle_error_deg": math.degrees(angle_error),
      }
      break
    guess = result.x
    accepted = {
      "forward_m": float(forward),
      "foot_distance_m": math.hypot(forward, width),
      "pelvis_x_m": float(result.x[0]),
      "max_position_error_m": float(position_error),
      "max_orientation_error_deg": math.degrees(angle_error),
      "joint_angles_deg": dict(zip(names, np.degrees(result.x[1:]).tolist())),
      "minimum_joint_margin_deg": math.degrees(float(np.minimum(result.x[1:] - lower[1:], upper[1:] - result.x[1:]).min()))
      + math.degrees(margin),
    }
  return {"pelvis_height_m": height, "width_m": width, "last_feasible": accepted, "first_rejected": rejected}


def main() -> None:
  """解析扫描条件，输出模型腿长和各高度下的静态可达记录"""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--model", type=Path, default=PKG_PATH / "assets/g1_gloria/xmls/g1_with_dual_gloria_m.xml")
  parser.add_argument("--heights", type=float, nargs="+", default=[0.70, 0.74, 0.78])
  parser.add_argument("--width", type=float, default=0.22)
  parser.add_argument("--increment", type=float, default=0.02)
  parser.add_argument("--margin-deg", type=float, default=3.0)
  args = parser.parse_args()
  if not 0 < args.increment <= 0.1 or not 0 < args.width < 0.5 or not 0 <= args.margin_deg < 10:
    parser.error("Invalid scan step, width or joint margin")
  if not all(0.4 <= height <= 0.85 for height in args.heights):
    parser.error("Heights must be between 0.4 and 0.85 m")
  model = load_kinematic_model(args.model)
  data = mujoco.MjData(model)
  mujoco.mj_kinematics(model, data)
  roll = model.body("left_hip_roll_link").id
  knee = model.body("left_knee_link").id
  ankle = model.body("left_ankle_pitch_link").id
  geometry = {
    "hip_roll_to_knee_m": float(np.linalg.norm(data.xpos[knee] - data.xpos[roll])),
    "knee_to_ankle_pitch_m": float(np.linalg.norm(data.xpos[ankle] - data.xpos[knee])),
    "zero_foot_site_spacing_m": float(
      np.linalg.norm(data.site_xpos[model.site("left_foot").id] - data.site_xpos[model.site("right_foot").id])
    ),
    "ankle_roll_to_foot_site_m": model.site("left_foot").pos.tolist(),
  }
  result = {
    "model": str(args.model),
    "assumptions": "Upright pelvis; flat feet, yaw=0; fixed height/width; pelvis X optimized within +/-0.2m. Local IK continuation, not a global bound. No collision, balance, torque, speed or swing-path checks.",
    "joint_limit_margin_deg": args.margin_deg,
    "increment_m": args.increment,
    "geometry": geometry,
    "scans": [scan(model, height, args.width, args.increment, math.radians(args.margin_deg)) for height in args.heights],
  }
  print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
  main()
