"""标定过的 31 自由度 Unitree G1 + 双 Gloria-M 夹爪。

执行器/碰撞/关节体定义直接复用 mjlab 自带的 G1 资产常量，本文件只补三样东西：
自己的 MJCF、夹爪偏心关节的执行器、以及为长夹爪加宽的初始姿态。
"""

import re

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  ACTUATOR_4010,
  FULL_COLLISION,
  G1_ARTICULATION,
)
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from g1_lower_rl import PKG_PATH

XML = PKG_PATH / "assets" / "g1_gloria" / "xmls" / "g1_with_dual_gloria_m.xml"
assert XML.exists(), XML


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(XML))


GRIPPER_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=("left_eccentric_joint", "right_eccentric_joint"),
  stiffness=10.0,
  damping=0.5,
  effort_limit=10.0,
  armature=ACTUATOR_4010.reflected_inertia,
)

ARTICULATION = EntityArticulationInfoCfg(
  actuators=(*G1_ARTICULATION.actuators, GRIPPER_ACTUATOR),
  soft_joint_pos_limit_factor=0.9,
)

# Gloria-M 夹爪比原来的橡胶手多伸出约 130 mm，G1 出厂站姿（shoulder_roll ±0.18）
# 会把夹爪塞进大腿里。把肩外展到 ±0.25 才有间隙。
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.8),
  joint_pos={
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_shoulder_pitch_joint": 0.35,
    ".*_elbow_joint": 0.87,
    "left_shoulder_roll_joint": 0.25,
    "right_shoulder_roll_joint": -0.25,
    ".*_eccentric_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)


def get_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=ARTICULATION,
  )


##
# 下肢子集。关节顺序与真机电机编号 0..14 一致，导出的策略在 deploy/ 侧
# 用 `joint_ids_map: [0..14]` 即可，不需要重映射。
##

WAIST_JOINTS: tuple[str, ...] = (
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
)

LOWER_BODY_JOINTS: tuple[str, ...] = (
  tuple(
    f"{_side}_{_j}_joint"
    for _side in ("left", "right")
    for _j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
  )
  + WAIST_JOINTS
)

ARM_JOINTS: tuple[str, ...] = tuple(
  f"{_side}_{_j}_joint"
  for _side in ("left", "right")
  for _j in (
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
  )
)

_ACTION_SCALE: dict[str, float] = {}
_EFFORT_LIMIT: dict[str, float] = {}
for _a in ARTICULATION.actuators:
  assert isinstance(_a, BuiltinPositionActuatorCfg)
  assert _a.effort_limit is not None
  for _expr in _a.target_names_expr:
    _ACTION_SCALE[_expr] = 0.25 * _a.effort_limit / _a.stiffness
    _EFFORT_LIMIT[_expr] = _a.effort_limit

# 只保留能解析到下肢关节的执行器模式。把完整字典交给 15 关节的动作项会让
# resolve_matching_names_values 报错——手臂/夹爪的模式一个也匹配不上。
LOWER_BODY_ACTION_SCALE: dict[str, float] = {
  _expr: _scale
  for _expr, _scale in _ACTION_SCALE.items()
  if any(re.fullmatch(_expr, _name) for _name in LOWER_BODY_JOINTS)
}

LOWER_BODY_ACTUATOR_EXPR: tuple[str, ...] = tuple(LOWER_BODY_ACTION_SCALE)

# 各下肢关节的力矩上限，按执行器模式索引。站立任务用它把力矩归一化成「占额定的几成」。
LOWER_BODY_EFFORT_LIMIT: dict[str, float] = {
  _expr: _limit
  for _expr, _limit in _EFFORT_LIMIT.items()
  if any(re.fullmatch(_expr, _name) for _name in LOWER_BODY_JOINTS)
}

# 全身 29 个 G1 关节（下肢 15 + 手臂 14），顺序与真机电机编号 0..28 一致。
# 动作跟踪任务用它；两个 Gloria-M 夹爪关节由真机上独立的控制器驱动，不属于策略动作空间。
WHOLE_BODY_JOINTS: tuple[str, ...] = LOWER_BODY_JOINTS + ARM_JOINTS

WHOLE_BODY_ACTION_SCALE: dict[str, float] = {
  _expr: _scale
  for _expr, _scale in _ACTION_SCALE.items()
  if any(re.fullmatch(_expr, _name) for _name in WHOLE_BODY_JOINTS)
}

WHOLE_BODY_ACTUATOR_EXPR: tuple[str, ...] = tuple(WHOLE_BODY_ACTION_SCALE)

# 上肢扰动事件写进去的手臂 PD 目标的采样范围。
#
# 这些是**目标**，而手臂执行器很软（kp=14.3），重力会把实际位姿往下拽 30-40 度，
# 挂上 0-2 kg 夹爪负载后更多。解静力平衡 kp*(ctrl - q) = qfrc_bias(q)，以 shoulder_pitch 为例：
#
#   目标 -1.6 rad (-92°)  ->  空载实际 -49°，带 2 kg 时 -30°
#   目标 -0.9 rad (-52°)  ->  空载实际 -21°，带 2 kg 时 -11°
#   目标 -0.3 rad (-17°)  ->  空载实际  -0°，带 2 kg 时  +6°
#
# 所以旧的 -0.3 下界根本没演示过“往前伸”：实际位姿几乎没离开自然下垂，而整个前半区间
# ——手臂实际使用时待的地方、质心前移约 5 cm 的地方——策略完全没见过。负 pitch 是向前。
#
# 两端一起放宽反而**降低**自碰撞率：易碰撞区在中段，长夹爪正好扫过髋部，两个极端都能避开。
# 按实际位姿统计，600 次采样：
#
#   (-0.3, 1.2)  空载 16.5% / 带 2 kg 31.0%
#   (-0.9, 1.2)  12.7% / 26.5%
#   (-1.6, 1.8)   7.8% / 17.5%   <- 当前
#
# 其它关节同理，但下垂量差别很大，取决于关节轴相对重力的方向。空载时实际张角占指令张角的比例：
#
#   shoulder_pitch 60%   shoulder_roll 59%   elbow 82%
#   wrist_pitch    93%   wrist_yaw     95%
#   shoulder_yaw  100%   wrist_roll   100%   <- 轴接近竖直，无需补偿
#
# shoulder_roll 在上界损失最大（空载 29°，带 2 kg 39°），所以上界推到 1.9 rad；下界保持
# 0.25，因为重力本来就把实际值拽到它之下，再往里拉只会换来手臂贴躯干。
#
# 左右保持镜像对称：矢状面镜像下 roll/yaw 变号而 pitch 不变，所以只有 shoulder_roll 需要
# 下面这样按侧拆开写，其余要么是 pitch 类要么本身关于零对称。
ARM_TARGET_RANGES: dict[str, tuple[float, float]] = {
  ".*_shoulder_pitch_joint": (-1.6, 1.8),
  "left_shoulder_roll_joint": (0.25, 1.90),
  "right_shoulder_roll_joint": (-1.90, -0.25),
  ".*_shoulder_yaw_joint": (-1.0, 1.0),
  ".*_elbow_joint": (0.1, 1.7),
  ".*_wrist_roll_joint": (-1.0, 1.0),
  ".*_wrist_pitch_joint": (-0.9, 0.9),
  ".*_wrist_yaw_joint": (-0.9, 0.9),
}
