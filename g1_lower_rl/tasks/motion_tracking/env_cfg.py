"""全身动作跟踪（GMT）的环境配置。

在 mjlab 自带的**单动作** tracking 配置之上改两处，这两处正是「换一段没训过的动作还能
跟」的来源：

* 命令项换成 :class:`GeneralMotionCommandCfg`——多动作语料 + 失败率自适应采样 +
  偏航/平移不变的前瞻观测。
* actor 的本体观测接上历史窗口。参考轨迹只说「该往哪走」，而接触与打滑这些信息只存在于
  最近几帧的本体量里；没有历史，策略在动作切换处会反复踩空。

奖励、终止、事件全部沿用上游取值，没有改动。
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.tracking_env_cfg import VELOCITY_RANGE, make_tracking_env_cfg

from g1_lower_rl.assets import (
  WHOLE_BODY_ACTION_SCALE,
  WHOLE_BODY_ACTUATOR_EXPR,
  WHOLE_BODY_JOINTS,
  get_robot_cfg,
)
from g1_lower_rl.tasks.motion_tracking.mdp import GeneralMotionCommandCfg

DEFAULT_MOTION_DIR = "motions/lafan1"
"""参考动作语料目录。目录下所有 NPZ 一起参与训练。"""

PROPRIO_HISTORY = 5
"""actor 本体观测保留的历史帧数（含当前帧）。"""

# 参与跟踪的刚体。**第一个必须是根节点**：前瞻观测按 body_names[0] 取参考根位姿。
TRACKED_BODIES: tuple[str, ...] = (
  "pelvis",
  "left_hip_roll_link",
  "left_knee_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_knee_link",
  "right_ankle_roll_link",
  "torso_link",
  "left_shoulder_roll_link",
  "left_elbow_link",
  "left_wrist_yaw_link",
  "right_shoulder_roll_link",
  "right_elbow_link",
  "right_wrist_yaw_link",
)

# 判「跟丢」的末端（只比高度，阈值 0.25m）。手腕在列是有代价的：Gloria-M 夹爪让手臂
# 惯量增加 70%，而手臂电机仍是原厂 5020，快速臂部动作会先在这里被判死。
END_EFFECTORS: tuple[str, ...] = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
)


def _self_collision_sensor() -> ContactSensorCfg:
  return ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )


def motion_tracking_env_cfg(
  motion_dir: str = DEFAULT_MOTION_DIR,
  has_state_estimation: bool = False,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """构造全身动作跟踪配置（平地）。

  ``has_state_estimation`` 默认关闭：真机上根节点的水平速度没有可靠来源，训练时喂进去
  会造出一个部署时补不上的观测。
  """
  cfg = make_tracking_env_cfg()

  commands: dict[str, CommandTermCfg] = {
    "motion": GeneralMotionCommandCfg(
      entity_name="robot",
      resampling_time_range=(1.0e9, 1.0e9),  # 只在动作放完或复位时重采样
      debug_vis=False,
      pose_range={
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (-0.01, 0.01),
        "roll": (-0.1, 0.1),
        "pitch": (-0.1, 0.1),
        "yaw": (-0.2, 0.2),
      },
      velocity_range=VELOCITY_RANGE,
      joint_position_range=(-0.1, 0.1),
      motion_dir=motion_dir,
      anchor_body_name="torso_link",
      body_names=TRACKED_BODIES,
      policy_joint_names=WHOLE_BODY_JOINTS,
    )
  }
  cfg.commands = commands

  actor = cfg.observations["actor"]
  for name in ("base_ang_vel", "joint_pos", "joint_vel", "actions"):
    if name in actor.terms:
      actor.terms[name].history_length = PROPRIO_HISTORY
      actor.terms[name].flatten_history_dim = True

  cfg.scene.entities = {"robot": get_robot_cfg()}
  cfg.scene.sensors = (_self_collision_sensor(),)

  # 动作：29 个 G1 关节；两个夹爪关节由真机上独立的控制器管，不进动作空间。
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = WHOLE_BODY_ACTUATOR_EXPR
  joint_pos_action.scale = WHOLE_BODY_ACTION_SCALE

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.terminations["ee_body_pos"].params["body_names"] = END_EFFECTORS
  cfg.viewer.body_name = "torso_link"

  # 上游默认 njmax=250 是按站立/行走估的；语料里 fallAndGetUp 这类躺地帧接触点远超
  # 该值，超出的约束会被**静默丢弃**（日志刷 nefc overflow），那些帧的物理是错的。
  cfg.sim.njmax = 600
  cfg.sim.nconmax = 200

  if not has_state_estimation:
    cfg.observations["actor"] = ObservationGroupCfg(
      terms={
        k: v
        for k, v in actor.terms.items()
        if k not in ("motion_anchor_pos_b", "base_lin_vel")
      },
      concatenate_terms=True,
      enable_corruption=True,
    )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, GeneralMotionCommandCfg)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.joint_position_range = (0.0, 0.0)
    motion_cmd.sampling_mode = "start"
    motion_cmd.debug_vis = True  # 参考动作画成 ghost，便于肉眼对照

  return cfg
