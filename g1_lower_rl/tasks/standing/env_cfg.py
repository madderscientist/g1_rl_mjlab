"""「最省电站立」任务的环境配置。

目标：**站着不摔的前提下，下肢电耗最小**。没有指令，也不限定姿势——站成什么样是
优化结果，不是输入。所以目标函数只有 ``power`` 一项，其余全是单边约束。

扰动与观测噪声/延时模型直接复用速度跟踪任务（``G1-Gloria-LowerBody-Flat`` 及其
GRU 变体）的配置。
"""

from __future__ import annotations

import copy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import GaussianNoiseCfg as Gnoise
from mjlab.utils.noise import NoiseModelWithAdditiveBiasCfg
from mjlab.viewer import ViewerConfig

from g1_lower_rl.assets import (
  LOWER_BODY_ACTION_SCALE,
  LOWER_BODY_ACTUATOR_EXPR,
  LOWER_BODY_EFFORT_LIMIT,
  WHOLE_BODY_ACTUATOR_EXPR,
  get_robot_cfg,
)
from g1_lower_rl.tasks.lower_body import mdp as lower_body_mdp
from g1_lower_rl.tasks.lower_body.cfg.constants import (
  ARM_DRIFT_LEVELS,
  ARM_TORQUE_LEVELS,
  BODY_IMPULSE_LEVELS,
  FEET_SENSOR,
  FOOT_SITES,
  ITER,
  RESET_LEVELS,
)
from g1_lower_rl.tasks.lower_body.cfg.events import make_events
from g1_lower_rl.tasks.standing import mdp

# 扰动课程只在这里抓闸：骨盆倾角小于约 14 度才算「这一档站住了」。
STAND_GATE = {"max_tilt": 0.25, "min_fraction": 0.6, "ema": 0.02}

# 高度**下限**，不是目标。实测自然站姿 0.78，留 10 cm 给抗扰动的屈膝缓冲。
STAND_HEIGHT_MIN = 0.68

# 延时是机器人的属性，不是每一拍的属性：每步重采一次等于给关节速度叠了一层白噪声。
_DELAY = {"delay_min_lag": 0, "delay_max_lag": 2, "delay_hold_prob": 0.98}


def _imu_noise(std: float, bias_std: float) -> NoiseModelWithAdditiveBiasCfg:
  """逐拍高斯噪声，再叠一个每个 episode 固定的偏置。"""
  return NoiseModelWithAdditiveBiasCfg(
    noise_cfg=Gnoise(std=std),
    bias_noise_cfg=Gnoise(std=bias_std),
  )


def _feet_ground_sensor() -> ContactSensorCfg:
  return ContactSensorCfg(
    name=FEET_SENSOR,
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )


def _lower_body(*, sites: bool = False) -> SceneEntityCfg:
  """每次新建一份实体配置：管理器会就地解析，多个项共用一个实例会让第二次解析失败。"""
  if sites:
    return SceneEntityCfg("robot", site_names=FOOT_SITES)
  return SceneEntityCfg("robot", joint_names=LOWER_BODY_ACTUATOR_EXPR)


def _whole_body() -> SceneEntityCfg:
  """全身 29 轴，**不含两个夹爪偏心关节**。夹爪在训练里恒为 0，观测归一化学到的标准差
  接近零，真机上一开合就会被除成巨值盖掉平衡信号——实测夹爪偏 0.2 rad 造成的动作扰动
  是左膝偏同样角度的 5 倍。"""
  return SceneEntityCfg("robot", joint_names=WHOLE_BODY_ACTUATOR_EXPR)


def _observations() -> dict[str, ObservationGroupCfg]:
  # 没有指令项，actor 只看本体量。关节量取全身 29 轴而不是只取受控的 15 轴：手臂由上层
  # VR IK 自顾自地驱动，它的位姿直接决定质心落在哪儿，挡在观测外面等于把一个可测的量
  # 硬做成扰动。事件里的 ``reset_arm_pose`` / ``arm_pose_drift`` 会把手臂在整个可达范围
  # 内摆起来，所以这 14 维在训练中是有方差的。
  actor = {
    "base_ang_vel": ObservationTermCfg(
      func=envs_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=_imu_noise(std=0.115, bias_std=0.03),
      **_DELAY,
    ),
    "projected_gravity": ObservationTermCfg(
      func=envs_mdp.projected_gravity,
      noise=_imu_noise(std=0.029, bias_std=0.01),
      **_DELAY,
    ),
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      params={"biased": True, "asset_cfg": _whole_body()},
      noise=Gnoise(std=0.006),
      **_DELAY,
    ),
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      params={"asset_cfg": _whole_body()},
      noise=Gnoise(std=0.87),
      **_DELAY,
    ),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
  }
  # critic 可以看到真机拿不到的量：根速度直接决定了「站稳没有」，
  # 给了它值函数才不用从含噪的角速度里反推。深拷贝一份，让两组的实体配置是独立对象。
  critic = copy.deepcopy(actor)
  for term in critic.values():
    term.delay_max_lag = 0
    term.delay_hold_prob = 0.0
  critic["base_lin_vel"] = ObservationTermCfg(
    func=envs_mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"}
  )
  return {
    "actor": ObservationGroupCfg(
      terms=actor, concatenate_terms=True, enable_corruption=True
    ),
    "critic": ObservationGroupCfg(
      terms=critic, concatenate_terms=True, enable_corruption=False
    ),
  }


def _rewards() -> dict[str, RewardTermCfg]:
  """目标函数只有 ``power``，其余全是「别摔、别趴下」的单边约束。权重依据见 README。"""
  return {
    # sum (tau/额定)^2，只算策略驱动的 15 轴。平方而非 L1：站着不动时 tau*omega ≈ 0，
    # 电耗几乎全是铜损 I²R，而 tau = n·kt·I，所以功率正比于 tau²。求和而非均值：总功率
    # 是各关节相加。手臂不计：实测它占 90% 以上且主要由事件采到的位姿决定，策略控制不了。
    "power": RewardTermCfg(
      func=lower_body_mdp.normalized_joint_effort_l2,
      weight=-20.0,
      params={"asset_cfg": _lower_body(), "effort_limits": LOWER_BODY_EFFORT_LIMIT},
    ),
    # 正向项之一。没它的话全是罚项，提前摔倒反而是止损。
    "feet_grounded": RewardTermCfg(
      func=mdp.feet_both_grounded,
      weight=1.0,
      params={"sensor_name": FEET_SENSOR},
    ),
    # 主力正向项：重心越靠近两脚中点给分越多。**它是把单步净回报抬到正的那一项。**
    "com_centered": RewardTermCfg(
      func=mdp.com_over_feet,
      weight=4.0,
      params={"asset_cfg": _lower_body(sites=True), "std": 0.08},
    ),
    # 单边下限，站高不罚。L1 而非平方：平方在刚跌破下限处梯度为零，拦不住缓慢下沉。
    "height_floor": RewardTermCfg(
      func=mdp.height_floor,
      weight=-40.0,
      params={"asset_cfg": _lower_body(sites=True), "minimum": STAND_HEIGHT_MIN},
    ),
    "com_margin": RewardTermCfg(
      func=mdp.com_inside_feet_margin,
      weight=-20.0,
      params={"asset_cfg": _lower_body(sites=True)},
    ),
    "stillness": RewardTermCfg(
      func=mdp.base_stillness,
      weight=-0.5,
      params={"asset_cfg": SceneEntityCfg("robot")},
    ),
    "joint_vel": RewardTermCfg(
      func=envs_mdp.joint_vel_l2, weight=-2e-3, params={"asset_cfg": _lower_body()}
    ),
    "joint_acc": RewardTermCfg(
      func=envs_mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": _lower_body()}
    ),
    "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.1),
    "terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-200.0),
  }


def _events() -> dict[str, EventTermCfg]:
  """扰动整套复用速度跟踪任务的 :func:`make_events`，只删 ``gait_phase``（本任务没有步态时钟）。

  不重新配一遍是有教训的：第一版自己写（外力 ±60 N 同时压在两个夹爪和躯干）实测只能撑30 也不行；
  换成官方 velocity 那一小套后回合长度回到 988 步，但又漏掉了手臂力矩冲量、手臂位姿漂移、
  夹爪负载、左右独立的地面摩擦。直接引用才不会再漏。
  """
  events = make_events()
  del events["gait_phase"]
  return events


def _curriculum() -> dict[str, CurriculumTermCfg]:
  """扰动课程，档位表与速度任务共用。

  没有它的话事件全部停在第 0 档，而第 0 档的手臂力矩、漂移、外力冲量、开局幅值
  **都是 0**——等于把整套扰动关掉。闸门换成「骨盆还立着」，因为本任务没有速度指令。
  """

  def gated(event_name: str, log_key: str, stages: list) -> CurriculumTermCfg:
    return CurriculumTermCfg(
      func=mdp.gated_event_params,
      params={
        "event_name": event_name,
        "log_key": log_key,
        "gate": STAND_GATE,
        "stages": stages,
      },
    )

  return {
    "reset_pose_level": gated(
      "reset_base",
      "scale",
      [{"step": s * ITER, "params": {"scale": v}} for s, v in RESET_LEVELS],
    ),
    "reset_joint_vel_level": gated(
      "reset_robot_joints",
      "scale",
      [{"step": s * ITER, "params": {"scale": v}} for s, v in RESET_LEVELS],
    ),
    "arm_torque_level": gated(
      "arm_torque",
      "torque_range",
      [
        {"step": s * ITER, "params": {"torque_range": (-t, t)}}
        for s, t in ARM_TORQUE_LEVELS
      ],
    ),
    "body_impulse_level": gated(
      "body_impulse",
      "force_range",
      [
        {
          "step": s * ITER,
          "params": {"force_range": (-f, f), "torque_range": (-f / 6.0, f / 6.0)},
        }
        for s, f in BODY_IMPULSE_LEVELS
      ],
    ),
    "arm_drift_level": gated(
      "arm_pose_drift",
      "blend",
      [{"step": s * ITER, "params": {"blend": b}} for s, b in ARM_DRIFT_LEVELS],
    ),
  }


def _terminations() -> dict[str, TerminationTermCfg]:
  return {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    # 站立任务里「摔了」判得比行走严：躯干倾角超过 0.8 rad 就已经救不回来了。
    "fell_over": TerminationTermCfg(
      func=envs_mdp.bad_orientation,
      params={"limit_angle": 0.8, "asset_cfg": SceneEntityCfg("robot")},
    ),
    "too_low": TerminationTermCfg(
      func=envs_mdp.root_height_below_minimum,
      params={"minimum_height": 0.45, "asset_cfg": SceneEntityCfg("robot")},
    ),
  }


def make_standing_env_cfg() -> ManagerBasedRlEnvCfg:
  """构造「最省力站立」的环境配置（平地、无指令）。"""
  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=LOWER_BODY_ACTUATOR_EXPR,
      scale=LOWER_BODY_ACTION_SCALE,
      use_default_offset=True,
    )
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1,
      extent=2.0,
      entities={"robot": get_robot_cfg()},
      sensors=(_feet_ground_sensor(),),
    ),
    observations=_observations(),
    actions=actions,
    commands={},  # 本任务没有指令
    events=_events(),
    rewards=_rewards(),
    terminations=_terminations(),
    curriculum=_curriculum(),
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="torso_link",
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      njmax=300,
      mujoco=MujocoCfg(
        timestep=0.005, iterations=10, ls_iterations=20, ccd_iterations=50
      ),
      contact_sensor_maxmatch=64,
    ),
    decimation=4,  # 0.005 * 4 = 50 Hz
    episode_length_s=20.0,
  )
  return cfg


def standing_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_standing_env_cfg()
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    # 课程清空后必须手动拨到末档，否则扰动会停在第 0 档（也就是全关），
    # 评估出来的抗扰能力全是假的。
    cfg.curriculum = {}
    tau = ARM_TORQUE_LEVELS[-1][1]
    force = BODY_IMPULSE_LEVELS[-1][1]
    cfg.events["arm_torque"].params["torque_range"] = (-tau, tau)
    cfg.events["body_impulse"].params["force_range"] = (-force, force)
    cfg.events["body_impulse"].params["torque_range"] = (-force / 6.0, force / 6.0)
    cfg.events["arm_pose_drift"].params["blend"] = ARM_DRIFT_LEVELS[-1][1]
    cfg.events["reset_base"].params["scale"] = RESET_LEVELS[-1][1]
    cfg.events["reset_robot_joints"].params["scale"] = RESET_LEVELS[-1][1]
  return cfg
