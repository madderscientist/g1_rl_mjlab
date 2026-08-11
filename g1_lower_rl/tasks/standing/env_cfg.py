"""「最省力站立」任务的环境配置。

和同项目的 lower_body 行走任务的根本差别：**这个任务没有指令**。策略不需要跟随任何
速度或高度目标，唯一要做的是在被推被拽的情况下站住，并且尽量少花力气。

观测取全身 29 轴（12 腿 + 3 腰 + 14 臂），动作只有下肢 15 轴。上肢在实机上由 VR IK
自行驱动，它摆到哪儿直接决定质心落在哪儿；把它挡在观测外面，下肢就只能从骨盆姿态里
事后反推。两个夹爪关节不进观测——训练里它们恒为 0，归一化会把真机上的开合放大成巨值。

奖励的设计只围绕一句话展开：*站着别动，重心压在两脚之间，力矩越小越好*。

为什么不能只写「力矩最小」：那样最优解是直接瘫到地上——地面替你承担全部重量，力矩确实
是零。所以省力项必须和「保持站姿」的几项一起构成一个有唯一解的目标：高度项钉住站姿，
重心项钉住平衡，双脚着地项排除单脚站立，最后才轮到省力项在剩下的自由度里做优化。

扰动是这个任务的核心难点，不是可选项：没有外力时「省力」退化成静态配平，学出来的策略
一推就倒。整套扰动直接复用速度跟踪任务的 :func:`make_events`，配套课程见 :func:`_curriculum`。
"""

from __future__ import annotations

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
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from g1_lower_rl.assets import (
  ARM_TARGET_RANGES,
  LOWER_BODY_ACTION_SCALE,
  LOWER_BODY_ACTUATOR_EXPR,
  LOWER_BODY_EFFORT_LIMIT,
  get_robot_cfg,
)
from g1_lower_rl.tasks.lower_body.cfg.constants import (
  ARM_DRIFT_LEVELS,
  ARM_JOINT_EXPR,
  ARM_TORQUE_LEVELS,
  BODY_IMPULSE_LEVELS,
  ITER,
  RESET_LEVELS,
)
from g1_lower_rl.tasks.lower_body.cfg.events import make_events
from g1_lower_rl.tasks.standing import mdp

FOOT_SITES = ("left_foot", "right_foot")
FOOT_GEOMS = tuple(
  f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
)
# 外力作用点：夹爪（手臂碰到或拿着的东西）和躯干（有人推机器人）。
IMPULSE_BODIES = (r".*_gripper_base", "torso_link")

# 观测里的关节集合：29 个受控轴，**不含两个夹爪偏心关节**。夹爪在训练里恒为 0，
# 观测归一化学到的标准差接近零，真机上夹爪一开合就会被除成巨值盖掉平衡信号——
# 实测夹爪偏 0.2 rad 造成的动作扰动是左膝偏同样角度的 5 倍。
WHOLE_BODY_JOINT_EXPR = LOWER_BODY_ACTUATOR_EXPR + ARM_JOINT_EXPR

# 扰动课程只在这里抓闸：骨盆倾角小于约 14 度才算「这一档站住了」。
STAND_GATE = {"max_tilt": 0.25, "min_fraction": 0.6, "ema": 0.02}

# 站立目标高度：躯干原点相对双脚的高度。取膝盖微屈的姿态，纯直腿站立虽然更省力，
# 但膝关节顶在限位上，一受扰动就没有可用的缓冲行程。
STAND_HEIGHT = 0.72


def _feet_ground_sensor() -> ContactSensorCfg:
  return ContactSensorCfg(
    name="feet_ground_contact",
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
  return SceneEntityCfg("robot", joint_names=WHOLE_BODY_JOINT_EXPR)


def _observations() -> dict[str, ObservationGroupCfg]:
  # 没有指令项，actor 只看本体量：真机上这些全都拿得到。
  #
  # 关节量取**全身 29 轴**（12 腿 + 3 腰 + 14 臂），动作只有下肢 15 轴。手臂由上层
  # VR IK 自顾自地驱动，它的位姿直接决定了质心落在哪儿；把它挡在观测外面，下肢就只能
  # 从骨盆姿态里事后反推，等于把一个可测的量硬做成扰动。事件里的 ``reset_arm_pose`` /
  # ``arm_pose_drift`` 会把手臂在整个可达范围内摆起来，所以这 14 维在训练中是有方差的，
  # 不会重蹈夹爪那种「训练里恒定、真机上乱动」的覆辙。
  actor = {
    "base_ang_vel": ObservationTermCfg(
      func=envs_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
    ),
    "projected_gravity": ObservationTermCfg(
      func=envs_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
    ),
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      params={"asset_cfg": _whole_body()},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      params={"asset_cfg": _whole_body()},
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
  }
  # critic 可以看到真机拿不到的量：根速度直接决定了「站稳没有」，
  # 给了它值函数才不用从含噪的角速度里反推。
  critic = {
    name: ObservationTermCfg(
      func=term.func,
      params={"asset_cfg": _whole_body()} if term.params else {},
    )
    for name, term in actor.items()
  }
  critic["base_lin_vel"] = ObservationTermCfg(func=envs_mdp.base_lin_vel)
  return {
    "actor": ObservationGroupCfg(
      terms=actor, concatenate_terms=True, enable_corruption=True
    ),
    "critic": ObservationGroupCfg(
      terms=critic, concatenate_terms=True, enable_corruption=False
    ),
  }


def _rewards() -> dict[str, RewardTermCfg]:
  return {
    # —— 站姿：先保证「站着」这件事有唯一解 ——
    "height": RewardTermCfg(
      func=mdp.upright_height,
      weight=2.0,
      params={
        "asset_cfg": _lower_body(sites=True),
        "target": STAND_HEIGHT,
        "std": 0.06,
      },
    ),
    "upright": RewardTermCfg(
      func=envs_mdp.flat_orientation_l2,
      weight=-2.0,
      params={"asset_cfg": SceneEntityCfg("robot")},
    ),
    "feet_grounded": RewardTermCfg(
      func=mdp.feet_both_grounded,
      weight=1.0,
      params={"sensor_name": "feet_ground_contact"},
    ),
    # —— 平衡：重心压在两脚之间 ——
    "com_centered": RewardTermCfg(
      func=mdp.com_over_feet,
      weight=3.0,
      params={"asset_cfg": _lower_body(sites=True), "std": 0.08},
    ),
    "com_margin": RewardTermCfg(
      func=mdp.com_inside_feet_margin,
      weight=-20.0,
      params={"asset_cfg": _lower_body(sites=True)},
    ),
    # —— 本任务的真正目标：省力 ——
    #
    # 权重是实测标定的，不是拍的：默认站姿静置 150 步后 mean(力矩/额定)^2 = 0.0047
    # （最大的两项是 ankle_pitch 26%、waist_yaw 15%，其余都在 3% 以内）。
    # 取 60 之后这一项在静态站立时约 0.28，占 shaping 总量的 7%；被推的时候升到 1~3，
    # 足以把「硬顶回去」压过成「顺势卸力」——那正是「轻松站住」想要的行为。
    # 真摔了有 -200 的终止罚兜底，所以不会退化成直接躺平。
    "effort": RewardTermCfg(
      func=mdp.effort_cost,
      weight=-60.0,
      params={"asset_cfg": _lower_body(), "effort_limits": LOWER_BODY_EFFORT_LIMIT},
    ),
    # —— 静止与平滑 ——
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
    "joint_limits": RewardTermCfg(
      func=envs_mdp.joint_pos_limits, weight=-5.0, params={"asset_cfg": _lower_body()}
    ),
    "terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-200.0),
  }


def _events() -> dict[str, EventTermCfg]:
  """扰动方案整套复用同项目速度跟踪任务的 :func:`make_events`，一项不减。

  这里不重新配一遍是有教训的：第一版自己写（外力 ±60 N 同时压在两个夹爪和躯干、每
  1.5~3.5 s 换向）实测只能撑 3.7 s；换成官方 velocity 那一小套之后回合长度回到 988 步，
  但又漏掉了手臂力矩冲量、手臂位姿漂移、夹爪负载、左右独立的地面摩擦这些真正会在实机上
  出现的量。速度任务那套是在这台机器人上调出来的，直接引用才不会再漏。

  只删 ``gait_phase``：本任务没有步态时钟，相位起点无处可用。
  """
  events = make_events()
  del events["gait_phase"]

  for term in ("reset_arm_pose", "arm_pose_drift"):
    events[term].params["ranges"] = ARM_TARGET_RANGES
  events["body_impulse"].params["asset_cfg"].body_names = IMPULSE_BODIES
  events["payload_mass"].params["asset_cfg"].body_names = (r".*_gripper_base",)
  events["foot_friction_left"].params["asset_cfg"].geom_names = tuple(
    g for g in FOOT_GEOMS if g.startswith("left")
  )
  events["foot_friction_right"].params["asset_cfg"].geom_names = tuple(
    g for g in FOOT_GEOMS if g.startswith("right")
  )
  events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
  return events


def _curriculum() -> dict[str, CurriculumTermCfg]:
  """扰动课程，档位表与速度任务共用。

  没有课程的话上面那套事件全部停在第 0 档，而第 0 档的手臂力矩、手臂漂移、外力冲量、
  开局姿态幅值**都是 0**——等于把整套扰动关掉。闸门条件换成「骨盆还立着」，因为本任务
  没有速度指令可跟随。
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
