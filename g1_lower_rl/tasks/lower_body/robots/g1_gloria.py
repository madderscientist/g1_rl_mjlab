"""Unitree G1 + 双 Gloria-M 的下肢行走配置。"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from g1_lower_rl.assets import (
  ARM_TARGET_RANGES,
  LOWER_BODY_ACTION_SCALE,
  LOWER_BODY_ACTUATOR_EXPR,
  get_robot_cfg,
)
from g1_lower_rl.tasks.lower_body import mdp
from g1_lower_rl.tasks.lower_body.cfg import make_lower_body_env_cfg, max_out_curriculum
from g1_lower_rl.tasks.lower_body.cfg.constants import BODY_IMPULSE_LEVELS, ITER

FOOT_SITES = ("left_foot", "right_foot")
FOOT_GEOMS = tuple(
  f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
)
# 外部推力作用在哪里：夹爪（手臂碰到或拿着的东西）和躯干（有人推机器人）。
IMPULSE_BODIES = (r".*_gripper_base", "torso_link")

# 资产里的动作缩放是从 effort/stiffness 推出来的，给 roll/pitch 的权限比膝盖还大
# （0.44 vs 0.35 rad / 单位动作），而它们的行程只有 ±0.52 rad，多出来的量程只是给躯干
# 抖动提供幅度。waist_yaw 不在内：它需要完整权限去做反向旋转。
#
# 有代价，要知道：``body_orientation_l2`` 量的是 torso_link，就在腰关节正上方，把这个系数
# 从 1.0 减半之后实测躯干偏离竖直的程度差 3 倍（−0.0066 → −0.0200，两边权重相同可直接比），
# 从零训练更难爬出“坐在地上”那个状态。真要再动，做成课程（参照 mdp.tightening_height_std），
# 别改常量。
WAIST_SCALE_FACTOR = 0.5
WAIST_SCALED_JOINTS = ("waist_roll_joint", "waist_pitch_joint")

# GRU 的 BPTT 截断窗口。环境配置也要用它去缩放课程时间表，所以提到模块级。
GRU_NUM_STEPS_PER_ENV = 48


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
    track_air_time=True,
  )


def _self_collision_sensor() -> ContactSensorCfg:
  # 只算左腿碰撞几何体与右腿子树的接触——腿绞在一起才是策略的错。
  #
  # 曾经两边都是 subtree("pelvis")，而 pelvis 是根体——等于整棵树，手臂全包在内。
  # 实测当前策略的自碰撞事件 **100% 都含手臂**：手臂撞躯干 73.1%、撞腿 26.1%、
  # 手臂互撞 0.9%。而手臂是被事件随机扰动的，撞上不是策略造成的——与 joint_acc_l2
  # 只限受控关节同一个理由。排掉之后好行为的代价恰好是 0，权重才能给重。
  #
  # 为什么不用 exclude：subtree 模式下 exclude 过滤的是“匹配到的根刚体名”，
  # 而 pattern 只匹配到 pelvis 一个，排不掉子树里的手臂；改成 torso_link 也不行，
  # 手臂就挂在它下面。另外 secondary 必须解析成单个名字，所以“非手臂 vs 非手臂”
  # 这个 API 表达不了，只能取“左腿 vs 右腿”——而那正是双足真正要防的失效模式。
  return ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(
      mode="geom",
      pattern=r"^left_.*_collision$",
      entity="robot",
      exclude=(r"(shoulder|elbow|wrist|gripper|camera|finger)",),
    ),
    secondary=ContactMatch(
      mode="subtree", pattern="right_hip_pitch_link", entity="robot"
    ),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )


def _apply_play_overrides(cfg: ManagerBasedRlEnvCfg) -> None:
  """把配置拨到“训练末期”的工况，用于评估与录像。"""
  cfg.episode_length_s = int(1e9)  # 相当于无限
  cfg.observations["actor"].enable_corruption = False
  for term in cfg.observations["actor"].terms.values():
    term.delay_max_lag = 0

  # 上肢扰动保留，且直接给到训练结束时的档位：它们就是这个任务的重点，拿课程的
  # *初始*档位去评估会低估策略实际被推到什么程度。指令量程同理，必须与训练末档一致，
  # 否则会系统性避开策略最容易失效的高速段——前进能力崩塌恰恰只在 vx 超过 1.0 之后才看得见。
  cfg.events.pop("push_robot", None)
  max_out_curriculum(cfg)


def flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """构造平地下肢行走配置。"""
  cfg = make_lower_body_env_cfg()

  cfg.scene.entities = {"robot": get_robot_cfg()}
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64

  feet_ground = _feet_ground_sensor()
  self_collision = _self_collision_sensor()
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground, self_collision)

  # 动作：12 个腿关节 + 3 个腰关节，顺序与真机电机编号一致。
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = LOWER_BODY_ACTUATOR_EXPR
  joint_pos_action.scale = {
    expr: scale * (WAIST_SCALE_FACTOR if expr in WAIST_SCALED_JOINTS else 1.0)
    for expr, scale in LOWER_BODY_ACTION_SCALE.items()
  }

  height_cmd = cfg.commands["height"]
  assert isinstance(height_cmd, mdp.BaseHeightCommandCfg)
  height_cmd.site_names = FOOT_SITES

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  for term in ("base_height", "foot_height"):
    cfg.observations["critic"].terms[term].params["asset_cfg"].site_names = FOOT_SITES

  for term in ("reset_arm_pose", "arm_pose_drift"):
    cfg.events[term].params["ranges"] = ARM_TARGET_RANGES
  cfg.events["body_impulse"].params["asset_cfg"].body_names = IMPULSE_BODIES
  cfg.events["payload_mass"].params["asset_cfg"].body_names = (r".*_gripper_base",)
  cfg.events["foot_friction_left"].params["asset_cfg"].geom_names = tuple(
    g for g in FOOT_GEOMS if g.startswith("left")
  )
  cfg.events["foot_friction_right"].params["asset_cfg"].geom_names = tuple(
    g for g in FOOT_GEOMS if g.startswith("right")
  )
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  for term in ("track_base_height", "foot_clearance", "foot_slip"):
    cfg.rewards[term].params["asset_cfg"].site_names = FOOT_SITES
  # 这一项作用在骨盆而不是躯干：骨盆后坐时腰关节会把偏差吸收掉，量躯干看不见。
  cfg.rewards["pelvis_backward_lean"].params["asset_cfg"].body_names = ("pelvis",)
  for term in ("body_orientation_l2", "body_ang_vel"):
    cfg.rewards[term].params["asset_cfg"].body_names = ("torso_link",)
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    # 传感器已排掉手臂（见上），好行为的代价恰好是 0，所以不再需要“压轻以免淹掉
    # 速度跟随”——从 -0.5 提到 -2.0。算一下量级：原始值是一个控制拍内接触力超
    # 10 N 的子步数（0~4），所以满接触的一拍要罚 2.0 × 4 × dt = 0.16，是同一拍
    # 线速度跟随满分（2.5 × dt = 0.05）的 3.2 倍——足够威慑，但远没到一撞就完。
    weight=-2.0,
    params={"sensor_name": self_collision.name, "force_threshold": 10.0},
  )

  cfg.terminations["collapsed"].params["asset_cfg"].site_names = FOOT_SITES
  cfg.viewer.body_name = "torso_link"

  if play:
    _apply_play_overrides(cfg)
  return cfg


def _rescale_curriculum_steps(cfg: ManagerBasedRlEnvCfg, factor: float) -> None:
  """课程档位的 ``step`` 是按「迭代数 x ITER」写死的，而 ``ITER`` 假设了每迭代 24 步。

  ``common_step_counter`` 每次 ``env.step()`` 加一，所以把 ``num_steps_per_env`` 调大
  之后，同一个 ``step`` 阈值会在更早的迭代就跨过——不缩放的话课程会整体提前。
  """
  for term in cfg.curriculum.values():
    for key, value in term.params.items():
      if key.endswith("stages") and isinstance(value, list):
        for stage in value:
          stage["step"] = round(stage["step"] * factor)


def flat_env_cfg_gru(play: bool = False) -> ManagerBasedRlEnvCfg:
  """GRU 版环境。与 MLP 版的差别都在这里。

  **actor 观测去掉 ``actions``（上一拍动作）。** 对 MLP 它是唯一的历史来源，非留不可；
  对 GRU 它是冗余的——隐状态本来就记得自己上一拍输出了什么，再从输入喂回去等于给策略接了
  一条显式正反馈通路。实测摆动确实不靠它传播（把它钉成 0、观测不变，跑 20 拍 waist_roll
  照样从 -1.36 漂到 +0.81），所以删掉它是**减少一个可能的振荡回路**，不是指望它单独治好
  摆动。critic 那份保留：它是 MLP，看得到上一拍动作才能把「这个状态是被什么动作带进来的」
  算进 value。``action_rate_l2`` 读的是 ``action_manager.prev_action``，不经过观测，不受影响。

  **观测延迟关掉。** 随机滞后让同一个物理状态对应到不同的观测帧，等于往 GRU 要学的时序
  关系里掺抖动。先确认 GRU 在干净时序下能不能学出来，再谈把延迟加回去。

  **外力冲量课程砍掉最高档。**
  """
  cfg = flat_env_cfg(play=play)
  del cfg.observations["actor"].terms["actions"]

  for term in cfg.observations["actor"].terms.values():
    term.delay_min_lag = 0
    term.delay_max_lag = 0
    term.delay_hold_prob = 0.0

  force = BODY_IMPULSE_LEVELS[-2][1]
  if play:
    # play 时课程已被清空，``max_out_curriculum`` 按常量拨到了原最高档，这里拨回来。
    cfg.events["body_impulse"].params["force_range"] = (-force, force)
    cfg.events["body_impulse"].params["torque_range"] = (-force / 6.0, force / 6.0)
  else:
    cfg.curriculum["body_impulse_level"].params["stages"].pop()
    _rescale_curriculum_steps(cfg, GRU_NUM_STEPS_PER_ENV / ITER)
  return cfg


def ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_gloria_lower_body",
    logger="tensorboard",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10001,
  )


def ppo_runner_cfg_gru() -> RslRlOnPolicyRunnerCfg:
  """只把 actor 的 MLP 换成 GRU

  为什么值得试：这个任务的观测里有几样东西是**当前帧看不出来的**——脚底打滑、地面
  软硬、手臂被外力拽住的方向，都要跨若干帧才能分辨。MLP 只能靠观测里显式给的历史项去
  推断，而 GRU 自带状态，理论上能把整段接触史压进隐状态。

  rsl_rl 5.4 的 ``RNNModel`` 自带 TorchScript/ONNX 导出，所以部署链路不用改。
  """
  cfg = ppo_runner_cfg()
  cfg.actor.class_name = "RNNModel"
  cfg.actor.rnn_type = "gru"
  cfg.actor.rnn_hidden_dim = 32
  cfg.actor.rnn_num_layers = 1
  # GRU 之后的 MLP 收窄一档：隐状态已经承担了大部分容量，再堆宽只是更难训。
  cfg.actor.hidden_dims = (256, 128)
  # RNN 靠 BPTT 学时序，24 拍（0.48 s）还不到一个步态周期（0.6 s），截断窗口跨不过一步。
  cfg.num_steps_per_env = GRU_NUM_STEPS_PER_ENV
  cfg.experiment_name = "g1_gloria_lower_body_gru"
  return cfg
