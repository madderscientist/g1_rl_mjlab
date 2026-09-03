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

import copy
import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg as EnvTerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.tracking_env_cfg import VELOCITY_RANGE, make_tracking_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from g1_lower_rl.assets import (
  ARM_JOINTS,
  WHOLE_BODY_ACTION_SCALE,
  WHOLE_BODY_ACTUATOR_EXPR,
  WHOLE_BODY_JOINTS,
  get_robot_cfg,
)
from g1_lower_rl.tasks.motion_tracking import mdp
from g1_lower_rl.tasks.motion_tracking.mdp import (
  GeneralMotionCommandCfg,
  GravityCompensatedJointPositionActionCfg,
  payload_mass,
)

DEFAULT_MOTION_DIR = "motions/lafan1"
"""参考动作语料目录。目录下所有 NPZ 一起参与训练。"""

PROPRIO_HISTORY = 5
"""actor 本体观测保留的历史帧数（含当前帧）。"""

RGMT_HISTORY = 10
"""RGMT 分支编码用的历史帧数，对应论文的 H=10。"""

RGMT_PROP_TERMS: tuple[str, ...] = (
  "projected_gravity",
  "base_ang_vel",
  "joint_pos",
  "joint_vel",
)
"""本体观测 o_prop 的组成，对应论文式 (1) 的 ``[g_proj, ω, q-q0, q̇]``。"""

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

# 判「跟丢」的末端（只比高度，阈值 0.25m）。**只留双脚，不含手腕**——这是相对上游
# 官方配置的有意偏离。
#
# 官方（原厂 G1）用双踝 + 双腕。我们换了 Gloria-M 夹爪后，肩 pitch 惯量 ×2.25、带宽从
# 1.64 Hz 掉到 1.10 Hz，而电机仍是原厂 5020，手臂的跟踪能力本来就低了三分之一，阈值却
# 没动，等于在要求手臂做它做不到的事。
#
# 实测（model_43800，关掉该终止后统计）：手腕触发次数是脚踝的 3 倍（357 vs 114），
# 且 65% 的终止是「脚跟得好好的、光手腕超了」；超阈持续时间中位 5~10 拍、最长 77 拍，
# 是持续状态而非抖动，说明策略无法改善——终止只是把腿部本可继续学习的样本一起销毁。
#
# 摔倒本身由 anchor_pos（躯干高度偏 0.25m）和 anchor_ori（0.8 rad）覆盖，去掉手腕不会
# 漏掉真正的失败；手臂精度仍由 motion_body_pos / motion_body_ori 奖励项驱动。
END_EFFECTORS: tuple[str, ...] = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
)

# 参考 token 里额外给笛卡尔位置的刚体，坐标系是**机器人当前** anchor 的 yaw 局部系。
# anchor 自身在内：它的当前帧分量就是「我离虚影多远」，是治漂移的误差反馈通道。
# 手腱在内：手臂原本只有关节角目标，没有笛卡尔目标可补偿重力/惯性下的稳态误差。
REFERENCE_KEY_BODIES: tuple[str, ...] = (
  "torso_link",
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
  "left_ankle_roll_link",
  "right_ankle_roll_link",
)
REFERENCE_KEY_BODY_POS = True
REFERENCE_KEY_BODY_VEL = True


# 手部单独立项的末端。与双脚同理：14 个 body 取平均再套 σ=0.3 的核，手腱误差会被
# 腿部稀释，而手在动捕里运动幅度最大、误差天然更大。ASAP/OmniH2O/DeepMimic 三家
# 都给上肢更严的 σ（分别严 2× / 50× / 2.2×）。
HANDS: tuple[str, ...] = (
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
)


def resolve_reference_key_bodies() -> tuple[str, ...]:
  """env_cfg / rl_cfg / export_onnx **必须**共用这一个来源。

  三处各自拼一遍曾经导致维度对不上，现在只留一个真值。
  """
  return REFERENCE_KEY_BODIES


# 真机手臂带重力补偿：补偿器算出力矩后按 kp 折算成位置偏移叠加到目标上。这里同样建模，
# 系数**按臂共享**（同一条臂共用一套模型参数和一个负载估计器，误差是相关的）。
# 上界略超 1.0：补偿器也会过补，不只会欠补。
ARM_GRAVCOMP_GAIN_RANGE: tuple[float, float] = (0.9, 1.01)

# 参考动作的播放倍率，每回合采样一个。
#
# 原速 LAFAN1 里很多动作在这台机器上物理不可达：肩关节幅值 1 rad 的正弦摆臂在
# 1.5 Hz 就需要 26.6 N·m，超过 25 N·m 上限。早期对照：整体放慢 1.5 倍后 iter 12000 的
# 回合长度从 12.91 涨到 50.44（×3.9）。
MOTION_SPEED_RANGE: tuple[float, float] = (0.8, 1.0)
#
# 下界取负是为了让「空手」占一块有限概率（这里约 1/3），而不是概率为零的边界点。
# 上界 1.0 kg 是量出来的：取语料 1941 帧真实臂姿算重力 + M*qddot，瓶颈不是肩而是腕
# （wrist_pitch/yaw 用 4010 电机，上限仅 5 N·m）。1.0 kg 时腕的 P95 需求已达上限 72%，
# 2.0 kg 时 106%，而同期肩关节还剩 28% 余量。再往上加只会逆选择出“拿重物就放弃手臂跟踪”。
#
# 左右独立采样（单手拿东西比双手对称更常见，也更难：重心横向偏移，腿要补偿）。
PAYLOAD_MASS_RANGE: tuple[float, float] = (-0.5, 1.0)

# PD 增益的相对标定误差。mjlab 的 kp 是按电机**转子反射惯量**定的（STIFFNESS =
# ARMATURE * ω²），完全没算连杆惯量，所以真机上的**等效**增益本来就不确定；
# 再加上固件电流环的实际带宽差异，±20% 是保守估计。
PD_GAIN_SCALE_RANGE: tuple[float, float] = (0.8, 1.2)

# 连杆质量密度的全局缩放（质量与惯量同时按 e^{2α} 变，COM 不变）。
# ±0.05 ≈ 质量 ±10%。用 pseudo_inertia 而不是 body_mass：后者只改质量不改惯量，
# 会造出密度不自洽的刚体，等于把策略往不存在的动力学上训。
LINK_INERTIA_ALPHA_RANGE: tuple[float, float] = (-0.05, 0.05)

# 关节干摩擦（N·m，绝对值）。模型里是 0，而真机谐波减速器 + 密封的静摩擦不可忽略；
# 不建模的话策略会学出依赖「零摩擦自由摆动」的步态。
JOINT_FRICTION_RANGE: tuple[float, float] = (0.0, 0.2)

# 电机实际出力相对额定值的缩放：温升降额、母线电压波动、力矩常数个体差异。
MOTOR_STRENGTH_SCALE_RANGE: tuple[float, float] = (0.8, 1.2)

# 关节电枢（转子反射惯量）的缩放。只往上不往下：模型里已经按额定转子惯量填了，
# 真机的联轴器、编码器盘、线缆只会再加一点。
JOINT_ARMATURE_SCALE_RANGE: tuple[float, float] = (1.0, 1.05)

# 参考指令观测的均匀噪声半宽。
COMMAND_NOISE: float = 0.02

# 初始状态随机化。复位总是把机器人写成**参考动作的姿态**再叠加这些扰动，
# 所以范围就是训练分布在参考流形周围的“半径”。原来的 ±0.1 rad / ±1 cm 太窄，
# 上机时只要初始姿态不是直立就落到分布外。
#
# 幅度必须**分阶段加宽**，不能一步到位。曾经直接拉到 ±0.3 / +0.15 训了 1600 迭代：
# 单步存活率看着还行（关节 ±0.3 -> 90%、倾角 ±0.3 -> 84%），但整条回合的 reward 从
# +0.97 掉到 -0.4，优势信号弱到 surrogate 只有 -0.005，自适应 KL 把学习率压到 1e-5
# （正常量级 5.8e-4），于是固定的 entropy_coef 反过来主导了目标——σ 从 0.265 单调
# 涨到 0.344，策略越训越随机。确定性评测：干净启动 30.2 s -> 15.4 s，扰动启动
# 16.3 s -> 8.4 s，两个场景一起退化。
#
# 所以这里是课程的**第一档**（约目标值的一半）。等 reward 回正、σ 回落到 0.27 附近
# 再往上加，判据见 README。
INIT_JOINT_RANGE: tuple[float, float] = (-0.15, 0.15)
INIT_TILT_RANGE: tuple[float, float] = (-0.15, 0.15)

# 高度必须**不对称**：向下会把脚压进地面造出虚假的穿透接触，向上才是有意义的
# “下落 + 落地”任务。目标上界 0.15 m（+0.15 存活 98%，+0.30 超过 ``anchor_pos``
# 的 0.25 m 阈值直接废掉），当前同样先取一半。
INIT_HEIGHT_RANGE: tuple[float, float] = (-0.01, 0.06)

# 复位后屏蔽“跟丢”终止的步数。0 = 关闭。
#
# 当初开这个是为了救 ±0.3/+0.15 那档随机化——三项误差相加后组合中位 0.344 m 超过
# ``ee_body_pos`` 的 0.25 m 阈值，84.6% 的回合复位当帧即死。收窄到半幅后这个前提
# 没了：实测复位首帧误差中位 0.058、p90 0.086，离阈值还差 3 倍，根本不会秒死。
#
# 而它的副作用是实打实的：已经跟丢的回合被强行多跑 15 步，那 15 步跟踪奖励接近 0、
# 惩罚项照算。同一策略同一语料的控制变量实测（512 env × 400 步）：
#   窄±0.1  无宽限 eplen 53.2 reward +0.721 | +宽限15 eplen 58.8 reward +0.371
#   半幅    无宽限 eplen 47.5 reward +0.454 | +宽限15 eplen 52.8 reward +0.180
# eplen 只换来 +10%，reward 却掉 50~60%——在优势信号本来就弱的时候，这等于把熵项
# 推上主导位（见 eval-gotchas 第 24 条）。
# 跟踪失败后每步的终止概率（Stubborn 式软终止，arXiv:2606.12814 §3.1）。
#
# 硬终止下策略永远见不到「已经摔了」之后的状态——失败即复位，倒地过程被剪掉，
# 那片状态空间对策略是未知的，恢复行为无从学起。改成概率终止后失败态会持续进入
# rollout，跟踪奖励自然把机器人往参考姿态拉，恢复行为作为副产品涌现，
# 不需要恢复专用奖励，也不需要独立的起身策略。
#
# 取值由物理恢复时间反推：存活步数服从 Geo(p)，令 1/p = f_ctrl * t_rec，
# 50 Hz、t_rec=4 s 得 p=0.005，期望恢复窗口 200 步。论文消融里强扰动下的
# 恢复成功率 77.5%/85.0%(硬终止) -> 100%(软终止)。本仓实测跨类别存活 46.9 -> 78.1 s。
FALL_RECOVERY_SECONDS: float = 4.0
PROB_TERMINATION: float = 1.0 / (50.0 * FALL_RECOVERY_SECONDS)

# 训练回合上限。上游默认 10 s（500 步），开了概率终止后不够用：
# 失败后平均还要跑 200 步，实测 37.2% 的回合撞上 500 步上限被截断。
# 而截断引入的价值估计偏差正是概率终止想避开的东西（几何分布的无记忆性
# 才能保证 TD 一致）。p=0.005 下要让 99% 的失败回合在上限前概率终止，
# 需约 ln(0.01)/ln(0.995) ≈ 920 步，所以取 20 s（1000 步），与论文 T_max 一致。
EPISODE_LENGTH_S: float = 20.0


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
  arm_gravcomp_gain_range: tuple[float, float] = ARM_GRAVCOMP_GAIN_RANGE,
  payload_mass_range: tuple[float, float] = PAYLOAD_MASS_RANGE,
  motion_speed_range: tuple[float, float] = MOTION_SPEED_RANGE,
  residual_action: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """构造全身动作跟踪配置（平地）。

  真机上根节点的水平速度没有可靠来源，所以策略侧观测里不能出现它。
  这一约束现在由 ``rg_*`` 组的构成天然保证（只有重力投影/角速度/关节/动作/参考）；
  critic 不受限，它拿得到特权信息。
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
        "z": INIT_HEIGHT_RANGE,
        "roll": INIT_TILT_RANGE,
        "pitch": INIT_TILT_RANGE,
        "yaw": (-0.2, 0.2),
      },
      velocity_range=VELOCITY_RANGE,
      joint_position_range=INIT_JOINT_RANGE,
      motion_dir=motion_dir,
      anchor_body_name="torso_link",
      body_names=TRACKED_BODIES,
      reference_key_bodies=resolve_reference_key_bodies(),
      reference_key_body_pos=REFERENCE_KEY_BODY_POS,
      reference_key_body_vel=REFERENCE_KEY_BODY_VEL,
      policy_joint_names=WHOLE_BODY_JOINTS,
      speed_range=motion_speed_range,
    )
  }
  cfg.commands = commands

  actor = cfg.observations["actor"]
  for name in ("base_ang_vel", "joint_pos", "joint_vel", "actions"):
    if name in actor.terms:
      actor.terms[name].history_length = PROPRIO_HISTORY
      actor.terms[name].flatten_history_dim = True

  # 对参考指令 s_t^g 本身加噪。真机上参考来自重定向或惯性动捕，带有相位偏差、根节点
  # 漂移和局部姿态不一致；只在干净参考上训练的策略一遇到这些就崩。
  # 幅度是本仓取值（与 encoder_bias 同量级），论文只说有这一项、未给具体区间。
  if "command" in actor.terms:
    actor.terms["command"].noise = Unoise(
      n_min=-COMMAND_NOISE, n_max=COMMAND_NOISE
    )

  # RGMT 的结构化观测：**每个分支单独成组**。分组而不是合并成一个大向量，是因为
  # mjlab 对每个项先展开历史再拼接，合并后的向量布局是 ``[项1全部历史 | 项2全部历史 | ...]``，
  # 直接 reshape 成 (H, d) 会把时间轴和特征轴拧反。每项一组后，组维度 / H 就是每拍维度，
  # 布局自描述，模型不需要额外配置就能还原 token 序列。
  #
  # ``GRU_ACTOR=1`` 走另一条路：保留单一 ``actor`` 组、不建 rg_*。GRU 靠隐状态自己记
  # 时序，再给 10 帧显式历史就不是干净的架构对比了。
  #
  # 只在这里分支、不提前 return：传感器、概率终止、奖励整形都在后面，early-return
  # 会把它们全跳过（self_collision 传感器没注册但奖励项还在，第一次算 reward 直接崩）。
  if os.environ.get("GRU_ACTOR", "0") == "1":
    cfg.observations["actor"].enable_corruption = True
  else:
    for name in RGMT_PROP_TERMS + ("actions",):
      if name in actor.terms:
        term = copy.deepcopy(actor.terms[name])
      else:
        term = ObservationTermCfg(func=getattr(mdp, name))
      term.history_length = RGMT_HISTORY
      term.flatten_history_dim = True
      cfg.observations[f"rg_{name}"] = ObservationGroupCfg(
        terms={name: term}, concatenate_terms=True, enable_corruption=True
      )
    cfg.observations["rg_reference"] = ObservationGroupCfg(
      terms={
        "reference_window": ObservationTermCfg(
          func=mdp.motion_reference_window,
          params={"command_name": "motion"},
          noise=Unoise(n_min=-COMMAND_NOISE, n_max=COMMAND_NOISE),
        )
      },
      concatenate_terms=True,
      enable_corruption=True,
    )
    # actor 组到此只剩下“当过 rg_* 的模板”这个用途：模型吃的是 rg_*，critic 吃 critic。
    # 留着它 ObservationManager 会每步白算一遍（项在 critic/rg_* 里都有），
    # 实测 512 env 下删掉提速 12.9%、观测总维度 3256 -> 2390。
    #
    # 导出部署契约时需要它，由 scripts/export_onnx.py 用 rg_* 各组的并集重建。
    del cfg.observations["actor"]

  cfg.scene.entities = {"robot": get_robot_cfg()}
  cfg.scene.sensors = (_self_collision_sensor(),)

  # 三个失败判据全部改成概率终止，倒地判据也包——摔了之后那段状态正是要学的。
  # 三者共享同一次伯努利采样，否则实际终止率会翻三倍、窗口缩到 1.3 s。
  #
  # 训练和回放都开：若评测走硬终止，误差一超阈就判死，恢复行为根本没机会发生——
  # 量到的只是「多不容易失败」而不是「失败后能不能爬起来」，两者正是 Stubborn 要统一的两个能力。
  for name in ("ee_body_pos", "anchor_pos", "anchor_ori"):
    term = cfg.terminations[name]
    cfg.terminations[name] = EnvTerminationTermCfg(
      func=mdp.with_probabilistic_termination,
      params={
        "base_func": term.func,
        "p_term": PROB_TERMINATION,
        **dict(term.params),
      },
    )

  # 上游 anchor_pos 判据是 `bad_anchor_pos_z_only`：**只看高度，漂 5 米也不终止**。
  # 于是「漂着但站得好好的」永远不被判负，反因存活久被 advantage 强化——漂移的正反馈。
  # 换上全 3D 版本，阈值 2.0 m：分离实验表明漂移的 −93% 里 **−76% 来自奖励**，
  # 只有最后 17 个点来自紧终止，而那 17 个点要用 9 s 存活 + 手部与全身精度去换。
  from mjlab.tasks.tracking.mdp import terminations as _tracking_terms

  p = cfg.terminations["anchor_pos"].params
  p["base_func"] = _tracking_terms.bad_anchor_pos
  p["threshold"] = 2.0

  # 动作：29 个 G1 关节；两个夹爪关节由真机上独立的控制器管，不进动作空间。
  # 手臂用带重力补偿的动作项，对应真机上把补偿力矩折算成位置偏移的做法。
  #
  # ``residual_action`` 把动作基准从默认站姿换成当前参考姿态（Extreme-RGMT 式 (3)）。
  # 零动作即“照抄参考”，网络只学修正量，于是 action_rate 惩罚不再随参考动作的幅度和
  # 速度一起放大——原来的写法等于给大幅度动作系统性加税。
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  cfg.actions["joint_pos"] = GravityCompensatedJointPositionActionCfg(
    entity_name=joint_pos_action.entity_name,
    actuator_names=WHOLE_BODY_ACTUATOR_EXPR,
    scale=WHOLE_BODY_ACTION_SCALE,
    offset=joint_pos_action.offset,
    preserve_order=joint_pos_action.preserve_order,
    use_default_offset=False if residual_action else joint_pos_action.use_default_offset,
    reference_command_name="motion" if residual_action else None,
    compensated_joint_names=ARM_JOINTS,
    gain_range=arm_gravcomp_gain_range,
  )

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # 下面三项都是 startup 模式：每个并行环境开局抽一次并保持不变，靠环境间的差异
  # 而不是回合间的差异提供多样性（MJWarp 改不了逐回合的模型参数）。
  cfg.events["pd_gains"] = EventTermCfg(
    func=dr.pd_gains,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "kp_range": PD_GAIN_SCALE_RANGE,
      "kd_range": PD_GAIN_SCALE_RANGE,
      "operation": "scale",
    },
  )
  cfg.events["link_inertia"] = EventTermCfg(
    func=dr.pseudo_inertia,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "alpha_range": LINK_INERTIA_ALPHA_RANGE,
    },
  )
  cfg.events["joint_friction"] = EventTermCfg(
    func=dr.joint_friction,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=WHOLE_BODY_JOINTS),
      "ranges": JOINT_FRICTION_RANGE,
      "operation": "abs",
    },
  )
  cfg.events["motor_strength"] = EventTermCfg(
    func=dr.effort_limits,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "effort_limit_range": MOTOR_STRENGTH_SCALE_RANGE,
      "operation": "scale",
    },
  )
  cfg.events["joint_armature"] = EventTermCfg(
    func=dr.joint_armature,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=WHOLE_BODY_JOINTS),
      "ranges": JOINT_ARMATURE_SCALE_RANGE,
      "operation": "scale",
    },
  )

  # 手上拿东西。负载建模成夹爪质心处的质点：绕肩/胘的 m*d^2 由质量自动带出，
  # 而负载绕自身质心的转动惯量对紧凑物体而言比手臂惯量小两个数量级，可忽略。
  cfg.events["payload"] = EventTermCfg(
    func=payload_mass,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg(
        "robot", body_names=(r"^(left|right)_gripper_base$",)
      ),
      "mass_range": payload_mass_range,
    },
  )
  cfg.terminations["ee_body_pos"].params["body_names"] = END_EFFECTORS

  # 末端单独立一项，否则抬脚学不会。``motion_body_pos`` 把 14 个 body 的误差取平均，
  # 拖着一只脚走（差 0.107 m）摊下来只剩 0.00082，奖励从 1.000 掉到 0.991——
  # 0.9% 的信号连噪声都不如，策略当然不抬。硬终止时不抬脚会直接摔死，是终止条件在
  # 替奖励施加压力；改成概率终止后那个压力没了，奖励里一直存在的稀释缺陷就暴露了
  # （实测抬脚比从 c8 的 0.95x 塔到 0.51x）。
  #
  # 只看双踝 + std 收到 0.15，区分度从 0.9% 提到 22%（拖脚 0.775 vs 抬脚 0.998）。
  cfg.rewards["motion_feet_pos"] = RewardTermCfg(
    func=cfg.rewards["motion_body_pos"].func,
    params={"command_name": "motion", "std": 0.15, "body_names": END_EFFECTORS},
    weight=0.5,
  )

  # 抬脚按**比值**评分。绝对误差形式对「系统性抬不够」几乎无感（拖脚只让奖励掉 0.9%），
  # 而实测抬脚比仅 0.63、且参考要求越高跟得越差（0.12 m 以上只有 0.45×）。
  # 换成比值后同一偏差的信号强 87 倍（区分度 0.9% -> 78.7%）。
  cfg.rewards["motion_swing_lift"] = RewardTermCfg(
    func=mdp.motion_swing_lift_ratio,
    params={
      "command_name": "motion",
      "body_names": END_EFFECTORS,
      "std": 0.3,
    },
    weight=1.5,
  )

  # 手部单独立项，σ 比整体 body_pos 紧一倍。完全照搬 `motion_feet_pos` 的做法——
  # 那一项当初就是为了解决「末端误差被 14 个 body 平均稀释」，手部是同一个病。
  # σ=0.07：实测手部误差 0.0668 时，σ=0.15 得 0.82、σ=0.10 得 0.64，收紧能提区分度且不饱和。
  cfg.rewards["motion_hand_pos"] = RewardTermCfg(
    func=cfg.rewards["motion_body_pos"].func,
    params={
      "command_name": "motion",
      "std": 0.07,
      "body_names": HANDS,
    },
    weight=1.0,
  )

  # 手相对躯干的**位形**跟踪，专供精细操作：上一项比的是重锚定后的世界位置，
  # 里面混着全身平移与朝向；这一项只问「手相对身体在哪」。两者互补。
  cfg.rewards["motion_hand_torso_rel"] = RewardTermCfg(
    func=mdp.motion_ee_pos_torso_relative_exp,
    params={
      "command_name": "motion",
      "body_names": HANDS,
      "std": 0.07,
      "still_std": 0.03,
      "still_speed": 0.5,
    },
    weight=0.75,
  )

  # 以下两项治「越漂越远」：250 s 漂移从 5.71 m 压到 0.41 m（−93%），其中 −76% 来自这两
  # 项奖励本身，不靠收紧终止阈值。
  cfg.rewards["motion_anchor_lin_vel"] = RewardTermCfg(
    func=mdp.motion_anchor_lin_vel_error_exp,
    params={"command_name": "motion", "std": 0.4},
    weight=0.4,
  )
  coarse = copy.deepcopy(cfg.rewards["motion_global_root_pos"])
  coarse.params = {**coarse.params, "std": 1.2}
  coarse.weight = 0.3
  cfg.rewards["motion_global_root_pos_coarse"] = coarse

  cfg.viewer.body_name = "torso_link"

  # 上游默认 njmax=250 / nconmax=35 是按站立/行走估的；语料里 fallAndGetUp、爬行、
  # 翻滚这类躺地帧的接触远超该值。
  #
  # **溢出不会报错，是越界写**，表现为 SIGSEGV / CUDA illegal memory access，而且多卡
  # 下先挂的那个 rank 会拖着另一个在 all_reduce 里等满 NCCL 10 分钟超时才暴露——现象
  # 离病因很远。2026-08-14 就是这么炸的两次。
  #
  # 实测（512 envs，语料含 AMASS 后）：单世界接触峰值 91，逐世界 nefc 峰值 395。
  # nconmax 是**逐世界**上限（总分配 = nconmax × nworld），按峰值留 2.8 倍余量。
  cfg.sim.njmax = 800
  cfg.sim.nconmax = 256

  # 回合上限要给概率终止留出空间：失败后平均还要跑 200 步，500 步上限下实测
  # 37.2% 的回合撞上限；延到 1000 步后失败回合的截断率降到 5.8%。
  cfg.episode_length_s = EPISODE_LENGTH_S

  if play:
    cfg.episode_length_s = int(1e9)
    # 必须遍历所有组：RGMT actor 吃的是 rg_* 组，只关 actor 组等于没关
    for group in cfg.observations.values():
      group.enable_corruption = False
    cfg.events.pop("push_robot", None)
    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, GeneralMotionCommandCfg)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.joint_position_range = (0.0, 0.0)
    motion_cmd.sampling_mode = "start"
    motion_cmd.speed_range = (1.0, 1.0)  # 回放时固定原速，便于与参考动作对照
    motion_cmd.debug_vis = True  # 参考动作画成 ghost，便于肉眼对照

  return cfg
