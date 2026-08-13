"""奖励配置。

分四组：跟随（速度/转向/高度）、姿态、正则、步态。每一项的权重都是撞出来的，改之前
先读注释里的实测数据。
"""

from __future__ import annotations

import math
import re

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from g1_lower_rl.assets import LOWER_BODY_EFFORT_LIMIT, WAIST_JOINTS
from g1_lower_rl.tasks.lower_body import mdp
from g1_lower_rl.tasks.lower_body.cfg.constants import (
  FOOT_SITES,
  HEIGHT_STD_STAGES,
  LATERAL_JOINT_EXPR,
  LOWER_BODY_JOINT_EXPR,
  SELF_COLLISION_SENSOR,
  joints,
)

# 力矩上限按执行器模式索引、覆盖全部 15 个下肢关节，整份喂给只含腰三轴的项会让
# resolve_matching_names_values 因为腿的模式一个也匹配不上而报错。
WAIST_EFFORT_LIMIT: dict[str, float] = {
  _expr: _limit
  for _expr, _limit in LOWER_BODY_EFFORT_LIMIT.items()
  if any(re.fullmatch(_expr, _name) for _name in WAIST_JOINTS)
}


def _tracking_rewards() -> dict[str, RewardTermCfg]:
  return {
    # 故意压过高度奖励。高度跟随几乎立刻就饱和（站着不动就能满足），权重相等的话
    # 策略的最优选择就是站着撑高度而不是去走。
    #
    # ``std_scale`` / ``std_knee``：指令超过 0.8 m/s 后核宽线性放宽。固定核宽下，大指令处
    # “站着”与“走到可达速度”的奖励落差低于走路的净成本，走路变成纯亏损。拐点以下不变。
    #
    # 2.5 拆成 2.0（瞬时）+ 0.8（平均），合计不变。z 速度惩罚和大指令放宽核只在瞬时项里，
    # 所以瞬时项仍占大头。
    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=2.0,
      params={
        "command_name": "twist",
        "std": math.sqrt(0.2),
        "z_penalty": 1.5,
        "std_scale": 1.0,
        "std_knee": 0.8,
      },
    ),
    # 平均项：指令重采后累计的平均速度，等价于“这段窗口实际走了多远”。指令游走删掉之后
    # 窗口内指令恒定，这个量才有意义。std 0.30 比瞬时项的 0.447 紧 33%（实测平均误差
    # p50 = 0.240），ramp_time 让阶跃后的加速暂态几乎不收费。
    "track_linear_velocity_avg": RewardTermCfg(
      func=mdp.track_linear_velocity_avg,
      weight=0.8,
      params={
        "command_name": "twist",
        "std": 0.30,
        "ramp_time": 1.5,
        "command_threshold": 0.1,
      },
    ),
    # 转向跟随拆成两项。瞬时项：std 放宽到 0.7，主要作用是压住 yaw 率振荡
    # （实测直行时 σ 能到 0.77，标杆是 0.47）。它对直流跟踪的区分度只有 4.8×。
    "track_angular_velocity": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=0.5,
      params={"command_name": "twist", "std": 0.7},
    ),
    # 平均项：从指令重采时刻开始累计平均角速度，专管直流跟踪。2.5 -> 2.0 -> 1.6。
    # 2.0 实测把转向拉到了压过直行的地步：yaw 误差确实降了 23~25%，但线速度跟随学不起来
    # ——转向合计 2.5 已经和 track_linear_velocity 持平，策略优先买更便宜的转向。
    # std 保持 0.5：收得更紧会掉进“误差大到没梯度”的平地。
    "track_angular_velocity_avg": RewardTermCfg(
      func=mdp.track_angular_velocity_avg,
      weight=1.6,
      params={"command_name": "twist", "std": 0.5, "command_threshold": 0.1},
    ),
    # 原地转弯/站立时的位移。速度项在 vx=vy=0 时罚的是速度不是位置，慢漂几乎免费。
    #
    # std=0.2：5 cm 只罚 0.03（步态本身的落脚散布，不该收费），20 cm 罚 0.32，
    # 40 cm 罚 0.49（有界核封顶 0.5）。权重 -0.5 明显低于转向合计 2.1，避免出现
    # “不转就不漂”的退化解。
    "stationary_drift": RewardTermCfg(
      func=mdp.stationary_drift,
      weight=-0.5,
      params={"command_name": "twist", "std": 0.2},
    ),
    # 上面那个指数核只在捕获域内有梯度，指令一大“原地不动”就落在平地上。这一项补那段
    # 死区：线性，梯度恒等于权重。目标取指令的 cap_frac 倍，不要求全速。
    #
    # 权重必须轻。实测 -0.3 会把策略带到“蹭地拖着走”：它奖励沿指令方向的速度而不关心
    # 怎么得来，而抬脚惩罚按脚的水平速度加权（脚不动就不罚），两者合起来使拖行最便宜。
    # 当前取值下单拍峰值约 0.06，只当预防性的兵。
    "velocity_shortfall": RewardTermCfg(
      func=mdp.velocity_shortfall,
      weight=-0.08,
      params={"command_name": "twist", "cap_frac": 0.5, "command_threshold": 0.1},
    ),
    "track_base_height": RewardTermCfg(
      func=mdp.track_base_height,
      weight=0.8,
      params={
        "command_name": "height",
        # 【由 height_std 课程接管】这里只是第一档的初值。
        "std": HEIGHT_STD_STAGES[0][1],
        "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITES),
      },
    ),
  }


def _posture_rewards() -> dict[str, RewardTermCfg]:
  return {
    # 静止且高度指令达到 0.78 m 以上时（已经是完全直腿 0.7919 m 前的最后一档），
    # 额外奖励膝关节接近伸直；运动时不约束步态所需的屈膝。
    "straight_knee": RewardTermCfg(
      func=mdp.straight_knee,
      weight=0.5,
      params={
        "command_name": "height",
        "movement_command_name": "twist",
        "height_threshold": 0.78,
        "command_threshold": 0.1,
        "std": 0.5,
        "asset_cfg": joints(r".*_knee_joint"),
      },
    ),
    # 保持躯干站直，这是上层手臂策略需要的。
    # 翻倍会把正常步态需要的那点躯干俯仰也一并压掉——它罚的是 sin²(倾角)，前后对称，
    # 实测走起来之后姿态正常的策略在这一项上反而付 1.5~4 倍（因为它前倾 6°）。
    # 要治后仰请用下面那个单边项。
    #
    # ``lateral_scale=2.0``：前后分量有 ``pelvis_backward_lean`` 单边管着，左右却只有
    # 这一项。实测策略把精度全放在前后、把代价推给了左右：站立时前后只偏 0.13°
    # 而左右歪 11.4°（iter 9000）。放大 2 倍后站立单拍代价 0.039 -> 0.078，约为同期
    # 扭腰惩罚的一半；而走路时左右本来就只有 1.7°（代价 0.0022），几乎不受影响。
    "body_orientation_l2": RewardTermCfg(
      func=mdp.body_orientation_l2,
      weight=-1.0,
      params={
        "lateral_scale": 2.0,
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
      },
    ),
    # 只罚骨盆后仰、且留 3° 死区。补的是姿态类里的一个真空：没有任何项约束骨盆俯仰，
    # 骨盆整体后坐时腰关节会把偏差吸收掉，``body_orientation_l2``（量 torso_link）看不见。
    #
    # 后仰侧线性、前倾侧平方，且死区分开给。后仰处的平方形式是二阶小量，推不动姿态；
    # 而骨盆一旦平铺在后仰死区边上，机体系 vx 就会出现一个恒定后退偏置。前倾侧不能跟着
    # 改成线性：正常步态的前倾随速度递增，会把好姿态一起罚上。
    "pelvis_backward_lean": RewardTermCfg(
      func=mdp.backward_lean,
      weight=-2.0,
      params={
        "deadband_deg": 3.0,
        "forward_scale": 0.2,
        "forward_deadband_deg": 6.0,
        "asset_cfg": SceneEntityCfg("robot", body_names=("pelvis",)),
      },
    ),
    # 腰关节绝对角约束。必须有：防止撅屁股
    "waist_deviation": RewardTermCfg(
      func=mdp.joint_deviation_l2,
      weight=-1.0,
      params={
        "asset_cfg": joints(*WAIST_JOINTS),
        "weights": (0.4, 0.85, 1.0),  # 顺序同上：扭腰 / 左右 / 前后
      },
    ),
    # 叠在上面那项之上，只在站立时生效。上面给 waist_yaw 的 0.4 是**走路专用**的折扣：
    # 它要与摆动腿反向旋转抵消角动量，压死会把 20 s 直行累计航向推到 99.8°。站立没有
    # 摆动腿，那个理由不成立，这个自由度就成了几乎免费的。
    "waist_deviation_still": RewardTermCfg(
      func=mdp.joint_deviation_l1,
      weight=-0.3,
      params={
        "asset_cfg": joints(*WAIST_JOINTS),
        "weights": (1.0, 1.0, 0.5),
        "command_name": "twist",
        "command_threshold": 0.1,
      },
    ),
    # 电机发热主要随持续电流平方增长，而位置偏差不能直接量到控制器为了维持姿态用了多少力。
    # 这里对三个腰电机的实际力矩按各自额定值归一化后平方求和；角度 L1 仍保留，避免策略
    # 单纯靠侧歪把上身质心挪到关节轴线上来卸力。只在无运动指令时收费，不压正常步态和转向。
    "waist_effort_still": RewardTermCfg(
      func=mdp.normalized_joint_effort_l2,
      weight=-1.0,
      params={
        "asset_cfg": joints(*WAIST_JOINTS),
        "effort_limits": WAIST_EFFORT_LIMIT,
        "command_name": "twist",
        "command_threshold": 0.1,
      },
    ),
    # 站住时把大腿旋转拉回 0，防内外八字
    #
    # pose 那一项虽然也管 hip_yaw，但它是 exp(-mean(误差²/std²))，**在 6 个关节上取平均**：
    # 两条腿各偏 7° 只让它从 0.5 掉到 0.33，单拍代价 0.17，太软。实测当前策略站立时
    # 左 -6.4° / 右 +8.1°（均为内八），而标杆是 -0.4° / 0.0°，几乎正前。
    #
    # 权重 -5.0：偏 3° 单拍罚 0.027，7° 罚 0.15（与 pose 自身损失同量级），15° 罚 0.69。
    # 只在“没叫它动”时收费——走路和原地转弯都需要这个关节参与。
    "hip_yaw_still": RewardTermCfg(
      func=mdp.joint_deviation_l2,
      weight=-5.0,
      params={
        "asset_cfg": joints(r".*_hip_yaw_joint"),
        "command_name": "twist",
        "command_threshold": 0.1,
      },
    ),
    # 只管那些不该动的关节——矢状面那条链承担蹲起，由 track_base_height 管。这里的 std
    # 看起来比 velocity 任务松，但那个奖励是在 29 个关节上取平均，这个只在 6 个上取，
    # 同样的偏差在这里罚得重约 5 倍。这里收得太紧会阻止抬脚所需的侧向重心转移，
    # 策略的回应就是干脆不迈步。
    "pose": RewardTermCfg(
      func=mdp.variable_posture,
      weight=0.5,
      params={
        "asset_cfg": joints(*LATERAL_JOINT_EXPR),
        "command_name": "twist",
        "std_standing": {k: 0.1 for k in LATERAL_JOINT_EXPR},
        "std_walking": {
          r".*_hip_roll_joint": 0.35,
          r".*_hip_yaw_joint": 0.3,
          r".*_ankle_roll_joint": 0.2,
        },
        "std_running": {
          r".*_hip_roll_joint": 0.5,
          r".*_hip_yaw_joint": 0.4,
          r".*_ankle_roll_joint": 0.25,
        },
        "walking_threshold": 0.1,
        "running_threshold": 1.4,
      },
    ),
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=-0.05,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",))},
    ),
    # 传感器已排掉手臂（见 env_cfg 里的传感器定义），好行为的代价恰好是 0，所以不再需要“压轻
    # 以免淹掉速度跟随”——从 -0.5 提到 -2.0。算一下量级：原始值是一个控制拍内接触力超
    # 10 N 的子步数（0~4），所以满接触的一拍要罚 2.0 × 4 × dt = 0.16，是同一拍
    # 线速度跟随满分（2.5 × dt = 0.05）的 3.2 倍——足够威慑，但远没到一撞就完。
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-2.0,
      params={"sensor_name": SELF_COLLISION_SENSOR, "force_threshold": 10.0},
    ),
  }


def _regularization_rewards() -> dict[str, RewardTermCfg]:
  return {
    # 这一项和腰部那组一样，属于“早期需要宽、后期需要紧”的量，真要再动它应该做成课程
    # 而不是改常量——参照 height_std 课程的做法。
    "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-120.0),
    # 限在受控关节上：手臂是故意被晃的，夹爪又贴在硬限位上，两者都不应该反馈给策略。
    "joint_acc_l2": RewardTermCfg(
      func=mdp.joint_acc_l2,
      weight=-2.5e-7,
      params={"asset_cfg": joints(*LOWER_BODY_JOINT_EXPR)},
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.02),
    # 叠在上面那项之上，只在静止指令下生效，所以站着不动时总权重是 -0.1。
    "action_rate_still": RewardTermCfg(
      func=mdp.action_rate_still,
      weight=-0.08,
      params={"command_name": "twist", "command_threshold": 0.1},
    ),
  }


def _gait_rewards() -> dict[str, RewardTermCfg]:
  return {
    # 唯一一个双脚支撑时恰好为零、只在单脚支撑时为正的项。没有它策略会收敛成雕像：
    # 站着不动就能拿满高度奖励加 56% 的 foot_gait，而所有运动惩罚都是零，迈出第一步
    # 严格更差。
    #
    # 2.0 -> 3.0：实测 1500 迭代时它只拿到 0.0334（满分 0.8，达成 4.2%），而 foot_gait
    # 在运动环境拿到 0.277，几乎正好等于“双脚全程不离地”的理论值 0.28——策略在蹭着走。
    "foot_air_time": RewardTermCfg(
      func=mdp.feet_air_time,
      weight=3.2,
      params={
        "sensor_name": "feet_ground_contact",
        "threshold": 0.4,
        "command_name": "twist",
        "command_threshold": 0.1,
      },
    ),
    "foot_gait": RewardTermCfg(
      func=mdp.shifted_feet_gait,
      weight=1.0,
      params={
        "period": 0.6,
        "offset": [0.0, 0.5],
        "threshold": 0.56,
        "command_threshold": 0.1,
        "command_name": "twist",
        "sensor_name": "feet_ground_contact",
      },
    ),
    # 用相对基准而不是世界 z：地形有起伏之后，绝对高度会把地面本身算进目标里。
    "foot_clearance": RewardTermCfg(
      func=mdp.feet_clearance_relative,
      weight=-1.2,
      params={
        "target_height": 0.11,
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITES),
      },
    ),
    # 权重给得很重：拖着脚滑是策略替代“真的走”的主要选项。它同样能满足速度指令，却避开了
    # 真迈一步要付的所有代价，所以按 velocity 任务的权重它会完胜。
    # 只管有运动指令的情形；静止时的蹭地由下面的 feet_slip_still 接管。
    "foot_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=-2.0,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITES),
      },
    ),
    "soft_landing": RewardTermCfg(
      func=mdp.soft_landing,
      weight=-2e-3,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.1,
      },
    ),
    # foot_air_time 和 foot_gait 奖励的是“迈步”而不是“位移”，所以原地踏步能把它们满额拿走。
    # 这一项让没叫你动的时候踏步徒劳无益，同时保持足够便宜，使得被推后迈一步仍然划算。
    "stand_still_feet": RewardTermCfg(
      func=mdp.feet_stationary,
      weight=-0.5,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.1,
      },
    ),
    # 变宽本身是好策略，但不允许蹭着地变宽。上游 foot_slip 只在有运动指令时生效，
    # 静止时蹭地免费，实测 16 s 内两脚间距棘轮式漂移 +10 cm（非双支撑帧仅 1%）。
    #
    # 权重按“蹭地必须比迈步贵”反推。L1 的时间积分就是路径长度：蹭出 10 cm 间距等于
    # 两脚各走 5 cm，代价 = 权重 × 0.10；抬脚一步（stand_still_feet，单脚腾空 0.3 s）
    # 约 0.15。所以 -1.5 只是打平，取 -4.0 留出 2.7 倍差价。
    # 代价是 L1 在残余抖动处不归零：站定后 Σ|v| 约 0.01，单拍还剩 8e-4。
    "feet_slip_still": RewardTermCfg(
      func=mdp.feet_slip_still,
      weight=-4.0,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITES),
      },
    ),
  }


def make_rewards() -> dict[str, RewardTermCfg]:
  return {
    **_tracking_rewards(),
    **_posture_rewards(),
    **_regularization_rewards(),
    **_gait_rewards(),
  }
