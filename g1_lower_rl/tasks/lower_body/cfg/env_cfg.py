"""下肢行走任务的环境配置。

策略控制 12 个腿关节加 3 个腰关节，跟随 ``[vx, vy, omega, height]`` 四个指令。
各 manager 的配置拆在同目录的模块里，这里负责组装，并给出 MLP / GRU 两个入口。
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from g1_lower_rl.assets import get_robot_cfg
from g1_lower_rl.tasks.lower_body import mdp
from g1_lower_rl.tasks.lower_body.cfg.actions import make_actions, make_commands
from g1_lower_rl.tasks.lower_body.cfg.constants import (
  BODY_IMPULSE_LEVELS,
  FEET_SENSOR,
  GRU_NUM_STEPS_PER_ENV,
  ITER,
  SELF_COLLISION_SENSOR,
)
from g1_lower_rl.tasks.lower_body.cfg.curriculum import (
  make_curriculum,
  max_out_curriculum,
)
from g1_lower_rl.tasks.lower_body.cfg.events import make_events
from g1_lower_rl.tasks.lower_body.cfg.observations import make_observations
from g1_lower_rl.tasks.lower_body.cfg.rewards import make_rewards
from g1_lower_rl.tasks.lower_body.cfg.terminations import make_terminations


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
    name=SELF_COLLISION_SENSOR,
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


def make_lower_body_env_cfg() -> ManagerBasedRlEnvCfg:
  """构造下肢行走的基础配置（平地）。"""
  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1,
      extent=2.0,
      entities={"robot": get_robot_cfg()},
      sensors=(_feet_ground_sensor(), _self_collision_sensor()),
    ),
    observations=make_observations(),
    actions=make_actions(),
    commands=make_commands(),
    events=make_events(),
    rewards=make_rewards(),
    terminations=make_terminations(),
    curriculum=make_curriculum(),
    # 该指标同时是所有 rank 每步都会进入的课程同步钩子，不要换回 mjlab 原函数。
    metrics={"mean_action_acc": MetricsTermCfg(func=mdp.synchronized_mean_action_acc)},
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="torso_link",
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=None,
      njmax=300,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
        ccd_iterations=50,
      ),
      contact_sensor_maxmatch=64,
    ),
    decimation=4,
    episode_length_s=20.0,
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
