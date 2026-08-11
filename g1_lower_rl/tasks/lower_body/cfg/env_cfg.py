"""下肢行走任务的基础环境配置（与机器人无关的部分）。

策略控制 12 个腿关节加 3 个腰关节，跟随 ``[vx, vy, omega, height]`` 四个指令。
各 manager 的配置拆在同目录的模块里，这里只负责组装。带 ``按机器人设置`` 注释的字段
留空，由 ``robots/<name>.py`` 填。
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from g1_lower_rl.tasks.lower_body import mdp
from g1_lower_rl.tasks.lower_body.cfg.actions import make_actions, make_commands
from g1_lower_rl.tasks.lower_body.cfg.curriculum import make_curriculum
from g1_lower_rl.tasks.lower_body.cfg.events import make_events
from g1_lower_rl.tasks.lower_body.cfg.observations import make_observations
from g1_lower_rl.tasks.lower_body.cfg.rewards import make_rewards
from g1_lower_rl.tasks.lower_body.cfg.terminations import make_terminations


def make_lower_body_env_cfg() -> ManagerBasedRlEnvCfg:
  """构造下肢行走的基础配置（平地）。"""
  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1,
      extent=2.0,
    ),
    observations=make_observations(),
    actions=make_actions(),
    commands=make_commands(),
    events=make_events(),
    rewards=make_rewards(),
    terminations=make_terminations(),
    curriculum=make_curriculum(),
    # 该指标同时是所有 rank 每步都会进入的课程同步钩子，不要换回 mjlab 原函数。
    metrics={
      "mean_action_acc": MetricsTermCfg(func=mdp.synchronized_mean_action_acc)
    },
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # 按机器人设置。
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
    ),
    decimation=4,
    episode_length_s=20.0,
  )
