"""脚步 GRU 的 PPO 配置，不复用速度任务专属增强或课程"""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from g1_lower_rl.rl.footstep_model import FootstepModelCfg
from g1_lower_rl.tasks.footstep_tracking.curriculum import NUM_STEPS_PER_ENV


def footstep_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """单任务64拍recurrent PPO基础配置；联合入口显式覆盖学习率和探索。"""
  return RslRlOnPolicyRunnerCfg(
    actor=FootstepModelCfg(),
    critic=RslRlModelCfg(hidden_dims=(256, 128), activation="elu", obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      entropy_coef=0.01,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_footstep_tracking",
    # 64拍覆盖慢频率下的一整个左右周期
    num_steps_per_env=NUM_STEPS_PER_ENV,
    max_iterations=10001,
    save_interval=100,
    logger="tensorboard",
    # 动作幅度由15轴比例映射决定，不额外改变已约定的部署控制公式
    clip_actions=None,
  )


def step_episode_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """一步精度任务固定小幅探索，不影响通用脚步策略的PPO配置"""
  cfg = footstep_ppo_runner_cfg()
  cfg.algorithm.entropy_coef = 0.
  cfg.actor.distribution_cfg.update(init_std=.25, learn_std=False, std_range=(.25, .25))
  return cfg
