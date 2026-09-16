"""脚步 GRU 的 PPO 配置，不复用速度任务专属增强或课程"""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from g1_lower_rl.rl.footstep_model import FootstepModelCfg


def footstep_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """建立 64 拍 recurrent rollout 和普通 PPO，参数尚未经步行训练调优"""
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
    # 64拍覆盖慢频率下的一整个左右周期，保持默认32维GRU隐藏状态
    num_steps_per_env=64,
    max_iterations=10001,
    save_interval=100,
    logger="tensorboard",
    # 动作幅度由15轴比例映射决定，不额外改变已约定的部署控制公式
    clip_actions=None,
  )
