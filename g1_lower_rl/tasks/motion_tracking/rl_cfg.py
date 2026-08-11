"""全身动作跟踪的 PPO 配置。"""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def motion_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      # 比单动作跟踪宽一档：要在同一组权重里塞下整个语料的动作流形。
      hidden_dims=(1024, 512, 256),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(1024, 512, 256),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_gloria_gmt",
    logger="tensorboard",
    save_interval=200,
    num_steps_per_env=24,
    # 语料从 10 段扩到 80 段真实动捕（含镜像），涵盖行走/跑步/冲刺/跳跃/格斗/舞蹈/
    # 摔倒起身，要学的动作流形大了一个量级，迭代上限跟着放大。收敛看
    # Train/mean_episode_length 与 Metrics/motion/error_body_pos 是否进平台期。
    max_iterations=60001,
  )
