"""全身动作跟踪的 PPO 配置。"""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

GRU_NUM_STEPS_PER_ENV = 48
"""RNN 靠 BPTT 学时序，截断窗口要能跨过一个步态周期（24 拍 = 0.48 s 不够）。"""


def motion_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """actor 用 GRU：负载与补偿器误差都只能从「指令 vs 实际响应」的历史里推断。

  拼接 5 帧本体历史只给了 0.1 s 的窗口，而隐状态没有这个上限。下肢行走任务换 GRU
  后是压倒性优势（iter 9800 时 reward 50.3 vs 14.4），这里加了负载随机化之后，需要
  隐式在线辨识的成分更重。

  rsl_rl 5.4 的 ``RNNModel`` 自带 ONNX 导出，部署链路不用改。
  """
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      class_name="RNNModel",
      rnn_type="gru",
      # 比下肢版（32）宽一倍：这里要辨识的量更多（负载 + 补偿器增益 + 接触）。
      rnn_hidden_dim=64,
      rnn_num_layers=1,
      # GRU 之后的 MLP 收窄：隐状态已承担大部分容量，再堆宽只是更难训。
      hidden_dims=(512, 256),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      # critic 吃特权观测且不上机，保持 MLP。
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
    num_steps_per_env=GRU_NUM_STEPS_PER_ENV,
    # 语料从 10 段扩到 80 段真实动捕（含镜像），涵盖行走/跑步/冲刺/跳跃/格斗/舞蹈/
    # 摔倒起身，要学的动作流形大了一个量级，迭代上限跟着放大。收敛看
    # Train/mean_episode_length 与 Metrics/motion/error_body_pos 是否进平台期。
    max_iterations=60001,
  )
