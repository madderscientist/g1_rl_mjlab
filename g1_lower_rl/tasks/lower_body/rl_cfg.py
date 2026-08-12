"""下肢行走任务的 PPO 配置。"""

import math

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg

from g1_lower_rl.rl import ScheduledPpoAlgorithmCfg
from g1_lower_rl.tasks.lower_body.cfg.constants import GRU_NUM_STEPS_PER_ENV


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
    algorithm=ScheduledPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      # MLP 版保持 0.01：实测 σ 在第 1200 迭代触底 0.704，然后爬到 0.79 就停住了
      # （3600 迭代 0.788 -> 9800 迭代 0.793，跨 6200 迭代纹丝不动），跑满全程站立完好。
      # 伤害阈值大致在 0.79 和 1.05 之间，0.79 从未进入侵蚀区。
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
  # **同样的 0.01，MLP 的 σ 能稳在 0.79，GRU 的停不下来。** 实测 GRU 从第 400 迭代的
  # 0.75 单调涨到 4600 的 1.08（4200 迭代没回过头），同期 Mean entropy loss 贡献 0.217
  # 而 surrogate loss 只有 -0.0058，相差 37 倍。
  #
  # σ=1.05 时膝目标角每拍被注入 ±16°、踝 ±30°，精度类奖励被直接吃掉
  # （track_base_height -32%、track_linear_velocity -48%），而步态类持平（抬脚与否是二值
  # 判据，噪声改不了）——策略于是转去做噪声打不烂的事，**站立被系统性侵蚀**。
  # A/B（同从 model_4400 出发各跑 600 迭代，判据事先定死）：
  #
  #   0.01 ：三个站立场景全部 100% 摔（连起点最稳的蹲站也摔），走路仍正常
  #   0.002：站立摔倒率 0%，晃动 0.128->0.082、双手前伸 0.144->0.065、
  #         蹲站 0.013->0.002，走路 0.95（护栏 0.93）；σ 退火到 0.32
  #
  # 但 0.002 从零训练**学不会走路**：σ 在 200~800 迭代本来有一段由熵项推动的反弹
  # （0.60 -> 0.84），而那正好是学会走路的阶段（reward 3.8 -> 29.9），0.002 把它压死了。
  # 所以做成课程：前 4000 迭代用 0.01 把走路学出来，之后逐档降。
  #
  # **第三列的 σ 上限不能省。** 只改系数的话退火太慢：实测 0.01 -> 0.002 只把漂移改变
  # -3.2e-4/iter，σ 从 1.05 走到 0.35 要约 3200 迭代，而站立在 σ>1.0 时 300 迭代就崩。
  # A/B 里有效的那一臂是「改系数 + 压 σ + 清零它在 Adam 里的动量」三件事一起做的。
  # 上限只下压不上抬，``inf`` 表示这一档不动 σ。
  #
  # 别改用 ``distribution_cfg["std_range"]`` 去压 σ：``GaussianDistribution`` 里是
  # ``std_param.clamp(...)``，区间外梯度为零，一旦顶到上界 std_param 就被永久冻结、
  # σ 再也不退火，等价于写死 ``learn_std=False``。
  assert isinstance(cfg.algorithm, ScheduledPpoAlgorithmCfg)
  cfg.algorithm.entropy_stages = (
    (0, 0.01, math.inf),
    (4000, 0.004, 0.55),  # σ* 按两个已知点对数插值估计约 0.5
    (5000, 0.002, 0.35),  # σ* 实测 0.30~0.32
  )
  cfg.experiment_name = "g1_gloria_lower_body_gru"
  return cfg
