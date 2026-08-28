"""全身动作跟踪的 PPO 配置。"""

import os

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
)

from g1_lower_rl.rl.rgmt_model import RgmtModelCfg
from g1_lower_rl.rl.runner import ScheduledPpoAlgorithmCfg
from g1_lower_rl.tasks.motion_tracking.env_cfg import (
  RGMT_PROP_TERMS,
  resolve_key_body_vel,
  resolve_reference_key_bodies,
)

RGMT_OBS_GROUPS: dict[str, tuple[str, ...]] = {
  "actor": tuple(f"rg_{n}" for n in RGMT_PROP_TERMS) + ("rg_actions", "rg_reference"),
  "critic": ("critic",),
}

GRU_OBS_GROUPS: dict[str, tuple[str, ...]] = {"actor": ("actor",), "critic": ("critic",)}


def _reference_dim() -> int:
  """token 维度必须与 ``reference_tokens`` 实际产出一致，否则模型 reshape 会算错 token 数。"""
  per_body = 6 if resolve_key_body_vel() else 3
  return 38 + len(resolve_reference_key_bodies()) * per_body


def motion_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """actor 用 RGMT 架构（Extreme-RGMT 第 IV-A 节）。

  三分支独立编码 + 逐分支 LayerNorm + 因果历史编码器 + cross-attention + FSQ 瓶颈。
  取代原来的 GRU：GRU 只能把历史压进一个隐状态，而 cross-attention 能按当前状态
  从局部参考窗口里**挑**相关帧——高动态动作里相位偏一点，该看的参考帧就完全不同。

  ``obs_normalization`` 必须关：RGMT 用逐分支 LayerNorm 代替经验归一化，后者的
  running statistics 在 Stage II 换语料后会漂。
  """
  if os.environ.get("GRU_ACTOR", "0") == "1":
    return _gru_runner_cfg()
  return RslRlOnPolicyRunnerCfg(
    actor=RgmtModelCfg(
      class_name="g1_lower_rl.rl.rgmt_model:RgmtActor",
      hidden_dims=(1024, 1024, 512, 256),
      activation="elu",
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
      history_len=10,
      reference_dim=_reference_dim(),
      token_dim=64,
      num_heads=4,
      history_layers=2,
      fsq_levels=5,
    ),
    critic=RslRlModelCfg(
      # critic 吃特权观测且不上机，保持 MLP。
      hidden_dims=(1024, 1024, 512, 512),
      activation="elu",
      obs_normalization=True,
    ),
    obs_groups=RGMT_OBS_GROUPS,
    algorithm=ScheduledPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      # 开概率终止后必须调小：倒地样本把跟踪奖励每步摊薄 58%，优势信号变弱，
      # 而熵项是固定系数，相对权重从 c8 的 5.6 倍策略梯度涨到 10.5 倍，
      # σ 随之单调发散（200 iter 涨 0.035，0.263->0.357 无收敛迹象）。
      # 按比例回压到 c8 那个稳定档位。探索够不够不靠 σ——恢复行为的探索来自
      # 失败态本身进了 rollout。
      entropy_coef=0.0025,
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
    # 论文写 24，但那是配它自己的环境数定的资源配比，不是架构主张。这里保持 48：
    # 更长的 rollout 让 GAE 少一次 bootstrap、偏差更小，且与上一轮 GRU 基线逐项可比。
    num_steps_per_env=48,
    max_iterations=140001,
  )


def _gru_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """GRU actor（架构对比用），取值照搬 53c97e6 换 GRU 那次。

  与 RGMT 版的唯一差别是 actor：观测走单一 ``actor`` 组（GRU 靠隐状态自己记时序，
  再喂 10 帧显式历史就不是干净对比了），奖励/终止/语料/算法超参全部相同。
  """
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      class_name="RNNModel",
      rnn_type="gru",
      # 比下肢版（32）宽一倍：要辨识的量更多（负载 + 补偿器增益 + 接触）。
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
      hidden_dims=(1024, 512, 256),
      activation="elu",
      obs_normalization=True,
    ),
    obs_groups=GRU_OBS_GROUPS,
    algorithm=ScheduledPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.0025,
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
    num_steps_per_env=48,
    max_iterations=140001,
  )
