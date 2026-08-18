"""全身动作跟踪的 PPO 配置。"""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

from g1_lower_rl.rl.rgmt_model import RgmtModelCfg
from g1_lower_rl.tasks.motion_tracking.env_cfg import RGMT_PROP_TERMS

RGMT_OBS_GROUPS: dict[str, tuple[str, ...]] = {
  "actor": tuple(f"rg_{n}" for n in RGMT_PROP_TERMS) + ("rg_actions", "rg_reference"),
  "critic": ("critic",),
}


def motion_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """actor 用 RGMT 架构（Extreme-RGMT 第 IV-A 节）。

  三分支独立编码 + 逐分支 LayerNorm + 因果历史编码器 + cross-attention + FSQ 瓶颈。
  取代原来的 GRU：GRU 只能把历史压进一个隐状态，而 cross-attention 能按当前状态
  从局部参考窗口里**挑**相关帧——高动态动作里相位偏一点，该看的参考帧就完全不同。

  ``obs_normalization`` 必须关：RGMT 用逐分支 LayerNorm 代替经验归一化，后者的
  running statistics 在 Stage II 换语料后会漂。
  """
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
      reference_dim=38,
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
    # 论文写 24，但那是配它自己的环境数定的资源配比，不是架构主张。这里保持 48：
    # 更长的 rollout 让 GAE 少一次 bootstrap、偏差更小，且与上一轮 GRU 基线逐项可比。
    num_steps_per_env=48,
    max_iterations=140001,
  )
