"""机器人特化的下肢任务配置。每个模块负责把基础配置补成一个可注册的任务。"""

from g1_lower_rl.tasks.lower_body.robots.g1_gloria import (
  flat_env_cfg as g1_gloria_flat_env_cfg,
)
from g1_lower_rl.tasks.lower_body.robots.g1_gloria import (
  flat_env_cfg_gru as g1_gloria_flat_env_cfg_gru,
)
from g1_lower_rl.tasks.lower_body.robots.g1_gloria import (
  ppo_runner_cfg as g1_gloria_ppo_runner_cfg,
)
from g1_lower_rl.tasks.lower_body.robots.g1_gloria import (
  ppo_runner_cfg_gru as g1_gloria_ppo_runner_cfg_gru,
)

__all__ = [
  "g1_gloria_flat_env_cfg",
  "g1_gloria_flat_env_cfg_gru",
  "g1_gloria_ppo_runner_cfg",
  "g1_gloria_ppo_runner_cfg_gru",
]
