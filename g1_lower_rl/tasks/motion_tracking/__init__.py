"""全身动作跟踪（GMT）任务。注册在 g1_lower_rl.tasks 里，和其它任务同一处。"""

from g1_lower_rl.tasks.motion_tracking.env_cfg import (
  motion_tracking_env_cfg as motion_tracking_env_cfg,
)
from g1_lower_rl.tasks.motion_tracking.rl_cfg import (
  motion_tracking_ppo_runner_cfg as motion_tracking_ppo_runner_cfg,
)
