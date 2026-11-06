"""导入本模块即注册所有任务。"""

from mjlab.tasks.registry import list_tasks as list_mjlab_tasks
from mjlab.tasks.registry import register_mjlab_task

from g1_lower_rl.rl import GloriaOnPolicyRunner
from g1_lower_rl.tasks.footstep_tracking.env_cfg import footstep_env_cfg
from g1_lower_rl.tasks.footstep_tracking.rl_cfg import footstep_ppo_runner_cfg, step_episode_ppo_runner_cfg
from g1_lower_rl.tasks.footstep_tracking.step_episode import step_episode_env_cfg
from g1_lower_rl.tasks.lower_body import (
  flat_env_cfg,
  flat_env_cfg_gru,
  ppo_runner_cfg,
  ppo_runner_cfg_gru,
)
from g1_lower_rl.tasks.lower_body.cfg.constants import ITER
from g1_lower_rl.tasks.motion_tracking import MOCAP_TASK as MOCAP_TASK
from g1_lower_rl.tasks.standing import standing_env_cfg, standing_ppo_runner_cfg

# 脚本的默认任务。别再各自硬编码字符串——改过任务名之后漏改一处就会静默跑错环境。
DEFAULT_TASK = "G1-Gloria-LowerBody-Flat"
FOOTSTEP_TASK = "G1-Gloria-FootstepTracking"
STEP_EPISODE_TASK = "G1-Gloria-FootstepEpisode"

register_mjlab_task(
  task_id=FOOTSTEP_TASK,
  env_cfg=footstep_env_cfg(),
  play_env_cfg=footstep_env_cfg(play=True),
  rl_cfg=footstep_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id=STEP_EPISODE_TASK,
  env_cfg=step_episode_env_cfg(),
  play_env_cfg=step_episode_env_cfg(play=True),
  rl_cfg=step_episode_ppo_runner_cfg(),
)

# 课程档位是按 ITER 步/迭代换算成环境步数的，对不上的话课程会整体提前或推迟，且不报错。
assert ppo_runner_cfg().num_steps_per_env == ITER, (
  "改了 num_steps_per_env 就必须同步改 ITER，或者像 GRU 版那样缩放课程 step"
)

register_mjlab_task(
  task_id="G1-Gloria-LowerBody-Flat",
  env_cfg=flat_env_cfg(),
  play_env_cfg=flat_env_cfg(play=True),
  rl_cfg=ppo_runner_cfg(),
  runner_cls=GloriaOnPolicyRunner,
)

# 与上面同一个环境，只把 actor 换成 GRU，并去掉 actor 观测里的上一拍动作。
register_mjlab_task(
  task_id="G1-Gloria-LowerBody-Flat-GRU",
  env_cfg=flat_env_cfg_gru(),
  play_env_cfg=flat_env_cfg_gru(play=True),
  rl_cfg=ppo_runner_cfg_gru(),
  runner_cls=GloriaOnPolicyRunner,
)

register_mjlab_task(
  task_id="G1-Gloria-Stand",
  env_cfg=standing_env_cfg(),
  play_env_cfg=standing_env_cfg(play=True),
  rl_cfg=standing_ppo_runner_cfg(),
  runner_cls=GloriaOnPolicyRunner,
)

def list_tasks() -> list[str]:
  """Project tasks include SuperMimic, which deliberately has no PPO registry entry."""
  return sorted([*list_mjlab_tasks(), MOCAP_TASK])
