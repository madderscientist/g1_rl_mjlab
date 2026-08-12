"""下肢任务的奖励项。

按主题分三块：``tracking`` 跟随、``gait`` 步态与足部、``posture`` 姿态与关节。
纯粹沿用 mjlab 且语义未变的项（``feet_slip`` / ``soft_landing`` /
``body_angular_velocity_penalty`` / ``self_collision_cost`` / ``variable_posture``）
不再复制一份，直接从 ``mjlab.tasks.velocity.mdp`` 转出。
"""

from mjlab.tasks.velocity.mdp import (
  body_angular_velocity_penalty as body_angular_velocity_penalty,
)
from mjlab.tasks.velocity.mdp import feet_slip as feet_slip
from mjlab.tasks.velocity.mdp import self_collision_cost as self_collision_cost
from mjlab.tasks.velocity.mdp import soft_landing as soft_landing
from mjlab.tasks.velocity.mdp import variable_posture as variable_posture

from g1_lower_rl.tasks.lower_body.mdp.rewards.gait import (
  feet_air_time as feet_air_time,
)
from g1_lower_rl.tasks.lower_body.mdp.rewards.gait import (
  feet_clearance_relative as feet_clearance_relative,
)
from g1_lower_rl.tasks.lower_body.mdp.rewards.gait import (
  feet_stationary as feet_stationary,
)
from g1_lower_rl.tasks.lower_body.mdp.rewards.gait import (
  shifted_feet_gait as shifted_feet_gait,
)
from g1_lower_rl.tasks.lower_body.mdp.rewards.posture import (
  action_rate_still as action_rate_still,
)
from g1_lower_rl.tasks.lower_body.mdp.rewards.posture import (
  backward_lean as backward_lean,
)
from g1_lower_rl.tasks.lower_body.mdp.rewards.posture import (
  body_orientation_l2 as body_orientation_l2,
)
from g1_lower_rl.tasks.lower_body.mdp.rewards.posture import (
  joint_deviation_l1 as joint_deviation_l1,
)
from g1_lower_rl.tasks.lower_body.mdp.rewards.posture import (
  joint_deviation_l2 as joint_deviation_l2,
)
from g1_lower_rl.tasks.lower_body.mdp.rewards.posture import (
  normalized_joint_effort_l2 as normalized_joint_effort_l2,
)
from g1_lower_rl.tasks.lower_body.mdp.rewards.posture import (
  straight_knee as straight_knee,
)
from g1_lower_rl.tasks.lower_body.mdp.rewards.tracking import (
  stationary_drift as stationary_drift,
)
from g1_lower_rl.tasks.lower_body.mdp.rewards.tracking import (
  track_angular_velocity as track_angular_velocity,
)
from g1_lower_rl.tasks.lower_body.mdp.rewards.tracking import (
  track_angular_velocity_avg as track_angular_velocity_avg,
)
from g1_lower_rl.tasks.lower_body.mdp.rewards.tracking import (
  track_base_height as track_base_height,
)
from g1_lower_rl.tasks.lower_body.mdp.rewards.tracking import (
  track_linear_velocity as track_linear_velocity,
)
from g1_lower_rl.tasks.lower_body.mdp.rewards.tracking import (
  track_linear_velocity_avg as track_linear_velocity_avg,
)
from g1_lower_rl.tasks.lower_body.mdp.rewards.tracking import (
  velocity_shortfall as velocity_shortfall,
)
