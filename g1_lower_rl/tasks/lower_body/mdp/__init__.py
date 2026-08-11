"""下肢任务的 MDP 项。

``mdp.<fn>`` 同时能解析到 mjlab 通用项（``mjlab.envs.mdp``）、mjlab 速度任务里可直接
复用的项，以及本任务自己的项。后者遮盖前者——``feet_air_time`` / ``body_orientation_l2``
在 mjlab 1.5.x 里语义变了或被删了，本包保留了原语义。
"""

from mjlab.envs.mdp import *  # noqa: F403
from mjlab.tasks.velocity.mdp import commands_vel as commands_vel

from g1_lower_rl.tasks.lower_body.mdp.commands import (
  BaseHeightCommand as BaseHeightCommand,
)
from g1_lower_rl.tasks.lower_body.mdp.commands import (
  BaseHeightCommandCfg as BaseHeightCommandCfg,
)
from g1_lower_rl.tasks.lower_body.mdp.commands import (
  ScenarioVelocityCommand as ScenarioVelocityCommand,
)
from g1_lower_rl.tasks.lower_body.mdp.commands import (
  ScenarioVelocityCommandCfg as ScenarioVelocityCommandCfg,
)
from g1_lower_rl.tasks.lower_body.mdp.curriculums import (
  commands_height as commands_height,
)
from g1_lower_rl.tasks.lower_body.mdp.curriculums import (
  gated_event_params as gated_event_params,
)
from g1_lower_rl.tasks.lower_body.mdp.curriculums import (
  tightening_height_std as tightening_height_std,
)
from g1_lower_rl.tasks.lower_body.mdp.events import (
  arm_torque_impulse as arm_torque_impulse,
)
from g1_lower_rl.tasks.lower_body.mdp.events import (
  disturbance_level as disturbance_level,
)
from g1_lower_rl.tasks.lower_body.mdp.events import gait_phase_offset as gait_phase_offset
from g1_lower_rl.tasks.lower_body.mdp.events import hold_arm_pose as hold_arm_pose
from g1_lower_rl.tasks.lower_body.mdp.events import (
  resample_disturbance_level as resample_disturbance_level,
)
from g1_lower_rl.tasks.lower_body.mdp.events import (
  resample_gait_phase as resample_gait_phase,
)
from g1_lower_rl.tasks.lower_body.mdp.events import (
  scaled_body_impulse as scaled_body_impulse,
)
from g1_lower_rl.tasks.lower_body.mdp.events import (
  scaled_reset_joints as scaled_reset_joints,
)
from g1_lower_rl.tasks.lower_body.mdp.events import (
  scaled_reset_root_state as scaled_reset_root_state,
)
from g1_lower_rl.tasks.lower_body.mdp.metrics import (
  synchronized_mean_action_acc as synchronized_mean_action_acc,
)
from g1_lower_rl.tasks.lower_body.mdp.observations import base_height as base_height
from g1_lower_rl.tasks.lower_body.mdp.observations import foot_air_time as foot_air_time
from g1_lower_rl.tasks.lower_body.mdp.observations import foot_contact as foot_contact
from g1_lower_rl.tasks.lower_body.mdp.observations import (
  foot_contact_forces as foot_contact_forces,
)
from g1_lower_rl.tasks.lower_body.mdp.observations import foot_height as foot_height
from g1_lower_rl.tasks.lower_body.mdp.observations import (
  height_above_feet as height_above_feet,
)
from g1_lower_rl.tasks.lower_body.mdp.observations import shifted_phase as shifted_phase
from g1_lower_rl.tasks.lower_body.mdp.rewards import *  # noqa: F403
from g1_lower_rl.tasks.lower_body.mdp.terminations import (
  base_height_below_minimum as base_height_below_minimum,
)
