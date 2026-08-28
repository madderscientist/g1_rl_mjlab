"""GMT 任务的 mdp 项：沿用 mjlab tracking 的奖励/终止/观测，只替换命令项。

奖励和终止全部按 ``MotionCommand`` 的属性名取值，而 ``GeneralMotionCommand`` 刻意
保持了同一套属性名，所以这里直接整体转出即可，不需要重写。
"""

from mjlab.tasks.tracking.mdp import *

from g1_lower_rl.tasks.motion_tracking.mdp.actions import (
  GravityCompensatedJointPositionActionCfg as GravityCompensatedJointPositionActionCfg,
)
from g1_lower_rl.tasks.motion_tracking.mdp.commands import (
  GeneralMotionCommand as GeneralMotionCommand,
)
from g1_lower_rl.tasks.motion_tracking.mdp.commands import (
  GeneralMotionCommandCfg as GeneralMotionCommandCfg,
)
from g1_lower_rl.tasks.motion_tracking.mdp.events import (
  payload_mass as payload_mass,
)
from g1_lower_rl.tasks.motion_tracking.mdp.motion_corpus import MotionCorpus as MotionCorpus
from g1_lower_rl.tasks.motion_tracking.mdp.observations import (
  motion_reference_window as motion_reference_window,
)
from g1_lower_rl.tasks.motion_tracking.mdp.rewards import (
  motion_anchor_lin_vel_error_exp as motion_anchor_lin_vel_error_exp,
)
from g1_lower_rl.tasks.motion_tracking.mdp.rewards import (
  motion_ee_pos_torso_relative_exp as motion_ee_pos_torso_relative_exp,
)
from g1_lower_rl.tasks.motion_tracking.mdp.rewards import (
  motion_swing_lift_ratio as motion_swing_lift_ratio,
)
from g1_lower_rl.tasks.motion_tracking.mdp.terminations import (
  with_probabilistic_termination as with_probabilistic_termination,
)
