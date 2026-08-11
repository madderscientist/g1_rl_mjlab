"""下肢行走环境配置。按 manager 拆分，``env_cfg.make_lower_body_env_cfg`` 负责组装。"""

from g1_lower_rl.tasks.lower_body.cfg.curriculum import (
  max_out_curriculum as max_out_curriculum,
)
from g1_lower_rl.tasks.lower_body.cfg.env_cfg import (
  make_lower_body_env_cfg as make_lower_body_env_cfg,
)
