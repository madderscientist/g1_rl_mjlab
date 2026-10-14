"""训练、部署和预览共用的策略接口常量，不依赖训练或推理库"""

CONTRACT_VERSION = "g1_footstep_gru_v7_torso_heading_command"
PREVIEW_FORMAT = "footstep-preview-v4"
RAW_OBS_DIM = 79
ENCODED_OBS_DIM = 82
ACTION_DIM = 15
FOOTSTEP_SLOTS = ("L_next", "R_next")
PLANNER_FOOTSTEP_SLOTS = ("L_support", "R_support", "L_next", "R_next")