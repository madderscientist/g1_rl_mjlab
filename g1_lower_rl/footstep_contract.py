"""训练、部署和预览共用的策略接口常量，不依赖训练或推理库"""

CONTRACT_VERSION = "g1_footstep_gru_v5_next_only"
PREVIEW_FORMAT = "footstep-preview-v4"
RAW_OBS_DIM = 78
ENCODED_OBS_DIM = 81
ACTION_DIM = 15
FOOTSTEP_SLOTS = ("L_next", "R_next")
PLANNER_FOOTSTEP_SLOTS = ("L_support", "R_support", "L_next", "R_next")