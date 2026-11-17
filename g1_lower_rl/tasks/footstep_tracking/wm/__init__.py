"""Footstep tracking with Twin-Mirror history MLP and differentiable Position WM v3."""

from __future__ import annotations

FOOTSTEP_WM_TASK = "G1-Gloria-FootstepTracking-WM"

from .playback import PlayConfig, export_onnx, run_play
from .runner import DEFAULT_PRETRAINED_WM, FootstepWmConfig, run_footstep_wm_training
from .task import AUTO_CURRICULUM_STAGES, footstep_env_cfg_wm

__all__ = [
  "AUTO_CURRICULUM_STAGES",
  "DEFAULT_PRETRAINED_WM",
  "FOOTSTEP_WM_TASK",
  "FootstepWmConfig",
  "PlayConfig",
  "export_onnx",
  "footstep_env_cfg_wm",
  "run_footstep_wm_training",
  "run_play",
]
