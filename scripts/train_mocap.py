#!/usr/bin/env python
"""Official full-body MLP + position-WM SuperMimic training, single GPU or torchrun."""

import tyro

from g1_lower_rl.tasks.motion_tracking.training import TrainingConfig, run_training

if __name__ == "__main__":
  run_training(tyro.cli(TrainingConfig))
