#!/usr/bin/env python
"""Footstep tracking with Twin-Mirror history MLP + Position WM v3 + Auto-Curriculum."""

import tyro

from g1_lower_rl.tasks.footstep_tracking.wm import FootstepWmConfig, run_footstep_wm_training


def main() -> None:
  run_footstep_wm_training(tyro.cli(FootstepWmConfig))


if __name__ == "__main__":
  main()
