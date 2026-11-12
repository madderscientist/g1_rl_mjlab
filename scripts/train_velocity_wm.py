#!/usr/bin/env python
"""Lower-body velocity tracking with MLP history + Position WM v3 (single GPU or torchrun)."""

import tyro

from g1_lower_rl.tasks.lower_body.wm import VelocityWmConfig, run_velocity_wm_training


def main() -> None:
  run_velocity_wm_training(tyro.cli(VelocityWmConfig))


if __name__ == "__main__":
  main()
