#!/usr/bin/env python
"""Pack mocap without changing source files or materializing the full corpus."""

import tyro

from g1_lower_rl.tasks.motion_tracking.corpus import PrepareMocapConfig, prepare_mocap

if __name__ == "__main__":
  prepare_mocap(tyro.cli(PrepareMocapConfig))
