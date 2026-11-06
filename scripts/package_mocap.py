"""Preserve all ranks and the exact runtime of an official MLP result."""

import tyro

from g1_lower_rl.tasks.motion_tracking.training.export import package_result

if __name__ == "__main__":
  tyro.cli(package_result)
