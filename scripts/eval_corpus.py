"""Evaluate the official MLP in native MuJoCo on held-out mapped mocap clips."""

import tyro

from g1_lower_rl.tasks.motion_tracking.training.playback import evaluate

if __name__ == "__main__":
  tyro.cli(evaluate)
