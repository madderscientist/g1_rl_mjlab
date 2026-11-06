"""Render the official MLP in native MuJoCo, not a WM imagination."""

import os

import tyro

from g1_lower_rl.tasks.motion_tracking.training.playback import render

os.environ.setdefault("MUJOCO_GL", "egl")

if __name__ == "__main__":
  tyro.cli(render)
