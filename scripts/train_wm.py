#!/usr/bin/env python
"""Bounded native-position WM pretraining (single process or torchrun DDP)."""

import os

if int(os.environ.get("WORLD_SIZE", "1")) > 1 and os.environ.get("WM_GPU_PINNED") != "1":
  local_rank = int(os.environ["LOCAL_RANK"])
  visible = os.environ.get("CUDA_VISIBLE_DEVICES")
  devices = visible.split(",") if visible is not None else [str(i) for i in range(int(os.environ["LOCAL_WORLD_SIZE"]))]
  os.environ["CUDA_VISIBLE_DEVICES"] = devices[local_rank]
  os.environ["WM_GPU_PINNED"] = "1"
  os.environ["SM_GPU_PINNED"] = "1"

import tyro

from g1_lower_rl.world_model.training import TrainingConfig, run_training


def main() -> None:
  run_training(tyro.cli(TrainingConfig))


if __name__ == "__main__":
  main()
