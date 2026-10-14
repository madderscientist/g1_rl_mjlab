"""Expand a legacy NEXT-only joint checkpoint for the scalar theta command without training."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from g1_lower_rl.footstep_contract import CONTRACT_VERSION
from g1_lower_rl.tasks.footstep_tracking.commands import TorsoHeadingCommandCfg
from g1_lower_rl.tasks.footstep_tracking.curriculum import TORSO_HEADING_STAGES


def insert_feature(value, index, fill=0.0):
  feature = torch.full((*value.shape[:-1], 1), fill, dtype=value.dtype, device=value.device)
  return torch.cat((value[..., :index], feature, value[..., index:]), dim=-1)


def migrate_model(state, actor=False):
  width, index = (81, 81) if actor else (104, 84)
  weight = "rnn.rnn.weight_ih_l0" if actor else "mlp.0.weight"
  if state[weight].shape != ((192, 81) if actor else (256, 104)):
    raise ValueError("Expected the legacy 78/81 actor and 104-feature critic architecture")
  state[weight] = insert_feature(state[weight], index)
  for name, fill in (("_mean", 0.0), ("_var", 1.0), ("_std", 1.0)):
    key = "obs_normalizer." + name
    if state[key].shape != (1, width):
      raise ValueError("Unexpected legacy observation normalizer shape")
    state[key] = insert_feature(state[key], index, fill)


def migrate_optimizer(optimizer):
  parameters = [index for group in optimizer["param_groups"] for index in group["params"]]
  if len(parameters) != 19:
    raise ValueError("Expected the legacy joint actor/critic Adam parameter order")
  for position, shape, index in ((9, (192, 81), 81), (13, (256, 104), 84)):
    moments = optimizer["state"].get(parameters[position])
    if moments is None:
      raise ValueError("Missing legacy input-weight Adam state")
    for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
      if name not in moments:
        continue
      if moments[name].shape != shape:
        raise ValueError("Unexpected input-weight Adam moment shape or parameter order")
      moments[name] = insert_feature(moments[name], index)


def migrate_checkpoint(saved, source_sha256):
  infos = saved["infos"]
  if not infos.get("shared_actor") or infos.get("actor_raw_obs_dim") != 78:
    raise ValueError("Expected an unmigrated 78-feature shared-actor checkpoint")
  if set(saved["task_critic_state_dicts"]) != {"walking", "single_step"}:
    raise ValueError("Expected both private task critics")
  if set(saved["task_optimizer_state_dicts"]) != {"walking", "single_step"}:
    raise ValueError("Expected both private task optimizers")
  result = copy.deepcopy(saved)
  migrate_model(result["actor_state_dict"], actor=True)
  for state in (result["critic_state_dict"], *result["task_critic_state_dicts"].values()):
    migrate_model(state)
  for optimizer in (result["optimizer_state_dict"], *result["task_optimizer_state_dicts"].values()):
    migrate_optimizer(optimizer)
  defaults = TorsoHeadingCommandCfg()
  counters = list(infos["task_environment_counters"])
  if len(counters) != 2 or any(not isinstance(value, int) or value < 0 for value in counters):
    raise ValueError("Expected two nonnegative task counters for curriculum origins")
  result["infos"].update(
    actor_raw_obs_dim=79, actor_encoded_obs_dim=82, policy_contract_version=CONTRACT_VERSION,
    theta_stage={
      "task_start_steps": counters,
      "stages": [list(stage) for stage in TORSO_HEADING_STAGES],
      "resampling_time_range_s": list(defaults.resampling_time_range),
      "speed_range_rad_s": list(defaults.speed_range),
    },
    theta_migration={
      "source_sha256": source_sha256, "source_iteration": saved["iter"],
      "new_input_weights": "zero", "new_normalizer_mean_var_std": [0.0, 1.0, 1.0],
      "optimizer": "all old moments and step counters preserved; new columns zero",
    },
  )
  return result


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("source", type=Path)
  parser.add_argument("destination", type=Path)
  args = parser.parse_args()
  source = args.source.resolve(strict=True)
  destination = args.destination.resolve()
  if destination.exists() or source == destination:
    parser.error("Destination must be new; never overwrite the original checkpoint")
  digest = hashlib.sha256(source.read_bytes()).hexdigest()
  saved = torch.load(source, map_location="cpu", weights_only=False)
  migrated = migrate_checkpoint(saved, digest)
  destination.parent.mkdir(parents=True, exist_ok=True)
  with destination.open("xb") as output:
    torch.save(migrated, output)
  if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
    raise RuntimeError("Source checkpoint changed during migration")
  receipt = {
    "source": str(source), "destination": str(destination), "source_sha256": digest,
    "destination_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    "iteration": migrated["iter"], "contract": CONTRACT_VERSION,
    "theta_stage": migrated["infos"]["theta_stage"], "training_started": False,
  }
  destination.with_suffix(".migration.json").write_text(json.dumps(receipt, indent=2) + "\n")
  print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
  main()