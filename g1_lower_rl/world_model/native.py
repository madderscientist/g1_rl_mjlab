"""Audited fixed29 native-policy loading and collection, without policy training."""

from __future__ import annotations

import hashlib
import importlib
import json
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path, PosixPath
from unittest.mock import patch

import torch
from torch import nn

from g1_lower_rl.assets.g1_gloria import get_robot_cfg, get_spec
from g1_lower_rl.tasks.domain_randomization import LINK_MASS_SCALE_RANGE, make_link_mass_event
from g1_lower_rl.tasks.motion_tracking.corpus import file_sha256
from g1_lower_rl.world_model.model import JOINT_NAMES

ADAM_FIELDS = (("wm_opt", "new_WM_Adam"), ("policy_opt", "new_actor_Adam"), ("critic_opt", "new_critic_Adam"))


def adam_steps(checkpoint):
  result = {}
  for field, counter in ADAM_FIELDS:
    steps = {int(state["step"]) for state in checkpoint[field]["state"].values()}
    if len(steps) != 1 or min(steps) < 1:
      raise ValueError(f"Missing/inconsistent serialized Adam steps: {field}: {steps}")
    result[counter] = steps.pop()
  return result


def audited_origin(checkpoint, path, asset_sha, manifest, motions):
  resume = "fixed29_transfer" in checkpoint
  origin = path
  if resume:
    meta = checkpoint["fixed29_transfer"]
    previous = json.loads((path.parent / "report.json").read_text())
    required = {
      "robot": "calibrated_gloria_fixed_closed29", "canonical_joint_names": list(JOINT_NAMES),
      "replica_source_manifest": manifest, "motion_source_provenance": motions,
      "mass_scale_range": list(LINK_MASS_SCALE_RANGE),
      "initialization": {"reference": .70, "half_squat": .15, "airborne": .15},
      "airborne_height_offset": [.15, .22],
    }
    for key, value in required.items():
      if previous[key] != value:
        raise ValueError(f"Fixed29 resume provenance mismatch: {key}")
    if meta["asset_sha256"] != asset_sha or previous["asset_sha256"] != asset_sha:
      raise ValueError("Fixed29 resume asset changed")
    if meta["source_sha256"] != previous["source_sha256"]:
      raise ValueError("Fixed29 checkpoint/report parent lineage mismatch")
    origin = Path(meta.get("origin_checkpoint", previous["source_checkpoint"]))
    adam_steps(checkpoint)
  accepted = json.loads((origin.parent.parent / "final_physical_geometry/report.json").read_text())
  if file_sha256(origin) != accepted["checkpoint_sha256"]:
    raise ValueError("Origin checkpoint differs from the previously audited stock5000 checkpoint")
  return accepted, origin, resume


def load_weights(path: Path) -> dict:
  allowed = {"pathlib.Path", "pathlib.PosixPath"}
  unsafe = set(torch.serialization.get_unsafe_globals_in_checkpoint(path))
  if unsafe - allowed:
    raise ValueError(f"Unexpected checkpoint globals: {sorted(unsafe - allowed)}")
  with torch.serialization.safe_globals([Path, PosixPath]):
    return torch.load(path, map_location="cpu", weights_only=True)


def tensor_hash(module: nn.Module) -> str:
  digest = hashlib.sha256()
  for name, value in sorted(module.state_dict().items()):
    if not isinstance(value, torch.Tensor):
      raise TypeError(f"Unexpected nontensor model state: {name}")
    digest.update(name.encode())
    array = value.detach().cpu().contiguous().numpy()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(array.tobytes())
  return digest.hexdigest()


@contextmanager
def fixed29_runtime(report):
  source = importlib.import_module("supermimic.envs.tracking_env_cfg")
  loop = importlib.import_module("supermimic.training.loop")
  commands = importlib.import_module("supermimic.envs.mdp.commands")
  factory = source.make_env_cfg

  class MixedCommand(commands.MultiClipMotionCommand):
    def _write_reference_state_to_sim(self, ids, position, orientation, linear, angular, joints, velocity):
      if self.cfg.mixed_initialization:
        mode = torch.rand(len(ids), device=self.device)
        crouch, airborne = mode < 0.15, (mode >= 0.15) & (mode < 0.30)
        counters = report["observed_initialization_counts"]
        counters["half_squat"] += int(crouch.sum())
        counters["airborne"] += int(airborne.sum())
        counters["reference"] += int((mode >= .30).sum())
        joints, velocity = joints.clone(), velocity.clone()
        linear, angular, position = linear.clone(), angular.clone(), position.clone()
        for side in ("left", "right"):
          for name, value in (("hip_pitch", -0.55), ("knee", 1.1), ("ankle_pitch", -0.55)):
            joints[crouch, JOINT_NAMES.index(f"{side}_{name}_joint")] = value
        position[crouch, 2] = self._env.scene.env_origins[ids[crouch], 2] + 0.62
        velocity[crouch], linear[crouch], angular[crouch] = 0, 0, 0
        position[airborne, 2] += torch.empty(int(airborne.sum()), device=self.device).uniform_(0.15, 0.22)
        linear[airborne, 2] = torch.empty(int(airborne.sum()), device=self.device).uniform_(-1, 1)
      return super()._write_reference_state_to_sim(ids, position, orientation, linear, angular, joints, velocity)

  @dataclass(kw_only=True)
  class MixedCommandCfg(commands.MultiClipMotionCommandCfg):
    mixed_initialization: bool = True

    def build(self, env):
      return MixedCommand(self, env)

  def make_cfg(args):
    cfg = factory(args)
    cfg.scene.entities["robot"] = get_robot_cfg()
    cfg.sim.njmax, cfg.sim.nconmax = 500, 64
    if args.domain_randomization:
      cfg.events["link_mass"] = make_link_mass_event()
    cfg.commands["motion"] = MixedCommandCfg(
      **asdict(cfg.commands["motion"]), mixed_initialization=args.reset_noise,
    )
    return cfg

  with patch.object(source, "make_env_cfg", make_cfg), patch.object(loop, "load_g1_model", lambda: get_spec().compile()):
    yield
