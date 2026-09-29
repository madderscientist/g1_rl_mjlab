"""Quarantine unstable physics worlds until the normal control-step reset."""

import json
import math
from pathlib import Path

import numpy as np
import torch


class NumericalSafety:
  def __init__(self, cfg, env):
    self.env = env
    self.sim = env.sim
    self.limits = {name: cfg.params.get(name, default) for name, default in (
      ("joint_speed_limit", 120.), ("root_speed_limit", 20.), ("root_angular_limit", 80.),
    )}
    if not all(math.isfinite(value) and value > 0 for value in self.limits.values()):
      raise ValueError("Numerical safety limits must be finite and positive")
    self.failed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.safe_qpos = self.sim.data.qpos.clone()
    self.safe_qvel = self.sim.data.qvel.clone()
    self.incidents = 0
    self.substeps = 0
    self._step = self.sim.step
    self.sim.step = self.step

  def reset(self, env_ids=None):
    selected = slice(None) if env_ids is None else env_ids
    self.failed[selected] = False
    self.safe_qpos[selected] = self.sim.data.qpos[selected]
    self.safe_qvel[selected] = self.sim.data.qvel[selected]

  def invalid_state(self):
    data = self.sim.data
    invalid = ~torch.isfinite(data.qpos).all(-1) | ~torch.isfinite(data.qvel).all(-1)
    invalid |= data.qvel[:, 6:].abs().amax(-1) > self.limits["joint_speed_limit"]
    invalid |= data.qvel[:, :3].norm(dim=-1) > self.limits["root_speed_limit"]
    invalid |= data.qvel[:, 3:6].norm(dim=-1) > self.limits["root_angular_limit"]
    quaternion_norm = data.qpos[:, 3:7].norm(dim=-1)
    invalid |= (quaternion_norm < .5) | (quaternion_norm > 1.5)
    for name in ("qacc", "qacc_warmstart", "sensordata"):
      value = getattr(data, name)
      invalid |= ~torch.isfinite(value).flatten(1).all(-1)
    invalid |= data.nefc > self.sim.wp_data.njmax
    return invalid

  def dump(self, env_ids, phase):
    self.incidents += 1
    output = Path(self.sim.cfg.nan_guard.output_dir) / "numerical_safety"
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"incident": self.incidents, "phase": phase, "substep": self.substeps,
      "common_step_counter": self.env.common_step_counter, "env_ids": env_ids.tolist(),
      "physics_dt": self.env.physics_dt, "limits": self.limits}
    prefix = output / f"incident_{self.env.common_step_counter}_{self.substeps}_{self.incidents}"
    if self.incidents <= 10:
      arrays = {"previous_qpos": self.safe_qpos[env_ids].cpu().numpy(),
        "previous_qvel": self.safe_qvel[env_ids].cpu().numpy()}
      for name in ("qpos", "qvel", "qacc", "qacc_warmstart", "ctrl", "qfrc_applied", "xfrc_applied", "nefc", "sensordata"):
        arrays[name] = getattr(self.sim.data, name)[env_ids].detach().cpu().numpy()
      for name in sorted(self.sim.expanded_fields):
        value = getattr(self.sim.model, name)
        arrays["model__" + name] = value[env_ids if value.shape[0] == self.env.num_envs else slice(None)].detach().cpu().numpy()
      arrays["actions"] = self.env.action_manager.action[env_ids].detach().cpu().numpy()
      np.savez_compressed(prefix.with_suffix(".npz"), **arrays)
      prefix.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (output / "incidents.jsonl").open("a") as stream:
      stream.write(json.dumps(metadata) + "\n")
    print("NUMERICAL_SAFETY " + json.dumps(metadata), flush=True)

  def quarantine(self, mask):
    env_ids = mask.nonzero(as_tuple=False).flatten()
    if env_ids.numel() == 0:
      return
    self.sim.reset(env_ids)
    self.sim.data.qpos[env_ids] = self.safe_qpos[env_ids]
    self.sim.data.qvel[env_ids] = 0.
    self.sim.data.ctrl[env_ids] = 0.
    self.sim.data.qfrc_applied[env_ids] = 0.
    self.sim.data.xfrc_applied[env_ids] = 0.
    self.sim.forward()

  def step(self):
    self.substeps += 1
    if not torch.isfinite(self.sim.data.ctrl).all():
      ids = (~torch.isfinite(self.sim.data.ctrl).all(-1)).nonzero(as_tuple=False).flatten()
      self.dump(ids, "nonfinite_control")
      raise FloatingPointError("Non-finite control input; refusing to advance physics")
    before = self.invalid_state() & ~self.failed
    if before.any():
      self.dump(before.nonzero(as_tuple=False).flatten(), "before_step")
      self.failed |= before
    self.quarantine(self.failed)
    self.safe_qpos.copy_(self.sim.data.qpos)
    self.safe_qvel.copy_(self.sim.data.qvel)
    self._step()
    after = self.invalid_state() & ~self.failed
    if after.any():
      self.dump(after.nonzero(as_tuple=False).flatten(), "after_step")
      self.failed |= after
    self.quarantine(self.failed)

  def __call__(self, env, joint_speed_limit=120., root_speed_limit=20., root_angular_limit=80.):
    return self.failed