"""Shared native position-control recording at control boundaries."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from mujoco._functions import mj_forward
from mujoco._structs import MjData, MjModel

from g1_lower_rl.world_model.model import (
  BODY_NAMES,
  CONTRACT_VERSION,
  JOINT_NAMES,
  TORQUE_DIM,
  pack_inputs,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(frozen=True)
class CollectorConfig:
  checkpoint: str
  num_envs: int = 512
  device: str = "cuda:0"
  seed: int = 0
  motion_dir: str | None = None
  coverage: bool = True
  physics_dt: float = 0.005
  action_noise: float = 0.1
  recover_nonfinite: bool = False
  njmax: int | None = None
  contact_sensor_maxmatch: int | None = None


@dataclass
class TransitionBatch:
  state: torch.Tensor
  inputs: torch.Tensor
  next_state: torch.Tensor
  foot_force: torch.Tensor
  valid: torch.Tensor
  segment: torch.Tensor
  contract_version: int = CONTRACT_VERSION


def canonical_indices(model: MjModel) -> dict[str, list[int]]:
  """Resolve the contract by names; never assume actuator and joint order match."""
  joints = [model.joint(f"robot/{name}").id for name in JOINT_NAMES]
  bodies = [model.body(f"robot/{name}").id for name in BODY_NAMES]
  actuators = []
  for joint in joints:
    matches = np.flatnonzero(model.actuator_trnid[:, 0] == joint)
    if len(matches) != 1:
      raise ValueError(f"Expected one actuator for {model.joint(joint).name}, got {matches.tolist()}")
    actuators.append(int(matches[0]))
  if (model.nq, model.nv, model.nu, model.na) != (7 + TORQUE_DIM, 6 + TORQUE_DIM, TORQUE_DIM, 0):
    raise ValueError(f"Unsupported G1 coordinates: {(model.nq, model.nv, model.nu, model.na)}")
  if not np.allclose(model.actuator_gear[actuators, 0], 1.0):
    raise ValueError("The position collector requires unit joint transmission gears")
  return {
    "joint": joints,
    "body": bodies,
    "actuator": actuators,
    "qpos": model.jnt_qposadr[joints].tolist(),
    "qvel": model.jnt_dofadr[joints].tolist(),
  }


def native_sensor_force_sign(model: MjModel, sensor_names: list[str], foot_geoms: list[int]) -> float:
  """Calibrate the sensor convention against gravity in a separate native scene."""
  data = MjData(model)
  data.qpos[:] = model.qpos0
  data.qpos[:3] = (0.0, 0.0, 1.0)
  mj_forward(model, data)
  lowest = min(float(data.geom_xpos[geom, 2] - model.geom_size[geom, 0]) for geom in foot_geoms)
  data.qpos[2] -= lowest + 0.002
  mj_forward(model, data)
  vertical = sum(float(data.sensordata[model.sensor(name).adr[0] + 2]) for name in sensor_names)
  if not math.isfinite(vertical) or abs(vertical) < 1.0:
    raise RuntimeError(f"Could not calibrate foot contact force convention: Fz={vertical}")
  return 1.0 if vertical > 0 else -1.0


class MotionCollector:
  """Record one actual position target held across every physical control substep."""

  env: ManagerBasedRlEnv
  config: CollectorConfig
  num_envs: int
  device: torch.device

  def _initialize_recording(self) -> None:
    from mjlab.sensor import ContactSensor

    model = self.env.sim.mj_model
    self.indices = canonical_indices(model)
    self._joint_ids = torch.tensor(self.indices["joint"], device=self.device)
    self._body_ids = torch.tensor(self.indices["body"], device=self.device)
    self._actuator_ids = torch.tensor(self.indices["actuator"], device=self.device)
    self._qpos_ids = torch.tensor(self.indices["qpos"], device=self.device)
    self._qvel_ids = torch.tensor(self.indices["qvel"], device=self.device)
    self._torso = model.body("robot/torso_link").id
    self._foot_geoms = [
      [i for i in range(model.ngeom) if re.fullmatch(f"robot/{side}_foot[1-7]_collision", model.geom(i).name or "")]
      for side in ("left", "right")
    ]
    if any(len(ids) != 7 for ids in self._foot_geoms):
      raise RuntimeError(f"Expected seven collision geoms on each foot: {self._foot_geoms}")
    self._sensor = self.env.scene["wm_feet"]
    assert isinstance(self._sensor, ContactSensor)
    expected_feet = ["left_ankle_roll_link", "right_ankle_roll_link"]
    if self._sensor.primary_names != expected_feet:
      raise RuntimeError(f"Unexpected foot sensor order: {self._sensor.primary_names}")
    self._force_sign = native_sensor_force_sign(
      model, [f"wm_feet_{name}_force" for name in expected_feet], [geom for ids in self._foot_geoms for geom in ids]
    )
    model_gpu = self.env.sim.model
    self._kp = model_gpu.actuator_gainprm[:, self._actuator_ids, 0].expand(self.num_envs, -1).clone()
    self._kd = -model_gpu.actuator_biasprm[:, self._actuator_ids, 2].expand(self.num_envs, -1).clone()
    self._effort = model_gpu.actuator_forcerange[:, self._actuator_ids].expand(self.num_envs, -1, -1).clone()
    self._nominal_torso_com = torch.tensor(model.body_ipos[self._torso], device=self.device, dtype=torch.float32)
    self._substep = 0
    self._tick_start: tuple[torch.Tensor, ...] | None = None
    self._held_target: torch.Tensor | None = None
    self._segment = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
    self._dirty = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
    self._last_next: torch.Tensor | None = None
    self._records: list[tuple[torch.Tensor, ...]] = []
    self._collecting = False
    self.nonfinite_envs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self._original_step = self.env.sim.step
    self._original_forward = self.env.sim.forward
    self.env.sim.step = self._step
    self.env.sim.forward = self._forward
    self.counters = {
      "substeps": 0,
      "control_steps": 0,
      "invalid_cache": 0,
      "nonfinite": 0,
      "state_discontinuities": 0,
      "nonfinite_resets": 0,
      "overflow_substeps": 0,
    }

  def _expanded(self, value: torch.Tensor) -> torch.Tensor:
    return value.expand(self.num_envs, *value.shape[1:])

  def _state(self) -> torch.Tensor:
    data = self.env.sim.data
    return torch.cat(
      (
        data.qpos[:, :3] - self.env.scene.env_origins,
        data.qpos[:, 3:7],
        data.qpos[:, self._qpos_ids],
        data.qvel[:, :6],
        data.qvel[:, self._qvel_ids],
      ),
      dim=-1,
    )

  def _context(self) -> torch.Tensor:
    model = self.env.sim.model
    return torch.cat(
      (
        self._expanded(model.body_mass)[:, self._body_ids],
        self._shared_friction(),
        self._expanded(model.body_ipos)[:, self._torso] - self._nominal_torso_com,
      ),
      dim=-1,
    )

  def _forward(self) -> None:
    self._original_forward()
    self._dirty.zero_()

  def _shared_friction(self) -> torch.Tensor:
    friction = self._expanded(self.env.sim.model.geom_friction)
    values = friction[:, [geom for ids in self._foot_geoms for geom in ids], 0]
    if not torch.allclose(values, values[:, :1].expand_as(values), atol=1e-6, rtol=0):
      raise ValueError("WM v3 requires shared friction on all left/right foot geoms; independent values cannot be averaged")
    return values[:, :1]

  def _step(self) -> None:
    data = self.env.sim.data
    if (data.qfrc_applied != 0).any() or (data.xfrc_applied != 0).any():
      raise ValueError("Applied external forces are not inputs to WM v3; disable them in its collection environment")
    if self._substep == 0:
      self._tick_invalid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
      state = self._state().clone()
      target = data.ctrl[:, self._actuator_ids].clone()
      native = self.env.sim.mj_model
      limited = torch.tensor(native.actuator_ctrllimited[self.indices["actuator"]], device=self.device, dtype=torch.bool)
      limits = torch.tensor(native.actuator_ctrlrange[self.indices["actuator"]], device=self.device, dtype=target.dtype)
      target = torch.where(limited, target.clamp(limits[:, 0], limits[:, 1]), target)
      self._held_target = target
      inputs = pack_inputs(state, target, self._context())
      if self._last_next is not None:
        changed = (state - self._last_next).abs().amax(-1) > 2e-5
        self._segment += changed.long()
        self.counters["state_discontinuities"] += int(changed.sum().item())
      self._tick_start = (state, inputs, self._segment.clone(), ~self._dirty)
    if self._held_target is None or self._tick_start is None:
      raise RuntimeError("Missing control-boundary state/position target")
    data.ctrl[:, self._actuator_ids] = self._held_target
    self._original_step()
    invalid = ~torch.isfinite(self._state()).all(-1) | (data.overflow != 0)
    self._tick_invalid |= invalid
    if self._collecting:
      self.nonfinite_envs |= invalid
    self._substep += 1
    self.counters["substeps"] += self.num_envs
    if self._substep < self.env.cfg.decimation:
      return
    self._substep = 0
    state, inputs, segment, clean = self._tick_start
    self._tick_start = None
    next_state = self._state()
    self._sensor._invalidate_cache()
    force = self._sensor.data.force
    if force is None:
      raise RuntimeError("Foot force sensor unexpectedly has no force field")
    force = force.clone() * self._force_sign
    finite = torch.isfinite(inputs).all(-1) & torch.isfinite(next_state).all(-1) & torch.isfinite(force).all((-1, -2))
    overflow = self.env.sim.data.overflow != 0
    if self._collecting:
      self.nonfinite_envs |= ~finite | overflow
      self._records.append((state, inputs, next_state, force, finite & clean & ~overflow & ~self._tick_invalid, segment))
      self.counters["control_steps"] += self.num_envs
      self.counters["invalid_cache"] += int((~clean).sum().item())
      self.counters["nonfinite"] += int((~finite).sum().item())
      self.counters["overflow_substeps"] += int(overflow.sum().item())
    self._last_next = next_state.detach()
