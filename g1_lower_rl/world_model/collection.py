"""Frozen-teacher collection of native position-control transitions at 50Hz."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from mujoco._functions import mj_forward
from mujoco._structs import MjData, MjModel

from g1_lower_rl.world_model.model import (
  BODY_NAMES,
  CONTRACT_VERSION,
  CONTROL_DT,
  JOINT_NAMES,
  TORQUE_DIM,
  pack_inputs,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnvCfg
  from mjlab.rl import RslRlOnPolicyRunnerCfg
  from tensordict import TensorDict

  from g1_lower_rl.rl.rgmt_model import RgmtActor
  from g1_lower_rl.tasks.motion_tracking.mdp.actions import GravityCompensatedJointPositionAction
  from g1_lower_rl.tasks.motion_tracking.mdp.commands import GeneralMotionCommand


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


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


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
  """Owns an isolated native-position environment and a frozen/injected policy.

  One actually executed target is held for all physical substeps in a control tick.
  Encoder/GC corrections occur outside the network, once per tick. Unconditioned
  actuator randomization is disabled only in this isolated collection config.
  Labels are captured before task events, resets or clip swaps.
  """

  _action: GravityCompensatedJointPositionAction
  _command: GeneralMotionCommand

  def __init__(
    self,
    config: CollectorConfig,
    *,
    env_cfg: ManagerBasedRlEnvCfg | None = None,
    policy_factory: Callable[[TensorDict, RslRlOnPolicyRunnerCfg, int], RgmtActor] | None = None,
  ):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
    from mjlab.sensor import ContactMatch, ContactSensorCfg
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

    import g1_lower_rl.tasks
    from g1_lower_rl.rl.motion_reference import restore_motion_reference_layout
    from g1_lower_rl.rl.rgmt_model import RgmtActor, RgmtModelCfg, actor_checkpoint_dtype
    from g1_lower_rl.tasks.motion_tracking.mdp.actions import GravityCompensatedJointPositionAction
    from g1_lower_rl.tasks.motion_tracking.mdp.commands import GeneralMotionCommand
    from g1_lower_rl.world_model.adapters import StaticKinematicTemplate

    _ = g1_lower_rl.tasks

    if config.num_envs < 2 or not math.isfinite(config.action_noise) or config.action_noise < 0 or config.physics_dt != 0.005:
      raise ValueError("Require >=2 environments, finite nonnegative action noise, and fixed5ms native physics")
    if any(value is not None and value < 1 for value in (config.njmax, config.contact_sensor_maxmatch)):
      raise ValueError("Simulator storage capacities must be positive")
    self.config = config
    self.device = torch.device(config.device)
    self.num_envs = config.num_envs
    torch.manual_seed(config.seed)
    checkpoint = Path(config.checkpoint).resolve() if config.checkpoint else None
    if checkpoint is not None and not checkpoint.is_file():
      raise FileNotFoundError(checkpoint)
    if policy_factory is None and checkpoint is None:
      raise ValueError("Frozen-teacher collection requires a checkpoint")
    task = "G1-Gloria-MotionTracking"
    cfg = load_env_cfg(task, play=False) if env_cfg is None else copy.deepcopy(env_cfg)
    agent_cfg = load_rl_cfg(task)
    assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg)
    assert isinstance(agent_cfg.actor, RgmtModelCfg)
    if policy_factory is None:
      assert checkpoint is not None
      restore_motion_reference_layout(cfg, agent_cfg, checkpoint)
    cfg.scene.num_envs = config.num_envs
    cfg.seed = config.seed
    cfg.sim.mujoco.timestep = config.physics_dt
    cfg.decimation = round(CONTROL_DT / config.physics_dt)
    for event in ("pd_gains", "motor_strength", "joint_friction", "joint_armature", "foot_friction"):
      cfg.events.pop(event, None)
    if config.njmax is not None:
      cfg.sim.njmax = config.njmax
    if config.contact_sensor_maxmatch is not None:
      cfg.sim.contact_sensor_maxmatch = config.contact_sensor_maxmatch
    command_cfg: Any = cfg.commands["motion"]
    if config.motion_dir is not None:
      command_cfg.motion_dir = config.motion_dir
    if config.coverage:
      cfg.events["payload"].params["mass_range"] = (-0.5, 2.0)
      cfg.events["base_com"].params["ranges"] = (-0.05, 0.05)
      cfg.events["encoder_bias"].params["bias_range"] = (-0.015, 0.015)
    cfg.scene.sensors = (
      *cfg.scene.sensors,
      ContactSensorCfg(
        name="wm_feet",
        primary=ContactMatch(mode="body", pattern=r"^(left|right)_ankle_roll_link$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
      ),
    )
    self.env = ManagerBasedRlEnv(cfg=cfg, device=config.device)
    if "geom_friction" not in self.env.sim.expanded_fields:
      self.env.sim.expand_model_fields(("geom_friction",))
    self.wrapper = RslRlVecEnvWrapper(self.env, clip_actions=agent_cfg.clip_actions)
    model = self.env.sim.mj_model
    self._initialize_recording()
    command = self.env.command_manager.get_term("motion")
    assert isinstance(command, GeneralMotionCommand)
    self._command = command
    self._original_resample = self._command._resample_command
    self._command._resample_command = self._resample
    action = self.env.action_manager.get_term("joint_pos")
    assert isinstance(action, GravityCompensatedJointPositionAction)
    self._action = action
    if config.coverage:
      self._original_gain_resample = self._action._resample_gain
      self._action._resample_gain = self._resample_gain
      self._resample_gain(slice(None))
    self.wrapper.reset()
    self._randomize_foot_friction(torch.arange(self.num_envs, device=self.device))
    observations = self.wrapper.get_observations()
    actor_kwargs = asdict(agent_cfg.actor)
    actor_class = actor_kwargs.pop("class_name")
    if actor_class != "g1_lower_rl.rl.rgmt_model:RgmtActor":
      raise ValueError(f"Expected the preserved RGMT teacher, got {actor_class}")
    if actor_kwargs.pop("cnn_cfg") is not None or actor_kwargs.pop("rnn_type") is not None:
      raise ValueError("The frozen RGMT collector cannot replace an active CNN or recurrent teacher")
    actor_kwargs.pop("rnn_hidden_dim")
    actor_kwargs.pop("rnn_num_layers")
    if policy_factory is None:
      assert checkpoint is not None
      self.teacher = RgmtActor(
        observations,
        obs_groups={name: list(groups) for name, groups in agent_cfg.obs_groups.items()},
        obs_set="actor",
        output_dim=self.wrapper.num_actions,
        **actor_kwargs,
      ).to(self.device)
      saved = torch.load(checkpoint, map_location=self.device, weights_only=False)
      self.teacher.to(dtype=actor_checkpoint_dtype(saved["actor_state_dict"]))
      self.teacher.load_state_dict(saved["actor_state_dict"], strict=True)
      self.teacher.eval().requires_grad_(False)
    else:
      self.teacher = policy_factory(observations, agent_cfg, self.wrapper.num_actions)
      self.teacher.eval()
    self._observations = observations
    self.nonfinite_envs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.metadata = {
      "collector_version": 2,
      "contract_version": CONTRACT_VERSION,
      "teacher_checkpoint": str(checkpoint) if checkpoint else None,
      "teacher_sha256": _sha256(checkpoint) if checkpoint else None,
      "teacher_reference_dim": agent_cfg.actor.reference_dim,
      "teacher_frozen": policy_factory is None,
      "policy_collection": "frozen_teacher" if policy_factory is None else "current_trainable_policy",
      "static_template": StaticKinematicTemplate.from_mujoco(model, name_prefix="robot/").to_dict(),
      "controller_template": {
        "kp_nominal": model.actuator_gainprm[self.indices["actuator"], 0].tolist(),
        "kd_nominal": (-model.actuator_biasprm[self.indices["actuator"], 2]).tolist(),
        "effort_range_nominal": model.actuator_forcerange[self.indices["actuator"]].tolist(),
        "encoder_bias_range": [-0.015, 0.015] if config.coverage else [-0.01, 0.01],
        "kp_scale_range": [1.0, 1.0],
        "kd_scale_range": [1.0, 1.0],
        "effort_scale_range": [1.0, 1.0],
        "gravity_compensation_nominal_kp": self._action._kp.cpu().tolist(),
        "gravity_compensation_gain_range": list(self._action.cfg.gain_range),
      },
      "config": asdict(config),
      "motion_dir": str(command_cfg.motion_dir),
      "integration": "native_position_actuator_fixed20ms_held_actual_target",
      "teacher_controller": "control_boundary_encoder_bias_GC_then_held_target",
      "native_position_actuators": True,
      "collection_overrides": {
        "fixed_unconditioned_parameters": ["pd_gains", "motor_strength", "joint_friction", "joint_armature"],
        "foot_friction": "shared_left_right",
        "gravity_compensation": "sample_actual_target_once_per_control_tick",
        "external_disturbances": "not_in_contract_rejected",
        "scope": "isolated_collector_only_not_default_task_configs",
      },
      "physics_dt": config.physics_dt,
      "control_dt": self.env.step_dt,
      "force_label": "final_physics_substep_contact_solution_world_ground_to_foot",
      "force_sign": self._force_sign,
      "joint_names": list(JOINT_NAMES),
      "body_names": list(BODY_NAMES),
      "indices": self.indices,
      "solver": {
        "integrator": int(model.opt.integrator),
        "iterations": int(model.opt.iterations),
        "ls_iterations": int(model.opt.ls_iterations),
        "gravity": model.opt.gravity.tolist(),
      },
      "storage": {
        "njmax": self.env.sim.wp_data.njmax,
        "naconmax": self.env.sim.wp_data.naconmax,
        "contact_sensor_maxmatch": cfg.sim.contact_sensor_maxmatch,
      },
      "condim": sorted(set(model.geom_condim[[geom for ids in self._foot_geoms for geom in ids]].tolist())),
      "coverage": {
        "foot_mu": [0.3, 1.6] if config.coverage else [0.3, 1.2],
        "payload_mass": [-0.5, 2.0] if config.coverage else [-0.5, 1.0],
        "initial_crouch_probability": 0.15 if config.coverage else 0,
        "initial_airborne_probability": 0.15 if config.coverage else 0,
        "gravity_compensation_off_probability": 0.25 if config.coverage else 0,
      },
    }
    manifest = Path(str(command_cfg.motion_dir)) / "manifest.json"
    if manifest.is_file():
      self.metadata["corpus_manifest"] = json.loads(manifest.read_text())

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

  def _resample_gain(self, env_ids) -> None:
    self._original_gain_resample(env_ids)
    gain = self._action._gain[env_ids]
    groups = (torch.rand((gain.shape[0], 2), device=self.device) >= 0.25).float()
    self._action._gain[env_ids] = gain * groups[:, self._action._gain_group]

  def _randomize_foot_friction(self, env_ids: torch.Tensor) -> None:
    friction = self.env.sim.model.geom_friction
    upper = 1.6 if self.config.coverage else 1.2
    mu = torch.empty((len(env_ids), 1), device=self.device).uniform_(0.3, upper)
    for ids in self._foot_geoms:
      friction[env_ids[:, None], torch.tensor(ids, device=self.device)[None, :], 0] = mu

  def _shared_friction(self) -> torch.Tensor:
    friction = self._expanded(self.env.sim.model.geom_friction)
    values = friction[:, [geom for ids in self._foot_geoms for geom in ids], 0]
    if not torch.allclose(values, values[:, :1].expand_as(values), atol=1e-6, rtol=0):
      raise ValueError("WM v3 requires shared friction on all left/right foot geoms; independent values cannot be averaged")
    return values[:, :1]

  def _resample(self, env_ids: torch.Tensor) -> None:
    self._original_resample(env_ids)
    self._segment[env_ids] += 1
    self._dirty[env_ids] = True
    self._randomize_foot_friction(env_ids)
    if not self.config.coverage:
      return
    mode = torch.rand(len(env_ids), device=self.device)
    crouch_ids = env_ids[mode < 0.15]
    air_ids = env_ids[(mode >= 0.15) & (mode < 0.3)]
    data = self.env.sim.data
    if len(crouch_ids):
      for side in ("left", "right"):
        for joint, value in (("hip_pitch", -0.55), ("knee", 1.1), ("ankle_pitch", -0.55)):
          index = self._qpos_ids[JOINT_NAMES.index(f"{side}_{joint}_joint")]
          data.qpos[crouch_ids, index] = value
      data.qpos[crouch_ids, 2] = self.env.scene.env_origins[crouch_ids, 2] + 0.62
      data.qvel[crouch_ids, :3] = 0
    if len(air_ids):
      data.qpos[air_ids, 2] += torch.empty(len(air_ids), device=self.device).uniform_(0.15, 0.45)
      data.qvel[air_ids, 2] = torch.empty(len(air_ids), device=self.device).uniform_(-1.0, 1.0)

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

  @torch.no_grad()
  def collect(self, num_control_steps: int = 1) -> TransitionBatch:
    if num_control_steps < 1:
      raise ValueError("num_control_steps must be positive")
    self._records.clear()
    self._collecting = True
    try:
      for _ in range(num_control_steps):
        self.nonfinite_envs.zero_()
        first_record = len(self._records)
        actions = self.teacher(self._observations, stochastic_output=False)
        if self.config.action_noise:
          actions = actions + self.config.action_noise * torch.randn_like(actions)
        actions = actions.to(self.env.sim.data.ctrl)
        self._observations, self.last_rewards, self.last_dones, self.last_extras = self.wrapper.step(actions)
        if self.config.recover_nonfinite:
          self._recover_nonfinite(first_record)
    finally:
      self._collecting = False
    if self._substep or len(self._records) != num_control_steps:
      raise RuntimeError("Collector did not produce exactly one transition per 20ms control tick")
    columns = list(zip(*self._records, strict=True))
    result = TransitionBatch(
      state=torch.stack(columns[0]),
      inputs=torch.stack(columns[1]),
      next_state=torch.stack(columns[2]),
      foot_force=torch.stack(columns[3]),
      valid=torch.stack(columns[4]),
      segment=torch.stack(columns[5]),
    )
    self._records.clear()
    if not torch.isfinite(self._state()).all():
      raise FloatingPointError("Nonfinite simulator state; collector refuses to continue corrupt trajectories")
    return result

  def _recover_nonfinite(self, first_record: int) -> None:
    from mjlab.utils.nan_guard import NanGuard
    from tensordict import TensorDict

    bad = self.nonfinite_envs | NanGuard.detect_nans(self.env.sim.data) | (self.env.sim.data.overflow != 0)
    attempts = torch.zeros(self.num_envs, dtype=torch.int8, device=self.device)
    while bad.any():
      exhausted = bad & (attempts >= 3)
      if exhausted.any():
        raise FloatingPointError(
          f"Invalid simulator data survived 3 targeted resets: envs={exhausted.nonzero().flatten().tolist()}"
        )
      ids = bad.nonzero().flatten()
      logging.getLogger(__name__).warning(
        "Invalid simulator data (nonfinite or overflow): quarantining control tick and resetting envs=%s motions=%s; "
        "invalid labels and unfinished returns must not enter training",
        ids.tolist(),
        self._command.motion_ids[ids].tolist(),
      )
      self.nonfinite_envs |= bad
      attempts[ids] += 1
      for record in self._records[first_record:]:
        record[4].logical_and_(~bad)
      observations, _ = self.env.reset(env_ids=ids)
      self._observations = TensorDict(observations, batch_size=[self.num_envs])
      self.last_dones[ids] = True
      self.counters["nonfinite_resets"] += ids.numel()
      bad = NanGuard.detect_nans(self.env.sim.data) | (self.env.sim.data.overflow != 0)
      for group in observations.values():
        for value in group.values() if isinstance(group, dict) else (group,):
          bad |= ~torch.isfinite(value).reshape(self.num_envs, -1).all(-1)

  def close(self) -> None:
    self.wrapper.close()
