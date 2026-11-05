"""Audited22650 frozen policy, full mocap references, native MuJoCo supervision."""

from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from unittest.mock import patch

import torch

from g1_lower_rl.tasks.motion_tracking.corpus import file_sha256
from g1_lower_rl.world_model.adapters.kinematics import StaticKinematicTemplate
from g1_lower_rl.world_model.collection import CollectorConfig, MotionCollector, TransitionBatch
from g1_lower_rl.world_model.mocap import MmapMotionLibrary
from g1_lower_rl.world_model.model import BODY_NAMES, CONTRACT_VERSION, CONTROL_DT, JOINT_NAMES

BASELINE_SHA256 = "47aa6ff79f6147a815e9460a3ffcc964635edf1786a221f53c18fd520d0091c1"


@runtime_checkable
class NativeMotionCommand(Protocol):
  library: MmapMotionLibrary
  frame: torch.Tensor
  clip_end: torch.Tensor
  _resample_command: Callable[[torch.Tensor], None]
  _sample_frames: Callable[[int], torch.Tensor]


class FullCorpusPositionCollector(MotionCollector):
  """Shared partition guards for native full-corpus position environments."""

  def _sample_frames(self, count: int) -> torch.Tensor:
    return self.library.sample(count, validation=self._sampling_validation)

  def _resample(self, env_ids: torch.Tensor) -> None:
    for validation in (False, True):
      selected = env_ids[self._validation[env_ids] == validation]
      if selected.numel():
        self._sampling_validation = validation
        self._original_native_resample(selected)
        clips = self.library.frame_clip[self._native_command.frame[selected]]
        if ((clips >= self.library.corpus.training_motions) != validation).any():
          raise RuntimeError("Native command reset crossed the fixed train/validation family partition")
        self._segment[selected] += 1
        self._dirty[selected] = True


class BaselineMotionCollector(FullCorpusPositionCollector):
  def __init__(self, config: CollectorConfig, *, replica_source: str, validation_envs: int):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.sensor import ContactMatch, ContactSensorCfg

    if not 0 < validation_envs < config.num_envs or config.physics_dt != 0.005:
      raise ValueError("Require train/validation environments and fixed5ms physics")
    if not config.coverage or config.recover_nonfinite:
      raise ValueError("The audited baseline requires mixed initialization/DR and fail-fast physical guards")
    checkpoint = Path(config.checkpoint).resolve()
    if file_sha256(checkpoint) != BASELINE_SHA256:
      raise ValueError("Frozen pretraining teacher must be the retained audited22650 checkpoint")
    source = Path(replica_source).resolve()
    report = json.loads((checkpoint.parent / "report.json").read_text())
    source_manifest = {str(path.relative_to(source)): file_sha256(path) for path in sorted((source / "src").rglob("*.py"))}
    if source_manifest != report["replica_source_manifest"]:
      raise ValueError("Pinned native-policy reproduction source changed")
    sys.path.insert(0, str(source / "src"))
    package = importlib.import_module("supermimic")
    if package.__file__ is None or Path(package.__file__).resolve().parents[2] != source:
      raise RuntimeError("A different supermimic package is already imported")
    bridge = importlib.import_module("g1_lower_rl.world_model.native")
    saved = bridge.load_weights(checkpoint)
    if saved["iteration"] != 22650:
      raise ValueError("Teacher iteration differs from22650")
    asset = Path(__file__).resolve().parents[1] / "assets/g1_gloria/xmls/g1_with_dual_gloria_m.xml"
    bridge.audited_origin(saved, checkpoint, file_sha256(asset), source_manifest, report["motion_source_provenance"])
    for record in report["motion_source_provenance"]:
      if not record["original_training_motion"] or file_sha256(Path(record["output"])) != record["output_sha256"]:
        raise ValueError("Frozen teacher's original motion lineage changed")
    nets = importlib.import_module("supermimic.policy.nets")
    self.config, self.num_envs = config, config.num_envs
    self.device = torch.device(config.device)
    torch.manual_seed(config.seed)
    self.teacher: Any = nets.Policy(nets.PolicyCfg(**saved["config"]["policy"])).to(self.device)
    self.teacher.load_state_dict(saved["policy"], strict=True)
    self.teacher.eval().requires_grad_(False)
    self._teacher_hash = bridge.tensor_hash(self.teacher)
    self._tensor_hash = bridge.tensor_hash
    del saved
    self._runtime = ExitStack()
    self._initialization = {"reference": 0, "half_squat": 0, "airborne": 0}
    try:
      self._runtime.enter_context(bridge.fixed29_runtime({"observed_initialization_counts": self._initialization}))
      native_cfg = importlib.import_module("supermimic.envs.tracking_env_cfg")
      commands = importlib.import_module("supermimic.envs.mdp.commands")

      def library_factory(npz_files, body_names, default_joint_pos, device, bin_frames, **_kwargs):
        if npz_files:
          raise ValueError("This collector accepts the full mapped corpus, not an NPZ subset")
        if not config.motion_dir:
          raise ValueError("Full mocap directory is required")
        return MmapMotionLibrary(Path(config.motion_dir), body_names, default_joint_pos, device, bin_frames)

      args = native_cfg.SuperMimicEnvCfgArgs(
        npz_files=[],
        num_envs=config.num_envs,
        action_noise_std=config.action_noise,
        record_transitions=False,
        seed=config.seed,
        sampling_mode="uniform",
      )
      cfg = native_cfg.make_env_cfg(args)
      cfg.sim.mujoco.timestep, cfg.decimation = config.physics_dt, 4
      if config.njmax is not None:
        cfg.sim.njmax = config.njmax
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
      with patch.object(commands, "MotionLibrary", library_factory):
        self.env = ManagerBasedRlEnv(cfg=cfg, device=config.device)
      if self.env.step_dt != CONTROL_DT:
        raise ValueError("Native baseline control period must be20ms")
      self._initialize_recording()
      command = self.env.command_manager.get_term("motion")
      if not isinstance(command, NativeMotionCommand) or not isinstance(command.library, MmapMotionLibrary):
        raise TypeError("Expected the native motion command with the full mapped library")
      self._native_command = command
      self.library: MmapMotionLibrary = self._native_command.library
      self._validation = torch.arange(self.num_envs, device=self.device) >= self.num_envs - validation_envs
      self._original_native_resample = self._native_command._resample_command
      self._native_command._resample_command = self._resample
      self._native_command._sample_frames = self._sample_frames
      self._sampling_validation = False
      self._observations, _ = self.env.reset()
      self.metadata = {
        "collector_version": 3,
        "contract_version": CONTRACT_VERSION,
        "teacher_kind": "supermimic22650",
        "teacher_checkpoint": str(checkpoint),
        "teacher_sha256": BASELINE_SHA256,
        "teacher_iteration": 22650,
        "teacher_policy_sha256": self._teacher_hash,
        "teacher_frozen": True,
        "policy_collection": "frozen_teacher",
        "old_WM_loaded_into_new_model": False,
        "replica_source_manifest": source_manifest,
        "config": asdict(config),
        "corpus": self.library.metadata,
        "static_template": StaticKinematicTemplate.from_mujoco(self.env.sim.mj_model, "robot/").to_dict(),
        "native_position_actuators": True,
        "integration": "native_position_actuator_fixed20ms_held_actual_target",
        "teacher_controller": "native_action_noise_scale_offset_clip_encoder_bias_no_GC",
        "physics_dt": config.physics_dt,
        "control_dt": self.env.step_dt,
        "joint_names": list(JOINT_NAMES),
        "body_names": list(BODY_NAMES),
        "indices": self.indices,
        "force_sign": self._force_sign,
        "force_label": "final_physics_substep_contact_solution_world_ground_to_foot",
        "collection_overrides": {
          "fixed_unconditioned_parameters": ["pd_gains", "motor_strength", "joint_friction", "joint_armature"],
          "external_disturbances": "not_in_contract_rejected",
          "gravity_compensation": "disabled_as_native_teacher",
          "scope": "isolated_collector_only_not_default_task_configs",
        },
        "coverage": {
          "mass_scale": [0.75, 1.05],
          "foot_mu": [0.3, 1.2],
          "shared_left_right_mu": True,
          "reference_probability": 0.70,
          "half_squat_probability": 0.15,
          "airborne_probability": 0.15,
          "airborne_height_offset": [0.15, 0.22],
        },
        "storage": {"njmax": self.env.sim.wp_data.njmax, "naconmax": self.env.sim.wp_data.naconmax},
        "fixed_actuator_parameters": {
          "kp": self._kp[0].tolist(),
          "kd": self._kd[0].tolist(),
          "effort_range": self._effort[0].tolist(),
          "dof_frictionloss": self.env.sim.mj_model.dof_frictionloss.tolist(),
          "dof_armature": self.env.sim.mj_model.dof_armature.tolist(),
        },
      }
    except BaseException:
      if hasattr(self, "env"):
        self.env.close()
      self._runtime.close()
      raise

  @torch.no_grad()
  def collect(self, num_control_steps: int = 1) -> TransitionBatch:
    if num_control_steps < 1:
      raise ValueError("num_control_steps must be positive")
    self._records.clear()
    self._collecting = True
    try:
      for _ in range(num_control_steps):
        self.nonfinite_envs.zero_()
        actions = self.teacher(self._observations["policy"], self._observations["goal"])
        if not torch.isfinite(actions).all():
          raise FloatingPointError("Nonfinite frozen-policy action")
        self._observations, self.last_rewards, terminated, truncated, self.last_extras = self.env.step(actions)
        self.last_dones = terminated | truncated
        if self.nonfinite_envs.any():
          ids = self.nonfinite_envs.nonzero().flatten()
          clips = self.library.frame_clip[self._native_command.frame[ids]].tolist()
          raise FloatingPointError(f"Native physics nonfinite/overflow; labels rejected: envs={ids.tolist()}, clips={clips}")
    finally:
      self._collecting = False
    if self._substep or len(self._records) != num_control_steps:
      raise RuntimeError("Expected exactly one native transition per20ms")
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
    self.counters.update(self.library.coverage())
    self.counters.update({f"initial_{name}": count for name, count in self._initialization.items()})
    return result

  def close(self) -> None:
    try:
      if self.teacher.training or any(p.requires_grad for p in self.teacher.parameters()):
        raise RuntimeError("Pretraining teacher was unfrozen")
      if self._tensor_hash(self.teacher) != self._teacher_hash:
        raise RuntimeError("Frozen22650 policy weights changed during collection")
    finally:
      self.env.close()
      self._runtime.close()
