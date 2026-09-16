"""CPU ONNX inference only; no simulator, phase scheduler, or hardware driver."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from numpy.typing import ArrayLike

from g1_lower_rl.footstep_phase import FootstepPhaseCfg


def validate_footstep_observation(obs: np.ndarray) -> None:
  if obs.shape != (1, 84) or not np.isfinite(obs).all():
    raise ValueError("Expected finite raw observations with shape (1, 84)")
  if not 0 <= obs[0, 70] < 2 * np.pi:
    raise ValueError("Phase must be wrapped into [0, 2*pi)")
  if obs[0, 71] < 0:
    raise ValueError("Frequency must be nonnegative; zero commands double-support standing")


def pack_footstep_observation(
  *,
  joint_pos: ArrayLike,
  joint_vel: ArrayLike,
  pelvis_ang_vel: ArrayLike,
  pelvis_projected_gravity: ArrayLike,
  torso_ang_vel: ArrayLike,
  torso_projected_gravity: ArrayLike,
  phase: float,
  frequency: float,
  footsteps: ArrayLike,
) -> np.ndarray:
  """Pack calibrated raw sensors and [L1,L2,R1,R2] XY/yaw commands without normalization."""
  parts = []
  for name, value, shape in (
    ("joint_pos", joint_pos, (29,)),
    ("joint_vel", joint_vel, (29,)),
    ("pelvis_ang_vel", pelvis_ang_vel, (3,)),
    ("pelvis_projected_gravity", pelvis_projected_gravity, (3,)),
    ("torso_ang_vel", torso_ang_vel, (3,)),
    ("torso_projected_gravity", torso_projected_gravity, (3,)),
    ("phase/frequency", [phase, frequency], (2,)),
    ("footsteps", footsteps, (4, 3)),
  ):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape:
      raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    parts.append(array.reshape(-1))
  obs = np.concatenate(parts)[None, :]
  validate_footstep_observation(obs)
  return obs


class FootstepPolicy:
  """One recurrent stream per robot, returning (normalized action, joint position target)."""

  def __init__(self, path: str | Path) -> None:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    metadata = self.session.get_modelmeta().custom_metadata_map
    if "footstep_contract" not in metadata:
      raise ValueError("ONNX has no footstep_contract metadata; use export_footstep_policy")
    self.contract = json.loads(metadata["footstep_contract"])
    if self.contract.get("version") != "g1_footstep_gru_v1" or self.contract.get("dtype") != "float32":
      raise ValueError("Unsupported footstep policy contract")
    schedule = self.contract.get("phase", {}).get("contact_schedule")
    if schedule is not None and schedule.get("parameterization") != "stance_centers_shared_half_width_v1":
      raise ValueError("Unsupported legacy phase-window metadata; re-export with the correct stance-center configuration")
    self.phase_cfg = (
      None
      if schedule is None
      else FootstepPhaseCfg(
        left_stance_phase=schedule["left_stance_phase"],
        right_stance_phase=schedule["right_stance_phase"],
        contact_half_width=schedule["contact_half_width"],
      )
    )
    if self.phase_cfg is not None:
      expected = self.phase_cfg.to_metadata()
      if schedule != expected or any(
        self.contract["phase"][key] != expected[key] for key in ("left_touchdown_rad", "right_touchdown_rad")
      ):
        raise ValueError("Inconsistent phase schedule metadata")
    inputs = {item.name: item.shape for item in self.session.get_inputs()}
    outputs = {item.name: item.shape for item in self.session.get_outputs()}
    if inputs != self.contract["inputs"] or outputs != self.contract["outputs"]:
      raise ValueError("ONNX interfaces do not match the embedded contract")
    if set(inputs) != {"obs", "h_in"} or set(outputs) != {"actions", "h_out"}:
      raise ValueError("Expected obs/h_in inputs and actions/h_out outputs")
    hidden_shape = inputs["h_in"]
    if (
      inputs["obs"] != [1, 84]
      or outputs["actions"] != [1, 15]
      or outputs["h_out"] != hidden_shape
      or len(hidden_shape) != 3
      or hidden_shape[1] != 1
      or not all(isinstance(size, int) and size > 0 for size in hidden_shape)
      or any(item.type != "tensor(float)" for item in (*self.session.get_inputs(), *self.session.get_outputs()))
    ):
      raise ValueError("Unsupported observation, action or GRU state tensor specification")
    self.joint_names = self.contract["joint_names"]
    self.action_joint_names = self.contract["actions"]["joint_names"]
    if len(set(self.joint_names)) != 29 or len(set(self.action_joint_names)) != 15:
      raise ValueError("Expected 29 distinct observed joints and 15 distinct controlled joints")
    if self.action_joint_names != self.joint_names[:15]:
      raise ValueError("Action order must match the first 15 observed joints")
    self.default_pos = np.asarray(self.contract["actions"]["default_joint_pos"], dtype=np.float32)
    self.action_scale = np.asarray(self.contract["actions"]["scale"], dtype=np.float32)
    if (
      self.default_pos.shape != (15,)
      or self.action_scale.shape != (15,)
      or not np.isfinite(self.default_pos).all()
      or not np.isfinite(self.action_scale).all()
      or (self.action_scale <= 0).any()
    ):
      raise ValueError("Invalid action offset or scale in contract")
    self.hidden_state = np.zeros(hidden_shape, dtype=np.float32)

  def reset(self) -> None:
    self.hidden_state.fill(0.0)

  def step(self, obs: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(obs, dtype=np.float32)
    validate_footstep_observation(raw)
    if raw[0, 71] == 0 and self.contract.get("phase", {}).get("standing_frequency") != 0.0:
      raise ValueError("This policy contract does not support zero-frequency standing; re-export a compatible model")
    actions, next_hidden = self.session.run(["actions", "h_out"], {"obs": raw, "h_in": self.hidden_state})
    targets = self.default_pos + self.action_scale * actions[0]
    if not all(np.isfinite(value).all() for value in (actions, next_hidden, targets)):
      raise RuntimeError("Non-finite policy output; do not send targets to the robot")
    self.hidden_state = next_hidden
    return actions[0], targets
