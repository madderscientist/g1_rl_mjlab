"""仅提供 CPU 上的 ONNX 推理，不包含仿真器、相位调度器或硬件驱动"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from numpy.typing import ArrayLike

from g1_lower_rl.footstep_contract import ACTION_DIM, CONTRACT_VERSION, FOOTSTEP_SLOTS, RAW_OBS_DIM
from g1_lower_rl.footstep_phase import FootstepPhaseCfg


def validate_footstep_observation(obs: np.ndarray) -> None:
  if obs.shape != (1, RAW_OBS_DIM) or not np.isfinite(obs).all():
    raise ValueError(f"Expected finite raw observations with shape (1, {RAW_OBS_DIM})")
  if not 0 <= obs[0, 70] < 2 * np.pi:
    raise ValueError("Phase must be wrapped into [0, 2*pi)")
  if obs[0, 71] < 0:
    raise ValueError("Frequency must be nonnegative; zero commands double-support standing")
  if not -np.float32(np.pi) <= obs[0, 78] < np.float32(np.pi):
    raise ValueError("Theta must be wrapped into [-pi, pi)")


def compute_torso_heading_error(torso_yaw: float, foot_yaws: ArrayLike) -> float:
  """用同一参考系中的实际 torso/左右足底航向计算 theta，单位 rad"""
  feet = np.asarray(foot_yaws, dtype=np.float64)
  if feet.shape != (2,) or not np.isfinite(feet).all() or not np.isfinite(torso_yaw):
    raise ValueError("Expected one finite torso yaw and two finite left/right foot yaws")
  sine, cosine = np.sin(feet).sum(), np.cos(feet).sum()
  mean = np.arctan2(sine, cosine) if np.hypot(sine, cosine) > 1e-6 else feet[0]
  return float((torso_yaw - mean + np.pi) % (2 * np.pi) - np.pi)


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
  theta: float,
) -> np.ndarray:
  """拼接原始传感器、左右脚下一落点及期望 torso 偏角指令 theta"""
  if not np.isfinite(theta) or not -np.pi <= theta < np.pi:
    raise ValueError("Theta must be finite and wrapped into [-pi, pi)")
  parts = []
  for name, value, shape in (
    ("joint_pos", joint_pos, (29,)),
    ("joint_vel", joint_vel, (29,)),
    ("pelvis_ang_vel", pelvis_ang_vel, (3,)),
    ("pelvis_projected_gravity", pelvis_projected_gravity, (3,)),
    ("torso_ang_vel", torso_ang_vel, (3,)),
    ("torso_projected_gravity", torso_projected_gravity, (3,)),
    ("phase/frequency", [phase, frequency], (2,)),
    ("footsteps", footsteps, (2, 3)),
    ("theta", [theta], (1,)),
  ):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape:
      raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    parts.append(array.reshape(-1))
  obs = np.concatenate(parts)[None, :]
  if obs[0, 78] == np.float32(np.pi):
    obs[0, 78] = -np.float32(np.pi)
  validate_footstep_observation(obs)
  return obs


class FootstepPolicy:
  """为每台机器人维护独立循环状态，返回归一化动作和关节位置目标"""

  def __init__(self, path: str | Path) -> None:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    metadata = self.session.get_modelmeta().custom_metadata_map
    if "footstep_contract" not in metadata:
      raise ValueError("ONNX has no footstep_contract metadata; use export_footstep_policy")
    self.contract = json.loads(metadata["footstep_contract"])
    if self.contract.get("version") != CONTRACT_VERSION or self.contract.get("dtype") != "float32":
      raise ValueError("Unsupported footstep policy contract")
    if self.contract.get("footsteps", {}).get("slots") != list(FOOTSTEP_SLOTS):
      raise ValueError("Expected fixed left/right next landing slots")
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
      inputs["obs"] != [1, RAW_OBS_DIM]
      or outputs["actions"] != [1, ACTION_DIM]
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
    actions, next_hidden = [
      np.asarray(value, dtype=np.float32)
      for value in self.session.run(["actions", "h_out"], {"obs": raw, "h_in": self.hidden_state})
    ]
    targets = self.default_pos + self.action_scale * actions[0]
    if not all(np.isfinite(value).all() for value in (actions, next_hidden, targets)):
      raise RuntimeError("Non-finite policy output; do not send targets to the robot")
    self.hidden_state = next_hidden
    return actions[0], targets
