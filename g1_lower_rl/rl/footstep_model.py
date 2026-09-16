"""Deployable full-body observations for a lower-body footstep GRU policy."""

from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch
from mjlab.rl import RslRlModelCfg
from rsl_rl.models import RNNModel
from rsl_rl.modules import HiddenState
from tensordict import TensorDict
from torch import nn

from g1_lower_rl.assets import LOWER_BODY_JOINTS, WHOLE_BODY_JOINTS, get_robot_cfg
from g1_lower_rl.footstep_phase import FootstepPhaseCfg, resolve_phase_cfg

CONTRACT_VERSION = "g1_footstep_gru_v1"
RAW_OBS_DIM = 84
ENCODED_OBS_DIM = 89
ACTION_DIM = 15
FOOTSTEP_SLOTS = ("L1", "L2", "R1", "R2")
OBSERVATION_LAYOUT = (
  ("joint_pos", 0, 29, "rad"),
  ("joint_vel", 29, 58, "rad/s"),
  ("pelvis_ang_vel", 58, 61, "rad/s"),
  ("pelvis_projected_gravity", 61, 64, "unit vector"),
  ("torso_ang_vel", 64, 67, "rad/s"),
  ("torso_projected_gravity", 67, 70, "unit vector"),
  ("phase", 70, 71, "rad"),
  ("frequency", 71, 72, "cycles/s"),
  ("footsteps", 72, 84, "[m, m, rad] per slot"),
)


def nominal_joint_positions() -> tuple[float, ...]:
  positions = get_robot_cfg().init_state.joint_pos
  return tuple(
    next((value for pattern, value in positions.items() if re.fullmatch(pattern, name)), 0.0) for name in WHOLE_BODY_JOINTS
  )


def nominal_action_parameters() -> dict[str, list[float]]:
  result: dict[str, list[float]] = {name: [] for name in ("scale", "stiffness", "damping", "effort_limit")}
  actuators = get_robot_cfg().articulation.actuators
  for joint_name in LOWER_BODY_JOINTS:
    matches = [
      actuator for actuator in actuators if any(re.fullmatch(pattern, joint_name) for pattern in actuator.target_names_expr)
    ]
    if len(matches) != 1:
      raise ValueError(f"Expected exactly one actuator for {joint_name}")
    actuator = matches[0]
    stiffness = float(actuator.stiffness)
    damping = float(actuator.damping)
    effort_limit = float(actuator.effort_limit)
    if not all(math.isfinite(value) and value > 0 for value in (stiffness, damping, effort_limit)):
      raise ValueError(f"Invalid actuator parameters for {joint_name}")
    result["scale"].append(0.25 * effort_limit / stiffness)
    result["stiffness"].append(stiffness)
    result["damping"].append(damping)
    result["effort_limit"].append(effort_limit)
  return result


class FootstepObservationEncoder(nn.Module):
  """Encode [..., 84] raw observations as [..., 89] continuous features."""

  def __init__(self, default_joint_pos: Sequence[float] | None = None) -> None:
    super().__init__()
    defaults = torch.tensor(
      nominal_joint_positions() if default_joint_pos is None else default_joint_pos,
      dtype=torch.float32,
    )
    if defaults.shape != (29,) or not torch.isfinite(defaults).all():
      raise ValueError("default_joint_pos must contain 29 finite joint angles")
    self.register_buffer("default_joint_pos", defaults)

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    if obs.shape[-1] != 84:
      raise ValueError("Footstep observations must have exactly 84 raw features")
    footprints = obs[..., 72:84].unflatten(-1, (4, 3))
    headings = footprints[..., 2:3]
    encoded_footprints = torch.cat((footprints[..., :2], headings.sin(), headings.cos()), dim=-1).flatten(-2)
    phase = obs[..., 70:71]
    return torch.cat(
      (
        obs[..., :29] - self.default_joint_pos,
        obs[..., 29:70],
        phase.sin(),
        phase.cos(),
        obs[..., 71:72],
        encoded_footprints,
      ),
      dim=-1,
    )


class FootstepActor(RNNModel):
  """rsl_rl recurrent actor with an explicit raw-observation deployment boundary."""

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (256, 128),
    activation: str = "elu",
    obs_normalization: bool = True,
    distribution_cfg: dict | None = None,
    rnn_type: str = "gru",
    rnn_hidden_dim: int = 32,
    rnn_num_layers: int = 1,
    default_joint_pos: Sequence[float] | None = None,
    phase_cfg: FootstepPhaseCfg | dict | None = None,
  ) -> None:
    if output_dim != ACTION_DIM:
      raise ValueError("FootstepActor controls exactly 15 leg and waist joints")
    if rnn_type != "gru":
      raise ValueError("FootstepActor requires a GRU")
    super().__init__(
      obs=obs,
      obs_groups=obs_groups,
      obs_set=obs_set,
      output_dim=output_dim,
      hidden_dims=hidden_dims,
      activation=activation,
      obs_normalization=obs_normalization,
      distribution_cfg=copy.deepcopy(distribution_cfg),
      rnn_type=rnn_type,
      rnn_hidden_dim=rnn_hidden_dim,
      rnn_num_layers=rnn_num_layers,
    )
    self.encoder = FootstepObservationEncoder(default_joint_pos)
    self.phase_cfg = resolve_phase_cfg(phase_cfg=FootstepPhaseCfg(**phase_cfg) if isinstance(phase_cfg, dict) else phase_cfg)
    for name, values in nominal_action_parameters().items():
      self.register_buffer(f"action_{name}", torch.tensor(values, dtype=torch.float32))

  def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
    groups, raw_dim = super()._get_obs_dim(obs, obs_groups, obs_set)
    if len(groups) != 1 or raw_dim != RAW_OBS_DIM:
      raise ValueError("FootstepActor requires one observation group with 84 raw features")
    return groups, ENCODED_OBS_DIM

  def get_latent(self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None) -> torch.Tensor:
    features = self.obs_normalizer(self.encoder(obs[self.obs_groups[0]]))
    return self.rnn(features, masks, hidden_state).squeeze(0)

  def update_normalization(self, obs: TensorDict) -> None:
    if self.obs_normalization:
      features = self.encoder(obs[self.obs_groups[0]]).reshape(-1, ENCODED_OBS_DIM)
      self.obs_normalizer.update(features)

  def as_jit(self) -> nn.Module:
    exported = super().as_jit()
    exported.obs_normalizer = nn.Sequential(copy.deepcopy(self.encoder), exported.obs_normalizer)
    return exported

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    exported = super().as_onnx(verbose)
    exported.obs_normalizer = nn.Sequential(copy.deepcopy(self.encoder), exported.obs_normalizer)
    exported.input_size = RAW_OBS_DIM
    return exported

  def deployment_contract(self) -> dict:
    hidden_shape = [self.rnn.rnn.num_layers, 1, self.rnn.rnn.hidden_size]
    defaults = self.encoder.default_joint_pos.detach().cpu().tolist()
    return {
      "version": CONTRACT_VERSION,
      "model_class": "g1_lower_rl.rl.footstep_model:FootstepActor",
      "control_dt_s": 0.02,
      "dtype": "float32",
      "inputs": {"obs": [1, RAW_OBS_DIM], "h_in": hidden_shape},
      "outputs": {"actions": [1, ACTION_DIM], "h_out": hidden_shape},
      "encoded_obs_dim": ENCODED_OBS_DIM,
      "observation_layout": [
        {"name": name, "start": start, "stop": stop, "units": units} for name, start, stop, units in OBSERVATION_LAYOUT
      ],
      "joint_names": list(WHOLE_BODY_JOINTS),
      "default_joint_pos": defaults,
      "raw_joint_pos": "absolute calibrated encoder angles; default subtraction is inside the model",
      "imu": {
        "pelvis": "angular velocity and unit gravity in calibrated pelvis link axes",
        "torso": "angular velocity and unit gravity in calibrated torso link axes",
        "gravity": "R_world_from_link.T @ [0, 0, -1]; not raw accelerometer readings",
      },
      "footsteps": {
        "slots": list(FOOTSTEP_SLOTS),
        "raw_fields": ["dx", "dy", "dtheta"],
        "encoded_fields": ["dx", "dy", "sin(dtheta)", "cos(dtheta)"],
        "reference_frame": "one common frozen stance-foot heading frame; x forward, y left, z up",
        "heading": "foot sole yaw, positive counterclockwise about +z",
        "preview": "two unconsumed future landings per foot; current support targets are external",
        "coordinates": "all four targets relative to the same anchor, not chained deltas",
        "reanchor": "transform unchanged physical targets using an estimated new anchor",
        "requires_external_coordinate_conversion": True,
        "all_slots_required": True,
        "zero_progress": "repeat each foot's fixed landing pose; f>0 marches, f=0 holds double support",
        "standing_slots": "at f=0 use [left_hold, left_hold, right_hold, right_hold] in the same frozen anchor",
      },
      "phase": {
        "raw_range_rad": [0.0, 2 * math.pi],
        "upper_bound_exclusive": True,
        "encoded_fields": ["sin(x)", "cos(x)"],
        "frequency_units": "complete left-right cycles/s; aggregate cadence is 2*f steps/s",
        "standing_frequency": 0.0,
        "standing": "f=0 commands double support; freeze phase, anchor and queues, but continue GRU inference",
        "stop_transition": "decelerate with f>0 until a double-support window, then set f=0; scheduler remains external",
        "phase_encoding_at_standstill": "keep sin(x),cos(x); do not zero the phase features",
        "update": "external x_next = x + 2*pi*f*dt; model does not advance time or queues",
        "left_touchdown_rad": self.phase_cfg.left_stance_phase,
        "right_touchdown_rad": self.phase_cfg.right_stance_phase,
        "contact_schedule": self.phase_cfg.to_metadata(),
        "events": "crossing the phase boundary, not equality; theoretical contact only",
        "frequency_range": None,
        "frequency_range_status": "pending training design; not a certified operating range",
      },
      "normalization": {"inside_model": True, "empirical": self.obs_normalization, "update_on_device": False},
      "actions": {
        "joint_names": list(LOWER_BODY_JOINTS),
        "default_joint_pos": defaults[:ACTION_DIM],
        **{
          name: getattr(self, f"action_{name}").detach().cpu().tolist()
          for name in ("scale", "stiffness", "damping", "effort_limit")
        },
        "target_formula": "q_des = default_joint_pos + scale * actions",
        "velocity_target_rad_s": 0.0,
        "feedforward_torque_nm": 0.0,
        "torque_formula": "clip(stiffness*(q_des-q) - damping*dq, -effort_limit, effort_limit)",
        "action_clip": None,
        "action_filter": None,
        "output": "deterministic normalized position offsets, not torque or absolute joint angles",
      },
      "hidden_state": {
        "initial": "zeros",
        "next": "h_in = previous h_out",
        "reset": "startup or controlled takeover; never on ordinary footstep/anchor or walk/stand changes",
      },
      "deployment_status": "model interface only; training, sensing, scheduler and hardware validation are required",
    }


def export_footstep_policy(model: FootstepActor, path: str | Path) -> Path:
  """Export deterministic ONNX and a matching JSON contract without changing the live actor."""
  import onnx

  destination = Path(path)
  if destination.suffix != ".onnx":
    raise ValueError("Footstep export path must end in .onnx")
  destination.parent.mkdir(parents=True, exist_ok=True)
  exported = model.as_onnx().cpu().eval()
  contract = model.deployment_contract()
  with torch.no_grad():
    torch.onnx.export(
      exported,
      exported.get_dummy_inputs(),
      str(destination),
      input_names=exported.input_names,
      output_names=exported.output_names,
      opset_version=17,
      dynamo=False,
    )
  graph = onnx.load(str(destination))
  onnx.helper.set_model_props(graph, {"footstep_contract": json.dumps(contract, allow_nan=False)})
  onnx.checker.check_model(graph)
  onnx.save(graph, str(destination))
  destination.with_suffix(".contract.json").write_text(json.dumps(contract, indent=2, allow_nan=False) + "\n", encoding="utf-8")
  return destination


@dataclass
class FootstepModelCfg(RslRlModelCfg):
  class_name: str = "g1_lower_rl.rl.footstep_model:FootstepActor"
  hidden_dims: tuple[int, ...] = (256, 128)
  activation: str = "elu"
  obs_normalization: bool = True
  rnn_type: str = "gru"
  rnn_hidden_dim: int = 32
  rnn_num_layers: int = 1
  default_joint_pos: tuple[float, ...] | None = None
  phase_cfg: FootstepPhaseCfg = field(default_factory=FootstepPhaseCfg)
  distribution_cfg: dict | None = field(
    default_factory=lambda: {
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    }
  )
