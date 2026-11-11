"""供下肢脚步 GRU 策略使用的可部署全身观测接口"""

from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.rl import RslRlModelCfg
from rsl_rl.models import RNNModel
from rsl_rl.modules import HiddenState
from tensordict import TensorDict
from torch import nn

from g1_lower_rl.assets import LOWER_BODY_JOINTS, WHOLE_BODY_JOINTS, get_robot_cfg
from g1_lower_rl.footstep_contract import ACTION_DIM, CONTRACT_VERSION, ENCODED_OBS_DIM, FOOTSTEP_SLOTS, RAW_OBS_DIM
from g1_lower_rl.footstep_phase import FootstepPhaseCfg, resolve_phase_cfg
from g1_lower_rl.rl.mirror import _pair_joints
from rsl_rl.utils import unpad_trajectories

OBSERVATION_LAYOUT = (
  ("joint_pos", 0, 29, "rad"),
  ("joint_vel", 29, 58, "rad/s"),
  ("pelvis_ang_vel", 58, 61, "rad/s"),
  ("pelvis_projected_gravity", 61, 64, "unit vector"),
  ("torso_ang_vel", 64, 67, "rad/s"),
  ("torso_projected_gravity", 67, 70, "unit vector"),
  ("phase", 70, 71, "rad"),
  ("frequency", 71, 72, "cycles/s"),
  ("footsteps", 72, 78, "[m, m, rad] per slot"),
  ("theta", 78, 79, "rad; desired torso yaw offset from actual mean foot yaw"),
)


def nominal_joint_positions() -> tuple[float, ...]:
  positions = get_robot_cfg().init_state.joint_pos
  assert positions is not None
  return tuple(
    next((value for pattern, value in positions.items() if re.fullmatch(pattern, name)), 0.0) for name in WHOLE_BODY_JOINTS
  )


def nominal_action_parameters() -> dict[str, list[float]]:
  result: dict[str, list[float]] = {name: [] for name in ("scale", "stiffness", "damping", "effort_limit")}
  articulation = get_robot_cfg().articulation
  assert articulation is not None and articulation.actuators is not None
  actuators = articulation.actuators
  for joint_name in LOWER_BODY_JOINTS:
    matches = [
      actuator for actuator in actuators if any(re.fullmatch(pattern, joint_name) for pattern in actuator.target_names_expr)
    ]
    if len(matches) != 1:
      raise ValueError(f"Expected exactly one actuator for {joint_name}")
    actuator = matches[0]
    assert isinstance(actuator, BuiltinPositionActuatorCfg)
    assert actuator.stiffness is not None and actuator.damping is not None and actuator.effort_limit is not None
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
  """将形状为 [..., 79] 的原始观测编码为 [..., 82] 的特征"""

  default_joint_pos: torch.Tensor

  def __init__(self, default_joint_pos: Sequence[float] | None = None, *, include_theta: bool = True) -> None:
    super().__init__()
    self.include_theta = include_theta
    self.raw_obs_dim = RAW_OBS_DIM if include_theta else 78
    defaults = torch.tensor(
      nominal_joint_positions() if default_joint_pos is None else default_joint_pos,
      dtype=torch.float32,
    )
    if defaults.shape != (29,) or not torch.isfinite(defaults).all():
      raise ValueError("default_joint_pos must contain 29 finite joint angles")
    self.register_buffer("default_joint_pos", defaults)

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    if obs.shape[-1] != self.raw_obs_dim:
      raise ValueError(f"Footstep observations must have exactly {self.raw_obs_dim} raw features")
    footprints = obs[..., 72:78].unflatten(-1, (2, 3))
    headings = footprints[..., 2:3]
    encoded_footprints = torch.cat((footprints[..., :2], headings.sin(), headings.cos()), dim=-1).flatten(-2)
    phase = obs[..., 70:71]
    features = torch.cat(
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
    return torch.cat((features, obs[..., 78:79]), dim=-1) if self.include_theta else features


def build_footstep_mirror_buffers(*, include_theta: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """构建编码观测空间（82/81维）与动作空间（15维）的矢状面左右镜像置换与符号"""
  q_perm, q_sign = _pair_joints(list(WHOLE_BODY_JOINTS))
  act_perm, act_sign = _pair_joints(list(LOWER_BODY_JOINTS))
  enc_perm = (
    q_perm
    + [29 + i for i in q_perm]
    + list(range(58, 73))
    + [77, 78, 79, 80, 73, 74, 75, 76]
    + ([81] if include_theta else [])
  )
  enc_sign = (
    q_sign
    + q_sign
    + [-1.0, 1.0, -1.0]  # pelvis_ang_vel
    + [1.0, -1.0, 1.0]   # pelvis_projected_gravity
    + [-1.0, 1.0, -1.0]  # torso_ang_vel
    + [1.0, -1.0, 1.0]   # torso_projected_gravity
    + [-1.0, -1.0]       # sin(phase), cos(phase): phase + pi -> (-sin, -cos)
    + [1.0]              # frequency
    + [1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0]  # L_next <-> R_next: (dx, -dy, -sin, +cos)
    + ([-1.0] if include_theta else [])  # theta -> -theta
  )
  return (
    torch.tensor(enc_perm, dtype=torch.long),
    torch.tensor(enc_sign, dtype=torch.float32),
    torch.tensor(act_perm, dtype=torch.long),
    torch.tensor(act_sign, dtype=torch.float32),
  )


class _TwinMirrorTorchGRUModel(nn.Module):
  """TorchScript-friendly Twin-Mirror GRU wrapper."""

  hidden_state: torch.Tensor
  enc_mirror_perm: torch.Tensor
  enc_mirror_sign: torch.Tensor
  act_mirror_perm: torch.Tensor
  act_mirror_sign: torch.Tensor

  def __init__(self, model: FootstepActor) -> None:
    super().__init__()
    self.encoder = copy.deepcopy(model.encoder)
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.rnn = copy.deepcopy(model.rnn.rnn)
    self.mlp = copy.deepcopy(model.mlp)
    if model.distribution is not None:
      self.deterministic_output = model.distribution.as_deterministic_output_module()
    else:
      self.deterministic_output = nn.Identity()
    self.rnn.cpu()
    self.register_buffer("enc_mirror_perm", model._enc_mirror_perm.detach().cpu().clone())
    self.register_buffer("enc_mirror_sign", model._enc_mirror_sign.detach().cpu().clone())
    self.register_buffer("act_mirror_perm", model._act_mirror_perm.detach().cpu().clone())
    self.register_buffer("act_mirror_sign", model._act_mirror_sign.detach().cpu().clone())
    self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, 2 * self.rnn.hidden_size))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    enc_orig = self.encoder(x)
    enc_mirr = enc_orig.index_select(-1, self.enc_mirror_perm) * self.enc_mirror_sign
    norm_pair = self.obs_normalizer(torch.cat((enc_orig, enc_mirr), dim=0))
    h_orig, h_mirr = self.hidden_state.chunk(2, dim=-1)
    h_pair = torch.cat((h_orig, h_mirr), dim=1)
    out_pair, next_h_pair = self.rnn(norm_pair.unsqueeze(0), h_pair)
    next_h_orig, next_h_mirr = next_h_pair.chunk(2, dim=1)
    self.hidden_state[:] = torch.cat((next_h_orig, next_h_mirr), dim=-1)
    mlp_pair = self.mlp(out_pair.squeeze(0))
    out_orig, out_mirr = mlp_pair.chunk(2, dim=0)
    out_mirr_back = out_mirr.index_select(-1, self.act_mirror_perm) * self.act_mirror_sign
    return self.deterministic_output(0.5 * (out_orig + out_mirr_back))

  @torch.jit.export
  def reset(self) -> None:
    self.hidden_state[:] = 0.0


class _TwinMirrorOnnxRNNModel(nn.Module):
  """ONNX-exportable Twin-Mirror GRU wrapper."""

  is_recurrent: bool = True
  enc_mirror_perm: torch.Tensor
  enc_mirror_sign: torch.Tensor
  act_mirror_perm: torch.Tensor
  act_mirror_sign: torch.Tensor

  def __init__(self, model: FootstepActor, verbose: bool = False) -> None:
    super().__init__()
    self.verbose = verbose
    self.encoder = copy.deepcopy(model.encoder)
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.rnn = copy.deepcopy(model.rnn.rnn)
    self.mlp = copy.deepcopy(model.mlp)
    if model.distribution is not None:
      self.deterministic_output = model.distribution.as_deterministic_output_module()
    else:
      self.deterministic_output = nn.Identity()
    self.register_buffer("enc_mirror_perm", model._enc_mirror_perm.detach().cpu().clone())
    self.register_buffer("enc_mirror_sign", model._enc_mirror_sign.detach().cpu().clone())
    self.register_buffer("act_mirror_perm", model._act_mirror_perm.detach().cpu().clone())
    self.register_buffer("act_mirror_sign", model._act_mirror_sign.detach().cpu().clone())
    self.input_size = model.raw_obs_dim
    self.hidden_size = 2 * int(self.rnn.hidden_size)
    self.num_layers = int(self.rnn.num_layers)

  def forward(
    self, obs: torch.Tensor, h_in: torch.Tensor, c_in: torch.Tensor | None = None
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    del c_in
    enc_orig = self.encoder(obs)
    enc_mirr = enc_orig.index_select(-1, self.enc_mirror_perm) * self.enc_mirror_sign
    norm_orig = self.obs_normalizer(enc_orig)
    norm_mirr = self.obs_normalizer(enc_mirr)
    half = self.hidden_size // 2
    h_orig = h_in[:, :, :half]
    h_mirr = h_in[:, :, half:]
    out_orig_seq, next_h_orig = self.rnn(norm_orig.unsqueeze(0), h_orig)
    out_mirr_seq, next_h_mirr = self.rnn(norm_mirr.unsqueeze(0), h_mirr)
    h_out = torch.cat((next_h_orig, next_h_mirr), dim=-1)
    out_orig = self.mlp(out_orig_seq.squeeze(0))
    out_mirr = self.mlp(out_mirr_seq.squeeze(0))
    out_mirr_back = out_mirr.index_select(-1, self.act_mirror_perm) * self.act_mirror_sign
    out = self.deterministic_output(0.5 * (out_orig + out_mirr_back))
    return out, h_out, None

  def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
    obs = torch.zeros(1, self.input_size)
    h_in = torch.zeros(self.num_layers, 1, self.hidden_size)
    return (obs, h_in)

  @property
  def input_names(self) -> list[str]:
    return ["obs", "h_in"]

  @property
  def output_names(self) -> list[str]:
    return ["actions", "h_out"]


class FootstepActor(RNNModel):
  """以原始观测作为明确部署输入边界的 rsl_rl 循环策略网络"""

  raw_obs_dim = RAW_OBS_DIM
  encoded_obs_dim = ENCODED_OBS_DIM
  default_twin_mirror = False
  _enc_mirror_perm: torch.Tensor
  _enc_mirror_sign: torch.Tensor
  _act_mirror_perm: torch.Tensor
  _act_mirror_sign: torch.Tensor

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (256, 256, 128),
    activation: str = "elu",
    obs_normalization: bool = True,
    distribution_cfg: dict | None = None,
    rnn_type: str = "gru",
    rnn_hidden_dim: int = 64,
    rnn_num_layers: int = 1,
    default_joint_pos: Sequence[float] | None = None,
    phase_cfg: FootstepPhaseCfg | dict | None = None,
    twin_mirror: bool | None = None,
  ) -> None:
    if output_dim != ACTION_DIM:
      raise ValueError("FootstepActor controls exactly 15 leg and waist joints")
    if rnn_type != "gru":
      raise ValueError("FootstepActor requires a GRU")
    self.twin_mirror = self.default_twin_mirror if twin_mirror is None else bool(twin_mirror)
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
    include_theta = self.raw_obs_dim == 79
    self.encoder = FootstepObservationEncoder(default_joint_pos, include_theta=include_theta)
    enc_perm, enc_sign, act_perm, act_sign = build_footstep_mirror_buffers(include_theta=include_theta)
    self.register_buffer("_enc_mirror_perm", enc_perm, persistent=False)
    self.register_buffer("_enc_mirror_sign", enc_sign, persistent=False)
    self.register_buffer("_act_mirror_perm", act_perm, persistent=False)
    self.register_buffer("_act_mirror_sign", act_sign, persistent=False)
    self.phase_cfg = resolve_phase_cfg(phase_cfg=FootstepPhaseCfg(**phase_cfg) if isinstance(phase_cfg, dict) else phase_cfg)
    for name, values in nominal_action_parameters().items():
      self.register_buffer(f"action_{name}", torch.tensor(values, dtype=torch.float32))

  @property
  def hidden_state_dim(self) -> int:
    rnn_module: Any = self.rnn.rnn
    return int(rnn_module.hidden_size) * (2 if self.twin_mirror else 1)

  def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
    groups, raw_dim = super()._get_obs_dim(obs, obs_groups, obs_set)
    if len(groups) != 1 or raw_dim != self.raw_obs_dim:
      raise ValueError(f"FootstepActor requires one observation group with {self.raw_obs_dim} raw features")
    return groups, self.encoded_obs_dim

  def get_latent(self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None) -> torch.Tensor:
    raw = obs[self.obs_groups[0]]
    if not self.twin_mirror:
      features = self.obs_normalizer(self.encoder(raw))
      return self.rnn(features, masks, hidden_state).squeeze(0)
    enc_orig = self.encoder(raw)
    enc_mirr = enc_orig.index_select(-1, self._enc_mirror_perm) * self._enc_mirror_sign
    if masks is not None:
      if hidden_state is None or not isinstance(hidden_state, torch.Tensor):
        raise ValueError("Hidden states not passed to RNN module during policy update")
      h_orig, h_mirr = hidden_state.chunk(2, dim=-1)
      h_pair = torch.cat((h_orig, h_mirr), dim=1).contiguous()
      norm_pair = self.obs_normalizer(torch.cat((enc_orig, enc_mirr), dim=1))
      out_pair, _ = self.rnn.rnn(norm_pair, h_pair)
      out_orig, out_mirr = out_pair.chunk(2, dim=1)
      lat_orig = unpad_trajectories(out_orig, masks)
      lat_mirr = unpad_trajectories(out_mirr, masks)
      return torch.cat((lat_orig, lat_mirr), dim=-1)
    h_in = self.rnn.hidden_state if hidden_state is None else hidden_state
    if h_in is None:
      h_pair = None
    else:
      if not isinstance(h_in, torch.Tensor):
        raise TypeError("Expected GRU hidden state tensor")
      h_orig, h_mirr = h_in.chunk(2, dim=-1)
      h_pair = torch.cat((h_orig, h_mirr), dim=1).contiguous()
    norm_pair = self.obs_normalizer(torch.cat((enc_orig, enc_mirr), dim=0))
    out_pair, next_h_pair = self.rnn.rnn(norm_pair.unsqueeze(0), h_pair)
    next_h_orig, next_h_mirr = next_h_pair.chunk(2, dim=1)
    self.rnn.hidden_state = torch.cat((next_h_orig, next_h_mirr), dim=-1)
    lat_orig, lat_mirr = out_pair.squeeze(0).chunk(2, dim=0)
    return torch.cat((lat_orig, lat_mirr), dim=-1)

  def forward(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
    stochastic_output: bool = False,
  ) -> torch.Tensor:
    if not self.twin_mirror:
      return super().forward(obs, masks=masks, hidden_state=hidden_state, stochastic_output=stochastic_output)
    latent = self.get_latent(obs, masks, hidden_state)
    lat_orig, lat_mirr = latent.chunk(2, dim=-1)
    mlp_pair = self.mlp(torch.stack((lat_orig, lat_mirr), dim=0))
    out_orig, out_mirr = mlp_pair[0], mlp_pair[1]
    out_mirr_back = out_mirr.index_select(-1, self._act_mirror_perm) * self._act_mirror_sign
    mlp_output = 0.5 * (out_orig + out_mirr_back)
    if self.distribution is not None:
      if stochastic_output:
        self.distribution.update(mlp_output)
        return self.distribution.sample()
      return self.distribution.deterministic_output(mlp_output)
    return mlp_output

  def update_normalization(self, obs: TensorDict) -> None:
    if self.obs_normalization:
      features = self.encoder(obs[self.obs_groups[0]]).reshape(-1, self.encoded_obs_dim)
      normalizer: Any = self.obs_normalizer
      if self.twin_mirror:
        mirrored = features.index_select(-1, self._enc_mirror_perm) * self._enc_mirror_sign
        normalizer.update(torch.cat((features, mirrored), dim=0))
      else:
        normalizer.update(features)

  def as_jit(self) -> nn.Module:
    if self.twin_mirror:
      return _TwinMirrorTorchGRUModel(self)
    exported: Any = super().as_jit()
    exported.obs_normalizer = nn.Sequential(copy.deepcopy(self.encoder), exported.obs_normalizer)
    return exported

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    if self.twin_mirror:
      return _TwinMirrorOnnxRNNModel(self, verbose=verbose)
    exported: Any = super().as_onnx(verbose)
    exported.obs_normalizer = nn.Sequential(copy.deepcopy(self.encoder), exported.obs_normalizer)
    exported.input_size = self.raw_obs_dim
    return exported

  def deployment_contract(self) -> dict:
    rnn_module: Any = self.rnn.rnn
    hidden_mult = 2 if self.twin_mirror else 1
    hidden_shape = [int(rnn_module.num_layers), 1, int(rnn_module.hidden_size) * hidden_mult]
    defaults = self.encoder.default_joint_pos.detach().cpu().tolist()
    return {
      "version": CONTRACT_VERSION,
      "model_class": "g1_lower_rl.rl.footstep_model:FootstepActor",
      "control_dt_s": 0.02,
      "dtype": "float32",
      "inputs": {"obs": [1, self.raw_obs_dim], "h_in": hidden_shape},
      "outputs": {"actions": [1, ACTION_DIM], "h_out": hidden_shape},
      "encoded_obs_dim": self.encoded_obs_dim,
      "observation_layout": [
        {"name": name, "start": start, "stop": stop, "units": units}
        for name, start, stop, units in OBSERVATION_LAYOUT
      ],
      "joint_names": list(WHOLE_BODY_JOINTS),
      "default_joint_pos": defaults,
      "raw_joint_pos": "absolute calibrated encoder angles; default subtraction is inside the model",
      "imu": {
        "pelvis": "angular velocity and unit gravity in calibrated pelvis link axes",
        "torso": "angular velocity and unit gravity in calibrated torso link axes",
        "gravity": "R_world_from_link.T @ [0, 0, -1]; not raw accelerometer readings",
      },
      "theta": {
        "role": "command",
        "raw_range_rad": [-math.pi, math.pi],
        "upper_bound_exclusive": True,
        "encoded_fields": ["theta"],
        "definition": "desired torso yaw offset from the circular mean of actual left/right sole headings; not the measured torso offset",
        "reference": "actual measured sole headings, not queued landing or support targets; all headings in one common frame",
        "positive": "desired torso counterclockwise relative to mean foot heading about +z",
        "degenerate_mean": "use left foot heading when the summed unit heading vector norm is <= 1e-6",
        "standing": "continue smoothly updating the desired theta at f=0; do not freeze with the planner anchor",
        "generation": "external uniformly sampled angle goals and positive speeds; rate-limit the current command without jumps when resampling",
      },
      "footsteps": {
        "slots": list(FOOTSTEP_SLOTS),
        "raw_fields": ["dx", "dy", "dtheta"],
        "encoded_fields": ["dx", "dy", "sin(dtheta)", "cos(dtheta)"],
        "reference_frame": "one common frozen stance-foot heading frame; x forward, y left, z up",
        "heading": "foot sole yaw, positive counterclockwise about +z",
        "support": "not an actor input; retained only by the physical planner and critic",
        "preview": "first unconsumed queued landing per foot, including the currently executing swing target; only two future steps",
        "side_order": "fixed left/right within each pair; never chronological order",
        "update": "liftoff leaves all world slots unchanged; at planned touchdown move that foot's next to support and advance its next goal",
        "coordinates": "all supplied poses relative to the same frozen anchor, not chained deltas",
        "reanchor": "transform unchanged physical targets using an estimated new anchor",
        "requires_external_coordinate_conversion": True,
        "all_slots_required": True,
        "zero_progress": "repeat each foot's fixed landing pose; f>0 marches, f=0 holds double support",
        "standing_slots": "at f=0 use [left_hold, right_hold]; holds do not follow measured foot drift",
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


class LegacyFootstepActor(FootstepActor):
  raw_obs_dim = 78
  encoded_obs_dim = 81
  default_twin_mirror = False

  def deployment_contract(self) -> dict:
    contract = super().deployment_contract()
    contract["version"] = "g1_footstep_gru_v5_next_only"
    contract["model_class"] = "g1_lower_rl.rl.footstep_model:LegacyFootstepActor"
    contract["observation_layout"] = [field for field in contract["observation_layout"] if field["name"] != "theta"]
    contract.pop("theta")
    return contract


def export_footstep_policy(model: FootstepActor, path: str | Path) -> Path:
  """导出确定性 ONNX 模型及配套 JSON 契约，不修改当前策略网络"""
  import onnx

  destination = Path(path)
  if destination.suffix != ".onnx":
    raise ValueError("Footstep export path must end in .onnx")
  destination.parent.mkdir(parents=True, exist_ok=True)
  exported: Any = model.as_onnx().cpu().eval()
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
  hidden_dims: tuple[int, ...] = (256, 256, 128)
  activation: str = "elu"
  obs_normalization: bool = True
  rnn_type: str | None = "gru"
  rnn_hidden_dim: int = 64
  rnn_num_layers: int = 1
  default_joint_pos: tuple[float, ...] | None = None
  phase_cfg: FootstepPhaseCfg = field(default_factory=FootstepPhaseCfg)
  twin_mirror: bool = False
  distribution_cfg: dict | None = field(
    default_factory=lambda: {
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    }
  )
