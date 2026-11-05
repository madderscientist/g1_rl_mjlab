"""Factories for checkpoint-captured geometry and optional external GC preprocessing."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import torch

from g1_lower_rl.world_model.model import TORQUE_DIM

from .controller import ARM_JOINT_INDICES, SubstepTorqueController
from .kinematics import StaticKinematicTemplate, TorchKinematics
from .rollout import ControllerContext


class ControllerSetup(NamedTuple):
  kinematics: TorchKinematics
  controller: SubstepTorqueController
  context: ControllerContext


def controller_from_metadata(
  controller_template: Mapping[str, Any],
  static_template: StaticKinematicTemplate | Mapping[str, Any],
  *,
  device: str | torch.device = "cpu",
  dtype: torch.dtype = torch.float32,
) -> ControllerSetup:
  """Build nominal external control from collector.metadata['controller_template'].

  Required exact collector keys: kp_nominal[29], kd_nominal[29],
  effort_range_nominal[29,2], gravity_compensation_nominal_kp[14].
  GC14 is left shoulder/elbow/wrist7 then right7.
  No random samples are drawn: effective kp/kd/effort initially use nominal
  values, encoder bias is zero, and GC gain is one. Use dataclasses.replace
  on the returned context for explicit measured/randomized per-environment values.
  """
  if not dtype.is_floating_point:
    raise ValueError("Controller dtype must be floating point")
  template = (
    static_template
    if isinstance(static_template, StaticKinematicTemplate)
    else StaticKinematicTemplate.from_dict(dict(static_template))
  )

  def tensor(key: str, shape: tuple[int, ...]) -> torch.Tensor:
    if key not in controller_template:
      raise ValueError(f"Controller provenance is missing {key}")
    value = torch.as_tensor(controller_template[key], device=device, dtype=dtype)
    if value.shape != shape or not torch.isfinite(value).all():
      raise ValueError(f"{key} must contain finite values with shape {shape}")
    return value

  kp = tensor("kp_nominal", (TORQUE_DIM,))
  kd = tensor("kd_nominal", (TORQUE_DIM,))
  limits = tensor("effort_range_nominal", (TORQUE_DIM, 2))
  gc14 = tensor("gravity_compensation_nominal_kp", (14,))
  if (kp < 0).any() or (kd < 0).any() or (gc14 <= 0).any():
    raise ValueError("PD gains must be nonnegative and GC nominal gains strictly positive")
  if (limits[:, 0] > limits[:, 1]).any():
    raise ValueError("Nominal effort limits must be ordered lower/upper")
  indices = torch.tensor(ARM_JOINT_INDICES, device=device)
  nominal_gc = torch.ones(TORQUE_DIM, device=device, dtype=dtype).index_copy(0, indices, gc14)
  kinematics = TorchKinematics(template).to(device=device, dtype=dtype)
  controller = SubstepTorqueController(kinematics, nominal_gc).to(device=device, dtype=dtype)
  context = ControllerContext(kp, kd, limits, torch.zeros_like(kp), torch.ones_like(kp))
  return ControllerSetup(kinematics, controller, context)
