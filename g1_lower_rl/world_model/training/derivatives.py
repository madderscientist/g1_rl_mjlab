"""Separate native 20ms position-target secants for training and validation."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from g1_lower_rl.world_model.model import (
  CONTRACT_VERSION,
  CONTROL_DT,
  DELTA_FIELDS,
  INPUT_FIELDS,
  JOINT_DIM,
  DynamicsPrediction,
  PositionWorldModel,
  state_delta,
)

if TYPE_CHECKING:
  from g1_lower_rl.world_model.collection import MotionCollector

logger = logging.getLogger(__name__)
SECANT_CONVENTION = "v3_position_target_radians_state_delta(minus,plus)/(2*epsilon)"
SECANT_EPSILONS = (0.005, 0.02)
SNAPSHOT_FIELDS = (
  "qpos",
  "qvel",
  "qacc",
  "qacc_warmstart",
  "ctrl",
  "qfrc_applied",
  "xfrc_applied",
  "time",
  "act",
  "mocap_pos",
  "mocap_quat",
  "overflow",
)


def snapshot(collector: MotionCollector) -> dict[str, torch.Tensor]:
  return {name: getattr(collector.env.sim.data, name).clone() for name in SNAPSHOT_FIELDS}


def restore(collector: MotionCollector, values: dict[str, torch.Tensor]) -> None:
  for name, value in values.items():
    getattr(collector.env.sim.data, name).copy_(value)
  collector._original_forward()
  collector.env.sim.data.qacc_warmstart.copy_(values["qacc_warmstart"])


@torch.no_grad()
def native_secants(
  collector: MotionCollector,
  before: dict[str, torch.Tensor],
  inputs: torch.Tensor,
  valid: torch.Tensor,
  expected_next_state: torch.Tensor,
  *,
  seed: int,
  strict: bool = True,
  repeatability_exclusion_limit: float = 0.0,
) -> dict:
  if not 0 <= repeatability_exclusion_limit < 1:
    raise ValueError("Native derivative repeatability exclusion limit must be in [0,1)")
  end = snapshot(collector)
  end_value = collector._state().clone()
  generator = torch.Generator(device=inputs.device).manual_seed(seed)
  directions = (
    torch.randint(2, (4, *inputs[:, INPUT_FIELDS["joint_target"]].shape), device=inputs.device, generator=generator) * 2 - 1
  ).to(inputs.dtype) / JOINT_DIM**0.5
  target = inputs[:, INPUT_FIELDS["joint_target"]]
  model = collector.env.sim.mj_model
  limited = torch.tensor(model.actuator_ctrllimited[collector.indices["actuator"]], device=target.device, dtype=torch.bool)
  limits = torch.tensor(model.actuator_ctrlrange[collector.indices["actuator"]], device=target.device, dtype=target.dtype)
  interior = ~limited | ((target > limits[:, 0] + max(SECANT_EPSILONS)) & (target < limits[:, 1] - max(SECANT_EPSILONS)))
  directions *= interior
  directions /= directions.norm(dim=-1, keepdim=True).clamp_min(1)
  mask = valid & interior.any(-1)
  if not mask.any():
    raise ValueError("No native position-target secant states away from target limits")

  def run(value):
    restore(collector, before)
    collector.env.sim.data.ctrl[:, collector._actuator_ids] = value
    for _ in range(round(CONTROL_DT / collector.config.physics_dt)):
      collector._original_step()
    return collector._state().clone()

  try:
    restore(collector, before)
    initial_state = collector._state().clone()
    baseline = run(target)
    repeat = run(target)
    repeat_per_state = (repeat - baseline).abs().amax(-1)
    baseline_per_state = (baseline - expected_next_state).abs().amax(-1)
    stable = torch.isfinite(baseline).all(-1) & torch.isfinite(repeat).all(-1)
    stable &= (repeat_per_state <= 1e-4) & (baseline_per_state <= 1e-4)
    excluded = int((mask & ~stable).sum())
    if excluded and (not strict or repeatability_exclusion_limit > 0):
      if strict and excluded / int(mask.sum()) > repeatability_exclusion_limit:
        raise RuntimeError(f"Native derivative repeatability exclusions exceed limit: {excluded}/{int(mask.sum())}")
      logger.warning(
        "Excluding %d/%d nonrepeatable native %s secant states at unchanged 1e-4 tolerance",
        excluded, int(mask.sum()), "validation" if strict else "training",
      )
      mask &= stable
      if not mask.any():
        raise RuntimeError("No repeatable native training secant states")
    repeat_error = float((repeat[mask] - baseline[mask]).abs().max())
    baseline_error = float((baseline[mask] - expected_next_state[mask]).abs().max())
    if not torch.isfinite(baseline[mask]).all() or repeat_error > 1e-4 or baseline_error > 1e-4:
      raise RuntimeError(f"Native derivative replay mismatch: repeat={repeat_error}, collected={baseline_error}")
    labels = {}
    for epsilon in SECANT_EPSILONS:
      labels[str(epsilon)] = torch.stack(
        [
          state_delta(run(target - epsilon * direction), run(target + epsilon * direction), validate=False)[
            :, DELTA_FIELDS["root_linear_velocity"].start :
          ]
          / (2 * epsilon)
          for direction in directions
        ]
      )
    if any(not torch.isfinite(value[:, mask]).all() for value in labels.values()):
      raise FloatingPointError("Native derivative labels are nonfinite")
    return {
      "state": initial_state[mask].clone(),
      "inputs": inputs[mask].clone(),
      "directions": directions[:, mask].clone(),
      "env_indices": mask.nonzero().flatten(),
      "labels": {key: value[:, mask].clone() for key, value in labels.items()},
      "native_repeat_max_abs": repeat_error,
      "valid_states": int(mask.sum()),
      "native_collected_max_abs": baseline_error,
      "excluded_nonrepeatable_states": excluded,
      "total_probe_states": int(valid.sum()),
      "excluded_saturated_states": int((valid & ~interior.any(-1)).sum()),
      "saturated_joint_fraction": float((~interior[valid]).float().mean()),
      "scope": "one-control-tick actual-position-target-to-root/joint velocity secants; not full actor gradients",
      "contract_version": CONTRACT_VERSION,
      "secant_convention": SECANT_CONVENTION,
      "label_partition": "validation" if strict else "training",
      "physics_dt": collector.config.physics_dt,
      "control_dt": CONTROL_DT,
    }
  finally:
    restore(collector, end)
    if not torch.allclose(collector._state(), end_value, rtol=0, atol=1e-5):
      raise RuntimeError("Native end state was not restored after secant validation")


@torch.no_grad()
def derivative_sums(model: PositionWorldModel, probe: dict) -> dict[str, torch.Tensor]:
  if probe.get("secant_convention") != SECANT_CONVENTION or probe.get("contract_version") != CONTRACT_VERSION:
    raise ValueError("Unsupported native secant sign convention")
  inputs, directions = probe["inputs"], probe["directions"]
  result = {}
  for key, native in probe["labels"].items():
    epsilon = float(key)
    x = inputs.unsqueeze(0).expand(directions.shape[0], -1, -1).clone()
    plus, minus = x.clone(), x.clone()
    plus[..., INPUT_FIELDS["joint_target"]] += epsilon * directions
    minus[..., INPUT_FIELDS["joint_target"]] -= epsilon * directions
    states = probe["state"].unsqueeze(0).expand(directions.shape[0], -1, -1).flatten(0, 1)
    context = x[..., INPUT_FIELDS["context"]].flatten(0, 1)
    learned = state_delta(
      model(states, minus[..., INPUT_FIELDS["joint_target"]].flatten(0, 1), context, validate=False).state,
      model(states, plus[..., INPUT_FIELDS["joint_target"]].flatten(0, 1), context, validate=False).state,
      validate=False,
    )[:, DELTA_FIELDS["root_linear_velocity"].start :].reshape_as(native) / (2 * epsilon)
    if not torch.isfinite(learned).all():
      raise FloatingPointError("Nonfinite learned position-target derivative")
    result[f"derivative/{key}/dot"] = (learned.double() * native.double()).sum()
    result[f"derivative/{key}/learned_sq"] = learned.double().square().sum()
    result[f"derivative/{key}/native_sq"] = native.double().square().sum()
    result[f"derivative/{key}/error_sq"] = (learned.double() - native.double()).square().sum()
  return result


def derivative_metrics(sums: dict[str, torch.Tensor]) -> dict[str, float]:
  result = {}
  for epsilon in map(str, SECANT_EPSILONS):
    prefix = f"derivative/{epsilon}"
    native = sums[f"{prefix}/native_sq"]
    learned = sums[f"{prefix}/learned_sq"]
    if native <= 0 or learned <= 0:
      raise ValueError("Derivative comparison requires nonzero native and learned response")
    result[f"{prefix}/cosine"] = float(sums[f"{prefix}/dot"] / (native * learned).sqrt())
    result[f"{prefix}/relative_l2"] = float((sums[f"{prefix}/error_sq"] / native).sqrt())
  return result


def secant_loss(
  forward: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], DynamicsPrediction],
  probe: dict,
  delta_scale: torch.Tensor,
  batch_size: int,
  generator: torch.Generator,
) -> torch.Tensor:
  if (
    probe.get("secant_convention") != SECANT_CONVENTION
    or probe.get("contract_version") != CONTRACT_VERSION
    or probe.get("label_partition") != "training"
  ):
    raise ValueError("Secant supervision requires independent training labels with the recorded sign convention")
  directions, states = probe["directions"].shape[:2]
  selected = torch.randint(directions * states, (batch_size,), device=delta_scale.device, generator=generator)
  direction, state = selected // states, selected % states
  inputs = probe["inputs"][state]
  epsilon = max(SECANT_EPSILONS)
  plus, minus = inputs.clone(), inputs.clone()
  plus[:, INPUT_FIELDS["joint_target"]] += epsilon * probe["directions"][direction, state]
  minus[:, INPUT_FIELDS["joint_target"]] -= epsilon * probe["directions"][direction, state]
  initial, context = probe["state"][state], inputs[:, INPUT_FIELDS["context"]]
  learned = state_delta(
    forward(initial, minus[:, INPUT_FIELDS["joint_target"]], context).state,
    forward(initial, plus[:, INPUT_FIELDS["joint_target"]], context).state,
    validate=False,
  )[:, DELTA_FIELDS["root_linear_velocity"].start :]
  learned = learned / (2 * epsilon)
  native = probe["labels"][str(epsilon)][direction, state]
  return ((learned - native) / delta_scale[DELTA_FIELDS["root_linear_velocity"].start :]).square().mean()
