"""CPU checkpoint/three-task gradient check, without simulation or optimizer steps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from g1_lower_rl.world_model.checkpoint import _model_from_checkpoint, read_checkpoint
from g1_lower_rl.world_model.model import INPUT_FIELDS, JOINT_DIM, STATE_DIM, STATE_FEATURE_FIELDS, STATE_FIELDS

from .contact import ContactMemory
from .kinematics import StaticKinematicTemplate, TorchKinematics
from .math import yaw
from .rollout import PositionTaskRollout
from .tasks import (
  FootstepGoal,
  FootstepTrackingAdapter,
  LowerBodyVelocityAdapter,
  MotionGoal,
  MotionTrackingAdapter,
  VelocityGoal,
)


def smoke_checkpoint(
  checkpoint: str | Path,
  *,
  rank: int = 0,
  transition: str | Path | None = None,
  steps: int = 1,
) -> dict[str, dict[str, float]]:
  """Verify API/gradients, not accuracy; optional sample has state71 and context48."""
  if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
    raise ValueError("steps must be a positive number of20ms control ticks")
  payload = read_checkpoint(checkpoint)
  model = _model_from_checkpoint(payload, "cpu").requires_grad_(False)
  dtype = model.input_mean.dtype
  provenance = payload["provenance"]
  if not isinstance(provenance, list) or not 0 <= rank < len(provenance):
    raise ValueError("rank must select collector provenance")
  collector = provenance[rank]["collector"]
  if "static_template" not in collector:
    raise ValueError("Collector provenance is missing static_template")
  fk = TorchKinematics(StaticKinematicTemplate.from_dict(collector["static_template"])).to(dtype=dtype)
  if transition is None:
    state = torch.zeros(1, STATE_DIM, dtype=dtype)
    state[:, 2], state[:, 3] = 0.8, 1
    state[:, STATE_FIELDS["joint_position"]] = model.input_mean[STATE_FEATURE_FIELDS["joint_position"]]
    context = model.input_mean[INPUT_FIELDS["context"]].detach().clone().unsqueeze(0)
  else:
    sample = torch.load(transition, map_location="cpu", weights_only=True)
    state, context = sample["state"].to(dtype=dtype), sample["context"].to(dtype=dtype)
    if state.ndim == 1:
      state, context = state.unsqueeze(0), context.unsqueeze(0)
  k = fk(state)
  batch = state.shape[:-1]
  phase = state.new_zeros(batch)
  targets = torch.cat((k.foot_position[..., :2] + 0.03, yaw(k.foot_quaternion).unsqueeze(-1)), dim=-1)
  goals = {
    "motion": MotionGoal(state.clone(), window=state.unsqueeze(-2)),
    "footstep": FootstepGoal(
      targets,
      torch.zeros(*batch, 2, dtype=torch.long),
      phase,
      torch.ones_like(phase),
      state.new_zeros(*batch, 2),
    ),
    "velocity": VelocityGoal(
      state.new_tensor([0.2, 0.0, 0.1]).expand(*batch, 3),
      state[..., 2] - k.foot_position[..., 2].amin(-1) + 0.01,
      phase,
    ),
  }
  adapters = {
    "motion": MotionTrackingAdapter(fk),
    "footstep": FootstepTrackingAdapter(fk),
    "velocity": LowerBodyVelocityAdapter(fk),
  }
  results = {}
  for name, adapter in adapters.items():
    rollout = PositionTaskRollout(model, adapter)
    action = torch.full((*batch, JOINT_DIM), 0.01, dtype=dtype, requires_grad=True)
    current = state
    contact = ContactMemory.zeros(state.new_zeros(*batch, 2))
    memory = None
    for _ in range(steps):
      output = rollout(
        current,
        action,
        goals[name],
        context,
        contact,
        memory,
        action_scale=state.new_full((JOINT_DIM,), 0.1),
        action_offset=state[..., STATE_FIELDS["joint_position"]],
      )
      current, contact, memory = output.prediction.state, output.contact_memory, output.footstep_memory
    if name == "footstep":
      loss = output.signals.costs["footprint_accuracy"].mean() - output.signals.rewards["soft_swing_position"].mean()
    elif name == "velocity":
      loss = -output.signals.rewards["linear_velocity"].mean() - output.signals.rewards["height"].mean()
    else:
      loss = -output.signals.rewards["body_position"].mean() - output.signals.rewards["body_linear_velocity"].mean()
    (gradient,) = torch.autograd.grad(loss, action)
    if not torch.isfinite(output.prediction.as_tensor()).all() or not torch.isfinite(gradient).all():
      raise FloatingPointError(f"{name} produced nonfinite predictions or action gradients")
    gradient_l1 = float(gradient.abs().sum())
    if gradient_l1 == 0:
      raise RuntimeError(f"{name} action gradient is zero; inspect target clipping or learned target sensitivity")
    results[name] = {"loss": float(loss.detach()), "action_gradient_l1": gradient_l1}
  return results


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("checkpoint", type=Path)
  parser.add_argument("--rank", type=int, default=0)
  parser.add_argument("--transition", type=Path)
  parser.add_argument("--steps", type=int, default=1)
  args = parser.parse_args()
  print(json.dumps(smoke_checkpoint(args.checkpoint, rank=args.rank, transition=args.transition, steps=args.steps), indent=2))


if __name__ == "__main__":
  main()
