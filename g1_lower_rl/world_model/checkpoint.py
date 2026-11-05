"""Portable world-model weights and simulator-independent checkpoint utilities."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, is_dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import torch

from g1_lower_rl.world_model.model import CONTRACT_VERSION, PositionWorldModel, PositionWorldModelCfg

CHECKPOINT_VERSION = 2
PRETRAINED_KIND = "pretrained_initialization"


def pretrained_checkpoint_path() -> Path:
  """The package-local training starting point; never download weights."""
  return Path(__file__).parent / "checkpoints" / "g1_wm.pt"


def primitive_metadata(value: Any) -> Any:
  """Keep config/provenance safe for JSON and Torch weights-only deserialization."""
  if value is None or isinstance(value, (str, bool)):
    return value
  if isinstance(value, Path):
    return str(value)
  if is_dataclass(value) and not isinstance(value, type):
    return primitive_metadata(asdict(value))
  if isinstance(value, Integral):
    return int(value)
  if isinstance(value, Real):
    return float(value)
  if isinstance(value, dict):
    return {primitive_metadata(key): primitive_metadata(item) for key, item in value.items()}
  if isinstance(value, (tuple, list)):
    return [primitive_metadata(item) for item in value]
  raise TypeError(f"Unsupported nonprimitive checkpoint metadata: {type(value).__name__}")


def finite_tree(value: Any) -> bool:
  if isinstance(value, torch.Tensor):
    return bool(torch.isfinite(value).all()) if value.is_floating_point() or value.is_complex() else True
  if isinstance(value, dict):
    return all(finite_tree(item) for item in value.values())
  if isinstance(value, (tuple, list)):
    return all(finite_tree(item) for item in value)
  return not isinstance(value, float) or math.isfinite(value)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
  temporary = path.with_name(f".{path.name}.{os.getpid()}.pending")
  try:
    with temporary.open("w") as stream:
      json.dump(payload, stream, indent=2, allow_nan=False)
      stream.write("\n")
      stream.flush()
      os.fsync(stream.fileno())
    temporary.replace(path)
  finally:
    temporary.unlink(missing_ok=True)


def atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
  if not finite_tree(payload):
    raise FloatingPointError("Refusing to persist nonfinite checkpoint state")
  temporary = path.with_name(f".{path.name}.{os.getpid()}.pending")
  try:
    with temporary.open("wb") as stream:
      torch.save(payload, stream)
      stream.flush()
      os.fsync(stream.fileno())
    temporary.replace(path)
  finally:
    temporary.unlink(missing_ok=True)


def read_checkpoint(
  path: str | Path | None = None,
  device: str | torch.device = "cpu",
) -> dict[str, Any]:
  """Read a trusted training snapshot or portable initialization with weights_only."""
  resolved = Path(path) if path is not None else pretrained_checkpoint_path()
  if not resolved.is_file():
    raise FileNotFoundError(
      f"No position-control WM v3 at {resolved}; download a v3 checkpoint "
      "or supply an explicit path. A fixed29 position-control contract v3 checkpoint is required."
    )
  payload = torch.load(resolved, map_location=device, weights_only=True)
  if not isinstance(payload, dict) or payload.get("format_version") != CHECKPOINT_VERSION:
    raise ValueError("Unsupported WM checkpoint version; position-control contract v3 requires checkpoint format_version=2")
  contract = payload.get("model", {}).get("_extra_state", {})
  if contract.get("contract_version") != CONTRACT_VERSION:
    raise ValueError(
      "Incompatible WM physical contract: v3 position-control weights are required; legacy weights cannot resume"
    )
  if not finite_tree(payload):
    raise FloatingPointError("Checkpoint contains nonfinite state")
  return payload


def load_world_model(
  path: str | Path | None = None,
  device: str | torch.device = "cpu",
) -> PositionWorldModel:
  """Load shared pretrained weights by default, retaining gradients for fine-tuning."""
  return _model_from_checkpoint(read_checkpoint(path, device), device)


def _model_from_checkpoint(payload: dict[str, Any], device: str | torch.device) -> PositionWorldModel:
  mean = payload["model"]["input_mean"]
  if not isinstance(mean, torch.Tensor) or not mean.is_floating_point():
    raise ValueError("Checkpoint normalization must use a floating-point dtype")
  model = PositionWorldModel(PositionWorldModelCfg(**payload["model_config"])).to(device=device, dtype=mean.dtype)
  model.load_state_dict(payload["model"], strict=True)
  if not bool(model.normalization_ready):
    raise RuntimeError("Checkpoint has no fitted normalization")
  if any(bool((scale <= 0).any()) for scale in (model.input_scale, model.delta_scale, model.force_scale, model.state_scale)):
    raise ValueError("Checkpoint normalization scales must be positive")
  return model.eval()


def _portable_provenance(provenance: list[dict[str, Any]]) -> list[dict[str, Any]]:
  collector_fields = (
    "collector_version",
    "contract_version",
    "teacher_kind",
    "teacher_iteration",
    "teacher_reference_dim",
    "teacher_frozen",
    "static_template",
    "controller_template",
    "integration",
    "teacher_controller",
    "not_native_position_actuator_equivalent",
    "native_position_actuators",
    "collection_overrides",
    "physics_dt",
    "control_dt",
    "force_label",
    "force_sign",
    "joint_names",
    "body_names",
    "indices",
    "solver",
    "condim",
    "coverage",
    "fixed_actuator_parameters",
  )
  result = []
  for rank in provenance:
    original = rank["collector"]
    collector = {key: original[key] for key in collector_fields if key in original}
    if "corpus" in original:
      corpus_fields = (
        "clips", "frames",
        "train_clips", "validation_clips", "family_split_disjoint", "content_split_disjoint",
        "joint_projection", "gripper_state", "body_reference", "sampling", "bin_semantics",
      )
      collector["corpus"] = {key: original["corpus"][key] for key in corpus_fields if key in original["corpus"]}
    if original.get("corpus_manifest") is not None:
      manifest = original["corpus_manifest"]
      contents = manifest["contents"]
      if "frame_width" in contents:
        manifest_fields = (
          "version", "dtype", "frame_width", "fps", "frames", "bytes",
          "joint_names", "source_body_names", "body_names", "split_seed", "validation_fraction",
          "limited_to_clips", "evaluation_caveat", "clip_count", "split_counts",
        )
      else:
        manifest_fields = ("version", "seed", "selected_clips", "source_clips", "selected_bytes")
      missing = set(manifest_fields) - contents.keys()
      if missing:
        raise ValueError(f"Corpus manifest is missing portable provenance fields: {sorted(missing)}")
      collector["corpus_manifest"] = {
        "contents": {key: contents[key] for key in manifest_fields},
      }
    result.append({key: rank[key] for key in ("rank", "seed", "validation_env_indices", "training_env_indices") if key in rank})
    result[-1]["collector"] = collector
  return primitive_metadata(result)


def export_pretrained_checkpoint(
  source: str | Path,
  destination: str | Path,
  *,
  overwrite: bool = False,
) -> Path:
  """Export full-precision model initialization and physical metadata."""
  source, destination = Path(source), Path(destination)
  manifest_path = destination.with_suffix(".json")
  if source.resolve() == destination.resolve():
    raise ValueError("A pretrained export must not overwrite its training source")
  if not overwrite and (destination.exists() or manifest_path.exists()):
    raise FileExistsError("Pretrained destination exists; choose a new version or explicitly overwrite")
  payload = read_checkpoint(source)
  if payload.get("artifact_kind") == PRETRAINED_KIND:
    raise ValueError("Export requires a full training snapshot, not an already exported initialization")
  _model_from_checkpoint(payload, "cpu")
  retained = {
    key: payload[key] for key in ("format_version", "model_config", "model", "metrics") if key in payload
  }
  retained["artifact_kind"] = PRETRAINED_KIND
  retained["provenance"] = _portable_provenance(payload["provenance"])
  retained["initialization"] = {
    "source_run": source.parent.name,
    "source_checkpoint": source.name,
    "source_timestamp_utc": payload["timestamp_utc"],
    "source_update": payload["update"],
    "source_transitions": payload["transitions"],
    "selection_metric": payload["best_metric"],
    "selection_score": payload["best_score"],
    "optimizer_included": False,
    "new_training_updates": 0,
  }
  destination.parent.mkdir(parents=True, exist_ok=True)
  atomic_checkpoint(destination, retained)
  atomic_json(
    manifest_path,
    {
      "format_version": CHECKPOINT_VERSION,
      "artifact_kind": PRETRAINED_KIND,
      "file": destination.name,
      "bytes": destination.stat().st_size,
      "model_config": retained["model_config"],
      "physical_contract": retained["model"]["_extra_state"],
      "initialization": retained["initialization"],
      "metrics": retained.get("metrics"),
      "physics": [
        {
          key: rank["collector"][key]
          for key in ("physics_dt", "control_dt", "integration", "solver", "condim")
          if key in rank["collector"]
        }
        for rank in retained["provenance"]
      ],
      "evaluation": {
        "open_loop_endpoints_seconds": payload["config"]["eval_horizons_seconds"],
        "closed_loop_policy_validation": "Not included in this training snapshot",
      },
      "scope": "G1 fixed29 position-control v3; use the recorded geometry, fixed actuator parameters and held20ms targets",
      "limitations": [
        "A warm start, not a complete policy trainer or a guarantee of native prediction/gradient accuracy.",
        "Consult collector provenance for held-out environments, motion exposure and fixed-validation scope.",
        "Open-loop validation does not establish 640ms closed-loop policy-imagination accuracy.",
        "Check per-horizon state skill and contact false-positive rates before policy optimization.",
        "Force-derived contact is not identical to native found events.",
      ],
    },
  )
  return destination
