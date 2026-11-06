"""Export and preserve the official full-body MLP without changing its controller."""

from __future__ import annotations

import importlib
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from g1_lower_rl.tasks.motion_tracking import MOCAP_TASK
from g1_lower_rl.tasks.motion_tracking.corpus import file_sha256
from g1_lower_rl.world_model.model import JOINT_NAMES

from .official import frozen_runtime, load_mlp_checkpoint
from .position_policy import NATIVE_TERMS, REFERENCE_OFFSETS, PositionMlpActor, actor_observations


def native_module(saved: dict):
  source = Path(saved["config"]["replica_source"]).resolve(strict=True)
  for name, digest in saved["collector"]["replica_source_manifest"].items():
    if file_sha256(source / name) != digest:
      raise ValueError(f"Native simulator source changed: {name}")
  sys.path.insert(0, str(source / "src"))
  package = importlib.import_module("supermimic")
  if package.__file__ is None or Path(package.__file__).resolve().parents[2] != source:
    raise RuntimeError("A different supermimic package is already imported")
  return importlib.import_module("supermimic.envs.tracking_env_cfg")


def load_actor(saved: dict, device: str = "cpu") -> PositionMlpActor:
  actor = PositionMlpActor().to(device)
  actor.load_state_dict(saved["actor_state_dict"], strict=True)
  if not all(torch.isfinite(value).all() for value in actor.state_dict().values()):
    raise FloatingPointError("Checkpoint actor contains nonfinite weights")
  return actor.eval().requires_grad_(False)


def controller_contract(saved: dict) -> dict:
  from mjlab.entity import Entity

  from g1_lower_rl.world_model.native import fixed29_runtime

  repository = Path(__file__).resolve().parents[4]
  for name, digest in saved["source_provenance"]["files"].items():
    if name.startswith("g1_lower_rl/assets/g1_gloria/") and file_sha256(repository / name) != digest:
      raise ValueError(f"Native robot asset differs from the checkpoint: {name}")
  native = native_module(saved)
  with fixed29_runtime({"observed_initialization_counts": {"reference": 0, "half_squat": 0, "airborne": 0}}):
    cfg = native.make_env_cfg(native.SuperMimicEnvCfgArgs(
      npz_files=[], num_envs=2, obs_noise=False, reset_noise=False, domain_randomization=False,
    ))
    robot_cfg = cfg.scene.entities["robot"]
    model = Entity(robot_cfg).spec.compile()
  joint_ids = [model.joint(name).id for name in JOINT_NAMES]
  actuator_ids = [model.actuator(name).id for name in JOINT_NAMES]
  scales = cfg.actions["joint_pos"].scale
  scale, default = [], []
  for name, joint in zip(JOINT_NAMES, joint_ids, strict=True):
    matches = [value for pattern, value in scales.items() if re.fullmatch(pattern, name)]
    if len(matches) != 1:
      raise ValueError(f"Expected exactly one native action scale for {name}")
    scale.append(matches[0])
    positions = [value for pattern, value in robot_cfg.init_state.joint_pos.items() if re.fullmatch(pattern, name)]
    if len(positions) > 1:
      raise ValueError(f"Ambiguous default position for {name}")
    default.append(positions[0] if positions else float(model.qpos0[model.jnt_qposadr[joint]]))
  return {
    "joint_names": list(JOINT_NAMES), "action_scale": scale, "default_joint_pos": default,
    "target_limits": model.actuator_ctrlrange[actuator_ids].tolist(),
    "target_limited": model.actuator_ctrllimited[actuator_ids].astype(bool).tolist(),
    "kp": model.actuator_gainprm[actuator_ids, 0].tolist(),
    "kd": (-model.actuator_biasprm[actuator_ids, 2]).tolist(),
    "effort_range": model.actuator_forcerange[actuator_ids].tolist(),
    "formula": "clamp_if_limited(q_reference_current + action_scale * raw_residual - encoder_bias, target_limits)",
    "gravity_compensation": False, "training_action_std": saved["config"]["action_noise"],
    "inference_action_noise": False, "history_action": "previous raw residual, before scaling/bias/clamping",
  }


class OnnxActor(nn.Module):
  def __init__(self, actor: PositionMlpActor):
    super().__init__()
    self.actor = actor

  def forward(self, history: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    # Export plain tensors; TensorDict and Python shape guards are not part of the ONNX graph.
    offset, parts = 0, []
    for _, width in NATIVE_TERMS:
      parts.append(history[..., offset:offset + width].flatten(1))
      offset += width
    latent = torch.tanh(self.actor.encoder(reference * self.actor.reference_scale))
    return self.actor.decoder(torch.cat((latent, torch.cat(parts, -1) * self.actor.history_scale), -1))


def export_onnx(checkpoint: Path, output_dir: Path, *, native_inputs: Path | None = None) -> Path:
  import onnx
  import onnxruntime as ort

  checkpoint = checkpoint.resolve(strict=True)
  if checkpoint.is_dir():
    checkpoint /= "rank_0.pt"
  saved = load_mlp_checkpoint(checkpoint)
  actor = load_actor(saved)
  controller = controller_contract(saved)
  output_dir.mkdir(parents=True, exist_ok=True)
  path = output_dir / "policy.onnx"
  contract_path = output_dir / "policy_contract.json"
  validation_path = output_dir / "onnx_validation.json"
  if any(p.exists() for p in (path, contract_path, validation_path)):
    raise FileExistsError("Export requires unused policy.onnx/contract/validation paths")
  contract = {
    "format": "g1_mocap_mlp_position_v3_onnx_v1", "task_id": MOCAP_TASK,
    "algorithm": "SuperMimic_position_WM_BPTT_GAE", "actor_architecture": saved["origin"]["architecture"],
    "checkpoint_sha256": file_sha256(checkpoint), "iteration": saved["iteration"],
    "parameter_count": sum(p.numel() for p in actor.parameters()),
    "physics_dt": .005, "decimation": 4, "control_dt": .02,
    "inputs": {
      "history": {
        "shape": ["batch", 10, 93], "dtype": "float32", "time_order": "oldest_to_newest",
        "per_frame": ["encoder_q_minus_default29", "joint_velocity29", "base_angular_velocity_body3",
                      "projected_gravity_body3", "previous_raw_residual29"],
        "reset": "native observation manager history reset; see the verified native reset sample in validation/",
      },
      "reference": {
        "shape": ["batch", 798], "dtype": "float32", "offsets": list(REFERENCE_OFFSETS),
        "per_token": ["root_linear_velocity_in_reference_body3", "root_angular_velocity_in_reference_body3",
                      "reference_projected_gravity_body3", "absolute_reference_joint_position29"],
        "clip_boundary": "clamp every offset to the current clip's first/last frame",
        "frame_timing": "native reset writes reference state, then advances the command by one frame before observation",
      },
    },
    "normalization": {
      "inside_onnx": True, "history_term_divisors": [1., 5., 2., 1., 1.],
      "reference_token_divisors": [1.] * 3 + [2.] * 3 + [1.] * 32,
    },
    "outputs": {"raw_residual": {"shape": ["batch", 29], "dtype": "float32", "not": "absolute joint targets or torque"}},
    "controller": controller, "task_contract": saved["task_contract"],
    "reference_data_included": False, "world_model_required_for_inference": False,
  }
  wrapper = OnnxActor(actor).eval()
  history, reference = torch.zeros(1, 10, 93), torch.zeros(1, 798)
  torch.onnx.export(
    wrapper, (history, reference), str(path), input_names=["history", "reference"], output_names=["raw_residual"],
    dynamic_axes={"history": {0: "batch"}, "reference": {0: "batch"}, "raw_residual": {0: "batch"}},
    opset_version=17, dynamo=False,
  )
  model = onnx.load(path)
  item = model.metadata_props.add()
  item.key, item.value = "mocap_contract", json.dumps(contract, sort_keys=True)
  onnx.checker.check_model(model)
  onnx.save(model, path)
  options = ort.SessionOptions()
  options.intra_op_num_threads = 1
  session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
  generator = torch.Generator().manual_seed(14160)
  samples = [
    ("zero", history, reference),
    ("random_batch1", torch.randn(1, 10, 93, generator=generator), torch.randn(1, 798, generator=generator)),
    ("random_batch8", torch.randn(8, 10, 93, generator=generator), torch.randn(8, 798, generator=generator)),
  ]
  frozen_actions = None
  if native_inputs:
    with np.load(native_inputs) as inputs:
      samples.append(("native_reset", torch.from_numpy(inputs["history"].copy()), torch.from_numpy(inputs["reference"].copy())))
      if "action" in inputs:
        frozen_actions = inputs["action"].copy()
  results = {}
  with torch.no_grad():
    for name, h, r in samples:
      expected = actor(actor_observations(h, r), stochastic_output=False).numpy()
      if name == "native_reset" and frozen_actions is not None:
        np.testing.assert_allclose(expected, frozen_actions, rtol=2e-5, atol=2e-5)
      actual = session.run(["raw_residual"], {"history": h.numpy(), "reference": r.numpy()})[0]
      if actual.shape != (len(h), 29) or not np.isfinite(actual).all():
        raise AssertionError(f"Invalid ONNX output: {name}")
      np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
      results[name] = {"batch": len(h), "max_absolute_error": float(np.abs(actual - expected).max())}
      if name == "native_reset" and frozen_actions is not None:
        results[name]["frozen_runtime_actor_max_absolute_error"] = float(np.abs(expected - frozen_actions).max())
  contract_path.write_text(json.dumps(contract, indent=2) + "\n")
  validation_path.write_text(json.dumps({
    "passed": True, "onnx_sha256": file_sha256(path), "rtol": 2e-5, "atol": 2e-5,
    "samples": results, "native_sample_included": native_inputs is not None,
  }, indent=2) + "\n")
  print(f"[mocap] exported and ORT-validated: {path}")
  return path


def package_result(checkpoint: Path, output: Path) -> None:
  checkpoint = checkpoint.resolve(strict=True)
  if checkpoint.is_dir():
    checkpoint /= "rank_0.pt"
  saved = load_mlp_checkpoint(checkpoint)
  runtime = frozen_runtime(checkpoint, saved)
  if output.exists():
    raise FileExistsError(output)
  output.mkdir(parents=True)
  training = output / "training" / f"update_{saved['iteration']:08d}"
  training.mkdir(parents=True)
  ranks = {}
  for rank in range(saved["world_size"]):
    source = checkpoint.parent / f"rank_{rank}.pt"
    state = load_mlp_checkpoint(source)
    if state["iteration"] != saved["iteration"] or state["rank"] != rank or state["world_size"] != saved["world_size"]:
      raise ValueError(f"Inconsistent complete checkpoint rank: {source}")
    target = training / source.name
    shutil.copy2(source, target)
    digest = file_sha256(source)
    if file_sha256(target) != digest:
      raise AssertionError(f"Copied checkpoint differs: {target}")
    ranks[str(rank)] = {"path": str(target.relative_to(output)), "sha256": digest}
  shutil.copytree(runtime, output / "runtime", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
  frozen_runtime(training / "rank_0.pt", saved)
  (output / "training_config.json").write_text(json.dumps(saved["config"], indent=2) + "\n")
  files = {
    str(path.relative_to(output)): file_sha256(path) for path in sorted(output.rglob("*")) if path.is_file()
  }
  (output / "bundle.json").write_text(json.dumps({
    "format": "g1_mocap_mlp_training_bundle_v1", "iteration": saved["iteration"],
    "world_size": saved["world_size"], "rank_checkpoints": ranks, "files": files,
    "source_checkpoint": str(checkpoint), "source_provenance": saved["source_provenance"],
    "optimizer_updates": saved["optimizer_updates"],
    "external_dependencies": {
      "mocap_dir": saved["config"]["mocap_dir"], "corpus": saved["collector"]["corpus"],
      "replica_source": saved["config"]["replica_source"],
      "bootstrap_world_model": saved["config"]["world_model"],
      "bootstrap_world_model_sha256": saved["origin"]["world_model_sha256"],
    },
    "resume": "Use the official --resume entrypoint with the saved world size; it verifies and executes runtime/",
    "resume_semantics": "all models/three Adam/sampler/coverage/budgets/RNG retained; native simulator and replay reinitialized",
    "portable_without_external_data_and_dependencies": False,
  }, indent=2) + "\n")
  print(f"[mocap] preserved complete training result: {output}")
