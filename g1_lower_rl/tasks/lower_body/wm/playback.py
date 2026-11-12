"""Native MuJoCo playback, headless smoke check, and ONNX export for velocity WM policies."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn

from g1_lower_rl.assets import LOWER_BODY_JOINTS
from g1_lower_rl.tasks.lower_body.wm import LOWER_BODY_WM_TASK
from g1_lower_rl.world_model.checkpoint import atomic_json
from g1_lower_rl.world_model.model import JOINT_NAMES

from .policy import (
  ACTION_DIM,
  OBS_DIM,
  OBS_TERMS,
  VelocityMlpActor,
  migrate_actor_state_dict,
  unflatten_native_history,
)
from .task import flat_env_cfg_wm

FORMAT = "g1_lower_body_velocity_wm_v1"


def resolve_checkpoint_file(path: str | Path) -> Path:
  resolved = Path(path).resolve(strict=True)
  if resolved.is_dir():
    latest_json = resolved / "latest.json"
    if latest_json.is_file():
      meta = json.loads(latest_json.read_text(encoding="utf-8"))
      resolved = (resolved / meta["directory"] / "rank_0.pt").resolve(strict=True)
    elif (resolved / "rank_0.pt").is_file():
      resolved = (resolved / "rank_0.pt").resolve(strict=True)
    else:
      raise FileNotFoundError(f"No rank_0.pt or latest.json found under {resolved}")
  return resolved


def load_velocity_wm_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
  ckpt_file = resolve_checkpoint_file(path)
  saved = torch.load(ckpt_file, map_location=device, weights_only=True)
  if not isinstance(saved, dict) or saved.get("format") != FORMAT:
    raise ValueError(f"Expected {FORMAT} checkpoint at {ckpt_file}, got {saved.get('format') if isinstance(saved, dict) else type(saved)}")
  return saved


def load_actor_from_checkpoint(saved: dict[str, Any], device: str | torch.device = "cpu") -> VelocityMlpActor:
  cfg = saved["config"]
  actor = VelocityMlpActor(
    history_length=int(cfg["history_length"]),
    hidden_dims=tuple(int(w) for w in cfg["actor_hidden_dims"]),
  ).to(device)
  migrated_state, _ = migrate_actor_state_dict(actor, saved["actor_state_dict"])
  actor.load_state_dict(migrated_state, strict=True)
  if not all(torch.isfinite(value).all() for value in actor.state_dict().values()):
    raise FloatingPointError("Velocity WM actor checkpoint contains nonfinite weights")
  return actor.eval().requires_grad_(False)


class StatelessOnnxWrapper(nn.Module):
  def __init__(self, actor: VelocityMlpActor):
    super().__init__()
    self.actor = actor

  def forward(self, history_or_obs: torch.Tensor) -> torch.Tensor:
    scaled = ((history_or_obs - self.actor.frame_offset) * self.actor.frame_scale).flatten(1)
    return self.actor.net(scaled)


class DeploySlidingHistoryWrapper(nn.Module):
  """Stateful 1-frame + (H-1)-frame history wrapper compatible with g1_lower_rl/deploy.py."""

  def __init__(self, actor: VelocityMlpActor):
    super().__init__()
    self.actor = actor

  def forward(self, obs: torch.Tensor, h_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    history = torch.cat((h_in, obs.unsqueeze(1)), dim=1)
    scaled = ((history - self.actor.frame_offset) * self.actor.frame_scale).flatten(1)
    return self.actor.net(scaled), history[:, 1:, :]


def export_onnx(
  checkpoint_or_saved: str | Path | dict[str, Any],
  output_dir: str | Path,
  *,
  overwrite: bool = False,
) -> dict[str, Path]:
  """Export stateless history ONNX (`policy.onnx`) and deploy-compatible ONNX (`policy_deploy.onnx`)."""
  import onnx
  import onnxruntime as ort

  saved = (
    checkpoint_or_saved
    if isinstance(checkpoint_or_saved, dict)
    else load_velocity_wm_checkpoint(checkpoint_or_saved, device="cpu")
  )
  actor = load_actor_from_checkpoint(saved, device="cpu")
  out = Path(output_dir)
  out.mkdir(parents=True, exist_ok=True)
  stateless_path = out / "policy.onnx"
  deploy_path = out / "policy_deploy.onnx"
  contract_path = out / "policy_contract.json"
  if not overwrite and any(p.exists() for p in (stateless_path, deploy_path, contract_path)):
    raise FileExistsError(f"Export destination already contains ONNX/contract files: {out}")

  h = actor.history_length
  controller = saved["controller_contract"]
  obs_names = [name for name, _ in OBS_TERMS]

  # 1. Stateless [batch, H, 57] -> [batch, 15]
  stateless = StatelessOnnxWrapper(actor).eval()
  dummy_hist = torch.zeros(1, h, OBS_DIM, dtype=torch.float32)
  torch.onnx.export(
    stateless,
    (dummy_hist,),
    str(stateless_path),
    input_names=["history"],
    output_names=["actions"],
    dynamic_axes={"history": {0: "batch"}, "actions": {0: "batch"}},
    opset_version=17,
    dynamo=False,
  )

  # 2. Deploy-compatible wrapper for scripts/check_deploy_policy.py
  if h > 1:
    deploy_wrapper: nn.Module = DeploySlidingHistoryWrapper(actor).eval()
    dummy_obs = torch.zeros(1, OBS_DIM, dtype=torch.float32)
    dummy_h_in = torch.zeros(1, h - 1, OBS_DIM, dtype=torch.float32)
    torch.onnx.export(
      deploy_wrapper,
      (dummy_obs, dummy_h_in),
      str(deploy_path),
      input_names=["obs", "h_in"],
      output_names=["actions", "h_out"],
      opset_version=17,
      dynamo=False,
    )
  else:
    dummy_obs = torch.zeros(1, OBS_DIM, dtype=torch.float32)
    torch.onnx.export(
      stateless,
      (dummy_obs,),
      str(deploy_path),
      input_names=["obs"],
      output_names=["actions"],
      opset_version=17,
      dynamo=False,
    )

  deploy_meta = {
    "task_id": LOWER_BODY_WM_TASK,
    "algorithm": "PositionWM_v3_BPTT_GAE_MLP",
    "iteration": str(saved["iteration"]),
    "history_length": str(h),
    "control_dt": "0.02",
    "joint_names": ",".join(JOINT_NAMES),
    "default_joint_pos": ",".join(f"{float(v):.8g}" for v in controller["default_joint_pos_29"]),
    "action_joint_names": ",".join(LOWER_BODY_JOINTS),
    "action_scale": ",".join(f"{float(v):.8g}" for v in controller["action_scale_15"]),
    "observation_names": ",".join(obs_names),
    "obs_joint_pos_joint_names": ",".join(LOWER_BODY_JOINTS),
    "obs_joint_vel_joint_names": ",".join(LOWER_BODY_JOINTS),
  }

  for onnx_path, extra in (
    (stateless_path, {"velocity_wm_contract": "stateless_history"}),
    (deploy_path, deploy_meta),
  ):
    model_proto = onnx.load(onnx_path)
    for k, v in extra.items():
      prop = model_proto.metadata_props.add()
      prop.key = str(k)
      prop.value = str(v)
    onnx.checker.check_model(model_proto)
    onnx.save(model_proto, onnx_path)

  # Validate numerical equivalence with ONNX Runtime on CPU
  opts = ort.SessionOptions()
  opts.intra_op_num_threads = 1
  rng = torch.Generator().manual_seed(20261111)
  test_hist = torch.randn(4, h, OBS_DIM, generator=rng, dtype=torch.float32)
  with torch.no_grad():
    ref_out = actor(test_hist).numpy()

  sess_stateless = ort.InferenceSession(str(stateless_path), sess_options=opts, providers=["CPUExecutionProvider"])
  ort_stateless = sess_stateless.run(["actions"], {"history": test_hist.numpy()})[0]
  max_err_stateless = float(np.max(np.abs(ref_out - ort_stateless)))
  if max_err_stateless > 1e-5:
    raise RuntimeError(f"Stateless ONNX mismatch vs PyTorch: {max_err_stateless}")

  sess_deploy = ort.InferenceSession(str(deploy_path), sess_options=opts, providers=["CPUExecutionProvider"])
  if h > 1:
    ort_deploy, ort_h_out = sess_deploy.run(
      ["actions", "h_out"],
      {"obs": test_hist[:1, -1, :].numpy(), "h_in": test_hist[:1, :-1, :].numpy()},
    )
    if float(np.max(np.abs(ort_h_out - test_hist[:1, 1:, :].numpy()))) > 1e-6:
      raise RuntimeError("Deploy ONNX sliding-window history output mismatch")
  else:
    ort_deploy = sess_deploy.run(["actions"], {"obs": test_hist[:1, 0, :].numpy()})[0]
  max_err_deploy = float(np.max(np.abs(ref_out[:1] - ort_deploy)))
  if max_err_deploy > 1e-5:
    raise RuntimeError(f"Deploy ONNX mismatch vs PyTorch: {max_err_deploy}")

  contract = {
    "format": "g1_lower_body_velocity_wm_onnx_v1",
    "task_id": LOWER_BODY_WM_TASK,
    "iteration": int(saved["iteration"]),
    "history_length": h,
    "obs_dim": OBS_DIM,
    "action_dim": ACTION_DIM,
    "architecture": saved["origin"]["architecture"],
    "controller": controller,
    "onnx_validation": {
      "stateless_max_abs_error": max_err_stateless,
      "deploy_max_abs_error": max_err_deploy,
    },
  }
  atomic_json(contract_path, contract)
  return {"stateless": stateless_path, "deploy": deploy_path, "contract": contract_path}


@dataclass(frozen=True)
class PlayConfig:
  checkpoint_file: str
  device: str = "cuda:0"
  num_envs: int = 16
  viewer: Literal["auto", "native", "viser"] = "auto"
  viser_host: str = "0.0.0.0"
  viser_port: int = 8082
  no_terminations: bool = False
  steps: int = 0
  """Positive count runs a headless native MuJoCo smoke evaluation instead of opening a viewer."""


def run_play(cfg: PlayConfig) -> None:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

  if cfg.steps < 0 or cfg.num_envs < 1:
    raise ValueError("steps must be >= 0 and num_envs >= 1")
  saved = load_velocity_wm_checkpoint(cfg.checkpoint_file, device="cpu")
  actor = load_actor_from_checkpoint(saved, device=cfg.device)
  h = actor.history_length

  env_cfg = flat_env_cfg_wm(history_length=h, play=True)
  env_cfg.scene.num_envs = cfg.num_envs
  if cfg.no_terminations:
    env_cfg.terminations = {}

  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  try:
    @torch.no_grad()
    def policy(observations: Any) -> torch.Tensor:
      raw = observations["actor"] if isinstance(observations, dict) or hasattr(observations, "__getitem__") else observations
      history = unflatten_native_history(raw, h)
      action = actor(history)
      if not torch.isfinite(action).all():
        raise FloatingPointError("Nonfinite playback action")
      return action

    if cfg.steps > 0:
      obs, _ = env.reset()
      total_reward = torch.zeros(cfg.num_envs, device=cfg.device)
      terminations = 0
      for _ in range(cfg.steps):
        obs, reward, terminated, truncated, _ = env.step(policy(obs))
        if env.sim.data.overflow.any() or not torch.isfinite(env.sim.data.qpos).all():
          raise FloatingPointError("Native playback physics overflow or nonfinite qpos")
        total_reward += reward
        terminations += int((terminated | truncated).sum())
      print(
        json.dumps(
          {
            "task": LOWER_BODY_WM_TASK,
            "iteration": int(saved["iteration"]),
            "steps": cfg.steps,
            "num_envs": cfg.num_envs,
            "mean_step_reward": float(total_reward.mean() / cfg.steps),
            "resets": terminations,
          }
        ),
        flush=True,
      )
      return

    wrapped = RslRlVecEnvWrapper(env)
    resolved = cfg.viewer
    if resolved == "auto":
      has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
      resolved = "native" if has_display else "viser"

    if resolved == "native":
      NativeMujocoViewer(wrapped, policy).run()
    else:
      import viser

      server = viser.ViserServer(host=cfg.viser_host, port=cfg.viser_port, label=f"VelocityWM {saved['iteration']}")
      try:
        ViserPlayViewer(wrapped, policy, viser_server=server).run()
      finally:
        server.stop()
  finally:
    env.close()
