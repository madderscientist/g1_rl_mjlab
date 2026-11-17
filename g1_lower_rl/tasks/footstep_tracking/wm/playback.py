"""ONNX export (stateless + FootstepPolicy-compatible) and MuJoCo/Viser playback for footstep WM."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn

from g1_lower_rl.assets import LOWER_BODY_JOINTS, WHOLE_BODY_JOINTS
from g1_lower_rl.footstep_contract import (
  ACTION_DIM as CONTRACT_ACTION_DIM,
)
from g1_lower_rl.footstep_contract import (
  CONTRACT_VERSION,
  FOOTSTEP_SLOTS,
  RAW_OBS_DIM,
)
from g1_lower_rl.footstep_phase import FootstepPhaseCfg
from g1_lower_rl.tasks.footstep_tracking.wm import FOOTSTEP_WM_TASK
from g1_lower_rl.world_model.checkpoint import atomic_json

from .policy import (
  ACTION_DIM,
  CMD_ENC_DIM,
  FRAME_DIM,
  FootstepTwinMirrorMlpActor,
  encode_command,
  encode_proprio_frame,
)
from .task import footstep_env_cfg_wm

FORMAT = "g1_footstep_tracking_wm_v1"


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


def load_footstep_wm_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
  ckpt_file = resolve_checkpoint_file(path)
  saved = torch.load(ckpt_file, map_location=device, weights_only=True)
  if not isinstance(saved, dict) or saved.get("format") != FORMAT:
    raise ValueError(
      f"Expected {FORMAT} checkpoint at {ckpt_file}, got {saved.get('format') if isinstance(saved, dict) else type(saved)}"
    )
  return saved


def load_actor_from_checkpoint(saved: dict[str, Any], device: str | torch.device = "cpu") -> FootstepTwinMirrorMlpActor:
  cfg = saved["config"]
  actor = FootstepTwinMirrorMlpActor(
    history_length=int(cfg["history_length"]),
    hidden_dims=tuple(int(w) for w in cfg["actor_hidden_dims"]),
    default_joint_pos=saved["controller_contract"]["default_joint_pos_29"],
  ).to(device)
  actor.load_state_dict(saved["actor_state_dict"], strict=True)
  if not all(torch.isfinite(v).all() for v in actor.state_dict().values()):
    raise FloatingPointError("Footstep WM actor checkpoint contains nonfinite weights")
  return actor.eval().requires_grad_(False)


class StatelessFootstepOnnxWrapper(nn.Module):
  def __init__(self, actor: FootstepTwinMirrorMlpActor):
    super().__init__()
    self.actor = actor

  def forward(self, history: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
    return self.actor(history, command)


class DeployFootstepHistoryWrapper(nn.Module):
  """Stateful ONNX wrapper matching g1_lower_rl.footstep_deploy.FootstepPolicy (`obs[1,79], h_in[H,1,86]`).

  `h_in[:H-1, 0, :85]` stores past H-1 encoded frames, `h_in[:H-1, 0, 85]` stores a 0/1 valid flag
  so `reset()` (`h_in.fill(0)`) automatically repeats the initial post-reset frame, and
  `h_in[H-1, 0, 70:85]` stores the previous 15D action!
  """

  def __init__(self, actor: FootstepTwinMirrorMlpActor):
    super().__init__()
    self.actor = actor

  def forward(self, obs: torch.Tensor, h_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h = self.actor.history_length
    prev_action = h_in[h - 1, :, 70:85]
    frame_t = encode_proprio_frame(obs[:, :70], prev_action, self.actor.default_joint_pos)

    if h > 1:
      past_frames = h_in[: h - 1, 0, :FRAME_DIM]
      past_valid = h_in[: h - 1, 0, FRAME_DIM : FRAME_DIM + 1]
      filled_past = past_valid * past_frames + (1.0 - past_valid) * frame_t
      history = torch.cat((filled_past.unsqueeze(0), frame_t.unsqueeze(1)), dim=1)
    else:
      history = frame_t.unsqueeze(1)

    phase = obs[:, 70]
    freq = obs[:, 71]
    footprints = obs[:, 72:78].unflatten(-1, (2, 3))
    theta = obs[:, 78]
    cmd = encode_command(phase, freq, footprints, theta)

    actions = self.actor(history, cmd)

    # Build h_out [H, 1, 86]
    ones_flag = torch.ones((h - 1, 1, 1), dtype=obs.dtype, device=obs.device)
    if h > 1:
      new_past = history[:, 1:, :].transpose(0, 1)  # [H-1, 1, 85]
      past_with_flag = torch.cat((new_past, ones_flag), dim=-1)
    else:
      past_with_flag = obs.new_zeros((0, 1, FRAME_DIM + 1))

    last_row = torch.cat(
      (
        obs.new_zeros((1, 70)),
        actions,
        obs.new_ones((1, 1)),
      ),
      dim=-1,
    ).unsqueeze(0)  # [1, 1, 86]
    h_out = torch.cat((past_with_flag, last_row), dim=0)
    return actions, h_out


def build_footstep_deploy_contract(
  saved: dict[str, Any],
  *,
  history_length: int,
  phase_cfg: FootstepPhaseCfg | None = None,
) -> dict[str, Any]:
  """Build v7 `footstep_contract` metadata accepted by g1_lower_rl.footstep_deploy.FootstepPolicy."""
  phase = phase_cfg or FootstepPhaseCfg()
  schedule_meta = phase.to_metadata()
  controller = saved["controller_contract"]
  hidden_shape = [history_length, 1, FRAME_DIM + 1]
  return {
    "version": CONTRACT_VERSION,
    "dtype": "float32",
    "task_id": FOOTSTEP_WM_TASK,
    "algorithm": "PositionWM_v3_BPTT_GAE_TwinMirror_MLP",
    "iteration": int(saved["iteration"]),
    "joint_names": list(WHOLE_BODY_JOINTS),
    "inputs": {
      "obs": [1, RAW_OBS_DIM],
      "h_in": hidden_shape,
    },
    "outputs": {
      "actions": [1, CONTRACT_ACTION_DIM],
      "h_out": hidden_shape,
    },
    "actions": {
      "joint_names": list(LOWER_BODY_JOINTS),
      "default_joint_pos": [float(v) for v in controller["default_joint_pos_29"][:ACTION_DIM]],
      "scale": [float(v) for v in controller["action_scale_15"]],
    },
    "footsteps": {
      "slots": list(FOOTSTEP_SLOTS),
      "coordinates": ["x", "y", "yaw"],
    },
    "phase": {
      "standing_frequency": 0.0,
      "left_touchdown_rad": schedule_meta["left_touchdown_rad"],
      "right_touchdown_rad": schedule_meta["right_touchdown_rad"],
      "contact_schedule": schedule_meta,
    },
  }


def export_onnx(
  checkpoint_or_saved: str | Path | dict[str, Any],
  output_dir: str | Path,
  *,
  overwrite: bool = False,
) -> dict[str, Path]:
  """Export stateless ONNX (`policy.onnx`) and FootstepPolicy-compatible (`policy_deploy.onnx`)."""
  import onnx
  import onnxruntime as ort

  from g1_lower_rl.footstep_deploy import FootstepPolicy

  saved = (
    checkpoint_or_saved
    if isinstance(checkpoint_or_saved, dict)
    else load_footstep_wm_checkpoint(checkpoint_or_saved, device="cpu")
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
  stateless = StatelessFootstepOnnxWrapper(actor).eval()
  torch.onnx.export(
    stateless,
    (torch.zeros(1, h, FRAME_DIM), torch.zeros(1, CMD_ENC_DIM)),
    str(stateless_path),
    input_names=["history", "command"],
    output_names=["actions"],
    dynamic_axes={"history": {0: "batch"}, "command": {0: "batch"}, "actions": {0: "batch"}},
    opset_version=17,
    dynamo=False,
  )

  deploy_wrapper = DeployFootstepHistoryWrapper(actor).eval()
  torch.onnx.export(
    deploy_wrapper,
    (torch.zeros(1, RAW_OBS_DIM), torch.zeros(h, 1, FRAME_DIM + 1)),
    str(deploy_path),
    input_names=["obs", "h_in"],
    output_names=["actions", "h_out"],
    opset_version=17,
    dynamo=False,
  )

  deploy_contract = build_footstep_deploy_contract(saved, history_length=h)
  for onnx_path, meta in (
    (stateless_path, {"footstep_wm_contract": "stateless_twin_mirror_history_mlp"}),
    (deploy_path, {"footstep_contract": json.dumps(deploy_contract)}),
  ):
    proto = onnx.load(onnx_path)
    for k, v in meta.items():
      prop = proto.metadata_props.add()
      prop.key = str(k)
      prop.value = str(v)
    onnx.checker.check_model(proto)
    onnx.save(proto, onnx_path)

  # Validate numerical parity against PyTorch AND FootstepPolicy CPU runner
  opts = ort.SessionOptions()
  opts.intra_op_num_threads = 1
  sess_stateless = ort.InferenceSession(str(stateless_path), sess_options=opts, providers=["CPUExecutionProvider"])
  rng = torch.Generator().manual_seed(20261112)
  test_hist = torch.randn(4, h, FRAME_DIM, generator=rng)
  test_cmd = torch.randn(4, CMD_ENC_DIM, generator=rng)
  with torch.no_grad():
    ref_stateless = actor(test_hist, test_cmd).numpy()
  ort_stateless = sess_stateless.run(["actions"], {"history": test_hist.numpy(), "command": test_cmd.numpy()})[0]
  max_err_stateless = float(np.max(np.abs(ref_stateless - ort_stateless)))
  if max_err_stateless > 1e-5:
    raise RuntimeError(f"Stateless footstep ONNX mismatch vs PyTorch: {max_err_stateless}")

  fp_policy = FootstepPolicy(deploy_path)
  fp_policy.reset()
  py_h = torch.zeros(h, 1, FRAME_DIM + 1)
  max_err_deploy = 0.0
  np_rng = np.random.default_rng(20261112)
  with torch.no_grad():
    for step_i in range(16):
      if step_i == 8:
        fp_policy.reset()
        py_h.zero_()
      raw = np_rng.normal(0.0, 0.05, (1, RAW_OBS_DIM)).astype(np.float32)
      raw[:, :29] += actor.default_joint_pos.numpy()
      raw[:, 63] = raw[:, 69] = -1.0
      raw[:, 70] = np_rng.uniform(0.0, 2.0 * np.pi)
      raw[:, 71] = 0.0 if step_i % 4 == 0 else np_rng.uniform(0.8, 1.8)
      raw[:, 78] = np_rng.uniform(-0.8, 0.8)
      expected_act, py_h = deploy_wrapper(torch.from_numpy(raw), py_h)
      ort_act, _ = fp_policy.step(raw)
      max_err_deploy = max(max_err_deploy, float(np.max(np.abs(expected_act.numpy()[0] - ort_act))))
  if max_err_deploy > 1e-5:
    raise RuntimeError(f"FootstepPolicy deploy ONNX mismatch vs PyTorch: {max_err_deploy}")

  atomic_json(
    contract_path,
    {
      "format": "g1_footstep_tracking_wm_onnx_v1",
      "deploy_contract": deploy_contract,
      "onnx_validation": {
        "stateless_max_abs_error": max_err_stateless,
        "deploy_max_abs_error": max_err_deploy,
      },
    },
  )
  return {"stateless": stateless_path, "deploy": deploy_path, "contract": contract_path}


@dataclass(frozen=True)
class PlayConfig:
  checkpoint_file: str
  profile: Literal["precision", "step-episode"] = "precision"
  device: str = "cuda:0"
  num_envs: int = 1
  viewer: Literal["auto", "native", "viser"] = "auto"
  viser_host: str = "0.0.0.0"
  viser_port: int = 8085
  label: str | None = None
  manual_step: bool = False
  no_terminations: bool = False
  hot_reload_checkpoint: bool = True
  steps: int = 0


def run_play(cfg: PlayConfig) -> None:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.viewer import NativeMujocoViewer

  from g1_lower_rl.tasks.footstep_tracking.frequency_preview import (
    FrequencyPlayViewer,
    ThetaPlayViewer,
    configure_frequency_preview,
    configure_theta_preview,
  )
  from g1_lower_rl.tasks.footstep_tracking.manual_preview import ManualStepViewer, configure_manual_preview

  if cfg.steps < 0 or cfg.num_envs < 1:
    raise ValueError("steps must be >= 0 and num_envs >= 1")
  ckpt_resolved = resolve_checkpoint_file(cfg.checkpoint_file)
  last_mtime_ns = ckpt_resolved.stat().st_mtime_ns
  saved = load_footstep_wm_checkpoint(ckpt_resolved, device="cpu")
  actor = load_actor_from_checkpoint(saved, device=cfg.device)
  print(
    f"[PLAY_START] profile={cfg.profile} port={cfg.viser_port} iteration={saved['iteration']} ckpt={ckpt_resolved}",
    flush=True,
  )
  h = actor.history_length

  env_cfg = footstep_env_cfg_wm(play=True, profile=cfg.profile)
  env_cfg.seed = 42
  env_cfg.scene.num_envs = 1 if cfg.manual_step else cfg.num_envs
  configure_theta_preview(env_cfg)
  if cfg.manual_step:
    configure_manual_preview(env_cfg)
  elif cfg.profile != "step-episode":
    configure_frequency_preview(env_cfg)
  if cfg.no_terminations:
    env_cfg.terminations = {k: v for k, v in env_cfg.terminations.items() if k == "footstep_fault"}

  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  try:
    n_envs = env.num_envs
    history = torch.zeros(n_envs, h, FRAME_DIM, device=cfg.device)
    prev_action = torch.zeros(n_envs, ACTION_DIM, device=cfg.device)
    fresh = torch.ones(n_envs, dtype=torch.bool, device=cfg.device)
    theta_cmd = env.command_manager.get_term("torso_heading")

    tick_counter = 0
    current_iter = int(saved["iteration"])
    model_badge: Any = None

    @torch.no_grad()
    def policy(observations: Any) -> torch.Tensor:
      nonlocal history, prev_action, fresh, last_mtime_ns, tick_counter, current_iter
      tick_counter += 1
      if cfg.hot_reload_checkpoint and tick_counter % 250 == 0:
        try:
          cand_file = resolve_checkpoint_file(cfg.checkpoint_file)
          mtime_ns = cand_file.stat().st_mtime_ns
          if mtime_ns != last_mtime_ns:
            new_saved = load_footstep_wm_checkpoint(cand_file, device="cpu")
            sd = new_saved["actor_state_dict"]
            if all(torch.isfinite(v).all() for v in sd.values()):
              actor.load_state_dict(sd, strict=True)
              last_mtime_ns = mtime_ns
              current_iter = int(new_saved["iteration"])
              if model_badge is not None:
                model_badge.content = f"**当前策略**: `iter {current_iter}`"
              print(
                f"[HOT_RELOAD] profile={cfg.profile} port={cfg.viser_port} iteration={current_iter}",
                flush=True,
              )
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
          print(f"[HOT_RELOAD_SKIP] {exc}", flush=True)
      raw = observations if isinstance(observations, torch.Tensor) else observations["actor"]
      ep_zero = env.episode_length_buf == 0
      is_fresh = fresh | ep_zero
      pa = torch.where(is_fresh.unsqueeze(-1), torch.zeros_like(prev_action), prev_action)
      frame = encode_proprio_frame(raw[:, :70], pa, actor.default_joint_pos)
      repeated = frame.unsqueeze(1).expand(-1, h, -1)
      shifted = torch.cat((history[:, 1:], frame.unsqueeze(1)), dim=1)
      history = torch.where(is_fresh[:, None, None], repeated, shifted)

      theta = theta_cmd.command[:, 0].to(raw.dtype)
      cmd = encode_command(raw[:, 70], raw[:, 71], raw[:, 72:78].unflatten(-1, (2, 3)), theta)
      action = actor(history, cmd)
      if not torch.isfinite(action).all():
        raise FloatingPointError("Nonfinite footstep playback action")
      prev_action = action.clone()
      fresh = torch.zeros_like(fresh)
      return action

    if cfg.steps > 0:
      obs, _ = env.reset()
      fresh.fill_(True)
      total_reward = torch.zeros(n_envs, device=cfg.device)
      resets = 0
      for _ in range(cfg.steps):
        obs, reward, terminated, truncated, _ = env.step(policy(obs))
        fresh = (terminated | truncated).clone()
        total_reward += reward
        resets += int(fresh.sum())
      print(
        json.dumps(
          {
            "task": FOOTSTEP_WM_TASK,
            "iteration": int(saved["iteration"]),
            "stage": saved.get("stage_index"),
            "steps": cfg.steps,
            "num_envs": n_envs,
            "mean_step_reward": float(total_reward.mean() / cfg.steps),
            "resets": resets,
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

      title = cfg.label or f"FootstepWM {cfg.profile} - iter {current_iter}"
      server = viser.ViserServer(host=cfg.viser_host, port=cfg.viser_port, label=title)
      model_badge = server.gui.add_markdown(f"**当前策略**: `iter {current_iter}`")
      try:
        if cfg.manual_step:
          viewer_cls = ManualStepViewer
        elif cfg.profile == "step-episode":
          viewer_cls = ThetaPlayViewer
        else:
          viewer_cls = FrequencyPlayViewer
        viewer_cls(wrapped, policy, viser_server=server).run()
      finally:
        server.stop()
  finally:
    env.close()
