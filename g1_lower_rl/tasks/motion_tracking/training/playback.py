"""Native, noise-free mocap playback/evaluation; no imagined WM or optimizer."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import numpy as np
import torch

from g1_lower_rl.world_model.checkpoint import atomic_json
from g1_lower_rl.world_model.collection import CollectorConfig

from .export import load_actor, native_module
from .official import load_mlp_checkpoint
from .position_collection import PositionPolicyCollector
from .position_policy import actor_observations, unflatten_native_history


class NativePlayback:
  def __init__(
    self, checkpoint: str, device: str, num_envs: int = 2, motions: str = "",
    *, validation: bool = False, width: int = 960, height: int = 720,
    render_mode: Literal["rgb_array"] | None = None,
  ):
    if num_envs < 2:
      raise ValueError("Native mapped playback requires at least two environments")
    self.saved = load_mlp_checkpoint(Path(checkpoint))
    self.actor = load_actor(self.saved, device)
    native = native_module(self.saved)
    factory = native.make_env_cfg

    def nominal(args):
      cfg = factory(replace(args, obs_noise=False, reset_noise=False, domain_randomization=False, debug_vis=True))
      cfg.episode_length_s = 1e5
      for name in ("root_height", "root_tilt", "time_out"):
        cfg.terminations[name] = replace(
          cfg.terminations[name], func=lambda env: torch.zeros(env.num_envs, device=env.device, dtype=torch.bool), params={},
        )
      cfg.viewer.width, cfg.viewer.height = width, height
      return cfg

    with patch.object(native, "make_env_cfg", nominal):
      self.collector = PositionPolicyCollector(
        CollectorConfig("", motion_dir=self.saved["config"]["mocap_dir"], num_envs=num_envs,
                        device=device, seed=14160, action_noise=0., njmax=500),
        replica_source=self.saved["config"]["replica_source"], validation_envs=1, actor=self.actor,
        task_definition=self.saved["config"]["task_definition"],
        gamma=self.saved["config"]["gamma"],
        render_mode=render_mode,
      )
    try:
      self.env = self.collector.env
      library = self.collector.library
      for key in ("manifest_sha256", "data_sha256", "joint_projection"):
        if library.metadata[key] != self.saved["collector"]["corpus"][key]:
          raise ValueError(f"Playback corpus differs from the training checkpoint: {key}")
      names = list(library.corpus.names)
      wanted = [name.strip() for name in motions.split(",") if name.strip()]
      if wanted:
        if len(wanted) > num_envs or any(name not in names for name in wanted):
          raise ValueError("Requested motions must exist and fit in num_envs")
        ids = [names.index(name) for name in wanted]
        ids = [ids[i % len(ids)] for i in range(num_envs)]
      else:
        first = library.corpus.training_motions if validation else 0
        if first + num_envs > library.num_clips:
          raise ValueError("Requested playback batch exceeds the selected corpus split")
        ids = list(range(first, first + num_envs))
      self.ids = torch.tensor(ids, device=device)
      self.names = [names[i] for i in ids]
      self.command = self.collector._native_command
      # Playback assigns explicit clips, not the trainer's fixed train/validation column counts.
      self.collector._validation = self.ids >= library.corpus.training_motions
      self.command.forced_frame[:] = library.clip_start[self.ids]
      self.observations, _ = self.env.reset()
      expected = torch.minimum(library.clip_start[self.ids] + 1, library.clip_last[self.ids])
      if not torch.equal(self.command.frame, expected):
        raise AssertionError("Native reset/reference frame timing changed")
      self.lengths = library.clip_len[self.ids]
    except BaseException:
      self.collector.close()
      raise

  @torch.no_grad()
  def policy(self, observations):
    history = unflatten_native_history(observations["policy"])
    reference = self.collector.reference_window(self.command.frame)
    action = self.actor(actor_observations(history, reference), stochastic_output=False)
    if not torch.isfinite(action).all():
      raise FloatingPointError("Nonfinite native playback action")
    return action

  @torch.no_grad()
  def step(self):
    self.observations, reward, terminated, truncated, _ = self.env.step(self.policy(self.observations))
    if self.env.sim.data.overflow.any() or not torch.isfinite(self.env.sim.data.qpos).all():
      raise FloatingPointError("Native playback physics overflow/nonfinite state")
    return reward, terminated | truncated

  def close(self):
    self.collector.close()


@dataclass(frozen=True)
class PlayConfig:
  checkpoint_file: str
  device: str = "cuda:0"
  num_envs: int = 2
  motions: str = ""
  viewer: str = "viser"
  viser_host: str = "0.0.0.0"
  viser_port: int = 8081
  steps: int = 0
  """Positive count runs a headless native smoke instead of starting a viewer."""


def run_play(cfg: PlayConfig) -> None:
  if cfg.steps < 0 or cfg.viewer not in ("viser", "native") or not 1 <= cfg.viser_port <= 65535:
    raise ValueError("steps must be nonnegative and viewer must be viser or native")
  session = NativePlayback(cfg.checkpoint_file, cfg.device, cfg.num_envs, cfg.motions)
  server = None
  try:
    if cfg.steps:
      for _ in range(cfg.steps):
        session.step()
      print(f"[mocap] native smoke passed: iteration {session.saved['iteration']}, {cfg.steps} controls, {session.names}")
      return
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

    print(f"[mocap] MLP iteration {session.saved['iteration']}: {session.names}", flush=True)
    if cfg.viewer == "viser":
      import viser

      server = viser.ViserServer(host=cfg.viser_host, port=cfg.viser_port, label=f"MLP {session.saved['iteration']}")
      if server.get_port() != cfg.viser_port:
        raise OSError(f"Requested Viser port {cfg.viser_port} is occupied")
      with server.gui.add_folder("MLP policy"):
        server.gui.add_markdown(
          f"**MLP iteration {session.saved['iteration']}**\n\n"
          "Native MuJoCo, deterministic reference-relative residuals, no WM imagination.\n\n"
          + "\n\n".join(f"{index}: `{name}`" for index, name in enumerate(session.names))
        )
      ViserPlayViewer(RslRlVecEnvWrapper(session.env), session.policy, viser_server=server).run()
    else:
      NativeMujocoViewer(RslRlVecEnvWrapper(session.env), session.policy).run()
  finally:
    if server is not None:
      server.stop()
    session.close()


def evaluate(
  checkpoint: str, output: str = "logs/evaluation/mocap.json", clips: int = 32,
  max_seconds: float = 20., device: str = "cuda:0",
) -> None:
  if clips < 2 or not 0 < max_seconds < float("inf"):
    raise ValueError("Require at least two clips and a finite positive duration")
  session = NativePlayback(checkpoint, device, clips, validation=True)
  try:
    total = torch.zeros(clips, device=device)
    count = torch.zeros(clips, device=device)
    active = torch.ones(clips, device=device, dtype=torch.bool)
    limit = min(max(1, int(max_seconds / .02)), int(session.lengths.max()))
    for _ in range(limit):
      reward, done = session.step()
      total += torch.where(active, reward, 0.)
      count += active
      active &= ~done
      if not active.any():
        break
    atomic_json(Path(output), {
      "iteration": session.saved["iteration"], "native": True, "optimizer_updates": 0,
      "protocol": "first held-out clips; nominal physics, reference reset, no noise; full clip or explicit duration cap",
      "not": "the paired-family heldout128 architecture benchmark or real-robot safety certification",
      "clips": [
        {"name": name, "controls": int(n), "mean_reward": float(reward / n.clamp_min(1)),
         "clip_completed": bool(not still_active)}
        for name, n, reward, still_active in zip(session.names, count, total, active, strict=True)
      ],
    })
  finally:
    session.close()


def render(
  checkpoint: str, output: str, motions: str = "", seconds: float = 12.,
  width: int = 960, height: int = 720, device: str = "cuda:0", fps: int = 25,
) -> None:
  import imageio.v2 as imageio
  from PIL import Image, ImageDraw

  if not 0 < seconds < float("inf") or fps < 1 or fps > 50 or 50 % fps or min(width, height) < 1:
    raise ValueError("Require a finite positive duration/size and an fps dividing 50")
  session = NativePlayback(
    checkpoint, device, max(2, len([name for name in motions.split(",") if name.strip()]) or 6), motions,
    width=width, height=height, render_mode="rgb_array",
  )
  try:
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output, fps=fps, macro_block_size=1) as writer:
      for name in dict.fromkeys(session.names):
        index = session.names.index(name)
        session.env.cfg.viewer.env_idx = index
        session.observations, _ = session.env.reset()
        for step in range(max(1, int(seconds / .02))):
          _, done = session.step()
          if step % (50 // fps) == 0:
            frame = session.env.render()
            if frame is None:
              raise RuntimeError("Native renderer returned no frame")
            image = Image.fromarray(frame)
            ImageDraw.Draw(image).text((8, 8), f"{name} | {(step + 1) * .02:.2f}s", fill="white", stroke_width=1)
            writer.append_data(np.asarray(image))
          if bool(done[index]):
            break
    print(f"[mocap] rendered native clips {session.names}: {output}")
  finally:
    session.close()
