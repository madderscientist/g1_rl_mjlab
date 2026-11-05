"""Full-corpus native-policy references with bounded, read-only frame residency."""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

import torch

from g1_lower_rl.assets.g1_gloria import get_spec
from g1_lower_rl.tasks.motion_tracking.corpus import TRACKED_BODIES, MappedMotionCorpus
from g1_lower_rl.world_model.adapters.kinematics import StaticKinematicTemplate, TorchKinematics
from g1_lower_rl.world_model.adapters.math import conjugate, rotate
from g1_lower_rl.world_model.model import BODY_NAMES


class FrameClipIndex:
  """Search clip boundaries instead of materializing a32-million-element GPU table."""

  def __init__(self, start: torch.Tensor, length: torch.Tensor):
    self.order = start.argsort()
    self.last = (start + length - 1)[self.order]
    ordered_start = start[self.order]
    if int(ordered_start[0]) != 0 or not torch.equal(ordered_start[1:], self.last[:-1] + 1):
      raise ValueError("Full mocap clips must cover the frame file exactly, without gaps or overlaps")

  def __getitem__(self, frame: torch.Tensor) -> torch.Tensor:
    if frame.dtype != torch.long or ((frame < 0) | (frame > self.last[-1])).any():
      raise IndexError("Mocap frame outside the full corpus")
    return self.order[torch.searchsorted(self.last, frame.contiguous())]


class MmapMotionLibrary:
  def __init__(
    self,
    directory: Path,
    body_names: tuple[str, ...],
    default_joint_pos: torch.Tensor,
    device: str,
    bin_frames: int = 50,
  ):
    if tuple(body_names) != TRACKED_BODIES or bin_frames < 1:
      raise ValueError("Require the native14 tracked bodies and positive bin_frames")
    self.device = torch.device(device)
    self.bin_frames = bin_frames
    self.body_names = body_names
    self.default_joint_pos = torch.as_tensor(default_joint_pos, device=device, dtype=torch.float32).reshape(29)
    body_ids = torch.tensor([BODY_NAMES.index(name) for name in body_names])
    self.corpus = MappedMotionCorpus(str(directory / "train"), body_ids, device, include_validation=True)
    self.corpus.verify()
    manifest = json.loads(self.corpus.manifest_path.read_text())
    families = {split: {r["family"] for r in manifest["clips"] if r["split"] == split} for split in ("train", "validation")}
    if families["train"] & families["validation"]:
      raise ValueError("Mocap train/validation family leakage")
    hashes = {split: {r["sha256"] for r in manifest["clips"] if r["split"] == split} for split in ("train", "validation")}
    if hashes["train"] & hashes["validation"]:
      raise ValueError("Mocap train/validation content leakage")
    if manifest.get("limited_to_clips") != 0 or len(manifest["clips"]) != self.corpus.num_motions:
      raise ValueError("Frozen-baseline pretraining requires the full, unfiltered mocap corpus")
    if self.corpus.time_step_total != manifest["frames"]:
      raise ValueError("Mocap split metadata does not cover every packed frame")
    self._native = import_module("supermimic.envs.mdp.motion_library")
    self.clips = [
      self._native.ClipInfo(name, int(start), int(length))
      for name, start, length in zip(self.corpus.names, self.corpus.start_idx.cpu(), self.corpus.num_frames.cpu(), strict=True)
    ]
    self.num_clips = self.corpus.num_motions
    self.num_frames = self.corpus.time_step_total
    self.clip_start, self.clip_len = self.corpus.start_idx, self.corpus.num_frames
    self.clip_last = self.clip_start + self.clip_len - 1
    self.frame_clip = FrameClipIndex(self.clip_start, self.clip_len)
    count = (self.clip_len + bin_frames - 1) // bin_frames
    self.clip_bin_offset = count.cumsum(0) - count
    self.num_bins = int(count.sum())
    self.bin_clip = torch.repeat_interleave(torch.arange(self.num_clips, device=device), count)
    phase = (torch.arange(self.num_bins, device=device) - self.clip_bin_offset[self.bin_clip]) * bin_frames
    self.bin_start = self.clip_start[self.bin_clip] + phase
    self.bin_end = torch.minimum(self.bin_start + bin_frames, self.clip_last[self.bin_clip] + 1)
    self.visited = torch.zeros(self.num_clips, device=device, dtype=torch.bool)
    self._fk = TorchKinematics(StaticKinematicTemplate.from_mujoco(get_spec().compile())).to(device=device, dtype=torch.float32)
    self._body_ids = body_ids.to(device)
    self.metadata = {
      "manifest_path": str(self.corpus.manifest_path),
      "manifest_sha256": self.corpus.manifest_sha256,
      "data_sha256": self.corpus.data_sha256,
      "data_sha256_verified": True,
      "clips": self.num_clips,
      "frames": self.num_frames,
      "train_clips": self.corpus.training_motions,
      "validation_clips": self.num_clips - self.corpus.training_motions,
      "family_split_disjoint": True,
      "content_split_disjoint": True,
      "storage": "read_only_CPU_mmap_maximum64MiB_GPU_frame_cache",
      "joint_projection": "legacy31_ignore_gripper_state_to_canonical29" if self.corpus.legacy31 else "canonical29",
      "gripper_state": "excluded_from_reference_policy_and_WM_coordinates_no_clip_filtering",
      "body_reference": "FK_of_actual_fixed29_asset_not_legacy_body_geometry",
      "sampling": "uniform_clip_then_uniform_frame_with_fixed_env_train_validation_partition",
      "bin_semantics": "ceil_including_short_tail_as_native_policy",
    }

  def ref(self, frame: torch.Tensor):
    c = self.corpus
    return self._native.RefState(
      c.body_pos_w[frame][..., 0, :],
      c.body_quat_w[frame][..., 0, :],
      c.joint_pos[frame],
      c.body_lin_vel_w[frame][..., 0, :],
      c.body_ang_vel_w[frame][..., 0, :],
      c.joint_vel[frame],
    )

  def bodies(self, frame: torch.Tensor):
    r = self.ref(frame)
    state = torch.cat(
      (
        r.root_pos_w,
        r.root_quat_w,
        r.joint_pos,
        r.root_lin_vel_w,
        rotate(conjugate(r.root_quat_w), r.root_ang_vel_w),
        r.joint_vel,
      ),
      dim=-1,
    )
    body = self._fk(state)
    return self._native.RefBodies(*(value[..., self._body_ids, :] for value in body[:4]))

  def bin_of_frame(self, frame: torch.Tensor) -> torch.Tensor:
    clip = self.frame_clip[frame]
    return self.clip_bin_offset[clip] + (frame - self.clip_start[clip]) // self.bin_frames

  def last_frame_of(self, frame: torch.Tensor) -> torch.Tensor:
    return self.clip_last[self.frame_clip[frame]]

  def goal_window(self, frame: torch.Tensor, k: int) -> torch.Tensor:
    if k < 1:
      raise ValueError("Goal window must have at least one frame")
    indices = torch.minimum(frame[:, None] + torch.arange(k, device=self.device), self.last_frame_of(frame)[:, None])
    return self._native.goal_features(self.ref(indices.reshape(-1)), self.default_joint_pos).reshape(len(frame), 68 * k)

  def sample(self, count: int, *, validation: bool) -> torch.Tensor:
    boundary = self.corpus.training_motions
    low, high = (boundary, self.num_clips) if validation else (0, boundary)
    clip = torch.randint(low, high, (count,), device=self.device)
    self.visited[clip] = True
    phase = (torch.rand(count, device=self.device) * self.clip_len[clip]).long()
    return self.clip_start[clip] + phase

  def coverage(self) -> dict[str, int]:
    boundary = self.corpus.training_motions
    return {
      "visited_train_clips": int(self.visited[:boundary].sum()),
      "visited_validation_clips": int(self.visited[boundary:].sum()),
    }

  def summary(self) -> str:
    return f"{self.num_clips} clips, {self.num_frames} frames; full CPU mmap, canonical29, disjoint family holdout"
