"""Versioned, streaming mocap packing and bounded CPU-mmap/GPU-gather access."""

from __future__ import annotations

import hashlib
import json
import math
import time
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from g1_lower_rl.tasks.motion_tracking.mdp.motion_corpus import canonical_joint_motion
from g1_lower_rl.world_model.model import BODY_NAMES, JOINT_NAMES

TRACKED_BODIES = (
  "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
  "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
  "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
  "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
)
FIELDS = {
  "joint_pos": (0, 29, (29,)),
  "joint_vel": (29, 58, (29,)),
  "body_pos_w": (58, 100, (14, 3)),
  "body_quat_w": (100, 156, (14, 4)),
  "body_lin_vel_w": (156, 198, (14, 3)),
  "body_ang_vel_w": (198, 240, (14, 3)),
}
FRAME_WIDTH = 240


def file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def motion_family(name: str) -> str:
  # Keep mirrored variants and overlapping 100style cuts in the same split.
  return name.removesuffix("_mir").split("__", 1)[0]


@dataclass(frozen=True)
class PrepareMocapConfig:
  source: str = "motions/expanded_all"
  output: str = "motions/mocap_supermimic_fixed29_v2"
  validation_fraction: float = 0.02
  split_seed: int = 0
  max_clips: int = 0


def prepare_mocap(cfg: PrepareMocapConfig) -> dict:
  if not 0 <= cfg.validation_fraction < 1 or cfg.max_clips < 0:
    raise ValueError("Require validation_fraction in [0,1) and nonnegative max_clips")
  source, output = Path(cfg.source).resolve(), Path(cfg.output).resolve()
  files = sorted(source.glob("*.npz"))
  if cfg.max_clips:
    files = files[:cfg.max_clips]
  if not files:
    raise FileNotFoundError(f"No mocap NPZ files in {source}")
  if output.exists() and any(output.iterdir()):
    raise FileExistsError(f"Refusing to overwrite an existing corpus: {output}")
  output.mkdir(parents=True, exist_ok=True)
  records = []
  parent: dict[str, str] = {}
  by_hash: dict[str, str] = {}

  def root(key: str) -> str:
    parent.setdefault(key, key)
    if parent[key] != key:
      parent[key] = root(parent[key])
    return parent[key]

  start_time = time.monotonic()
  for index, path in enumerate(files):
    with zipfile.ZipFile(path) as archive, archive.open("joint_pos.npy") as stream:
      version = np.lib.format.read_magic(stream)
      if version == (1, 0):
        shape, _, _ = np.lib.format.read_array_header_1_0(stream)
      elif version == (2, 0):
        shape, _, _ = np.lib.format.read_array_header_2_0(stream)
      else:
        raise ValueError(f"Unsupported NPY header version {version}: {path}")
    if len(shape) != 2 or shape[1] not in (29, 31) or shape[0] < 2:
      raise ValueError(f"Invalid joint_pos shape {shape}: {path}")
    digest = file_sha256(path)
    family = motion_family(path.stem)
    if digest in by_hash:
      a, b = root(family), root(by_hash[digest])
      parent[max(a, b)] = min(a, b)
    else:
      by_hash[digest] = family
      root(family)
    records.append({
      "name": path.stem, "frames": int(shape[0]), "sha256": digest,
      "family": family, "resolved_source": str(path.resolve()),
    })
    if (index + 1) % 5000 == 0:
      print(json.dumps({"indexing": index + 1, "seconds": time.monotonic() - start_time}), flush=True)
  total = sum(r["frames"] for r in records)
  partial = output / "frames.bin.partial"
  mapped = np.memmap(partial, mode="w+", dtype="<f4", shape=(total, FRAME_WIDTH))
  offset = 0
  bodies = [BODY_NAMES.index(name) for name in TRACKED_BODIES]
  try:
    for i, (path, record) in enumerate(zip(files, records, strict=True)):
      count = record["frames"]
      with np.load(path, allow_pickle=False) as data:
        if "fps" not in data or not np.allclose(data["fps"], 50.0):
          raise ValueError(f"Mocap must explicitly be 50 Hz: {path}")
        joint_pos, joint_vel = canonical_joint_motion(data["joint_pos"], data["joint_vel"])
        for field, (lo, hi, tail) in FIELDS.items():
          array = {"joint_pos": joint_pos, "joint_vel": joint_vel}.get(field)
          if array is None:
            array = data[field]
          expected = (count, 29) if field.startswith("joint_") else (count, 44, tail[-1])
          if array.shape != expected or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError(f"Invalid {field}: expected finite float32 {expected}: {path}")
          if field.startswith("body_"):
            selected = array[:, bodies].copy()
            if field == "body_pos_w":
              selected[..., :2] -= array[0, 0, :2]
            if field == "body_quat_w" and not np.allclose(np.linalg.norm(selected, axis=-1), 1, atol=1e-3):
              raise ValueError(f"Non-unit reference quaternions: {path}")
          else:
            selected = array
          mapped[offset:offset + count, lo:hi] = selected.reshape(count, hi - lo)
      family = root(record["family"])
      fraction = int(hashlib.sha256(f"{cfg.split_seed}:{family}".encode()).hexdigest()[:16], 16) / 2**64
      record.update(offset=offset, family=family, split="validation" if fraction < cfg.validation_fraction else "train")
      offset += count
      if (i + 1) % 1000 == 0:
        print(json.dumps({"packed": i + 1, "frames": offset, "seconds": time.monotonic() - start_time}), flush=True)
    mapped.flush()
    del mapped
    data_hash = file_sha256(partial)
    partial.replace(output / "frames.bin")
    manifest = {
      "version": 1, "dtype": "<f4", "frame_width": FRAME_WIDTH, "fps": 50,
      "frames": total, "bytes": total * FRAME_WIDTH * 4, "data_sha256": data_hash,
      "joint_names": list(JOINT_NAMES), "source_body_names": list(BODY_NAMES),
      "robot_contract": "gloria_fixed_closed29_v2",
      "body_names": list(TRACKED_BODIES), "source": str(source),
      "split_seed": cfg.split_seed, "validation_fraction": cfg.validation_fraction,
      "limited_to_clips": cfg.max_clips, "clips": records,
      "evaluation_caveat": "The warm-start actor/WM may have seen these motions. This is not a zero-shot holdout.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    for split in ("train", "validation"):
      directory = output / split
      directory.mkdir()
      (directory / "index.json").write_text(json.dumps({"version": 1, "split": split}) + "\n")
    print(json.dumps({"complete": len(records), "frames": total, "hours": total / 50 / 3600,
                      "bytes": manifest["bytes"], "sha256": data_hash}), flush=True)
    return manifest
  except BaseException:
    # Only remove the incomplete file owned by this invocation; never the corpus directory.
    partial.unlink(missing_ok=True)
    raise


class _FrameField:
  def __init__(self, corpus: MappedMotionCorpus, field: str):
    self.corpus, self.field = corpus, field

  def __getitem__(self, key) -> torch.Tensor:
    first, *rest = key if isinstance(key, tuple) else (key,)
    if not isinstance(first, torch.Tensor):
      raise TypeError("Mapped mocap requires explicit tensor frame indices; full-corpus materialization is disabled")
    lo, hi, tail = FIELDS[self.field]
    rows = self.corpus.gather(first)[..., lo:hi].reshape(*first.shape, *tail)
    return rows[(slice(None),) * first.ndim + tuple(rest)]


class MappedMotionCorpus:
  """Only indices/metadata live permanently on GPU; frame gathers are bounded."""

  def __init__(
    self, motion_dir: str, body_indexes: torch.Tensor, device: str = "cpu", *,
    include_validation: bool = False,
  ):
    directory = Path(motion_dir).resolve()
    descriptor = json.loads((directory / "index.json").read_text())
    if descriptor.get("version") != 1 or descriptor.get("split") not in ("train", "validation"):
      raise ValueError(f"Invalid mocap split descriptor: {directory}")
    self.manifest_path = directory.parent / "manifest.json"
    self.manifest_sha256 = file_sha256(self.manifest_path)
    manifest = json.loads(self.manifest_path.read_text())
    expected_bodies = [BODY_NAMES[i] for i in body_indexes.cpu().tolist()]
    legacy_names = list(JOINT_NAMES)
    legacy_names.insert(22, "left_eccentric_joint")
    legacy_names.append("right_eccentric_joint")
    self.legacy31 = manifest.get("joint_names") == legacy_names
    source_width = 244 if self.legacy31 else FRAME_WIDTH
    if (manifest.get("version") != 1 or manifest.get("dtype") != "<f4"
        or manifest.get("frame_width") != source_width or manifest.get("fps") != 50
        or manifest.get("joint_names") not in (list(JOINT_NAMES), legacy_names)
        or manifest.get("body_names") != expected_bodies):
      raise ValueError("Packed mocap contract/body order does not match this task")
    self.data_path = directory.parent / "frames.bin"
    total = manifest["frames"]
    if not isinstance(total, int) or total < 2 or self.data_path.stat().st_size != total * source_width * 4:
      raise ValueError("Packed mocap frame file size/shape mismatch")
    self.data_sha256 = manifest["data_sha256"]
    train = [r for r in manifest["clips"] if r["split"] == "train"]
    validation = [r for r in manifest["clips"] if r["split"] == "validation"]
    if include_validation and (descriptor["split"] != "train" or not train or not validation):
      raise ValueError("Joint training requires a nonempty train split and a nonempty held-out motion split")
    records = train + validation if include_validation else (train if descriptor["split"] == "train" else validation)
    if not records:
      raise ValueError(f"Empty {descriptor['split']} mocap split: {directory}")
    for record in records:
      if record["frames"] < 2 or not 0 <= record["offset"] <= total - record["frames"]:
        raise ValueError(f"Invalid packed clip bounds: {record['name']}")
    self.names = [r["name"] for r in records]
    self.training_motions = len(train) if include_validation else len(records)
    self.device = torch.device(device)
    self.num_frames = torch.tensor([r["frames"] for r in records], device=device)
    self.start_idx = torch.tensor([r["offset"] for r in records], device=device)
    self.num_motions = len(records)
    self.time_step_total = sum(r["frames"] for r in records)
    self._mapped = np.memmap(self.data_path, mode="r", dtype="<f4", shape=(total, source_width))
    self._cache: OrderedDict[int, tuple[torch.Tensor, int, torch.Tensor]] = OrderedDict()
    self.joint_pos = _FrameField(self, "joint_pos")
    self.joint_vel = _FrameField(self, "joint_vel")
    self.body_pos_w = _FrameField(self, "body_pos_w")
    self.body_quat_w = _FrameField(self, "body_quat_w")
    self.body_lin_vel_w = _FrameField(self, "body_lin_vel_w")
    self.body_ang_vel_w = _FrameField(self, "body_ang_vel_w")

  def verify(self) -> None:
    if file_sha256(self.data_path) != self.data_sha256:
      raise ValueError(f"Packed mocap SHA256 mismatch: {self.data_path}")

  def gather(self, indices: torch.Tensor) -> torch.Tensor:
    if indices.dtype != torch.long or indices.numel() > 262144:
      raise ValueError("Require int64 indices and at most 262144 frames per gather")
    key = id(indices)
    cached = self._cache.get(key)
    if cached is not None and cached[0] is indices and cached[1] == indices._version:
      self._cache.move_to_end(key)
      return cached[2]
    cpu = indices.detach().cpu().numpy()
    if (cpu < 0).any() or (cpu >= self._mapped.shape[0]).any():
      raise IndexError("Mocap frame index outside the packed corpus")
    array = np.array(self._mapped[cpu], copy=True)
    if self.legacy31:
      flat = array.reshape(-1, 244)
      position, velocity = canonical_joint_motion(flat[:, :31], flat[:, 31:62])
      array = np.concatenate((position, velocity, flat[:, 62:]), axis=-1).reshape(*cpu.shape, FRAME_WIDTH)
    if not np.isfinite(array).all():
      raise ValueError("Nonfinite packed mocap frames")
    result = torch.from_numpy(array).to(self.device)
    self._cache[key] = (indices, indices._version, result)
    # Cache bytes, not just entries: a large reference batch must not grow GPU residency.
    while len(self._cache) > 16 or sum(v[2].numel() * 4 for v in self._cache.values()) > 64 * 1024**2:
      self._cache.popitem(last=False)
    return result

  def make_bins(self, bin_frames: int):
    from .mdp.motion_corpus import MotionBins

    if bin_frames < 1:
      raise ValueError("bin_frames must be positive")
    count = (self.num_frames // bin_frames).clamp_min(1)
    offset = count.cumsum(0) - count
    motion = torch.repeat_interleave(torch.arange(self.num_motions, device=self.device), count)
    phase = (torch.arange(int(count.sum()), device=self.device) - offset[motion]) * bin_frames
    return MotionBins(motion, phase, offset, count, bin_frames)

  def describe(self, fps: float = 50.0) -> str:
    if not math.isclose(fps, 50):
      raise ValueError("Packed mocap is fixed at 50 Hz")
    return f"{self.num_motions} clips, {self.time_step_total} frames; CPU mmap, <=64MiB GPU frame cache"
