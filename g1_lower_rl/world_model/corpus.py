"""Build a deterministic, memory-bounded symlink corpus without altering motions."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def prepare_corpus(
  source: Path,
  destination: Path,
  *,
  max_clips: int = 384,
  max_bytes: int = 512 * 1024 * 1024,
  seed: int = 20261026,
) -> dict:
  if max_clips < 1 or max_bytes < 1:
    raise ValueError("Corpus limits must be positive")
  source = source.resolve()
  files = sorted(source.glob("*.npz"))
  if not files:
    raise FileNotFoundError(f"No motions under {source}")
  groups: dict[str, list[Path]] = defaultdict(list)
  for path in files:
    parts = path.stem.split("_")
    family = "_".join(parts[:2]) if parts[0] in ("amass", "style", "ground") else parts[0].rstrip("0123456789")
    groups[family].append(path)
  rng = random.Random(seed)
  for paths in groups.values():
    rng.shuffle(paths)
  families = sorted(groups)
  rng.shuffle(families)
  selected = []
  used_bytes = 0
  while families and len(selected) < max_clips:
    remaining = []
    for family in families:
      paths = groups[family]
      while paths:
        path = paths.pop()
        size = path.stat().st_size
        if used_bytes + size <= max_bytes:
          selected.append({"name": path.name, "source": str(path), "bytes": size, "family": family})
          used_bytes += size
          break
      if paths:
        remaining.append(family)
      if len(selected) == max_clips:
        break
    families = remaining
  if not selected:
    raise ValueError("No source motion fits the corpus memory budget")
  manifest = {
    "version": 1,
    "source": str(source),
    "seed": seed,
    "max_clips": max_clips,
    "max_bytes": max_bytes,
    "selected_clips": len(selected),
    "source_clips": len(files),
    "selected_bytes": used_bytes,
    "motions": selected,
  }
  manifest["selection_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
  destination.mkdir(parents=True, exist_ok=True)
  manifest_path = destination / "manifest.json"
  if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
    raise FileExistsError(f"Refusing to replace a different corpus manifest: {manifest_path}")
  expected_names = {item["name"] for item in selected}
  unexpected = {path.name for path in destination.glob("*.npz")} - expected_names
  if unexpected:
    raise FileExistsError(f"Unexpected motions already exist in {destination}: {sorted(unexpected)[:5]}")
  for item in selected:
    link = destination / item["name"]
    target = Path(item["source"])
    if link.is_symlink() and link.resolve() == target:
      continue
    if link.exists() or link.is_symlink():
      raise FileExistsError(f"Refusing to overwrite {link}")
    link.symlink_to(target)
  manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
  return manifest


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source", type=Path, default=Path("motions/lafan1"))
  parser.add_argument("--destination", type=Path, default=Path("run_logs/wm_teacher_corpus"))
  parser.add_argument("--max-clips", type=int, default=384)
  parser.add_argument("--max-mib", type=int, default=512)
  parser.add_argument("--seed", type=int, default=20261026)
  args = parser.parse_args()
  manifest = prepare_corpus(
    args.source,
    args.destination,
    max_clips=args.max_clips,
    max_bytes=args.max_mib * 1024 * 1024,
    seed=args.seed,
  )
  print(json.dumps({key: value for key, value in manifest.items() if key != "motions"}, indent=2))


if __name__ == "__main__":
  main()
