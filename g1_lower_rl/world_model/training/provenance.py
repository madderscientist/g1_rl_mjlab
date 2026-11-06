"""Content-addressed, immutable snapshots of the code used for model fitting."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def snapshot_sources(output_dir: Path, *, write: bool) -> dict[str, Any]:
  training = Path(__file__).resolve().parent
  model_package = training.parent
  repository = model_package.parent.parent
  paths = [
    *sorted(model_package.glob("*.py")),
    repository / "g1_lower_rl/assets/g1_gloria/__init__.py",
    *sorted((repository / "g1_lower_rl/assets/g1_gloria/xmls").glob("*.xml")),
    *sorted(training.glob("*.py")),
    *sorted((model_package / "adapters").glob("*.py")),
    *sorted((repository / "g1_lower_rl/tasks/motion_tracking/mdp").glob("*.py")),
    repository / "g1_lower_rl/tasks/motion_tracking/corpus.py",
    repository / "g1_lower_rl/tasks/domain_randomization.py",
    repository / "g1_lower_rl/rl/supermimic.py",
    *sorted((repository / "g1_lower_rl/tasks/motion_tracking/training").glob("*.py")),
  ]
  script = repository / "scripts" / "train_wm.py"
  if script.is_file():
    paths.append(script)
  for name in ("train_mocap.py", "package_mocap.py", "eval_corpus.py", "render_policy.py"):
    paths.append(repository / "scripts" / name)
  contents = {str(path.relative_to(repository)): path.read_bytes() for path in paths}
  hashes = {name: hashlib.sha256(data).hexdigest() for name, data in sorted(contents.items())}
  digest = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
  directory = output_dir / "sources" / digest
  manifest = {"format_version": 1, "sha256": digest, "files": hashes}
  if write:
    for name, data in contents.items():
      target = directory / name
      target.parent.mkdir(parents=True, exist_ok=True)
      if target.exists():
        if target.read_bytes() != data:
          raise RuntimeError(f"Immutable source snapshot was modified: {target}")
        continue
      with target.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    manifest_path = directory / "manifest.json"
    serialized = json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    if manifest_path.exists():
      if manifest_path.read_text() != serialized:
        raise RuntimeError(f"Immutable source manifest was modified: {manifest_path}")
    else:
      with manifest_path.open("x") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())
  return {**manifest, "snapshot_directory": str(directory.resolve())}
