"""Export a portable full-precision WM starting point, without optimizer state."""

from pathlib import Path

import tyro

from g1_lower_rl.world_model import export_pretrained_checkpoint, pretrained_checkpoint_path


def main(source: Path, destination: Path | None = None, overwrite: bool = False) -> None:
  saved = export_pretrained_checkpoint(
    source,
    destination if destination is not None else pretrained_checkpoint_path(),
    overwrite=overwrite,
  )
  print(f"Shared WM initialization: {saved}")
  print(f"Manifest: {saved.with_suffix('.json')}")


if __name__ == "__main__":
  tyro.cli(main)
