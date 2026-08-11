from pathlib import Path

PKG_PATH: Path = Path(__file__).parent
"""Root of the installed package; asset paths are resolved against it."""

__all__ = ["PKG_PATH"]
