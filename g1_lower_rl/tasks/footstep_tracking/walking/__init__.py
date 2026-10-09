"""Continuous walking profiles: basic gait and precision training."""

from .env_cfg import walk_first_env_cfg
from .precision import precision_env_cfg

__all__ = ["walk_first_env_cfg", "precision_env_cfg"]