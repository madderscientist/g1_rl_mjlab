"""Shared per-episode link density randomization for G1 tasks."""

from __future__ import annotations

import math

import torch
from mjlab.managers.event_manager import EventTermCfg, RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform

LINK_MASS_SCALE_RANGE = (0.95, 1.05)


@requires_model_fields("body_mass", "body_inertia", recompute=RecomputeLevel.set_const)
class RandomizeLinkMass:
  """Resample nominal link density while preserving startup additive payloads."""

  def __init__(self, cfg: EventTermCfg, env):
    asset_cfg = cfg.params["asset_cfg"]
    self.body_ids = env.scene[asset_cfg.name].indexing.body_ids[asset_cfg.body_ids].to(device=env.device, dtype=torch.long)
    indices = self.body_ids.cpu().numpy()
    self.nominal_mass = torch.as_tensor(env.sim.mj_model.body_mass[indices], device=env.device).clone()
    self.nominal_inertia = torch.as_tensor(env.sim.mj_model.body_inertia[indices], device=env.device).clone()
    self.mass_offset: torch.Tensor | None = None

  def __call__(
    self,
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    scale_range: tuple[float, float] = LINK_MASS_SCALE_RANGE,
  ) -> None:
    del asset_cfg
    lower, upper = scale_range
    if not (math.isfinite(lower) and math.isfinite(upper) and 0 < lower <= upper):
      raise ValueError("Link mass scales must be finite, positive and ordered")
    if env_ids is None:
      env_ids = torch.arange(env.num_envs, device=env.device)
    else:
      env_ids = env_ids.to(device=env.device, dtype=torch.long)
    if env_ids.numel() == 0:
      return
    model = env.sim.model
    if self.mass_offset is None:
      self.nominal_mass = self.nominal_mass.to(dtype=model.body_mass.dtype)
      self.nominal_inertia = self.nominal_inertia.to(dtype=model.body_inertia.dtype)
      self.mass_offset = (model.body_mass[:, self.body_ids] - self.nominal_mass).clone()
    scale = sample_uniform(lower, upper, (len(env_ids), len(self.body_ids)), device=env.device)
    env_grid, body_grid = torch.meshgrid(env_ids, self.body_ids, indexing="ij")
    model.body_mass[env_grid, body_grid] = self.nominal_mass * scale + self.mass_offset[env_ids]
    model.body_inertia[env_grid, body_grid] = self.nominal_inertia * scale.unsqueeze(-1)


def make_link_mass_event() -> EventTermCfg:
  """Independent uniform +/-5% per robot body, held constant until its next reset."""
  return EventTermCfg(
    func=RandomizeLinkMass,
    mode="reset",
    params={"asset_cfg": SceneEntityCfg("robot"), "scale_range": LINK_MASS_SCALE_RANGE},
  )