"""动作跟踪任务的事件项。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@requires_model_fields("body_mass", recompute=RecomputeLevel.set_const)
def payload_mass(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor | None,
  mass_range: tuple[float, float],
  asset_cfg: SceneEntityCfg,
) -> None:
  """在指定刚体上挂负载，质量按 ``mass_range`` **均匀**采样，负值一律视作空载。

  下界取负是为了让「空手」有一块有限概率，而不是概率为零的边界点——真机大部分
  时间是空手的。例如 ``(-0.5, 1.0)`` 有 1/3 的环境空载，其余在 (0, 1] 上均匀。

  **不用 dr.pseudo_inertia**：那个按 e^(2*alpha) 整体缩放，一是质量分布被指数
  拉向重端（alpha 均匀不等于质量均匀），二是它把刚体自身的惯量张量一起放大，
  相当于假设负载和夹爪一样"摊开"。实际负载是握在夹爪里的紧凑物体：绕肩/肘的
  m*d^2 由质量自动带出，被忽略的只是它绕自身质心的转动惯量——对 1 kg、尺度
  10 cm 的物体约 1e-3 kg·m^2，比手臂惯量小两个数量级。

  **不用 dr.body_mass**：那个不支持把负值截断成空载，会真的把夹爪减到比实物还轻。
  """
  asset = env.scene[asset_cfg.name]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.int)

  body_ids = asset.indexing.body_ids.to(env.device)[asset_cfg.body_ids]
  # 标称质量取自未展开的单份模型，避免多次触发时在已改过的值上累加。
  nominal = torch.as_tensor(
    env.sim.mj_model.body_mass[body_ids.cpu().numpy()],
    device=env.device,
    dtype=env.sim.model.body_mass.dtype,
  )
  payload = sample_uniform(
    mass_range[0], mass_range[1], (len(env_ids), len(body_ids)), device=env.device
  ).clamp_min(0.0)

  env_grid, body_grid = torch.meshgrid(env_ids, body_ids, indexing="ij")
  env.sim.model.body_mass[env_grid, body_grid] = nominal + payload
