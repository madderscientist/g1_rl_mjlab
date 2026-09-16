"""由调用方定义模型构造方式；配置和权重加载不依赖 V1/V2 命名。"""

import copy
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDict
from torch import nn


def resolve_model_config(
  config: dict[str, Any] | None,
  saved: dict[str, Any] | None = None,
  fallback: dict[str, Any] | None = None,
) -> dict[str, Any]:
  """统一配置优先级：显式配置、存档配置、应用默认；空字典也是显式配置。"""
  if config is None and saved is not None:
    config = saved.get("model_config")
    if config is None:
      config = saved.get("infos", {}).get("distillation", {}).get("actor_config")
  if config is None:
    config = fallback
  if config is None:
    raise ValueError("Model config is required: supply it explicitly or use a checkpoint containing model_config")
  if not isinstance(config, dict):
    raise TypeError("Model configuration must be a dictionary")
  return copy.deepcopy(config)


@dataclass
class ModelSpec:
  """模型配置或检查点，与确定性 TensorDict -> actions 构造函数组合使用。

  factory(observations, config) 返回 nn.Module；配置由 factory 自己解释。
  config 显式提供时优先，否则从本包存档的 model_config 或旧蒸馏 actor_config
  恢复。普通 PPO state_dict 不包含激活函数等信息，必须额外提供 config。
  只加载可信的本地检查点。模块应返回 [环境数, 动作数]，不能默认采样探索噪声。
  """

  factory: Callable[[TensorDict, dict[str, Any]], nn.Module]
  config: dict[str, Any] | None = None
  checkpoint: str | Path | None = None
  state_key: str = "actor_state_dict"

  def build(self, observations: TensorDict, device: str) -> tuple[nn.Module, dict[str, Any]]:
    saved = None
    if self.checkpoint is not None:
      saved = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
    config = resolve_model_config(self.config, saved)
    model = self.factory(observations, copy.deepcopy(config)).to(device)
    if saved is not None:
      model.load_state_dict(saved[self.state_key], strict=True)
    return model, config