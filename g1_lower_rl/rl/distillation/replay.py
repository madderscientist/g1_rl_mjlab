"""DAgger 的有界样本缓存和分布式梯度同步，不依赖模型结构。"""

import torch
from tensordict import TensorDict
from torch import distributed


class DaggerBuffer:
  """每项观测为二维张量；存入时复制，避免 env.step 原地更新导致标签错配。"""

  def __init__(self, observations: TensorDict, action_dim: int, capacity: int):
    if capacity <= 0:
      raise ValueError("Replay capacity must be positive")
    self.capacity = capacity
    self.position = 0
    self.size = 0
    self.observations = TensorDict(
      {name: value.new_empty((capacity, value.shape[-1])) for name, value in observations.items()},
      batch_size=[capacity],
    )
    self.labels = next(iter(observations.values())).new_empty((capacity, action_dim))

  @torch.no_grad()
  def add(self, observations: TensorDict, labels: torch.Tensor) -> None:
    count = min(len(labels), self.capacity)
    indexes = (torch.arange(count, device=labels.device) + self.position) % self.capacity
    for name, values in self.observations.items():
      values[indexes] = observations[name][-count:]
    self.labels[indexes] = labels[-count:]
    self.position = (self.position + count) % self.capacity
    self.size = min(self.capacity, self.size + count)

  def sample(self, batch_size: int) -> tuple[TensorDict, torch.Tensor]:
    if self.size == 0:
      raise ValueError("Cannot sample empty replay")
    indexes = torch.randint(self.size, (batch_size,), device=self.labels.device)
    return self.observations[indexes], self.labels[indexes]


def teacher_probability(iteration: int, teacher_steps: int) -> float:
  """只退火谁来控制，不改变监督标签的来源；按累计轮次计算。"""
  if teacher_steps <= 0:
    return 0.0
  return max(0.0, 1.0 - iteration / teacher_steps)


def average_gradients(model) -> None:
  """各进程必须具有相同的可训练参数和梯度结构；先平均，再统一裁剪和更新。"""
  if not distributed.is_initialized():
    return
  gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
  flat = torch.cat([gradient.reshape(-1) for gradient in gradients])
  distributed.all_reduce(flat)
  flat.div_(distributed.get_world_size())
  offset = 0
  for gradient in gradients:
    gradient.copy_(flat[offset:offset + gradient.numel()].view_as(gradient))
    offset += gradient.numel()