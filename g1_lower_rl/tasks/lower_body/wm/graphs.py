"""CUDA Graph wrappers for fixed-shape WM autoregressive backward and H32 velocity BPTT ticks."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from torch import nn

GRAD_INPUT_NAMES = ("state", "history", "aux")


class CapturedWmMicrobatch:
  """Capture one fixed-shape WM autoregressive loss + backward microbatch (accumulating grads)."""

  def __init__(
    self,
    loss_fn: Callable[[dict[str, torch.Tensor]], tuple[torch.Tensor, torch.Tensor]],
    example: dict[str, torch.Tensor],
    parameters: Iterable[nn.Parameter],
    *,
    pool: tuple[int, int] | None = None,
  ):
    self.inputs = {name: value.detach().clone() for name, value in example.items()}
    self.parameters = tuple(p for p in parameters if p.requires_grad)
    device = next(iter(self.inputs.values())).device

    # Ensure persistent .grad buffers exist before graph capture so backward accumulates in-place.
    for p in self.parameters:
      if p.grad is None:
        p.grad = torch.zeros_like(p)

    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
      for _ in range(2):
        for p in self.parameters:
          assert p.grad is not None
          p.grad.zero_()
        total_loss, _ = loss_fn(self.inputs)
        total_loss.backward()
    torch.cuda.current_stream(device).wait_stream(stream)

    for p in self.parameters:
      assert p.grad is not None
      p.grad.zero_()

    self.graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(self.graph, stream=stream, pool=pool):
      total_loss, state_loss = loss_fn(self.inputs)
      total_loss.backward()
    torch.cuda.current_stream(device).wait_stream(stream)
    self.outputs = (total_loss.detach(), state_loss.detach())
    for p in self.parameters:
      assert p.grad is not None
      p.grad.zero_()

  def zero_grad(self) -> None:
    for p in self.parameters:
      if p.grad is not None:
        p.grad.zero_()

  def __call__(self, values: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    if values.keys() != self.inputs.keys():
      raise ValueError("WM CUDA graph input keys changed")
    with torch.no_grad():
      for name, value in values.items():
        target = self.inputs[name]
        if value.shape != target.shape or value.dtype != target.dtype or value.device != target.device:
          raise ValueError(f"WM CUDA graph input contract changed: {name}")
        target.copy_(value)
    self.graph.replay()
    return self.outputs


class CapturedVelocityRolloutStep:
  """Checkpoint one 20ms velocity imagination tick; replay forward & VJP CUDA graphs across H32."""

  def __init__(
    self,
    forward: Callable[[dict[str, torch.Tensor]], tuple[torch.Tensor, ...]],
    example: dict[str, torch.Tensor],
    parameters: Iterable[nn.Parameter],
    *,
    pool: tuple[int, int] | None = None,
  ):
    self.names = tuple(example)
    if self.names[:3] != GRAD_INPUT_NAMES:
      raise ValueError(f"Rollout step inputs must begin with {GRAD_INPUT_NAMES}, got {self.names[:3]}")
    self.inputs = {
      name: value.detach().clone().requires_grad_(name in GRAD_INPUT_NAMES)
      for name, value in example.items()
    }
    self.parameters = tuple(p for p in parameters if p.requires_grad)
    device = self.inputs["state"].device

    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream), torch.enable_grad():
      outputs = forward(self.inputs)
      self.cotangents = tuple(torch.ones_like(value) for value in outputs)

      def differentiate() -> tuple[torch.Tensor | None, ...]:
        result = forward(self.inputs)
        return torch.autograd.grad(
          result,
          (self.inputs["state"], self.inputs["history"], self.inputs["aux"], *self.parameters),
          grad_outputs=self.cotangents,
          allow_unused=True,
        )

      gradients = differentiate()
      for parameter, gradient in zip(self.parameters, gradients[3:], strict=True):
        if gradient is not None and parameter.grad is None:
          parameter.grad = torch.zeros_like(parameter)
      differentiate()
      del outputs, gradients
    torch.cuda.current_stream(device).wait_stream(stream)

    self.forward_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(self.forward_graph, stream=stream, pool=pool), torch.enable_grad():
      outputs = forward(self.inputs)
    self.outputs = tuple(value.detach() for value in outputs)
    del outputs

    self.backward_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(self.backward_graph, stream=stream, pool=pool), torch.enable_grad():
      gradients = differentiate()
    self.gradients = tuple(None if value is None else value.detach() for value in gradients)
    torch.cuda.current_stream(device).wait_stream(stream)

  def copy_inputs(self, values: tuple[torch.Tensor, ...]) -> None:
    if len(values) != len(self.names):
      raise ValueError("Velocity rollout graph input count changed")
    with torch.no_grad():
      for name, value in zip(self.names, values, strict=True):
        target = self.inputs[name]
        if value.shape != target.shape or value.dtype != target.dtype or value.device != target.device:
          raise ValueError(f"Velocity rollout graph input contract changed: {name}")
        target.copy_(value)

  def __call__(self, values: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    if tuple(values) != self.names:
      raise ValueError("Velocity rollout graph input ordering changed")
    return _VelocityRolloutStepFn.apply(self, *values.values(), *self.parameters)


class _VelocityRolloutStepFn(torch.autograd.Function):
  @staticmethod
  def forward(ctx, graph: CapturedVelocityRolloutStep, *values):
    n_in = len(graph.names)
    inputs = values[:n_in]
    ctx.graph = graph
    ctx.save_for_backward(*inputs)
    ctx.set_materialize_grads(False)
    graph.copy_inputs(inputs)
    graph.forward_graph.replay()
    return tuple(value.clone() for value in graph.outputs)

  @staticmethod
  def backward(ctx, *cotangents):
    graph: CapturedVelocityRolloutStep = ctx.graph
    graph.copy_inputs(ctx.saved_tensors)
    for target, gradient in zip(graph.cotangents, cotangents, strict=True):
      if gradient is None:
        target.zero_()
      else:
        target.copy_(gradient)
    graph.backward_graph.replay()
    gradients = tuple(None if value is None else value.clone() for value in graph.gradients)
    n_non_grad_inputs = len(graph.names) - 3
    return (None, *gradients[:3], *((None,) * n_non_grad_inputs), *gradients[3:])
