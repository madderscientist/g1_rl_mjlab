"""Fixed-shape forward/backward replay with stable parameter-gradient storage."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from torch import nn


class CapturedBackward:
  def __init__(
    self, forward: Callable[[dict[str, torch.Tensor]], tuple[torch.Tensor, ...]],
    example: dict[str, torch.Tensor], parameters: Iterable[nn.Parameter],
    *, pool: tuple[int, int] | None = None,
  ):
    self.inputs = {name: value.detach().clone() for name, value in example.items()}
    self.parameters = list(parameters)
    stream = torch.cuda.Stream(device=next(iter(example.values())).device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
      for _ in range(2):
        self.zero_grad()
        forward(self.inputs)[0].backward()
    torch.cuda.current_stream().wait_stream(stream)
    self.zero_grad()
    self.graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(self.graph, stream=stream, pool=pool):
      outputs = forward(self.inputs)
      outputs[0].backward()
    torch.cuda.current_stream().wait_stream(stream)
    self.outputs = tuple(value.detach() for value in outputs)
    self.zero_grad()

  def zero_grad(self) -> None:
    for parameter in self.parameters:
      if parameter.grad is not None:
        parameter.grad.zero_()

  def __call__(self, values: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    if values.keys() != self.inputs.keys():
      raise ValueError("CUDA graph input fields changed")
    for name, value in values.items():
      target = self.inputs[name]
      if value.shape != target.shape or value.dtype != target.dtype or value.device != target.device:
        raise ValueError(f"CUDA graph input contract changed: {name}")
      target.copy_(value)
    self.graph.replay()
    return self.outputs


class CapturedRolloutStep:
  """Checkpoint one control tick; reuse its forward/VJP graphs along the BPTT chain."""

  def __init__(
    self, forward: Callable[[dict[str, torch.Tensor]], tuple[torch.Tensor, ...]],
    example: dict[str, torch.Tensor], parameters: Iterable[nn.Parameter],
    *, pool: tuple[int, int],
  ):
    self.names = tuple(example)
    if self.names[:2] != ("state", "history"):
      raise ValueError("Rollout step inputs must begin with state and history")
    self.inputs = {name: value.detach().clone().requires_grad_(name in ("state", "history"))
                   for name, value in example.items()}
    self.parameters = tuple(parameters)
    stream = torch.cuda.Stream(device=example["state"].device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.enable_grad():
      outputs = forward(self.inputs)
      self.cotangents = tuple(torch.ones_like(value) for value in outputs)

      def differentiate():
        result = forward(self.inputs)
        return torch.autograd.grad(result, (self.inputs["state"], self.inputs["history"], *self.parameters),
                                   grad_outputs=self.cotangents, allow_unused=True)

      gradients = differentiate()
      for parameter, gradient in zip(self.parameters, gradients[2:], strict=True):
        if gradient is not None and parameter.grad is None:
          parameter.grad = torch.zeros_like(parameter)
      differentiate()
      del outputs, gradients
    torch.cuda.current_stream().wait_stream(stream)
    self.forward_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(self.forward_graph, stream=stream, pool=pool), torch.enable_grad():
      outputs = forward(self.inputs)
    self.outputs = tuple(value.detach() for value in outputs)
    del outputs
    self.backward_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(self.backward_graph, stream=stream, pool=pool), torch.enable_grad():
      gradients = differentiate()
    self.gradients = tuple(None if value is None else value.detach() for value in gradients)
    torch.cuda.current_stream().wait_stream(stream)

  def copy_inputs(self, values: tuple[torch.Tensor, ...]) -> None:
    if len(values) != len(self.names):
      raise ValueError("Rollout graph input fields changed")
    with torch.no_grad():
      for name, value in zip(self.names, values, strict=True):
        target = self.inputs[name]
        if value.shape != target.shape or value.dtype != target.dtype or value.device != target.device:
          raise ValueError(f"Rollout graph input contract changed: {name}")
        target.copy_(value)

  def __call__(self, values: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    if tuple(values) != self.names:
      raise ValueError("Rollout graph input ordering changed")
    return _RolloutStep.apply(self, *values.values(), *self.parameters)


class _RolloutStep(torch.autograd.Function):
  @staticmethod
  def forward(ctx, graph: CapturedRolloutStep, *values):
    inputs = values[:len(graph.names)]
    ctx.graph = graph
    ctx.save_for_backward(*inputs)
    ctx.set_materialize_grads(False)
    graph.copy_inputs(inputs)
    graph.forward_graph.replay()
    return tuple(value.clone() for value in graph.outputs)

  @staticmethod
  def backward(ctx, *cotangents):
    graph = ctx.graph
    graph.copy_inputs(ctx.saved_tensors)
    for target, gradient in zip(graph.cotangents, cotangents, strict=True):
      if gradient is None:
        target.zero_()
      else:
        target.copy_(gradient)
    graph.backward_graph.replay()
    gradients = tuple(None if value is None else value.clone() for value in graph.gradients)
    return (None, *gradients[:2], *((None,) * (len(graph.names) - 2)), *gradients[2:])
