"""Small differentiable wxyz quaternion primitives; no simulator dependency."""

from __future__ import annotations

import torch


def unit(q: torch.Tensor) -> torch.Tensor:
  return q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(torch.finfo(q.dtype).eps)


def conjugate(q: torch.Tensor) -> torch.Tensor:
  return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
  return torch.cat((aw * bw - (av * bv).sum(-1, keepdim=True), aw * bv + bw * av + torch.linalg.cross(av, bv, dim=-1)), dim=-1)


def rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  qv = q[..., 1:]
  uv = torch.linalg.cross(qv, v, dim=-1)
  return v + 2 * (q[..., :1] * uv + torch.linalg.cross(qv, uv, dim=-1))


def yaw(q: torch.Tensor) -> torch.Tensor:
  w, x, y, z = q.unbind(-1)
  return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))


def heading(q: torch.Tensor) -> torch.Tensor:
  angle = yaw(q) / 2
  zero = torch.zeros_like(angle)
  return torch.stack((angle.cos(), zero, zero, angle.sin()), dim=-1)


def wrap(angle: torch.Tensor) -> torch.Tensor:
  return torch.atan2(angle.sin(), angle.cos())


def orientation_error_squared(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  relative = unit(multiply(conjugate(unit(a)), unit(b)))
  sine = torch.linalg.vector_norm(relative[..., 1:], dim=-1)
  return (2 * torch.atan2(sine, relative[..., 0].abs())).square()


def rotation_6d(q: torch.Tensor) -> torch.Tensor:
  """First two matrix columns, in column-major order."""
  x = torch.zeros_like(q[..., 1:])
  x[..., 0] = 1
  y = torch.zeros_like(x)
  y[..., 1] = 1
  return torch.cat((rotate(q, x), rotate(q, y)), dim=-1)
