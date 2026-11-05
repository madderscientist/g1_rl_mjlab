"""Differentiable link/site kinematics from a compiled MuJoCo static template."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import nn

from g1_lower_rl.world_model.model import BODY_NAMES, JOINT_NAMES, STATE_DIM, STATE_FIELDS

from .math import multiply, rotate, unit

FOOT_NAMES = ("left_foot", "right_foot")


@dataclass(frozen=True)
class StaticKinematicTemplate:
  """JSON-safe MJCF constants in canonical body/joint order, independent of MuJoCo."""

  body_names: tuple[str, ...]
  joint_names: tuple[str, ...]
  parents: tuple[int, ...]
  body_position: tuple[tuple[float, ...], ...]
  body_quaternion: tuple[tuple[float, ...], ...]
  body_joint: tuple[int, ...]
  joint_position: tuple[tuple[float, ...], ...]
  joint_axis: tuple[tuple[float, ...], ...]
  joint_reference: tuple[float, ...]
  foot_body: tuple[int, ...]
  foot_position: tuple[tuple[float, ...], ...]
  foot_quaternion: tuple[tuple[float, ...], ...]
  body_com_position: tuple[tuple[float, ...], ...]
  gravity: tuple[float, ...]
  source_prefix: str = ""
  source_body_ids: tuple[int, ...] = ()
  source_joint_ids: tuple[int, ...] = ()
  source_site_ids: tuple[int, ...] = ()
  version: int = 1

  def __post_init__(self) -> None:
    if self.version != 1 or self.body_names != BODY_NAMES or self.joint_names != JOINT_NAMES:
      raise ValueError("Unsupported template version or noncanonical names")
    n = len(BODY_NAMES)
    for name, width in (("body_position", 3), ("body_quaternion", 4), ("body_com_position", 3)):
      if torch.as_tensor(getattr(self, name)).shape != (n, width):
        raise ValueError(f"Invalid {name} shape")
    for name, width in (("joint_position", 3), ("joint_axis", 3)):
      if torch.as_tensor(getattr(self, name)).shape != (len(JOINT_NAMES), width):
        raise ValueError(f"Invalid {name} shape")
    if len(self.parents) != n or self.parents[0] != -1 or any(not 0 <= p < i for i, p in enumerate(self.parents[1:], 1)):
      raise ValueError("Body parents must be in canonical topological order")
    if len(self.body_joint) != n or self.body_joint[0] != -1:
      raise ValueError("Invalid body-joint mapping")
    if any(j < -1 for j in self.body_joint):
      raise ValueError("Invalid body-joint index")
    if sorted(j for j in self.body_joint if j >= 0) != list(range(len(JOINT_NAMES))):
      raise ValueError("Each canonical hinge must occur exactly once")
    if len(self.joint_reference) != len(JOINT_NAMES):
      raise ValueError("Invalid joint references")
    if len(self.gravity) != 3:
      raise ValueError("Invalid gravity")
    if len(self.foot_body) != 2 or any(not 0 <= b < n for b in self.foot_body):
      raise ValueError("Invalid foot-body mapping")
    for name, width in (("foot_position", 3), ("foot_quaternion", 4)):
      if torch.as_tensor(getattr(self, name)).shape != (2, width):
        raise ValueError(f"Invalid {name} shape")
    for name in (
      "body_position",
      "body_quaternion",
      "joint_position",
      "joint_axis",
      "joint_reference",
      "foot_position",
      "foot_quaternion",
      "body_com_position",
      "gravity",
    ):
      if not torch.isfinite(torch.as_tensor(getattr(self, name))).all():
        raise ValueError(f"Nonfinite {name}")
    for name in ("body_quaternion", "foot_quaternion", "joint_axis"):
      norms = torch.linalg.vector_norm(torch.as_tensor(getattr(self, name)), dim=-1)
      if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5):
        raise ValueError(f"{name} must be normalized")

  @classmethod
  def from_mujoco(cls, model: Any, name_prefix: str | None = None) -> StaticKinematicTemplate:
    """Extract once from the *actual compiled* teacher model (including frame edits)."""
    mujoco: Any = import_module("mujoco")

    if name_prefix is None:
      name_prefix = "" if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis") >= 0 else "robot/"
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name_prefix + name) for name in BODY_NAMES]
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name_prefix + name) for name in JOINT_NAMES]
    site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name_prefix + name) for name in FOOT_NAMES]
    if min(body_ids + joint_ids + site_ids) < 0:
      raise ValueError("Compiled model is missing canonical bodies, joints or foot sites")
    free_id = int(model.body_jntadr[body_ids[0]])
    if (
      model.body_parentid[body_ids[0]] != 0
      or model.body_jntnum[body_ids[0]] != 1
      or model.jnt_type[free_id] != mujoco.mjtJoint.mjJNT_FREE
    ):
      raise ValueError("Pelvis must be a world-parented free body")
    if any(model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE for j in joint_ids):
      raise ValueError("Only the canonical scalar hinge joints are supported")
    body_map, joint_map = {b: i for i, b in enumerate(body_ids)}, {j: i for i, j in enumerate(joint_ids)}
    parents, body_joints = [-1], [-1]
    for b in body_ids[1:]:
      if int(model.body_parentid[b]) not in body_map or model.body_jntnum[b] > 1:
        raise ValueError("Unsupported noncanonical ancestor or multi-joint body")
      parents.append(body_map[int(model.body_parentid[b])])
      j = int(model.body_jntadr[b])
      if model.body_jntnum[b] and j not in joint_map:
        raise ValueError("Unsupported extra body joint")
      body_joints.append(joint_map[j] if model.body_jntnum[b] else -1)
    rows = lambda array, ids: tuple(tuple(float(v) for v in array[i]) for i in ids)
    return cls(
      BODY_NAMES,
      JOINT_NAMES,
      tuple(parents),
      rows(model.body_pos, body_ids),
      rows(model.body_quat, body_ids),
      tuple(body_joints),
      rows(model.jnt_pos, joint_ids),
      rows(model.jnt_axis, joint_ids),
      tuple(float(model.qpos0[model.jnt_qposadr[j]]) for j in joint_ids),
      tuple(body_map[int(model.site_bodyid[s])] for s in site_ids),
      rows(model.site_pos, site_ids),
      rows(model.site_quat, site_ids),
      rows(model.body_ipos, body_ids),
      tuple(float(x) for x in model.opt.gravity),
      name_prefix,
      tuple(body_ids),
      tuple(joint_ids),
      tuple(site_ids),
    )

  @classmethod
  def from_xml(cls, path: str | Path | None = None) -> StaticKinematicTemplate:
    """Compile an MJCF once; prefer from_mujoco for teacher-specific compiled edits."""
    mujoco: Any = import_module("mujoco")

    if path is None:
      path = Path(__file__).resolve().parents[2] / "assets/g1_gloria/xmls/scene_g1_with_dual_gloria_m.xml"
    return cls.from_mujoco(mujoco.MjModel.from_xml_path(str(path)))

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> StaticKinematicTemplate:
    def tuples(value: Any) -> Any:
      return tuple(tuples(v) for v in value) if isinstance(value, (list, tuple)) else value

    return cls(**{key: tuples(value) for key, value in data.items()})

  def save(self, path: str | Path) -> None:
    Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

  @classmethod
  def load(cls, path: str | Path) -> StaticKinematicTemplate:
    return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class KinematicState(NamedTuple):
  body_position: torch.Tensor
  body_quaternion: torch.Tensor
  body_linear_velocity: torch.Tensor
  body_angular_velocity: torch.Tensor
  foot_position: torch.Tensor
  foot_quaternion: torch.Tensor
  foot_linear_velocity: torch.Tensor
  foot_angular_velocity: torch.Tensor


class TorchKinematics(nn.Module):
  """All output positions and velocities are world-frame *link/site origins*, not COMs."""

  joint_axis: torch.Tensor
  joint_position: torch.Tensor

  def __init__(self, template: StaticKinematicTemplate):
    super().__init__()
    self.template = template
    self.level_batching = False
    depths = [0]
    for parent in template.parents[1:]:
      depths.append(depths[parent] + 1)
    self.levels = tuple(
      tuple(i for i, depth in enumerate(depths) if depth == level) for level in range(1, max(depths) + 1)
    )
    for level, ids in enumerate(self.levels):
      joints = [template.body_joint[i] for i in ids]
      self.register_buffer(f"level_bodies_{level}", torch.tensor(ids), persistent=False)
      self.register_buffer(f"level_joints_{level}", torch.tensor([max(j, 0) for j in joints]), persistent=False)
      self.register_buffer(
        f"level_active_{level}", torch.tensor([j >= 0 for j in joints], dtype=torch.float64), persistent=False,
      )
    for name in (
      "body_position",
      "body_quaternion",
      "joint_position",
      "joint_axis",
      "joint_reference",
      "foot_position",
      "foot_quaternion",
    ):
      self.register_buffer(name, torch.tensor(getattr(template, name), dtype=torch.float64))

  def forward(self, state: torch.Tensor, *, validate: bool = True) -> KinematicState:
    if validate and (state.ndim < 1 or state.shape[-1] != STATE_DIM or not state.is_floating_point()):
      raise ValueError(f"state must be floating point [..., {STATE_DIM}]")
    if validate:
      finite = torch.isfinite(state).all()
      nonzero = (torch.linalg.vector_norm(state[..., 3:7], dim=-1) > torch.finfo(state.dtype).eps).all()
      if state.is_cuda and torch.cuda.is_current_stream_capturing():
        torch._assert_async(finite, "state must be finite")
        torch._assert_async(nonzero, "Root quaternion must be nonzero")
      else:
        if not finite:
          raise ValueError("state must be finite")
        if not nonzero:
          raise ValueError("Root quaternion must be nonzero")
    constants = {
      name: getattr(self, name).to(state)
      for name in (
        "body_position",
        "body_quaternion",
        "joint_position",
        "joint_axis",
        "joint_reference",
        "foot_position",
        "foot_quaternion",
      )
    }
    if self.level_batching:
      return self._forward_levels(state, constants)
    p, q, v = [state[..., :3]], [unit(state[..., 3:7])], [state[..., STATE_FIELDS["root_linear_velocity"]]]
    w = [rotate(q[0], state[..., STATE_FIELDS["root_angular_velocity"]])]
    for i, parent in enumerate(self.template.parents[1:], 1):
      local_p = constants["body_position"][i].expand_as(p[parent])
      local_q = constants["body_quaternion"][i].expand_as(q[parent])
      base_q = multiply(q[parent], local_q)
      offset = rotate(q[parent], local_p)
      base_p = p[parent] + offset
      base_v = v[parent] + torch.linalg.cross(w[parent], offset, dim=-1)
      j = self.template.body_joint[i]
      if j < 0:
        p.append(base_p)
        q.append(base_q)
        v.append(base_v)
        w.append(w[parent])
        continue
      angle = (state[..., 7 + j] - constants["joint_reference"][j]) / 2
      axis = constants["joint_axis"][j].expand_as(base_p)
      joint_q = torch.cat((angle.cos().unsqueeze(-1), angle.sin().unsqueeze(-1) * axis), dim=-1)
      anchor = constants["joint_position"][j].expand_as(base_p)
      displacement = rotate(base_q, anchor - rotate(joint_q, anchor))
      position = base_p + displacement
      angular = rotate(base_q, axis) * state[..., STATE_FIELDS["joint_velocity"]][..., j : j + 1]
      anchor_w = base_p + rotate(base_q, anchor)
      p.append(position)
      q.append(multiply(base_q, joint_q))
      v.append(
        base_v + torch.linalg.cross(w[parent], displacement, dim=-1) + torch.linalg.cross(angular, position - anchor_w, dim=-1)
      )
      w.append(w[parent] + angular)
    fp, fq, fv, fw = [], [], [], []
    for s, b in enumerate(self.template.foot_body):
      offset = rotate(q[b], constants["foot_position"][s].expand_as(p[b]))
      fp.append(p[b] + offset)
      fq.append(multiply(q[b], constants["foot_quaternion"][s].expand_as(q[b])))
      fv.append(v[b] + torch.linalg.cross(w[b], offset, dim=-1))
      fw.append(w[b])
    return KinematicState(*(torch.stack(values, dim=-2) for values in (p, q, v, w, fp, fq, fv, fw)))

  def _forward_levels(self, state: torch.Tensor, constants: dict[str, torch.Tensor]) -> KinematicState:
    count = len(self.template.parents)
    p = [state[..., :3]] * count
    q = [unit(state[..., 3:7])] * count
    v = [state[..., STATE_FIELDS["root_linear_velocity"]]] * count
    w = [rotate(q[0], state[..., STATE_FIELDS["root_angular_velocity"]])] * count
    for level, bodies in enumerate(self.levels):
      parents = [self.template.parents[i] for i in bodies]
      pp, qp, vp, wp = (
        torch.stack([values[i] for i in parents], dim=-2) for values in (p, q, v, w)
      )
      ids = getattr(self, f"level_bodies_{level}").to(device=state.device)
      joints = getattr(self, f"level_joints_{level}").to(device=state.device)
      active = getattr(self, f"level_active_{level}").to(state)
      base_q = multiply(qp, constants["body_quaternion"][ids].expand_as(qp))
      offset = rotate(qp, constants["body_position"][ids].expand_as(pp))
      base_p = pp + offset
      base_v = vp + torch.linalg.cross(wp, offset, dim=-1)
      angle = (state[..., 7 + joints] - constants["joint_reference"][joints]) * active / 2
      axis = constants["joint_axis"][joints].expand_as(pp)
      joint_q = torch.cat((angle.cos().unsqueeze(-1), angle.sin().unsqueeze(-1) * axis), dim=-1)
      anchor = constants["joint_position"][joints].expand_as(pp)
      displacement = rotate(base_q, anchor - rotate(joint_q, anchor))
      position = base_p + displacement
      angular = rotate(base_q, axis) * (state[..., STATE_FIELDS["joint_velocity"]][..., joints] * active).unsqueeze(-1)
      anchor_w = base_p + rotate(base_q, anchor)
      velocity = (
        base_v + torch.linalg.cross(wp, displacement, dim=-1)
        + torch.linalg.cross(angular, position - anchor_w, dim=-1)
      )
      results = (position, multiply(base_q, joint_q), velocity, wp + angular)
      for destination, values in zip((p, q, v, w), results, strict=True):
        for i, value in zip(bodies, values.unbind(-2), strict=True):
          destination[i] = value
    fp, fq, fv, fw = [], [], [], []
    for s, b in enumerate(self.template.foot_body):
      offset = rotate(q[b], constants["foot_position"][s].expand_as(p[b]))
      fp.append(p[b] + offset)
      fq.append(multiply(q[b], constants["foot_quaternion"][s].expand_as(q[b])))
      fv.append(v[b] + torch.linalg.cross(w[b], offset, dim=-1))
      fw.append(w[b])
    return KinematicState(*(torch.stack(values, dim=-2) for values in (p, q, v, w, fp, fq, fv, fw)))

  def get_extra_state(self) -> dict[str, Any]:
    return self.template.to_dict()

  def set_extra_state(self, state: dict[str, Any]) -> None:
    if StaticKinematicTemplate.from_dict(state) != self.template:
      raise RuntimeError("Kinematic checkpoint static template does not match this adapter")
