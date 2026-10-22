"""Mixed standing, crouching and airborne episode initialization."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import mujoco
import mujoco_warp as mjwarp
import torch
import warp as wp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.utils.lab_api.math import quat_apply, sample_uniform, yaw_quat

from g1_lower_rl.tasks.lower_body.cfg.constants import FOOT_GEOMS

from .leg_symmetry import LEG_JOINTS

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class RandomizeFootstepInitialization:
  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.robot = env.scene["robot"]
    self.joint_ids = torch.tensor(
      self.robot.find_joints(LEG_JOINTS, preserve_order=True)[0], device=env.device, dtype=torch.long,
    )
    local_geom_ids = self.robot.find_geoms(FOOT_GEOMS, preserve_order=True)[0]
    self.geom_ids = self.robot.indexing.geom_ids[local_geom_ids].long()
    indices = self.geom_ids.cpu().numpy()
    model = env.sim.mj_model
    if not (model.geom_type[indices] == mujoco.mjtGeom.mjGEOM_CAPSULE).all():
      raise ValueError("Footstep initialization requires capsule foot collision geometry")
    sizes = torch.as_tensor(model.geom_size[indices], device=env.device, dtype=torch.float32)
    self.radii = sizes[:, 0]
    self.half_lengths = sizes[:, 1]
    self.modes = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
    data = mujoco.MjData(model)
    q_ids = self.robot.indexing.joint_q_adr.cpu().numpy()
    root_ids = self.robot.indexing.free_joint_q_adr.cpu().numpy()
    data.qpos[q_ids] = self.robot.data.default_joint_pos[0].cpu().numpy()
    data.qpos[root_ids] = (0., 0., 0., 1., 0., 0., 0.)
    site_ids = self.robot.indexing.site_ids[self.robot.find_sites(("left_foot", "right_foot"), preserve_order=True)[0]]
    mujoco.mj_kinematics(model, data)
    self.home_feet_x = float(data.site_xpos[int(site_ids[0]), 0])
    data.qpos[q_ids[self.joint_ids.cpu().numpy()]] = 0.
    mujoco.mj_kinematics(model, data)
    joints = self.robot.indexing.joint_ids[self.joint_ids].cpu().numpy()
    hip, knee, ankle = (data.xanchor[joints[index]].copy() for index in (0, 3, 4))
    self.upper_x, self.upper_z = float(knee[0] - hip[0]), float(knee[2] - hip[2])
    self.lower_x, self.lower_z = float(ankle[0] - knee[0]), float(ankle[2] - knee[2])
    self.hip_x, self.hip_z = float(hip[0]), float(hip[2])
    self.foot_offset_x = float(data.site_xpos[int(site_ids[0]), 0] - ankle[0])
    capsule_axes_z = data.geom_xmat[indices].reshape(-1, 3, 3)[:, 2, 2]
    sole_bottom = (data.geom_xpos[indices, 2]
      - abs(capsule_axes_z) * model.geom_size[indices, 1] - model.geom_size[indices, 0]).min()
    self.foot_offset_z = float(sole_bottom - ankle[2])
    self.upper_length = math.hypot(self.upper_x, self.upper_z)
    self.lower_length = math.hypot(self.lower_x, self.lower_z)
    self.knee_phase = math.atan2(
      self.upper_x * self.lower_z - self.upper_z * self.lower_x,
      self.upper_x * self.lower_x + self.upper_z * self.lower_z,
    )
    limits = self.robot.data.soft_joint_pos_limits[0, self.joint_ids]
    self.hip_lower = limits[[0, 6], 0].max()
    self.hip_upper = limits[[0, 6], 1].min()
    self.ankle_lower = limits[[4, 10], 0].max()
    self.ankle_upper = limits[[4, 10], 1].min()
    self.knee_upper = self.robot.data.joint_pos_limits[0, self.joint_ids[[3, 9]], 1].min()

  def _solve_crouch(self, height: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    target_x = self.home_feet_x - self.hip_x - self.foot_offset_x
    target_z = -height - self.hip_z - self.foot_offset_z
    cosine = (target_x**2 + target_z.square() - self.upper_length**2 - self.lower_length**2
      ) / (2 * self.upper_length * self.lower_length)
    if (cosine.abs() > 1 + 1e-6).any():
      raise ValueError("Requested pelvis height is unreachable with fixed feet")
    knee = self.knee_phase + torch.acos(cosine.clamp(-1., 1.))
    x = self.upper_x + knee.cos() * self.lower_x + knee.sin() * self.lower_z
    z = self.upper_z - knee.sin() * self.lower_x + knee.cos() * self.lower_z
    hip = torch.atan2(x, -z) - torch.atan2(torch.full_like(height, target_x), -target_z)
    ankle = -hip - knee
    bounded_ankle = ankle.clamp(min=self.ankle_lower, max=self.ankle_upper)
    limited = ankle != bounded_ankle
    lower_z = bounded_ankle.sin() * self.lower_x + bounded_ankle.cos() * self.lower_z
    upper_cosine = -(target_z - lower_z) / self.upper_length
    if (limited & (upper_cosine.abs() > 1 + 1e-6)).any():
      raise ValueError("Requested pelvis height is unreachable within ankle limits")
    upper_angle = math.atan2(self.upper_x, -self.upper_z)
    angle = torch.acos(upper_cosine.clamp(-1., 1.))
    bounded_hip = upper_angle + torch.where(hip < upper_angle, -angle, angle)
    hip = torch.where(limited, bounded_hip, hip)
    ankle = bounded_ankle
    knee = -hip - ankle
    if ((hip < self.hip_lower - 1e-6) | (hip > self.hip_upper + 1e-6)
        | (knee < 0) | (knee > self.knee_upper)).any():
      raise ValueError("Requested pelvis height is unreachable within hip/knee limits")
    x = self.upper_x + knee.cos() * self.lower_x + knee.sin() * self.lower_z
    z = self.upper_z - knee.sin() * self.lower_x + knee.cos() * self.lower_z
    # At an ankle limit, solve height exactly and move the pelvis horizontally to keep both feet fixed.
    pelvis_x = target_x - (hip.cos() * x + hip.sin() * z)
    return hip, knee, ankle, pelvis_x

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    probabilities: tuple[float, float, float],
    pelvis_height_range: tuple[float, float],
    airborne_clearance_range: tuple[float, float],
  ) -> None:
    if (len(probabilities) != 3 or any(not math.isfinite(p) or p < 0 for p in probabilities)
        or not math.isclose(sum(probabilities), 1., abs_tol=1e-8)):
      raise ValueError("Standing, crouching and airborne probabilities must be nonnegative and sum to one")
    lower, upper = pelvis_height_range
    if not (math.isfinite(lower) and math.isfinite(upper) and 0 < lower <= upper):
      raise ValueError("Pelvis height must be finite, positive and ordered")
    lower, upper = airborne_clearance_range
    if not (math.isfinite(lower) and math.isfinite(upper) and 0 < lower <= upper):
      raise ValueError("Airborne sole clearance must be finite, positive and ordered")
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids.to(env.device)
    if ids.numel() == 0:
      return
    draw = torch.rand(len(ids), device=env.device)
    modes = (draw >= probabilities[0]).long() + (draw >= sum(probabilities[:2])).long()
    self.modes[ids] = modes
    selected = ids[modes != 0]
    if selected.numel() == 0:
      return
    crouching = modes[modes != 0] == 1
    crouch_ids = selected[crouching]
    joint_pos = self.robot.data.default_joint_pos[selected[:, None], self.joint_ids]
    height = sample_uniform(*pelvis_height_range, (len(selected),), device=env.device)
    translation = torch.zeros((len(selected), 3), device=env.device)
    if crouch_ids.numel():
      hip, knee, ankle, pelvis_x = self._solve_crouch(height[crouching])
      legs = joint_pos.view(-1, 2, 6)
      legs[crouching, :, 0] = hip[:, None]
      legs[crouching, :, 3] = knee[:, None]
      legs[crouching, :, 4] = ankle[:, None]
      translation[crouching, 0] = pelvis_x
    self.robot.write_joint_state_to_sim(
      joint_pos, torch.zeros_like(joint_pos), joint_ids=self.joint_ids, env_ids=selected,
    )
    pose = env.sim.data.qpos[selected[:, None], self.robot.indexing.free_joint_q_adr].clone()
    pose[:, 3:7] = yaw_quat(pose[:, 3:7])
    pose[:, :3] += quat_apply(pose[:, 3:7], translation)
    pose[crouching, 2] = env.scene.env_origins[crouch_ids, 2] + height[crouching]
    self.robot.write_root_link_pose_to_sim(pose, env_ids=selected)
    self.robot.write_root_link_velocity_to_sim(torch.zeros((len(selected), 6), device=env.device), env_ids=selected)
    # Only refresh transforms: do not step physics or run the contact solver at the temporary root height.
    with wp.ScopedDevice(env.sim.wp_device):
      mjwarp.kinematics(env.sim.wp_model, env.sim.wp_data)
    grid = selected[:, None]
    positions = env.sim.data.geom_xpos[grid, self.geom_ids, 2]
    axes_z = env.sim.data.geom_xmat[grid, self.geom_ids, 2, 2]
    bottom = (positions - axes_z.abs() * self.half_lengths - self.radii).amin(-1)
    clearance = sample_uniform(*airborne_clearance_range, (len(selected),), device=env.device)
    clearance = torch.where(crouching, 0., clearance)
    pose[:, 2] += env.scene.env_origins[selected, 2] + clearance - bottom
    self.robot.write_root_link_pose_to_sim(pose, env_ids=selected)
