import math

import mujoco
import numpy as np
import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from g1_lower_rl.tasks.footstep_tracking import make_footstep_env_cfg
from g1_lower_rl.tasks.footstep_tracking.initialization import RandomizeFootstepInitialization
from g1_lower_rl.tasks.lower_body.cfg.constants import FOOT_GEOMS


@pytest.mark.parametrize("profile", ["tracking", "walk-first", "precision", "step-episode"])
@pytest.mark.parametrize("play", [False, True])
def test_mixture_is_shared_by_training_profiles_only(profile, play):
  cfg = make_footstep_env_cfg(profile, play=play)
  assert ("reset_initialization" in cfg.events) == (not play)
  if not play:
    event = cfg.events["reset_initialization"]
    assert event.func is RandomizeFootstepInitialization and event.mode == "reset"
    assert event.params == {
      "probabilities": (.5, .4, .1),
      "pelvis_height_range": (.50, .79),
      "airborne_clearance_range": (.02, .10),
    }
    names = list(cfg.events)
    assert names.index("reset_initialization") > names.index("reset_robot_joints")


@pytest.mark.parametrize("task", [
  "G1-Gloria-LowerBody-Flat", "G1-Gloria-LowerBody-Flat-GRU", "G1-Gloria-Stand", "G1-Gloria-MotionTracking",
])
def test_other_tasks_keep_original_initialization(task):
  assert "reset_initialization" not in load_env_cfg(task).events


@pytest.fixture(scope="module", params=["precision", "step-episode"])
def environment(request):
  cfg = make_footstep_env_cfg(request.param)
  cfg.scene.num_envs = 128
  cfg.seed = 42
  cfg.commands["footsteps"].compile_backend = False
  env = ManagerBasedRlEnv(cfg, device="cuda:0" if torch.cuda.is_available() else "cpu")
  yield env
  env.close()


def cpu_sole_heights(env, ids):
  model = env.sim.mj_model
  data = mujoco.MjData(model)
  geom_ids = np.array([model.geom(f"robot/{name}").id for name in FOOT_GEOMS])
  heights = []
  for qpos in env.sim.data.qpos[ids].cpu().numpy():
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    axes_z = data.geom_xmat[geom_ids].reshape(-1, 3, 3)[:, 2, 2]
    bottom = data.geom_xpos[geom_ids, 2] - np.abs(axes_z) * model.geom_size[geom_ids, 1] - model.geom_size[geom_ids, 0]
    heights.append(bottom.reshape(2, 7).min(-1))
  return np.asarray(heights)


def test_mixture_distribution_and_physical_ranges(environment):
  env = environment
  event = env.event_manager.get_term_cfg("reset_initialization").func
  counts = torch.zeros(3, device=env.device, dtype=torch.long)
  pelvis_heights = []
  air_heights = []
  for _ in range(16):
    observations, _ = env.reset()
    assert all(torch.isfinite(value).all() for value in observations.values())
    counts += torch.bincount(event.modes, minlength=3)
    crouch = event.modes == 1
    pelvis = env.scene["robot"].data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    assert (pelvis[crouch] >= .50 - 2e-6).all() and (pelvis[crouch] <= .79 + 2e-6).all()
    pelvis_heights.append(pelvis[crouch])
    selected = (event.modes != 0).nonzero().flatten()
    heights = cpu_sole_heights(env, selected)
    is_crouch = crouch[selected].cpu().numpy()
    np.testing.assert_allclose(heights[is_crouch], 0., atol=2e-6)
    assert (heights[~is_crouch] >= .02 - 2e-6).all()
    assert (heights[~is_crouch] <= .10 + 2e-6).all()
    np.testing.assert_allclose(heights[:, 0], heights[:, 1], atol=2e-6)
    air_heights.extend(heights[~is_crouch, 0])
  np.testing.assert_allclose((counts / counts.sum()).cpu().numpy(), [.5, .4, .1], atol=.035)
  pelvis = torch.cat(pelvis_heights)
  assert pelvis.min() < .505 and pelvis.max() > .785
  expected_std = .29 / math.sqrt(12)
  assert abs(pelvis.mean().item() - .645) < 3 * expected_std / math.sqrt(pelvis.numel())
  assert abs(pelvis.std().item() - expected_std) < 3 * expected_std / math.sqrt(2 * (pelvis.numel() - 1))
  assert min(air_heights) < .025 and max(air_heights) > .095


@pytest.mark.parametrize("mode,height,clearance", [
  (1, .50, .02), (1, .60, .02), (1, .79, .02), (2, .70, .02), (2, .70, .10),
])
def test_endpoints_and_partial_reset(environment, mode, height, clearance):
  env = environment
  env.reset()
  event = env.event_manager.get_term_cfg("reset_initialization").func
  selected = torch.tensor([1, 3, 7], device=env.device)
  untouched = torch.tensor([0, 2, 4], device=env.device)
  qpos, qvel, modes = env.sim.data.qpos.clone(), env.sim.data.qvel.clone(), event.modes.clone()
  probabilities = tuple(float(index == mode) for index in range(3))
  event(env, selected, probabilities, (height, height), (clearance, clearance))
  env.sim.forward()
  torch.testing.assert_close(env.sim.data.qpos[untouched], qpos[untouched], rtol=0, atol=0)
  torch.testing.assert_close(env.sim.data.qvel[untouched], qvel[untouched], rtol=0, atol=0)
  torch.testing.assert_close(event.modes[untouched], modes[untouched], rtol=0, atol=0)
  np.testing.assert_allclose(cpu_sole_heights(env, selected), 0. if mode == 1 else clearance, atol=2e-6)
  if mode == 1:
    actual = env.scene["robot"].data.root_link_pos_w[selected, 2] - env.scene.env_origins[selected, 2]
    torch.testing.assert_close(actual, torch.full_like(actual, height), atol=2e-6, rtol=0)
  assert torch.isfinite(env.sim.data.qpos).all() and torch.isfinite(env.sim.data.qvel).all()


def test_standing_and_empty_selection_do_not_change_state(environment):
  env = environment
  env.reset()
  cfg = env.event_manager.get_term_cfg("reset_initialization")
  qpos, qvel = env.sim.data.qpos.clone(), env.sim.data.qvel.clone()
  cfg.func(env, None, **{**cfg.params, "probabilities": (1., 0., 0.)})
  cfg.func(env, torch.empty(0, device=env.device, dtype=torch.long), **cfg.params)
  torch.testing.assert_close(env.sim.data.qpos[:], qpos, rtol=0, atol=0)
  torch.testing.assert_close(env.sim.data.qvel[:], qvel, rtol=0, atol=0)


def test_ik_preserves_nominal_foot_xy_and_flatness_within_joint_limits(environment, monkeypatch):
  env = environment
  robot = env.scene["robot"]
  cfg = env.event_manager.get_term_cfg("reset_initialization")
  event = cfg.func
  site_ids = robot.find_sites(("left_foot", "right_foot"), preserve_order=True)[0]
  global_sites = robot.indexing.site_ids[site_ids].cpu().numpy()
  reference = mujoco.MjData(env.sim.mj_model)
  reference.qpos[robot.indexing.joint_q_adr.cpu().numpy()] = robot.data.default_joint_pos[0].cpu().numpy()
  reference.qpos[robot.indexing.free_joint_q_adr.cpu().numpy()] = (0., 0., 0., 1., 0., 0., 0.)
  mujoco.mj_kinematics(env.sim.mj_model, reference)
  nominal_feet = reference.site_xpos[global_sites].copy()
  monkeypatch.setitem(cfg.params, "probabilities", (0., 1., 0.))
  for height in (.50, .60, .70, .79):
    monkeypatch.setitem(cfg.params, "pelvis_height_range", (height, height))
    env.reset()
    actual = robot.data.site_pos_w[:, site_ids].cpu().numpy()
    root_quaternions = robot.data.root_link_quat_w.cpu().numpy()
    origin = (env.scene.env_origins + robot.data.default_root_state[:, :3]).cpu().numpy()
    for index, quaternion in enumerate(root_quaternions):
      rotation = np.empty(9)
      mujoco.mju_quat2Mat(rotation, quaternion.astype(np.float64))
      expected = nominal_feet @ rotation.reshape(3, 3).T + origin[index]
      np.testing.assert_allclose(actual[index, :, :2], expected[:, :2], atol=2e-6)
    quaternion = robot.data.site_quat_w[:, site_ids]
    vertical = 1 - 2 * quaternion[..., 1:3].square().sum(-1)
    torch.testing.assert_close(vertical, torch.ones_like(vertical), atol=2e-6, rtol=0)
    ids = event.joint_ids[[0, 4, 6, 10]]
    angles = robot.data.joint_pos[:, ids]
    limits = robot.data.soft_joint_pos_limits[:, ids]
    assert (angles >= limits[..., 0] - 1e-6).all()
    assert (angles <= limits[..., 1] + 1e-6).all()
    leg_angles = robot.data.joint_pos[:, event.joint_ids]
    hard_limits = robot.data.joint_pos_limits[:, event.joint_ids]
    assert (leg_angles >= hard_limits[..., 0] - 1e-6).all()
    assert (leg_angles <= hard_limits[..., 1] + 1e-6).all()
    pelvis_height = robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    torch.testing.assert_close(pelvis_height, torch.full_like(pelvis_height, height), atol=2e-6, rtol=0)
    assert (robot.data.root_link_pos_w[:, :2] - torch.as_tensor(origin[:, :2], device=env.device)).norm(dim=-1).max() < .15


def test_airborne_initialization_falls_under_gravity(environment, monkeypatch):
  env = environment
  cfg = env.event_manager.get_term_cfg("reset_initialization")
  monkeypatch.setitem(cfg.params, "probabilities", (0., 0., 1.))
  monkeypatch.setitem(cfg.params, "airborne_clearance_range", (.10, .10))
  env.reset()
  robot = env.scene["robot"]
  initial_height = robot.data.root_link_pos_w[:, 2].clone()
  env.step(torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device))
  assert (robot.data.root_link_pos_w[:, 2] < initial_height).all()
  assert (robot.data.root_link_lin_vel_w[:, 2] < 0).all()


@pytest.mark.parametrize("probabilities", [(0., 0., 1.), (.5, .4, .1)])
def test_ik_only_solves_crouching_environments(environment, monkeypatch, probabilities):
  cfg = environment.event_manager.get_term_cfg("reset_initialization")
  event = cfg.func
  solve = event._solve_crouch
  counts = []

  def record(height):
    counts.append(height.numel())
    return solve(height)

  monkeypatch.setattr(event, "_solve_crouch", record)
  monkeypatch.setitem(cfg.params, "probabilities", probabilities)
  environment.reset()
  crouching = int((event.modes == 1).sum())
  assert counts == ([crouching] if crouching else [])


@pytest.mark.parametrize("params", [
  {"probabilities": (.5, .5, .5)},
  {"probabilities": (-.1, .5, .6)},
  {"probabilities": (float("nan"), .4, .1)},
  {"pelvis_height_range": (0., .8)},
  {"pelvis_height_range": (.8, .6)},
  {"pelvis_height_range": (.6, float("inf"))},
  {"airborne_clearance_range": (0., .1)},
  {"airborne_clearance_range": (.1, .02)},
  {"airborne_clearance_range": (.02, float("inf"))},
])
def test_invalid_parameters_are_rejected(environment, params):
  env = environment
  cfg = env.event_manager.get_term_cfg("reset_initialization")
  with pytest.raises(ValueError):
    cfg.func(env, None, **{**cfg.params, **params})


def test_unreachable_pelvis_height_is_rejected(environment):
  env = environment
  cfg = env.event_manager.get_term_cfg("reset_initialization")
  with pytest.raises(ValueError, match="unreachable"):
    cfg.func(env, None, **{**cfg.params, "probabilities": (0., 1., 0.), "pelvis_height_range": (1., 1.)})


def test_mixed_initialization_advances_physics_without_nonfinite_state(environment):
  env = environment
  env.reset()
  for _ in range(12):
    observations, rewards, _, _, _ = env.step(torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device))
    assert all(torch.isfinite(value).all() for value in observations.values())
    assert torch.isfinite(rewards).all()
    assert torch.isfinite(env.sim.data.qpos).all() and torch.isfinite(env.sim.data.qvel).all()
