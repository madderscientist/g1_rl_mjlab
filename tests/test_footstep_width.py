import math
from html.parser import HTMLParser

import mujoco
import numpy as np
import pytest

from g1_lower_rl.assets.g1_gloria import get_spec
from g1_lower_rl.footsteps import FootstepManager, FootstepSampler
from g1_lower_rl.footsteps.config import DEFAULT_FOOT_WIDTH, FootstepManagerCfg, FootstepSamplerCfg
from g1_lower_rl.footsteps.sampler import to_local
from g1_lower_rl.footsteps.webui import ASSETS, PreviewSession


def zero_joint_foot_spacing():
  model = get_spec().compile()
  data = mujoco.MjData(model)
  data.qpos[:] = model.qpos0
  for joint_index in range(model.njnt):
    if model.jnt_type[joint_index] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
      data.qpos[model.jnt_qposadr[joint_index]] = 0.0
  mujoco.mj_forward(model, data)
  positions = data.site_xpos[[model.site(name).id for name in ("left_foot", "right_foot")]]
  return abs(float(positions[0, 1] - positions[1, 1]))


def test_default_width_center_covers_zero_joint_foot_spacing():
  spacing = zero_joint_foot_spacing()
  assert spacing == pytest.approx(0.23701291)
  assert spacing <= DEFAULT_FOOT_WIDTH < spacing + 0.01
  assert FootstepSamplerCfg().min_width == 0.12
  assert FootstepSamplerCfg().width_center == DEFAULT_FOOT_WIDTH
  assert FootstepSamplerCfg().distance_range == (0.05, 0.48)
  assert FootstepSamplerCfg().max_width == 0.36
  assert FootstepManagerCfg().hold_width == DEFAULT_FOOT_WIDTH


@pytest.mark.parametrize("relative_direction", [0.0, math.pi, math.pi / 2, -math.pi / 2])
def test_sampled_width_is_centered_and_retains_geometric_bounds(relative_direction):
  sampler = FootstepSampler(seed=42)
  heading = math.pi / 3
  sampler.set_direction(heading + relative_direction, heading)
  previous = np.array([3.0, -2.0, heading])
  widths = []
  for step in range(2000):
    side = step % 2
    target = sampler.sample(previous, side)
    local = to_local(target, previous)
    width = local[1] * (1 if side == 0 else -1)
    assert sampler.cfg.min_width - 1e-10 <= width <= sampler.cfg.max_width + 1e-10
    assert np.linalg.norm(local[:2]) <= sampler.cfg.distance_range[1] + 1e-10
    assert abs(local[2]) <= sampler.cfg.max_yaw_change + 1e-10
    widths.append(width)
    previous = target
  assert np.mean(widths) == pytest.approx(0.24, abs=0.004)
  assert min(widths) < 0.20 and max(widths) > 0.28


def test_stopping_uses_wider_hold_spacing_without_moving_committed_targets():
  manager = FootstepManager(seed=42)
  manager.reset(np.array([[0.0, 0.12, 0.0], [0.0, -0.12, 0.0]]))
  committed = manager.command().footsteps_w.copy()
  last = manager.queue[-1]
  manager.request_stop()
  np.testing.assert_array_equal(manager.command().footsteps_w, committed)
  closing_side = 1 - last.side
  closing = to_local(manager.terminal_feet[closing_side], last.pose_w)
  np.testing.assert_allclose(closing, [0.0, (1 if closing_side == 0 else -1) * 0.24, 0.0], atol=1e-12)


def test_preview_form_and_server_use_shared_width_defaults():
  class InputDefaults(HTMLParser):
    def __init__(self):
      super().__init__()
      self.values = {}

    def handle_starttag(self, tag, attrs):
      attributes = dict(attrs)
      if tag == "input" and attributes.get("name") in ("min_width", "hold_width", "distance_max", "stop_probability"):
        self.values[attributes["name"]] = float(attributes["value"])

  parser = InputDefaults()
  parser.feed((ASSETS / "index.html").read_text())
  assert parser.values == {"min_width": 0.12, "hold_width": 0.24, "distance_max": 0.48, "stop_probability": 0.30}
  for options in ({}, parser.values):
    session = PreviewSession(options)
    assert session.manager.cfg.sampler.min_width == 0.12
    assert session.manager.cfg.sampler.width_center == DEFAULT_FOOT_WIDTH
    assert session.manager.cfg.hold_width == DEFAULT_FOOT_WIDTH
    assert session.manager.cfg.sampler.distance_range[1] == 0.48
    assert session.source.cfg.stop_probability == 0.30
    np.testing.assert_allclose(session.manager.supports[:, 1], [0.12, -0.12])


@pytest.mark.parametrize("direction", [0.0, math.pi])
def test_total_step_distance_can_reach_48cm_with_24cm_width(direction):
  from dataclasses import replace

  cfg = replace(FootstepSamplerCfg(), distance_range=(0.48, 0.48), direction_noise=(0.0, 0.0), yaw_noise=(0.0, 0.0))
  sampler = FootstepSampler(cfg, seed=42)
  sampler.set_direction(direction, 0.0)
  for side in (0, 1):
    target = sampler.sample([0.0, 0.0, 0.0], side)
    assert np.linalg.norm(target[:2]) == pytest.approx(0.48)
    assert abs(target[1]) == pytest.approx(0.24)
    assert abs(target[0]) == pytest.approx(math.sqrt(0.48**2 - 0.24**2))


@pytest.mark.parametrize("direction", [math.pi / 2, -math.pi / 2])
def test_centered_width_keeps_commanded_sideways_progress(direction):
  from dataclasses import replace

  cfg = replace(FootstepSamplerCfg(), direction_noise=(0.0, 0.0), yaw_noise=(0.0, 0.0))
  sampler = FootstepSampler(cfg, seed=42)
  sampler.set_direction(direction, 0.0)
  initial = np.array([0.0, -0.12, 0.0])
  previous = initial.copy()
  widths = []
  for step in range(2000):
    target = sampler.sample(previous, step % 2)
    widths.append(abs(to_local(target, previous)[1]))
    previous = target
  assert np.mean(widths) == pytest.approx(0.24, abs=0.003)
  assert (previous[1] - initial[1]) * math.sin(direction) > 100.0


def test_opposite_feet_keep_width_mean_for_matched_samples():
  for direction in np.linspace(-math.pi, math.pi, 13):
    left = FootstepSampler(seed=42)
    right = FootstepSampler(seed=42)
    left.set_direction(direction, 0.0)
    right.set_direction(direction, 0.0)
    for _ in range(100):
      left_pose = left.sample([0.0, 0.0, 0.0], 0)
      right_pose = right.sample([0.0, 0.0, 0.0], 1)
      assert (left_pose[1] - right_pose[1]) / 2 == pytest.approx(0.24)