import pickle
from types import SimpleNamespace

import numpy as np
import torch

from g1_lower_rl.footsteps import FootstepManager
from g1_lower_rl.footsteps.footprint_geometry import load_footprint_geometry
from g1_lower_rl.tasks.footstep_tracking.commands import FootstepCommand
from g1_lower_rl.tasks.footstep_tracking.env_cfg import footstep_env_cfg


class RecordingVisualizer:
  def __init__(self, indices):
    self.indices = indices
    self.arrows = []
    self.cylinders = []
    self.spheres = []

  def get_env_indices(self, num_envs):
    return self.indices

  def add_arrow(self, start, end, **kwargs):
    self.arrows.append((start, end, kwargs))

  def add_cylinder(self, start, end, **kwargs):
    self.cylinders.append((start, end, kwargs))

  def add_sphere(self, center, **kwargs):
    self.spheres.append((center, kwargs))


def test_world_footprints_match_commands_without_changing_state():
  manager = FootstepManager(seed=42)
  manager.reset(np.array(((3.0, 4.11, 0.0), (3.0, 3.89, 0.0))))
  manager.queue[0].pose_w[2] = np.pi / 2
  before = pickle.dumps(manager.state_dict())
  term = SimpleNamespace(
    num_envs=2,
    pending_reset=np.array((True, False)),
    managers=[None, manager],
    _footprint_geometry=None,
    reward_state=SimpleNamespace(
      ground_height=torch.full((2, 2), 0.3),
      targets_w=torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[4.0, 5.0, 0.4], [6.0, 7.0, -0.7]]]),
    ),
  )
  visualizer = RecordingVisualizer([1])
  FootstepCommand._debug_vis_impl(term, visualizer)
  assert pickle.dumps(manager.state_dict()) == before
  assert len(visualizer.arrows) == 6
  assert len(visualizer.cylinders) == 50
  assert len(visualizer.spheres) == 84
  for slot, (origin, end, kwargs) in enumerate(visualizer.arrows[:4]):
    pose = manager.command().footsteps_w[slot]
    np.testing.assert_allclose(origin[:2], pose[:2])
    np.testing.assert_allclose(origin[2], 0.35)
    np.testing.assert_allclose(end - origin, (0.16 * np.cos(pose[2]), 0.16 * np.sin(pose[2]), 0.0), atol=1e-12)
    assert kwargs["label"] == f"footstep_1_{('L1', 'L2', 'R1', 'R2')[slot]}"
  assert visualizer.arrows[0][2]["color"][2] > visualizer.arrows[0][2]["color"][0]
  assert visualizer.arrows[2][2]["color"][0] > visualizer.arrows[2][2]["color"][2]
  for side, (origin, end, kwargs) in enumerate(visualizer.arrows[4:]):
    pose = term.reward_state.targets_w[1, side].numpy()
    np.testing.assert_allclose(origin[:2], pose[:2])
    np.testing.assert_allclose(origin[2], 0.36)
    np.testing.assert_allclose(end - origin, (0.16 * np.cos(pose[2]), 0.16 * np.sin(pose[2]), 0.0), atol=1e-8)
    assert kwargs["label"] == f"footstep_1_{'L' if side == 0 else 'R'}_current"
    assert kwargs["width"] > visualizer.arrows[side * 2][2]["width"]
  borders = [entry for entry in visualizer.cylinders if "_border_" in entry[2]["label"]]
  assert len(borders) == 8
  for side in (0, 1):
    edges = borders[side * 4:side * 4 + 4]
    for edge, (_, end, _) in enumerate(edges):
      np.testing.assert_allclose(end, edges[(edge + 1) % 4][0])
    center = np.mean([entry[0][:2] for entry in edges], axis=0)
    pose = term.reward_state.targets_w[1, side].numpy()
    lower, upper = np.asarray(load_footprint_geometry()["feet"][side]["bounds"])
    rotation = np.array(((np.cos(pose[2]), -np.sin(pose[2])), (np.sin(pose[2]), np.cos(pose[2]))))
    np.testing.assert_allclose(center, (lower + upper) / 2 @ rotation.T + pose[:2], atol=1e-8)
  capsule = load_footprint_geometry()["feet"][0]["capsules"][0]
  pose = manager.command().footsteps_w[0]
  for point, key in zip(visualizer.cylinders[0][:2], ("start", "end")):
    offset = capsule[key]
    np.testing.assert_allclose(point[:2], pose[:2] + np.array((-offset[1], offset[0])))
    np.testing.assert_allclose(point[2], 0.315)
  empty = RecordingVisualizer([0])
  FootstepCommand._debug_vis_impl(term, empty)
  assert not empty.arrows and not empty.cylinders and not empty.spheres


def test_footprints_enabled_only_for_play():
  assert footstep_env_cfg(play=True).commands["footsteps"].debug_vis
  assert not footstep_env_cfg().commands["footsteps"].debug_vis


def test_current_target_survives_queue_rollover_and_uses_completed_snapshot():
  manager = FootstepManager(seed=42)
  manager.reset(np.array(((0.0, 0.11, 0.0), (0.0, -0.11, 0.0))))
  for _ in range(100):
    old_ids = manager.target_ids.copy()
    update = manager.advance()
    if update.landed_sides or not np.array_equal(old_ids, manager.target_ids):
      term = SimpleNamespace(
        num_envs=1, pending_reset=np.array([False]), managers=[manager], _footprint_geometry=None,
        reward_state=SimpleNamespace(ground_height=torch.zeros(1, 2), targets_w=torch.from_numpy(update.completed.targets_w[None])),
      )
      before = pickle.dumps(manager.state_dict())
      visualizer = RecordingVisualizer([0])
      FootstepCommand._debug_vis_impl(term, visualizer)
      assert pickle.dumps(manager.state_dict()) == before
      np.testing.assert_allclose([entry[0][:2] for entry in visualizer.arrows[4:]], update.completed.targets_w[:, :2])
      np.testing.assert_allclose([entry[0][:2] for entry in visualizer.arrows[:4]], update.command.footsteps_w[:, :2])
      for side in update.landed_sides:
        assert update.completed.target_ids[side] not in update.command.future_ids