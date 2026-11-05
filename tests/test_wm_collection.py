import json
from pathlib import Path

import numpy as np
import pytest
import torch
from mjlab.scene import Scene
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.registry import load_env_cfg

from g1_lower_rl.tasks.motion_tracking.env_cfg import motion_tracking_env_cfg
from g1_lower_rl.world_model.collection import MotionCollector, canonical_indices, native_sensor_force_sign
from g1_lower_rl.world_model.corpus import prepare_corpus
from g1_lower_rl.world_model.model import BODY_NAMES, JOINT_NAMES


@pytest.fixture
def position_collector(native_model):
  from types import SimpleNamespace

  from g1_lower_rl.world_model.collection import CollectorConfig

  collector = MotionCollector.__new__(MotionCollector)
  collector.config = CollectorConfig(checkpoint="", num_envs=2, device="cpu", coverage=False)
  collector.num_envs, collector.device = 2, torch.device("cpu")
  collector.indices = canonical_indices(native_model)
  for field, name in (("body", "_body_ids"), ("actuator", "_actuator_ids"), ("qpos", "_qpos_ids"), ("qvel", "_qvel_ids")):
    setattr(collector, name, torch.tensor(collector.indices[field]))
  collector._torso = native_model.body("robot/torso_link").id
  collector._foot_geoms = [
    [native_model.geom(f"robot/{side}_foot{i}_collision").id for i in range(1, 8)] for side in ("left", "right")
  ]
  physics = SimpleNamespace(
    body_mass=torch.tensor(native_model.body_mass, dtype=torch.float32)[None].repeat(2, 1),
    body_ipos=torch.tensor(native_model.body_ipos, dtype=torch.float32)[None].repeat(2, 1, 1),
    geom_friction=torch.tensor(native_model.geom_friction, dtype=torch.float32)[None].repeat(2, 1, 1),
  )
  collector._nominal_torso_com = physics.body_ipos[0, collector._torso].clone()
  data = SimpleNamespace(
    qpos=torch.tensor(native_model.qpos0, dtype=torch.float32)[None].repeat(2, 1),
    qvel=torch.zeros(2, native_model.nv),
    ctrl=torch.zeros(2, native_model.nu),
    qfrc_applied=torch.zeros(2, native_model.nv),
    xfrc_applied=torch.zeros(2, native_model.nbody, 6),
    overflow=torch.zeros(2, dtype=torch.int32),
  )
  collector.env = SimpleNamespace(
    sim=SimpleNamespace(data=data, model=physics, mj_model=native_model),
    scene=SimpleNamespace(env_origins=torch.zeros(2, 3)),
    cfg=SimpleNamespace(decimation=4),
  )
  collector._randomize_foot_friction(torch.arange(2))
  collector._sensor = SimpleNamespace(data=SimpleNamespace(force=torch.zeros(2, 2, 3)), _invalidate_cache=lambda: None)
  collector._force_sign, collector._substep = -1.0, 0
  collector._held_target, collector._tick_start, collector._last_next = None, None, None
  collector._dirty = torch.zeros(2, dtype=torch.bool)
  collector._segment = torch.zeros(2, dtype=torch.long)
  collector._records, collector._collecting = [], True
  collector.nonfinite_envs = torch.zeros(2, dtype=torch.bool)
  collector.counters = dict.fromkeys(
    ("substeps", "control_steps", "state_discontinuities", "invalid_cache", "nonfinite", "overflow_substeps"), 0
  )
  seen = []

  def step():
    seen.append(data.ctrl[:, collector._actuator_ids].clone())
    data.qpos[:, 0] += 0.001
    collector._sensor.data.force[:, :, 2] = -len(seen)

  collector._original_step = step
  return collector, seen


def test_collector_records_one_held_target_transition_and_final_force_not_average(position_collector):
  collector, seen = position_collector
  for substep in range(4):
    collector.env.sim.data.ctrl[:, collector._actuator_ids] = 0.2 + substep
    collector._step()
    assert len(collector._records) == (1 if substep == 3 else 0)
  initial, inputs, final, forces, valid, segment = collector._records[0]
  assert inputs.shape == (2, 145) and initial.shape == final.shape == (2, 71)
  assert valid.all() and segment.eq(0).all()
  assert all(torch.allclose(target, torch.full_like(target, 0.2)) for target in seen)
  torch.testing.assert_close(final[:, 0] - initial[:, 0], torch.full((2,), 0.004))
  torch.testing.assert_close(inputs[:, 68:97], torch.full((2, 29), 0.2))
  torch.testing.assert_close(forces[:, :, 2], torch.full((2, 2), 4.0))
  assert collector.counters["control_steps"] == 2 and collector.counters["substeps"] == 8


def test_context_uses_actual_payload_mass_com_offset_and_rejects_independent_friction(position_collector):
  collector, _ = position_collector
  physics = collector.env.sim.model
  hand = collector.indices["body"][BODY_NAMES.index("left_gripper_base")]
  physics.body_mass[:, hand] += 0.7
  physics.body_ipos[:, collector._torso, 0] += 0.03
  context = collector._context()
  assert context.shape == (2, 48)
  torch.testing.assert_close(context[:, BODY_NAMES.index("left_gripper_base")], physics.body_mass[:, hand])
  torch.testing.assert_close(context[:, 45], torch.full((2,), 0.03), atol=1e-7, rtol=0)
  physics.geom_friction[:, collector._foot_geoms[1], 0] = 1.8
  with pytest.raises(ValueError, match="independent values cannot be averaged"):
    collector._context()


def test_collector_rejects_unmodeled_external_force(position_collector):
  collector, _ = position_collector
  collector.env.sim.data.qfrc_applied[:, 0] = 1
  with pytest.raises(ValueError, match="external forces"):
    collector._step()


def test_intermediate_overflow_quarantines_entire_control_transition(position_collector):
  collector, _ = position_collector
  for substep in range(4):
    collector.env.sim.data.overflow[0] = int(substep == 1)
    collector._step()
  assert collector._records[0][4].tolist() == [False, True]


@pytest.fixture(scope="module")
def native_model():
  cfg = motion_tracking_env_cfg()
  cfg.scene.num_envs = 1
  cfg.scene.sensors = (
    ContactSensorCfg(
      name="wm_feet",
      primary=ContactMatch(mode="body", pattern=r"^(left|right)_ankle_roll_link$", entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="netforce",
    ),
  )
  return Scene(cfg.scene, device="cpu").compile()


def test_canonical_contract_matches_compiled_robot(native_model):
  indices = canonical_indices(native_model)
  assert [native_model.joint(i).name.removeprefix("robot/") for i in indices["joint"]] == list(JOINT_NAMES)
  assert [native_model.body(i).name.removeprefix("robot/") for i in indices["body"]] == list(BODY_NAMES)
  assert len(set(indices["qpos"])) == len(set(indices["qvel"])) == len(set(indices["actuator"])) == 29
  assert indices["actuator"] != list(range(29))
  assert native_model.na == 0
  for joint, actuator in zip(indices["joint"], indices["actuator"], strict=True):
    assert native_model.actuator_trnid[actuator, 0] == joint


def test_ground_force_sensor_calibration_is_unambiguous(native_model):
  foot_geoms = [native_model.geom(f"robot/{side}_foot{i}_collision").id for side in ("left", "right") for i in range(1, 8)]
  sign = native_sensor_force_sign(
    native_model,
    [f"wm_feet_{side}_ankle_roll_link_force" for side in ("left", "right")],
    foot_geoms,
  )
  assert sign == -1.0


def test_bounded_corpus_is_reproducible_and_does_not_replace_files(tmp_path: Path):
  source = tmp_path / "source"
  source.mkdir()
  for family in ("walk", "jump", "amass_ACCAD"):
    for i in range(4):
      (source / f"{family}_{i}.npz").write_bytes(b"test-data")
  destination = tmp_path / "chosen"
  manifest = prepare_corpus(source, destination, max_clips=6, max_bytes=54, seed=42)
  assert manifest["selected_clips"] == 6
  assert manifest["selected_bytes"] == 54
  assert len({motion["family"] for motion in manifest["motions"]}) == 3
  assert all(path.is_symlink() for path in destination.glob("*.npz"))
  assert prepare_corpus(source, destination, max_clips=6, max_bytes=54, seed=42) == manifest
  assert json.loads((destination / "manifest.json").read_text()) == manifest
  with pytest.raises(FileExistsError, match="different corpus"):
    prepare_corpus(source, destination, max_clips=5, max_bytes=54, seed=42)
  with pytest.raises(ValueError, match="No source motion fits"):
    prepare_corpus(source, tmp_path / "too-small", max_bytes=1)


@pytest.mark.parametrize("clips,byte_limit", [(0, 10), (1, 0)])
def test_invalid_corpus_limits_are_explicit(tmp_path, clips, byte_limit):
  with pytest.raises(ValueError, match="positive"):
    prepare_corpus(tmp_path, tmp_path / "chosen", max_clips=clips, max_bytes=byte_limit)


def test_transition_state_layout_uses_native_body_angular_velocity(native_model):
  indices = canonical_indices(native_model)
  assert indices["qpos"] == list(range(7, 36))
  assert indices["qvel"] == list(range(6, 35))
  assert torch.tensor(indices["qvel"]).unique().numel() == 29


@pytest.mark.parametrize("reset_works", [True, False])
def test_nonfinite_recovery_is_targeted_logged_and_keeps_invalid_labels(caplog, reset_works):
  from types import SimpleNamespace

  collector = MotionCollector.__new__(MotionCollector)
  collector.num_envs = 3
  collector.device = torch.device("cpu")
  data = SimpleNamespace(**{name: torch.ones(3, 4) for name in ("qpos", "qvel", "qacc", "qacc_warmstart", "sensordata")})
  data.qpos[1, 0] = torch.nan
  data.overflow = torch.zeros(3, dtype=torch.int32)
  observations = {"history": torch.arange(12.0).reshape(3, 4)}
  untouched = observations["history"][[0, 2]].clone()
  resets = []

  def reset(*, env_ids):
    resets.append(env_ids.clone())
    if reset_works:
      data.qpos[env_ids] = 0
      observations["history"][env_ids] = 0
    return observations, {}

  collector.env = SimpleNamespace(sim=SimpleNamespace(data=data), reset=reset)
  collector._command = SimpleNamespace(motion_ids=torch.tensor([10, 11, 12]))
  collector.nonfinite_envs = torch.zeros(3, dtype=torch.bool)
  collector.last_dones = torch.zeros(3, dtype=torch.bool)
  collector.counters = {"nonfinite_resets": 0}
  collector._records = [
    (
      torch.ones(3, 71),
      torch.ones(3, 145),
      torch.ones(3, 71),
      torch.ones(3, 2, 3),
      torch.ones(3, dtype=torch.bool),
      torch.zeros(3, dtype=torch.long),
    )
    for _ in range(4)
  ]
  collector._records[-1][2][1, 0] = torch.nan
  if reset_works:
    collector._recover_nonfinite(0)
    assert torch.isfinite(data.qpos).all()
    assert collector.last_dones.tolist() == [False, True, False]
    assert collector.counters["nonfinite_resets"] == 1
  else:
    with pytest.raises(FloatingPointError, match="survived 3 targeted resets"):
      collector._recover_nonfinite(0)
  assert len(resets) == (1 if reset_works else 3) and all(ids.tolist() == [1] for ids in resets)
  assert "quarantining" in caplog.text and "motions=[11]" in caplog.text
  assert all(record[4].tolist() == [True, False, True] for record in collector._records)
  assert torch.isnan(collector._records[-1][2][1, 0])
  torch.testing.assert_close(observations["history"][[0, 2]], untouched, rtol=0, atol=0)


@pytest.mark.parametrize("propagates", ["state", "observation", "overflow"])
def test_reset_forward_quarantines_newly_invalid_other_environments(propagates):
  from types import SimpleNamespace

  collector = MotionCollector.__new__(MotionCollector)
  collector.num_envs, collector.device = 3, torch.device("cpu")
  data = SimpleNamespace(**{name: torch.ones(3, 4) for name in ("qpos", "qvel", "qacc", "qacc_warmstart", "sensordata")})
  data.qpos[1, 0] = torch.nan
  data.overflow = torch.zeros(3, dtype=torch.int32)
  observations = {"history": torch.ones(3, 10)}
  resets = []

  def reset(*, env_ids):
    resets.append(env_ids.tolist())
    for name in ("qpos", "qvel", "qacc", "qacc_warmstart", "sensordata"):
      getattr(data, name)[env_ids] = 0
    data.overflow[env_ids] = 0
    observations["history"][env_ids] = 0
    if len(resets) == 1:
      if propagates == "state":
        data.qacc[2, 0] = torch.nan
      elif propagates == "observation":
        observations["history"][2, 0] = torch.nan
      else:
        data.overflow[2] = 64
    return observations, {}

  collector.env = SimpleNamespace(sim=SimpleNamespace(data=data), reset=reset)
  collector._command = SimpleNamespace(motion_ids=torch.arange(3))
  collector.nonfinite_envs = torch.zeros(3, dtype=torch.bool)
  collector.last_dones = torch.zeros(3, dtype=torch.bool)
  collector.counters = {"nonfinite_resets": 0}
  collector._records = [
    (
      torch.ones(3, 71),
      torch.ones(3, 145),
      torch.ones(3, 71),
      torch.ones(3, 2, 3),
      torch.ones(3, dtype=torch.bool),
      torch.zeros(3, dtype=torch.long),
    )
    for _ in range(4)
  ]
  collector._recover_nonfinite(0)
  assert resets == [[1], [2]]
  assert collector.nonfinite_envs.tolist() == collector.last_dones.tolist() == [False, True, True]
  assert all(record[4].tolist() == [True, False, False] for record in collector._records)
  assert collector.counters["nonfinite_resets"] == 2
  assert torch.isfinite(data.qacc).all() and torch.isfinite(observations["history"]).all()


@pytest.mark.parametrize("task", ["G1-Gloria-FootstepTracking", "G1-Gloria-LowerBody-Flat"])
def test_three_task_families_share_the_learned_static_physics(native_model, task):
  cfg = load_env_cfg(task)
  cfg.scene.num_envs = 1
  other = Scene(cfg.scene, device="cpu").compile()
  assert canonical_indices(other) == canonical_indices(native_model)
  for field in (
    "body_parentid",
    "body_pos",
    "body_quat",
    "body_iquat",
    "jnt_type",
    "jnt_axis",
    "jnt_pos",
    "jnt_limited",
    "jnt_range",
    "dof_damping",
    "geom_type",
    "geom_size",
    "geom_pos",
    "geom_quat",
    "geom_condim",
    "geom_contype",
    "geom_conaffinity",
    "geom_solref",
    "geom_solimp",
    "geom_margin",
    "geom_gap",
  ):
    np.testing.assert_array_equal(getattr(other, field), getattr(native_model, field), err_msg=field)
  assert cfg.sim.mujoco.integrator == "implicitfast"
  assert cfg.sim.mujoco.iterations == 10 and cfg.sim.mujoco.ls_iterations == 20
  assert cfg.sim.mujoco.timestep in (0.005, 0.0025)
  assert cfg.sim.mujoco.timestep * cfg.decimation == pytest.approx(0.02)
