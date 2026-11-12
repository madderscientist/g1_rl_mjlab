"""Unit tests for lower-body velocity WM policy, task differentiability, and ONNX/deploy contracts."""

from __future__ import annotations

from pathlib import Path

import mujoco
import torch

from g1_lower_rl import deploy
from g1_lower_rl.tasks import LOWER_BODY_WM_TASK, list_tasks
from g1_lower_rl.tasks.lower_body.wm.playback import FORMAT, export_onnx
from g1_lower_rl.tasks.lower_body.wm.policy import (
  ACTION_DIM,
  OBS_DIM,
  OBS_TERMS,
  VelocityMlpActor,
  actor_architecture_metadata,
  migrate_actor_state_dict,
  unflatten_native_history,
  velocity_observation,
  velocity_target,
)
from g1_lower_rl.tasks.lower_body.wm.runner import DEFAULT_EXPLORATION_STAGES, VelocityWmConfig
from g1_lower_rl.tasks.lower_body.wm.task import (
  AUX_DIM,
  CRITIC_DIM,
  VelocityReplay,
  VelocityWmTask,
  _probabilistic_termination,
  flat_env_cfg_wm,
  pack_task_aux,
)
from g1_lower_rl.world_model.adapters.kinematics import StaticKinematicTemplate, TorchKinematics
from g1_lower_rl.world_model.collection import TransitionBatch
from g1_lower_rl.world_model.model import JOINT_NAMES, PositionWorldModel, PositionWorldModelCfg, pack_inputs

ROOT = Path(__file__).resolve().parents[1]


def test_task_registered_and_wm_env_contract():
  assert LOWER_BODY_WM_TASK in list_tasks()
  cfg = flat_env_cfg_wm(history_length=10, curriculum_steps_per_iter=2.0)
  assert cfg.observations["actor"].history_length == 10
  for term in cfg.observations["actor"].terms.values():
    assert term.delay_max_lag == 0
  assert "foot_friction" in cfg.events
  assert "foot_friction_left" not in cfg.events
  assert "foot_friction_right" not in cfg.events
  assert "arm_torque" not in cfg.events
  assert "body_impulse" not in cfg.events
  assert "push_robot" not in cfg.events
  assert any(sensor.name == "wm_feet" for sensor in cfg.scene.sensors)
  assert cfg.terminations["fell_over"].func is _probabilistic_termination
  assert cfg.terminations["collapsed"].func is _probabilistic_termination
  assert cfg.terminations["fell_over"].params["p_term"] == 0.05


def test_exploration_schedule_and_overrides():
  cfg = VelocityWmConfig()
  assert cfg.exploration_stages == DEFAULT_EXPLORATION_STAGES
  assert cfg.active_exploration(0) == (0.25, 0.08)
  assert cfg.active_exploration(6499) == (0.25, 0.08)
  assert cfg.active_exploration(6500) == (0.08, 0.015)
  override = VelocityWmConfig(action_noise=0.14)
  assert override.active_exploration(0) == (0.14, 0.08)
  assert override.active_exploration(7000) == (0.14, 0.015)


def test_unflatten_native_history_preserves_frame_order():
  batch, h = 3, 10
  frames = torch.randn(batch, h, OBS_DIM)
  # Construct mjlab term-major flattened tensor [B, sum(H * width_i)]
  term_major_parts = []
  offset = 0
  for _, width in OBS_TERMS:
    term_major_parts.append(frames[:, :, offset : offset + width].reshape(batch, h * width))
    offset += width
  term_major = torch.cat(term_major_parts, dim=-1)
  reconstructed = unflatten_native_history(term_major, history_length=h)
  torch.testing.assert_close(reconstructed, frames, rtol=0, atol=0)


def test_differentiable_velocity_task_gradients_from_standing():
  mj_model = mujoco.MjModel.from_xml_path(
    str(ROOT / "g1_lower_rl/assets/g1_gloria/xmls/scene_g1_with_dual_gloria_m.xml")
  )
  template = StaticKinematicTemplate.from_mujoco(mj_model)
  fk = TorchKinematics(template).to(dtype=torch.float32)
  fk.level_batching = True

  default_pos = torch.zeros(29, dtype=torch.float32)
  soft_limits = torch.stack((torch.full((29,), -2.5), torch.full((29,), 2.5)), dim=-1)
  task = VelocityWmTask(
    kinematics=fk,
    default_position=default_pos,
    soft_limits=soft_limits,
    waist_kp=torch.full((3,), 200.0),
    waist_kd=torch.full((3,), 5.0),
    waist_effort_limit=torch.full((3,), 88.0),
  )

  b = 4
  prev_state = torch.zeros(b, 71, dtype=torch.float32)
  prev_state[:, 2] = 0.76
  prev_state[:, 3] = 1.0

  next_state = prev_state.clone().requires_grad_(True)
  # Double-support standing forces (180 N per foot), swing phase active on one leg
  foot_force = torch.zeros(b, 2, 3, dtype=torch.float32)
  foot_force[:, :, 2] = 180.0
  foot_force.requires_grad_(True)

  action = torch.zeros(b, ACTION_DIM, dtype=torch.float32, requires_grad=True)
  arm_target = torch.zeros(b, 14, dtype=torch.float32)
  limits = torch.stack((torch.full((29,), -3.0), torch.full((29,), 3.0)), dim=-1)
  limited = torch.ones(29, dtype=torch.bool)
  target29 = velocity_target(
    action,
    arm_target,
    default_pos[:ACTION_DIM],
    torch.full((ACTION_DIM,), 0.4),
    torch.zeros(b, ACTION_DIM),
    limits,
    limited,
  )

  context = torch.ones(b, 48)
  twist = torch.tensor([[0.8, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.6], [-0.5, 0.2, 0.1]])
  height_cmd = torch.full((b,), 0.75)
  episode_step = torch.tensor([10, 10, 20, 25], dtype=torch.long)
  phase_offset = torch.tensor([0.45, 0.0, 0.2, 0.7])
  cmd_elapsed = torch.full((b,), 0.2)
  drift_anchor = torch.zeros(b, 2)
  is_stationary = torch.tensor([False, True, True, False])
  aux = pack_task_aux(
    lin_accum=torch.zeros(b, 2),
    ang_accum=torch.zeros(b),
    contact_prob=torch.ones(b, 2),
    air_time=torch.zeros(b, 2),
    contact_time=torch.full((b, 2), 0.2),
  )
  assert aux.shape == (b, AUX_DIM)

  reward, term, next_aux, foot_z = task.step(
    prev_state=prev_state,
    next_state=next_state,
    foot_force=foot_force,
    target29=target29,
    action=action,
    prev_action=torch.zeros_like(action),
    aux=aux,
    twist=twist,
    height_cmd=height_cmd,
    next_episode_step=episode_step + 1,
    phase_offset=phase_offset,
    cmd_elapsed=cmd_elapsed,
    drift_anchor=drift_anchor,
    is_stationary=is_stationary,
  )
  assert reward.shape == (b,)
  assert term.shape == (b,)
  assert next_aux.shape == (b, AUX_DIM)
  assert foot_z.shape == (b, 2)
  assert torch.isfinite(reward).all() and torch.isfinite(term).all()
  assert (term < 1e-4).all()

  g_state, g_force, g_act = torch.autograd.grad(reward.sum(), (next_state, foot_force, action))
  assert torch.isfinite(g_state).all() and g_state.abs().sum() > 0
  assert torch.isfinite(g_force).all() and g_force.abs().sum() > 0
  assert torch.isfinite(g_act).all()

  wm = PositionWorldModel(PositionWorldModelCfg())
  wm.set_normalization(
    torch.zeros(145),
    torch.ones(145),
    torch.ones(70),
    torch.ones(6),
    torch.ones(70),
  )
  feat = task.critic_features(
    wm,
    next_state,
    context,
    action,
    next_aux,
    arm_target,
    twist,
    height_cmd,
    episode_step + 1,
    phase_offset,
    cmd_elapsed + 0.02,
    drift_anchor,
    is_stationary,
    foot_z=foot_z,
  )
  assert feat.shape == (b, CRITIC_DIM)
  assert torch.isfinite(feat).all()

  obs = velocity_observation(
    next_state.detach(),
    action.detach(),
    twist,
    height_cmd,
    episode_step + 1,
    phase_offset,
    torch.zeros(b, ACTION_DIM),
    default_pos[:ACTION_DIM],
    torch.zeros(b, 6),
  )
  assert obs.shape == (b, OBS_DIM)
  assert torch.isfinite(obs).all()


def test_velocity_replay_filters_upright_starts():
  num_envs = 8
  replay = VelocityReplay(capacity=16, num_envs=num_envs, validation_envs=2, device="cpu")
  state = torch.zeros(1, num_envs, 71)
  state[:, :3, 2] = 0.74
  state[:, :3, 3] = 1.0
  # Mark envs 3..5 as fallen on floor (z=0.18, 90deg pitch)
  state[:, 3:6, 2] = 0.18
  state[:, 3:6, 3] = 0.7071
  state[:, 3:6, 5] = 0.7071
  state[:, 6:, 2] = 0.74
  state[:, 6:, 3] = 1.0
  batch = TransitionBatch(
    state=state,
    inputs=pack_inputs(state, torch.zeros(1, num_envs, 29), torch.ones(1, num_envs, 48)),
    next_state=state.clone(),
    foot_force=torch.zeros(1, num_envs, 2, 3),
    segment=torch.zeros(1, num_envs, dtype=torch.int64),
    valid=torch.ones(1, num_envs, dtype=torch.bool),
  )
  record = {
    "latest": torch.zeros(num_envs, OBS_DIM),
    "aux": torch.zeros(num_envs, AUX_DIM),
    "encoder_bias": torch.zeros(num_envs, ACTION_DIM),
    "imu_bias": torch.zeros(num_envs, 6),
    "arm_target": torch.zeros(num_envs, 14),
    "arm_ramp_start": torch.zeros(num_envs, 14),
    "arm_ramp_target": torch.zeros(num_envs, 14),
    "arm_ramp_steps": torch.zeros(num_envs, dtype=torch.long),
    "arm_ramp_active": torch.zeros(num_envs, dtype=torch.bool),
    "twist": torch.zeros(num_envs, 3),
    "height_cmd": torch.full((num_envs,), 0.74),
    "height_target": torch.full((num_envs,), 0.74),
    "episode_step": torch.zeros(num_envs, dtype=torch.long),
    "phase_offset": torch.zeros(num_envs),
    "remaining": torch.full((num_envs,), 500, dtype=torch.long),
    "cmd_elapsed": torch.full((num_envs,), 0.2),
    "drift_anchor": torch.zeros(num_envs, 2),
    "is_stationary": torch.zeros(num_envs, dtype=torch.bool),
  }
  for _ in range(12):
    replay.append_control(batch, record)
  starts = replay.sample_starts(batch_size=12, history_length=10)
  assert (starts.state[:, 2] > 0.40).all()


def test_onnx_export_and_cpu_deploy_compatibility(tmp_path: Path):
  actor = VelocityMlpActor(history_length=10, hidden_dims=(128, 64))
  saved = {
    "format": FORMAT,
    "iteration": 0,
    "config": {"history_length": 10, "actor_hidden_dims": [128, 64]},
    "origin": {"architecture": actor_architecture_metadata(actor)},
    "actor_state_dict": actor.state_dict(),
    "controller_contract": {
      "default_joint_pos_29": [0.0] * len(JOINT_NAMES),
      "action_scale_15": [0.4] * ACTION_DIM,
    },
  }
  exported = export_onnx(saved, tmp_path / "onnx")
  assert exported["stateless"].is_file()
  assert exported["deploy"].is_file()
  assert exported["contract"].is_file()

  # Verify deploy.load_policy and deploy.rollout run cleanly on policy_deploy.onnx!
  policy = deploy.load_policy(str(exported["deploy"]))
  model = deploy.build_model()
  data = mujoco.MjData(model)
  index = deploy.Index(model, policy)
  deploy.reset(model, data, policy.joint_names, policy.default_pos)
  log, _ = deploy.rollout(
    model,
    data,
    policy.session,
    index,
    policy.action_default_pos,
    policy.action_scale,
    [0.5, 0.0, 0.0, 0.75],
    seconds=0.2,
    video=None,
  )
  assert len(log["vel_b"]) >= 1


def test_legacy_uncentered_actor_migration_preserves_exact_outputs():
  new_actor = VelocityMlpActor(history_length=10, hidden_dims=(64, 32))
  legacy_state = {k: v.clone() for k, v in new_actor.state_dict().items() if k != "frame_offset"}
  legacy_scale = legacy_state["frame_scale"].clone()
  legacy_scale[9] = 1.0
  legacy_state["frame_scale"] = legacy_scale

  x = torch.randn(8, 10, OBS_DIM)
  x[:, :, 9] = torch.linspace(0.50, 0.80, 8).unsqueeze(1)
  with torch.no_grad():
    expected = new_actor.net((x * legacy_scale).flatten(1))
    migrated, changed = migrate_actor_state_dict(new_actor, legacy_state)
    assert changed
    new_actor.load_state_dict(migrated, strict=True)
    actual = new_actor(x)
  torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
