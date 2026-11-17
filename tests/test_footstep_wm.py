"""Unit tests for Footstep Twin-Mirror WM policy, differentiable re-anchoring, auto-curriculum, and ONNX."""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import torch

from g1_lower_rl.assets import LOWER_BODY_JOINTS, WHOLE_BODY_JOINTS
from g1_lower_rl.footstep_deploy import FootstepPolicy
from g1_lower_rl.rl.mirror import _pair_joints
from g1_lower_rl.tasks import FOOTSTEP_WM_TASK, list_tasks
from g1_lower_rl.tasks.footstep_tracking.wm.playback import FORMAT, export_onnx
from g1_lower_rl.tasks.footstep_tracking.wm.policy import (
  ACTION_DIM,
  CMD_ENC_DIM,
  FRAME_DIM,
  FootstepTwinMirrorMlpActor,
  actor_architecture_metadata,
  encode_command,
  encode_proprio_frame,
)
from g1_lower_rl.tasks.footstep_tracking.wm.task import (
  AUTO_CURRICULUM_STAGES,
  AUX_DIM,
  CRITIC_DIM,
  FootstepWmTask,
  _probabilistic_termination,
  footstep_env_cfg_wm,
  pack_footstep_aux,
  stage_qualifies,
)
from g1_lower_rl.world_model.adapters.kinematics import StaticKinematicTemplate, TorchKinematics
from g1_lower_rl.world_model.model import JOINT_NAMES, PositionWorldModel, PositionWorldModelCfg

ROOT = Path(__file__).resolve().parents[1]


def test_task_registered_and_wm_env_contract():
  assert FOOTSTEP_WM_TASK in list_tasks()
  cfg = footstep_env_cfg_wm()
  assert math.isclose(cfg.sim.mujoco.timestep, 0.005)
  assert cfg.decimation == 4
  assert "torso_heading" in cfg.commands
  assert "foot_friction" in cfg.events
  assert "foot_friction_left" not in cfg.events
  assert "foot_friction_right" not in cfg.events
  assert "arm_torque" not in cfg.events
  assert "body_impulse" not in cfg.events
  assert "push_robot" not in cfg.events
  assert "numerical_safety" not in cfg.terminations
  assert any(sensor.name == "wm_feet" for sensor in cfg.scene.sensors)
  assert cfg.terminations["fell_over"].func is _probabilistic_termination
  assert cfg.terminations["collapsed"].func is _probabilistic_termination
  assert cfg.terminations["fell_over"].params["p_term"] == 0.05


def test_twin_mirror_actor_exact_sagittal_equivariance_and_zero_dc():
  actor = FootstepTwinMirrorMlpActor(history_length=10, hidden_dims=(128, 64))
  default_q = actor.default_joint_pos

  # 1. Verify upright default standing proprioception encodes to exact 0.0
  raw_upright = torch.zeros(2, 70)
  raw_upright[:, :29] = default_q
  raw_upright[:, 63] = -1.0
  raw_upright[:, 69] = -1.0
  frame0 = encode_proprio_frame(raw_upright, torch.zeros(2, ACTION_DIM), default_q)
  assert torch.allclose(frame0, torch.zeros_like(frame0), atol=1e-7)

  # 2. Verify exact sagittal reflection equivariance on random trajectories & commands
  q_perm, q_sign = _pair_joints(list(WHOLE_BODY_JOINTS))
  act_perm, act_sign = _pair_joints(list(LOWER_BODY_JOINTS))
  q_perm_t = torch.tensor(q_perm, dtype=torch.long)
  q_sign_t = torch.tensor(q_sign, dtype=torch.float32)
  act_perm_t = torch.tensor(act_perm, dtype=torch.long)
  act_sign_t = torch.tensor(act_sign, dtype=torch.float32)

  b, h = 6, 10
  rng = torch.Generator().manual_seed(12345)
  raw_p = torch.randn(b, h, 70, generator=rng) * 0.2
  raw_p[..., :29] += default_q
  raw_p[..., 63] = -0.95
  raw_p[..., 69] = -0.95
  prev_a = torch.randn(b, h, ACTION_DIM, generator=rng) * 0.3

  # Reflect raw physical sensors across sagittal plane y = 0
  mir_p = torch.empty_like(raw_p)
  q_rel = raw_p[..., :29] - default_q
  mir_p[..., :29] = q_rel.index_select(-1, q_perm_t) * q_sign_t + default_q
  mir_p[..., 29:58] = raw_p[..., 29:58].index_select(-1, q_perm_t) * q_sign_t
  mir_p[..., 58:61] = raw_p[..., 58:61] * torch.tensor([-1.0, 1.0, -1.0])
  mir_p[..., 61:64] = raw_p[..., 61:64] * torch.tensor([1.0, -1.0, 1.0])
  mir_p[..., 64:67] = raw_p[..., 64:67] * torch.tensor([-1.0, 1.0, -1.0])
  mir_p[..., 67:70] = raw_p[..., 67:70] * torch.tensor([1.0, -1.0, 1.0])
  mir_prev_a = prev_a.index_select(-1, act_perm_t) * act_sign_t

  hist_orig = encode_proprio_frame(raw_p, prev_a, default_q)
  hist_mir = encode_proprio_frame(mir_p, mir_prev_a, default_q)

  phase = torch.rand(b, generator=rng) * (2.0 * math.pi)
  freq = torch.tensor([0.0, 1.2, 0.9, 1.6, 0.0, 1.4])
  fp_orig = torch.tensor(
    [
      [[0.0, 0.0, 0.0], [0.0, -0.24, 0.0]],
      [[0.25, 0.12, 0.15], [0.10, -0.14, -0.10]],
      [[-0.12, 0.20, -0.25], [0.18, -0.05, 0.20]],
      [[0.30, 0.05, 0.0], [0.05, -0.22, 0.05]],
      [[0.0, 0.24, 0.0], [0.0, 0.0, 0.0]],
      [[0.22, 0.15, 0.30], [-0.05, -0.15, -0.20]],
    ]
  )
  # Sagittal mirror of [L, R] is [reflect(R), reflect(L)]
  fp_mir = torch.stack(
    (
      torch.stack((fp_orig[:, 1, 0], -fp_orig[:, 1, 1], -fp_orig[:, 1, 2]), dim=-1),
      torch.stack((fp_orig[:, 0, 0], -fp_orig[:, 0, 1], -fp_orig[:, 0, 2]), dim=-1),
    ),
    dim=1,
  )
  theta = torch.linspace(-0.6, 0.6, b)

  cmd_orig = encode_command(phase, freq, fp_orig, theta)
  cmd_mir = encode_command(phase + math.pi, freq, fp_mir, -theta)
  assert cmd_orig.shape == (b, CMD_ENC_DIM)

  with torch.no_grad():
    act_orig = actor(hist_orig, cmd_orig)
    act_mir = actor(hist_mir, cmd_mir)
    expected_mir = act_orig.index_select(-1, act_perm_t) * act_sign_t
  torch.testing.assert_close(act_mir, expected_mir, rtol=1e-5, atol=1e-6)


def test_differentiable_footstep_task_and_reanchoring_gradients():
  mj_model = mujoco.MjModel.from_xml_path(
    str(ROOT / "g1_lower_rl/assets/g1_gloria/xmls/scene_g1_with_dual_gloria_m.xml")
  )
  template = StaticKinematicTemplate.from_mujoco(mj_model)
  fk = TorchKinematics(template).to(dtype=torch.float32)
  fk.level_batching = True

  default_pos = torch.zeros(29, dtype=torch.float32)
  soft_limits = torch.stack((torch.full((29,), -2.5), torch.full((29,), 2.5)), dim=-1)
  task = FootstepWmTask(
    kinematics=fk,
    default_position=default_pos,
    soft_limits=soft_limits,
    lower_kp=torch.full((ACTION_DIM,), 150.0),
    lower_kd=torch.full((ACTION_DIM,), 5.0),
    lower_effort_limit=torch.full((ACTION_DIM,), 88.0),
    action_scale=torch.full((ACTION_DIM,), 0.4),
  )

  b = 4
  prev_state = torch.zeros(b, 71)
  prev_state[:, 2] = 0.75
  prev_state[:, 3] = 1.0

  next_state = prev_state.clone().requires_grad_(True)
  foot_force = torch.zeros(b, 2, 3)
  foot_force[:, :, 2] = 175.0
  foot_force.requires_grad_(True)
  action = torch.full((b, ACTION_DIM), 0.1, requires_grad=True)

  aux = pack_footstep_aux(
    contact_prob=torch.ones(b, 2),
    air_time=torch.zeros(b, 2),
    contact_time=torch.full((b, 2), 0.2),
    anchor=torch.zeros(b, 3),
  ).requires_grad_(True)
  assert aux.shape == (b, AUX_DIM)

  # Env 0 is in double support with pending_anchor=0 to verify differentiable re-anchoring!
  end_phase = torch.tensor([0.5 * math.pi, 0.1 * math.pi, 1.0 * math.pi, 0.5 * math.pi])
  frequency = torch.tensor([1.2, 1.2, 1.2, 0.0])
  supports = torch.tensor([[[0.0, 0.12, 0.0], [0.0, -0.12, 0.0]]] * b)
  targets = torch.tensor([[[0.25, 0.12, 0.0], [0.0, -0.12, 0.0]]] * b)
  future_world = torch.tensor([[[0.25, 0.12, 0.0], [0.50, -0.12, 0.0]]] * b)
  pending_anchor = torch.tensor([0, -1, -1, -1], dtype=torch.long)
  double_support = torch.tensor([True, False, False, False])
  theta = torch.zeros(b)

  reward, term, next_aux, frame, next_pending = task.step(
    prev_state=prev_state,
    next_state=next_state,
    foot_force=foot_force,
    action=action,
    prev_action=torch.zeros_like(action),
    aux=aux,
    end_phase=end_phase,
    frequency=frequency,
    targets=targets,
    supports=supports,
    future_world=future_world,
    pending_anchor=pending_anchor,
    double_support=double_support,
    theta=theta,
  )
  assert reward.shape == (b,)
  assert term.shape == (b,)
  assert next_aux.shape == (b, AUX_DIM)
  assert int(next_pending[0]) == -1

  # Verify BOTH reward AND re-anchored next_aux[:, 6:9] carry nonzero gradients wrt next_state and foot_force!
  g_anchor_state = torch.autograd.grad(next_aux[0, 6:9].sum(), next_state, retain_graph=True)[0]
  assert torch.isfinite(g_anchor_state).all() and g_anchor_state.abs().sum() > 0
  g_state, g_force, g_act = torch.autograd.grad(reward.sum(), (next_state, foot_force, action))
  assert torch.isfinite(g_state).all() and g_state.abs().sum() > 0
  assert torch.isfinite(g_force).all() and g_force.abs().sum() > 0
  assert torch.isfinite(g_act).all() and g_act.abs().sum() > 0

  wm = PositionWorldModel(PositionWorldModelCfg())
  wm.set_normalization(torch.zeros(145), torch.ones(145), torch.ones(70), torch.ones(6), torch.ones(70))
  queue = torch.zeros(b, 4, 3)
  feat = task.critic_features(
    wm,
    next_state,
    torch.ones(b, 48),
    action,
    next_aux,
    torch.zeros(b, 14),
    end_phase,
    frequency,
    targets,
    supports,
    queue,
    torch.zeros(b, dtype=torch.long),
    theta,
    foot_pos=frame.foot_pos,
    foot_yaw=frame.foot_yaw,
  )
  assert feat.shape == (b, CRITIC_DIM)
  assert torch.isfinite(feat).all()


def test_auto_curriculum_qualification():
  from dataclasses import replace

  from g1_lower_rl.footsteps import FootstepManagerCfg, RandomCommandCfg

  # Verify all 8 stages construct valid FootstepManagerCfg and RandomCommandCfg without error
  base_mgr = FootstepManagerCfg()
  base_src = RandomCommandCfg()
  for st in AUTO_CURRICULUM_STAGES:
    mgr = replace(
      base_mgr,
      frequency_range=st.frequency_range,
      initial_frequency=1.2 if st.initial_frequency is None else st.initial_frequency,
    )
    src = replace(
      base_src,
      frequency_range=st.frequency_range,
      initial_frequency=st.initial_frequency,
      initial_standing_probability=st.initial_standing_prob,
      stop_probability=st.stop_prob,
      hold_time_s=(4.0, 14.0),
    )
    assert mgr.frequency_range == src.frequency_range

  s0 = AUTO_CURRICULUM_STAGES[0]
  bad = {
    "native/validation_failure_rate": 0.002,
    "native/det_mtbf_s": 12.0,
    "native/det_unbroken_mean_s": 5.0,
    "native/validation_ss_match": 0.70,
    "native/validation_swing_lift_m": 0.05,
    "native/validation_stance_xy_err_m": 0.08,
    "native/validation_stance_yaw_err_rad": 0.05,
    "native/validation_theta_err_rad": 0.05,
  }
  good = {**bad, "native/det_mtbf_s": 60.0, "native/det_unbroken_mean_s": 24.0}
  assert not stage_qualifies(s0, bad)
  assert stage_qualifies(s0, good)


def test_onnx_export_and_footstep_deploy_policy_compatibility(tmp_path: Path):
  actor = FootstepTwinMirrorMlpActor(history_length=10, hidden_dims=(128, 64))
  saved = {
    "format": FORMAT,
    "iteration": 0,
    "config": {"history_length": 10, "actor_hidden_dims": [128, 64]},
    "origin": {"architecture": actor_architecture_metadata(actor)},
    "actor_state_dict": actor.state_dict(),
    "controller_contract": {
      "default_joint_pos_29": actor.default_joint_pos.tolist(),
      "action_scale_15": [0.4] * ACTION_DIM,
    },
  }
  exported = export_onnx(saved, tmp_path / "onnx")
  policy = FootstepPolicy(exported["deploy"])
  policy.reset()
  obs = np.zeros((1, 79), dtype=np.float32)
  obs[:, :29] = actor.default_joint_pos.numpy()
  obs[:, 63] = obs[:, 69] = -1.0
  obs[:, 71] = 1.2
  act, targets = policy.step(obs)
  assert act.shape == (15,)
  assert targets.shape == (15,)
  assert np.isfinite(act).all() and np.isfinite(targets).all()
  assert FRAME_DIM == 85 and len(JOINT_NAMES) == 29


def test_ppo_aligned_standing_reward_prefers_straight_legs():
  mj_model = mujoco.MjModel.from_xml_path(
    str(ROOT / "g1_lower_rl/assets/g1_gloria/xmls/scene_g1_with_dual_gloria_m.xml")
  )
  template = StaticKinematicTemplate.from_mujoco(mj_model)
  fk = TorchKinematics(template).to(dtype=torch.float32)
  fk.level_batching = True

  joint_ids = [mj_model.joint(f"{name}").id for name in JOINT_NAMES]
  hard_limits = torch.tensor(mj_model.jnt_range[joint_ids], dtype=torch.float32)
  center = 0.5 * (hard_limits[:, 0] + hard_limits[:, 1])
  half_span = 0.45 * (hard_limits[:, 1] - hard_limits[:, 0])
  soft_limits = torch.stack((center - half_span, center + half_span), dim=-1)

  default_pos = torch.zeros(29, dtype=torch.float32)
  task = FootstepWmTask(
    kinematics=fk,
    default_position=default_pos,
    soft_limits=soft_limits,
    hard_limits=hard_limits,
    lower_kp=torch.full((ACTION_DIM,), 150.0),
    lower_kd=torch.full((ACTION_DIM,), 5.0),
    lower_effort_limit=torch.full((ACTION_DIM,), 88.0),
    action_scale=torch.full((ACTION_DIM,), 0.4),
  )

  # Batch of 2 standing states (f == 0): row 0 straight-leg (-2 deg knee), row 1 crouched (50 deg knee)
  state = torch.zeros(2, 71)
  state[:, 3] = 1.0
  state[0, 2] = 0.79
  state[0, 7 + 3] = math.radians(-2.0)
  state[0, 7 + 9] = math.radians(-2.0)

  state[1, 2] = 0.73
  state[1, 7 + 0] = math.radians(-20.5)
  state[1, 7 + 6] = math.radians(-20.5)
  state[1, 7 + 3] = math.radians(50.5)
  state[1, 7 + 9] = math.radians(50.5)
  state[1, 7 + 4] = math.radians(-30.0)
  state[1, 7 + 10] = math.radians(-30.0)
  state.requires_grad_(True)

  # Equilibrium PD action: knee extension against gravity is negative torque (-q_knee), plantarflexion is positive (+q_ankle)
  eq_act = state[:, 7 : 7 + ACTION_DIM].detach().clone() / 0.4
  eq_act[0, [3, 9]] -= 5.5 / (150.0 * 0.4)
  eq_act[1, [3, 9]] -= 12.5 / (150.0 * 0.4)
  eq_act[1, [4, 10]] += 11.5 / (150.0 * 0.4)

  foot_force = torch.zeros(2, 2, 3)
  foot_force[:, :, 2] = 175.0
  aux = pack_footstep_aux(torch.ones(2, 2), torch.zeros(2, 2), torch.full((2, 2), 0.5), torch.zeros(2, 3))
  with torch.no_grad():
    fr_init = fk(state.detach(), validate=False)
    targets = torch.cat((fr_init.foot_position[:, :, :2], torch.zeros(2, 2, 1)), dim=-1)

  reward, _, _, frame, _ = task.step(
    prev_state=state.detach(),
    next_state=state,
    foot_force=foot_force,
    action=eq_act,
    prev_action=eq_act,
    aux=aux,
    end_phase=torch.zeros(2),
    frequency=torch.zeros(2),
    targets=targets,
    supports=targets,
    future_world=targets,
    pending_anchor=torch.full((2,), -1, dtype=torch.long),
    double_support=torch.zeros(2, dtype=torch.bool),
    theta=torch.zeros(2),
  )
  assert float(frame.head_z[0].detach()) > 1.255 and float(frame.head_z[1].detach()) < 1.215
  assert float(reward[0].detach()) > float(reward[1].detach())
  grad_state = torch.autograd.grad(reward[1], state)[0]
  # Gradient wrt crouched knee angles must be strictly negative (pushing knees straighter toward 0)
  assert float(grad_state[1, 7 + 3]) < 0.0 and float(grad_state[1, 7 + 9]) < 0.0

  # Verify soft_landing penalizes high first-contact touchdown force spikes when moving
  aux_air = pack_footstep_aux(torch.zeros(2, 2), torch.full((2, 2), 0.25), torch.zeros(2, 2), torch.zeros(2, 3))
  soft_ff = torch.zeros(2, 2, 3)
  soft_ff[0, :, 2] = 200.0
  soft_ff[1, :, 2] = 700.0
  r_impact, *_ = task.step(
    prev_state=state.detach(),
    next_state=state.detach(),
    foot_force=soft_ff,
    action=eq_act,
    prev_action=eq_act,
    aux=aux_air,
    end_phase=torch.full((2,), 0.5 * math.pi),
    frequency=torch.full((2,), 1.2),
    targets=targets,
    supports=targets,
    future_world=targets,
    pending_anchor=torch.full((2,), -1, dtype=torch.long),
    double_support=torch.zeros(2, dtype=torch.bool),
    theta=torch.zeros(2),
  )
  assert float(r_impact[0]) > float(r_impact[1])

