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

  # Batch of 2 standing states (f == 0): row 0 natural straight (+3 deg knee), row 1 crouched (50.5 deg knee)
  state = torch.zeros(2, 71)
  state[:, 3] = 1.0
  state[0, 2] = 0.79
  state[0, 7 + 3] = math.radians(3.0)
  state[0, 7 + 9] = math.radians(3.0)

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
  assert aux.shape == (2, AUX_DIM) and AUX_DIM == 11
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
  # Gradient wrt crouched knee angles must be strictly negative (pushing knees straighter toward +3.5 deg)
  assert float(grad_state[1, 7 + 3]) < 0.0 and float(grad_state[1, 7 + 9]) < 0.0

  # Verify differentiable 50x limit-pushing penalty repels hyperextended knees (-4.9 deg near -5.0 deg stop) with positive gradient!
  hyp_state = state.detach().clone()
  hyp_state[1] = hyp_state[0]
  hyp_state[1, 7 + 3] = math.radians(-4.9)
  hyp_state[1, 7 + 9] = math.radians(-4.9)
  hyp_state.requires_grad_(True)
  hyp_act = eq_act.clone()
  hyp_act[1, [3, 9]] = (math.radians(-4.9) - 10.0 / 150.0) / 0.4
  r_hyp, *_ = task.step(
    prev_state=hyp_state.detach(),
    next_state=hyp_state,
    foot_force=foot_force,
    action=hyp_act,
    prev_action=hyp_act,
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
  assert float(r_hyp[0].detach()) > float(r_hyp[1].detach())
  grad_hyp = torch.autograd.grad(r_hyp[1], hyp_state)[0]
  assert float(grad_hyp[1, 7 + 3]) > 0.0 and float(grad_hyp[1, 7 + 9]) > 0.0

  # Verify soft_landing penalizes high first-contact touchdown force spikes when moving (holding state/action identical)
  aux_air = pack_footstep_aux(torch.zeros(2, 2), torch.full((2, 2), 0.25), torch.zeros(2, 2), torch.zeros(2, 3))
  soft_state = state[0:1].detach().expand(2, -1)
  soft_act = eq_act[0:1].expand(2, -1)
  soft_targets = targets[0:1].expand(2, -1, -1)
  soft_ff = torch.zeros(2, 2, 3)
  soft_ff[0, :, 2] = 200.0
  soft_ff[1, :, 2] = 700.0
  r_impact, *_ = task.step(
    prev_state=soft_state,
    next_state=soft_state,
    foot_force=soft_ff,
    action=soft_act,
    prev_action=soft_act,
    aux=aux_air,
    end_phase=torch.full((2,), 0.5 * math.pi),
    frequency=torch.full((2,), 1.2),
    targets=soft_targets,
    supports=soft_targets,
    future_world=soft_targets,
    pending_anchor=torch.full((2,), -1, dtype=torch.long),
    double_support=torch.zeros(2, dtype=torch.bool),
    theta=torch.zeros(2),
  )
  assert float(r_impact[0]) > float(r_impact[1])


def test_twin_mirror_gru_actor_parity_symmetry_and_onnx(tmp_path: Path):
  import inspect
  from dataclasses import asdict

  from tensordict import TensorDict

  from g1_lower_rl.rl.footstep_model import FootstepActor, FootstepModelCfg
  from g1_lower_rl.tasks.footstep_tracking.wm.policy import FootstepTwinMirrorGruActor

  gru_actor = FootstepTwinMirrorGruActor(gru_hidden_dim=64, mlp_hidden_dims=(256, 256, 128)).train()
  # Verify obs_normalizer stays frozen in eval() even when gru_actor.train() is called
  assert gru_actor.training and gru_actor.actor.rnn.training
  assert not gru_actor.actor.obs_normalizer.training

  # Verify bit-identical parity against reference FootstepActor(twin_mirror=True)
  cfg_kwargs = {
    k: v for k, v in asdict(FootstepModelCfg(twin_mirror=True)).items() if k in inspect.signature(FootstepActor).parameters
  }
  ref = FootstepActor(TensorDict({"actor": torch.zeros(1, 79)}, batch_size=[1]), {"actor": ["actor"]}, "actor", 15, **cfg_kwargs)
  gru_actor.load_ppo_actor_state_dict(ref.state_dict())
  gru_actor.set_actuator_metadata(
    default_joint_pos=gru_actor.default_joint_pos,
    action_scale=torch.full((ACTION_DIM,), 0.4),
    stiffness=torch.full((ACTION_DIM,), 150.0),
    damping=torch.full((ACTION_DIM,), 5.0),
    effort_limit=torch.full((ACTION_DIM,), 88.0),
  )
  ref.load_state_dict(gru_actor.actor.state_dict(), strict=False)
  ref.eval()

  rng = torch.Generator().manual_seed(20261114)
  raw_seq = torch.randn(4, 5, 79, generator=rng) * 0.2
  raw_seq[..., :29] += gru_actor.default_joint_pos
  raw_seq[..., 70] = torch.rand(4, 5, generator=rng) * (2.0 * math.pi)
  raw_seq[..., 71] = 1.2
  raw_seq[..., 78] = (torch.rand(4, 5, generator=rng) - 0.5) * 0.6

  h = torch.zeros(5, gru_actor.hidden_state_dim)
  ref.reset()
  for t in range(4):
    obs_t = raw_seq[t]
    act_wm, h = gru_actor.step(obs_t, h)
    with torch.no_grad():
      act_ref = ref(TensorDict({"actor": obs_t}, batch_size=[5]))
    torch.testing.assert_close(act_wm, act_ref, rtol=1e-6, atol=1e-6)

  # Verify exact sagittal reflection equivariance over multi-step trajectories
  q_perm, q_sign = _pair_joints(list(WHOLE_BODY_JOINTS))
  act_perm, act_sign = _pair_joints(list(LOWER_BODY_JOINTS))
  q_perm_t = torch.tensor(q_perm, dtype=torch.long)
  q_sign_t = torch.tensor(q_sign, dtype=torch.float32)
  act_perm_t = torch.tensor(act_perm, dtype=torch.long)
  act_sign_t = torch.tensor(act_sign, dtype=torch.float32)
  default_q = gru_actor.default_joint_pos

  h_orig = torch.zeros(5, gru_actor.hidden_state_dim)
  h_mirr = torch.zeros(5, gru_actor.hidden_state_dim)
  for t in range(4):
    obs_t = raw_seq[t]
    obs_m = torch.empty_like(obs_t)
    obs_m[:, :29] = (obs_t[:, :29] - default_q).index_select(-1, q_perm_t) * q_sign_t + default_q
    obs_m[:, 29:58] = obs_t[:, 29:58].index_select(-1, q_perm_t) * q_sign_t
    obs_m[:, 58:61] = obs_t[:, 58:61] * torch.tensor([-1.0, 1.0, -1.0])
    obs_m[:, 61:64] = obs_t[:, 61:64] * torch.tensor([1.0, -1.0, 1.0])
    obs_m[:, 64:67] = obs_t[:, 64:67] * torch.tensor([-1.0, 1.0, -1.0])
    obs_m[:, 67:70] = obs_t[:, 67:70] * torch.tensor([1.0, -1.0, 1.0])
    obs_m[:, 70] = (obs_t[:, 70] + math.pi).remainder(2.0 * math.pi)
    obs_m[:, 71] = obs_t[:, 71]
    obs_m[:, 72:75] = obs_t[:, 75:78] * torch.tensor([1.0, -1.0, -1.0])
    obs_m[:, 75:78] = obs_t[:, 72:75] * torch.tensor([1.0, -1.0, -1.0])
    obs_m[:, 78] = -obs_t[:, 78]
    act_o, h_orig = gru_actor.step(obs_t, h_orig)
    act_m, h_mirr = gru_actor.step(obs_m, h_mirr)
    torch.testing.assert_close(act_m, act_o.index_select(-1, act_perm_t) * act_sign_t, rtol=1e-5, atol=1e-5)

  # Verify ONNX export produces [1, 1, 128] Twin-Mirror GRU deploy contract compatible with FootstepPolicy
  saved = {
    "format": FORMAT,
    "iteration": 12,
    "config": {
      "actor_type": "gru",
      "history_length": 10,
      "actor_hidden_dims": [1024, 512, 256],
      "gru_hidden_dim": 64,
      "gru_mlp_hidden_dims": [256, 256, 128],
    },
    "origin": {"architecture": actor_architecture_metadata(gru_actor)},
    "actor_state_dict": gru_actor.state_dict(),
    "controller_contract": {
      "default_joint_pos_29": gru_actor.default_joint_pos.tolist(),
      "action_scale_15": [0.4] * ACTION_DIM,
    },
  }
  exported = export_onnx(saved, tmp_path / "onnx_gru")
  policy = FootstepPolicy(exported["deploy"])
  assert policy.contract["inputs"]["h_in"] == [1, 1, 128]
  policy.reset()
  obs_np = np.zeros((1, 79), dtype=np.float32)
  obs_np[:, :29] = gru_actor.default_joint_pos.numpy()
  obs_np[:, 63] = obs_np[:, 69] = -1.0
  obs_np[:, 71] = 1.2
  act_np, targets_np = policy.step(obs_np)
  assert act_np.shape == (15,) and targets_np.shape == (15,)
  assert np.isfinite(act_np).all() and np.isfinite(targets_np).all()

  # Verify zero-init residual skip_net + prev_action warm-start
  widened = FootstepTwinMirrorGruActor(
    gru_hidden_dim=64,
    mlp_hidden_dims=(256, 256, 128),
    include_prev_action=True,
    skip_connection=True,
  ).train()
  assert sum(p.numel() for p in widened.parameters()) == 208478
  widened.load_ppo_actor_state_dict(gru_actor.state_dict())

  pa_t = torch.randn(5, ACTION_DIM, generator=rng) * 0.3
  act_narrow, h_narrow = gru_actor.step(raw_seq[0], torch.zeros(5, 128))
  act_wide, h_wide = widened.step(raw_seq[0], torch.zeros(5, 128), pa_t)
  torch.testing.assert_close(act_wide, act_narrow, rtol=1e-6, atol=1e-6)
  torch.testing.assert_close(h_wide, h_narrow, rtol=1e-6, atol=1e-6)

  # First backward pass immediately writes nonzero gradients into GRU prev_action cols & skip_net output head
  loss_w = act_wide.square().sum() + 0.1 * h_wide.square().sum()
  loss_w.backward()
  assert widened.actor.rnn.rnn.weight_ih_l0.grad[:, 82:].abs().sum() > 0
  assert widened.skip_net is not None and widened.skip_net[-1].weight.grad.abs().sum() > 0

  # Verify sagittal equivariance with nonzero weights on prev_action & skip_net
  with torch.no_grad():
    widened.actor.rnn.rnn.weight_ih_l0[:, 82:].normal_(0.0, 0.1)
    widened.skip_net[-1].weight.normal_(0.0, 0.1)
  h_wo = torch.zeros(5, 128)
  h_wm = torch.zeros(5, 128)
  pa_o = pa_t.clone()
  pa_m = pa_o.index_select(-1, act_perm_t) * act_sign_t
  for t in range(4):
    obs_t = raw_seq[t]
    obs_m = torch.empty_like(obs_t)
    obs_m[:, :29] = (obs_t[:, :29] - default_q).index_select(-1, q_perm_t) * q_sign_t + default_q
    obs_m[:, 29:58] = obs_t[:, 29:58].index_select(-1, q_perm_t) * q_sign_t
    obs_m[:, 58:61] = obs_t[:, 58:61] * torch.tensor([-1.0, 1.0, -1.0])
    obs_m[:, 61:64] = obs_t[:, 61:64] * torch.tensor([1.0, -1.0, 1.0])
    obs_m[:, 64:67] = obs_t[:, 64:67] * torch.tensor([-1.0, 1.0, -1.0])
    obs_m[:, 67:70] = obs_t[:, 67:70] * torch.tensor([1.0, -1.0, 1.0])
    obs_m[:, 70] = (obs_t[:, 70] + math.pi).remainder(2.0 * math.pi)
    obs_m[:, 71] = obs_t[:, 71]
    obs_m[:, 72:75] = obs_t[:, 75:78] * torch.tensor([1.0, -1.0, -1.0])
    obs_m[:, 75:78] = obs_t[:, 72:75] * torch.tensor([1.0, -1.0, -1.0])
    obs_m[:, 78] = -obs_t[:, 78]
    pa_o, h_wo = widened.step(obs_t, h_wo, pa_o)
    pa_m, h_wm = widened.step(obs_m, h_wm, pa_m)
    torch.testing.assert_close(pa_m, pa_o.index_select(-1, act_perm_t) * act_sign_t, rtol=1e-5, atol=1e-5)

  saved_wide = {
    **saved,
    "config": {**saved["config"], "gru_include_prev_action": True, "gru_skip_connection": True},
    "origin": {"architecture": actor_architecture_metadata(widened)},
    "actor_state_dict": widened.state_dict(),
  }
  exported_wide = export_onnx(saved_wide, tmp_path / "onnx_gru_wide")
  policy_wide = FootstepPolicy(exported_wide["deploy"])
  assert policy_wide.contract["inputs"]["h_in"] == [1, 1, 143]
  policy_wide.reset()
  act_w1, _ = policy_wide.step(obs_np)
  act_w2, _ = policy_wide.step(obs_np)
  assert np.isfinite(act_w1).all() and np.isfinite(act_w2).all()


def test_twin_mirror_skip_gru_integrator_zero_dc_symmetry_and_onnx(tmp_path: Path):
  from g1_lower_rl.tasks.footstep_tracking.wm.policy import FootstepTwinMirrorSkipGruActor

  actor = FootstepTwinMirrorSkipGruActor(gru_hidden_dim=256, hidden_dims=(512, 256, 128)).train()
  assert sum(p.numel() for p in actor.parameters()) == 621342

  # 1. Upright standing at f=0 has exact zero sagittal asymmetry and < 5e-3 initial pose offset
  zero_f = torch.zeros(3, FRAME_DIM)
  zero_c = encode_command(torch.zeros(3), torch.zeros(3), torch.zeros(3, 2, 3), torch.zeros(3))
  zero_h = torch.zeros(3, actor.hidden_state_dim)
  act0, h0 = actor.step(zero_f, zero_c, zero_h)
  assert float(act0.abs().max()) < 5e-3
  h0_orig, h0_mirr = h0.chunk(2, dim=-1)
  torch.testing.assert_close(h0_orig, h0_mirr, rtol=1e-6, atol=1e-6)

  # 2. Asymmetric alpha_logit perturbation still produces exact sagittal equivariance across multi-step rollouts
  with torch.no_grad():
    actor.alpha_logit.copy_(torch.linspace(-1.5, 1.5, ACTION_DIM))
  q_perm, q_sign = _pair_joints(list(WHOLE_BODY_JOINTS))
  act_perm, act_sign = _pair_joints(list(LOWER_BODY_JOINTS))
  q_perm_t = torch.tensor(q_perm, dtype=torch.long)
  q_sign_t = torch.tensor(q_sign, dtype=torch.float32)
  act_perm_t = torch.tensor(act_perm, dtype=torch.long)
  act_sign_t = torch.tensor(act_sign, dtype=torch.float32)
  default_q = actor.default_joint_pos

  rng = torch.Generator().manual_seed(20261118)
  h_o = torch.zeros(4, actor.hidden_state_dim)
  h_m = torch.zeros(4, actor.hidden_state_dim)
  pa_o = torch.randn(4, ACTION_DIM, generator=rng) * 0.3
  pa_m = pa_o.index_select(-1, act_perm_t) * act_sign_t
  for _ in range(4):
    raw_p = torch.randn(4, 70, generator=rng) * 0.2
    raw_p[:, :29] += default_q
    mir_p = torch.empty_like(raw_p)
    mir_p[:, :29] = (raw_p[:, :29] - default_q).index_select(-1, q_perm_t) * q_sign_t + default_q
    mir_p[:, 29:58] = raw_p[:, 29:58].index_select(-1, q_perm_t) * q_sign_t
    mir_p[:, 58:61] = raw_p[:, 58:61] * torch.tensor([-1.0, 1.0, -1.0])
    mir_p[:, 61:64] = raw_p[:, 61:64] * torch.tensor([1.0, -1.0, 1.0])
    mir_p[:, 64:67] = raw_p[:, 64:67] * torch.tensor([-1.0, 1.0, -1.0])
    mir_p[:, 67:70] = raw_p[:, 67:70] * torch.tensor([1.0, -1.0, 1.0])
    ph = torch.rand(4, generator=rng) * (2.0 * math.pi)
    fr = torch.tensor([0.0, 1.1, 1.5, 0.9])
    fp = torch.randn(4, 2, 3, generator=rng) * 0.15
    fp_m = torch.stack((fp[:, 1] * torch.tensor([1.0, -1.0, -1.0]), fp[:, 0] * torch.tensor([1.0, -1.0, -1.0])), dim=1)
    th = torch.randn(4, generator=rng) * 0.4
    f_o = encode_proprio_frame(raw_p, pa_o, default_q)
    f_m = encode_proprio_frame(mir_p, pa_m, default_q)
    c_o = encode_command(ph, fr, fp, th)
    c_m = encode_command(ph + math.pi, fr, fp_m, -th)
    pa_o, h_o = actor.step(f_o, c_o, h_o)
    pa_m, h_m = actor.step(f_m, c_m, h_m)
    torch.testing.assert_close(pa_m, pa_o.index_select(-1, act_perm_t) * act_sign_t, rtol=1e-5, atol=1e-5)

  # 3. Verify ONNX export ([1, 1, 527] hidden contract) & FootstepPolicy parity
  saved = {
    "format": FORMAT,
    "iteration": 0,
    "config": {
      "actor_type": "skip_gru",
      "history_length": 10,
      "gru_hidden_dim": 256,
      "actor_hidden_dims": [512, 256, 128],
    },
    "origin": {"architecture": actor_architecture_metadata(actor)},
    "actor_state_dict": actor.state_dict(),
    "controller_contract": {
      "default_joint_pos_29": actor.default_joint_pos.tolist(),
      "action_scale_15": [0.4] * ACTION_DIM,
    },
  }
  exported = export_onnx(saved, tmp_path / "onnx_skip_gru")
  policy = FootstepPolicy(exported["deploy"])
  assert policy.contract["inputs"]["h_in"] == [1, 1, 527]
  policy.reset()
  obs_np = np.zeros((1, 79), dtype=np.float32)
  obs_np[:, :29] = actor.default_joint_pos.numpy()
  obs_np[:, 63] = obs_np[:, 69] = -1.0
  obs_np[:, 71] = 1.2
  act_np, targets_np = policy.step(obs_np)
  assert act_np.shape == (15,) and targets_np.shape == (15,)
  assert np.isfinite(act_np).all() and np.isfinite(targets_np).all()


def test_autoregressive_loss_outlier_huber_guard():
  from g1_lower_rl.world_model.model import CONTEXT_DIM, STATE_DIM, pack_inputs
  from g1_lower_rl.world_model.training.objectives import LossConfig, autoregressive_loss
  from g1_lower_rl.world_model.training.replay import ReplayWindow

  wm = PositionWorldModel(PositionWorldModelCfg())
  wm.set_normalization(torch.zeros(145), torch.ones(145), torch.ones(70), torch.ones(6), torch.ones(70))
  b, h = 16, 4
  state = torch.zeros(b, h, STATE_DIM)
  state[..., 3] = 1.0
  next_state = state.clone()
  next_state[..., 41:] += 0.2
  inputs = pack_inputs(state, torch.zeros(b, h, 29), torch.ones(b, h, CONTEXT_DIM))
  foot_force = torch.zeros(b, h, 2, 3)
  env = torch.arange(b).unsqueeze(-1).expand(b, h)
  seq = torch.arange(h).unsqueeze(0).expand(b, h)

  win_clean = ReplayWindow(state=state, inputs=inputs, next_state=next_state, foot_force=foot_force, env=env, sequence=seq)
  wm.zero_grad()
  loss_l2, _ = autoregressive_loss(wm, win_clean, wm.state_scale, wm.force_scale, LossConfig())
  loss_l2.backward()
  grad_clean = torch.nn.utils.clip_grad_norm_(wm.parameters(), float("inf")).item()

  wm.zero_grad()
  loss_hub, _ = autoregressive_loss(wm, win_clean, wm.state_scale, wm.force_scale, LossConfig(outlier_huber_delta=8.0))
  torch.testing.assert_close(loss_hub, loss_l2, rtol=1e-6, atol=1e-6)

  # Inject a 1000-sigma MuJoCo contact-penetration velocity glitch (both step=0 input shock and step=1 target shock)
  state_spike = state.clone()
  state_spike[0, 0, 50] = 1000.0
  inputs_spike = pack_inputs(state_spike, torch.zeros(b, h, 29), torch.ones(b, h, CONTEXT_DIM))
  next_spike = next_state.clone()
  next_spike[0, 1, 50] = 1000.0
  win_spike = ReplayWindow(state=state_spike, inputs=inputs_spike, next_state=next_spike, foot_force=foot_force, env=env, sequence=seq)

  wm.zero_grad()
  l2_spike, _ = autoregressive_loss(wm, win_spike, wm.state_scale, wm.force_scale, LossConfig())
  l2_spike.backward()
  grad_l2 = torch.nn.utils.clip_grad_norm_(wm.parameters(), float("inf")).item()

  wm.zero_grad()
  hub_spike, _ = autoregressive_loss(wm, win_spike, wm.state_scale, wm.force_scale, LossConfig(outlier_huber_delta=8.0))
  hub_spike.backward()
  grad_hub = torch.nn.utils.clip_grad_norm_(wm.parameters(), float("inf")).item()

  assert grad_l2 > 100.0 * grad_hub
  assert grad_hub < 1.2 * grad_clean


def test_advance_queue_schedule_stopping_uses_terminal_feet():
  mj_model = mujoco.MjModel.from_xml_path(
    str(ROOT / "g1_lower_rl/assets/g1_gloria/xmls/scene_g1_with_dual_gloria_m.xml")
  )
  fk = TorchKinematics(StaticKinematicTemplate.from_mujoco(mj_model)).to(dtype=torch.float32)
  joint_ids = [mj_model.joint(f"{name}").id for name in JOINT_NAMES]
  hard_limits = torch.tensor(mj_model.jnt_range[joint_ids], dtype=torch.float32)
  task = FootstepWmTask(
    kinematics=fk,
    default_position=torch.zeros(29),
    soft_limits=hard_limits,
    hard_limits=hard_limits,
    lower_kp=torch.full((ACTION_DIM,), 150.0),
    lower_kd=torch.full((ACTION_DIM,), 5.0),
    lower_effort_limit=torch.full((ACTION_DIM,), 88.0),
    action_scale=torch.full((ACTION_DIM,), 0.4),
  )

  # Batch of 2: row 0 WALKING (mode=0), row 1 STOPPING (mode=2) crossing left-foot touchdown center
  c0 = float(task.centers[0])
  phase = torch.tensor([c0 - 0.05, c0 - 0.05])
  freq = torch.tensor([1.2, 1.2])
  mode = torch.tensor([0, 2], dtype=torch.long)
  queue = torch.tensor(
    [[[0.25, 0.11, 0.0], [0.50, -0.11, 0.0], [0.75, 0.11, 0.0], [1.00, -0.11, 0.0]]] * 2,
    dtype=torch.float32,
  )
  terminal = torch.tensor([[[1.00, 0.11, 0.0], [1.00, -0.11, 0.0]]] * 2, dtype=torch.float32)
  supports = torch.zeros(2, 2, 3)
  targets = queue[:, :2].clone()
  _, _, _, _, _, _, next_queue, *_ = task.advance_queue_schedule(
    phase=phase,
    frequency=freq,
    mode=mode,
    elapsed=torch.zeros(2),
    req_freq=freq,
    stop_init_freq=freq,
    stop_ramp_start=torch.full((2,), 5.0),
    stop_phase=torch.full((2,), 20.0),
    start_dur=torch.full((2,), 0.4),
    stop_dur=torch.full((2,), 0.4),
    terminal_feet=terminal,
    targets=targets,
    supports=supports,
    queue=queue,
    first_side=torch.zeros(2, dtype=torch.long),
    pending_anchor=torch.full((2,), -1, dtype=torch.long),
  )
  # Row 0 (WALKING) extrapolates x = 0.75 + (1.00 - 0.50) = 1.25; Row 1 (STOPPING) appends terminal_feet[0] = 1.00!
  torch.testing.assert_close(next_queue[0, 3], torch.tensor([1.25, 0.11, 0.0]))
  torch.testing.assert_close(next_queue[1, 3], torch.tensor([1.00, 0.11, 0.0]))


