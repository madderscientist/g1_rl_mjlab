from types import SimpleNamespace

import numpy as np
import pytest
import torch
from mjlab.tasks.registry import load_env_cfg

import g1_lower_rl.tasks
from g1_lower_rl.tasks.domain_randomization import RandomizeLinkMass, make_link_mass_event


def make_fixture():
  mass = np.array([0.0, 4.0, 0.5, 2.0])
  inertia = np.arange(12, dtype=float).reshape(4, 3) / 1000
  model = SimpleNamespace(
    body_mass=torch.tensor(mass, dtype=torch.float32).repeat(4, 1),
    body_inertia=torch.tensor(inertia, dtype=torch.float32).repeat(4, 1, 1),
  )
  cfg = make_link_mass_event()
  env = SimpleNamespace(
    device="cpu", num_envs=4,
    scene={"robot": SimpleNamespace(indexing=SimpleNamespace(body_ids=torch.tensor([3, 1, 2], dtype=torch.int32)))},
    sim=SimpleNamespace(model=model, mj_model=SimpleNamespace(body_mass=mass, body_inertia=inertia)),
  )
  term = RandomizeLinkMass(cfg, env)
  model.body_mass[:, 2] += torch.tensor([0.1, 0.5, 1.0, 2.0])
  return env, cfg, term


def test_partial_reset_preserves_payload_and_never_accumulates():
  env, cfg, term = make_fixture()
  initial_mass = env.sim.model.body_mass.clone()
  initial_inertia = env.sim.model.body_inertia.clone()
  selected = torch.tensor([1, 3])
  untouched = torch.tensor([0, 2])
  term(env, selected, cfg.params["asset_cfg"], (1.05, 1.05))
  expected_mass = initial_mass.clone()
  expected_inertia = initial_inertia.clone()
  expected_mass[selected] += torch.tensor(env.sim.mj_model.body_mass, dtype=torch.float32) * 0.05
  expected_inertia[selected, 1:] *= 1.05
  torch.testing.assert_close(env.sim.model.body_mass, expected_mass)
  torch.testing.assert_close(env.sim.model.body_inertia, expected_inertia)
  for _ in range(100):
    term(env, selected, cfg.params["asset_cfg"], (1.05, 1.05))
  torch.testing.assert_close(env.sim.model.body_mass, expected_mass)
  torch.testing.assert_close(env.sim.model.body_mass[untouched], initial_mass[untouched], rtol=0, atol=0)
  term(env, None, cfg.params["asset_cfg"], (1.0, 1.0))
  torch.testing.assert_close(env.sim.model.body_mass, initial_mass, rtol=0, atol=0)
  torch.testing.assert_close(env.sim.model.body_inertia, initial_inertia, rtol=0, atol=0)


def test_independent_uniform_samples_are_bounded_and_scale_inertia_together():
  torch.manual_seed(42)
  env, cfg, term = make_fixture()
  draws = []
  for _ in range(100):
    term(env, None, **cfg.params)
    scale = (env.sim.model.body_mass[:, term.body_ids] - term.mass_offset) / term.nominal_mass
    assert (scale >= 0.95 - 1e-6).all() and (scale <= 1.05 + 1e-6).all()
    actual_inertia = env.sim.model.body_inertia[:, term.body_ids]
    torch.testing.assert_close(actual_inertia, term.nominal_inertia * scale.unsqueeze(-1))
    draws.append(scale.flatten())
  values = torch.cat(draws)
  assert abs(values.mean().item() - 1.0) < 0.003
  assert 0.026 < values.std().item() < 0.032
  assert not torch.equal(draws[0], draws[1])
  assert torch.unique(draws[0]).numel() == draws[0].numel()


@pytest.mark.parametrize("scale_range", [(0, 1), (1.05, 0.95), (0.95, float("inf")), (float("nan"), 1)])
def test_invalid_ranges_rejected(scale_range):
  env, cfg, term = make_fixture()
  with pytest.raises(ValueError):
    term(env, None, cfg.params["asset_cfg"], scale_range)


@pytest.mark.parametrize("play", [False, True])
@pytest.mark.parametrize("task", [
  "G1-Gloria-LowerBody-Flat", "G1-Gloria-LowerBody-Flat-GRU", "G1-Gloria-Stand",
  "G1-Gloria-FootstepTracking", "G1-Gloria-MotionTracking",
])
def test_every_task_uses_shared_reset_randomization(task, play):
  cfg = load_env_cfg(task, play=play)
  term = cfg.events["link_mass"]
  assert term.func is RandomizeLinkMass
  assert term.mode == "reset" and term.min_step_count_between_reset == 0
  assert term.params["scale_range"] == (0.95, 1.05)
  assert term.params["asset_cfg"].name == "robot"
  assert "link_inertia" not in cfg.events
  assert sum(event.func is RandomizeLinkMass for event in cfg.events.values()) == 1


@pytest.mark.parametrize("payload_source", ["lower_body", "motion_tracking"])
@torch.inference_mode()
def test_real_environment_reset_recomputes_constants_and_holds_mass(payload_source):
  from mjlab.envs import ManagerBasedRlEnv

  cfg = load_env_cfg("G1-Gloria-LowerBody-Flat", play=True)
  cfg.scene.num_envs = 4
  cfg.seed = 42
  cfg.terminations = {}
  if payload_source == "motion_tracking":
    cfg.events["payload_mass"] = load_env_cfg("G1-Gloria-MotionTracking").events["payload"]
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env = ManagerBasedRlEnv(cfg, device=device)
  try:
    model = env.sim.model
    initial_mass = model.body_mass.clone()
    initial_com = model.body_ipos.clone()
    initial_iquat = model.body_iquat.clone()
    nominal_mass = torch.tensor(env.sim.mj_model.body_mass, device=device, dtype=model.body_mass.dtype)
    initial_offset = initial_mass - nominal_mass
    env.reset()
    term = env.event_manager.get_term_cfg("link_mass").func
    assert len(term.body_ids) == len(env.scene["robot"].body_names) == 44
    torch.testing.assert_close(term.mass_offset, initial_offset[:, term.body_ids], rtol=0, atol=0)
    assert (term.mass_offset > 0).any()
    selected = torch.tensor([1, 3], device=device)
    untouched = torch.tensor([0, 2], device=device)
    for _ in range(5):
      before_mass = model.body_mass.clone()
      before_inertia = model.body_inertia.clone()
      before_invweight = model.dof_invweight0.clone()
      for _ in range(3):
        env.step(torch.zeros((4, env.action_manager.total_action_dim), device=device))
      torch.testing.assert_close(model.body_mass[:], before_mass, rtol=0, atol=0)
      torch.testing.assert_close(model.body_inertia[:], before_inertia, rtol=0, atol=0)
      env.reset(env_ids=selected)
      assert not torch.equal(model.body_mass[selected], before_mass[selected])
      assert not torch.equal(model.dof_invweight0[selected], before_invweight[selected])
      torch.testing.assert_close(model.body_mass[untouched], before_mass[untouched], rtol=0, atol=0)
      torch.testing.assert_close(model.body_inertia[untouched], before_inertia[untouched], rtol=0, atol=0)
      scale = (model.body_mass[:, term.body_ids] - initial_offset[:, term.body_ids]) / term.nominal_mass
      assert (scale >= 0.95 - 1e-6).all() and (scale <= 1.05 + 1e-6).all()
      torch.testing.assert_close(model.body_inertia[:, term.body_ids], term.nominal_inertia * scale.unsqueeze(-1))
      root_id = env.scene["robot"].indexing.body_ids[0]
      torch.testing.assert_close(model.body_subtreemass[:, root_id], model.body_mass[:, term.body_ids].sum(-1))
      torch.testing.assert_close(model.body_ipos[:], initial_com, rtol=0, atol=0)
      torch.testing.assert_close(model.body_iquat[:], initial_iquat, rtol=0, atol=0)
      assert torch.isfinite(env.sim.data.qpos[:]).all() and torch.isfinite(env.sim.data.qvel[:]).all()
  finally:
    env.close()