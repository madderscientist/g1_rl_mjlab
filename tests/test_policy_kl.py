from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from g1_lower_rl.tasks.motion_tracking.training.policy_kl import (
  NativePolicyRollout,
  constrain_policy_kl,
  policy_kl,
)
from g1_lower_rl.tasks.motion_tracking.training.position_runner import PositionTrainer, PositionTrainingConfig


def _rollout(train_envs=4, total_envs=6, steps=3):
  rollout = NativePolicyRollout(train_envs)
  for _ in range(steps):
    mean = torch.zeros(total_envs, 29)
    rollout.append(
      history=torch.randn(total_envs, 10, 93),
      reference=torch.randn(total_envs, 798),
      mean=mean,
    )
  return rollout


def test_kl_only_rollout_needs_no_rewards_values_or_actions_and_excludes_validation():
  rollout = NativePolicyRollout(2)
  history = torch.zeros(3, 10, 93)
  history[-1].fill_(float("nan"))
  rollout.append(history=history, reference=torch.zeros(3, 798), mean=torch.zeros(3, 29))
  history[0].fill_(7)
  batch = rollout.finish_policy()
  assert len(batch) == 2 and torch.isfinite(batch.history).all() and batch.history.sum() == 0
  assert not hasattr(batch, "returns") and not hasattr(batch, "features")
  assert policy_kl(lambda history, reference: torch.ones(len(history), 29) * 0.01, batch, 0.1, 1) > 0
  with pytest.raises(ValueError, match="fresh"):
    rollout.finish_policy()


def test_native_kl_only_training_does_not_run_ppo_or_update_real_critic(monkeypatch):
  trainer = PositionTrainer.__new__(PositionTrainer)
  trainer.cfg = PositionTrainingConfig("wm", "replica", "output", iterations=1, native_actor_kl=True)
  trainer.cfg.validate()
  trainer.device, trainer.rank, trainer.world_size = torch.device("cpu"), 1, 1
  trainer.actor = torch.nn.Linear(1, 29)
  torch.nn.init.zeros_(trainer.actor.weight)
  torch.nn.init.zeros_(trainer.actor.bias)
  trainer.optimizers = {"actor": torch.optim.Adam(trainer.actor.parameters())}
  trainer.iteration = trainer.scheduled_controls = trainer.scheduled_training_transitions = trainer.scheduled_actor_starts = 0
  trainer.optimizer_updates = {"actor": 0, "critic": 0, "world_model": 0}
  trainer.preflight = {}
  trainer.native_sums = torch.zeros(8)
  trainer.replay = SimpleNamespace(train_envs=2, candidates=lambda *args, **kwargs: torch.ones(1))
  trainer.collector = SimpleNamespace(counters={})
  trainer.last_save = float("inf")
  trainer.progress = lambda *args, **kwargs: None
  trainer.validate_model = dict
  trainer.audit_imagination = trainer.save = lambda: None
  trainer.stop_requested = lambda: False
  trainer.required_controls = lambda: 1
  trainer.policy_mean = lambda history, reference: trainer.actor(history[:, 0, :1])
  assert not hasattr(PositionTrainer, "fit_native_ppo")

  def collect(count, *, audit=False, policy_rollout=None):
    if policy_rollout is not None:
      policy_rollout.append(history=torch.zeros(2, 10, 93), reference=torch.zeros(2, 798), mean=torch.zeros(2, 29))

  def fit_model():
    trainer.optimizer_updates["world_model"] += 10
    return {"wm_loss": 1.0}

  def improve_policy():
    trainer.optimizer_updates["actor"] += 1
    with torch.no_grad():
      trainer.actor.bias.add_(0.001)
    return {}, (torch.zeros(1),)

  def fit_critic(batch):
    trainer.optimizer_updates["critic"] += 20
    return {"critic_loss": 1.0}

  trainer.collect, trainer.fit_model = collect, fit_model
  trainer.improve_policy, trainer.fit_critic = improve_policy, fit_critic
  import io

  stream = io.StringIO()
  from contextlib import nullcontext

  class MetricsOutput:
    def __truediv__(self, name):
      return self

    def open(self, mode):
      return nullcontext(stream)

  trainer.output = MetricsOutput()
  monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 0)
  trainer.run()
  assert trainer.optimizer_updates == {"world_model": 10, "actor": 1, "critic": 20}
  import json

  metrics = json.loads(stream.getvalue())
  assert metrics["policy/kl"] <= 0.01
  assert "ppo/policy_loss" not in metrics and "timing/ppo_seconds" not in metrics


def test_exact_gaussian_kl_and_backtracking_preserve_adam_and_restore_on_rejection():
  actor = torch.nn.Linear(1, 29, bias=False)
  torch.nn.init.zeros_(actor.weight)
  optimizer = torch.optim.Adam(actor.parameters())
  batch = _rollout().finish_policy()
  predict = lambda history, reference: actor(torch.ones(len(history), 1))
  before = [p.detach().clone() for p in actor.parameters()]
  adam_before = copy.deepcopy(optimizer.state_dict())
  with torch.no_grad():
    actor.weight.fill_(0.1)
  expected = 0.5 * 29
  torch.testing.assert_close(policy_kl(predict, batch, 0.1, 5), torch.tensor(expected, dtype=torch.float64))
  result = constrain_policy_kl(actor, optimizer, before, adam_before, lambda: policy_kl(predict, batch, 0.1, 5), 0.01)
  assert result["kl"] <= 0.01 and 0 < result["step_fraction"] < 1
  result = constrain_policy_kl(actor, optimizer, before, adam_before, lambda: torch.tensor(1.0), 0.01)
  assert result["rejected"] == 1
  torch.testing.assert_close(actor.weight, before[0])
  assert optimizer.state_dict() == adam_before


def test_imagination_uses_sampled_raw_action_for_history_and_rate_penalty(monkeypatch):
  from g1_lower_rl.tasks.motion_tracking.training import position_runner

  trainer = PositionTrainer.__new__(PositionTrainer)
  trainer.cfg = PositionTrainingConfig("wm", "replica", "output")
  trainer.actor = lambda observations, stochastic_output: torch.ones(2, 29)
  trainer.scale, trainer.target_limits, trainer.target_limited = torch.ones(29), torch.zeros(29, 2), torch.zeros(29).bool()
  trainer.default_position = torch.zeros(29)
  trainer.forward_model = lambda state, target, context: SimpleNamespace(state=state)
  trainer.task = SimpleNamespace(termination=lambda state, reference: torch.zeros(2))
  captured = {}

  def reward(state, reference, action, previous):
    captured["action"] = action
    return torch.zeros(2)

  trainer.reward_fn = reward
  monkeypatch.setattr(position_runner, "residual_target", lambda action, *args: action)
  state = torch.zeros(2, 71)
  state[:, 3] = 1
  _, history, _, _ = trainer._step(
    state,
    torch.zeros(2, 10, 93),
    torch.zeros(2, 48),
    torch.zeros(2, 29),
    state,
    torch.zeros(2, 798),
    torch.zeros(2, 93),
    torch.full((2, 29), 0.25),
  )
  expected = torch.full((2, 29), 1.25)
  torch.testing.assert_close(captured["action"], expected)
  torch.testing.assert_close(history[:, -1, -29:], expected)
