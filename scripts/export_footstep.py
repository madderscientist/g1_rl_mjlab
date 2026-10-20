"""Export and validate a theta-enabled recurrent footstep deployment policy."""

import argparse
import copy
import hashlib
import inspect
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnx
import torch
from tensordict import TensorDict

from g1_lower_rl.footstep_deploy import FootstepPolicy
from g1_lower_rl.rl.footstep_model import FootstepActor, FootstepModelCfg, export_footstep_policy


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("checkpoint", type=Path)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--simulation-steps", type=int, default=0)
  args = parser.parse_args()
  if args.simulation_steps < 0:
    parser.error("Simulation steps must be nonnegative")
  torch.set_num_threads(1)
  checkpoint = args.checkpoint.resolve(strict=True)
  digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
  saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
  assert saved["infos"]["actor_raw_obs_dim"] == 79
  config = {name: value for name, value in asdict(FootstepModelCfg()).items()
    if name in inspect.signature(FootstepActor).parameters}
  observations = TensorDict({"actor": torch.zeros(1, 79)}, batch_size=[1])
  actor = FootstepActor(observations, {"actor": ["actor"]}, "actor", 15, **config).eval()
  actor.load_state_dict(saved["actor_state_dict"], strict=True)
  for name, value in actor.state_dict().items():
    assert torch.isfinite(value).all()
    torch.testing.assert_close(value, saved["actor_state_dict"][name], rtol=0, atol=0)
  destination = args.output_dir / f"model_{saved['iter']}.onnx"
  if destination.exists():
    raise FileExistsError(f"Refusing to overwrite {destination}")
  export_footstep_policy(actor, destination)
  onnx.checker.check_model(onnx.load(destination), full_check=True)
  policy = FootstepPolicy(destination)
  contract = json.loads(destination.with_suffix(".contract.json").read_text())
  assert contract == policy.contract == actor.deployment_contract()
  reference = actor.as_onnx().cpu().eval()
  maxima = {"actions": 0., "hidden": 0.}

  def check(raw, hidden):
    expected, next_hidden, *_ = reference(torch.from_numpy(raw), hidden)
    actual, targets = policy.step(raw)
    np.testing.assert_allclose(actual, expected.numpy()[0], rtol=1e-4, atol=3e-5)
    np.testing.assert_allclose(policy.hidden_state, next_hidden.numpy(), rtol=1e-4, atol=3e-5)
    np.testing.assert_array_equal(targets, policy.default_pos + policy.action_scale * actual)
    maxima["actions"] = max(maxima["actions"], float(np.abs(actual - expected.numpy()[0]).max()))
    maxima["hidden"] = max(maxima["hidden"], float(np.abs(policy.hidden_state - next_hidden.numpy()).max()))
    return next_hidden

  generator = np.random.default_rng(42)
  hidden = torch.zeros(1, 1, 64)
  with torch.inference_mode():
    for step in range(1024):
      if step % 128 == 0:
        policy.reset()
        hidden.zero_()
      raw = generator.normal(0., .1, (1, 79)).astype(np.float32)
      raw[:, :29] += actor.encoder.default_joint_pos.numpy()
      raw[:, 63] = raw[:, 69] = -1.
      raw[:, 70] = generator.uniform(0., 2 * np.pi)
      raw[:, 71] = 0. if step % 3 == 0 else generator.uniform(.8, 1.8)
      raw[:, 78] = generator.uniform(-np.pi, np.pi)
      hidden = check(raw, hidden)

  simulations = []
  if args.simulation_steps:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends
    from g1_lower_rl.tasks.footstep_tracking.commands import FootstepCommandCfg
    from g1_lower_rl.tasks.footstep_tracking.theta_stage import make_theta_stage_env_cfg

    configure_torch_backends()
    for rank, profile in enumerate(("precision", "step-episode")):
      cfg = make_theta_stage_env_cfg(profile, saved["infos"]["theta_stage"], rank, play=True,
        precision_stage=saved["infos"].get("precision_stage") if rank == 0 else None)
      cfg.scene.num_envs = 1
      cfg.seed = 42
      footsteps = cfg.commands["footsteps"]
      assert isinstance(footsteps, FootstepCommandCfg)
      footsteps.compile_backend = False
      env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device="cuda:0"), clip_actions=None)
      try:
        env.unwrapped.common_step_counter = saved["infos"]["task_environment_counters"][rank]
        model = copy.deepcopy(actor).to("cuda:0").eval()
        with torch.inference_mode():
          env.reset()
          policy.reset()
          hidden = torch.zeros(1, 1, 64)
          resets = 0
          for step in range(args.simulation_steps):
            reset = env.episode_length_buf == 0
            model.reset(reset)
            if reset.any():
              policy.reset()
              hidden.zero_()
              resets += 1
            obs = env.get_observations()
            hidden = check(obs["actor"].cpu().numpy(), hidden)
            env.step(model(obs, stochastic_output=False))
            assert torch.isfinite(env.unwrapped.scene["robot"].data.joint_pos).all()
        simulations.append({"profile": profile, "steps": args.simulation_steps, "resets": resets})
        print("SIMULATION_ONNX_PARITY_PASS", profile, args.simulation_steps, flush=True)
      finally:
        env.close()
  assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
  receipt = {"status": "passed", "iteration": saved["iter"], "checkpoint": str(checkpoint),
    "checkpoint_sha256": digest, "onnx_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    "contract_sha256": hashlib.sha256(destination.with_suffix(".contract.json").read_bytes()).hexdigest(),
    "actor_exactly_loaded": True, "contract_matches": True, "synthetic_recurrent_steps": 1024,
    "maximum_absolute_errors": maxima, "simulations": simulations,
    "hardware_validation": "not performed; numerical parity is not hardware safety approval"}
  (args.output_dir / "validation.json").write_text(json.dumps(receipt, indent=2) + "\n")
  print("FOOTSTEP_EXPORT_VALIDATED", json.dumps(receipt), flush=True)


if __name__ == "__main__":
  main()