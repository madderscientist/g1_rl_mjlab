"""无头定量评估一个训练好的下肢策略。

    python scripts/eval_policy.py [checkpoint.pt] [no-disturb] [fixed-height]

不给检查点就取 ``logs/rsl_rl/<experiment>`` 下最新的一个。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import torch  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402

import g1_lower_rl.tasks  # noqa: F401,E402  注册任务
from g1_lower_rl.rl import latest_checkpoint, load_trained_runner  # noqa: E402
from g1_lower_rl.tasks import DEFAULT_TASK  # noqa: E402
from g1_lower_rl.tasks.lower_body.mdp import height_above_feet  # noqa: E402

TASK = DEFAULT_TASK
STEPS = 1000
NUM_ENVS = 256
LOG_ROOT = Path("logs/rsl_rl")


def main() -> None:
  args = sys.argv[1:]
  flags = {a for a in args if not a.endswith(".pt")}
  ckpt_arg = next((a for a in args if a.endswith(".pt")), None)

  configure_torch_backends()
  agent_cfg = load_rl_cfg(TASK)
  ckpt = (
    Path(ckpt_arg)
    if ckpt_arg
    else latest_checkpoint(LOG_ROOT / agent_cfg.experiment_name)
  )
  print("checkpoint:", ckpt, "flags:", sorted(flags) or "-")

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env_cfg = load_env_cfg(TASK, play=True)
  env_cfg.scene.num_envs = NUM_ENVS
  env_cfg.episode_length_s = 20.0  # 让 time_out 结束 episode，存活时长才可测
  if "no-disturb" in flags:
    for name in ("arm_torque", "body_impulse", "payload_mass", "arm_pose_drift"):
      env_cfg.events.pop(name, None)
  if "fixed-height" in flags:
    env_cfg.commands["height"].ranges.height = (0.79, 0.79)  # type: ignore[attr-defined]

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = load_trained_runner(TASK, wrapped, agent_cfg, ckpt, device)
  policy = runner.get_inference_policy(device=device)

  robot = env.scene["robot"]
  site_ids, _ = robot.find_sites(["left_foot", "right_foot"])
  sensor = env.scene["feet_ground_contact"]
  obs = wrapped.get_observations()

  vel_err = height_err = air_time = 0.0
  falls = 0
  h_by_band = {b: [0.0, 0] for b in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75)}
  vx_bins = {b: [0.0, 0] for b in (-0.5, -0.25, 0.0, 0.25, 0.5, 0.75)}
  with torch.inference_mode():
    for _ in range(STEPS):
      obs, _, dones, extras = wrapped.step(policy(obs))
      twist = env.command_manager.get_command("twist")
      h_cmd = env.command_manager.get_command("height")[:, 0]
      h = height_above_feet(robot, site_ids)
      v = robot.data.root_link_lin_vel_b
      vel_err += (twist[:, :2] - v[:, :2]).norm(dim=1).mean().item()
      height_err += (h_cmd - h).abs().mean().item()
      air_time += sensor.data.current_air_time.max(dim=1).values.mean().item()
      falls += int((dones.bool() & ~extras["time_outs"]).sum().item())
      for lo in h_by_band:
        m = (h_cmd >= lo) & (h_cmd < lo + 0.05)
        if m.any():
          h_by_band[lo][0] += (h_cmd[m] - h[m]).abs().sum().item()
          h_by_band[lo][1] += int(m.sum().item())
      for lo in vx_bins:
        m = (twist[:, 0] >= lo) & (twist[:, 0] < lo + 0.25)
        if m.any():
          vx_bins[lo][0] += v[m, 0].sum().item()
          vx_bins[lo][1] += int(m.sum().item())

  print(f"mean |v_xy error|  : {vel_err / STEPS:.3f} m/s")
  print(f"mean |height error|: {height_err / STEPS:.4f} m")
  print(f"mean swing air time: {air_time / STEPS:.3f} s  (0 = never lifts a foot)")
  print(f"falls per env-min  : {falls / NUM_ENVS / (STEPS * env.step_dt / 60):.2f}")
  for lo, (s, n) in sorted(h_by_band.items()):
    if n:
      print(f"  height cmd [{lo:.2f},{lo + 0.05:.2f}) : err {s / n:.4f} m  (n={n})")
  for lo, (s, n) in sorted(vx_bins.items()):
    if n:
      print(f"  vx cmd [{lo:+.2f},{lo + 0.25:+.2f}) : achieved {s / n:+.3f} m/s (n={n})")
  env.close()


if __name__ == "__main__":
  main()
