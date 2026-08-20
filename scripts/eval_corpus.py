"""逐条动作评测已训练的 GMT 策略。

对语料里的每一条动作，从第 0 帧起放，统计：能跟住多久、跟到哪一步失败、跟踪误差多大。
训练日志里的「平均回合长度」是所有动作混在一起的均值，看不出是「每条都跟一半」还是
「几条跟满、几条一开始就废」——而这两种情况对能不能上机是完全不同的结论。

``--tilt/--joint/--height-lo/--height-hi`` 给初始状态加扰动，用来测「非直立启动能不能
站住」。默认全 0 即干净启动。
"""

from __future__ import annotations

from pathlib import Path

import torch
import tyro

import g1_lower_rl.tasks  # noqa: F401  注册任务


def main(
  checkpoint: str,
  motion_dir: str | None = None,
  envs_per_motion: int = 16,
  device: str = "cuda:0",
  max_seconds: float = 200.0,
  tilt: float = 0.0,
  joint: float = 0.0,
  height_lo: float = 0.0,
  height_hi: float = 0.0,
) -> None:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  cfg = load_env_cfg("G1-Gloria-MotionTracking", play=True)
  agent_cfg = load_rl_cfg("G1-Gloria-MotionTracking")
  if motion_dir is not None:
    cfg.commands["motion"].motion_dir = motion_dir

  # play 会清空初始扰动，这里按需放回。默认全 0 = 干净启动，测跟踪上限；
  # 带上扰动才是测「非直立启动能不能站住」。两者要一起看：只看前者会低估鲁棒性训练，
  # 只看后者会掩盖跟踪能力的退化。
  if tilt or joint or height_lo or height_hi:
    mc = cfg.commands["motion"]
    mc.pose_range = {
      "roll": (-tilt, tilt),
      "pitch": (-tilt, tilt),
      "yaw": (-tilt, tilt),
      "z": (height_lo, height_hi),
    }
    mc.joint_position_range = (-joint, joint)

  # 先建一个最小环境问出动作条数，再按条数定环境数。
  cfg.scene.num_envs = 1
  probe = ManagerBasedRlEnv(cfg=cfg, device=device)
  n_motions = probe.command_manager.get_term("motion").motion.num_motions
  names = list(probe.command_manager.get_term("motion").motion.names)
  probe.close()

  cfg.scene.num_envs = n_motions * envs_per_motion
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  from dataclasses import asdict

  runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), log_dir=None, device=device)
  runner.load(checkpoint)
  policy = runner.get_inference_policy(device=device)

  cmd = env.command_manager.get_term("motion")
  assign = torch.arange(n_motions, device=device).repeat_interleave(envs_per_motion)

  obs, _ = wrapped.reset()
  cmd.motion_ids[:] = assign
  cmd.phase[:] = 0
  cmd._write_state_from_motion(torch.arange(cfg.scene.num_envs, device=device))

  n = cfg.scene.num_envs
  alive = torch.ones(n, dtype=torch.bool, device=device)
  survived = torch.zeros(n, dtype=torch.long, device=device)
  err_sum = torch.zeros(n, device=device)
  err_cnt = torch.zeros(n, device=device)

  total_frames = cmd.motion.num_frames[assign]
  steps = min(int(max_seconds * 50), int(total_frames.max().item()))

  with torch.inference_mode():
    for _ in range(steps):
      actions = policy(obs)
      obs, _, dones, _ = wrapped.step(actions)

      err = torch.norm(cmd.body_pos_relative_w - cmd.robot_body_pos_w, dim=-1).mean(-1)
      err_sum += torch.where(alive, err, torch.zeros_like(err))
      err_cnt += alive.float()
      survived += alive.long()

      alive &= ~dones.bool()
      if not alive.any():
        break

  # 跟满整条才算完成。相位回绕不能用来判定——失败复位同样会让相位变小。
  finished = survived >= (total_frames - 1)

  print(f"\n检查点: {Path(checkpoint).name}   每条动作 {envs_per_motion} 个环境\n")
  print(f"{'动作':<28}{'总帧':>7}{'存活帧':>8}{'存活秒':>8}{'跟满%':>8}{'体位误差m':>10}")
  print("-" * 70)
  for i, name in enumerate(names):
    sl = slice(i * envs_per_motion, (i + 1) * envs_per_motion)
    surv = survived[sl].float().mean().item()
    tot = float(total_frames[sl][0].item())
    err = (err_sum[sl] / err_cnt[sl].clamp(min=1)).mean().item()
    print(f"{name:<28}{tot:>7.0f}{surv:>8.0f}{surv / 50:>8.1f}{100 * surv / tot:>7.0f}%{err:>10.3f}")
  print("-" * 70)
  print(
    f"{'合计':<28}{'':>7}{survived.float().mean().item():>8.0f}"
    f"{survived.float().mean().item() / 50:>8.1f}"
    f"{finished.float().mean().item() * 100:>7.0f}%"
    f"{(err_sum / err_cnt.clamp(min=1)).mean().item():>10.3f}"
  )
  # 存活分布极偏（均值/中位差过 3 倍），只报均值会把「几条跑满拉高全场」看成普遍变好。
  secs = survived.float() / 50
  print(f"中位存活 {secs.median().item():.1f} s   存活>5s {(secs > 5).float().mean().item() * 100:.1f}%")


if __name__ == "__main__":
  tyro.cli(main)
