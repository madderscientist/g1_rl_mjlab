"""按 base policy 的完成率把语料切成 mastered / challenging 两堆（Extreme-RGMT 第 IV-C 节）。

把长于 ``clip_seconds`` 的动作切成等长片段，每段从首帧起跑 ``rollouts`` 次随机 rollout，
能跟到片段末尾算一次成功；完成率 ≥ ``mastered_threshold`` 的进 mastered，其余进 challenging。

Stage II 用这两堆做非对称训练：consolidation 环境在 mastered 上均匀采样约束策略漂移，
acquisition 环境在 challenging 上自适应采样攻高动态。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import tyro

import g1_lower_rl.tasks  # noqa: F401  导入即注册任务


def main(
  checkpoint: str,
  motion_dir: str | None = None,
  output: str = "motions/strata.json",
  clip_seconds: float = 10.0,
  rollouts: int = 5,
  mastered_threshold: float = 0.8,
  max_envs: int = 2048,
  device: str = "cuda:0",
) -> None:
  from dataclasses import asdict

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  cfg = load_env_cfg("G1-Gloria-MotionTracking", play=True)
  agent_cfg = load_rl_cfg("G1-Gloria-MotionTracking")
  if motion_dir is not None:
    cfg.commands["motion"].motion_dir = motion_dir
  # 分层要量的是策略的真实能力，不是抗扰动能力：关掉播放倍率随机化，固定原速。
  cfg.commands["motion"].speed_range = (1.0, 1.0)

  cfg.scene.num_envs = 1
  probe = ManagerBasedRlEnv(cfg=cfg, device=device)
  mt = probe.command_manager.get_term("motion")
  num_frames = mt.motion.num_frames.clone()
  names = list(mt.motion.names)
  probe.close()

  clip_len = int(clip_seconds * 50)
  clips: list[tuple[int, int, int]] = []  # (motion_id, start_frame, end_frame)
  for mid, n in enumerate(num_frames.tolist()):
    if n <= clip_len:
      clips.append((mid, 0, n - 1))
      continue
    for s in range(0, n - 1, clip_len):
      e = min(s + clip_len, n - 1)
      # 末尾不足半段的并进前一段，避免造出几十帧的碎片
      if e - s < clip_len // 2 and clips and clips[-1][0] == mid:
        clips[-1] = (mid, clips[-1][1], e)
      else:
        clips.append((mid, s, e))
  print(f"语料 {len(names)} 段 -> 切成 {len(clips)} 个 {clip_seconds}s 片段，每段跑 {rollouts} 次")

  per_batch = max(1, max_envs // rollouts)
  survived_frac = torch.zeros(len(clips))

  cfg.scene.num_envs = per_batch * rollouts
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), log_dir=None, device=device)
  runner.load(checkpoint)
  policy = runner.get_inference_policy(device=device)
  cmd = env.command_manager.get_term("motion")
  n_env = cfg.scene.num_envs
  all_ids = torch.arange(n_env, device=device)

  for b0 in range(0, len(clips), per_batch):
    chunk = clips[b0 : b0 + per_batch]
    mids = torch.tensor([c[0] for c in chunk], device=device).repeat_interleave(rollouts)
    starts = torch.tensor([c[1] for c in chunk], device=device).repeat_interleave(rollouts)
    ends = torch.tensor([c[2] for c in chunk], device=device).repeat_interleave(rollouts)
    k = len(mids)

    # reset 也必须在 inference_mode 内：上一批推理已把传感器缓冲区变成 inference
    # tensor，在模式外对它原地写会直接抛错。
    with torch.inference_mode():
      obs, _ = wrapped.reset()
      cmd.motion_ids[:k] = mids
      cmd.phase[:k] = starts.float()
      cmd._write_state_from_motion(all_ids)

      need = (ends - starts).max().item()
      alive = torch.zeros(n_env, dtype=torch.bool, device=device)
      alive[:k] = True
      reached = torch.zeros(n_env, dtype=torch.bool, device=device)
      for _ in range(int(need) + 1):
        obs, _, dones, _ = wrapped.step(policy(obs))
        # 相位走到片段末尾即算跟满。必须在结算 dones **之前**判：复位会把 phase 打回
        # 片段起点，最后一拍到达终点的样本会被误判成失败。
        reached[:k] |= alive[:k] & (cmd.phase[:k] >= ends.float())
        alive &= ~dones.bool()
        if not alive.any():
          break

    got = reached[:k].float().view(len(chunk), rollouts).mean(dim=1).cpu()
    survived_frac[b0 : b0 + len(chunk)] = got
    print(f"  {b0 + len(chunk):>6}/{len(clips)}  本批完成率均值 {got.mean():.3f}")

  env.close()

  mastered = [c for c, f in zip(clips, survived_frac.tolist()) if f >= mastered_threshold]
  challenging = [c for c, f in zip(clips, survived_frac.tolist()) if f < mastered_threshold]
  out = {
    "checkpoint": checkpoint,
    "clip_frames": clip_len,
    "rollouts": rollouts,
    "mastered_threshold": mastered_threshold,
    "names": names,
    "mastered": [list(c) for c in mastered],
    "challenging": [list(c) for c in challenging],
    "completion": survived_frac.tolist(),
  }
  Path(output).parent.mkdir(parents=True, exist_ok=True)
  Path(output).write_text(json.dumps(out), encoding="utf-8")

  m_sec = sum(c[2] - c[1] for c in mastered) / 50 / 3600
  c_sec = sum(c[2] - c[1] for c in challenging) / 50 / 3600
  print(f"\nmastered   {len(mastered):>6} 段  {m_sec:.2f} h")
  print(f"challenging{len(challenging):>6} 段  {c_sec:.2f} h")
  print(f"写入 {output}")


if __name__ == "__main__":
  tyro.cli(main)
