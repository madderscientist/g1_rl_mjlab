"""打印独立脚步规划结果，不运行机器人或加载策略"""

import argparse
import json
from dataclasses import asdict, replace

from g1_lower_rl.footstep_contract import FOOTSTEP_SLOTS
from g1_lower_rl.footsteps import FootstepManager, FootstepManagerCfg, RandomCommandCfg, RandomCommandSource


def main() -> None:
  """解析种子和模拟时长，逐拍生成脚印并演示停止与自动再起步"""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--seed", type=int, default=7)
  parser.add_argument("--seconds", type=float, default=20.0)
  parser.add_argument("--stop-at", type=float, default=3.0)
  args = parser.parse_args()
  cfg = FootstepManagerCfg(require_contact_confirmation=False)
  source = RandomCommandSource(RandomCommandCfg(automatic_commands=False), seed=args.seed)
  manager = FootstepManager(cfg, seed=args.seed)
  manager.reset([[0.0, 0.11, 0.0], [0.0, -0.11, 0.0]], source.reset())
  print(
    json.dumps(
      {
        "demo": "planning only, no physics/contact verification",
        "footstep_slots": list(FOOTSTEP_SLOTS),
        "initial_mode": manager.mode,
        "config": {"manager": asdict(cfg), "source": asdict(source.cfg)},
      }
    )
  )
  stopped = False
  for _ in range(int(args.seconds / cfg.control_dt)):
    if not stopped and manager.elapsed >= args.stop_at:
      source.set_request(replace(source.request, walking=False))
      manager.apply_request(source.request)
      stopped = True
    mode = manager.mode
    update = manager.advance()
    request = source.advance(cfg.control_dt, mode=update.command.mode, frequency=update.command.frequency)
    manager.apply_request(request)
    command = manager.command()
    if update.landed_sides or command.mode != mode:
      print(
        json.dumps(
          {
            "time": round(manager.elapsed, 3),
            "mode": command.mode,
            "phase": command.phase,
            "frequency": command.frequency,
            "landed_sides": update.landed_sides,
            "footsteps": command.footsteps.tolist(),
          }
        )
      )


if __name__ == "__main__":
  main()
