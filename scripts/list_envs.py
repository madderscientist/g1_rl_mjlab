"""列出已注册的任务。"""

from __future__ import annotations

from mjlab.tasks.registry import load_env_cfg
from prettytable import PrettyTable

from g1_lower_rl.tasks import MOCAP_TASK, list_tasks


def main() -> None:
  table = PrettyTable(["#", "Task ID", "Entities", "Envs"])
  table.align["Task ID"] = "l"
  for index, task_id in enumerate(sorted(list_tasks())):
    if task_id == MOCAP_TASK:
      table.add_row([index + 1, task_id, "robot (MLP + SuperMimic, position WM v3)", "training config"])
      continue
    cfg = load_env_cfg(task_id)
    table.add_row(
      [index + 1, task_id, ", ".join(cfg.scene.entities), cfg.scene.num_envs]
    )
  print(table)


if __name__ == "__main__":
  main()
