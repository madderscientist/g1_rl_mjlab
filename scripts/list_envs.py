"""列出已注册的任务。"""

from __future__ import annotations

from mjlab.tasks.registry import list_tasks, load_env_cfg
from prettytable import PrettyTable


def main() -> None:
  import g1_lower_rl.tasks  # noqa: F401  注册任务

  table = PrettyTable(["#", "Task ID", "Entities", "Envs"])
  table.align["Task ID"] = "l"
  for index, task_id in enumerate(sorted(list_tasks())):
    cfg = load_env_cfg(task_id)
    table.add_row(
      [index + 1, task_id, ", ".join(cfg.scene.entities), cfg.scene.num_envs]
    )
  print(table)


if __name__ == "__main__":
  main()
