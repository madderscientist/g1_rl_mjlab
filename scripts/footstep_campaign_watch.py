"""独立systemd定时巡检，恢复意外停止的八天监督器但不越过用户停止和截止时间"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


def choose_action(active, phase, stopped, deadline, now, updated):
  if stopped or now >= deadline:
    return "stop" if active else None
  if phase in {"paused", "blocked", "completed"}:
    return None
  if not active:
    return "start"
  if updated is not None and now - updated > 1200:
    return "restart"
  return None


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("manifest", type=Path)
  parser.add_argument("--dry-run", action="store_true")
  args = parser.parse_args()
  manifest = json.loads(args.manifest.read_text())
  root = args.manifest.resolve().parent
  now = time.time()
  deadline = datetime.fromisoformat(manifest["deadline"]).timestamp()
  records = []
  for lane in manifest["lanes"]:
    lane_root = Path(lane["root"])
    state_path = lane_root / "state.json"
    error = None
    try:
      state = json.loads(state_path.read_text()) if state_path.exists() else {}
    except (OSError, ValueError) as exception:
      state, error = {}, str(exception)
    process = subprocess.run(["systemctl", "--user", "is-active", lane["unit"]],
      capture_output=True, text=True, timeout=15)
    active = process.stdout.strip() in {"active", "activating", "deactivating"}
    updated = datetime.fromisoformat(state["updated_at_utc"]).timestamp() if state.get("updated_at_utc") else None
    stopped = (root / "STOP").exists() or (lane_root / "STOP").exists()
    action = choose_action(active, state.get("phase"), stopped, deadline, now, updated)
    if error and action in {"start", "restart"}:
      action = None
    result = {"lane": lane["name"], "active": active, "phase": state.get("phase"), "action": action,
      "last_update": state.get("updated_at_utc"), "last_iteration": state.get("last_iteration"),
      "cycle": state.get("cycle"), "error": error}
    if action and not args.dry_run:
      command = subprocess.run(["systemctl", "--user", action, lane["unit"]],
        capture_output=True, text=True, timeout=660)
      result.update(action_returncode=command.returncode, action_error=command.stderr[-1000:])
    records.append(result)
  record = {"time": datetime.now(timezone.utc).isoformat(), "deadline": manifest["deadline"], "lanes": records}
  if not args.dry_run:
    with (root / "watchdog.jsonl").open("a") as output:
      output.write(json.dumps(record) + "\n")
    temporary = root / "watchdog_latest.tmp"
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(root / "watchdog_latest.json")
    with (root / "RUN_LOG.md").open("a") as output:
      output.write(f"- {record['time']} 独立定时巡检: {json.dumps(records, ensure_ascii=False)}\n")
    if now >= deadline and not any(item["active"] for item in records):
      subprocess.run(["systemctl", "--user", "stop", manifest["timer_unit"]], timeout=15, check=False)
  print(json.dumps(record, indent=2))


if __name__ == "__main__":
  main()