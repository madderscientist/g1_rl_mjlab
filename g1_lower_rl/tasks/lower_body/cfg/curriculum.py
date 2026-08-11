"""课程配置。

档位表本身在 ``constants`` 里；这里只是把它们接到具体的指令/事件/奖励项上。
除 ``command_vel`` / ``command_height`` 外全部带闸门——纯按步数推进的课程会跑在策略
前面，一到满强度策略就用前进速度换更低的摔倒概率，而且再也没恢复。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mjlab.managers.curriculum_manager import CurriculumTermCfg

from g1_lower_rl.tasks.lower_body import mdp
from g1_lower_rl.tasks.lower_body.cfg.constants import (
  ARM_DRIFT_LEVELS,
  ARM_TORQUE_LEVELS,
  BODY_IMPULSE_LEVELS,
  HEIGHT_STAGES,
  HEIGHT_STD_GATE,
  HEIGHT_STD_STAGES,
  ITER,
  RESET_LEVELS,
  TRACKING_GATE,
  VELOCITY_STAGES,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnvCfg


def max_out_curriculum(cfg: ManagerBasedRlEnvCfg) -> str:
  """把所有课程档位直接拨到最后一档，再清空课程本身。返回一行档位摘要。

  评估和录像都需要它：课程被清空后不手动拨到终态的话，扰动和指令量程会停在第 0 档，
  会系统性地低估策略实际被推到什么程度，也避开了它最容易失效的高速段。
  """
  vel = cfg.curriculum["command_vel"].params["velocity_stages"][-1]
  cfg.curriculum = {}

  tau = ARM_TORQUE_LEVELS[-1][1]
  force = BODY_IMPULSE_LEVELS[-1][1]
  drift = ARM_DRIFT_LEVELS[-1][1]
  reset = RESET_LEVELS[-1][1]

  cfg.events["arm_torque"].params["torque_range"] = (-tau, tau)
  cfg.events["body_impulse"].params["force_range"] = (-force, force)
  cfg.events["body_impulse"].params["torque_range"] = (-force / 6.0, force / 6.0)
  cfg.events["arm_pose_drift"].params["blend"] = drift
  cfg.events["reset_base"].params["scale"] = reset
  cfg.events["reset_robot_joints"].params["scale"] = reset
  # 高度奖励核也归课程管，不拨的话报出来的高度奖励会偏高（宽核更宽容）。
  cfg.rewards["track_base_height"].params["std"] = HEIGHT_STD_STAGES[-1][1]

  twist = cfg.commands["twist"]
  twist.ranges.lin_vel_x = vel["lin_vel_x"]  # type: ignore[attr-defined]
  twist.ranges.lin_vel_y = vel["lin_vel_y"]  # type: ignore[attr-defined]
  twist.ranges.ang_vel_z = vel["ang_vel_z"]  # type: ignore[attr-defined]
  height = HEIGHT_STAGES[-1][1]
  cfg.commands["height"].ranges.height = height  # type: ignore[attr-defined]

  return (
    f"MAX CURRICULUM  arm_tau +-{tau:.1f}Nm  impulse +-{force:.0f}N  "
    f"arm_drift {drift:.1f}  reset {reset:.1f}  "
    f"vx[{vel['lin_vel_x'][0]:+.1f},{vel['lin_vel_x'][1]:+.1f}] "
    f"wz[{vel['ang_vel_z'][0]:+.2f},{vel['ang_vel_z'][1]:+.2f}] "
    f"h[{height[0]:.2f},{height[1]:.2f}]"
  )


def _gated_event(event_name: str, log_key: str, stages: list) -> CurriculumTermCfg:
  return CurriculumTermCfg(
    func=mdp.gated_event_params,
    params={
      "event_name": event_name,
      "log_key": log_key,
      "gate": TRACKING_GATE,
      "stages": stages,
    },
  )


def make_curriculum() -> dict[str, CurriculumTermCfg]:
  return {
    "command_vel": CurriculumTermCfg(
      func=mdp.commands_vel,
      params={
        "command_name": "twist",
        "velocity_stages": [{"step": s * ITER, **rng} for s, rng in VELOCITY_STAGES],
      },
    ),
    # 高度量程按步数逐档放开（不抓闸：深蹲本身不会把策略带崩，只是开局学不过来）。
    "command_height": CurriculumTermCfg(
      func=mdp.commands_height,
      params={
        "command_name": "height",
        "height_stages": [{"step": s * ITER, "height": h} for s, h in HEIGHT_STAGES],
      },
    ),
    # 高度奖励核逐档收紧。这是“宽核自举、窄核精度”那条经验的落地：同一个量在训练早期
    # 和部署时需要的宽窄相反，正确做法是课程，不是挑一个折中的常量。
    # 日志里看 Curriculum/height_std（当前 std）和 Curriculum/track_base_height_score
    # （闸门分数）——闸门一直不开的话 std 会停在 0.08，必须能看见。
    "height_std": CurriculumTermCfg(
      func=mdp.tightening_height_std,
      params={
        "term_name": "track_base_height",
        "command_name": "height",
        "stages": [{"step": s * ITER, "std": v} for s, v in HEIGHT_STD_STAGES],
        "gate": HEIGHT_STD_GATE,
      },
    ),
    "reset_pose_level": _gated_event(
      "reset_base",
      "scale",
      [{"step": s * ITER, "params": {"scale": v}} for s, v in RESET_LEVELS],
    ),
    "reset_joint_vel_level": _gated_event(
      "reset_robot_joints",
      "scale",
      [{"step": s * ITER, "params": {"scale": v}} for s, v in RESET_LEVELS],
    ),
    "arm_torque_level": _gated_event(
      "arm_torque",
      "torque_range",
      [
        {"step": s * ITER, "params": {"torque_range": (-t, t)}}
        for s, t in ARM_TORQUE_LEVELS
      ],
    ),
    "body_impulse_level": _gated_event(
      "body_impulse",
      "force_range",
      [
        {
          "step": s * ITER,
          "params": {"force_range": (-f, f), "torque_range": (-f / 6.0, f / 6.0)},
        }
        for s, f in BODY_IMPULSE_LEVELS
      ],
    ),
    "arm_drift_level": _gated_event(
      "arm_pose_drift",
      "blend",
      [{"step": s * ITER, "params": {"blend": b}} for s, b in ARM_DRIFT_LEVELS],
    ),
  }
