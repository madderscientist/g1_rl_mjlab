"""下肢任务的课程项。

三类：按步数放开指令量程（``commands_height``）、按“任务完成得好不好”抓闸后再抬扰动
（``gated_event_params``）、以及逐档收紧奖励核（``tightening_height_std``）。

后两类都带 ``state_dict``/``load_state_dict``：档位和闸门分数必须跟着 checkpoint 走，
否则续训时步数条件全部满足而分数从 0 重来，档位能不能爬回去纯看运气。
参见 ``g1_lower_rl.rl.runner``。
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, TypedDict

import torch

from g1_lower_rl.tasks.lower_body.mdp.observations import height_above_feet

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class EventStage(TypedDict):
  step: int
  params: dict[str, Any]


class HeightStage(TypedDict):
  step: int
  height: tuple[float, float]


class StdStage(TypedDict):
  step: int
  std: float


def _distributed_training_requested() -> bool:
  return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _fraction_from_counts(
  success_count: torch.Tensor, sample_count: torch.Tensor, fallback: float
) -> float:
  success_count_value, sample_count_value = torch.stack(
    (success_count, sample_count)
  ).tolist()
  return (
    success_count_value / sample_count_value
    if sample_count_value > 0
    else fallback
  )


def commands_height(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  height_stages: list[HeightStage],
) -> torch.Tensor:
  """按步数逐档放开高度指令的量程。

  一上来就在 (0.50, 0.80) 全量程均匀采样，等于同时要求它学会站直、深蹲和在两者之间
  切换；先把量程收在标称站姿附近，等站稳了再往下放。返回下界作为日志值——上界基本
  不动，会变的是允许蹲多深。
  """
  del env_ids  # 未使用。
  term = env.command_manager.get_term(command_name)
  assert term is not None
  for stage in height_stages:
    if env.common_step_counter > stage["step"]:
      term.cfg.ranges.height = stage["height"]  # type: ignore[attr-defined]
  return torch.tensor(float(term.cfg.ranges.height[0]))  # type: ignore[attr-defined]


class _GatedStages:
  """按“步数达到 + 闸门分数达标”推进档位的公共部分。

  纯按步数推进的课程会跑在策略前面：上肢扰动一到满强度，策略就不再往前走、改成
  原地踏步，用前进速度换更低的摔倒概率，而且再也没恢复。这个闸只在任务确实
  撑得住的时候才放下一档进来。
  """

  def __init__(self):
    self._stage = 0
    # 初值给 0 而不是 1：给 1 的话，续训时所有档位的步数阈值早已越过，闸门一开始就是开的，
    # 二十个迭代内直接顶满，闸门形同虚设。
    self._score = 0.0

  def state_dict(self) -> dict:
    return {"stage": self._stage, "score": self._score}

  def load_state_dict(self, state: dict) -> None:
    self._stage = int(state["stage"])
    self._score = float(state["score"])

  def _local_counts(
    self, env: ManagerBasedRlEnv, **params: Any
  ) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError

  def _advance_from_global(
    self, env: ManagerBasedRlEnv, good: float, **params: Any
  ) -> None:
    raise NotImplementedError

  def _apply_stage(self, env: ManagerBasedRlEnv, **params: Any) -> torch.Tensor:
    raise NotImplementedError

  def _clamp_stage(self, stages: list) -> None:
    self._stage = min(self._stage, len(stages) - 1)

  def _advance(
    self,
    env: ManagerBasedRlEnv,
    log_name: str,
    good: float,
    stages: list,
    gate: dict[str, Any],
  ) -> None:
    self._score += (good - self._score) * gate["ema"]
    # 必须可见：上一版里这个分数没进日志，结果一个 12000 迭代的从零训练全程卡在第 0 档
    # 都没人发现。课程比奖励项先跑，"log" 这时可能还没建，所以用 setdefault。
    env.extras.setdefault("log", {})[f"Curriculum/{log_name}_score"] = torch.tensor(
      self._score
    )
    nxt = self._stage + 1
    if (
      nxt < len(stages)
      and env.common_step_counter > stages[nxt]["step"]
      and self._score >= gate["min_fraction"]
    ):
      self._stage = nxt
    # 夹紧：档位可能是从存档恢复的，而档位表在两次运行之间可能被改短。
    self._clamp_stage(stages)


class gated_event_params(_GatedStages):
  """逐档抬高事件参数，闸门看速度跟随质量。"""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg, env
    super().__init__()

  def _local_counts(
    self,
    env: ManagerBasedRlEnv,
    event_name: str,
    stages: list[EventStage],
    log_key: str,
    gate: dict[str, Any],
  ) -> tuple[torch.Tensor, torch.Tensor]:
    del event_name, stages, log_key
    command = env.command_manager.get_command(gate["command_name"])
    assert command is not None
    robot = env.scene["robot"]
    error = torch.norm(command[:, :2] - robot.data.root_link_lin_vel_b[:, :2], dim=1)
    eligible = torch.norm(command[:, :2], dim=1) > gate["min_command"]
    return ((error < gate["max_error"]) & eligible).sum(), eligible.sum()

  def _advance_from_global(
    self,
    env: ManagerBasedRlEnv,
    good: float,
    event_name: str,
    stages: list[EventStage],
    log_key: str,
    gate: dict[str, Any],
  ) -> None:
    del log_key
    self._advance(env, event_name, good, stages, gate)

  def _apply_stage(
    self,
    env: ManagerBasedRlEnv,
    event_name: str,
    stages: list[EventStage],
    log_key: str,
    gate: dict[str, Any],
  ) -> torch.Tensor:
    del gate
    self._clamp_stage(stages)
    cfg = env.event_manager.get_term_cfg(event_name)
    cfg.params.update(stages[self._stage]["params"])
    value = cfg.params[log_key]
    return torch.tensor(float(value[-1] if isinstance(value, (tuple, list)) else value))

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str,
    stages: list[EventStage],
    log_key: str,
    gate: dict[str, Any],
  ) -> torch.Tensor:
    del env_ids  # 未使用。
    params = {
      "event_name": event_name,
      "stages": stages,
      "log_key": log_key,
      "gate": gate,
    }
    if not _distributed_training_requested():
      success_count, sample_count = self._local_counts(env, **params)
      good = _fraction_from_counts(success_count, sample_count, self._score)
      self._advance_from_global(env, good, **params)
    return self._apply_stage(env, **params)


class tightening_height_std(_GatedStages):
  """逐档收紧 ``track_base_height`` 的 std，且要求当前档位已经被满足。

  **为什么这一项必须是课程而不是常量。** 指数核 ``exp(-e²/std²)`` 在 ``e >> std``
  时是数值上的一片平地，梯度为零：

  | 高度误差 | std=0.08 | std=0.05 |
  |---|---|---|
  | 0.08 m | 0.276 | 0.062 |
  | 0.15 m | 0.0223 | 0.000099 |
  | 0.20 m | 0.0014 | 4e-7 |

  从零训练的机器人在成型期正好活在 0.15~0.25 m 误差这个区间。窄核在那里给不出任何
  “站高一点”的信号，机器人一直坐在地上，局长只有正常的三分之一，永远攒不够连续经验
  去长出步态。但窄核对部署又是必要的：std=0.05 时比指令低 5 cm 只剩 0.37 的奖励，
  能把“蹲着走”压掉。

  两个需求不冲突，冲突的是“从第 0 迭代就用最终值”。宽核负责自举，窄核负责精度，
  中间用这条课程过渡。**同类的收紧（腰部抖动抑制、摔倒代价）也该按这个思路做，
  而不是直接改常量。**

  闸门用的是绝对精度而不是相对当前 std 的精度：只有当机器人确实已经站到指令高度
  附近，收紧才有意义；否则收紧只会把它推回没有梯度的那片平地。
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg, env
    super().__init__()

  def _local_counts(
    self,
    env: ManagerBasedRlEnv,
    term_name: str,
    command_name: str,
    stages: list[StdStage],
    gate: dict[str, Any],
  ) -> tuple[torch.Tensor, torch.Tensor]:
    del stages
    cfg = env.reward_manager.get_term_cfg(term_name)
    asset_cfg = cfg.params["asset_cfg"]
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    error = torch.abs(command[:, 0] - height_above_feet(asset, asset_cfg.site_ids))
    return (error < gate["max_error"]).sum(), error.new_tensor(error.numel())

  def _advance_from_global(
    self,
    env: ManagerBasedRlEnv,
    good: float,
    term_name: str,
    command_name: str,
    stages: list[StdStage],
    gate: dict[str, Any],
  ) -> None:
    del command_name
    self._advance(env, term_name, good, stages, gate)

  def _apply_stage(
    self,
    env: ManagerBasedRlEnv,
    term_name: str,
    command_name: str,
    stages: list[StdStage],
    gate: dict[str, Any],
  ) -> torch.Tensor:
    del command_name, gate
    self._clamp_stage(stages)
    cfg = env.reward_manager.get_term_cfg(term_name)
    cfg.params["std"] = stages[self._stage]["std"]
    return torch.tensor(float(cfg.params["std"]))

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    term_name: str,
    command_name: str,
    stages: list[StdStage],
    gate: dict[str, Any],
  ) -> torch.Tensor:
    del env_ids  # 未使用。
    params = {
      "term_name": term_name,
      "command_name": command_name,
      "stages": stages,
      "gate": gate,
    }
    if not _distributed_training_requested():
      success_count, sample_count = self._local_counts(env, **params)
      good = _fraction_from_counts(success_count, sample_count, self._score)
      self._advance_from_global(env, good, **params)
    return self._apply_stage(env, **params)


def synchronize_distributed_curriculum(env: ManagerBasedRlEnv) -> None:
  """每步一次性聚合多卡闸门统计，并以 rank 0 状态统一所有课程。"""
  distributed = torch.distributed
  if not (distributed.is_available() and distributed.is_initialized()):
    return

  manager = env.curriculum_manager
  entries: list[tuple[_GatedStages, Any]] = []
  reset_count = env.reset_buf.sum()
  rows = [
    torch.stack((reset_count, torch.zeros_like(reset_count))).to(torch.float64)
  ]
  rank = distributed.get_rank()

  for name in manager.active_terms:
    term_cfg = manager.get_term_cfg(name)
    term = term_cfg.func
    if not isinstance(term, _GatedStages):
      continue
    success_count, sample_count = term._local_counts(env, **term_cfg.params)
    rows.append(torch.stack((success_count, sample_count)).to(torch.float64))
    rows.append(
      torch.tensor(
        [term._stage, term._score] if rank == 0 else [0.0, 0.0],
        device=env.device,
        dtype=torch.float64,
      )
    )
    entries.append((term, term_cfg))

  stats = torch.stack(rows)
  distributed.all_reduce(stats, op=distributed.ReduceOp.SUM)
  values = stats.tolist()
  any_reset = values[0][0] > 0

  row = 1
  for term, term_cfg in entries:
    success_count, sample_count = values[row]
    stage, score = values[row + 1]
    term._stage = int(stage)
    term._score = score
    if any_reset:
      good = success_count / sample_count if sample_count > 0 else term._score
      term._advance_from_global(env, good, **term_cfg.params)
    term._apply_stage(env, **term_cfg.params)
    row += 2

  if any_reset:
    # 所有 rank 同时重放，应用同步后的闸门档位和纯步数指令量程。
    manager.compute()
