"""带课程状态持久化的 PPO runner。

mjlab 的 ``MjlabOnPolicyRunner`` 只把 ``common_step_counter`` 存进 checkpoint，课程项
自己的档位和闸门分数没存。这两者缺一不可：续训时步数条件全部满足、而分数从 0 重来，
档位能不能爬回去完全取决于分数能不能重新越过 ``min_fraction``。

**实测代价：** 闸门取 ``err<0.4 比例>=0.5`` 时分数封顶 0.43，三条上肢扰动课程在整整
3000 迭代里一次都没升过档——力矩/外力/漂移全是字面意义的 0，而同期对照组是满扰动
（4.0 Nm / 20 N / drift 1.0）。两者的所有指标对比因此全部作废。高度 std 课程同理，
续训会把它从 0.05 退回 0.08。

这个坑踩过两次：早先分数初值是 1.0，续训时闸门一开就是开的、二十迭代顶满；改成 0.0
之后变成另一个极端。**根因不是初值，是状态没跟着 checkpoint 走。**
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from mjlab.rl import MjlabOnPolicyRunner, RslRlPpoAlgorithmCfg, RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx
from mjlab.tasks.registry import load_runner_cls
from mjlab.tasks.velocity.rl.runner import VelocityOnPolicyRunner


@dataclass
class ScheduledPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """PPO 配置，多一个 ``entropy_coef`` 课程。

  ``entropy_stages`` 形如 ``((迭代数, entropy_coef, σ 上限), ...)``，空则不做课程。
  σ 上限取 ``math.inf`` 表示这一档不动 σ，且它只下压不上抬。

  这个字段会被 ``GloriaOnPolicyRunner`` 在构造时摘走：rsl_rl 最后把
  ``cfg["algorithm"]`` 整个 splat 进 PPO 的构造函数，多一个键就 TypeError。
  """

  entropy_stages: tuple[tuple[int, float, float], ...] = ()

  # AMP（arXiv:2104.02180）。>0 才启用，此时算法类切到 AmpPPO。
  amp_coef: float = 0.0
  amp_lr: float = 1.0e-4
  amp_grad_penalty: float = 10.0
  amp_epochs: int = 1


def action_joint_names(env) -> list[str]:
  """动作项实际驱动的关节，按动作向量的顺序。"""
  return list(env.action_manager.get_term("joint_pos").target_names)


def deployment_metadata(env) -> dict[str, str]:
  """mjlab 的 ``get_base_metadata`` 漏掉、而部署端必需的那几个关节名单。

  ``joint_names`` 写的是模型**全部**关节，``action_scale`` 只覆盖动作关节，两者长度不同。
  靠 ``joint_names[:n]`` 截断去猜在别的任务上会错位（29 自由度那版里 ``left_eccentric_joint``
  夹在左右臂中间，右臂整体差一格）。观测里的关节子集也不一定等于动作关节：站立任务
  观测全身 29 轴、只驱动下肢 15 轴。两份名单都得显式写出去。
  """
  metadata = {"action_joint_names": ",".join(action_joint_names(env))}
  all_joints = env.scene["robot"].joint_names
  for term_name in ("joint_pos", "joint_vel"):
    asset_cfg = env.observation_manager.get_term_cfg("actor", term_name).params.get(
      "asset_cfg"
    )
    # 没有 asset_cfg 的观测项取的是全部关节（动作跟踪任务就是这样，含两个夹爪轴）。
    ids = asset_cfg.joint_ids if asset_cfg is not None else slice(None)
    selected = (
      all_joints[ids] if isinstance(ids, slice) else [all_joints[i] for i in ids]
    )
    metadata[f"obs_{term_name}_joint_names"] = ",".join(selected)
  return metadata


def latest_checkpoint(root: Path) -> Path:
  """``{log_root}/{experiment_name}`` 下 mtime 最新的检查点。"""
  ckpts = sorted(root.glob("*/model_*.pt"), key=lambda p: p.stat().st_mtime)
  if not ckpts:
    raise FileNotFoundError(f"{root} 下没有找到检查点")
  return ckpts[-1]


def load_trained_runner(task_id: str, env, agent_cfg, checkpoint: Path, device: str):
  """建 runner 并载入 actor 权重。

  必须走任务注册表拿 runner 类：本包的 ``GloriaOnPolicyRunner`` 才认得存档里的
  课程状态，用基类会静默丢掉。
  """
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
  )
  return runner


class GloriaOnPolicyRunner(VelocityOnPolicyRunner):
  """保存课程状态

  与任务无关：无课程的任务存空字典，关节名单从环境里读。所有任务都该用它，
  否则落回基类就只存 .pt、没有 ONNX 和部署元数据。
  """

  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    # 必须赶在基类把 algorithm 字典 splat 进 PPO 之前摘走。存档写在 runner 创建之前，yaml 不受影响。
    stages = train_cfg.get("algorithm", {}).pop("entropy_stages", ()) or ()
    self._entropy_stages = tuple(tuple(stage) for stage in stages)
    self._entropy_stage = -1

    # Stage II：命令项里配了分层文件就切到 PACE+STAR 的算法类。
    cmd = getattr(env.unwrapped, "command_manager", None)
    motion = cmd.get_term("motion") if cmd is not None else None
    self._stage2 = motion is not None and getattr(motion, "bin_pool_acq", None) is not None
    if self._stage2:
      alg = train_cfg.setdefault("algorithm", {})
      alg["class_name"] = "g1_lower_rl.rl.pace_star:PaceStarPPO"
      alg["acquisition_fraction"] = motion.cfg.acquisition_fraction

    # AMP（Stage II 的另一种形态，与 PACE 互斥）：amp_coef>0 才切算法类。
    alg_cfg = train_cfg.setdefault("algorithm", {})
    self._amp_coef = float(alg_cfg.get("amp_coef", 0.0) or 0.0)
    self._amp_on = self._amp_coef > 0.0 and not self._stage2
    if self._amp_on:
      alg_cfg["class_name"] = "g1_lower_rl.rl.amp:AmpPPO"
    else:
      # 没启用就摘干净，否则这些键会被 splat 进不认识它们的 PPO。
      for k in ("amp_coef", "amp_lr", "amp_grad_penalty", "amp_epochs"):
        alg_cfg.pop(k, None)

    super().__init__(env, train_cfg, log_dir, device)

    if self._stage2:
      self._motion_cmd = motion
      self.alg.bin_weight_fn = _BinWeight(motion)
      self.alg.pace_requested = True
      self._wrap_step_for_bin_record()

    if self._amp_on:
      self._setup_amp(motion)

  def _setup_amp(self, motion_cmd) -> None:
    """挂上判别器，并把风格奖励注入 env.step 的返回值。"""
    from g1_lower_rl.rl.amp import MotionExpertSampler, _local_features

    robot = self.env.unwrapped.scene["robot"]
    body_names = list(motion_cmd.cfg.body_names)
    robot_bi = [robot.body_names.index(n) for n in body_names]
    robot_ai = robot.body_names.index(motion_cmd.cfg.anchor_body_name)
    motion_bi = list(range(len(body_names)))
    motion_ai = motion_cmd.motion_anchor_body_index

    sampler = MotionExpertSampler(motion_cmd.motion, motion_bi, motion_ai)
    self.alg.attach_amp(sampler.feature_dim(), sampler, self.device)

    def policy_features() -> torch.Tensor:
      d = robot.data
      return _local_features(
        d.joint_pos,
        d.body_link_pos_w[:, robot_bi],
        d.body_link_pos_w[:, robot_ai],
        d.body_link_quat_w[:, robot_ai],
        d.body_link_lin_vel_w[:, robot_ai],
      )

    env = self.env
    inner = env.step

    def step(actions):
      prev = policy_features()
      obs, rewards, dones, extras = inner(actions)
      cur = policy_features()
      self.alg.record_amp_pair(prev, cur)
      return obs, rewards + self.alg.style_reward(prev, cur).to(rewards.device), dones, extras

    env.step = step

  def _wrap_step_for_bin_record(self) -> None:
    """每个环境步把起点 bin 记进环形缓冲，供 STAR 在更新时按 fragment 还原难度。"""
    env, cmd = self.env, self._motion_cmd
    inner = env.step
    n_steps = self.cfg["num_steps_per_env"]
    state = {"i": 0}

    def step(actions):
      cmd.record_bins(state["i"] % n_steps, n_steps)
      state["i"] += 1
      return inner(actions)

    env.step = step

  @staticmethod
  def _get_export_paths(checkpoint_path: str) -> tuple[Path, str, Path]:
    export_dir = Path(checkpoint_path).parent
    filename = "policy.onnx"
    return export_dir, filename, export_dir / filename

  def _stateful_curriculum_terms(self):
    """返回 (名字, 课程实例)——只挑实现了 state_dict 的。"""
    manager = getattr(self.env.unwrapped, "curriculum_manager", None)
    for name in getattr(manager, "active_terms", ()):
      func = manager.get_term_cfg(name).func
      if hasattr(func, "state_dict") and hasattr(func, "load_state_dict"):
        yield name, func

  def _entropy_stage_index(self) -> int:
    it = self.current_learning_iteration
    return max(i for i, stage in enumerate(self._entropy_stages) if stage[0] <= it)

  def _advance_entropy_stage(self) -> None:
    """跨档时换 ``entropy_coef``，并把 σ 压到该档上限以下。

    **σ 必须跟着一起压。** 只改系数退火太慢（实测 0.01->0.002 只把漂移改变
    -3.2e-4/iter，σ 从 1.05 到 0.35 要约 3200 迭代），而站立在 σ>1.0 时 300 迭代就崩了。
    Adam 动量也要清：``std_param`` 之前累积的是“往上推”的历史，不清就会立刻顶回来。

    σ 上限是档位的**不变量**，不是一次性事件：续训进来也要重新压一遍。曾经在这里
    信过“存档里的 σ 已经压过了”，结果档位阈值设在 4000、训练只跑到 3436 就中断，
    续训时直接落到第 1 档却不动 σ，1.02 的噪声原样带进新 run。σ 本来就低于上限时
    ``before <= cap`` 会提前返回，重复执行没有副作用。
    """
    stage = self._entropy_stage_index()
    if stage == self._entropy_stage:
      return
    self._entropy_stage = stage
    _, coef, cap = self._entropy_stages[stage]
    self.alg.entropy_coef = coef
    print(
      f"[INFO]: entropy 课程第 {stage} 档，entropy_coef -> {coef}"
      f"（迭代 {self.current_learning_iteration}）"
    )

    distribution = self.alg.get_policy().distribution
    assert distribution is not None
    std = distribution.std_param
    assert isinstance(std, torch.Tensor)
    before = float(std.mean().item())
    if before <= cap:
      return
    with torch.no_grad():
      std.clamp_(max=cap)
    state = self.alg.optimizer.state.get(std)
    if state is not None:
      state["exp_avg"].zero_()
      state["exp_avg_sq"].zero_()
    print(f"[INFO]: 探索噪声 σ {before:.3f} -> {cap:.3f}，已清零其 Adam 动量")

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
    if self._entropy_stages:
      self._advance_entropy_stage()
      # rsl_rl 的 learn 循环没有逐迭代钩子，包一层 update 是侵入最小的接法。
      # update 在 rollout 之后调用，所以档位比阈值晚一拍生效，无所谓。
      inner_update = self.alg.update

      def update():
        self._advance_entropy_stage()
        return inner_update()

      self.alg.update = update
    super().learn(num_learning_iterations, init_at_random_ep_len)

  def save(self, path: str, infos=None):
    curriculum_state = {
      name: term.state_dict() for name, term in self._stateful_curriculum_terms()
    }
    # 父类会覆盖 infos["env_state"]，但其余键原样保留，所以另起一个键。
    super().save(path, {**(infos or {}), "curriculum_state": curriculum_state})

    # 父类导出时不写这几项，不补的话只有手动跑 export_onnx.py 的产物才带。
    _, _, onnx_path = self._get_export_paths(path)
    if not onnx_path.exists():
      return  # 父类的导出是 try/except 包着的，失败时不该连累存档。
    try:
      attach_metadata_to_onnx(str(onnx_path), deployment_metadata(self.env.unwrapped))
    except Exception as e:
      print(f"[WARN] 写入部署元数据失败（训练继续）: {e}")

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    infos = super().load(path, load_cfg, strict, map_location)
    saved = (infos or {}).get("curriculum_state") or {}
    restored, missing = [], []
    for name, term in self._stateful_curriculum_terms():
      if name in saved:
        term.load_state_dict(saved[name])
        restored.append(f"{name}={saved[name]['stage']}")
      else:
        missing.append(name)
    if restored:
      print(f"[INFO]: 已恢复课程档位: {', '.join(restored)}")
    if missing:
      # 必须刺眼。静默回退到第 0 档正是这个 bug 当初没被发现的原因。
      print(
        "\n"
        + "!" * 78
        + "\n[警告]: 以下课程没有存档状态，将从第 0 档、分数 0 重新开始：\n"
        f"        {', '.join(missing)}\n"
        "        存档早于课程状态持久化功能。续训后这些课程可能永远升不回原来的档位，\n"
        "        指标会因为“扰动变小”而虚高，与其它运行不可比。\n"
        "        训练开始后请立刻确认 Curriculum/<名字> 是否回到了预期值。\n"
        + "!" * 78
        + "\n"
      )
    return infos


class _BinWeight:
  """STAR 的难度权重 w_t = B * p_{b_t}，>1 表示该 bin 的采样概率高于均匀基线。"""

  def __init__(self, motion) -> None:
    self._m = motion

  def bins(self):
    return self._m.bin_history

  def __call__(self):
    hist, probs = self._m.bin_history, self._m.last_bin_probs
    if hist is None or probs is None:
      return None
    return probs.numel() * probs[hist.clamp(min=0)]
