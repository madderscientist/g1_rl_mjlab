# 脚步跟踪

平地脚印跟随，控制12腿轴和3腰轴。连续任务名为 `G1-Gloria-FootstepTracking`，单步任务名为 `G1-Gloria-FootstepEpisode`。

## 阅读导航

| 目的 | 入口 |
| --- | --- |
| 选择训练阶段、初始化或迁移 | [训练阶段](#训练阶段) |
| 恢复、保存或停止训练 | [续训与运行管理](#续训与运行管理) |
| 查观测与网络接口 | [策略接口](#策略接口)、[模型与部署契约](MODEL_CONTRACT.md) |
| 查采样、奖励与安全规则 | [环境与训练规则](#环境与训练规则)、[单步场景规则](step_episode/README.md) |
| 验证或回放检查点 | [验证](#验证)、[Viser回放](#viser-回放) |
| 定位实现 | [代码入口](#代码入口)、[独立脚步模块](../../footsteps/README.md) |

<a id="训练与续训"></a>
<a id="指令课程"></a>

## 训练阶段

主流程：基础行走 → 精度训练 → 联合训练 → Theta训练。单步预训练可选。

| 阶段 | 输入 | 目标 |
| --- | --- | --- |
| [1. 基础行走](#第一阶段基础行走) | 从零初始化 | 固定步频，从直行扩展到全向、变步距/站距/脚yaw |
| [2. 精度训练](#第二阶段精度训练) | 第一阶段检查点 | 保留空间课程，开启步频游走、精确引导和落脚代价 |
| [3. 联合训练](#第三阶段联合训练) | 精度检查点，可选单步critic种子 | 一个共享actor，同时学习连续走与单步，两任务各有私有critic |
| [4. Theta训练](#第四阶段theta迁移与训练) | 已训练的第三阶段模型，经显式迁移 | 只推进躯干期望偏角课程，固定旧课程参数 |

命令在仓库根目录执行；检查点和目录变量需自行设置。前三阶段沿用累计课程计数，第四阶段另设课程起点。

### 第一阶段：基础行走

使用 `--profile walk-first` 从零开始，示例为双GPU、每卡128环境：

```bash
OMP_NUM_THREADS=1 MUJOCO_GL=egl micromamba run -n mj python scripts/train_footstep_resume.py \
  --profile walk-first --from-scratch --envs-per-rank 128 \
  --run-root logs/rsl_rl/footstep_walk_first --tag walk_first --save-interval 1000
```

步频固定1.2Hz。0–1499轮直行、步距0.27m、站距0.24m；1500轮起扩大方向与站距，2500轮起扩大步距和回合内变向，4000轮达到全方向、步距0.12–0.48m、站距0.12–0.36m。脚yaw在4500/5000/5500轮分别放开至±10°/±20°/±30°。

### 第二阶段：精度训练

设置 `WALK_CHECKPOINT` 为第一阶段完整检查点。保留模型、优化器和课程，示例追加2000轮：

```bash
OMP_NUM_THREADS=1 MUJOCO_GL=egl micromamba run -n mj python scripts/train_footstep_resume.py \
  "$WALK_CHECKPOINT" --profile precision --max-updates 2000 \
  --save-interval 1000 --tag precision
```

| 设置 | 第一阶段 `walk-first` | 第二阶段 `precision` |
| --- | --- | --- |
| 行走步频 | 固定1.2Hz | 0.8–1.8Hz游走，reset/重新起步以1.2Hz为目标 |
| XY引导：权重 / 核宽 | 1 / 0.40m | 5 / 0.20m |
| yaw引导：权重 / 核宽 | 0.5 / 0.30rad | 5 / 0.10rad |
| 摆动进度权重 | s | s² |
| 落脚事件权重 | 关闭 | 0渐入至-0.1，默认2000轮 |
| 位置 / yaw代价尺度 | 不适用 | 0.05m / 0.20rad，无免罚区 |

步频变化率每2秒采样，按累计0/800/2000轮取±0.01/±0.15/±0.3Hz/s，在频率边界反射；执行频率变化不超过0.2Hz/s。站立f=0，起停不受行走频率下限限制。落脚渐入进度随 `precision_stage` 恢复。

### 第三阶段：联合训练

设置 `PRECISION_CHECKPOINT` 为第二阶段检查点，`NEW_JOINT_RUN_DIR` 为尚不存在的运行目录。首次建立联合模型：

```bash
OMP_NUM_THREADS=1 MUJOCO_GL=egl micromamba run -n mj python scripts/train_footstep_joint.py \
  --walking-checkpoint "$PRECISION_CHECKPOINT" --run-dir "$NEW_JOINT_RUN_DIR" \
  --continuous --max-updates 200 --envs-per-rank 1024 --save-interval 200 \
  --task-weights 0.4 0.6 --learning-rate 1e-5 --distributed-backend nccl
```

actor与连续critic来自精度检查点；单步critic默认复制同一初值，之后独立更新。首次初始化建立两套新Adam。两任务分别使用 `precision` 与 `step-episode`，参数见[联合训练参数](#联合训练参数)。

### 可选分支：单步预训练

在第二阶段后独立执行：

```bash
OMP_NUM_THREADS=1 MUJOCO_GL=egl micromamba run -n mj python scripts/train_footstep_resume.py \
  "$PRECISION_CHECKPOINT" --profile step-episode --max-updates 2000 \
  --run-root logs/rsl_rl/footstep_single_step --tag single_step --save-interval 200
```

同时训练actor与critic，探索std固定0.25。将结果设为 `STEP_CHECKPOINT`，第三阶段首次初始化时增加 `--step-checkpoint "$STEP_CHECKPOINT"`；仅采用其critic和环境计数，不采用actor或Adam。

### 第四阶段：Theta迁移与训练

源模型须为完成旧课程的v5联合检查点，包含共享actor、两个私有critic、两套Adam和任务计数；单任务模型或已迁移模型不能作为输入。

`JOINT_CHECKPOINT` 指向源模型；`NEW_THETA_SEED` 和 `NEW_THETA_RUN_DIR` 分别为新文件、新目录，不复用旧阶段目录。

```bash
# 迁移，生成新检查点和.migration.json回执
OMP_NUM_THREADS=1 MUJOCO_GL=egl micromamba run -n mj python scripts/migrate_footstep_theta.py \
  "$JOINT_CHECKPOINT" "$NEW_THETA_SEED"

# 从迁移种子启动第四阶段
OMP_NUM_THREADS=1 MUJOCO_GL=egl micromamba run -n mj python scripts/train_footstep_joint.py \
  --resume-joint "$NEW_THETA_SEED" --run-dir "$NEW_THETA_RUN_DIR" \
  --continuous --max-updates 200 --envs-per-rank 1024 --save-interval 200 \
  --task-weights 0.4 0.6 --learning-rate 1e-5 --distributed-backend nccl
```

迁移保留旧权重、归一化、Adam矩与步数、累计轮号，为新增输入补零权重与零矩；不训练或覆盖源模型。迁移种子尚需学习theta跟踪。

本阶段只推进 `torso_heading_range`，固定旧课程，保留随机指令、扰动和控制。腰yaw回零改为theta跟踪，移除 `standing_pelvis_upright`；连续走与单步仍联合训练。

theta是torso相对实际双脚圆周平均航向的期望偏角，以一维rad输入。奖励为 `-2.0 * wrap(actual_offset-theta)^2`，站立与行走均启用。

课程按 `(当前任务计数-迁移起点) // 64` 选档，以下为第四阶段新增PPO轮数：

| 新增轮数 | 均匀采样目标范围 |
| --- | --- |
| 0–999 | ±20° |
| 1000–2499 | ±40° |
| 2500–4999 | ±60° |
| 5000及以后 | ±90° |

每环境独立每8–12秒采样目标与5–30°/s速度，从当前theta限速过渡，不超调；episode reset置零。参数见 `TorsoHeadingCommandCfg` 与 `TORSO_HEADING_STAGES`。

续训时设置 `THETA_CHECKPOINT` 和 `THETA_RUN_DIR` 为第四阶段存档与已有目录，直接恢复，不再次迁移：

```bash
OMP_NUM_THREADS=1 MUJOCO_GL=egl micromamba run -n mj python scripts/train_footstep_joint.py \
  --resume-joint "$THETA_CHECKPOINT" --run-dir "$THETA_RUN_DIR" \
  --continuous --max-updates 200 --envs-per-rank 1024 --save-interval 200 \
  --task-weights 0.4 0.6 --learning-rate 1e-5 --distributed-backend nccl
```

首次迁移设定新增第0轮；续训保留课程原点和进度。第四阶段仅由 `infos.theta_stage` 显式启用，默认任务不含theta。

## 续训与运行管理

### 恢复已有训练

单任务用 [train_footstep_resume.py](../../../scripts/train_footstep_resume.py) 和对应 `--profile` 完整恢复，不加 `--reset-optimizer`。

第三阶段联合续训时，设置 `JOINT_CHECKPOINT` 为完整联合存档、`JOINT_RUN_DIR` 为已有目录：

```bash
OMP_NUM_THREADS=1 MUJOCO_GL=egl micromamba run -n mj python scripts/train_footstep_joint.py \
  --resume-joint "$JOINT_CHECKPOINT" --run-dir "$JOINT_RUN_DIR" \
  --continuous --max-updates 200 --envs-per-rank 1024 --save-interval 200 \
  --task-weights 0.4 0.6 --learning-rate 1e-5 --distributed-backend nccl
```

第四阶段使用[对应续训命令](#第四阶段theta迁移与训练)。联合恢复保留共享actor、私有critic/Adam和任务计数；已有目录优先选累计轮号最大的完整 `model_*.pt`。首次初始化用 `--walking-checkpoint`，续训用 `--resume-joint`，不能混用。

### 联合训练参数

第三、第四阶段示例均采用以下设置：

| 项目 | 设置 |
| --- | --- |
| 任务 / 设备 | GPU0连续走，GPU1单步；每卡1024环境 |
| 采样与更新 | 每轮64拍，5epochs / 4minibatches |
| 共享actor | `0.4*g_walking + 0.6*g_step`，NCCL SUM allreduce |
| 任务私有状态 | critic、Adam、GAE和优势标准化各自独立 |
| PPO参数 | 学习率1e-5，gamma0.99，lambda0.95，entropy0 |
| 探索与归一化 | std0.25固定，actor normalizer冻结 |

### 保存、停止与审计

- **持续运行**：`--continuous --max-updates 200`无限运行，200为分段长度；`--save-interval 200`每200轮存档。单任务不指定时长或轮数也持续运行。
- **正常停止**：run根目录创建 `STOP`，或向worker发送SIGTERM/SIGINT；完整更新后保存退出，重启前移除 `STOP`。
- **恢复边界**：从最新完整存档和新episode继续，不恢复仿真现场或回合随机流。
- **审计**：`initialization.json`核对模型与Adam恢复；`status_rank_*.json`看进度，`launch.json`看命令。TensorBoard与环境配置由rank0写入。
- **源码快照**：不改写运行中或历史快照；换实现时另建快照并完整恢复。一个run目录只供一份训练写入。

## 策略接口

| 接口 | 第一至第三阶段、可选单步预训练 | 第四阶段 |
| --- | --- | --- |
| 版本 / actor类 | v5 / `LegacyFootstepActor` | v7 / `FootstepActor` |
| Actor原始 / 编码维度 | 78 / 81 | 79 / 82 |
| Critic维度 | 104 | 105 |
| 额外指令 | 无theta | 一维期望偏角theta |
| 契约 | `g1_footstep_gru_v5_next_only` | `g1_footstep_gru_v7_torso_heading_command` |

Actor使用GRU(64)和MLP(256,256,128)，输出15轴位置增量；critic使用MLP(256,128)。脚印输入为 `[L_next,R_next]`。v5不能直接加载到v7；导出用 `export_footstep_policy`，回放和部署匹配源码与接口。详细布局见[模型与部署契约](MODEL_CONTRACT.md)。

## 环境与训练规则

### 指令与完成时序

连续任务reset时站立/行走各50%，随机起脚侧。指令每3–8秒更新，停止概率30%，站立保持2–5秒；起停渐变各0.5秒，停脚需2拍双接触确认，超时0.5秒判故障。f为左右完整周期/s，f=0站立。单步时序见[单步场景规则](step_episode/README.md)。

奖励读完成拍目标，actor读下一拍next目标；reset后刷新实测脚位，普通换步和停走保持GRU状态。[TensorFootstepManager](../../footsteps/tensor_manager.py)在CUDA上默认编译，NumPy版本用于部署与Web预览。

单任务须显式指定 `--profile`；默认 `tracking` 是兼容配方，不属于四阶段主流程。课程实现见 [curriculum.py](curriculum.py) 与 [walking/curriculum.py](walking/curriculum.py)。

### 奖励与约束

常规奖励乘控制周期0.02秒；落脚与非超时终止按事件计分。第一阶段关闭落脚事件，第二阶段渐入至-0.1，单步直接使用-0.1。姿态、站定、接触、铜损和对称项的公式与权重见[奖励契约](MODEL_CONTRACT.md#10-训练奖励契约)，单步专属项见[单步说明](step_episode/README.md)。奖励中的限位代价不是硬件保护。

### 初始化与随机化

- 手臂14轴reset归零，5秒过渡到随机PD目标；每8–12秒采样新姿势，再插值5秒。幅度在0/1500/2000轮对应0.2/0.5/1.0，前800轮保持0.2。
- 质量、负载、摩擦、编码器偏置与质心随机化见 [lower_body事件](../lower_body/cfg/events.py)。
- Actor训练启用观测噪声与回合IMU偏置；critic和play关闭观测噪声，保留编码器偏置，观测延迟为0。

### 终止与数值保护

控制周期0.02秒，物理步长0.0025秒，每拍8个子步；连续回合60秒。

- 支撑脚偏离目标超过1m、规划器故障或接触确认超时，立即终止；回合到期截断。
- 骨盆倾角超过70°或离最低足底低于0.35m，每拍以0.005概率终止。
- 关节速度超过120rad/s、根线速度超过20m/s、根角速度超过80rad/s或状态非有限，触发子步保护并记录、隔离、reset。

仿真测试与无噪声回放不能代替实机安全验证。

## 验证

分进程运行，避免共享Dynamo缓存触及重编译上限：

```bash
OMP_NUM_THREADS=1 MUJOCO_GL=egl micromamba run -n mj python -m pytest -q \
  tests/test_footstep_precision.py tests/test_footstep_resume.py tests/test_footstep_supervised.py \
  tests/test_footstep_walk_first.py tests/test_footstep_walk_curriculum.py tests/test_step_episode.py \
  tests/test_footstep_posture.py tests/test_footstep_standing_copper.py \
  tests/test_footstep_tracking_objectives.py tests/test_footstep_leg_symmetry.py \
  tests/test_arm_reset_ranges.py tests/test_footstep_arm_curriculum.py

OMP_NUM_THREADS=1 MUJOCO_GL=egl micromamba run -n mj python -m pytest -q \
  tests/test_footstep_still_velocity.py tests/test_footstep_numerical_safety.py

# CPU双进程：actor加权同步、私有critic/Adam及联合恢复自测，不启动正式训练
OMP_NUM_THREADS=1 MUJOCO_GL=egl micromamba run -n mj python scripts/train_footstep_joint.py --self-test
```

评测关注XY/yaw误差、存活时间、起停成功率、力矩与接触冲击。

## Viser 回放

使用[检查点回放入口](../../../run_logs/play_footstep_checkpoint.py)，显式指定 `step-episode` 或 `precision`。

### 准备源码与检查点

设置 `FROZEN_SOURCE_ROOT` 为模型对应的源码快照，`PLAYBACK_RUN_DIR` 为运行目录。两路回放固定使用同一存档：

```bash
export REPO_ROOT="$(pwd)"
export SOURCE_ROOT="${FROZEN_SOURCE_ROOT:?请先设置目标阶段的源码快照目录}"
RUN_DIR="${PLAYBACK_RUN_DIR:?请先设置目标阶段的运行目录}"
export CHECKPOINT="$(find "$RUN_DIR" -maxdepth 1 -type f -name 'model_*.pt' | sort -V | tail -n 1)"

[[ -f "$CHECKPOINT" && -d "$SOURCE_ROOT/g1_lower_rl" ]] && \
  printf 'export REPO_ROOT=%q\nexport SOURCE_ROOT=%q\nexport CHECKPOINT=%q\n' \
  "$REPO_ROOT" "$SOURCE_ROOT" "$CHECKPOINT"
```

将打印的三行 `export` 在第二个终端执行；没有输出则检查路径。`PYTHONPATH` 选择源码，脚本不接受 `--source`。

### 单步：GPU1，端口8086

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl \
PYTHONPATH="$SOURCE_ROOT:$SOURCE_ROOT/scripts" \
  "$HOME/.local/bin/micromamba" run -n mj python \
  "$REPO_ROOT/run_logs/play_footstep_checkpoint.py" "$CHECKPOINT" \
  --profile step-episode --port 8086 --label "Single step"
```

打开 `http://127.0.0.1:8086/`：站1秒、随机迈步、保持5秒后reset；手臂目标为零。

### 连续步：GPU0，端口8085

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MUJOCO_GL=egl \
PYTHONPATH="$SOURCE_ROOT:$SOURCE_ROOT/scripts" \
  "$HOME/.local/bin/micromamba" run -n mj python \
  "$REPO_ROOT/run_logs/play_footstep_checkpoint.py" "$CHECKPOINT" \
  --profile precision --port 8085 --label "Continuous walking"
```

打开 `http://127.0.0.1:8085/`：保留随机脚印、停走和步频游走；手动频率请求仍按速率限制过渡。

### 访问、核验与停止

- 远程访问需转发8085/8086；端口占用时修改 `--port`。
- `CUDA_VISIBLE_DEVICES`选择物理GPU；进程内 `cuda:0` 对应该卡。
- 核对 `PLAYBACK_READY` 的源码、profile、检查点哈希与 `actor_exactly_loaded=true`；两路哈希应相同。
- `--check-steps`只做有限步检查，不启动网页。回放固定启动时的模型，不热更新。
- Ctrl+C关闭回放，不影响后台训练；回放会占用GPU资源。

### 可选：手动前移调试

沿用上述变量，在独立终端运行：

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl \
PYTHONPATH="$SOURCE_ROOT:$SOURCE_ROOT/scripts" \
  "$HOME/.local/bin/micromamba" run -n mj python \
  "$REPO_ROOT/run_logs/play_footstep_checkpoint.py" "$CHECKPOINT" \
  --profile precision --manual-step --port 8087 --label "Manual forward step"
```

打开 `http://127.0.0.1:8087/`。`x`为前移距离（默认0.10m，范围0–0.35m），`f`为步频（0.8–1.8Hz）；默认两次落脚，`--single-foot`只迈选中脚。此模式关闭随机调度并约束手臂零位，执行中锁定参数，失败时暂停保留现场；不是单步训练场景。

## 代码入口

| 文件 | 职责 |
| --- | --- |
| [统一工厂](./__init__.py)、[env_cfg.py](env_cfg.py)、[commands.py](commands.py) | 默认任务配方、共享物理场景、观测与命令时序 |
| [walking/env_cfg.py](walking/env_cfg.py)、[walking/precision.py](walking/precision.py) | 第一、第二阶段连续配方 |
| [curriculum.py](curriculum.py)、[walking/curriculum.py](walking/curriculum.py) | 手臂幅度、步频、空间采样和theta阶段表 |
| [step_episode/env_cfg.py](step_episode/env_cfg.py)、[commands.py](step_episode/commands.py)、[rewards.py](step_episode/rewards.py) | 单步场景、回合时序、失败和落脚事件 |
| [theta_stage.py](theta_stage.py)、[迁移工具](../../../scripts/migrate_footstep_theta.py) | 第四阶段独立工厂、输入与Adam状态迁移 |
| [rewards_cfg.py](rewards_cfg.py)、[rewards.py](rewards.py)、[leg_symmetry.py](leg_symmetry.py) | 奖励权重、公式和腿部对称状态 |
| [rl_cfg.py](rl_cfg.py)、[策略模型](../../rl/footstep_model.py) | PPO基础配置、模型接口与导出 |
| [单任务入口](../../../scripts/train_footstep_resume.py)、[联合入口](../../../scripts/train_footstep_joint.py) | 初始化、完整恢复、持续训练和原子存档 |
| [回放入口](../../../run_logs/play_footstep_checkpoint.py) | 连续、单步和手动Viser回放 |

[监督工具](../../../scripts/train_footstep_supervised.py)与[实验调度工具](../../../scripts/footstep_campaign.py)保留固定预算/独立任务实验兼容，不属于四阶段主流程。