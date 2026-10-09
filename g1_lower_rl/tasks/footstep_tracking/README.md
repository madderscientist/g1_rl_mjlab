# 脚步跟踪

平地脚印跟随，控制12腿轴和3腰轴。连续任务名为 `G1-Gloria-FootstepTracking`，单步任务名为 `G1-Gloria-FootstepEpisode`。

## 阅读导航

| 目的 | 入口 |
| --- | --- |
| 初始化或恢复训练 | 本页[阶段总览](#阶段总览)、[训练与续训](#训练与续训) |
| 查单步时序、奖励与随机化 | [单步场景规则](step_episode/README.md) |
| 查观测布局、ONNX、执行器与奖励公式 | [模型与部署契约](MODEL_CONTRACT.md) |
| 查独立脚印规划器 | [脚步模块](../../footsteps/README.md) |
| 启动单步或连续步Viser回放 | 本页[Viser回放](#viser-回放) |

## 阶段总览

主流程为基础行走、精度训练、联合训练。单步预训练是可选分支，不替代连续行走，也不是进入联合训练的前置条件。

| 阶段 | 入口 | 目标与对应代码 |
| --- | --- | --- |
| 1. 基础行走与空间拓展 | `--profile walk-first` | [walking/env_cfg.py](walking/env_cfg.py)设置固定1.2Hz及宽容引导；[walking/curriculum.py](walking/curriculum.py)按累计轮数从直行扩展到全向、变步距/站距/脚yaw |
| 2. 扩展与精度 | `--profile precision` | [walking/precision.py](walking/precision.py)保留空间课程，开启步频游走、精确摆动引导及落脚代价渐入 |
| 3. 联合训练，当前方案 | 首次 `--walking-checkpoint`；续训 `--resume-joint` | [联合入口](../../../scripts/train_footstep_joint.py)使用一个共享actor、连续/单步两个私有critic，按40/60合并actor梯度 |
| 2.5. 可选的单步预训练 | `--profile step-episode` | [step_episode/env_cfg.py](step_episode/env_cfg.py)配置回合与专属奖励；[设备规划器](../../footsteps/step_episode.py)执行先站、迈步、再站；可为联合训练提供单步critic种子 |

首次联合初始化时，共享actor与连续critic来自精度检查点。省略`--step-checkpoint`时，单步critic也从同一检查点复制，但它与连续critic是两套独立参数，使用各自任务的回报更新。若提供单步预训练检查点，仅采用其critic与环境计数，不采用其actor或Adam；预训练critic换到共享actor后仍需适应。连续与单步的奖励、终止和未来目标不同，因此采用两个任务私有critic，而不是要求两个critic的初值不同。

统一配置入口是 [make_footstep_env_cfg](./__init__.py)，单任务续训与联合入口都使用它。工作区按任务场景分为`walking/`与`step_episode/`；profile字符串、工厂函数名及checkpoint键不变。单步包入口保留原有公开符号；旧行走模块路径迁至`walking`，工作区调用同步更新，历史冻结源码保持原样。

空间课程的扩展已经包含在阶段1中，并由阶段2继续使用；不是另一个需要重新初始化的策略。阶段切换由命令显式指定，课程计数从检查点恢复，不自动回到零。

| 设置 | 第一阶段 `walk-first` | 第二阶段 `precision` |
| --- | --- | --- |
| 行走步频 | 固定1.2Hz | 0.8–1.8Hz游走，reset/重新起步以1.2Hz为目标 |
| XY引导：权重 / 核宽 | 1 / 0.40m | 5 / 0.20m |
| yaw引导：权重 / 核宽 | 0.5 / 0.30rad | 5 / 0.10rad |
| 摆动进度权重 | s | s² |
| 落脚事件权重 | 关闭 | 0渐入至-0.1，默认2000轮 |
| 位置 / yaw代价尺度 | 不适用 | 0.05m / 0.20rad，无免罚区 |

此表中的早期阶段差异仍保留，用于重建训练流程，不把旧阶段权重覆盖到当前联合训练。当前连续场景使用 `precision`，单步场景使用 `step-episode`。

`tracking` 是兼容基础配方，落脚事件权重-1；不是当前连续训练阶段。直接续训脚本仍保留原默认值，因此使用时应显式指定 `--profile`。

## 代码分工

| 文件 | 职责 |
| --- | --- |
| [env_cfg.py](env_cfg.py)、[commands.py](commands.py) | 共享物理场景、84维观测、零延迟、命令时序与完成拍奖励快照 |
| [curriculum.py](curriculum.py)、[walking/curriculum.py](walking/curriculum.py) | 手臂幅度、步频变化率与空间采样课程；不修改共享奖励权重 |
| [walking/env_cfg.py](walking/env_cfg.py)、[walking/precision.py](walking/precision.py) | 连续行走的基础与精度配方，共用场景和课程 |
| [step_episode/env_cfg.py](step_episode/env_cfg.py) | 单步场景配方，组合共享场景与单步专属项 |
| [step_episode/commands.py](step_episode/commands.py)、[step_episode/rewards.py](step_episode/rewards.py) | 单步完成与统计、失败事件、落脚奖金和站立防抖 |
| [rewards_cfg.py](rewards_cfg.py)、[rewards.py](rewards.py)、[leg_symmetry.py](leg_symmetry.py) | 共享权重、奖励公式和腿部对称状态 |
| [rl_cfg.py](rl_cfg.py) | 单任务PPO基础配置；联合入口覆盖学习率、熵系数和固定探索 |
| [续训入口](../../../scripts/train_footstep_resume.py) | 单任务初始化、阶段切换、完整恢复与停止保存 |
| [联合入口](../../../scripts/train_footstep_joint.py) | actor同步、私有critic/Adam、原子联合存档和持续训练 |

模型/部署接口见[模型契约](MODEL_CONTRACT.md)，独立几何和停走规划见[脚步模块](../../footsteps/README.md)，单步规则见[单步说明](step_episode/README.md)。

## 策略接口

- Actor：84维输入，编码为89维，经GRU(64)和MLP(256,256,128)输出15轴位置增量。
- 输入：29轴角度与速度、骨盆/躯干角速度与重力、相位、频率、四个XY/yaw脚印。
- 脚印顺序：`[L_support,R_support,L_next,R_next]`。support在计划落地时更新，next包含正在迈向的目标。
- Critic：104维，额外观测线速度、接触和上一拍动作，使用MLP(256,128)。
- 控制周期0.02s，PPO每轮64拍，回合60s。契约为 `g1_footstep_gru_v4`。

物理步长0.0025s，每控制拍8个子步，约束容量1024。子步数值保护在关节速度超过120rad/s、根线速度超过20m/s、根角速度超过80rad/s或状态非有限时隔离对应环境，记录现场并按失败终止后reset；正常倒地仍使用概率终止。首10次事件保存状态、控制、外力与随机化模型字段，后续持续记录摘要。

## 训练与续训

以下是流程示例，在仓库根目录运行；已有联合训练只使用联合恢复命令，不重新初始化。环境使用 `micromamba run -n mj`，无头仿真设置 `MUJOCO_GL=egl`。

`WALK_CHECKPOINT`、`PRECISION_CHECKPOINT`、`JOINT_CHECKPOINT` 指向对应完整检查点；`NEW_JOINT_RUN_DIR` 必须尚不存在，`JOINT_RUN_DIR` 是既有联合运行目录。这些变量应由操作者设置，不绑定某个历史轮号。

```bash
# 第一阶段：双GPU、每卡128环境，从零学习行走
OMP_NUM_THREADS=1 MUJOCO_GL=egl micromamba run -n mj python scripts/train_footstep_resume.py \
  --profile walk-first --from-scratch --envs-per-rank 128 \
  --run-root logs/rsl_rl/footstep_walk_first --tag walk_first --save-interval 1000

# 第二阶段：保留模型、优化器及课程，追加2000轮精度训练
OMP_NUM_THREADS=1 MUJOCO_GL=egl micromamba run -n mj python scripts/train_footstep_resume.py \
  "$WALK_CHECKPOINT" --profile precision --max-updates 2000 \
  --save-interval 1000 --tag precision

# 第三阶段：直接从精度检查点首次建立联合模型，跳过单步预训练
OMP_NUM_THREADS=1 MUJOCO_GL=egl micromamba run -n mj python scripts/train_footstep_joint.py \
  --walking-checkpoint "$PRECISION_CHECKPOINT" --run-dir "$NEW_JOINT_RUN_DIR" \
  --continuous --max-updates 200 --envs-per-rank 1024 --save-interval 200 \
  --task-weights 0.4 0.6 --learning-rate 1e-5 --distributed-backend nccl

# 当前主流程：恢复既有联合训练；200只是内部更新分段，不是总预算
OMP_NUM_THREADS=1 MUJOCO_GL=egl micromamba run -n mj python scripts/train_footstep_joint.py \
  --resume-joint "$JOINT_CHECKPOINT" --run-dir "$JOINT_RUN_DIR" \
  --continuous --max-updates 200 --envs-per-rank 1024 --save-interval 200 \
  --task-weights 0.4 0.6 --learning-rate 1e-5 --distributed-backend nccl
```

可选的单步预训练可在精度训练后独立执行，不是上述联合启动命令的前置步骤：

```bash
OMP_NUM_THREADS=1 MUJOCO_GL=egl micromamba run -n mj python scripts/train_footstep_resume.py \
  "$PRECISION_CHECKPOINT" --profile step-episode --max-updates 2000 \
  --run-root logs/rsl_rl/footstep_single_step --tag single_step --save-interval 200
```

若要采用预训练critic，在首次联合命令中额外指定`--step-checkpoint "$STEP_CHECKPOINT"`，其中`STEP_CHECKPOINT`是单步完整检查点。联合入口不支持随机初始化actor和两个critic；这里的“首次”是从已有精度模型开始联合学习，不是从零训练。

- **单任务恢复**：actor、critic、normalizer、Adam、学习率和计数恢复，从下一轮的新episode开始。单步预训练同时训练actor和critic，只将探索标准差固定为0.25，均值网络、critic和Adam不重置。不要用`--reset-optimizer`续训当前方案。
- **落脚课程**：`precision_stage`保存渐入起点和时长，续训不重新渐入；首次切换默认从检查点计数开始。单步落脚项立即使用-0.1。
- **首次建立联合模型**：`--walking-checkpoint`必填，`--step-checkpoint`可选；此路径初始化新的两套联合Adam，不能用它代替当前`--resume-joint`的完整恢复。
- **联合恢复**：同一原子检查点保存`actor_state_dict`、`task_critic_state_dicts`、`task_optimizer_state_dicts`和`task_environment_counters`。每卡恢复对应critic、Adam及计数，逐张量核验；每次更新检查actor副本一致。既有run目录会选择数值轮号最大的完整`model_*.pt`，不选历史最佳或旧种子。
- **当前联合参数**：GPU0连续走、GPU1单步；每卡1024环境、64拍rollout；actor梯度为`0.4*g_walking + 0.6*g_step`，NCCL SUM allreduce；critic、GAE和优势标准化各自独立。学习率1e-5，5epochs/4minibatches，gamma0.99、lambda0.95、entropy0，探索std0.25固定，actor normalizer冻结。
- **持续与保存**：`--continuous`无限期运行，`--max-updates 200`仅为内部段长度，每200次更新同步存档。单任务入口不指定`--hours`或`--max-updates`也持续运行；两种入口不要混用恢复格式。
- **正常停止**：在联合run根目录创建`STOP`或向worker发送SIGTERM/SIGINT，双卡协调在完整更新后保存退出；残留`STOP`会阻止重启。训练结束/崩溃后的恢复从最新完整联合存档继续，不恢复仿真现场或回合随机流。
- **审计文件**：联合`initialization.json`核对`actor_mean_exact`、`critic_exact`、`optimizer_exact`及`optimizer_reset=false`；`status_rank_*.json`核对进度；rank1不生成TensorBoard或`params/env.yaml`。`launch.json`记录命令，记录文件不是恢复状态的替代品。
- **源码与回放**：当前运行使用独立冻结源，编辑工作区不会热更新训练。不要回写正在使用或历史快照；需要应用新实现时另建快照并正常完整恢复。回放指定对应源码、检查点和profile，导出使用`export_footstep_policy`。

`train_footstep_supervised.py`与`footstep_campaign.py`是已有固定预算/独立任务实验工具，保留兼容，不属于当前联合持续训练主流程；不要另开一份与当前训练争用同一run目录。

## 指令课程

**共同调度：** reset时50%站立、50%直接行走，随机起脚侧；指令每3–8s更新，停止概率30%，站立保持2–5s。
起停渐变各0.5s；停脚后连续2拍双接触确认，等待超过0.5s判故障。频率f表示完整左右周期/s，f=0为站立。

**walk-first课程：** 0–1499轮固定直行、总步距0.27m、站距0.24m；1500轮起扩大方向与站距，2500轮起扩大步距和回合内变向，4000轮达到全方向、步距0.12–0.48m、站距0.12–0.36m。逐脚yaw在4500/5000/5500轮分别放开至±10°/±20°/±30°。奖励精度保持表中设置。

**precision步频：** 正常行走每2秒重采样一次有符号变化率，在0.8–1.8Hz边界反射；执行频率每秒变化不超过0.2Hz。变化率课程复用0/800/2000轮的±0.01/±0.15/±0.3Hz/s，按恢复后的累计轮数选档，不重新渐入。站立仍为0，起停渐变不受正常行走的0.8Hz下限限制。

**tracking课程：** 回合内方向增量在0/1500/2500/4000轮取0°/±45°/±90°/±180°；频率变化率在0/800/2000轮取±0.01/±0.15/±0.3Hz/s。

课程按 `common_step_counter // 64` 在reset时选档，作用于后续采样的目标。
walk-first配置见 [walking/env_cfg.py](walking/env_cfg.py)，阶段表见 [walking/curriculum.py](walking/curriculum.py)；共享课程见 [curriculum.py](curriculum.py)。

## 第二阶段约束

常规项每拍贡献为 `weight * raw_value * 0.02`；落脚和非超时终止按事件计分。

- **落脚精度**：每目标离地后首次触地扣 `0.1 * (0.5*d/0.05 + 0.5*abs(yaw_error)/0.20)`，渐入结束后使用完整系数；不按支撑时长重复收费，f=0关闭。
- **站定**：脚位相对冻结目标超过3cm才计罚；所有脚的XYZ速度超过0.02m/s计罚，包含离地脚；零速度正奖励乘 `1 / (1 + mean(d²)/0.10²)`，限制碎步漂移。
- **关节姿态**：髋yaw归零-0.1、腰yaw归零-0.8；六对腿关节EMA对称项-0.1，允许正常交替摆动。腰roll/pitch只保留避限位项-2，不要求关节归零；躯干竖直项-2负责姿态。
- **平衡与能耗**：保留接触节拍、足高、躯干/骨盆直立、头高、膝弯、足底平放、动作平滑和铜损。铜损基础权重-0.25，站立乘2；全部15轴在硬限位附近0.1°且力矩向外时乘50，站立合计乘100。向内卸载不乘限位倍率；这是软代价，不是硬件保护。

当前连续场景30项、单步33项，完整定义见[奖励契约](MODEL_CONTRACT.md#10-训练奖励契约)，工厂与公式见 [rewards_cfg.py](rewards_cfg.py)、[rewards.py](rewards.py)。第一阶段不含落脚事件，共29项。单步保留共享权重，但失败改为-100，并额外加入落脚奖金10、站立关节速度-0.2及站立动作差分-0.04；详见[单步说明](step_episode/README.md)。本次整理不重新调权。

腿部对称项：pitch/knee取左减右，roll/yaw取左加右；先EMA再计算代价。时间常数为 `min(1/f, 2s)`，f=0时为1秒，新样本权重为 `1-exp(-dt/tau)`。仅稳定行走且计划移动方向与脚朝向相差不超过15°时启用；站定时要求两脚目标的前后错位不超过5cm。两种模式均要求目标脚yaw相对计划朝向的镜像和偏差不超过30°。起停和不符合条件时清空历史；reset、重新启用或意图变更时按当前偏差初始化并重新渐入1秒。同拍重复计算不推进滤波，局部reset不影响其他环境。实现见 [leg_symmetry.py](leg_symmetry.py)。

## 初始化与随机化

- 手臂14轴实际角度、速度和初始PD目标设为0。最终目标独立采样 `U(lower-pi/2, upper+pi/2) * target_scale`，每个环境在reset后5秒内线性渐变到终点。
- 每个环境每8–12秒从 `ARM_TARGET_RANGES` 采样新姿势并乘 `target_scale`，从当前PD目标线性插值5秒；过渡期间不重采样，不改实际关节角和速度。
- `target_scale` 在0–800轮为0.2，随后线性升至1500轮0.5、2000轮1.0；只影响后续reset和定时采样。越界目标及渐变路径可能产生限位力矩或自碰撞。
- 基础随机化复用 [lower_body事件](../lower_body/cfg/events.py)，包括连杆质量、夹爪负载、脚底摩擦、编码器偏置和质心偏移。主动手臂力矩与身体平移推力处于初始档，身体冲量仍含±2Nm力矩；定时手臂目标插值独立启用。
- 训练actor使用高斯观测噪声：关节角0.006rad、关节速度0.87rad/s、IMU角速度0.115rad/s、重力分量0.029；IMU偏置每回合采样。critic和play关闭观测噪声，编码器偏置仍保留；观测延迟为0。

## 终止与时序

- 计划支撑脚距当前执行目标超过1m、规划器故障或停脚接触确认超时，立即终止；60s回合到期截断。
- 骨盆倾角超过70°或离最低足底低于0.35m时，每拍以0.005概率终止，两项共享同一次抽样。
- `footstep_fault` 在物理步后推进命令，奖励读取完成拍的频率和切换前目标；actor读取下一拍support/next。
- reset在运动学刷新后读取实测脚位，局部reset只更新对应环境；普通换步和停走保持GRU状态。

训练使用设备驻留的 [TensorFootstepManager](../../footsteps/tensor_manager.py)，CUDA默认编译执行。NumPy生成器用于独立部署与Web预览。

## 验证

仿真测试分进程运行，避免大量配方共享Dynamo编译缓存时触及重编译次数上限；不通过修改训练配置绕过测试。

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

评测关注XY/yaw误差、存活时间、起停成功率、力矩与接触冲击。测试和仿真回放不能代替实机安全验证。

## Viser 回放

统一使用[检查点回放入口](../../../run_logs/play_footstep_checkpoint.py)。单步选`--profile step-episode`，连续步选`--profile precision`；必须显式指定，联合训练目录的manifest不提供单一profile，不能依赖自动推断。

### 准备源码与检查点

在仓库根目录的第一个终端执行。以下示例对应当前联合运行，先选定数值轮号最大的已保存`model_*.pt`，两路回放共用这一文件；不读取`.pt.tmp`，也不在第二个终端重新挑选最新模型。

```bash
export REPO_ROOT="$(pwd)"
export SOURCE_ROOT="$REPO_ROOT/run_logs/footstep_joint_posture_limit50_source_20261009"
RUN_DIR="$REPO_ROOT/logs/rsl_rl/footstep_joint_step60_continuous_20261008"
export CHECKPOINT="$(find "$RUN_DIR" -maxdepth 1 -type f -name 'model_*.pt' | sort -V | tail -n 1)"

[[ -f "$CHECKPOINT" && -d "$SOURCE_ROOT/g1_lower_rl" ]] && \
  printf 'export REPO_ROOT=%q\nexport SOURCE_ROOT=%q\nexport CHECKPOINT=%q\n' \
  "$REPO_ROOT" "$SOURCE_ROOT" "$CHECKPOINT"
```

将打印出的三行`export`在第二个终端执行，然后分别运行下面的启动命令。若没有打印三行，先核对运行目录、已保存检查点和冻结源码，不启动回放。

历史检查点应改用它对应的`SOURCE_ROOT`与`CHECKPOINT`。回放脚本没有`--source`参数：通过`PYTHONPATH`选择冻结源码，脚本本身使用工作区入口。只有明确要检查工作区新实现时才设置`SOURCE_ROOT="$REPO_ROOT"`；不能把新源码回放误当作历史训练工况。

### 单步：GPU1，端口8086

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl \
PYTHONPATH="$SOURCE_ROOT:$SOURCE_ROOT/scripts" \
  "$HOME/.local/bin/micromamba" run -n mj python \
  "$REPO_ROOT/run_logs/play_footstep_checkpoint.py" "$CHECKPOINT" \
  --profile step-episode --port 8086 --label "Single step"
```

浏览器打开`http://127.0.0.1:8086/`。这是[单步训练场景](step_episode/README.md)的回放：先站1秒，随机执行单脚一步或双脚收齐，再保持5秒并重置。不是固定前移距离的手动调试；预览手臂目标为零，不额外加手臂等式锁定。

### 连续步：GPU0，端口8085

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MUJOCO_GL=egl \
PYTHONPATH="$SOURCE_ROOT:$SOURCE_ROOT/scripts" \
  "$HOME/.local/bin/micromamba" run -n mj python \
  "$REPO_ROOT/run_logs/play_footstep_checkpoint.py" "$CHECKPOINT" \
  --profile precision --port 8085 --label "Continuous walking"
```

浏览器打开`http://127.0.0.1:8085/`。连续场景保留随机脚印、停走调度和自动步频游走；频率控件可切换为手动请求频率，实际频率仍按速率限制平滑变化，不是锁定匀速一直走。精度落脚课程从检查点恢复，GRU在对应回合重置时清零。

### 访问、核验与停止

- VS Code Remote或SSH环境需要把服务端8085、8086端口转发到本地；浏览器地址使用实际转发后的本地端口。若直接访问训练机，使用训练机地址加对应端口。
- `CUDA_VISIBLE_DEVICES`选择物理GPU，脚本内部的`cuda:0`表示该进程唯一可见的GPU。端口若已占用，改`--port`并使用新地址，不关闭不明进程腾端口。
- 初次加载/编译需要时间。核对`PLAYBACK_READY`中的`profile`、`source_root`、`checkpoint`、`checkpoint_sha256`与`actor_exactly_loaded=true`，再等待Viser打印访问地址。两路应为同一检查点哈希，使用同一actor，但回合与GRU状态各自独立。
- 不加`--check-steps`才能打开Viser；该参数用于有限步检查，检查结束即退出，不启动网页。
- 页面中的Pause/Reset只控制对应回放。模型在启动时固定，训练新存档不会热更新回放；需要新模型时关闭并重新指定检查点。
- 两个命令均以前台进程运行。在各自终端按Ctrl+C关闭回放，不停止或重启后台训练。回放会占用额外GPU资源；无噪声预览不能代替训练工况或实机验证。

### 可选：手动前移调试

沿用上面的源码与检查点变量，在独立终端运行；这是连续配方上的预览适配，不是`step-episode`训练场景。

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl \
PYTHONPATH="$SOURCE_ROOT:$SOURCE_ROOT/scripts" \
  "$HOME/.local/bin/micromamba" run -n mj python \
  "$REPO_ROOT/run_logs/play_footstep_checkpoint.py" "$CHECKPOINT" \
  --profile precision --manual-step --port 8087 --label "Manual forward step"
```

- 四槽仍为`[当前左脚,当前右脚,前方左目标,前方右目标]`；默认计划消费两次落脚，首脚可选左或右。加`--single-foot`只执行选中脚的一步，另一脚保持原地，该参数同时启用手动模式。
- `x`为沿初始双脚共同朝向的前移距离，默认0.10m、范围0–0.35m；`f`为完整左右周期频率，范围0.8–1.8Hz；保持站宽和共同朝向。
- 初始双脚并列且f=0，站稳后执行；起步/收步各0.5秒插值，计划结束后确认双脚接触并保持f=0。
- 手动模式关闭随机脚印、随机停走及步频游走，手臂初始角度、速度和PD目标为零；添加仅预览生效的14轴零位等式约束，允许求解器数值误差。执行中锁定参数，不重置GRU、不瞬移机器人。
- 显示实际XY/yaw误差，失败时暂停并保留现场。计划结束不等于准确到达，超限距离拒绝执行；重置站姿后可再次尝试。访问端口8087，停止方式与普通回放相同。