# 两阶段脚步跟踪

任务：`G1-Gloria-FootstepTracking`，平地脚印跟随，控制12腿轴和3腰轴，手臂由独立PD驱动。

先用 `walk-first` 学习行走，再从已有检查点切换到 `precision` 精确跟踪；两阶段保持同一策略接口、1.2Hz请求步频和脚印课程，第二阶段只收紧精度引导并渐入落脚代价。阶段切换由训练命令显式指定。

| 设置 | 第一阶段 `walk-first` | 第二阶段 `precision` |
| --- | --- | --- |
| XY引导：权重 / 核宽 | 1 / 0.40m | 5 / 0.20m |
| yaw引导：权重 / 核宽 | 0.5 / 0.30rad | 5 / 0.10rad |
| 摆动进度权重 | s | s² |
| 落脚事件权重 | 关闭 | 0渐入至-0.1，默认2000轮 |
| 位置 / yaw代价尺度 | 不适用 | 0.05m / 0.20rad，无免罚区 |

`tracking` 是独立的默认配置：请求步频0.8–1.8Hz，落脚事件权重-1；第二阶段使用 `precision`。

## 策略接口

- Actor：84维输入，编码为89维，经GRU(64)和MLP(256,256,128)输出15轴位置增量。
- 输入：29轴角度与速度、骨盆/躯干角速度与重力、相位、频率、四个XY/yaw脚印。
- 脚印顺序：`[L_support,R_support,L_next,R_next]`。support在计划落地时更新，next包含正在迈向的目标。
- Critic：104维，额外观测线速度、接触和上一拍动作，使用MLP(256,128)。
- 控制周期0.02s，PPO每轮64拍，回合60s。契约为 `g1_footstep_gru_v4`。

物理步长0.0025s，每控制拍8个子步，约束容量1024。子步数值保护在关节速度超过120rad/s、根线速度超过20m/s、根角速度超过80rad/s或状态非有限时隔离对应环境，记录现场并按失败终止后reset；正常倒地仍使用概率终止。首10次事件保存状态、控制、外力与随机化模型字段，后续持续记录摘要。

部署布局与导出见[模型契约](../../../FOOTSTEP_TRACKING.md)，生成器见[脚步模块](../../footsteps/README.md)。

## 训练与续训

以下命令在仓库根目录运行，`WALK_CHECKPOINT` 指向第一阶段检查点，`PRECISION_CHECKPOINT` 指向第二阶段检查点。

```bash
# 第一阶段：双GPU、每卡128环境，从零学习行走
OMP_NUM_THREADS=1 micromamba run -n mj python scripts/train_footstep_resume.py \
  --profile walk-first --from-scratch --envs-per-rank 128 \
  --run-root logs/rsl_rl/footstep_walk_first --tag walk_first --save-interval 1000

# 第二阶段：保留模型、优化器及课程，追加2000轮精度训练
OMP_NUM_THREADS=1 micromamba run -n mj python scripts/train_footstep_resume.py \
  "$WALK_CHECKPOINT" --profile precision --max-updates 2000 \
  --save-interval 1000 --tag precision

# 长时续训：双GPU、每卡128环境，固定预算，故障后从完整检查点恢复
OMP_NUM_THREADS=1 micromamba run -n mj python scripts/train_footstep_supervised.py \
  "$PRECISION_CHECKPOINT" --run-root logs/rsl_rl/footstep_precision_continuation \
  --hours 48 --save-interval 1000 --max-restarts 3
```

- **状态恢复**：actor、critic、归一化、Adam、学习率和全局计数精确恢复，从下一轮、新episode开始；`--reset-optimizer` 才会重建Adam。`precision_stage` 保存落脚渐入起点和时长，后续续训不重新渐入；首次切换默认从检查点计数开始，`--landing-ramp-updates` 可覆盖时长。
- **预算与保存**：直接入口支持 `--max-updates N` 或 `--hours H`，二者互斥；都不指定则持续训练。默认每100轮保存，上例改为累计整千轮保存；停止时额外保存最终模型。
- **监督运行**：固定 `precision`，检查点必须包含第二阶段元数据；新建独立run-root，自动引用旧检查点。初始化和故障恢复均计入预算，重试不延长截止时间；300秒无进度触发恢复。以 `supervisor.json` 为准，拒绝覆盖已有监督状态。
- **正常停止**：直接入口在运行目录创建 `STOP` 或向worker发送SIGTERM。监督入口应向 `supervisor.json` 的 `pid` 发送SIGTERM，由它协调保存退出；仅停止worker会被当成需要恢复的中断。
- **复现与回放**：每次改配置后冻结源码到独立目录并从该目录启动；回放使用对应快照与profile。`launch.json` 记录启动参数，`resume_rank_*.json` 验证恢复，`status_rank_*.json` 记录进度；日志与模型不入库。导出使用 `export_footstep_policy`。

## 指令课程

**共同调度：** reset时50%站立、50%直接行走，随机起脚侧；指令每3–8s更新，停止概率30%，站立保持2–5s。
起停渐变各0.5s；停脚后连续2拍双接触确认，等待超过0.5s判故障。频率f表示完整左右周期/s，f=0为站立。

**walk-first课程：** 0–1499轮固定直行、总步距0.27m、站距0.24m；1500轮起扩大方向与站距，2500轮起扩大步距和回合内变向，4000轮达到全方向、步距0.12–0.48m、站距0.12–0.36m。逐脚yaw在4500/5000/5500轮分别放开至±10°/±20°/±30°。奖励精度保持表中设置。

**tracking课程：** 回合内方向增量在0/1500/2500/4000轮取0°/±45°/±90°/±180°；频率变化率在0/800/2000轮取±0.01/±0.15/±0.3Hz/s。

课程按 `common_step_counter // 64` 在reset时选档，作用于后续采样的目标。
walk-first配置见 [walk_first/env_cfg.py](walk_first/env_cfg.py)，阶段表见 [walk_first/curriculum.py](walk_first/curriculum.py)；共享课程见 [curriculum.py](curriculum.py)。

## 第二阶段约束

常规项每拍贡献为 `weight * raw_value * 0.02`；落脚和非超时终止按事件计分。

- **落脚精度**：每目标离地后首次触地扣 `0.1 * (0.5*d/0.05 + 0.5*abs(yaw_error)/0.20)`，渐入结束后使用完整系数；不按支撑时长重复收费，f=0关闭。
- **站定**：脚位相对冻结目标超过3cm才计罚；所有脚的XYZ速度超过0.02m/s计罚，包含离地脚；零速度正奖励乘 `1 / (1 + mean(d²)/0.10²)`，限制碎步漂移。
- **关节姿态**：髋yaw归零-0.1、腰yaw归零-0.4；六对腿关节EMA对称项-0.1，允许正常交替摆动。站姿和对称项为两阶段共享。
- **平衡与能耗**：保留接触节拍、足高、躯干/骨盆直立、头高、膝弯、足底平放、动作平滑和铜损。铜损基础权重-0.25，站立乘2；硬限位附近0.1°且力矩向外的关节乘10。

完整30项定义见[奖励契约](../../../FOOTSTEP_TRACKING.md#10-训练奖励契约)，工厂与公式见 [rewards_cfg.py](rewards_cfg.py)、[rewards.py](rewards.py)。第一阶段不含落脚事件，共29项。

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

```bash
OMP_NUM_THREADS=1 MUJOCO_GL=egl micromamba run -n mj python -m pytest -q \
  tests/test_footstep_precision.py tests/test_footstep_resume.py tests/test_footstep_supervised.py \
  tests/test_footstep_tracking_objectives.py tests/test_footstep_still_velocity.py \
  tests/test_footstep_leg_symmetry.py tests/test_footstep_numerical_safety.py \
  tests/test_arm_reset_ranges.py tests/test_footstep_arm_curriculum.py
```

评测关注XY/yaw误差、存活时间、起停成功率、力矩与接触冲击。测试和仿真回放不能代替实机安全验证。

实验记录（2026-09-24）：MYS适配版对齐了支撑脚相对目标、XY/yaw核与支撑期缓存、摆动窗口及论文列出的姿态/高度/动作变化/防滑系数，平地关闭Z跟踪和膝高项，使用自研生成器与15轴控制；整套奖励未等价，回放效果很差、不如自研实现，代码及结果已删除。