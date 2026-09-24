# 脚步跟踪训练

任务：`G1-Gloria-FootstepTracking`，平地脚印跟随，控制12腿轴和3腰轴，手臂由独立PD驱动。

## 策略接口

- Actor：84维输入，编码为89维，经GRU(64)和MLP(256,256,128)输出15轴位置增量。
- 输入：29轴角度与速度、骨盆/躯干角速度与重力、相位、频率、四个XY/yaw脚印。
- 脚印顺序：`[L_support,R_support,L_next,R_next]`。support在计划落地时更新，next包含正在迈向的目标。
- Critic：104维，额外观测线速度、接触和上一拍动作，使用MLP(256,128)。
- 控制周期0.02s，PPO每轮64拍，回合60s。契约为 `g1_footstep_gru_v4`。

部署布局与导出见[模型契约](../../../FOOTSTEP_TRACKING.md)，生成器见[脚步模块](../../footsteps/README.md)。

## 启动

以下命令在仓库根目录的 `mj` 环境运行。

```bash
# 默认tracking配置，单GPU
micromamba run -n mj python scripts/train.py G1-Gloria-FootstepTracking \
  --gpu-ids '[0]' --env.scene.num-envs 64

# walk-first从零训练，双GPU、每卡128环境；持续至停止请求
OMP_NUM_THREADS=1 micromamba run -n mj python scripts/train_footstep_resume.py \
  --profile walk-first --from-scratch --envs-per-rank 128 \
  --run-root logs/rsl_rl/footstep_walk_first --tag walk_first

# 从19490续训，使用当前walk-first配置
OMP_NUM_THREADS=1 micromamba run -n mj python scripts/train_footstep_resume.py \
  logs/rsl_rl/split_mys_ours_15h_20260923/ours_gpu1/model_19490.pt --profile walk-first
```

续训入口默认双GPU、每卡128环境，每100轮保存。恢复actor、critic、Adam、学习率及全局计数，从下一轮、新episode开始。
`--max-updates N` 限定追加轮数；`--reset-optimizer` 保留模型并重建Adam。
运行目录中的 `STOP` 文件或worker的SIGTERM请求会在完整更新后保存退出。
回放已有模型应使用对应源码快照及profile；专用ONNX导出使用 `export_footstep_policy`。

## 配置与课程

| 设置 | tracking（默认） | walk-first（19490训练使用） |
| --- | --- | --- |
| XY引导：权重 / 核宽 | 5 / 0.20m | 1 / 0.40m |
| yaw引导：权重 / 核宽 | 5 / 0.10rad | 0.5 / 0.30rad |
| 摆动进度权重 | s² | s |
| 落脚事件代价 | -1 | 关闭 |
| 请求步频 | 每回合0.8–1.8Hz，变化率课程 | 1.2Hz |
| 奖励项数 | 28 | 27 |

**共同调度：** reset时50%站立、50%直接行走，随机起脚侧；指令每3–8s更新，停止概率30%，站立保持2–5s。
起停渐变各0.5s；停脚后连续2拍双接触确认，等待超过0.5s判故障。频率f表示完整左右周期/s，f=0为站立。

**walk-first课程：** 0–1499轮固定直行、总步距0.27m、站距0.24m；1500轮起扩大方向与站距，2500轮起扩大步距和回合内变向，4000轮达到全方向、步距0.12–0.48m、站距0.12–0.36m。逐脚yaw在4500/5000/5500轮分别放开至±10°/±20°/±30°。奖励精度保持表中设置。

**tracking课程：** 回合内方向增量在0/1500/2500/4000轮取0°/±45°/±90°/±180°；频率变化率在0/800/2000轮取±0.01/±0.15/±0.3Hz/s。

课程按 `common_step_counter // 64` 在reset时选档，作用于后续采样的目标。
walk-first配置见 [walk_first/env_cfg.py](walk_first/env_cfg.py)，阶段表见 [walk_first/curriculum.py](walk_first/curriculum.py)；共享课程见 [curriculum.py](curriculum.py)。

## 奖励

常规项每拍贡献为 `weight * raw_value * 0.02`；落脚和非超时终止按事件计分。
下表列出默认tracking配置，walk-first仅覆盖上表中的跟踪项。

| 项 | 权重 | 定义 |
| --- | ---: | --- |
| `footstep_swing_position` | +5 | 摆动脚XY指数精度，核宽0.20m，乘s²；要求计划支撑脚实际接触 |
| `footstep_swing_yaw` | +5 | 最短yaw角差指数精度，核宽0.10rad，门控同上 |
| `footstep_landing` | -1/事件 | 每目标摆动离地后首次触地扣 `0.5*d/0.08 + 0.5*abs(yaw_error)/0.20`，f=0关闭 |
| `contact_schedule` | +1 | 实测双脚接触模式与计划完全匹配，f>0启用 |
| `foot_air_time` | +3.2 | 正确侧单支撑时长按摆动时长归一化，受摆动进度限制，原始值最高0.4 |
| `swing_clearance` | -0.5 | 双脚足高绝对误差除以0.11m；摆动目标为 `0.11*sin(pi*s)^2`，支撑目标为地面 |
| `swing_contact` | -0.5 | 计划摆动脚触地乘sin²相位权重 |
| `foot_slip` | -2 | 接触脚水平速度平方和，f>0启用 |
| `feet_slip_still` | -4 | 接触脚水平速度模长之和，f=0启用 |
| `stand_still_feet` | -0.5 | 未接触脚数，f=0启用 |
| `feet_hold_position` | -0.2 | f=0时脚位偏离冻结目标超过3cm的部分，按10cm归一化后平方、双脚平均 |
| `feet_flatness` | -0.2 | 计划支撑或实际触地脚的法向倾角按15°归一化后平方、按脚平均 |
| `soft_landing` | -0.002 | 首次触地力模长之和，f>0启用 |
| `lower_body_copper_proxy` | -0.25 | 15轴实际执行器力矩平方和，共用100Nm尺度；f=0乘2，向外顶限位的关节乘10 |
| `waist_yaw_zero` | -0.4 | 腰yaw绝对角平方 |
| `waist_roll_pitch_edges` | -2 | 腰roll/pitch硬限位两侧各15%行程的边缘平方代价 |
| `torso_upright` | -0.5 | 躯干相对竖直倾角平方 |
| `body_ang_vel` | -0.05 | 躯干世界系X/Y角速度平方和 |
| `head_height` | +0.4 | `clip(h/1.254,0,1)`；至少一脚接触且未终止时启用 |
| `head_height_low` | -1 | `relu(1.15-h)/0.2`，h为头部几何中心离地高度 |
| `stance_knee_bend` | -0.2 | 支撑膝超限角按45°归一化后平方、按腿平均；站立阈值30°，行走45° |
| `pelvis_upright_filtered` | -1 | 低通后的骨盆重力XY平方和；截止频率为f/4，站立为0.2Hz |
| `stand_still_linear_velocity` | +2 | f=0时 `exp(-(vx²+vy²+1.5*vz²)/0.2)` |
| `stand_still_angular_velocity` | +0.5 | f=0时 `exp(-(wz²+0.05*(wx²+wy²))/0.49)` |
| `action_rate` | -0.02 | 相邻动作差平方和 |
| `controlled_joint_acc` | -2.5e-7 | 15轴实测加速度平方和 |
| `self_collisions` | -2 | 腿间接触力超过10N的物理子步数 |
| `fall` | -10/次 | 非超时终止事件 |

铜损的限位容差为0.1°，只有接近或越过硬限位且力矩继续向外时放大；站立倍率与限位倍率相乘。
腰yaw项约束相对骨盆的扭腰角。奖励配置与公式见 [rewards_cfg.py](rewards_cfg.py)、[rewards.py](rewards.py)。

## 初始化与随机化

- 手臂14轴实际角度、速度和初始PD目标设为0。最终目标独立采样 `U(lower-pi/2, upper+pi/2) * target_scale`，每个环境在reset后5秒内线性渐变到终点。
- `target_scale` 在0–800轮为0.2，随后线性升至1500轮0.5、2000轮1.0；只影响后续reset。渐变期间暂停目标漂移。越界目标及渐变路径可能产生限位力矩或自碰撞。
- 基础随机化复用 [lower_body事件](../lower_body/cfg/events.py)，包括连杆质量、夹爪负载、脚底摩擦、编码器偏置和质心偏移。主动手臂力矩、目标漂移与身体平移推力处于初始档，身体冲量仍含±2Nm力矩。
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
  tests/test_footstep_*.py tests/test_tensor_footsteps.py tests/test_single_duration.py
```

评测关注XY/yaw误差、存活时间、起停成功率、力矩与接触冲击。测试和仿真回放不能代替实机安全验证。

实验记录（2026-09-24）：MYS适配版对齐了支撑脚相对目标、XY/yaw核与支撑期缓存、摆动窗口及论文列出的姿态/高度/动作变化/防滑系数，平地关闭Z跟踪和膝高项，使用自研生成器与15轴控制；整套奖励未等价，回放效果很差、不如自研实现，代码及结果已删除。