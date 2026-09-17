# 脚印跟踪 GRU 模型契约

版本：`g1_footstep_gru_v1`。

2026-09-15 增补 f=0 双支撑站立，输入输出 shape 不变。新导出契约声明
`phase.standing_frequency=0.0`；缺少该字段的旧 ONNX 只允许正频率输入。
这只是接口声明，旧行走权重仅重新导出不会自动获得站立能力。

已实现模型、输入编码、PPO 模型配置、TorchScript/ONNX 导出、CPU 推理封装，以及训练奖励和防摔终止配置。
另已实现独立的 NumPy 脚印采样、四步管理及频率停走调度模块。
已接入可运行的 `G1-Gloria-FootstepTracking` 环境，完成真实 CPU 仿真和短 PPO 更新验证
**没有训练好的策略。** 大规模吞吐、课程和训练分布调优、硬件验证仍待完成，不能用于机器人
训练入口、参数和时序见 [训练说明](g1_lower_rl/tasks/footstep_tracking/README.md)

实现位置：

- [独立脚步模块](g1_lower_rl/footsteps/README.md)：Mind Your Steps 风格采样、四步队列、慢变频率和概率停走。
- [模型和导出](g1_lower_rl/rl/footstep_model.py)：`FootstepActor`、`FootstepModelCfg`、`export_footstep_policy`。
- [部署输入和推理](g1_lower_rl/footstep_deploy.py)：`pack_footstep_observation`、`FootstepPolicy`。
- [相位配置](g1_lower_rl/footstep_phase.py)：`FootstepPhaseCfg`，奖励与模型导出共用的理论双支撑窗口。
- [测试](tests/test_footstep_model.py)：输入布局、编码、GRU、PPO 更新、导出和连续推理。
- [奖励配置](g1_lower_rl/tasks/footstep_tracking/rewards_cfg.py)：`make_rewards`、`make_terminations`。
- [奖励实现](g1_lower_rl/tasks/footstep_tracking/rewards.py)及[测试](tests/test_footstep_tracking_objectives.py)：摆动指数奖励、落地锁定线性代价、实时支撑与接触节拍。

## 1. 任务边界

- 观测全身 29 轴，不包括两个 Gloria-M 夹爪关节。
- 只控制 15 轴：左腿 6、右腿 6、腰 yaw/roll/pitch。
- 骨盆和 torso 各输入角速度与单位重力方向。
- 左右脚各提供两个未来落点，合计四步；每个目标是平地 XY 和脚掌 yaw。
- 输入全局相位和当前频率；不输入身体高度、线速度指令、骨盆相对支撑系位姿、接触力或外力真值。
- f>0 为行走/踏步，f=0 为双脚站立；原地踏步通过正频率和重复左右各自固定的落点表达。
- 上肢和夹爪由独立控制器控制，本策略不得覆盖其输出。

## 2. ONNX 输入

所有输入输出为 `float32`；部署固定 batch=1。下表下标从零开始，切片为左闭右开。

| 输入 | 默认 shape | 含义 |
| --- | --- | --- |
| `obs` | `[1,84]` | 未归一化、未编码的原始观测 |
| `h_in` | `[1,1,32]` | GRU `[层数,batch,hidden_size]` 隐状态 |

| obs 切片 | 维数 | 内容 | 单位/约定 |
| --- | --- | --- | --- |
| `0:29` | 29 | 全身关节位置 q | 标定后的编码器绝对角度，rad，不减默认角 |
| `29:58` | 29 | 全身关节速度 dq | rad/s |
| `58:61` | 3 | 骨盆角速度 | 骨盆 link 坐标系，rad/s |
| `61:64` | 3 | 骨盆系重力方向 | 单位向量，正立时 `[0,0,-1]` |
| `64:67` | 3 | torso 角速度 | torso link 坐标系，rad/s |
| `67:70` | 3 | torso 系重力方向 | 单位向量，正立时 `[0,0,-1]` |
| `70:71` | 1 | 相位 x | rad，`0 <= x < 2*pi` |
| `71:72` | 1 | 当前频率 f | 完整左右周期/s，非负；0 为双脚站立，正数时总步频为 `2*f` |
| `72:75` | 3 | L1 | 左脚下一次落地的 `[dx,dy,dtheta]` |
| `75:78` | 3 | L2 | 左脚再下一次落地的 `[dx,dy,dtheta]` |
| `78:81` | 3 | R1 | 右脚下一次落地的 `[dx,dy,dtheta]` |
| `81:84` | 3 | R2 | 右脚再下一次落地的 `[dx,dy,dtheta]` |

四个脚印槽位必须有效。本版没有 padding/valid 掩码；缺少远期指令时，必须由调度器显式补齐
可执行的重复落点或进行安全处理，不能用全零表示数据缺失。输入形状为 `[4,3]`，排列为
`[L1,L2,R1,R2]`，不是时间交错的 `[L1,R1,L2,R2]`。

f=0 时四个槽位为 `[left_hold,left_hold,right_hold,right_hold]`，两个不同的足底保持目标
仍在同一参考系表达，不输入远处行走目标，也不把两脚同时设为原点。

### 关节顺序

名称以 ONNX 契约中的 `joint_names` 为准，按名称映射硬件反馈，不依赖 MJCF 中的物理排列。

| 下标 | 名称顺序 |
| --- | --- |
| 0..5 | left_hip_pitch_joint, left_hip_roll_joint, left_hip_yaw_joint, left_knee_joint, left_ankle_pitch_joint, left_ankle_roll_joint |
| 6..11 | right_hip_pitch_joint, right_hip_roll_joint, right_hip_yaw_joint, right_knee_joint, right_ankle_pitch_joint, right_ankle_roll_joint |
| 12..14 | waist_yaw_joint, waist_roll_joint, waist_pitch_joint |
| 15..21 | left_shoulder_pitch_joint, left_shoulder_roll_joint, left_shoulder_yaw_joint, left_elbow_joint, left_wrist_roll_joint, left_wrist_pitch_joint, left_wrist_yaw_joint |
| 22..28 | right_shoulder_pitch_joint, right_shoulder_roll_joint, right_shoulder_yaw_joint, right_elbow_joint, right_wrist_roll_joint, right_wrist_pitch_joint, right_wrist_yaw_joint |

## 3. 模型内编码与网络

确定性前处理在模型内部执行，训练和导出使用同一实现：

1. `q_rel = q - default_joint_pos`。默认角存于 checkpoint 的 `encoder.default_joint_pos` buffer。
2. 相位 `x` 转为 `[sin(x),cos(x)]`。
3. 每个脚印的 `[dx,dy,dtheta]` 转为 `[dx,dy,sin(dtheta),cos(dtheta)]`。
4. 对编码后的 89 维特征做经验归一化。运行均值和方差保存于 checkpoint，并随模型导出。

编码后布局：`q_rel[0:29]`、`dq[29:58]`、两组 IMU `[58:70]`、
`sin(x),cos(x)[70:72]`、`f[72:73]`、四个脚印 `[73:89]`，每项 4 维。

```text
obs[batch,84]
  -> FootstepObservationEncoder[batch,89]
  -> EmpiricalNormalization
  -> GRU(input=89, hidden=32, layers=1)
  -> MLP(32 -> 256 -> 128 -> 15, ELU)
  -> actions[batch,15]
```

`FootstepModelCfg` 默认使用可训练的标量 Gaussian std=1.0，供 PPO 探索。
`FootstepActor.forward(..., stochastic_output=True)` 采样动作；默认 forward 和部署导出均为确定性输出。
Actor 支持 rsl_rl 的 padded 序列、mask、初始隐状态、按环境 reset 和截断反传。
模型配置可改变隐藏层大小；本版部署布局仍为 84/89/15，实际隐状态 shape 必须读取导出契约。

**部署端不能再次减默认角、做 sin/cos、做经验归一化，或每拍清零 GRU。**

站立时保留冻结相位的 sin/cos，不置为 `[0,0]`；f=0 本身就是模式标识。
只冻结外部时钟/队列/参考系，传感器读取和 GRU 推理仍每拍运行。正常停走切换不 reset 隐状态。

## 4. 输出与执行器

| 输出 | 默认 shape | 含义 |
| --- | --- | --- |
| `actions` | `[1,15]` | 确定性的归一化关节位置偏移，不是力矩，不是绝对角度 |
| `h_out` | `[1,1,32]` | 下一拍的 `h_in`；没有 LSTM cell state |

动作顺序为上表前 15 轴。各轴的位置目标和名义 PD 力矩：

```text
q_des = action_default_joint_pos + action_scale * actions
dq_des = 0
tau_ff = 0
tau = clip(Kp * (q_des - q) - Kd * dq, -effort_limit, effort_limit)
```

当前 scale 与 lower_body 一致：`0.25 * effort_limit / Kp`。
scale、Kp、Kd、effort_limit 在模型构造时从资产快照取得，并作为 buffers 保存在 checkpoint；
导出以这些 buffers 为准。默认动作偏移为已保存的 29 轴默认角的前 15 项。

输出层没有 tanh，本版没有动作裁剪、位置目标裁剪或滤波。这里的“归一化”不代表严格落在 `[-1,1]`。
常规裁剪/滤波要与未来训练环境一致；硬件保护独立执行，触发保护不等同于正常策略行为。
`FootstepPolicy.step` 返回 `(actions[15], q_des[15])`，不执行 PD、不发送电机命令。

## 5. 脚印坐标与相位调度边界

四个脚印相对于**同一个**冻结的水平支撑参考系 A：原点取建系时估计的支撑足底位置，
X 为该脚的水平前向，Y 向左，Z 向上。`dtheta` 是足底 yaw 相对 A 的角度，绕 +Z 逆时针为正。
足底的参考点应与训练中的 `left_foot/right_foot` site 一致，而不是另换脚踝或鞋尖。

所有四项直接相对 A，不是未来目标相对前一个目标的链式增量。一次支撑阶段内目标指令固定；
重建 A 时统一变换尚未消费的目标，不能移动其地面位置来抹掉误差。理论支撑事件不等于实际可靠接地；
定位/运动学模块必须处理迟落地和打滑，模型本身不能提供全局定位。

外部调度器按 `x_next = x + 2*pi*f*dt` 推进时钟，以未取模相位或圈数检测跨界，
仅向网络传取模的 x。默认计划事件（自定义时以导出配置为准）：

- 跨过 `pi/2 + 2*k*pi`：左脚理论落地，L1 成为支撑目标，左列表后移。
- 跨过 `3*pi/2 + 2*k*pi`：右脚理论落地，R1 成为支撑目标，右列表后移。
- 双脚最近的支撑目标另存于调度器，不占未来四步槽位。
- 相位、列表索引、参考系和观测必须在同一拍一致更新。普通换步不 reset GRU。
- 网络中的 sin/cos 是全局相位编码，不分别绑定左右脚，所以不隐含 1:3 步间隔。
- 模型不判断是否踩准，不输出换步信号，也不自行推进脚印。

独立管理器支持行走正频率随机游走、触边反弹及停止收步；实际数值配置见模块 README。
训练适配器为每个环境组合一个 CPU 单实例管理器与随机源，尚未 GPU 向量化。模型机器契约的频率能力范围仍标为 null，
不意味着任意正频率都可执行，也不将管理器配置视为已验证的硬件范围。
f=0 单独表示双支撑站立，不作为行走随机游走的下界。正频率重复原地脚印仍持续踏步，
硬件安全停机不等于零频率站立命令。停走调度方案见第 11 节。

## 6. 双 IMU 与部署数据来源

| 数据 | 获取方式 | 注意事项 |
| --- | --- | --- |
| 29 轴 q/dq | 电机反馈 | 关节名称、正方向、零位、单位与资产对齐；不含夹爪 |
| 两处角速度 | 各自陀螺仪 | 旋转到对应 link 坐标轴；不能将 torso 数据冒充骨盆数据 |
| 两处重力方向 | 姿态估计 | `R_world_from_link.T @ [0,0,-1]`；不是直接归一化加速度计 |
| x/f | 调度器 | 相位、脚印来自同一个控制时间基准 |
| 四个脚印 | 规划器 + 坐标变换 | 地图绝对脚印需要定位或视觉反馈；相对里程计可能累积漂移 |

原始加速度、世界线速度、身体高度、骨盆平面位姿、接触力和外力均不作为 actor 输入。
双 IMU 可以有不同测量噪声，但必须标定安装轴，并进行时间同步。
训练从骨盆和躯干各自的 link 姿态与角速度构造两组观测，没有修改 MJCF 传感项
真实硬件的双 IMU 反馈、安装标定和时间同步仍需单独验证

训练观测噪声已对齐 lower_body GRU：关节角/速度标准差0.006rad/0.87rad/s，
两组IMU角速度每拍标准差0.115rad/s、每episode偏置标准差0.03rad/s；
重力方向每拍标准差0.029、每episode偏置标准差0.01，两个IMU及各轴独立
偏置在episode内固定、reset重新采样而非累加，随机延迟关闭；重力加噪后不再归一化
critic与play关闭此观测噪声，但既有startup编码器偏置U[-0.015,0.015]rad仍保留
IMU在环境中拆成四个3维项以兼容噪声reset，最终84维排列不变，脚印/相位/频率不加噪声
详见[任务噪声说明与论文对照](g1_lower_rl/tasks/footstep_tracking/README.md)

## 7. 构造、保存和导出

以下是接口示例，不是训练命令，构造出的模型尚无行走能力：

```python
from dataclasses import asdict
import torch
from tensordict import TensorDict
from g1_lower_rl.rl.footstep_model import FootstepActor, FootstepModelCfg, export_footstep_policy

cfg = FootstepModelCfg()
options = {key: value for key, value in asdict(cfg).items()
           if value is not None and key != "class_name"}
example = TensorDict({"actor": torch.zeros(1, 84)}, batch_size=[1])
actor = FootstepActor(example, {"actor": ["actor"]}, "actor", 15, **options)
torch.save({"actor_state_dict": actor.state_dict(), "actor_config": asdict(cfg)}, "footstep_actor.pt")
export_footstep_policy(actor, "export/footstep/policy.onnx")
```

训练接入时传一个 84 维原始观测组，不要提前编码成 89 维；类路径可由 rsl_rl 的 resolver 加载。
训练已绑定奖励与104维 MLP critic 观测；不继承旧速度任务的课程，脚步课程仍待训练调优
加载 checkpoint 时同时恢复模型配置和完整 state_dict，包括默认角、归一化和动作参数 buffers。

`export_footstep_policy` 使用仓库现有 rsl_rl 导出包装，导出 opset 17、固定 batch=1：

- ONNX 内 `footstep_contract` 元数据是 JSON 字符串。
- 同目录生成同名 `.contract.json`，内容与内嵌契约一致。
- 包含输入切片、关节名单、默认角、动作参数、坐标与相位约定、隐状态 shapes。
- 不修改在线模型的训练模式、设备、权重、Gaussian 缓存或隐状态。
- `as_onnx()` 本身只提供标准图包装；需要带契约的部署文件时必须调用 `export_footstep_policy`。
- `torch.jit.script(actor.as_jit().cpu().eval())` 支持 TorchScript，其隐状态存在模块内部，用 `.reset()` 重置。

不要把新模型交给旧的 lower_body 观测拼装器或默认导出元数据函数；它们仍按速度/高度任务解析输入。
脚步任务已独立注册并使用普通 MjlabOnPolicyRunner，部署时仍需显式使用新契约导出函数

## 8. 实机每拍运行

CPU 端只需 NumPy 和 ONNX Runtime，不需要 PyTorch、MuJoCo 或 mjlab。
加载 `FootstepPolicy` 时自动从 ONNX 读取并校验契约，默认使用单线程 CPU inference。
运行节拍先约定为 50 Hz (`dt=0.02 s`)，PD 循环由硬件保持自己的更高频率。

```python
from g1_lower_rl.footstep_deploy import FootstepPolicy, pack_footstep_observation

policy = FootstepPolicy("export/footstep/policy.onnx")
policy.reset()

# 在外部控制循环中，以下变量由已同步的硬件反馈和规划器提供。
obs = pack_footstep_observation(
    joint_pos=q, joint_vel=dq,
    pelvis_ang_vel=pelvis_gyro, pelvis_projected_gravity=pelvis_gravity,
    torso_ang_vel=torso_gyro, torso_projected_gravity=torso_gravity,
    phase=phase, frequency=frequency, footsteps=footprints,
)
actions, q_des = policy.step(obs)
# 按 policy.action_joint_names 和契约中的 PD 参数下发 q_des。
```

每台机器人使用独立实例。封装自动维护 h，不要每步重建实例。它仅验证形状、数值、非负频率、
相位范围和导出接口，不能验证传感器安装/时戳、地面安全、目标可达性或足底真实接触。
推理产生非有限值时抛出错误且不提交新隐状态；控制进程必须捕获异常并安全处理，不是直接崩溃退出。

真正闭环运行前，还必须把独立脚步管理器接入控制时序，提供坐标估计、传感器过期检测、推理超时、
电机限位/力矩保护和安全接管。控制长时间中断时不能简单跳过多个脚印继续推理。

## 9. 验证与后续训练设计

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 micromamba run -n mj python -m unittest discover -s tests -p test_footstep_model.py -v
```

测试包括 84/89 维排列、关节默认值、四步顺序、角度周期性、批次和时间维度、GRU reset、
checkpoint 恢复、真实 CPU PPO 更新（含 episode 截断）、TorchScript 连续推理、ONNX Runtime
多拍动作/隐状态对齐、契约一致性和部署端动作映射。测试使用合成张量，不验证行走、铜损或抗扰性能。

下一轮训练仍遵循已讨论的目标：落点/脚掌朝向/节拍跟踪、不摔倒、受控 15 轴铜损代理最小，
腰 yaw 靠近零、腰 roll/pitch 临近限位惩罚；加入骨盆高度单调封顶奖励，但不增加身体高度指令。
上肢摆动、外力扰动及课程参考 lower_body GRU，课程升级改用脚印跟踪与存活质量。
没有电机电阻/力矩常数/传动标定时，力矩平方仅称为铜损代理。
脚印范围和频率参数已有可调实现初值，仍需验证可行性；轨迹分布、课程阈值和训练预算仍待确定。
奖励初始权重见下一节，训练中需根据分项指标调节，不代表已经验证的最优权重。

## 10. 训练奖励契约

入口为 `make_rewards(command_name="footsteps", sensor_name="feet_ground_contact")`。
配合 `RewardManager(scale_by_dt=True)` 使用：常规项为每秒奖励率，环境每拍乘 `step_dt`。
只有摔倒项是事件代价，函数内部除以 `step_dt`，所以每次真正终止固定扣 10 分；
时间上限结束不算摔倒。不要再额外对总奖励做非负裁剪。

### 10.1 当前奖励表

| 名称 | 权重 | 意图/原始值 |
| --- | --- | --- |
| `footstep_landing` | -4.0 | 首次触地锁定线性XY/yaw代价；漏落地/错时触地加代价；站立改为实时双脚代价 |
| `footstep_support` | -1.0 | 计划支撑脚相对原目标的实时线性XY/yaw代价，不因失去接触清零 |
| `footstep_swing_position` | +5.0 | 计划摆动脚 `exp(-(XY_distance/0.10)^2)`，无摆动进度权重、无实际接触门控 |
| `footstep_swing_yaw` | +5.0 | 计划摆动脚 `exp(-(wrapped_yaw_error/0.10)^2)`，阶段门控同上 |
| `contact_schedule` | +2.0 | 左右脚同时满足接触模式才得 1 分；f>0 按相位，f=0 始终要求双接触 |
| `swing_clearance` | -0.5 | 摆动脚低于相位相关离地间隙的归一化平方代价，高于目标不额外惩罚 |
| `foot_slip` | -2.0 | 实际触地脚的 XY 速度模长 + 0.05 m × 竖直轴角速度绝对值之和 |
| `soft_landing` | -0.002 | 复用速度跟踪任务的首次触地接触力模长惩罚，无速度/频率门控，持续支撑不收费 |
| `lower_body_copper_proxy` | -2.0 | 15 轴实际力矩的统一尺度平方和，见下文 |
| `waist_yaw_zero` | -0.4 | `waist_yaw_joint` 绝对角度平方，目标是 0 rad，不是默认姿态偏差 |
| `waist_roll_pitch_edges` | -2.0 | 两个腰轴靠近硬限位的边缘平方惩罚，内部区域为零 |
| `torso_upright` | -0.5 | torso 重力向量 XY 分量平方和，不约束世界 yaw，不控制身体高度 |
| `pelvis_height` | +2.0 | `clip((pelvis_z-ground_z)/0.78,0,1)`，至少一脚接触且未失败时启用 |
| `pelvis_upright_filtered` | -1.0 | 随f调整的一阶低通骨盆重力向量XY分量平方和，不罚步频摆动的原始幅度 |
| `action_rate` | -0.02 | 15 维动作相邻拍差的平方和 |
| `controlled_joint_acc` | -2.5e-7 | 仅 15 轴关节加速度平方和，不罚外部控制的手臂/夹爪 |
| `self_collisions` | -2.0 | 复用 lower_body 的腿间自碰撞传感器，统计控制拍内超过 10 N 的接触子步数 |
| `fall` | -10.0/次 | 真正终止事件代价，不罚 time-out |

这是带权 RL 目标，而非保证不摔的约束优化或安全证明。以可达脚印下的落地精度和存活为主要目标，
再降低能耗。能耗项过重可能导致不愿迈步，过轻则可能动作剧烈，需要结合实际训练调权。
时钟由外部指定，不能由策略选择停住或减慢时钟以逃避跟踪。

### 10.2 精度、节拍与防刷分

2026-09-17起采用摆动指数奖励、落地及支撑线性惩罚。
摆动核参考 [Mind Your Steps §IV-C及附录D、F](https://arxiv.org/html/2606.08253v1)，
位置尺度0.10m、角度尺度0.10rad，两个分量各权重+5，不乘摆动进度平方。
按计划摆动阶段启用，不以实际接触门控；双支撑及f=0时关闭。
默认工厂参数 `position_std=0.08 m`、`yaw_std=0.20 rad` 保留名称，但改为落地/支撑线性代价尺度。
这些尺度不是容许误差上界或已实现的实机精度。

```text
r_swing_xy = 5 * sum_swing(exp(-(XY_distance/0.10)^2))
r_swing_yaw = 5 * sum_swing(exp(-(wrapped_yaw_error/0.10)^2))
cost = 0.5 * XY_distance/position_std + 0.5 * abs(wrapped_yaw_error)/yaw_std
r_landing = -4 * mean_planned_stance(latched_or_pending_cost)
r_support = -1 * mean_planned_stance(cost)
```

线性代价不截断，XY整体距离与yaw各占50%，不把X和Y拆开。
旧 `footstep_approach`、`footstep_distance`、旧指数落地/支撑原语及 `distance_penalty` 参数已删除。
当前只有一套18项配置（4正、14负）；历史A/B脚本须使用对应源码快照，不能直接调用当前奖励工厂。
持续续训入口为 `scripts/train_footstep_resume.py`，支持双卡精确恢复及更新结束后同步保存退出。
比较run需使用原始XY/yaw误差和存活率，不能通过不同定义下的总奖励判断效果。
下肢平滑保持动作差分-0.02及实际关节加速度-2.5e-7，不添加站立增量或力矩差分项

两只脚的 yaw 均为 `left_foot/right_foot` site 的世界 yaw，使用 wxyz 四元数转换，并将角差 wrap。
默认单脚支撑占一个完整周期的 0.6，摆动占 0.4，总双支撑占 0.2；
左右默认理论落地边界为 pi/2、3pi/2。通过 `FootstepPhaseCfg` 显式修改窗口，频率变化不重置相位。
旧 `stance_fraction` 参数仍可使用，但不能与 `phase_cfg` 同时传入；其派生配置必须同步给 actor。
`phase_windows` 是后续调度器和奖励应共同使用的窗口定义，不得另外写一套起脚边界。

正常首次落地要求该脚在本目标摆动期真正离地，随后在理论落地前后各 0.1 个完整周期内首次接触。
默认 `landing_window=0.25` 表示摆动时长的 25%，即 `0.25*(1-0.6)=0.1` 个周期。
正常首次触地锁定当时的线性代价，过早/过晚则锁定 `cost_touch+landing_miss_cost`，默认附加代价1。
尚未完成离地再触地（包括reset后）的计划支撑脚使用 `cost_current+landing_miss_cost`，不再给零分。
附加代价属于落地项，随-4权重、计划支撑脚平均及dt一起累计，不是独立事件罚款。
同一目标再跳一次或滑到正确落点不能修改首次触地代价；目标ID变化及局部reset清理对应历史。

以上触地锁定针对 f>0。f=0 时改为实时双脚线性代价，清理历史，不要求reset站立先抬脚，也不收未落地附加代价。
支撑mask覆盖为双脚支撑，两个摆动奖励及clearance关闭；重新起步后须重新离地再触地才能锁定落地代价。
精确零频率才切换模式，小的正频率仍按行走评分。

落地代价在计划支撑期持续累计，不是每个touchdown单独罚一次。
支撑代价实时计算，两个惩罚都不因失去接触清零；节拍奖励仍要求两只脚实际接触模式匹配计划。
摆动间隙为 `0.08 m * sin(pi*swing_progress)` 的下界引导，不规定身体高度或完整关节轨迹。
实际接触来自传感器 `found > 0`，当前没有迟滞滤波；将来改变接触判据时需同步测试和时序。

落地力度项 `soft_landing` 直接复用 `mjlab.tasks.velocity.mdp.soft_landing`，权重与 lower_body 一致，
但不传 `command_name`：脚印任务没有 twist 指令，原地踏步和站立恢复中的再次触地也应避免猛踩。

```text
first_contact = feet_sensor.compute_first_contact(dt=step_dt)
landing_force = sum_feet norm(contact_force_xyz) * first_contact
reward_soft_landing = -0.002 * landing_force * step_dt
```

它按实际首次接触事件计费，不依赖理论相位或落地精度是否合格，也不使用落地精度项的目标编号锁分。
两脚同拍落地时求和；持续支撑即使受力较大也不收费。使用完整 XYZ 力模长，单位 N，
不是力矩平方、冲量、物理子步最大峰值或按体重归一化的值。保留原任务的 dt 缩放，不除以 step_dt。
复用函数记录 `Metrics/landing_force_mean`，为本拍所有首次触地脚的平均力模长；本拍无落地时为零，
因此跨时间直接平均该指标不等于跨所有落地事件的平均冲击力。
接触力仅作为训练奖励数据，不增加 actor 的 84 维输入，也不引入夹爪观测。

### 10.3 铜损代理与腰部

```text
copper_proxy = sum_j copper_weight[j] * (actual_actuator_torque[j] / 100 Nm)^2
```

`actual_actuator_torque` 读取奖励采样时刻的 `asset.data.actuator_force`，
按执行器名称精确选择下肢 15 轴，不用 joint_ids 索引 actuator_force；不包含关节约束力、外力、手臂或夹爪。
目前是控制拍采样值，不是物理子步 RMS，也没有电机热模型。
所有轴共用 100 Nm 作为数值尺度，默认系数全为 1；不是按各轴峰值力矩分别归一化，
因此在未知电机参数时相同 Nm 的力矩具有相同代价。

如有电阻、力矩常数和传动标定，可传入按 15 个执行器完整命名的正 `copper_weights`。
实际铜损对应 `sum R_j * I_j^2`，电流与关节力矩之间还需传动/效率模型；
当前默认值只表示力矩平方代理，不能标注为真实瓦特数。

骨盆高度奖励在0到0.78m内单调线性增加、之后封顶，双脚腾空或非超时失败拍不发此奖励。
高度参考 `ground_height` 而不是运动中的脚底；它不改变84维actor输入，也不加入高度命令或固定膝角。
骨盆倾斜使用真实骨盆局部单位重力向量的一阶低通，`fc=f/4`（f>0），f=0时`fc=0.2Hz`。
每个控制拍用当前执行频率计算 `alpha=1-exp(-2*pi*fc*step_dt)`，
`g_filtered=(1-alpha)*g_filtered+alpha*g`，惩罚 `g_filtered[:2]` 的平方和。
先滤波再平方；该滤波衰减而非完全消除步频分量。各环境独立reset，首次读当前重力初始化，
同拍重复读取不再积分；原有torso即时倾斜项及骨盆跌倒终止不变。

腰 yaw 目标为腰相对骨盆的扭转角 0，不是把整个机器人世界朝向锁到 0；
允许转弯时整个机器人随脚印转向，回零只是软约束。
腰 roll/pitch 不添加全程回零项，只罚各自硬限位两侧 15% 行程：

```text
margin = 0.15 * (upper_limit - lower_limit)
edge_cost = relu((lower_limit + margin - q)/margin)^2
          + relu((q - upper_limit + margin)/margin)^2
```

中央 70% 行程为零，单轴到硬边界代价为 1，越界继续增加；两轴求和。
不使用机器人 soft_joint_pos_limits，避免把膝的软限位当作姿态目标间接强迫屈膝。
仅腰 roll/pitch 使用此项，不对整条腿施加名义姿态回归或直膝奖励。

`make_terminations()` 增加 `footstep_distance`：任一计划支撑脚到本拍执行目标的 XY 距离严格超过1m，立即非超时终止。
使用时钟推进后的 `reward_state.targets_w` 快照，不使用下一步目标；不要求实际接触，不设置持续时间宽限。
摆动脚不参与此判断，f=0检查双脚。它同样触发固定10分的 `fall` 事件代价。
其余条件复用 lower_body：骨盆倾角超过 70 度或相对最低足底高度低于 0.35 m 结束。
0.35 m 是塌坐失败阈值，不是身体高度指令/跟踪奖励。

脚印采样默认横向间距范围12–36cm，以24cm为中心叠加有向横移分量并裁剪，交替左右脚的整体均值约24cm。
收步间距固定24cm，零关节足距实测23.701291cm；24cm不再是采样下限，也不是独立均匀步宽分布。
总步距上限48cm、横向上限36cm；改变脚印生成配置不改变实际机器人reset姿态，也不会自动重训旧检查点。

### 10.4 训练命令状态接口

命令项必须提供 `env.command_manager.get_term("footsteps").reward_state`，类型为
`FootstepRewardState`。这是训练特权状态，不增加 actor/ONNX 的 84 维输入：

| 字段 | shape | 含义 |
| --- | --- | --- |
| `phase` | `[N]` | 与正在评价的物理状态对应的全局相位，可 wrap 或连续累计 |
| `targets_w` | `[N,2,3]` | 当前两脚执行/保持的世界 XY/yaw，顺序 left/right，不是未来四步列表 |
| `target_ids` | `[N,2]` int64 | 非负目标编号，换新摆动目标时更新；即使原地重复落点也要换编号 |
| `ground_height` | `[N,2]` | 两脚对应接触平面高度，用于摆动间隙及骨盆高度奖励的地面基准 |
| `frequency` | `[N]` | 必填非负实际发布频率，必须与 actor 同步；0 为站立，不是未来停止请求 |

目标编号和世界目标在该脚整个摆动及随后支撑期间保持不变，到该脚下次起脚才换成下一目标。
首次着地得分锁定后，不能在相位落地边界立刻把 reward_state 替换成未来队列首项。
actor 的未来列表可在理论落地事件后移，但奖励的支撑目标需要另存，生命周期不同。

站立时 `targets_w` 为双脚保持目标，其位置和编号保持不变，actor 槽位重复这两个目标。
不能每帧把保持目标更新到实际脚下，否则滑移误差会被抹掉。

mjlab 当前步序为 termination -> reward -> reset -> command update -> observation。
训练适配器通过首个终止项在奖励前推进，提供与物理步末一致的相位和旧目标快照，再发布下一动作命令；
不能把旧相位误当步末相位，或先消费目标再评价刚刚落下的脚。
独立管理器 `advance()` 返回命令切换前的通用执行快照 `update.completed` 和当前新命令 `update.command`。
环境适配器需在物理步末、奖励计算前调用 advance，将 completed 中的数组堆叠为本接口并另提供地面高度；
不能在奖励计算后才更新快照。当前已实现每环境独立状态的适配，平地 ground_height=0

需要的实体/传感器：`robot`、按 left/right 输出的 `feet_ground_contact`，以及只检测腿间接触的
`self_collision`（沿用 lower_body 配置，不能把上肢扰动导致的碰撞也归咎于策略）。
足底传感器需提供 `found`、`force`（每脚 XYZ 接触合力），并开启 `track_air_time=True`，
供 `soft_landing` 判断实际首次接触；工厂参数 `sensor_name` 会同时传给落地力度项。
所有奖励配置都使用独立的 `SceneEntityCfg`，由管理器解析名称；重复建环境不会共享解析后的实体对象。

奖励测试命令：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 micromamba run -n mj python -m unittest discover -s tests -p 'test_footstep*.py' -v
```

覆盖执行器乱序和上肢排除、腰边缘曲线、绝对 yaw 回零、角度 wrap、首次落地锁分、早/迟落地、
按环境 reset、抬脚和滑移、完整 RewardManager 调用、与 dt 无关的终止代价及 time-out 排除。
落地力度测试覆盖首次接触逐脚选择、两脚合计、力幅值比例、持续支撑零罚分、日志指标和自定义传感器名。
另有真实 CPU 环境及短 PPO 更新测试，均不代表收敛训练，尚不能据此宣称已学会精准低铜损行走

## 11. 停走过渡与站立奖励

已实现 f=0 模型/部署输入、导出能力声明和奖励模式切换；`RandomCommandSource` 决定概率停止和保持后的起步，
`FootstepManager` 执行减速、接触确认及收步/起步，二者已接入 RL 环境，尚无硬件验证，不训练单脚站立

管理器维护 walking -> stopping -> standing -> starting 状态，不新增 actor 输入：

1. 停止意图采样概率设为 `stop_probability=0.30`。约定每次 WALK 的行走指令重采样时抽一次，
  不是每个控制拍抽一次，也不是保证 30% 时间站立；STOP_PENDING/STAND 期间不重复抽样。
  默认重采样间隔3-8s、保持时长2-5s，可配置；也支持 reset 直接站立。
2. 保留已经承诺的四步，在远端补可站立的终止脚印。选择有足够时间减速和完成必要收步的双支撑窗口，
  当前实现先执行原四步，再补一个与远端末脚对齐的收步，随后重复双脚终止目标。
3. 到达目标窗口前缓慢降低 f，但维持正的过渡频率下限，避免在单支撑中途相位停住或无限趋近边界。
  默认减速度0.3Hz/s、过渡下限0.6Hz；在停止条件满足前保持正频率，窗口内再离散置零。
4. 在约定双支撑窗口内完成落脚，实际接触/运动估计确认可双脚支撑后再发布 f=0。
  默认需要连续2拍双接触确认，满足停止相位条件后等待超过0.5s会 fault 并抛出异常，由外部安全处理。
5. STAND 冻结时钟/参考系/足端目标，持续运行策略。再次起步时提供新四步和有准备时间的起步相位，
  恢复正频率，不清空 GRU。

`stop_probability=0.30` 已由独立模块读取执行，不是每个控制拍的概率；站立后仍保持2–5s并自动再起步。
现有 lower_body 速度跟踪任务在观测与步态奖励中使用完整左右周期 `period=0.6 s`，
对应本契约 `f=1/0.6=1.6667 Hz`，左右合计 `2*f=3.3333 步/秒`（200 步/分钟）。
这是目标节拍，不是已测得的实际落地频率。管理器初始 f 取该值，正常慢变范围初值为0.8-1.8Hz，
默认每2秒采样 `df/dt ~ U(-0.3,+0.3)Hz/s` 并保持，每20ms按该变化率积分、触边反向。
随机指令范围、变化率范围和保持时间归属 source；manager 对所有目标用0.2Hz/s执行限速，
随机率以实际f为起点生成下一拍目标，超出执行限速的变化被截住。这些初值尚未训练调优。

默认相位配置的理论双支撑窗口为周期比例 `[0.2,0.3)` 和 `[0.7,0.8)`，
即 rad 下 `[0.4*pi,0.6*pi)`、`[1.4*pi,1.6*pi)`，等价于两脚各 `stance_fraction=0.6`。
调度应共用下面的配置和查询接口，不另写起脚窗口。
站立奖励在任意冻结相位下都要求双支撑，但这不代表允许部署在单支撑时刻突然停止。

### 显式配置与导出

```python
import math
from g1_lower_rl.footstep_phase import FootstepPhaseCfg
from g1_lower_rl.rl.footstep_model import FootstepModelCfg
from g1_lower_rl.tasks.footstep_tracking.rewards_cfg import make_rewards

phase_cfg = FootstepPhaseCfg(
  left_stance_phase=math.pi / 2,
  right_stance_phase=3 * math.pi / 2,
  contact_half_width=0.1 * math.pi,
)
actor_cfg = FootstepModelCfg(phase_cfg=phase_cfg)
rewards_cfg = make_rewards(phase_cfg=phase_cfg)
```

配置单位为弧度，共用半宽应用于理论落地中心两侧；区间起点包含、终点不包含。
窗口起点允许该脚接触，中心是队列的理论落地事件，终点是另一只脚起脚。默认事件表：

| 相位角 x | 计划接触状态 |
| --- | --- |
| `[0,0.4*pi)` | 右脚支撑，左脚摆动 |
| `[0.4*pi,0.6*pi)`，72° 至 108° | 双脚支撑，左脚理论落地中心90° |
| `[0.6*pi,1.4*pi)` | 左脚支撑，右脚摆动 |
| `[1.4*pi,1.6*pi)`，252° 至 288° | 双脚支撑，右脚理论落地中心270° |
| `[1.6*pi,2*pi)` | 右脚支撑，左脚摆动 |

窗口按2pi取模，可跨周期；共用半宽必须保证两个双支撑窗口不重叠，中间保留单支撑。
奖励的接触 mask、摆动进度和落地容许窗口均从这份配置计算，不独立写死落地角。
ONNX/JSON 的 `phase.contact_schedule` 保存中心/半宽、弧度窗口及派生接触边界。
重新加载 checkpoint 必须恢复保存的 `FootstepModelCfg.phase_cfg`，不能只加载权重后使用另一套时序。

部署端通过 `policy.phase_cfg.in_double_support(x)` 查询理论双支撑，不依赖 Torch/MuJoCo。
旧 ONNX 缺少窗口元数据时 `policy.phase_cfg is None`，调用方不得猜测可停止窗口。
旧的两个区间字段格式不再静默转换，需使用对应配置重新导出。相位查询本身不是实测接触判断；
减速、积分和f=0切换由独立 FootstepManager 执行，模型前处理不做这些事。

当前站立奖励率（常规项乘 dt，摔倒另扣固定 10 分）：

```text
hold_cost = mean_left_right(0.5*XY_error/0.08 + 0.5*abs(wrapped_yaw_error)/0.20)
r_stand = -5.0*hold_cost + 2.0*both_contact
      - 2.0*slip - 2.0*copper_proxy - 0.4*waist_yaw^2
       - 2.0*waist_roll_pitch_edges - 0.5*torso_tilt
       - 0.02*action_rate - 2.5e-7*controlled_joint_acc - 2.0*self_collisions
      + 2.0*pelvis_height_score - 1.0*pelvis_upright_filtered_cost
```

-5.0 的保持项来自 landing 的-4.0加support的-1.0，使用实时误差，不保留历史成绩，无未落地附加代价。
一脚未接触不会减少保持代价，且双接触奖励为零；不会仅因恢复抬脚而终止，但f=0时任一脚XY偏差超过1m仍会离轨终止。
铜损、腰部、打滑和防摔始终生效；骨盆高度封顶奖励与低通倾斜项也在站立时生效。
不增加外部高度指令、全身名义姿态回归或双脚 50/50 承重约束。
不惩罚独立控制的上肢运动。若训练后仍晃动，可再讨论轻量速度惩罚及落地后的渐进启用；
当前没有新增站立专用基座速度惩罚，接口测试也不验证物理站立能力。