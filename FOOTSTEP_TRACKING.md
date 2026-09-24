# 脚印跟踪 GRU 模型契约

版本：`g1_footstep_gru_v4`。

当前输入为 `[L_support,R_support,L_next,R_next]`：双脚上一次支撑落点，加双脚各自下一落点，仅预览未来两步。
原始84维、编码89维保持不变，但槽位语义不兼容；v1/v2/v3 ONNX不得换标签复用。
默认GRU隐藏维度64，MLP为256→256→128。当前采用50%站立、50%直接行走开局；此前全站立试跑已停止并删除结果。

已实现模型、输入编码、PPO 模型配置、TorchScript/ONNX 导出、CPU 推理封装，以及训练奖励和防摔终止配置。
另已实现独立的 NumPy 脚印采样、四步管理及频率停走调度模块。
已接入 `G1-Gloria-FootstepTracking` 环境，完成walk-first的15小时训练；目前仅保留最终19490所属的完整训练及源码快照，回放命令见[训练说明](g1_lower_rl/tasks/footstep_tracking/README.md#保留的训练)。
最终19490已通过500步无窗口回放；定量跟踪精度、大规模吞吐和硬件安全仍须独立验证，不能直接用于机器人。
训练入口、参数和时序见 [训练说明](g1_lower_rl/tasks/footstep_tracking/README.md)

实现位置：

- [共享契约](g1_lower_rl/footstep_contract.py)：版本、槽位及维度常量；模型、部署和预览共用，不依赖训练库。
- [独立脚步模块](g1_lower_rl/footsteps/README.md)：Mind Your Steps 风格采样、四步队列、慢变频率和概率停走。
- [模型和导出](g1_lower_rl/rl/footstep_model.py)：`FootstepActor`、`FootstepModelCfg`、`export_footstep_policy`。
- [部署输入和推理](g1_lower_rl/footstep_deploy.py)：`pack_footstep_observation`、`FootstepPolicy`。
- [相位配置](g1_lower_rl/footstep_phase.py)：`FootstepPhaseCfg`，奖励与模型导出共用的理论双支撑窗口。
- [测试](tests/test_footstep_model.py)：输入布局、编码、网络容量、环境时序、导出和连续推理。
- [奖励配置](g1_lower_rl/tasks/footstep_tracking/rewards_cfg.py)：`make_rewards`、`make_terminations`。
- [奖励实现](g1_lower_rl/tasks/footstep_tracking/rewards.py)及[测试](tests/test_footstep_tracking_objectives.py)：摆动指数奖励、落地锁定线性代价与计划接触节拍。

## 1. 任务边界

- 观测全身 29 轴，不包括两个 Gloria-M 夹爪关节。
- 只控制 15 轴：左腿 6、右腿 6、腰 yaw/roll/pitch。
- 骨盆和 torso 各输入角速度与单位重力方向。
- 固定左右顺序输入双脚支撑基准和双脚各自下一落点，共四个 XY/yaw 位姿；只有后两个是未来落点。
- 输入全局相位和当前频率；不输入身体高度、线速度指令、骨盆相对支撑系位姿、接触力或外力真值。
- f>0 为行走/踏步，f=0 为双脚站立；原地踏步通过正频率和重复左右各自固定的落点表达。
- 上肢和夹爪由独立控制器控制，本策略不得覆盖其输出。

## 2. ONNX 输入

所有输入输出为 `float32`；部署固定 batch=1。下表下标从零开始，切片为左闭右开。

| 输入 | 默认 shape | 含义 |
| --- | --- | --- |
| `obs` | `[1,84]` | 未归一化、未编码的原始观测 |
| `h_in` | `[1,1,64]` | GRU `[层数,batch,hidden_size]` 隐状态 |

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
| `72:75` | 3 | L_support | 左脚上次计划落脚点；摆动时保留起脚前基准 |
| `75:78` | 3 | R_support | 右脚上次计划落脚点；摆动时保留起脚前基准 |
| `78:81` | 3 | L_next | 左脚首个未消费落点，包含正在迈向的目标 |
| `81:84` | 3 | R_next | 右脚首个未消费落点，包含正在迈向的目标 |

四个脚印槽位必须有效。本版没有 padding/valid 掩码；缺少远期指令时，必须由调度器显式补齐
可执行的重复落点或进行安全处理，不能用全零表示数据缺失。输入形状为 `[4,3]`，排列为
`[L_support,R_support,L_next,R_next]`，每对固定左右顺序，不按时间交换槽位。
support重置时取实测双脚位姿，此后仅在计划落地事件更新为该次计划落点；不跟随实测脚位或滑移。
抬脚不改变四个世界槽位；落地时将该脚next移到support，再选择该脚的下一个队列目标。
next是每脚队列中的首个未消费目标，不再排除正在执行的摆动目标；两项恰好是内部队列最近两步。
例如左脚支撑在L0、右脚从R0迈向R1：本版输入 `[L0,R0,L1,R1]`，不是v3的 `[L0,R1,L1,R2]`。
重复原地踏步时位姿可以相同，但目标编号不同；计划落地不等于传感器确认落地。

f=0 时四个槽位为 `[left_hold,right_hold,left_hold,right_hold]`，两个不同的足底保持目标
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
  -> GRU(input=89, hidden=64, layers=1)
  -> MLP(64 -> 256 -> 256 -> 128 -> 15, ELU)
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
| `h_out` | `[1,1,64]` | 下一拍的 `h_in`；没有 LSTM cell state |

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

- 跨过左/右计划抬脚边界：内部执行目标切换为该脚next，供控制拍奖励使用；观测的世界槽位不变。
- 跨过 `pi/2 + 2*k*pi` 或 `3*pi/2 + 2*k*pi`：消费该侧队头，将落点写入support，next推进至该脚下一目标。
- 内部仍维护四个承诺落点，停车仍完成四步加收步；对actor只显示两脚support和最近两个未来落点。
- 奖励读取独立的完成拍执行目标，不能将actor的support或已经推进的next直接用作奖励目标。
- 相位、列表索引、参考系和观测必须在同一拍一致更新。普通换步不 reset GRU。
- 网络中的 sin/cos 是全局相位编码，不分别绑定左右脚，所以不隐含 1:3 步间隔。
- 模型不判断是否踩准，不输出换步信号，也不自行推进脚印。

独立管理器支持行走正频率随机游走、触边反弹及停止收步；实际数值配置见模块 README。
训练适配器使用批量Tensor管理器，将脚步队列、时钟、随机调度及局部重置保留在仿真设备上，CUDA默认编译执行。
独立NumPy管理器继续服务部署和预览，也是状态机的行为对照；训练热路径不再逐环境创建Python对象或回读脚位。
模型机器契约的频率能力范围仍标为 null，
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
import inspect
import torch
from tensordict import TensorDict
from g1_lower_rl.rl.footstep_model import FootstepActor, FootstepModelCfg, export_footstep_policy

cfg = FootstepModelCfg()
options = {key: value for key, value in asdict(cfg).items()
           if key in inspect.signature(FootstepActor).parameters}
example = TensorDict({"actor": torch.zeros(1, 84)}, batch_size=[1])
actor = FootstepActor(example, {"actor": ["actor"]}, "actor", 15, **options)
torch.save({"actor_state_dict": actor.state_dict(), "actor_config": asdict(cfg)}, "footstep_actor.pt")
export_footstep_policy(actor, "export/footstep/policy.onnx")
```

训练接入时传一个 84 维原始观测组，不要提前编码成 89 维；类路径可由 rsl_rl 的 resolver 加载。
训练已绑定奖励与104维 MLP critic 观测；不继承旧速度任务的课程，默认tracking的方向变化与频率变化率分别注册为独立课程
默认tracking每回合reset随机初始方向和0.8–1.8Hz目标步频，并独立抽取站立/行走模式及左右起脚
方向变化在1500/2500/4000次逐档放开，频率变化率从±0.01Hz/s开始，在800/2000次放开至±0.15/±0.3Hz/s
阶段表见[指令课程](g1_lower_rl/tasks/footstep_tracking/README.md#指令变化课程)，独立脚步库不依赖课程或训练轮数
独立 `--profile walk-first` 使用 [walk_first/curriculum.py](g1_lower_rl/tasks/footstep_tracking/walk_first/curriculum.py#L10)：
0–1499轮固定向前；1500/1750/2000/2250轮分别开放初始方向±30/60/120/180度，2500轮前回合内方向不变。
1500轮起同步放宽站距到0.21–0.27m，使侧移可行；候选距离仍固定0.27m。
2500/3000/3500/4000轮逐档放宽候选步距到0.24–0.30/0.20–0.36/0.16–0.40/0.12–0.48m，
回合内指令方向增量到±15/30/60/180度。完整站距范围和采样语义见[独立课程](g1_lower_rl/tasks/footstep_tracking/README.md#先行走后扩展的独立配置)。
保持1.2Hz请求步频、固定名义脚掌朝向和原有宽容XY/yaw奖励，不加入精准落脚惩罚。
4500/5000/5500轮继续开放每个新脚印独立的yaw随机量±10/20/30度；4500前为0，逐脚方向噪声保持0。
yaw相对初始名义脚掌朝向，不跟随行进方向；仍保留相邻脚印yaw变化限幅。
轮号按累计训练进度选择，从17899恢复后下次reset直接进入±30度档，不重新走一遍小角度课程。
范围原地更新设备参数，不重写承诺脚印或正在进行的相位；已启动作业和Viser使用冻结源码，不自动采用新课程。
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
OMP_NUM_THREADS=1 micromamba run -n mj python -m pytest tests/test_footstep_model.py tests/test_tensor_footsteps.py -q
```

测试包括84/89维布局、扩容网络、固定左右support/next槽位、角度周期性、GRU reset、ONNX Runtime
多拍动作/隐状态对齐、旧契约拒绝及真实环境支撑基准输入与完成拍执行奖励的分离。
规划器测试覆盖NumPy/编译GPU状态一致性、摆动期间基准冻结、计划落地更新和仅预览最近两步及局部重置；不验证策略已学会行走。

下一轮训练遵循落点/脚掌朝向/节拍跟踪和不摔倒的目标；2026-09-21铜损权重调整为-0.25，
腰 yaw 靠近零、腰 roll/pitch 临近限位惩罚；加入头部高度单调封顶奖励，但不增加身体高度指令。
当前先保持抗扰动初始档，不增加上肢摆动或外力难度；指令课程只按已约定阶段放开。
没有电机电阻/力矩常数/传动标定时，力矩平方仅称为铜损代理。
脚印范围和频率参数已有可调实现初值，仍需验证可行性；阶段范围和训练预算仍须结合实际训练评估。
奖励初始权重见下一节，训练中需根据分项指标调节，不代表已经验证的最优权重。

## 10. 训练奖励契约

训练端加速实现见 [批量管理器](g1_lower_rl/footsteps/tensor_manager.py) 和
[训练适配与基准说明](g1_lower_rl/tasks/footstep_tracking/README.md#设备驻留批量脚步后端)。
加速不改变本节奖励定义，也不修复已观察到的短回合/频繁跌倒问题；部署观测及动作契约不变。

入口为 `make_rewards(command_name="footsteps", sensor_name="feet_ground_contact")`。
配合 `RewardManager(scale_by_dt=True)` 使用：常规项为每秒奖励率，环境每拍乘 `step_dt`。
摔倒和落脚精度是事件代价，函数内部除以 `step_dt`，所以每次真正终止固定扣10分，每个有效落脚目标扣一次归一化误差；
时间上限结束不算摔倒。不要再额外对总奖励做非负裁剪。

### 10.1 当前奖励表

| 名称 | 权重 | 意图/原始值 |
| --- | --- | --- |
| `footstep_landing` | -1.0/事件 | 每目标在计划摆动中离地后的首次触地评价一次XY/yaw误差；无漏落或时序附加罚；f=0关闭 |
| `footstep_swing_position` | +5.0 | 计划摆动脚 `s^2*exp(-(XY_distance/0.20)^2)`，须有计划支撑脚实际接触 |
| `footstep_swing_yaw` | +5.0 | 计划摆动脚 `exp(-(wrapped_yaw_error/0.10)^2)`，阶段门控同上 |
| `contact_schedule` | +1.0 | 双脚实测接触模式完全匹配计划才得1，任一脚不符得0，f=0关闭 |
| `foot_air_time` | +3.2 | 仅计划正确侧单支撑，时长按当前计划摆动时长归一化并受摆动进度限制，原始值最高0.4 |
| `swing_clearance` | -0.5 | 双脚计划足高绝对误差除以0.11m后求和；摆动峰值0.11m，支撑及停脚为0 |
| `swing_contact` | -0.5 | 计划摆动脚实际触地的sin平方相位加权代价；起落边界为0、中点为1 |
| `foot_slip` | -2.0 | 实际触地脚的 XY 速度平方和，f>0启用 |
| `feet_slip_still` | -4.0 | 实际触地脚的 XY 速度模长之和，f=0启用 |
| `stand_still_feet` | -0.5 | 未接触脚数，f=0启用 |
| `feet_hold_position` | -0.2 | 仅f=0，双脚相对冻结停脚目标的XY距离超过3cm部分按10cm归一化平方后取平均 |
| `soft_landing` | -0.002 | 首次触地接触力模长惩罚，f>0启用，持续支撑不收费 |
| `lower_body_copper_proxy` | -0.25（基础） | 15个受控腿腰执行器的实际力矩统一尺度平方和；f=0乘2，有效权重-0.5 |
| `waist_yaw_zero` | -0.4 | `waist_yaw_joint` 绝对角度平方，目标是 0 rad，不是默认姿态偏差 |
| `waist_roll_pitch_edges` | -2.0 | 两个腰轴靠近硬限位的边缘平方惩罚，内部区域为零 |
| `torso_upright` | -0.5 | torso 相对竖直倾角的平方（弧度），`theta=atan2(norm(g_b.xy),-g_b.z)`，不约束世界 yaw，不控制身体高度 |
| `body_ang_vel` | -0.05 | `torso_link` 实际世界系X/Y角速度平方和，站立和行走均启用，不罚Z轴转向 |
| `head_height` | +0.4 | `clip((head_z-ground_z)/1.254,0,1)`，直接取 `head_collision` 几何体中心，至少一脚接触且未失败时启用 |
| `head_height_low` | -1.0 | `relu(1.15-head_height_above_ground)/0.2`，站立和行走均启用，不因腾空免罚 |
| `stance_knee_bend` | -0.2 | 计划支撑腿过度屈膝平方代价，停脚容许30度、行走支撑45度，尺度45度；摆动腿不罚 |
| `pelvis_upright_filtered` | -1.0 | 随f调整的一阶低通骨盆重力向量XY分量平方和，不罚步频摆动的原始幅度 |
| `action_rate` | -0.02 | 15 维动作相邻拍差的平方和 |
| `controlled_joint_acc` | -2.5e-7 | 仅 15 轴关节加速度平方和，不罚外部控制的手臂/夹爪 |
| `self_collisions` | -2.0 | 复用 lower_body 的腿间自碰撞传感器，统计控制拍内超过 10 N 的接触子步数 |
| `fall` | -10.0/次 | 真正终止事件代价，不罚 time-out |

这是带权 RL 目标，而非保证不摔的约束优化或安全证明。以可达脚印下的落地精度和存活为主要目标，
再降低能耗。能耗项过重可能导致不愿迈步，过轻则可能动作剧烈，需要结合实际训练调权。
时钟由外部指定，不能由策略选择停住或减慢时钟以逃避跟踪。

### 10.2 精度、节拍与防刷分

2026-09-20采用密集摆动指数奖励和一次性落脚线性惩罚，默认移除持续支撑位置精度项。
摆动核参考 [Mind Your Steps §IV-C及附录D、F](https://arxiv.org/html/2606.08253v1)，
位置尺度0.20m、角度尺度0.10rad，两个分量各权重+5，乘共享相位计算的摆动进度平方。
按计划摆动阶段启用，且至少一只计划支撑脚必须实际接触；双支撑及f=0时关闭。
XY/yaw引导不要求摆动脚已经离地，但只有摆动脚接触或双脚腾空时不给逼近奖励；另以swing_contact对计划摆动触地收费。
进度平方使起脚边界奖励从0增长，避免一起脚就能获得终点满分；这不保证足速或物理稳定性，需要重训验证。
默认工厂参数 `position_std=0.08 m`、`yaw_std=0.20 rad` 保留名称，但用于落脚线性代价尺度。
这些尺度不是容许误差上界或已实现的实机精度。

```text
supported = any_feet(planned_stance & actual_contact)
r_swing_xy = 5 * supported * sum_swing(swing_progress^2 * exp(-(XY_distance/0.20)^2))
r_swing_yaw = 5 * supported * sum_swing(swing_progress^2 * exp(-(wrapped_yaw_error/0.10)^2))
cost = 0.5 * XY_distance/position_std + 0.5 * abs(wrapped_yaw_error)/yaw_std
raw_landing = sum_feet(cost * first_contact_after_swing_air) / step_dt
reward_landing_per_step = -1 * raw_landing * step_dt
```

线性代价不截断，XY整体距离与yaw各占50%，不把X和Y拆开。
旧 `footstep_approach`、`footstep_distance`、旧指数落地/支撑原语及 `distance_penalty` 参数已删除。
默认配置28项（7正、21负），独立walk-first配置27项（移除落脚事件代价）；铜损行走-0.25、f=0时-0.5。
历史A/B脚本须使用对应源码快照，不能直接调用当前奖励工厂。
持续续训入口为 `scripts/train_footstep_resume.py`，支持双卡精确恢复及更新结束后同步保存退出。
比较run需使用原始XY/yaw误差和存活率，不能通过不同定义下的总奖励判断效果。
下肢平滑保持动作差分-0.02及实际关节加速度-2.5e-7，不添加站立增量或力矩差分项

`body_ang_vel` 直接复用速度任务的同名奖励，绑定 `torso_link`，奖励率
`-0.05 * sum(torso_angular_velocity_world_xy^2)`，站立和行走全程启用，不做低通或接触门控。
不惩罚世界Z轴转向，不替代原torso倾角项或静止骨盆零角速度奖励，也不直接改变手臂PD或物理阻尼。
需重新训练后评估torso晃动，历史冻结模型不自动采用。

静止零速度约束沿用lower_body瞬时零指令指数核，读取根刚体机体系实际速度：

```text
stand_still_linear_velocity = 2.0 * exp(-(vx^2 + vy^2 + 1.5*vz^2) / 0.2) * (f == 0)
stand_still_angular_velocity = 0.5 * exp(-(wz^2 + 0.05*(wx^2 + wy^2)) / 0.49) * (f == 0)
```

门控使用本拍冻结执行频率，精确f=0（包括settling）启用，任意正频率关闭。
这是静止满分、移动扣减的正奖励率，不约束固定朝向，不依赖实际接触，不添加历史状态。
不增加平均速度、骨盆漂移或静止动作差分项；行走精度、课程和其他奖励不变，需训练后验证碎步改善。

两只脚的 yaw 均为 `left_foot/right_foot` site 的世界 yaw，使用 wxyz 四元数转换，并将角差 wrap。
默认单脚支撑占一个完整周期的 0.6，摆动占 0.4，总双支撑占 0.2；
左右默认理论落地边界为 pi/2、3pi/2。通过 `FootstepPhaseCfg` 显式修改窗口，频率变化不重置相位。
旧 `stance_fraction` 参数仍可使用，但不能与 `phase_cfg` 同时传入；其派生配置必须同步给 actor。
`phase_windows` 是后续调度器和奖励应共同使用的窗口定义，不得另外写一套起脚边界。

有效落脚要求该脚在本目标计划摆动期真正离地，随后发生首次接触边沿；只在该拍评价误差，不要求处于计划支撑窗口。
每个目标最多计一次，后续滑动、抬落和接触抖动不重复计分；目标ID变化及局部reset只清理相应历史。
两脚同拍触地时事件代价求和，不按支撑脚数平均；内部除以dt后每个事件的总罚分不随控制步长变化。
事件权重设为-1而不是沿用旧持续奖励率-4：例如单脚XY误差5cm且yaw误差0，本次触地扣0.3125分。
不落脚、始终贴地或早晚触地没有额外的精度附加罚，时序交给独立 `contact_schedule` 引导；原有执行器失败终止不变。

以上事件评价针对 f>0。f=0清理历史，落脚代价为0，不要求reset站立先抬脚。
支撑mask覆盖为双脚支撑，两个摆动奖励关闭；clearance继续要求两脚高度为地面，重新起步的新计划步须先离地再触地才评价落脚。
精确零频率才切换模式，小的正频率仍按行走评分。

默认移除 `footstep_support`，不在落地后的支撑阶段持续追罚计划落点XY/yaw误差；支撑观测槽位和独立规划器不变。
防滑、接触时序、原有1m离轨和摔倒终止仍保留，放松支撑精度不等于允许无限滑移。
2026-09-21的步态引导不再直接照搬速度任务：使用同一 `phase_windows` 的计划侧别和摆动进度。
支撑脚目标离地高度为0，摆动脚为 `0.11*sin(pi*s)^2`，起落边界高度和斜率均为0。
净空原始代价为 `sum_feet(abs(site_z-ground_height-target_height)/clearance)`，默认clearance=0.11m，权重-0.5。
不乘水平脚速，不裁剪高度误差。单脚中点完全漏抬11cm时罚0.5/秒，旧版本罚0.132/秒，实际增强约3.79倍。
过高、过低和支撑脚悬空均收费；f=0时仍要求两脚贴地。不规定完整XYZ或关节轨迹，脚步几何、起停及课程不变。

接触相位奖励要求 `current_contact_time>0` 和 `found>0` 均完整匹配双脚计划接触模式。
计划单支撑时：正确侧得1，双脚接触、错侧或腾空均得0；计划双支撑缺脚得0，f=0关闭。
新增 `swing_contact` 原始成本 `sum_planned_swing(found_contact*sin(pi*s)^2)`，权重-0.5，
中点触地额外扣0.5/秒，起落边界平滑为0；支撑及f=0不计此项，不按触地事件锁分。
XY/yaw密集引导不增加离地门控，仍可在尝试抬脚前获得，避免把所有过程学习信号关闭。
单支撑时长奖励要求计划与实测侧别完全匹配（found及接触计时器均匹配），仅在计划单支撑时计算：

```text
held_time = min(stance_contact_time, swing_air_time)
T_swing = (1-stance_fraction_of_swing_foot)/f
raw_air_time = 0.4 * min(clip(held_time/T_swing,0,1), swing_progress)
```

错侧、双接触、腾空、计划双支撑和f=0不给奖励；传感器接触中断重置时长，摆动进度上限限制跨周期的旧时长。
参数 `air_time_scale=0.4` 是保留的幅值上限，不再是固定秒数阈值，替换 `air_time_threshold`。
每拍按当前f归一化（默认T_swing为0.4/f）；变频时是瞬时周期尺度，不预测未来精确落地时刻。
不新增独立时钟或历史。站立抬脚项仍使用接触计时器，实际触地防滑项继续使用found。
walk-first配置已有闭环训练；它与默认精度配置不同，不能仅凭训练完成或公式测试认定异常步态已解决。

落地力度项 `soft_landing` 直接复用 `mjlab.tasks.velocity.mdp.soft_landing`，权重与 lower_body 一致，
不传 twist 指令，而由适配函数使用 `f>0` 门控，对齐速度任务的运动状态限定。

```text
first_contact = feet_sensor.compute_first_contact(dt=step_dt)
landing_force = sum_feet norm(contact_force_xyz) * first_contact
reward_soft_landing = -0.002 * landing_force * (f>0) * step_dt
```

它按实际首次接触事件计费，不依赖理论相位或落地精度是否合格，也不使用落地精度项的目标编号锁分。
两脚同拍落地时求和；持续支撑即使受力较大也不收费。使用完整 XYZ 力模长，单位 N，
不是力矩平方、冲量、物理子步最大峰值或按体重归一化的值。保留原任务的 dt 缩放，不除以 step_dt。
复用函数记录 `Metrics/landing_force_mean`，为本拍所有首次触地脚的平均力模长；本拍无落地时为零，
因此跨时间直接平均该指标不等于跨所有落地事件的平均冲击力。
接触力仅作为训练奖励数据，不增加 actor 的 84 维输入，也不引入夹爪观测。

### 10.3 铜损代理与腰部

铜损基础权重保持-0.25，精确零执行频率时通过 `standing_scale=2.0` 将代价加倍，有效权重-0.5。
门控读取本拍冻结 `reward_state.frequency`，包括settling；任意正频率仍为-0.25，不用实际接触作为开关。

```text
pushing_limit[j] = (q[j] <= lower[j] + margin and torque[j] < 0)
                   or (q[j] >= upper[j] - margin and torque[j] > 0)
limit_multiplier[j] = 10 if pushing_limit[j] else 1
copper_proxy = sum_j copper_weight[j] * limit_multiplier[j] * (actual_actuator_torque[j] / 100 Nm)^2
reward_rate = -0.25 * copper_proxy * (2 if f == 0 else 1)
```

`limit_margin=0.1°`，`limit_scale=10.0`，使用实际关节角和模型硬限位，已越界也触发。
每个关节独立、每拍重算，仅放大继续向限位外施力的铜损；帮助退出限位的反向力矩维持原权重。
倍率作用于平方项的系数而非力矩本身；站立限位外推关节总倍率20，行走10。
`at_lower_limit`、`at_upper_limit` 和 `pushing_limit` 张量可从铜损项实例读取，不新增actor观测或历史锁存。
关节与执行器按名字分别映射；真实模型测试确认当前15轴传动均为正向1:1。
该项不裁剪动作或力矩，训练效果须验证；原语默认 `limit_scale=1.0` 可关闭放大。

`actual_actuator_torque` 读取奖励采样时刻的 `asset.data.actuator_force`，
按执行器名称精确选择下肢 15 轴，不用 joint_ids 索引 actuator_force；不包含关节约束力、外力、手臂或夹爪。
目前是控制拍采样值，不是物理子步 RMS，也没有电机热模型。
所有轴共用 100 Nm 作为数值尺度，默认系数全为 1；不是按各轴峰值力矩分别归一化，
因此在未知电机参数时相同 Nm 的力矩具有相同代价。

如有电阻、力矩常数和传动标定，可传入按 15 个执行器完整命名的正 `copper_weights`。
实际铜损对应 `sum R_j * I_j^2`，电流与关节力矩之间还需传动/效率模型；
当前默认值只表示力矩平方代理，不能标注为真实瓦特数。

头部高度奖励直接取 `asset.data.geom_pos_w` 中 `head_collision` 的世界Z，奖励率为 `0.4*clip(h/1.254,0,1)`。
在0到1.254m内线性增长，之后封顶0.4；另以 `head_height_low` 排斥明显偏低的姿态。
低高度原始代价为 `relu(1.15-h)/0.2`、权重-1.0，1.15m及以上不罚，低10cm罚0.5/秒。
使用同一头部几何体和地面基准，站立及行走均启用，且不因腾空或终止拍免罚；不是新的终止阈值。
该阈值允许名义封顶高度以下约10cm变化，不规定膝角。权重和阈值为待验证初值，效果仍需重训验证。
当前模型的 `head_link` 是网格名而非独立body；使用已有头部几何体，不增加刚体，也不再使用手工 `body_point`。
封顶基准为原骨盆0.78m加腰部0.044m和头部中心局部Z偏移0.43m；双脚腾空或非超时失败拍不发此奖励。
高度参考 `ground_height` 而不是运动中的脚底；它不改变84维actor输入，也不加入高度命令或固定膝角。
原 `pelvis_height` 及中间的 `torso_height` 项均由 `head_height` 替换；它仍不是保证站直的硬约束。
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

另加三项软约束：脚位保持和膝角保留容差，脚掌平放直接惩罚倾角平方，不改终止或动作映射：

```text
hold_error = norm(actual_foot_xy - reward_state.targets_w.xy)
feet_hold_position = -0.2 * mean_feet((relu(hold_error - 0.03) / 0.10)^2) * (f == 0)
q_free = pi/6 if f == 0 else pi/4
stance_knee_bend = -0.2 * mean_planned_support((relu(actual_knee_angle - q_free) / (pi/4))^2)
active_feet = planned_support OR actual_contact OR (f == 0)
feet_flatness = -0.2 * mean_active_feet((sole_normal_tilt / (pi/12))^2)
```

停脚目标在重置时取实测脚位，正常收步时为最终计划脚位，不跟随实际漂移更新；
3cm内允许调整，不固定24cm站距，任意正频率时关闭脚位保持项，因此不限制正常侧移和变向。
f=0包括停止后的settling阶段。屈膝项按共享相位选择支撑腿、f=0选择双腿；
使用实际关节角，停脚允许30度、行走支撑允许45度，摆动腿不罚，不要求膝角回零。
脚位保持和膝角两项均不按实际接触门控，不会因抬脚而免除应承担的代价；不新增历史状态。
脚掌平放以真实足底site法向相对平地法向的夹角计算，不直接惩罚踝角，不限制yaw。
行走和静止均生效：计划支撑或实际触地就计罚，f=0覆盖双脚；提前落地也约束，离地摆动不罚。
无容差，直接惩罚倾角平方，归一化尺度15度（不是允许误差）；按参与脚数取平均，持续计奖励率而非首次触地事件，不依赖接触面积估计。
仍为权重-0.2的软约束，不是倾角硬上限，也不自动改变已训练模型。
它们都是奖励率，外部乘step_dt；新增尺度需训练验证，当前冻结模型与预览不自动改变。

`make_terminations()` 增加 `footstep_distance`：任一计划支撑脚到本拍执行目标的 XY 距离严格超过1m，立即非超时终止。
使用时钟推进后的 `reward_state.targets_w` 快照，不使用下一步目标；不要求实际接触，不设置持续时间宽限。
摆动脚不参与此判断，f=0检查双脚。它同样触发固定10分的 `fall` 事件代价。
仅摔倒类条件复用motion_tracking的概率终止：骨盆倾角超过70度（fell_over）或相对最低足底高度低于0.35m（collapsed），每控制拍以0.005概率结束。
两判据同拍共享伯努利结果，每拍重新判断，恢复到阈值内就不再判死；50Hz下持续失败平均等待约4s，不是固定或保证的宽限。
脚步离轨、规划器故障（含收步后0.5s接触确认超时）和60s回合超时均不采用此概率，仍立即终止/截断，即使同时摔倒也不延后。
概率未命中的失败拍不触发fall事件罚分，但常规奖励继续计算；训练与play同配置，恢复能力需重训验证，不会自动扶起机器人。
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
| `ground_height` | `[N,2]` | 两脚对应接触平面高度，用于摆动间隙及头部高度奖励的地面基准 |
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
`FootstepManager` 执行相位对齐的末段减速、接触确认及定时升频起步，二者已接入 RL 环境，尚无硬件验证，不训练单脚站立

管理器维护 walking -> stopping -> settling -> standing -> starting 状态，不新增 actor 输入：

1. 停止意图采样概率设为 `stop_probability=0.30`。约定每次 WALK 的行走指令重采样时抽一次，
  不是每个控制拍抽一次，也不是保证 30% 时间站立；STOP_PENDING/STAND 期间不重复抽样。
  默认重采样间隔3–8s、保持时长2–5s，可配置。训练和Viser使用 `initial_standing_probability=0.5`，独立预览保留全站立开局
  站立组reset从实测双脚位姿建立原地目标，f=0并冻结在90°或270°双支撑中心；保持后按半秒斜坡起步
  行走组直接使用抽取的正频率，从随机选中脚的抬脚边界开始，默认左脚288°或右脚108°，不经过等待及起步斜坡
  两组都独立随机选择左右起脚，队头、当前执行目标和参考脚与相位一致；站立组90°后右脚先抬、270°后左脚先抬
  局部重置只重采样所选环境，不改物理初始姿态或速度；命令为行走不代表已处于稳定行走状态，f=0也不保证已经站稳
2. 保留已经承诺的四步，在远端补可站立的终止脚印。选择完成必要收步后的双支撑窗口，
  当前实现先执行原四步，再补一个与远端末脚对齐的收步，随后重复双脚终止目标。
3. STOPPING 前段保持请求时的频率，末段用 `stop_duration_s=0.5` 线性减速至零。
  终点取最后收脚的理论落地中心之后半个双支撑半宽，保证已经消费收脚目标、但尚未进入下一次起脚。
  连续线性减速的相位积分为 `pi*f_start*stop_duration_s`，据此安排开始减速的时刻；每拍发布该拍的平均频率，
  精确积分包含减速起止边界的控制拍，并在终拍锁定停止相位，避免离散步长越过窗口。
  已删除旧 `stop_deceleration`、`stop_frequency_floor` 参数；停步时长仍包含承诺四步与必要收步，不是立即停车。
4. 到达停止相位即发布 f=0 并进入 SETTLING，冻结相位、参考系、队列及双脚终止目标，不再触发起脚。
  默认连续2拍双接触确认后进入 STANDING；若此前已满足确认，可当拍完成转换。
  等待从到达终点开始计时，超过0.5s仍未确认则 fault；等待期间不重新踏步，也不开始站立保持/自动重启计时。
  因此 f=0 表示双支撑指令，不保证已经实际站稳；它可对应 SETTLING 或 STANDING。
5. STAND 冻结时钟/参考系/足端目标，持续运行策略。再次起步时生成内部四步计划，发布双脚next并保留冻结相位和双脚support基准，
  按 `start_duration_s=0.5` 线性升到请求频率，不清空 GRU。每次起步重置进度，`f=request_frequency*progress`，
  进度每控制拍增加 `control_dt/start_duration_s`、最多为1；发布首拍即取一个进度增量，默认25拍内达到目标。
  例如目标1.6667Hz，首拍约0.0667Hz、之后每拍增加约0.0667Hz；不再使用固定 `start_acceleration`。

`stop_probability=0.30` 已由独立模块读取执行，不是每个控制拍的概率；站立后仍保持2–5s并自动再起步。
现有 lower_body 速度跟踪任务在观测与步态奖励中使用完整左右周期 `period=0.6 s`，
对应本契约 `f=1/0.6=1.6667 Hz`，左右合计 `2*f=3.3333 步/秒`（200 步/分钟）。
这是起步后的目标节拍，不是重置时实际频率或已测得的落地频率。默认重置 f=0，正常行走慢变范围为0.8-1.8Hz，
默认每2秒采样 `df/dt ~ U(-0.3,+0.3)Hz/s` 并保持，每20ms按该变化率积分、触边反向。
随机指令范围、变化率范围和保持时间归属 source；manager 在 WALKING 对目标用0.2Hz/s执行限速，
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
末段减速、落地等待、起步升频、相位积分和f=0切换由独立 FootstepManager 执行，模型前处理不做这些事。

当前站立奖励率（常规项乘 dt，摔倒另扣固定 10 分）：

```text
r_stand = -0.5*airborne_foot_count
  - 4.0*contact_slip_speed - 0.5*sum_feet(abs(site_z-ground_height)/0.11)
  - 0.2*mean_feet((relu(hold_xy_error-0.03)/0.10)^2)
  - 0.2*mean_knees((relu(knee_angle-pi/6)/(pi/4))^2)
  - 0.25*copper_proxy - 0.4*waist_yaw^2
       - 2.0*waist_roll_pitch_edges - 0.5*torso_tilt^2
       - 0.02*action_rate - 2.5e-7*controlled_joint_acc - 2.0*self_collisions
  + 0.4*clamp(head_height/1.254,0,1) - relu(1.15-head_height)/0.2
  - 1.0*pelvis_upright_filtered_cost
```

站立时没有landing或旧support项的持续精度代价，另以带容差的feet_hold_position限制漂移，没有未落地附加罚。
站立时接触相位奖励关闭，改用未接触脚数惩罚；不会仅因恢复抬脚而终止，但f=0时任一脚XY偏差超过1m仍会离轨终止。
铜损、归一化双脚足高、防滑、脚位保持、膝角超限、腰部、防摔、线性头高及低头高度代价、骨盆低通倾斜项在站立时生效。
不增加外部高度指令、全身名义姿态回归或双脚 50/50 承重约束。
不惩罚独立控制的上肢运动。若训练后仍晃动，可再讨论轻量速度惩罚及落地后的渐进启用；
当前没有新增站立专用基座速度惩罚，接口测试也不验证物理站立能力。