# 最省电站立任务配置说明

任务 ID `G1-Gloria-Stand`，机器人 Unitree G1 + 双 Gloria-M 夹爪（31 关节），50 Hz
（`dt=0.005`，`decimation=4`），回合 20 s（1000 拍），平地。

**目标只有一句话：站着不摔的前提下，下肢电耗最小。姿势不限定。**

站成什么样是优化的**结果**，不是输入——配置里没有一项要求「腰要正」「重心居中」
「站到 0.72 m」，理由见第 3 节。训出来的策略学会了**利用机械限位省力**：把关节顶到
行程末端，让限位而不是电机去扛重力。

```bash
# 训练（列表字面量必须加引号；--agent.max-iterations 是增量）
PYTHONPATH=. MUJOCO_GL=egl python scripts/train.py G1-Gloria-Stand \
    --env.scene.num-envs=4096 --gpu-ids '[0]' \
    --mirror-schedule '((0,0.5),)' \
    --agent.resume True \
    --agent.max-iterations 8000 --agent.run-name stand

# 导出部署用 ONNX（会写入 action_joint_names / obs_joint_*_joint_names）
PYTHONPATH=. python scripts/export_onnx.py \
    --checkpoint logs/rsl_rl/g1_gloria_standing/<run>/model_12000.pt \
    --task-id G1-Gloria-Stand
```

---

## 1. 策略接口

### Actor 观测（79 维）

噪声与延时模型整套复用 `G1-Gloria-LowerBody-Flat` / `-GRU`，一个参数都没改。

| 顺序 | 项 | 维数 | 噪声与延时 |
|---:|---|---:|---|
| 1 | `base_ang_vel` | 3 | 高斯 `std=0.115`，每局偏置 `std=0.03`，0–2 拍延时 |
| 2 | `projected_gravity` | 3 | 高斯 `std=0.029`，每局偏置 `std=0.01`，0–2 拍延时 |
| 3 | `joint_pos` | 29 | 相对默认位姿，含编码器零位偏置，高斯 `std=0.006`，0–2 拍延时 |
| 4 | `joint_vel` | 29 | 高斯 `std=0.87`，0–2 拍延时 |
| 5 | `actions` | 15 | 上一拍策略动作（未裁剪的原始输出） |

**没有任何指令项。** 这是与行走任务最大的结构差异：没有 `command_twist`、
没有 `command_height`、没有 `phase`。

**关节量取全身 29 轴（12 腿 + 3 腰 + 14 臂），但动作只有下肢 15 轴。**
上肢在实机上由 VR IK 自行驱动，它摆到哪儿直接决定质心落在哪儿；挡在观测外面，下肢就
只能从骨盆姿态事后反推，等于把一个可测的量硬做成扰动。`reset_arm_pose` /
`arm_pose_drift` 保证这 14 维在训练分布里有方差。

**两个夹爪偏心轴不进观测。** 它们在训练里恒为 0，观测归一化学到的标准差接近零，
真机上夹爪一开合就会被除成巨值盖掉平衡信号。

延时不每拍重采：滞后是机器人的属性，不是每一拍的属性；每步换一次等于给关节速度叠一层
白噪声。`delay_hold_prob=0.98` 让它在一局内几乎恒定。

### Critic 观测（82 维）

actor 的 5 项深拷贝一份（关掉延时），再加 `base_lin_vel`（3 维，取自
`robot/imu_lin_vel`）。根线速度直接决定「站稳没有」，给了它值函数才不用从含噪的
角速度里反推。

### 动作（15 维）

`q_target = q_default + scale · action`，PD 由 MuJoCo 内置位置执行器执行。
目标**不按关节行程裁剪**——底层是 PD，关节靠到限位还想要力就必须把目标顶到行程外。

| 关节模式 | 每单位动作的目标角缩放（rad） | 额定力矩 (N·m) |
|---|---:|---:|
| hip pitch / hip yaw / waist yaw | 0.547546 | 88 |
| hip roll / knee | 0.350661 | 139 |
| ankle pitch / ankle roll | 0.438577 | 50 |
| waist roll / waist pitch | 0.438577 | 50 |

---

## 2. 终止条件

| 项 | 判据 | 说明 |
|---|---|---|
| `time_out` | 20 s | 正常结束，不算失败 |
| `fell_over` | 骨盆倾角 > 0.8 rad（46°） | 到这个角度已经救不回来 |
| `too_low` | 骨盆世界高度 < 0.45 m | 坐下/跪下 |

---

## 3. 奖励

只有 10 项。**目标函数只有 `power` 一项，其余 9 项全是「别摔、别趴下」的约束。**

| 项 | 权重 | 定义 | 角色 |
|---|---:|---|---|
| `power` | **-5.0** | $\sum_{j\in\text{下肢15轴}} (\tau_j/\tau^{max}_j)^2$ | **目标** |
| `com_centered` | +4.0 | $\exp[-(\lVert p^{com}_{xy}-p^{mid}_{xy}\rVert/0.08)^2]$ | 主力正向项 |
| `feet_grounded` | +1.0 | 双脚同时着地时为 1，否则 0 | 约束（正向） |
| `height_floor` | -40.0 | $\max(0,\ 0.68 - h)$，单位米，$h$ 为骨盆相对双脚高度 | 约束 |
| `com_margin` | -20.0 | $\max(0,\ \lVert p^{com}_{xy}-p^{mid}_{xy}\rVert - \tfrac{1}{2}d_{feet})$ | 约束 |
| `stillness` | -0.5 | $\lVert v_{root}\rVert^2 + \lVert \omega_{root}\rVert^2$ | 约束 |
| `joint_vel` | -2e-3 | $\lVert \dot q \rVert^2$（下肢 15 轴） | 平滑 |
| `joint_acc` | -2.5e-7 | $\lVert \ddot q \rVert^2$（下肢 15 轴） | 平滑 |
| `action_rate` | -0.1 | $\lVert a_t - a_{t-1} \rVert^2$ | 平滑，抗抖 |
| `terminated` | -200.0 | 摔倒时一次性 | 约束 |

### 3.1 `power` —— 为什么是这个形式

站着不动时机械功率 $\tau\omega \approx 0$，电耗几乎全是铜损 $I^2R$。而
$\tau = n k_t I$，所以

$$P_j = \frac{R_j}{(n_j k_t)^2}\,\tau_j^2$$

- **平方而不是 L1**：功率本来就正比于力矩平方，用绝对值反而不对。
- **按额定归一化**：同族执行器的 $R$ 与峰值电流接近，上式系数 $\propto 1/\tau_{max}^2$。
  不归一化的话 ankle（额定 50）出 1 N·m 和 knee（额定 139）出 1 N·m 记同样的账，
  而两者的发热完全不是一回事。
- **求和而不是均值**：总功率是各关节相加，取均值等于把每笔账除以 15。
- **手臂不计入**：手臂力矩主要由事件采到的位姿决定，策略控制不了，量级还比下肢大一个
  数量级，算进去只会把可控信号淹在外生方差里。**手臂仍然在观测里，只是不计费**——
  位姿要看得见（决定质心），电耗不归它管（上层 VR IK 的事）。

**权重定标。** `power` 的权重要让它在逐拍净收益里占一个不大不小的比例：占太多，站满
一局的收益压不过终止代价，回合长度会卡住不涨；占太少，退化解就封不死。

- 下限由退化解定：蹲下去顶膝限位那条解力矩很大，单拍代价必须显著高于全部逐拍净收益。
- 上限由「好姿态应当近乎免费」定：静止直腿时膝力矩趋零，代价应可忽略。

**权重必须按当前实际的下肢力矩量级重新定标，不能照抄。** 如果 `power` 读到的不是下肢
真实力矩（比如索引取错读成手臂），量级会差一个数量级，据此标的权重全部作废。

### 3.2 `height_floor` 为什么是下限而不是打靶

打靶（exp 对准某个高度）会直接制造蹲姿，而且是烧电机的直接原因：自然站姿本来就高于
常取的靶心，站直反而扣分，扣的量还远大于省力项的量级，于是最优解变成**蹲下去硬扛
力矩**——腿弯着、腰部电机过热。

改成单边下限之后，「站多高」交给 `power` 自己决定：直腿时重力线贴着膝轴过，膝力矩
趋零，**所以「站直」会作为省电的结果自己出现**，不需要另给高度奖励。这一项只负责
拦住「坐下去更省」这条退化解。

取 L1 而不是平方：平方在刚跌破下限处梯度为零，拦不住缓慢下沉。权重要和唯一那个正向项
同量级。

0.68 的取法：自然站姿约 0.78，留 10 cm 给抗扰动的屈膝缓冲；再低就不是「站」了。

### 3.3 已经删掉的项，以及为什么

| 删掉的项 | 原权重 | 删除理由 |
|---|---:|---|
| `height`（exp 打靶 0.72） | +2.0 | 见 3.2，姿态先验，直接导致蹲姿 |
| `upright`（`flat_orientation_l2`） | -2.0 | 姿态先验。骨盆正不正应该是省电的结果，不是输入；真摔了有 `fell_over` 兜底 |
| `com_centered`（exp 贴两脚中点） | +3.0 | 姿态先验。「重心在中间」是安全余量，不是省电目标；`com_margin` 已经在快出支撑面时收费了 |
| `joint_limits`（`joint_pos_limits`） | -5.0 | 变相的姿态约束：膝的软限位下界是 0.061 rad，留着它等于禁止把腿完全拉直 |
| `effort`（mean，下肢，权重 -60） | -60.0 | 被 `power` 取代：mean 改 sum、权重重新定标 |

**删 `joint_limits` 的代价要知道：** 策略会把关节顶到**硬**限位上让机械限位承重
（那确实是零力矩，也确实是字面意义上的最省电）。上机前看一眼实际关节角，如果发现膝
长期贴在行程端点，那不是训练出错，是它找到了最优解——但机械上不一定接受。

### 3.4 为什么正向项只留一个

`feet_grounded` (+1.0) 是唯一的正向项，留着它是为了让「活着」本身有收益：全是罚项
时，提前摔倒反而是止损。站立每拍的净收益是个很小的负数，而立刻摔倒是 -200 一次性 +
之后什么都拿不到，折算下来差三个数量级，退化解不成立。

---

## 4. 扰动与课程

**扰动是这个任务的核心难点，不是可选项。** 没有外力时「省电」退化成静态配平，
学出来的策略一推就倒。

整套事件直接复用行走任务的 `make_events()`，**一项不减**，只删 `gait_phase`
（本任务没有步态时钟）。自己另配一套很容易两头不讨好：要么幅值给得过猛直接站不住，
要么漏掉手臂力矩冲量、手臂位姿漂移、夹爪负载、左右独立的地面摩擦。

| 事件 | 模式 | 说明 |
|---|---|---|
| `disturbance_level` | reset | 每局采一个 $U(0,1)$ 的强度系数，**下面四个事件共用** |
| `reset_base` | reset | 开局整身倾角 ±0.35 rad + 初速度，幅值 × 课程 × 强度系数 |
| `reset_robot_joints` | reset | 关节角 ±0.15 rad，初速度 ±0.5 rad/s |
| `reset_arm_pose` | reset | 手臂 PD 目标在 `ARM_TARGET_RANGES` 内重采（**必须有**，否则手臂被拉到零位） |
| `arm_pose_drift` | interval 1–4 s | 局中让手臂真的摆起来 |
| `arm_torque` | step | 手臂关节力矩冲量，0.2–0.6 s，冷却 1–3 s |
| `body_impulse` | step | 夹爪 + 躯干外力冲量，0.1–0.4 s，冷却 1.5–4 s |
| `push_robot` | interval 5–6 s | 直接改根速度 |
| `payload_mass` | startup | 两个夹爪各加 0–2 kg |
| `foot_friction_left/right` | startup | 0.3–1.6，**左右独立**（合成一个的话接触完全对称，学不到主动保持航向） |
| `encoder_bias` | startup | ±0.015 rad |
| `base_com` | startup | 躯干质心 ±0.05 m |

### 安静段

`disturbance_level` 在 $(0,1)$ 上采样并被 `arm_torque` / `body_impulse` /
`arm_pose_drift` / `reset_*` **共用**，采到 0 附近就是一整局什么都不发生的 episode。
共用而不是各采各的，是因为各自独立时四者同时接近零的概率是乘积，几乎碰不到。

**策略必须见过「几分钟什么都没发生」这种工况**，否则高增益镇定器在真机上静止时会
自激抖动，而抖动就是持续电流。

### 课程

档位表与行走任务共用（`ARM_TORQUE_LEVELS` 等），**闸门换成「骨盆还立着」**——本任务
没有速度指令可跟随，父类那个跟随误差闸门会在 `command_manager` 上取空。

判据：`projected_gravity_b` 的水平分量模长就是 $\sin(\text{倾角})$，终止阈值 0.8 rad
对应 0.717，所以 `max_tilt=0.45`（约 27°）；至少 60% 的环境达标才放下一档。

**阈值不能收得太紧。** 本任务没有任何一项要求骨盆摆正（见 3.3），而 `reset_arm_pose`
会把手臂摆到全可达范围、质心随之偏移，策略就歪着站来代偿——稳态倾角本来就有相当
一部分超过 0.25。定得太严会让闸门永远开不了，整套扰动全程停在第 0 档（全关），
看着回合长度很高，其实是在零扰动平地上取得的。

**已知缺陷：这个判据有反向激励。** 它量的是全体环境的当前倾角，而复位瞬间骨盆是正的。
所以摔得越勤、刚复位的环境占比越高，分数反而越好看。放宽阈值只是绕开，根治要改成
按存活率度量。

| 课程项 | 驱动的事件参数 | 档位（迭代 → 幅值） |
|---|---|---|
| `arm_torque_level` | `arm_torque.torque_range` | 0→0, 500→1.5, 1500→2.5, 2600→3.2, 3700→4.0 N·m |
| `body_impulse_level` | `body_impulse.force_range` | 0→0, 600→8, 1800→12, 2900→17, 3950→20 N |
| `arm_drift_level` | `arm_pose_drift.blend` | 0→0, 550→0.3, 1650→0.6, 2800→1.0 |
| `reset_pose_level` | `reset_base.scale` | 0→0, 650→0.2, 2000→0.6, 2700→1.0 |
| `reset_joint_vel_level` | `reset_robot_joints.scale` | 同上 |

**没有课程 = 扰动全关**：第 0 档的手臂力矩、手臂漂移、外力冲量、开局幅值**都是 0**。
训练时盯 TensorBoard 里的 `Curriculum/arm_drift_level` 和 `arm_torque_level`，
如果全程卡在 0，说明闸门没开，那套手臂扰动等于白配。

`play=True` 时课程被清空，所以会手动把五项全部拨到末档——不拨的话评估用的是第 0 档，
会系统性高估抗扰能力。

---

## 5. 网络

MLP，actor/critic 都是 `(256, 128, 64)` + ELU，`GaussianDistribution` 标量 std
初值 0.5（站立不需要大幅探索），`entropy_coef=0.002`，`num_steps_per_env=24`。

观测里有 0–2 拍延时，严格说来不是马尔可夫的，循环结构理论上能把滞后补回来；但站立是
近似静态的镇定问题、观测已含全身关节角，行走任务上 GRU 的优势未必迁移。要做对照的话
照抄 `lower_body/rl_cfg.py` 里的 `ppo_runner_cfg_gru()` 注册一个 `G1-Gloria-Stand-GRU`。

---

## 6. 部署契约

导出的 ONNX metadata 里带两份权威名单，部署端（`Unitree_G1_Workspace` 的 `stand`
分支）照着装配观测，代码不写死布局：

| metadata 键 | 内容 |
|---|---|
| `observation_names` | `base_ang_vel,projected_gravity,joint_pos,joint_vel,actions` |
| `obs_joint_pos_joint_names` / `obs_joint_vel_joint_names` | 29 个观测关节，顺序 = 真机电机 0..28 |
| `action_joint_names` | 15 个动作关节 |
| `default_joint_pos` / `action_scale` | 动作偏置与缩放 |

模型解析出的 29 轴顺序恰好等于真机电机编号 0..28（腿 12 + 腰 3 + 左臂 7 + 右臂 7），
部署端不需要重映射。**不能靠截断 `joint_names` 去猜子集**——模型里
`left_eccentric_joint` 夹在左右臂中间。
