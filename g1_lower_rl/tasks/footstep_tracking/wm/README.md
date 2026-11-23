# 脚印跟踪世界模型训练说明（`G1-Gloria-FootstepTracking-WM`）

本目录用可微世界模型（[Position WM v3](../../../world_model/README.md)）训练脚印跟踪策略。策略网络结构与部署接口见主文档 **[策略接口与网络结构](../README.md#策略接口与网络结构)**；原版 PPO 代码（[footstep_model.py](../../../rl/footstep_model.py)、[rewards_cfg.py](../rewards_cfg.py)）保持未动。

---

## 1. 基本原理：和 PPO 到底差在哪？

两者的核心区别只有一件事——**怎么算策略参数的梯度**：

- **PPO**：把仿真器当黑盒，给动作加随机噪声去试，整条轨迹跑完看谁总分高就往哪边改。不要求奖励可导，但靠随机试探找方向，落脚误差到 `6~7 cm` 后就很难继续压低。
- **世界模型训练**：
  1. 用神经网络拟合单步物理规律：`当前状态 + 29 轴目标位置 -> 20ms 后的下一状态 + 双脚接触力`。双卡同时在真实 MuJoCo 里边跑边采数（GPU 0 跑连续走 `precision`，GPU 1 跑单步启停 `step-episode`），持续更新这个世界模型。
  2. 从回放池取真实状态作为起点，让策略和世界模型闭环往后推演 **`48` 步（`0.96 s`）**，算出每一步的可微奖励，第 48 步末尾接价值网络 $V(s_{48})$ 估计后续长期回报。
  3. 直接对这 `0.96 s` 的总回报求导（`backward()`），算出“当前动作往哪边改一点，未来落脚更准、更稳”，更新共享策略网络。

---

## 2. 直接求导会带来什么问题？（奖励与任务改动）

一旦从“随机试错”换成“顺着导数改动作”，微积分链式法则决定了三件事：**导数为 0 的地方学不动**、**有捷径的地方一定走捷径**、**推演规则和真实环境不一致就会学偏**。[task.py](./task.py) 相对原版 PPO 的改动全部由此而来：

### (1) 阶跃开关没导数，宽高斯在中心没斜率
- **原因**：
  - 原版 PPO 里的“是否触地 (`fz >= 10N`)”、“是否靠近硬限位 (`< 0.1°`)”、“是否摔倒”都是布尔 `0/1` 开关，对状态求导处处为 `0`，梯度传不过去。
  - 原版摆动引导用的是宽高斯 $\exp(-(d / 0.20)^2)$，距离 $d$ 进了 `4 cm` 以内曲线几乎是平的（导数趋近 `0`），拉不动毫米级精度。
  - 但如果直接加上陡峭的距离惩罚，机器人被外力推歪快摔倒时，导数依然死拉着脚往原脚印踩，不让迈大步撑地救命。
- **改法**：
  - 把接触、关节限位、摔倒全部写成连续平滑的 Sigmoid 函数。
  - 在宽高斯基础上增加窄高斯（`4.5 cm`）和中心斜率恒定的 `Smooth-L1` 距离项，把落脚拉进 `1~3 cm`。
  - 给脚印距离惩罚乘一个不传梯度的直立权重（上身倾角 `<14°` 且角速度正常时约为 `1`，被推歪失稳时平滑降到 `0`）：平稳走路时修精度，快摔倒时自动松开脚印约束先保命。
  - 摆动后期不再乘离地高度开关，防止脚底降到 `<2.5 cm` 快触地时奖励突然变小、诱导策略“半空悬停再跺脚”。

### (2) 头高满分线设得太高，直接求导会把膝盖拉直
- **原因**：原版头高奖励满分线设在 `1.254 m`（双腿完全伸直才够得到的高度），同时铜损项惩罚关节力矩平方。直接求导时，策略立刻发现“把膝盖锁直既省力矩又能多拿头高分”，于是走出双腿绷直、靠歪骨盆迈步的圆规步；站立时还会因为硬限位前 `0.1°` 阶跃惩罚没导数，把膝盖顶死在 `-5°` 反弓限位上。
- **改法**：
  - **走路时把头高满分线降到 `1.185 m`**：正常屈膝（`37°~57°`）走路就已经拿满头高分，不再产生把腿往上拽直的梯度；停步站立时再恢复 `1.254 m`。
  - **加单边下界与站立微屈**：走路时只有膝角过直（摆动中期 `<53°` 或支撑期 `<26°`）才扣分，正常弯腿走路扣分为 `0`；站立时让膝角停在 `+4.7°` 自然微屈，并在硬限位前 `1.5°` 平滑放大铜损回推力矩。

### (3) 相对角求导会双向拉扯，导致站定后脚底原地打转
- **原因**：三个因素叠加让踩在地上的脚转圈：① 上身偏角误差是 `(上身朝向 - 双脚朝向) - 目标角`，直接求导不仅会转腰，还会反向去拧地上的脚来凑这个差值；② 单步到位奖若在落地 `0.25 s` 后才发，等于鼓励踩实后继续磨脚底对准角度；③ 脚绕脚心原地自转时中心点线速度为 `0`，只看线速度的防滑项完全不管。
- **改法**：
  - 算上身偏角误差时对双脚朝向 `foot_yaw` 做 `.detach()` 截断梯度，让这项梯度只能驱动腰和髋转上身，不许动脚；
  - 单步到位奖只在**刚触地那一拍**结算，并且落脚偏航惩罚只在空中摆动期生效；
  - 在防滑惩罚里加上 `0.10 m × 脚自转角速度`（对应脚尖/脚跟擦地速度），站立时把足底转角速度惩罚放大 `3` 倍。

### (4) 推演步长与队列规则和真实环境对齐
- **推演步长取 `48` 步（`0.96 s`）**：最低步频 `0.8 Hz` 迈一步需要约 `0.6 s`，推演 `0.96 s` 才能完整包含“摆腿 $\to$ 落地 $\to$ 支撑”全过程（CUDA Graph 逐步复用显存，峰值显存不增加）。
- **参考系与收步目标对齐**：脚印局部参考系在双支撑期按足底法向接触力确认踩实后再完成切换，避免脚还在半空就提前切换参考系；进入减速停步（`STOPPING`）或单步模式时，队列末尾填入停止脚位 `terminal_feet`，不再继续往前外推新步。
- **连续走与单步分开设质心余量**：连续行走限制质心捕获点偏离不超过 `5.5 cm` 防跑偏；单步任务放宽到 `11.5 cm`，允许先把重心移到支撑脚正上方再稳稳抬脚。

---

## 3. 怎么加外力抗推、怎么防异常梯度冲坏模型？

1. **外力只在控制步交界处瞬间改速度，并切分轨迹段**：世界模型的输入只有机器人状态和关节目标，没有“外力”。如果在 `20 ms` 步内持续推机器人，世界模型会把外力当成机器人自己乱动而学坏。因此在采集端每次 `20 ms` 步结束后，以低概率对训练环境瞬间加一次躯干速度冲击（`--collector-vel-push-*`），同时采集器自动在该处把轨迹切成新的一段（`_segment += 1`）。这样世界模型不会跨冲击点算预测误差，而策略和价值网络又能从被推歪的真实状态里学会跨步恢复。
2. **用 Huber 损失和偏离截断防梯度爆炸**：极少数摔倒或碰撞穿模帧会产生巨大的单步误差，用平方损失求导并在 8 步自回归中连乘后，梯度会暴涨上万倍冲坏 Adam 动量。因此世界模型训练在正常误差范围（`≤8σ`）内保持平方损失，超出部分转为线性斜率（Huber），并在自回归预测已严重跑偏时停止往后续步传梯度，配合全局梯度上限跳过极端批次，保证长时间训练平稳。

---

## 4. 文件职责与实测结果

| 文件 | 作用 |
| --- | --- |
| [policy.py](./policy.py) | 观测与指令编码、`MLP / GRU / Skip-GRU` 策略网络实现 |
| [task.py](./task.py) | 环境配置、8 阶段自动课程表、回放池、可微队列推进与奖励计算 |
| [collection.py](./collection.py) | 50 Hz MuJoCo 采样、控制步交界速度冲击、验证集存活统计 |
| [runner.py](./runner.py) | 在线世界模型训练、`H=48` 轨迹求导更新策略、双卡独立 Critic 训练与存档 |
| [playback.py](./playback.py) | ONNX 导出与数值校验、Viser 可视化回放 |

标准续训与部署包位于 `export/footstep_wm_skip_gru256_52400/`（含双卡续训权重 `training/update_00052400/rank_{0,1}.pt` 与直接供 `FootstepPolicy` 加载的 `policy_deploy.onnx`）：

| 同种子 64 环境闭环测试 (`seed=123`) | WM `Skip-GRU-256` (`0.62M`, `Iter 52.4k`) | WM 10帧 `MLP` (`1.54M`, `71k`) | 原版 PPO `GRU` (`0.15M`, `189k`) |
| --- | ---: | ---: | ---: |
| 连续全向走存活 / 落脚 XY / Yaw / $\theta$ | **`64/64` / `3.78 cm` (`≥1.35Hz` `3.22 cm`) / `1.59°` / `2.89°`** | `51/64` / `2.86 cm` / `1.33°` / — | `64/64` / `6.59 cm` / `2.70°` / `3.42°` |
| 触地冲击力 `P50 / P90` | **`217.2 N` / `724.7 N`** | `475 N` / `784 N` | `214 N` / `576 N` |
| 行走屈膝（支撑 / 摆动）· 站立膝角 | **`+36.9° / +57.0°` · 站立 `+4.69°`（不反弓）** | `+41.2° / +72.2°` · `+14.5°` | `+56.3° / +75.0°` · `-5.0°`（顶限位） |
| **连续 3 次 `0.55 m/s` 强推存活 / 摔倒数** | **`96.9% (62/64)` / `0 摔倒`** | — | `76.6% (49/64)` / `0 摔倒` |
| 单步落脚 XY (`≤3cm` 率) / Yaw / 站定脚扭转 | **`1.39 cm (90.6%)` / `5.05°` / `1.29°`** | `1.63 cm` / — / — | `6.75 cm` / `27.40°` / `7.16°` |

---

## 5. 常用命令

```bash
# 1. 从标准包 export/footstep_wm_skip_gru256_52400 继续双卡联合训练
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 MUJOCO_GL=egl \
torchrun --standalone --nproc_per_node=2 scripts/train_footstep_wm.py \
  --resume export/footstep_wm_skip_gru256_52400 \
  --output logs/footstep_wm/continue_from_52400 \
  --actor-type skip_gru --gru-hidden-dim 256 --actor-hidden-dims 512 256 128 \
  --auto-curriculum --joint-step-episode --joint-task-weights 0.65 0.35 \
  --initial-stage 7 --final-polish --horizon 48 --wm-batches 5 \
  --learning-rate 1.0e-5 --wm-learning-rate 2.5e-5 --action-noise 0.025 --policy-target-kl 0.003 \
  --collector-vel-push-prob 0.002 --collector-vel-push-lin 0.45 --collector-vel-push-ang 0.45 \
  --iterations 0

# 2. 从头启动双卡 8 阶段自动课程训练
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 MUJOCO_GL=egl \
torchrun --standalone --nproc_per_node=2 scripts/train_footstep_wm.py \
  --output logs/footstep_wm/skip_gru256_scratch \
  --actor-type skip_gru --gru-hidden-dim 256 --actor-hidden-dims 512 256 128 \
  --gru-burnin-steps 24 --horizon 48 --iterations 0

# 3. 启动 Viser 可视化（8085 连续走 / 8086 单步）
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MUJOCO_GL=egl \
python scripts/play.py G1-Gloria-FootstepTracking-WM \
  --checkpoint-file export/footstep_wm_skip_gru256_52400 --profile precision --viewer viser --viser-port 8085
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl \
python scripts/play.py G1-Gloria-FootstepTracking-WM \
  --checkpoint-file export/footstep_wm_skip_gru256_52400 --profile step-episode --viewer viser --viser-port 8086

# 4. 导出 ONNX（直接供 FootstepPolicy 加载 policy_deploy.onnx）
python scripts/export_onnx.py \
  --checkpoint export/footstep_wm_skip_gru256_52400 \
  --output-dir export/footstep_wm_skip_gru256_52400
```
