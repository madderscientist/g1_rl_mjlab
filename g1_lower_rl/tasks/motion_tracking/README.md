# 全身动作跟踪（GMT）

让 G1 + 双 Gloria-M 夹爪（29 轴动作 / 31 轴观测）跟随任意一段参考动作，目标是
**换一段没训过的动作也能跟住**，而不是把固定几段背下来。

方法参考 [GMT: General Motion Tracking for Humanoid Whole-Body Control](https://arxiv.org/abs/2506.14770)。
官方仓库只放了推理代码和一个 23 自由度的预训练模型，没有训练代码，所以这里是按论文
在 mjlab 上重新实现的。

## 相对 mjlab 自带 tracking 的两点改动

奖励、终止、事件全部沿用上游取值，只改了两处——而这两处正是泛化能力的来源：

**1. 多动作语料 + bin 级自适应采样**（`mdp/commands.py`、`mdp/motion_corpus.py`）

上游的 `MotionCommand` 一次只跟一段动作。这里换成 `GeneralMotionCommand`：语料目录下
所有 NPZ 一起参与训练，并且整份语料被切成**固定时长的 bin**（默认 1 秒 / 50 帧，与
SONIC 对齐），复位时按每个 bin 的失败率加权挑一个、再在 bin 内随机取起点。

按整条动作加权（GMT 原版）不够用：难点往往只占一条长动作里的几秒，把整条上调之后起点
仍然撒满全篇，额外算力大部分花在早就练熟的段落上。bin 级把加权粒度降到和难点同量级。

三个参数决定行为：

| 参数 | 默认 | 作用 |
| --- | --- | --- |
| `bin_frames` | 50 | bin 长度。bin 总数要和并行环境数同量级——太粗则难点被稀释，太细则单个 bin 攒不够样本估失败率 |
| `adaptive_failure_cap` | 0.5 | 失败率上限，也是初值 |
| `adaptive_uniform_ratio` | 0.1 | 均匀分布占的比重，保底覆盖 |

**上限不能去掉。** 本机手臂在 25 N·m 下做不到的片段（冲刺、跳跃）失败率恒为 1，不封顶
的话它们会把采样预算吸光，而再练也不会变好。上限同时用作 `bin_failed` 的初值（乐观
初始化），保证冷启动时先把全语料扫一遍，而不是卡在最早出现失败的那几个 bin 上。

失败率是 **P(失败 | 从该 bin 起步)**，按 bin 的**访问次数**做 EMA（不是按仿真步——
一个 bin 平均几百步才被抽中一次，按步衰减的话绝大多数 bin 会一直停在初值）。采样单位
和统计单位是同一个，所以加权依据和实际观测严格对得上。

诊断看 `Metrics/motion/sampling_entropy`（1 = 摊平，低 = 失败集中在少数 bin）和
`Metrics/motion/sampling_failure_rate`（接近 `cap` 说明上限在起作用，该考虑是不是语料
里塞了太多物理不可达的片段）。

**2. 偏航/平移不变的前瞻观测**

策略看到的不是参考轨迹在世界系里的绝对位姿，而是未来若干帧在**根坐标系**下的：离地
高度、投影重力、线/角速度、关节角。绝对位置和朝向被彻底剔除，于是同一段动作平移或
转向之后对策略完全相同——这是泛化到新动作的前提。用投影重力而非 roll/pitch 角是为了
避开角度回绕。

actor 的本体观测还接了 5 帧历史：参考轨迹只说"该往哪走"，而接触与打滑这些信息只存在
于最近几帧的本体量里；没有历史，策略在动作切换处会反复踩空。

## 用法

```bash
# 训练（镜像增强必须关闭，见下）
python scripts/train.py G1-Gloria-MotionTracking --env.scene.num-envs 4096 \
    --mirror-schedule '()'

# 逐条动作定量评测：能跟多久、跟到哪一步失败、误差多大
python scripts/eval_corpus.py --checkpoint <run>/model_59200.pt --envs-per-motion 8

# 渲染策略跟踪效果（实体 = 策略，半透明 ghost = 参考动作）
python scripts/render_policy.py --checkpoint <run>/model_59200.pt \
    --motions walk1_subject1,jumps1_subject1 --output logs/render/policy.mp4

# 网页实时预览，同样带 ghost 对照
python scripts/play.py G1-Gloria-MotionTracking --viewer viser

# 只回放参考动作本身（运动学，不跑物理），用来检查动作数据对不对
python scripts/render_motion.py --motion motions/lafan1/walk1_subject1.npz

# 导出部署用 ONNX + 可读契约
python scripts/export_onnx.py --checkpoint <run>/model_59200.pt --output-dir export

# 给部署包瘦身用的动作裁剪
python scripts/slim_motion.py
```

> **`--mirror-schedule '()'` 不能省。** 镜像增强是按下肢 15 轴的对称关系写的，套到
> 29 轴上会静默地把手臂镜错。语料本身已经在 `build_corpus.py` 里做过镜像了。

## 已知限制：手臂跟不动

这是当前最主要的性能瓶颈，且**根因是硬件而非训练**。

Gloria-M 夹爪把手臂惯量抬高了一大截（按 400 个随机臂姿的中位数：肩 pitch ×2.16、
肩 roll ×2.06、肩 yaw ×2.80、肘 ×2.68、腕各轴 ×3.4~3.6），而手臂电机仍是原厂 5020
（25 N·m）。两个后果：

**静态下垂**。mjlab 的 kp 是按电机**转子反射惯量**定的（`STIFFNESS = ARMATURE * ω²`，
ω 取 10 Hz），完全没算连杆惯量。肩 pitch 的实际惯量是转子的 83 倍，所以实际带宽只有
1.10 Hz（原厂 G1 也只有 1.64 Hz），阻尼比 0.22（设计意图是 2.0）。手臂前平举时静态
下垂 32°，腕部位置误差 0.30 m，而判"跟丢"的阈值是 0.25 m（只比高度）。

**力矩天花板**。幅值 1 rad 的正弦摆臂，肩 pitch 在 1.5 Hz 就需要 26.6 N·m，超过 25 N·m
上限。这条线 kp 救不了。

实测（60 秒窗口，80 条语料逐条评测）正好对上：

| 类别 | 平均存活 | |
| --- | --- | --- |
| walk1 / walk4 | 60.0 s（撑满） | 解决 |
| walk2 / walk3 | 50~54 s | 基本解决 |
| dance2 | 36 s | 一半 |
| sprint1 / run2 / jumps1 / fight1 / fallAndGetUp | 6~10 s | 崩 |

存活期间的体位误差是 0.027~0.056 m，也就是说**不是跟不准，是撑不久**。

> 上面那句“`Episode_Termination/ee_body_pos` 约 30，其余终止项不到 0.7”是**移除手腕终止之前**的旧数据。
> 2026-08-14 在 model_59200 上重测：`anchor_pos` **71.7%**、`ee_body_pos` 28.3%、`anchor_ori` **0%**。

## 已排除：“终止太严”和“腿部力矩不够”都不是原因

两个自然猜想都被实测推翻（model_59200，逐条语料确定性评估），别再走这两条路：

**不是终止判据太严。** 把 `anchor_pos` + `ee_body_pos` 阈值从 0.25 放宽到 1.0 m，
存活只从 20.9 s 涨到 26.7 s（+28%），且**全部 361 次终止改判为 `anchor_ori`（真倾倒）**。
机器人是真在摔，`anchor_pos` 只是先一步探测到“腿塌陷下沉”。

**腿也不是硬件受限**（和手臂相反）。摔前 1 s 腿部力矩峰值饱和度均值仅 **0.69**，
＞95% 上限的时间占比只有 1.3%。决定性反证：**死得最快的类别力矩余量反而最大**
（sprint1 0.56、fightAndSports1 0.47，而活得久的 run1 是 0.90）。

真实画像：存活时躯干高度误差只有 0.005~0.07 m，一出事就急剧崩溃——**双峰，没有渐进退化**，
这是“掉出训练流形”的特征，指向数据覆盖不足而非能力上限。

## 已知限制：手臂的负载自适应仍弱

当前的域随机化（全部在 `env_cfg.py`）：

| 项 | 范围 | 来源 |
| --- | --- | --- |
| 整机速度扰动 `push_robot` | 上游默认 | mjlab |
| torso_link 质心偏移 `base_com` | ±2.5/5/5 cm | mjlab |
| 编码器零位 `encoder_bias` | ±0.01 rad | mjlab |
| 脚底摩擦 `foot_friction` | 0.3~1.2 | mjlab |
| 夹爪负载 `payload` | −0.5~1.0 kg，左右独立 | 本仓 |
| 手臂重力补偿增益 | 0.9~1.01 | 本仓 |
| PD 增益 `pd_gains` | ±20%（缩放） | 本仓 |
| 连杆质量/惯量 `link_inertia` | α ±0.05 ≈ 质量 ±10% | 本仓 |
| 关节干摩擦 `joint_friction` | 0~0.2 N·m | 本仓 |

负载随机化已经有了，但观测里仍没有任何量直接告诉策略“现在拿了多重的东西”，只能从
「指令 vs 实际响应」的历史里隐式辨识（这也是 actor 用 GRU 的原因）。上机做抓取类任务前
应该验证这一点够不够。

**还缺一项：对参考指令本身的扰动。** SONIC 明确对 $s_t^g$ 加噪，并把它归为“对 planner
输出鲁棒”的主因。接运动学规划器或遥操流之前应补上。

## 语料

见根目录 README 的「训练数据」一节。当前语料两部分，均含镜像：

| 来源 | 段数 | 时长 |
| --- | --- | --- |
| LAFAN1（Unitree 重定向） | 40 × 2 = 80 | 4.9 h |
| AMASS（`ember-lab-berkeley/AMASS_Retargeted_for_G1`） | 1332 × 2 = 2664 | 12.0 h |

扩充用 `scripts/import_amass.py`，它带**物理可行性筛选**（太短/脚够不到地/骸盆高度异常/
关节速度尖峰）——首批 6 h 预算里筛掉了 271 段，其中 219 段是重定向毛刺造成的速度尖峰。
SONIC 同样把 700 h 筛到 611 h 才训；不筛的话这些片段会永远占着采样预算却学不会。

AMASS 那份的 `dof_names` 顺序与 `build_corpus.JOINT_NAMES` 完全一致、fps 同为 30，
所以转换只是取根位姿 + 关节角、把四元数从 **wxyz** 换成 CSV 用的 xyzw。（wxyz 是实测定的：
站立片段里按 wxyz 解释时根的局部 +z 指向世界 [0,0,0.999]，按 xyzw 则是侧躺。）

早期版本曾用程序化生成的站立/下蹲/摆臂凑语料，那些动作跟得准但看上去不自然，也教不会
策略真实人体运动的动力学，已全部删除。换成真实动捕后 walk4 的存活从 5.1 s 涨到 60 s。

## 仿真参数

`env_cfg.py` 里把 `njmax` 从上游默认的 250 提到 600、`nconmax` 提到 200。上游默认值是
按站立/行走估的，语料里 fallAndGetUp 这类躺地帧的接触约束远超该值，超出部分会被**静默
丢弃**（日志刷 `nefc overflow`），那些帧的物理是错的。同样的坑在 `build_corpus.py` 里
表现为 CUDA 非法内存访问。
