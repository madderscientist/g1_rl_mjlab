# 全身动作跟踪（GMT）

让 G1 + 双 Gloria-M 夹爪（29 轴动作 / 31 轴观测）跟随任意一段参考动作，目标是
**换一段没训过的动作也能跟住**，而不是把固定几段背下来。

方法参考 [GMT: General Motion Tracking for Humanoid Whole-Body Control](https://arxiv.org/abs/2506.14770)。
官方仓库只放了推理代码和一个 23 自由度的预训练模型，没有训练代码，所以这里是按论文
在 mjlab 上重新实现的。

## 相对 mjlab 自带 tracking 的两点改动

奖励、终止、事件全部沿用上游取值，只改了两处——而这两处正是泛化能力的来源：

**1. 多动作语料 + 自适应采样**（`mdp/commands.py`、`mdp/motion_corpus.py`）

上游的 `MotionCommand` 一次只跟一段动作。这里换成 `GeneralMotionCommand`：语料目录下
所有 NPZ 一起参与训练，复位时按每条动作的**最近失败率**（EMA）加权挑一条。均匀采样会
把预算浪费在早就学会的简单动作上，难的那些始终学不动。

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
`Episode_Termination/ee_body_pos` 约 30，其余终止项加起来不到 0.7，相差 40 倍以上。

## 已知限制：没有负载自适应

训练时的域随机化只有四项：整机速度扰动、**仅 torso_link** 的质心偏移、编码器零位、
脚底摩擦。**没有任何质量或惯量随机化，手臂上一项都没有。**

策略因此只见过一种手臂动力学，学到的是一套针对当前配置写死的重力补偿。手上一旦拿
东西，这套补偿就是错的，而观测里没有任何量能让它感知到负载变化。上机做抓取类任务前
必须先解决这一点。

## 语料

见根目录 README 的「训练数据」一节。当前语料是 40 段 LAFAN1 × (原速 + 镜像) = 80 条，
882360 帧 / 约 4.9 小时。

早期版本曾用程序化生成的站立/下蹲/摆臂凑语料，那些动作跟得准但看上去不自然，也教不会
策略真实人体运动的动力学，已全部删除。换成真实动捕后 walk4 的存活从 5.1 s 涨到 60 s。

## 仿真参数

`env_cfg.py` 里把 `njmax` 从上游默认的 250 提到 600、`nconmax` 提到 200。上游默认值是
按站立/行走估的，语料里 fallAndGetUp 这类躺地帧的接触约束远超该值，超出部分会被**静默
丢弃**（日志刷 `nefc overflow`），那些帧的物理是错的。同样的坑在 `build_corpus.py` 里
表现为 CUDA 非法内存访问。
