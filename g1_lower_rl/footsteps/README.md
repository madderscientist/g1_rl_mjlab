# 脚步生成与管理

仅依赖 Python 标准库和 NumPy，随机指令源、四步执行器和单步采样器可独立调用
参考 [Mind Your Steps](https://arxiv.org/abs/2606.08253) 的单步采样思路，改为平地四步预览和截断正态采样，不是原算法的等价复现

## 职责

| 对象 | 负责 |
| --- | --- |
| `RandomCommandSource` | f 游走、定期重采样两个方向、概率停止及实际站立后的起步决策 |
| `GaitRequest` | 世界系移动方向、脚掌朝向、正频率目标和 `walking` 布尔意图 |
| `FootstepManager` | 接收意图，维护四步/相位/参考系，执行限速、收步及起步 |
| `FootstepSampler` | 按前一个异侧计划脚印生成单个 XY/yaw 目标 |

训练和 Web UI 组合这些对象，库不反向依赖训练环境；cfg 只保存参数，不执行生成

## 算法

入口为 [sampler.py](sampler.py) 的 `sample(previous, side)`，输入前一个**异侧计划脚印**的世界 `[x,y,yaw]` 和待生成脚侧（0左、1右），输出下一脚印的世界位姿，单位 m/m/rad

1. 在配置的 `distance_range` 内采样截断正态距离 d
2. 移动方向加随机扰动，再减参考脚 yaw，得到局部方向角
3. 以前后候选位移配合居中的步宽：横向分量围绕步宽区间中心偏移，再按脚侧限制范围
4. 裁剪前后分量，使最终总距离不超过距离上限
5. 独立采样脚掌 yaw 并限制相邻转角，最后将局部位移旋转、平移回世界系

设参考脚 yaw 为 $\psi_p$、移动方向为 $\theta$、扰动为 $\alpha$，左脚 $s=1$、右脚 $s=-1$：

$$
x_0=d\cos(\theta+\alpha-\psi_p),\qquad
y=s\operatorname{clip}\bigl(w_c+d\sin(\theta+\alpha-\psi_p)s,w_{\min},w_{\max}\bigr),\qquad
w_c=(w_{\min}+w_{\max})/2
$$

$$
x_{\max}=\sqrt{\max(0,d_{\max}^2-y^2)},\qquad
x=\operatorname{clip}(x_0,-x_{\max},x_{\max})
$$

**d 是候选偏移，不是同一只脚或机器人的净移动距离**
例如无角度扰动、脚掌朝前的纯左移，候选 d=25cm 时，左脚宽度裁为36cm，右脚宽度裁为12cm，
两脚平均24cm，稳态一对左右步净横移24cm。前进或后退且无方向扰动时，宽度为区间中心24cm。
步宽不是独立均匀采样：方向扰动与候选距离共同决定偏离中心的幅度，边界处可能出现截断堆积。
侧移时展开脚偏宽、跟随脚偏窄；交替左右脚的整体分布均值约24cm，不要求每只脚各自的条件均值相同。

## 四步与时钟

[manager.py](manager.py) 按时间顺序生成 `L1 -> R1 -> L2 -> R2`，但输出固定按 `[L1,L2,R1,R2]` 分组
新目标从计划队尾续接，`apply_request()` 和便捷方法 `set_direction()` 都不改已经发布的四步

- 世界目标统一转换到共同参考系 A，不能把逐步生成时的相对位移直接作为四步输入
- 每个控制拍结束调用一次 `advance()`，用上一拍发布的 f 积分相位，理论落地时消费队头并补一目标
- f 是完整左右周期 Hz，总步频为2f；频率与几何距离范围独立
- 默认左右理论落地中心为90°、270°，双支撑窗口为72°–108°、252°–288°，均为左闭右开
- 请求停止后保留承诺四步并收步，同时减速；收步完成且达到双支撑确认条件后发布 f=0，站立冻结相位与目标
- manager 不会自己随机换向或决定起步；指令源收到实际 standing 状态后才开始保持计时

主要输出：`command.footsteps` 为 A 系 float32 `[4,3]`，`footsteps_w` 为世界目标，`future_ids` 为目标编号
`phase`、`frequency` 与四步一起供下一拍观测使用；`required_contact` 是计划接触，不是实测接触

## 默认参数

完整定义见 [config.py](config.py)，分为 `FootstepSamplerCfg`、`FootstepManagerCfg`、`RandomCommandCfg`，角度使用弧度

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `distance_range` | 0.05–0.48 m | 候选距离截断范围及最终总距离上限 |
| `distance_mean` / `distance_std` | 0.25 / 0.10 m | 截断前正态均值和标准差 |
| `min_width` / `max_width` | 0.12 / 0.36 m | 参考脚系中的横向间距范围，中心及整体均值约0.24m |
| `hold_width` | 0.24 m | 收步间距 |
| `direction_noise` / `yaw_noise` | ±40° / ±30° | 移动方向和脚掌朝向扰动 |
| `max_yaw_change` | 30° | 相对前一个异侧目标的转角上限 |
| `control_dt` | 0.02 s | 固定控制拍长 |
| `frequency_range` / `initial_frequency` | 0.8–1.8 / 1.667 Hz | source 的采样范围/初值；manager 同名字段为执行范围/缺省值 |
| `frequency_rate_range` / `frequency_rate_interval_s` | ±0.3 Hz/s / 2 s | 随机变化率及保持时间，触边反向 |
| `frequency_slew_rate` | 0.2 Hz/s | manager 对所有频率目标的执行限速 |
| `command_interval_s` / `stop_probability` | 3–8 s / 30% | 每次行走指令重采样时抽一次停止概率 |
| `hold_time_s` | 2–5 s | 自动再起步前的站立保持时间 |
| `direction_range` / `foot_heading_range` | ±180° / 0° | source 的整体方向范围，不是逐脚扰动 |

`hold_width` 属于 manager；整体方向范围、随机变化率、重采样间隔和自动开关属于 source
停步概率从10%提高至30%，在成功收步、站立保持2–5s后自动再起步，不额外抽起步概率。
这会增加停走机会，但不保证站立时间占比或成功次数；短回合、失败重置仍可能打断过渡。
24cm 是默认采样中心及收步间距，不是硬下限；MJCF 全关节为0时左右足底 site 间距为0.23701291m，
不是左右髋关节原点的12.89cm距离。前进、后退、左右横移各两万步验证，均值为23.98–24.02cm。
总步距上限为48cm，横向上限仍为36cm；24cm步宽时前后分量最多约41.6cm。
不修改实际机器人初始关节姿态，候选距离的截断前均值25cm和标准差10cm也不变。
随机源用实际 f 加 `rate*dt` 产生下一拍目标，不另做平滑；实际变化仍受 manager 的执行限速约束
默认随机率可达±0.3Hz/s，执行限速为0.2Hz/s，更快变化会被截住；需要完整跟随时显式提高执行限速

## 调用 demo

在仓库根目录执行以下代码，组合 f 游走、方向变化和自动停走，不运行物理仿真或控制电机

```python
import math
from g1_lower_rl.footsteps import FootstepManager, FootstepManagerCfg, RandomCommandSource, RandomCommandCfg

source = RandomCommandSource(
  RandomCommandCfg(
    foot_heading_range=(-math.pi, math.pi),  # 默认范围为零，此处启用整体朝向变化
    automatic_commands=True,
    automatic_restart=True,
  ),
  seed=7,
)
cfg = FootstepManagerCfg(require_contact_confirmation=False)  # 仅离线 demo 关闭接触确认
manager = FootstepManager(cfg, seed=7)
command = manager.reset([[0, 0.12, 0], [0, -0.12, 0]], source.reset(heading_origin=0))

for frame in range(3000):  # 60秒模拟时间，概率停止不保证每次短演示都发生
  before = manager.command()
  update = manager.advance()  # 真实闭环在执行一拍物理控制后调用
  completed = update.completed  # 本拍实际频率和切换前目标，由训练适配器转换为奖励输入
  request = source.advance(cfg.control_dt, mode=update.command.mode, frequency=update.command.frequency)
  manager.apply_request(request)  # 只设置下一拍意图，不推进相位
  command = manager.command()
  if update.landed_sides or before.mode != command.mode:
    print(round((frame + 1) * cfg.control_dt, 2), command.mode, round(command.frequency, 3))
    print(command.footsteps.round(3).tolist())  # A 系 L1/L2/R1/R2 的 XY/yaw

  assert completed.frequency == before.frequency
```

真实闭环顺序：**读取 command → 执行一拍控制 → `advance(feet_w=实测XY/yaw, contacts=实测左右接触)`**
奖励适配器使用 `update.completed` 的步末、切换前快照，并另提供地面高度
随后调用 source 和 `apply_request`，再读取下一拍 `command`；旧快照不会自动更新
真实闭环保持 `require_contact_confirmation=True`，接触等待超时由外部安全处理；生成器不持有策略或 GRU

手动模式跳过 source，直接 `manager.apply_request(GaitRequest(...))`；`walking=False` 请求停止，正频率保留为起步目标
`set_frequency()` 不再接受 None，reset 改为接收 `GaitRequest`；切换随机模式由调用方选择指令源

## 检查

```bash
python -m unittest discover -s tests -p test_footstep_generation.py
```

## 使用边界

- 这是单实例 CPU 规划器，未接入批量训练，不计算 IK、碰撞、连续摆脚轨迹或动力学可行性
- 计划落地不代表真实接触，采样器不检测足底重叠
- 实测脚位只用于在双支撑时重建 A，不移动已经承诺的世界目标；无反馈时 A 保持初始值
- 相同种子及调用序列可重现；检查点需同时保存 manager 和 source 的 `state_dict()`，不兼容旧版混合状态，也不保证与旧版随机序列逐位一致