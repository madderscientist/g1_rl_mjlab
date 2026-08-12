# g1_lower_rl

Unitree G1 + 双 Gloria-M 夹爪的**下肢行走**强化学习任务，基于 [mjlab](https://github.com/mujocolab/mjlab) 1.5.x。

从 `unitree_rl_mjlab`（mjlab 1.2.0）迁移而来，只保留下肢任务，并重整了项目结构。

## 环境

```bash
micromamba activate mj
```

`mj` 环境已装好 mjlab 1.5.3（mujoco 3.10 / mujoco-warp 3.10 / warp-lang 1.16 / rsl-rl-lib 5.4）
和本包（`pip install -e .`）。

环境里加了一个 activate 钩子把 `$CONDA_PREFIX/lib` 挂到 `LD_LIBRARY_PATH`——
conda-forge 的 `libicui18n` 需要比系统更新的 `libstdc++`，不加的话
`import mjlab` 会在 `sqlite3` 处以 `CXXABI_1.3.15 not found` 失败。
**所以必须先 `micromamba activate mj`，不要直接调 `envs/mj/bin/python`。**

## 用法

```bash
# 列出任务
python scripts/list_envs.py

# 训练（默认开左右镜像数据增强 DUP，档位见 cfg/constants.py 的 MIRROR_STAGES）
python scripts/train.py G1-Gloria-LowerBody-Flat --env.scene.num-envs 4096

# 续训（课程档位与闸门分数会一起恢复）
python scripts/train.py G1-Gloria-LowerBody-Flat --agent.resume True \
    --agent.load-run 2026-08-07_14-00-32 --agent.load-checkpoint model_2000.pt

# 回放（不给 --checkpoint-file 就取最新的）
python scripts/play.py G1-Gloria-LowerBody-Flat

# 无头定量评估
python scripts/eval_policy.py logs/rsl_rl/g1_gloria_lower_body/<run>/model_8800.pt

# 导出 ONNX（训练时每次存档也会自动导出一份带元数据的）
python scripts/export_onnx.py logs/rsl_rl/g1_gloria_lower_body/<run>/model_8800.pt

# 上机前检查：CPU 上闭环重跑部署链路
python scripts/check_deploy_policy.py <policy>.onnx

# 多个策略的纵向对比视频（统一机位，两遍式渲染）
python scripts/compare_video.py A.onnx B.onnx out.mp4

# 满档扰动下的训练环境实况，6 宫格每格跟拍一个环境
python scripts/disturb_video.py <run>/model_8800.pt out.mp4 20
```

动作跟踪任务另有一组脚本，见 [`tasks/motion_tracking/README.md`](g1_lower_rl/tasks/motion_tracking/README.md)。

## 训练数据（motions/，不入库）

`motions/` 整个目录在 `.gitignore` 里，需要自己下载重建。

| 目录 | 内容 | 体积 |
| --- | --- | --- |
| `motions/csv/` | 40 段 LAFAN1 重定向到 G1 29 轴的 CSV，36 列 | 87 MB |
| `motions/lafan1/` | 由上面转出的训练语料，80 条 NPZ | 2.1 GB |

**来源**：`motions/csv/` 取自 HuggingFace 上的 LAFAN1 重定向数据集
[`lvhaidong/LAFAN1_Retargeting_Dataset`](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)。
每行是 `[root_pos(3), root_quat_xyzw(4), dof(29)]`，原生 30 fps，可直接喂
`scripts/build_corpus.py`。

> Unitree 官方的同名仓库 `unitreerobotics/LAFAN1_Retargeting_Dataset` 目前返回 401，
> 上面这个是可用的公开镜像。动作本身源自 Ubisoft 的 LAFAN1 动捕数据集，商用前请自行
> 确认授权。

**重建语料**：

```bash
python scripts/build_corpus.py --input-dir motions/csv --output-dir motions/lafan1
```

40 段 CSV × (原速 + 左右镜像) = 80 条，共 882360 帧 / 17647 秒（约 4.9 小时）。
脚本支持断点续传（输出已存在就跳过）。`--input-fps 20` 可把动作整体放慢 1.5 倍
（CSV 原生 30 fps），早期用它验证过「放慢能显著改善跟踪」。

## 结构

```
g1_lower_rl/
├── assets/g1_gloria/          机器人：MJCF + 执行器/关节常量
│                              （执行器与碰撞定义直接复用 mjlab 自带的 G1 资产）
├── tasks/
│   ├── __init__.py            注册任务
│   └── lower_body/
│       ├── mdp/               MDP 项的**实现**
│       │   ├── rewards/       ← 奖励按主题拆开
│       │   │   ├── tracking.py    速度 / 转向 / 原地位移 / 高度
│       │   │   ├── gait.py        步态相位 / 腾空 / 抬脚 / 打滑
│       │   │   └── posture.py     躯干姿态 / 骨盆前后倾 / 关节偏差
│       │   ├── commands.py    ScenarioVelocityCommand、BaseHeightCommand
│       │   ├── curriculums.py 带闸门与状态持久化的课程
│       │   ├── events.py      开局随机化 + 上肢扰动
│       │   ├── observations.py
│       │   └── terminations.py
│       ├── cfg/               环境**配置**，按 manager 拆开
│       │   ├── constants.py   关节集合 / 课程档位表 / 闸门 / 场景占比
│       │   ├── observations.py  actions.py  events.py
│       │   ├── rewards.py     terminations.py  curriculum.py
│       │   └── env_cfg.py     组装（含机器人特化）
│       └── rl_cfg.py          PPO 配置
│   ├── motion_tracking/       全身 29 轴动作跟踪（GMT），见该目录 README
│   │   ├── mdp/commands.py    GeneralMotionCommand：多动作 + 自适应采样 + 前瞻观测
│   │   ├── mdp/motion_corpus.py  多片段语料加载
│   │   ├── env_cfg.py         rl_cfg.py
│   └── standing/              最小力矩站立（无 command，重心保持在双脚之间）
│       ├── mdp.py             env_cfg.py  rl_cfg.py
├── rl/
│   ├── runner.py              课程状态持久化 + 每次存档导出带部署元数据的 ONNX
│   └── mirror.py              左右镜像数据增强（DUP）
└── deploy.py                  CPU 闭环部署链路仿真骨架
```

分层原则：`mdp/` 放**怎么算**，`cfg/` 放**用什么参数算**，`robots/` 放**这台机器人叫什么名字**。
档位表（`cfg/constants.py`）被三处共享：事件取第一档作初值、课程按表推进、
`cfg.max_out_curriculum()` 一次拨到末档（`play` 配置和 `disturb_video.py` 都用它）。

## 相对 unitree_rl_mjlab 的变化

**保留不变**（逐字段比对过，除 mjlab 新增字段的默认值外完全一致）：
观测顺序（actor 57 维，部署契约）、全部奖励项与权重、事件、课程档位与闸门、
PPO 超参、镜像增强、课程状态持久化。

> 验证方式：在两个环境里各自导出归一化的 env cfg 快照并 diff，目前只差
> `rel_world_envs` / `rel_forward_envs` 两个 mjlab 1.5 新增且取 0 的字段。

**结构上的调整**

| 旧 | 新 |
|---|---|
| `src/tasks/lower_body/lower_body_env_cfg.py`（914 行） | `cfg/` 下按 manager 拆成 7 个模块 |
| `src/tasks/lower_body/mdp/rewards.py` + `src/tasks/velocity/mdp/rewards.py` | `mdp/rewards/{tracking,gait,posture}.py` |
| `src/assets/robots/`（7 台机器人 + G1 常量 305 行副本） | 只留 G1+Gloria，执行器常量从 mjlab 导入 |
| `scripts/check_deploy_policy.py` 被 `_compare_video.py` 用 `importlib` 按路径加载 | 公共部分提到 `g1_lower_rl/deploy.py`，正常 import |
| `setup.py` | `pyproject.toml` |
| 顶层包名 `src` | `g1_lower_rl` |

**API 迁移要点（1.2.0 → 1.5.3）**

- `mjlab.utils.os.update_assets` 已删除；`mujoco.MjSpec.from_file` 自己解析 `meshdir`。
- `mjlab.tasks.velocity.mdp.feet_air_time` 换成了“腾空时长落在区间内计一分”的语义，
  与本任务用的单脚支撑判据不同 → 本包保留原实现（`mdp/rewards/gait.py`）。
- `mjlab.envs.mdp.body_orientation_l2` 被 `upright` 类取代，语义不同 → 保留原实现。
- `track_linear_velocity` 上游没有 `z_penalty` / `std_scale` / `std_knee` → 保留本任务版本。
- `feet_slip` / `soft_landing` / `variable_posture` / `self_collision_cost` /
  `body_angular_velocity_penalty` / `commands_vel` 与上游逐字相同 → 直接从 mjlab 转出，不再复制。
- `MujocoCfg.multiccd` → `disableflags` / `enableflags`；`SimulationCfg` 新增 `broadphase*`。
  本任务没用到这些字段。
- `BuiltinPositionActuatorCfg.frictionloss` 默认由 `0.0` 变为 `None`（保留 XML 值）。
  本 MJCF 未设该属性，MuJoCo 默认为 0，行为不变。

**暂未迁移**

- `velocity` / `tracking` 任务（按计划最后再迁）。
- `LowerBody-Flat-V2`（复刻历史配置 `2026-07-30_16-36-09` 的消融任务）。
- `scripts/` 里的一次性分析脚本：`_cmp_runs.py`、`_diff_env_yaml.py`、`_rank_policies.py`、
  `_scan_arm_ranges.py`；`csv_to_npz.py` / `visualize_terrain.py` 已是 mjlab 内置命令。
