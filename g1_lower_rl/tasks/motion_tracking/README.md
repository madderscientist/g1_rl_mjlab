# 全身动作捕捉：MLP + SuperMimic

正式任务名为 `G1-Gloria-MotionTracking`，只实现MLP + SuperMimic。
其他脚步、速度和站立任务保持原算法。历史权重、实验日志和冻结源码不改写。

## 正式实现

- [训练入口](../../../scripts/train_mocap.py)与
  [`train.py G1-Gloria-MotionTracking`](../../../scripts/train.py)调用同一实现。
- [MLP actor](training/position_policy.py)：参考编码器
  `798 -> 2048 -> 1024 -> 512 -> 256 -> 32`，末端tanh；
  解码器 `962 -> 2048 -> 1024 -> 512 -> 256 -> 29`，隐藏层SiLU。
  共9,132,861参数。正式入口不暴露RGMT、门控或GRU架构选项。
- [SuperMimic训练核心](training/position_runner.py)：原生位置控制采集，
  在线拟合[位置WM v3](../../world_model/README.md)，H32想象轨迹BPTT更新actor，GAE训练critic。
  使用原生样本约束actor KL，**当前不做PPO更新**。这是项目适配，不宣称逐项复刻论文。
- 原生控制为50Hz、每拍四个5ms物理子步；无显式上肢重力补偿。
  正式任务版本为 `recovery_xy_light_v1`，保留低概率失败终止及轻量躯干速度/XY轨迹奖励。
- 全量语料98,049条，训练96,152条、留出1,897条；CPU mmap，不把整库塞入显存。
  family与重复内容不跨划分。域随机化和混合初始化由原生桥接控制，续训严格继承其来源契约。

## 交付结果：MLP14160

目录：[export/mocap_mlp_14160](../../../export/mocap_mlp_14160/)。

| 文件 | 用途 |
|---|---|
| [policy.onnx](../../../export/mocap_mlp_14160/policy.onnx) | FP32、动态batch的确定性actor，不含WM和参考动作 |
| [policy_contract.json](../../../export/mocap_mlp_14160/policy_contract.json) | 输入布局、缩放、关节顺序、控制器、增益、限位、来源 |
| [training/update_00014160](../../../export/mocap_mlp_14160/training/update_00014160/) | rank_0/rank_1完整检查点 |
| [runtime](../../../export/mocap_mlp_14160/runtime/) | 与这份完整检查点来源严格匹配的冻结代码 |
| [bundle.json](../../../export/mocap_mlp_14160/bundle.json) | 权重/源码SHA256与外部依赖 |
| [onnx_validation.json](../../../export/mocap_mlp_14160/onnx_validation.json) | ORT与PyTorch输出一致性报告 |
| [validation/strict_restore](../../../export/mocap_mlp_14160/validation/strict_restore/) | 双rank严格零更新恢复报告和原生输入样本 |

选择依据是[同128条留出动作的配对比较](../../../logs/evaluation/mlp14160_gated14930_20387_heldout128_20261106/paired_comparison.json)：
未发现门控的明确跟踪优势，MLP前向成本更低。这不是架构因果优劣或真机安全证明。
14160是保存的训练轮次，不是从随机初始化只训练了14160轮；保留原始蒸馏、PPO历史和后续BPTT来源。

## 输入输出与控制器

ONNX有两个输入，均为**未归一化的FP32物理量**，归一化已经包含在网络中：

1. `history [B,10,93]`：最旧到最新；每帧依次是
   `编码器q减默认位姿29 + qd29 + 本体角速度3 + 本体重力方向3 + 上一拍raw残差29`。
   qd在网络内除5，角速度除2。reset遵循原生观测管理器的历史清零/填充语义。
2. `reference [B,798]`：21个38维token按时间排列；每token是
   `参考本体局部线速度3 + 参考本体局部角速度3 + 参考重力方向3 + 绝对参考关节角29`。
   参考角速度在网络内除2。偏移（单位20ms）为：
   `[-15,-13,-11,-9,-8,-6,-5,-3,-2,-1,0,1,2,3,5,6,8,9,11,13,15]`。
   跨clip边界时重复首/末帧，不跨动作取样。

输出 `raw_residual [B,29]`，**不是绝对关节目标，也不是力矩**。外部控制器执行：

```text
target = q_reference_current + action_scale * raw_residual - encoder_bias
target = clamp_if_actuator_limited(target, actuator_ctrlrange)
```

使用契约中的29关节顺序、动作缩放、默认位姿、PD参数和限位。
无默认站姿动作偏移；无额外GC；确定性回放不加训练探索噪声。
历史回填的是raw残差，而不是缩放/限位后的目标。不能把仿真的真实编码器bias或质量作为actor输入。
原生reset先写参考状态，再推进一帧command；不得假定reset后的参考仍停在clip第0帧。
参考动作需要单独提供，推理不需要WM。旧下肢速度策略的CPU部署检查器明确拒绝此接口。

## 原生回放与检查

在仓库根目录激活项目环境后：

```bash
# 交互回放（不启动训练；不要与已有Viser抢同一端口）
python scripts/play.py G1-Gloria-MotionTracking \
  --checkpoint-file export/mocap_mlp_14160/training/update_00014160 \
  --device cuda:0 --viewer viser --viser-port 8081

# 无界面原生控制链smoke
python scripts/play.py G1-Gloria-MotionTracking \
  --checkpoint-file export/mocap_mlp_14160/training/update_00014160 \
  --device cuda:0 --steps 100

# 固定留出分区前32条，名义物理，无噪声；独立于已完成的128-family配对基准
python scripts/eval_corpus.py \
  --checkpoint export/mocap_mlp_14160/training/update_00014160 \
  --clips 32 --max-seconds 20 --output logs/evaluation/mlp_native.json

python scripts/render_policy.py \
  --checkpoint export/mocap_mlp_14160/training/update_00014160 \
  --output logs/render/mlp.mp4
```

回放和评估用原生MuJoCo，不是WM想象；关闭噪声/DR、使用参考初始化，不因root高度/倾斜提前重置。
这与训练的混合初始化和低概率失败终止不同，不能把名义回放当成鲁棒性认证。
`--motions`可传逗号分隔的manifest动作名。完整语料仍须存在。

## 严格继续训练

完整检查点包含actor、**已在线更新的WM**、critic、三套Adam、sampler、
覆盖率、累计采集/更新预算和RNG。不是只加载actor，也不会把预训练WM覆盖回在线状态。
原生模拟器轨迹和replay按已有恢复协议重新建立，不承诺中断前后逐物理步bitwise连续。

正式入口自动读取检查点配置，核验并执行其冻结runtime；不放宽源码白名单，不伪造新的来源。
这使当前代码整理不会悄悄改变14160的训练算法。必须维持保存时的**双rank**和所有关键配置，
并使用新的输出目录；不能原地覆盖交付目录。无限训练必须由使用者明确执行：

```bash
# 只验证恢复，零优化器更新；不是训练
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
torchrun --standalone --nproc-per-node=2 scripts/train.py G1-Gloria-MotionTracking \
  --resume export/mocap_mlp_14160/training/update_00014160 \
  --output logs/supermimic/mlp14160_restore_check --preflight

# 真正继续训练；此命令不会在交付时自动执行
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
torchrun --standalone --nproc-per-node=2 scripts/train.py G1-Gloria-MotionTracking \
  --resume export/mocap_mlp_14160/training/update_00014160 \
  --output logs/supermimic/mlp14160_continue --iterations 0
```

在新run目录写入 `STOP` 后，当前完整轮结束时保存退出。常规检查点仅滚动保留两代，
最终结果应独立打包保留，不把交付物放到滚动清理目录。

**外部依赖不随权重包复制**：项目Python环境、完整mmap语料、固定第三方原生实现、
最初构造WM用的预训练检查点。精确路径和hash见
[bundle.json](../../../export/mocap_mlp_14160/bundle.json)。
本机恢复已验证；这不是脱离语料和依赖即可在任意机器运行的独立安装包。
更换路径/物理/任务/架构不是严格恢复，当前正式入口会拒绝，不会静默重建Adam。

## 从头训练与再次导出

```bash
python scripts/train_mocap.py \
  --world-model path/to/position_v3_pretraining.pt \
  --replica-source path/to/pinned/SuperMimic-Reproduced \
  --mocap-dir motions/mocap_supermimic_v1 \
  --output logs/supermimic/new_mlp --iterations 0

# 完整权重与精确runtime另存一份，不修改原件；目标目录必须不存在
python scripts/package_mocap.py \
  --checkpoint path/to/checkpoints/update_00014160 \
  --output export/another_mlp_result

# 导出并检查FP32 PyTorch/ORT一致性；目标目录不能已有同名导出
python scripts/export_onnx.py \
  --checkpoint export/another_mlp_result/training/update_00014160 \
  --output-dir export/another_mlp_result
```

从头训练默认MLP、原生KL、H32 BPTT与GAE，配置数量可显式覆盖；
新run会先冻结runtime，再启动训练。续训不采用这些新run默认值，而是继承原存档。
工作区不保留RGMT、门控actor、PPO/混合更新、蒸馏重启或配置/任务迁移分支。
WM预训练保留独立的原生22650教师采集器；策略训练只采集当前MLP。
已交付14160按其原始冻结runtime续训，代码清理不改写封存权重或来源。
