# SiT 弱 MLP 增加一层：实现、速度与短程训练

本次只检查容量扩展的正确性、完整 GAN 更新速度及短程可训练性；不据训练 loss 判断 FID 改善。
额外外推系数固定为 **a=1.05**，对应 `S + a f(t)(S-W)`。

## 结构与起点

SiT-S/2、ImageNet-100，仍读取冻结强模型第 4 层、宽度 384 的特征。
旧头为条件与位置融合的一层隐藏 MLP：

```text
h = SiLU(token(x) + condition(c) + position(p))
W_old = output(h)
```

新增一层 `Linear(384,384)` 及 SiLU，并保留残差：

```text
h_new = h + SiLU(hidden_residual(h))
W_new = output(h_new)
```

- 参数：304,528 → 452,368，增加 147,840（48.55%）。
- 新层权重和偏置均为零；由于 `SiLU(0)=0`，初始预测与旧头一致。新层梯度不为零，能够开始学习。
- 所有旧弱头参数仍参与训练，旧 Adam 一、二阶矩继续使用；只有新增参数的 Adam 状态从零开始。
- 强模型、VAE、Inception 冻结；保持完整输入梯度、64 步 Heun、原时间窗、RGB 特征判别器、D/W 交替更新、R1 和 FP32 SiT 计算。
- 从旧 RGB GAN v2 的 `checkpoint_001456.pt` 继续，而不是从随机弱头开始。该起点含 256 步 D 预热和 1,200 次 W 更新。

这是扩大原模型函数族的受控实现，不是 SSG 中更大 Transformer adapter 的复现，也不预设更大的头必然改善图像质量。

## 单卡配对测速

同一 RTX 4090（GPU 1）、batch 12、相同保存状态和输入；每轮恢复 D/W/Adam，预热 1 轮后测量 10 轮中位数。
包含完整采样、解码与 Inception 反馈、D/W 反传和 Adam；不含首次加载、CUDA 图捕获、数据读取与 checkpoint 写盘。

| 项目 | 原 MLP | 加一层 MLP |
|---|---:|---:|
| 一次完整 D/W 更新 | 2.823174 秒 | 2.837061 秒 |
| 相对耗时 | 1.0000 | 1.0049 |
| 峰值 allocated | 9.383725 GiB | 9.387096 GiB |
| 峰值 reserved | 11.185547 GiB | 11.279297 GiB |

观察到约 **0.49%** 的耗时增加；如此小的差异不应被视为精确的长期性能预测。显存开销很小。
批量 12 的完整采样初始端点与旧头逐元素完全一致；首次 D/G loss 也一致。
新增参数参与总梯度裁剪后，训练更新当然不再与旧结构相同。

## 正确性检查

- CPU：8 项通过；4 项需要 CUDA 的既有测试在 CPU 调用中跳过。
- 加层零初始化保持旧输出、输入梯度和原有参数梯度；新增层梯度非零，更新后预测发生变化。
- 旧 Adam 状态迁移、新增参数状态初始化、深头权重和 Adam 再次恢复均通过。
- 真实 SiT 的完整 64 步轨迹检查覆盖新噪声、新标签及一次参数更新；CUDA 图采样端点与常规实现完全相同。
- 两次完整参数梯度相对误差分别为 `8.46e-8`、`9.37e-8`；新增层单独检查的最大相对误差为 `1.10e-7`。

## 连续训练

使用两张 4090，每卡 batch 12，全局 batch 仍为 24；已从 1456 完成 100 次 D/W 更新到 1556，正常退出。
W 学习率 `1e-6`、D 学习率 `1e-4`，本次不重复 D 预热。
旧 checkpoint 使用 4 个 rank，此处 2 个 rank，因此全局 batch 相同但不承诺随机数据流逐项衔接；恢复记录明确标注 `exact_global_stream=false`。

| 连续训练指标 | 结果 |
|---|---:|
| 稳态更新中位数（除第一步） | 2.951669 秒 |
| 稳态更新均值 | 2.957641 秒 |
| 100 步计时之和 | 296.35 秒 |
| rank 0 峰值 allocated | 9.388482 GiB |
| rank 0 峰值 reserved | 11.414063 GiB |

此处每步计时包含数据读取、D/W 更新、EMA 及梯度同步，不含外部加载和保存。
全部 100 步指标与最终 head/EMA/D 状态均有限；新增层权重范数为 `0.00253427`、偏置范数为 `0.00010129`，两个参数的 Adam 步数均为 100。
旧弱头参数均发生更新，Adam 步数保留为 1,300；多卡保存时参数一致。
按稳态速度线性估计，两卡每 1,000 步约 49 分钟，不含启动和保存开销。

主检查点为 `training_deeper_hidden_100/checkpoint_001556.pt`。本次没有进行 5K FID，不能据此判断生成质量是否胜过原头。

随后从该检查点独立恢复一次 D/W 更新到 1557，不传 `--extra-hidden-layer` 也能自动识别结构。
恢复后的初始头与保存权重哈希完全一致；没有新建或清空已有 Adam 状态，旧参数步数继续到 1,301、新层继续到 101，两个 rank 一致。
该一步仅作恢复验证，不计入上述训练速度统计。两个试跑均正常退出，GPU 已释放。

## 文件与复现

实现：[`heads.py`](../../classifier_guidance/heads.py)，训练参数为 `--extra-hidden-layer`。
后续从深头 checkpoint 恢复时自动识别，无需再次加层；评测器也支持深头结构。

完整实验目录：`/home/zhoushunyu/data/eqvae/projects/classifier_guidance/sit_deeper_mlp_20260919/`。

- `benchmark_shallow_b12/result.json`：原头配对测速。
- `benchmark_deeper_hidden_b12/result.json`：最终结构配对测速。
- `deeper_hidden_replay_audit.json`：真实模型端点与梯度检查。
- `training_deeper_hidden_100/`：连续训练、源码快照、恢复记录、逐步日志和 checkpoint。
- `resume_deeper_hidden_one_update/`：自动识别深头结构的独立一步恢复验证，不并入 100 步测速。
- `summary.json`：本次测速、训练有限性、参数更新及恢复检查的汇总。

```bash
$HOME/miniconda3/envs/myenv/bin/python -m classifier_guidance.launch \
  --gpus 1,2 \
  --output "$HOME/data/eqvae/projects/classifier_guidance/sit_deeper_mlp_20260919/training_deeper_hidden_100" -- \
  --model sit_small --updates 100 --global-batch 24 --coefficient 1.05 \
  --resume "$HOME/data/eqvae/experiments/adversarial_weak_training_20260915/endpoint_binary_gan_rgb_v2/checkpoint_001456.pt" \
  --extra-hidden-layer --warmup 0 --save-every 50
```

复跑须使用新输出目录。实验目录中另保留早期 `h + extra(SiLU(h))` 的试跑，已停止；其 12 次更新没有用于最终版本的起点。
这些早期文件在目录 README 中标记为 retired，不纳入本报告比较；使用不同 checkpoint 参数名，防止误按最终结构加载。
