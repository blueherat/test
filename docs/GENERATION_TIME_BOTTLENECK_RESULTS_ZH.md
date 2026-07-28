# 生成时间瓶颈研究结果

> **范围说明：** 本文记录的是 MNIST 上的显式时间门控旁支，其中
> `high_noise` 模型被人为禁止在后期读取 latent。它不等同于原始的“让 decoder
> 自然学习责任曲线”方案，也不能用这里的停止结论终止该主线。主线受控实验与
> 最终判断见 `docs/IMAGENETTE_NOISE_RESPONSIBILITY_RESULTS_ZH.md`。

## 1. 最终结论

本轮实验按预注册顺序完成了：

1. 公开 PiD checkpoint 不训练筛查；
2. MNIST 小型受控条件模型；
3. 固定预算 latent prior；
4. 端到端联合训练门槛判断。

结论分成两部分：

> **两阶段生成成立。** 一个 8 维 latent 可以先确定数字类别和全局形状，随机
> 像素 flow 再完成笔画；其端到端分布质量明显好于同预算的无条件像素 flow。

> **显式 high-noise 门控失败。** 只允许 latent 在高噪声阶段介入，并没有让
> latent prior 更容易；它反而移除了后半程有用的条件校正，使最终 feature FID
> 在两个 seed 上都比全时条件模型差。因此不进入联合端到端训练。

这不是“整个两阶段想法失败”，而是更精确的负结论：

```text
条件主要在早期决定语义
不等于
只在早期提供条件会更好
```

## 2. Phase A：PiD 公开模型筛查

对象为官方四步蒸馏 checkpoint：DINOv2、SigLIP2 和 SD3-VAE。数据为 5 张
真实原生 2K Wikimedia 图像，两个噪声 seed；所有 real/null/shuffle 分支复用
相同状态、噪声、时刻、文本和 velocity target。

主指标是：

```text
Delta_shuffle = MSE(shuffled latent) - MSE(real latent)
```

### 2.1 Teacher-forced 结果

| model | t=0.999 | t=0.866 | t=0.634 | t=0.342 | 第二步/第一步 |
|---|---:|---:|---:|---:|---:|
| DINOv2 | 0.364652 | 0.002864 | 0.000864 | 0.000016 | 0.79% |
| SigLIP2 | 0.472117 | 0.002114 | 0.000176 | 0.000047 | 0.45% |
| SD3-VAE | 0.813709 | 0.046288 | 0.001475 | -0.000061 | 5.69% |

### 2.2 Real-rollout 结果

| model | t=0.999 | t=0.866 | t=0.634 | t=0.342 |
|---|---:|---:|---:|---:|
| DINOv2 | 0.365892 | -0.005014 | 0.001918 | 0.000053 |
| SigLIP2 | 0.487529 | -0.002072 | -0.000402 | -0.000257 |
| SD3-VAE | 0.819763 | 0.044455 | -0.001885 | -0.000591 |

所有 checkpoint 的 identity 与 batch-order 控制均精确为 0；三个模型的总体
`Delta_shuffle` 在两个 seed 上都为正。

### 2.3 对原预测的修正

原预测“VAE latent 在低噪声仍明显有用”不成立。更准确的结论是：

- DINOv2/SigLIP2 条件几乎只在第一步决定内容；
- SD3-VAE 比语义 latent 多持续一个阶段；
- 到 `t=0.342`，三个 latent 都不再提供明显额外信息，因为当前像素状态已经
  携带足够内容。

该修正可由条件信息随状态逐渐变得冗余解释，不是实现异常。它支持研究
“信息何时交给像素状态”，但不能证明某种 latent 本身更好。

## 3. Phase B：小型受控条件模型

### 3.1 设置

- MNIST 官方 split，train 10,000 / test 2,000；
- 8 维 encoder latent，32-channel tiny pixel flow；
- encoder 和 pixel flow 联合使用 velocity loss 训练 4,000 steps；
- 无重构 loss、无类别监督、无外部 representation encoder；
- `all_time`、`high_noise`、`low_noise`、`none` 四组共享初始化、数据顺序、
  像素噪声和训练预算；
- `high_noise` 在 `t=0.55` 到 `0.75` 平滑开启，低噪声端精确为 0；
- 两个 seed，50-step 真实 rollout；
- 独立分类器 test accuracy 为 98.0% 到 98.6%。

### 3.2 条件是否真的控制生成

| seed | all-time 源类别 | high-noise 源类别 | high-noise shuffled | low-noise 源类别 | none 源类别 | high/all |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 82.8% | 81.9% | 9.7% | 12.4% | 10.7% | 98.9% |
| 1 | 85.0% | 82.0% | 8.7% | 12.9% | 11.5% | 96.5% |

这证明：

- 高噪声阶段注入的 8 维码能把类别信息写入像素状态；
- 后半程不再读取 latent，类别仍能保留到终点；
- 只在低噪声读取 latent 太晚，表现接近无条件；
- shuffled code 生成的是 shuffled code 对应的类别，而不是原图类别，因果对照
  很清楚。

但 `all_time` 与 `high_noise` 的 latent 线性分类准确率几乎相同：seed 0 为
87.95%/88.20%，seed 1 为 89.35%/86.90%。时间门没有自动制造更简单的 latent。

## 4. Phase C：固定预算 latent prior

### 4.1 设置

- 冻结 Phase B encoder 和 pixel flow；
- `all_time`、`high_noise` 分别训练同一个 8 维 residual MLP flow prior；
- 每个 prior 恰好 233,992 参数、6,000 steps、batch 256、fp32；
- prior 和 pixel flow 各 50 NFE；
- 每组生成 2,000 张图；
- 不使用 test 数据训练 prior。

### 4.2 结果

| seed | none 图像 FID | all oracle FID | high oracle FID | all prior FID | high prior FID | all latent SWD | high latent SWD |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 142.78 | 20.28 | 29.97 | 24.62 | 35.42 | 0.0783 | 0.0755 |
| 1 | 186.59 | 40.18 | 42.93 | 44.13 | 52.57 | 0.1017 | 0.1018 |

两个 prior 都没有类别塌缩：latent/image 类别熵约为 2.29，高于预注册的
`log(8)=2.079`，有效类别数约 9.83 到 9.95。

两阶段模型相对无条件像素 flow 的 FID 改善很大：

- `all_time`：seed 0 改善 82.8%，seed 1 改善 76.4%；
- `high_noise`：seed 0 改善 75.2%，seed 1 改善 71.8%。

但 high-noise 对 all-time 的方法门槛失败：

- 图像 FID 更差 43.9% 和 19.1%；
- latent SWD 只改善 3.6% 和恶化 0.1%，既不到 10%，也不跨 seed 一致；
- oracle code 下 high-noise FID 已更差 47.8% 和 6.9%；
- prior code 替换 oracle code 后，额外 FID 退化为 all-time 的 4.34/3.95，
  high-noise 的 5.45/9.63。

因此不是 prior 容量不足导致 high-noise 单独失败。两种 prior 对各自 latent 的
拟合难度接近，主要差距来自条件 decoder 的后半程校正能力。

## 5. 机制解释

### 5.1 Responsibility 是系统属性，不是 latent 单独属性

`Delta_shuffle(t)` 同时取决于：

- latent 保存了什么；
- 当前噪声状态已经保存了什么；
- decoder 是否有能力和训练压力继续读取 latent；
- 采样轨迹是否仍在训练分布附近。

所以 PiD 中“第二步以后几乎不读 DINOv2”不能直接推出“人工禁止第二步以后
读取 latent 会更好”。前者是训练后系统的行为，后者是一个结构干预。

### 5.2 类别交接成功，误差校正失败

high-noise 模型在前半程成功把类别写入像素状态，因此类别保持率几乎不降。
但 prior 生成的 latent 和早期形成的像素状态都存在小误差。`all_time` 可以在
中低噪声继续利用同一个 latent 修正笔画粗细、闭合、倾斜和局部形状；
`high_noise` 在门关闭后失去这条参考通道，只能相信已经形成的状态。

这解释了一个看似矛盾的组合：

```text
类别准确率几乎不变
但 feature FID 稳定变差
```

### 5.3 时间门控没有减少 latent 难度

两组 latent 维数相同，且都用 covariance regularizer 保持 8 个维度活跃。
时间门控只改变 decoder 何时读取 code，并没有直接限制 code 的熵、容量、
曲率或类内变化。因此期待 prior 自动更容易，本身缺少必要机制。实测相同的
latent SWD 正好证实这一点。

## 6. 与文献的边界

- Diffusion Autoencoders、InfoDiffusion 和 SWYCC 已证明低维语义 encoder 与
  随机 diffusion decoder 可以联合学习；本实验再次验证这条路线，但不能把
  它当新颖贡献。
- RCG 已覆盖 representation prior + conditional pixel generator。
- 已有工作也观察到条件在早期去噪阶段对高层语义更敏感。

本轮原本可能的新点是“用生成时间定义两阶段的信息边界”。结果说明，简单
时间门控不足以成为方法：它没有使 prior 更容易，还损失了后期校正。

## 7. 停止决定

按照预注册，联合端到端训练要求 high-noise 在两个 seed 上通过质量 guardrail
并显示稳定方法价值。实际两个 seed 的 `end_to_end_gate_pass` 均为 false。

因此本轮在 latent prior 后停止：

- 不训练联合 encoder/prior/decoder；
- 不扫描门阈值、latent 维度、loss 权重或更大模型；
- 不用更多 epoch 挽救 high-noise 结论；
- 保留“两阶段模型有效、硬时间分工无增益”的真实负结果。

若未来重新立项，必须提出一个直接降低 latent 生成难度、同时保留后期误差
校正的机制，例如可学习的软信任调度或显式信息率约束；这将是新假设，不能
作为本轮实验的事后补丁。

## 8. 可复核材料

- PiD 预注册：`docs/NOISE_RESPONSIBILITY_PROFILE_PREREG_ZH.md`
- PiD 探针：`experiments/run_pid_noise_responsibility.py`
- 小模型预注册：`docs/GENERATION_TIME_BOTTLENECK_PREREG_ZH.md`
- 小模型实现：`experiments/mnist_generation_time_bottleneck.py`
- prior 预注册：`docs/GENERATION_TIME_LATENT_PRIOR_PREREG_ZH.md`
- prior 实现：`experiments/mnist_generation_time_prior.py`
- 外部结果：`~/data/eqvae/pid_responsibility_screen`、
  `~/data/eqvae/generation_time_bottleneck_comparison`、
  `~/data/eqvae/generation_time_prior_comparison`
