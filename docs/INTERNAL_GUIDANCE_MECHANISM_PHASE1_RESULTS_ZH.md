# Internal Guidance 机制研究：第一阶段结果

日期：2026-08-03

## 1. 研究问题

本阶段不训练模型，只研究一个问题：

> Internal Guidance 使用的 `full - base`，究竟是在逐时刻修正监督误差，还是通过整条采样动力学把终点分布推向更高质量但不一定更真实的位置？

记：

- `Y`：监督目标；RAEv2 中是 clean RAE latent，SiT 中是目标 velocity；
- `F`：full head 预测；
- `B`：base/early head 预测；
- `d = F - B`：IG 方向；
- `s`：代码中的 IG scale；
- `gamma = s - 1`：相对 full head 的外推系数。

于是：

```text
guided = F + gamma * d
```

配对平方误差满足：

```text
R(gamma) - R(0)
  = 2 gamma E[<F-Y, d>] + gamma^2 E[||d||^2]
```

总体局部最优系数为：

```text
gamma* = E[<d, Y-F>] / E[||d||^2]
```

只有 `gamma* > 0`，正向 IG 才是局部监督误差修正。

## 2. RAEv2 局部方向审计

配置：

- 官方 RAEv2 DINOv3-L K7 checkpoint；
- EMA state；
- ImageNet validation；
- 512 张图；
- 5 个随机种子；
- 8 个官方 shifted Euler solver 时刻；
- 模型冻结，bf16 推理；
- 每个种子独立抽图和噪声；
- 总计 40 个“种子 × 时刻”组合。

结果：

- 40/40 个组合的总体 `gamma*` 都为负；
- 跨种子 `gamma*` 标准差很小；
- 官方 `gamma=+0.78` 在所有时刻都增大配对 clean-latent MSE；
- base head 大多数时刻确实比 full head 更差，但 `full-base` 不朝向 full 的剩余误差。

跨种子平均 `gamma*`：

| solver time | gamma* |
|---:|---:|
| 0.0748 | -0.1055 |
| 0.1404 | -0.1128 |
| 0.1983 | -0.1010 |
| 0.3380 | -0.0573 |
| 0.4972 | -0.0408 |
| 0.6524 | -0.0384 |
| 0.7976 | -0.0424 |
| 0.9492 | -0.0962 |

结论：

> RAEv2 的正向 IG 不是 forward-noised 训练分布上的局部 clean-target MSE 纠错器。

结果目录：

```text
~/data/eqvae/experiments/raev2_ig_direction_audit/ema_solvergrid_n512_5seed_v1/
```

## 3. 原始 SiT-IG 跨模型复核

配置：

- 官方 `SiT-XL/2+IG` 800 epoch checkpoint；
- EMA state；
- 677,786,144 参数；
- ImageNet validation 的 SD-VAE latent；
- 512 张图；
- 7 个 flow 时刻；
- fp32，关闭 TF32；
- 模型冻结，不做 teacher-path rollout。

总体最优系数：

| time | gamma* | scale*=1+gamma* |
|---:|---:|---:|
| 0.10 | -0.0553 | 0.9447 |
| 0.20 | -0.0707 | 0.9293 |
| 0.35 | -0.0935 | 0.9065 |
| 0.50 | -0.1341 | 0.8659 |
| 0.65 | -0.2616 | 0.7384 |
| 0.80 | -0.5167 | 0.4833 |
| 0.95 | -0.4632 | 0.5368 |

同时，`scale=1.2/1.5/1.8` 在 7/7 个时刻都增大局部 velocity MSE。

结论：

> “正向 IG 不是逐时刻监督 MSE 纠错”同时出现在 RAEv2 和原始 SiT-IG，因而不是 RAE latent 独有现象。

结果目录：

```text
~/data/eqvae/experiments/internal_guidance_sit_audit/n512_seed20260803_v1/
```

## 4. RAEv2 单窗口终点响应

将 100 步采样分为五段：

```text
[0.8, 1.0]
[0.6, 0.8]
[0.4, 0.6]
[0.2, 0.4]
[0.1, 0.2]
```

只在一个窗口使用 `gamma=+0.05` 或 `-0.05`，其余步骤保持 full head。所有条件共享 checkpoint、类别和初始噪声。为避免小扰动被 bf16 舍入污染，采样和 endpoint 存储均使用 fp32。

正负扰动被分解为：

```text
odd  = (z(+gamma) - z(-gamma)) / 2
even = (z(+gamma) + z(-gamma)) / 2 - z(0)
```

- `odd/gamma` 近似一阶终点响应；
- `even/odd` 衡量二阶和非线性作用。

64 样本结果：

| window | direct injected norm | odd endpoint RMS | odd/gamma | even/odd |
|---|---:|---:|---:|---:|
| [0.8, 1.0] | 0.000339 | 0.05733 | 1.1466 | 0.7350 |
| [0.6, 0.8] | 0.000774 | 0.00522 | 0.1044 | 0.2377 |
| [0.4, 0.6] | 0.001400 | 0.00367 | 0.0735 | 0.3288 |
| [0.2, 0.4] | 0.002154 | 0.00359 | 0.0718 | 0.3360 |
| [0.1, 0.2] | 0.003487 | 0.00406 | 0.0811 | 0.2991 |

关键现象：

1. `h/t` 使直接注入能量向低噪声末端集中；
2. 终点传播增益却在最高噪声开端集中；
3. 高噪声窗口的单位 `gamma` 终点响应约为其余窗口的 7–16 倍；
4. 高噪声窗口 `even/odd` 较大，说明其作用不是干净的线性传播，而包含明显轨迹弯曲或分叉。

因此：

> “哪个窗口直接 impulse 最大”和“哪个窗口最终最能改变样本”是两个不同问题，而且在 RAEv2 中呈现近乎相反的排序。

结果目录：

```text
~/data/eqvae/experiments/raev2_ig_window_response/fp32_n64_seed20260809_g005_v1/
```

## 5. 64 样本解码筛选

同一批 endpoint 经官方 RAEv2 decoder 解码，再用 ImageNet Inception-v3 分类器筛选语义置信度。

高噪声窗口的主要变化：

| condition | max probability delta | true-class log-prob delta | entropy trend |
|---|---:|---:|---|
| gamma=+0.05 | +0.00910 | -0.00224 | 下降 |
| gamma=-0.05 | -0.00832 | +0.01343 | 上升 |

top-1 accuracy 在所有条件中均为 0.8125，没有变化。

这个小样本结果支持：

> 正向 IG 在高噪声阶段首先产生“更自信、更集中”的质量倾斜，而不是提高配对正确性或局部监督精度。

这与原始 IG 论文在二维例子中观察到分布变窄是一致的。但 N=64 只能作为筛选信号，不能报告为 FID 结论。

## 6. 当前能说与不能说的结论

已经有可靠证据：

1. RAEv2 与原始 SiT-IG 的正向 `full-base` 都不是局部监督 MSE 纠错方向；
2. RAEv2 的终点响应不能由 raw head gap 或 `h/t` 单独解释；
3. 高噪声小扰动具有异常高的终点杠杆和明显非线性；
4. 小样本解码信号更符合分布集中/置信度提高，而不是配对误差修正。

目前不能声称：

1. 高噪声窗口一定改善 FID；
2. 固定 scale 的最优值可以由当前局部量直接计算；
3. decoder 是全部排名反转的原因；
4. 已经得到可在线使用的反馈控制器；
5. 当前结果足以形成通用 IG 理论。

## 7. 下一步及验收标准

下一项大实验只保留三路：

```text
full baseline
high-noise gamma=+0.05
high-noise gamma=-0.05
```

先做 1k 配对样本，报告：

- decoded FID/KID，仅作为方向筛选；
- precision/recall 或等价覆盖指标；
- Inception true-class log-prob、confidence、entropy；
- endpoint latent 分布指标；
- 两个独立 seed。

扩到 5k 的门槛：

1. 正向与负向条件在两个 seed 上给出稳定相反的 concentration 信号；
2. 正向条件的 FID/KID 方向优于 baseline，且不是仅靠类别置信度提高；
3. endpoint 与 decoded 指标的排序能复现；
4. 结果大于 1k 评估噪声。

之后才做 replay 与 recursive 对照，判断高噪声高杠杆来自 baseline dynamics 的传播，还是 guidance 方向随状态更新形成的反馈。

## 8. 文献定位

- [Internal Guidance, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Zhou_Guiding_a_Diffusion_Transformer_with_the_Internal_Dynamics_of_Itself_CVPR_2026_paper.html)
- [RAEv2, arXiv:2605.18324](https://arxiv.org/abs/2605.18324)
- [The Unreasonable Effectiveness of Guidance](https://arxiv.org/abs/2411.10257)
- [Limited Guidance Interval, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/dd540e1c8d26687d56d296e64d35949f-Abstract-Conference.html)

现有论文说明“强模型减弱模型”常常有效，也说明 guidance 的时间窗口很重要；本阶段的新证据集中在它们之间缺失的一环：

> 局部 head difference 并不修正监督误差，但其时间局部扰动可被后续动力学以高度不均匀、非线性的方式传播到终点。
