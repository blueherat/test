# Imagenette-64 Prior-Decoder 断层：机制诊断结果

## 1. 结论先说

这轮研究得到一个清楚的负结论和一个新的正发现。

**被否定的机制：** 256d prior 的误差并没有特别落在 decoder 的高敏感方向上。
decoder 对真实数据方向和 prior 误差方向的等角响应几乎相同，因此不能把大 FID gap
解释为“prior 恰好偏到了一个危险方向”。

**得到支持的现象：** 普通 latent/condition 指标会严重低估 decoder 真正在意的分布
差异。256d 的 prior 与真实 condition 在普通分类器中不容易区分，但经过同一个冻结
decoder 后，二者的输出特征最容易区分，同时出现最大的类别比例和多样性偏移。

更准确地说，当前看到的不是简单的局部 Jacobian 放大，而是：

> prior 在 decoder 相关的细粒度模式和多样性上没有正确分配概率质量。普通 latent
> 距离没有把这种差异显露出来，decoder 作为一个任务相关的非线性观察器把它显露了。

这仍然是 Imagenette-64 小模型上的机制证据，不是 RAE/ImageNet 结论，也还不是一篇
完整的方法论文。

## 2. 从哪里开始

上一轮 `16/64/256d x 5 seeds` 已经确认：

| latent | Empirical FID | Prior FID | Modeling gap |
|---:|---:|---:|---:|
| 16 | 100.65 | 103.44 | 2.79 |
| 64 | 96.53 | 106.44 | 9.92 |
| 256 | 96.02 | 118.44 | 22.42 |

真实 latent 越丰富，冻结 decoder 的上限越好；但相同预算 prior 替代真实 latent 后，
完整系统反而越差。反常之处在于，256d 的 held-out flow MSE 和 raw SWD 都不差。
因此“高维 prior 普通意义上学不会”已经不足以解释结果。

## 3. 正式等角干预否定了什么

本轮首先在 decoder 实际使用的 192 维 condition embedding 球面上做受控干预。每个
真实 condition 都构造三个完全相同角度 `0.15 rad` 的方向：

1. 指向 Hungarian 匹配 prior condition 的方向；
2. 指向另一个真实 condition 的方向；
3. 随机切向方向。

所有分支共享 pixel noise，冻结 decoder，使用 fp32 和相同 50 步 Euler 解码。

关键结果：

| 指标 | 16d | 64d | 256d |
|---|---:|---:|---:|
| prior 方向 feature response | .533 | .605 | .621 |
| empirical 方向 feature response | .534 | .606 | .629 |
| random 方向 feature response | .316 | .423 | .453 |
| prior / random | 1.689 | 1.428 | 1.369 |
| prior / empirical | .998 | .997 | .988 |

decoder 确实随容量略微变敏感，256d 对 prior 方向的响应也在 `5/5` seed 中高于 16d；
但均值比只有 `1.165`，未达到预注册的 `1.20`。更重要的是，prior 方向并不比真实数据
方向敏感，`prior / empirical` 在 256d 甚至略低于 1。

预注册的 decoder-weighted mismatch 也没有预测 modeling gap：leave-one-seed-out
RMSE 为 `3.47`，反而差于只使用名义维数的 `1.84`。所以强机制 gate 正式失败。

## 4. 最反常、也最有价值的发现

随后使用全部 3,925 对 empirical/prior 输出做 post-hoc witness audit。训练/测试按
pixel-noise index 分组，保证共享同一噪声的 empirical/prior 样本不会跨越分类边界。

| latent | condition-space MLP AUC | decoded-feature MLP AUC | 输出类别 TV |
|---:|---:|---:|---:|
| 16 | .527 | .498 | .023 |
| 64 | .819 | .602 | .066 |
| 256 | .574 | **.637** | **.091** |

这个反序在五个 seed 中完全一致：

- condition space 中，64d 比 256d 更容易区分：`5/5`；
- decoder 输出后，256d 比 64d 更容易区分：`5/5`；
- 256d 的输出类别 TV 高于 64d：`5/5`。

因此，普通 condition-space C2ST 的大小并不能预测 decoded gap。64d 在输入空间有很
明显的可分误差，却只产生中等 gap；256d 的误差对普通线性/MLP witness 较隐蔽，经过
decoder 后却变成最大的输出分布差异。

这不违反数据处理不等式。理想的任意强判别器所能测得的真实统计散度不会被确定性
decoder 凭空增大；这里变化的是有限模型、有限样本和固定 ResNet feature 下的
**可见性**。decoder 是一个已经学会如何使用 latent 的任务相关映射，它把普通欧氏
指标忽略的差异变成了当前观察器容易检测的差异。

## 5. 类别质量分配能解释多少

为了判断输出类别偏移是原因还是伴随现象，又做了一个不训练模型的干预：

- 用四折数据估计 empirical/prior 的 Imagenette 类别质量比；
- 将比值只应用到第五折 prior 输出，五折交叉拟合；
- 重新计算加权 feature FID；
- 用 9 种错误的循环类别映射作为负对照。

结果如下：

| latent | 原 modeling gap | 类别重加权收回 FID | 收回比例 | 重加权 ESS |
|---:|---:|---:|---:|---:|
| 16 | 2.79 | .82 | 33.0% | 3,895 |
| 64 | 9.92 | 2.54 | 25.6% | 3,780 |
| 256 | 22.42 | 3.57 | **15.9%** | 3,515 |

三个容量均在 `5/5` seed 中改善，并且 `5/5` 优于错误类别映射对照。256d 的类别 TV
从 `.0912` 降到 `.0032`，说明粗类别质量分配确实是因果相关的一部分；但它只消掉约
16% 的 256d gap，所以不是主因。

FID 的精确分解进一步显示：

| latent | 特征均值 gap | 特征协方差 gap |
|---:|---:|---:|
| 16 | 1.28 | 1.51 |
| 64 | 4.60 | 5.32 |
| 256 | 7.92 | **14.50** |

256d 剩余问题主要落在协方差、覆盖和类别内部多样性，而不只是十个类别各生成多少。

## 6. 当前机制应该怎样表述

目前可靠的表述分三层：

1. **已确认：** latent 容量提高带来更好的 empirical decoder 上限，也带来更大的
   prior-to-decoder 完整系统 gap。
2. **已否定：** 这个 gap 不是由 prior 误差特殊对齐 decoder 局部高敏感方向解释的。
3. **新证据支持：** 256d prior 在 decoder 相关的细粒度模式和多样性上存在错配；
   普通 latent loss、SWD 和有限 C2ST 没有可靠测到这种错配。

第 3 层还不能改写为已证明的唯一因果机制。当前类别重加权只解释 16%，说明还需要
区分“更细的语义模式质量分配”和“每个模式内部的纹理/外观质量下降”。

## 7. 与已有工作的关系

这个问题并非完全空白。WAE 和 Sinkhorn Autoencoder 已经把 aggregated posterior 与
prior matching 放在生成自编码器核心位置；Sinkhorn Autoencoder 还明确联系了 latent
Wasserstein 误差和 decoder 的 Lipschitz 能力。decoder 诱导非欧氏 latent 几何也早已由
[Latent Space Oddity](https://arxiv.org/abs/1710.11379) 系统讨论。

较新的 RAE 工作进一步靠近当前问题：[LV-RAE](https://arxiv.org/abs/2602.08620)
关注高维信息型 latent 下 decoder 对偏离数据流形方向的脆弱性；
[Prior-Aligned Autoencoders](https://arxiv.org/abs/2605.07915) 则从 tokenizer 端改善
prior alignment。因此，仅仅提出“用 decoder-aware distance 训练 prior”还不足以构成
新颖贡献。[Sinkhorn Autoencoders](https://proceedings.mlr.press/v115/patrini20a.html)
也已经说明 latent matching 与 decoder 能力必须一起看。

当前可能有价值的新切口是更具体的：

> 在高容量两阶段生成中，通用 latent 指标可能与下游生成误差发生系统性反序；真正
> 需要匹配的是 decoder 可读出的细粒度模式质量，而不是平均 latent 几何。

这个切口需要方法和真实 RAE/ImageNet 验证，现有小模型结果本身还达不到 ICLR 论文量。

## 8. 下一步最小实验与停止线

不应继续调局部 Jacobian、等角扰动或新的 raw latent proxy。下一步只做一个低成本、
预注册的多尺度 mode 分解：

1. 在 empirical decoded ResNet feature 上固定聚类器；
2. 用 `K=10/25/50/100` 的细粒度 mode 做交叉拟合质量重加权；
3. 对照错误 cluster 映射，并报告有效样本数；
4. 测量 256d gap 能被 mode-mass 重加权收回多少。

继续方法阶段的门槛建议固定为：

- 256d 在至少 `4/5` seed 中收回 `>= 40%` modeling gap；
- 均值至少比错误映射对照多收回 `5 FID`；
- 重加权有效样本数保持原样本的 `>= 50%`；
- 趋势随 `K` 增大后稳定，而不是只在一个 K 上偶然出现。

若通过，再训练一个小型 frozen decoder witness，并只微调 256d prior，检验 witness
matching 能否同时降低 mode TV 和完整 FID。若多尺度 mode 分解仍只能解释少量 gap，
就停止“质量分配”路线，结论改为类内条件生成质量或 decoder-prior 联合动力学问题。

## 9. 高名义维数是否只增加了低维结构

现有 covariance spectrum 支持一个较弱但明确的结论：从 64d 增加到 256d，增加的
**高方差结构**远少于新增的 192 个坐标。

| latent | raw effective rank | raw 90% / 99% PC | condition effective rank | condition 90% / 99% PC |
|---:|---:|---:|---:|---:|
| 16 | 9.94 | 12 / 15 | 8.36 | 12 / 17.4 |
| 64 | 16.31 | 26.6 / 38.6 | 16.17 | 28.4 / 39.4 |
| 256 | 20.14 | 33.0 / 48.6 | 17.91 | 35.2 / 48.4 |

256d raw latent 的前 32 个 PC 已解释 `89.6%` 方差，前 64 个解释 `99.7%`；decoder
实际读取的 192d condition embedding 也只有约 `17.9` 的 participation-ratio rank。
因此“256d 的主要方差仍集中在几十个方向”已经得到五 seed 支持。

但这不能严格证明“256d 相比 64d 新增的信息只有约 10 个维度”。原因有三点：

1. effective rank 只测二阶方差，不测互信息；低方差方向仍可能编码重要细节；
2. 64d 和 256d encoder 分别训练，坐标系没有嵌套，不能直接把后 192 个坐标解释为
   新增信息；
3. condition MLP 是非线性的，PCA 方差低不等于 decoder 功能依赖低。

严格验证需要在同一个 256d 模型内做 PCA 截断干预：只保留 top-k condition 成分，
在共享 pixel noise 下测 Oracle FID、pixel MSE 和 class match 随 `k` 的恢复曲线；再单独
保留 residual，检查低方差部分是否仍提供不可替代的信息。这个干预尚未完成。

## 10. 复核材料

- 预注册：`docs/IMAGENETTE_DECODER_AMPLIFICATION_PREREG_ZH.md`
- 等角干预：`experiments/analyze_imagenette_decoder_amplification.py`
- witness audit：`experiments/analyze_imagenette_decoder_witness_gap.py`
- semantic reweight：`experiments/analyze_imagenette_decoder_semantic_reweighting.py`
- spectrum audit：`experiments/analyze_imagenette_latent_spectrum.py`
- 外部结果：`~/data/eqvae/imagenette_latent_prior_tradeoff/comparison_p0/`
- 主图：`decoder_amplification_summary.png`、`decoder_witness_gap.png`、
  `decoder_semantic_reweight.png`

所有 checkpoint、feature 和结果仍位于 `$HOME/data/eqvae`，仓库内没有复制大型数据。
