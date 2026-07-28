# RAE 上 LPL 的改进研究

## 现在真正知道了什么

四个严格配对训练 seed 都显示：加入冻结 decoder 中间特征损失后，普通 latent flow MSE 略差，但 5k FID 平均改善约 `10.84%`，KID 和 IS 也全部同向改善。

这支持一个具体机制：

> RAE latent 中相同大小的欧氏误差，经 decoder 放大后的视觉代价并不相同；均匀 MSE 没有正确衡量这些方向。

把 decoder 中间特征记作 `phi(z)`。在真实 latent 附近做一阶展开：

```text
phi(z + delta) - phi(z) ~= J_phi(z) delta
```

于是 LPL 局部近似于：

```text
delta^T [J_phi(z)^T J_phi(z)] delta
```

也就是说，LPL 不只是“又加一个感知损失”，而是在用 decoder 诱导的局部度量替代各向同性的 latent MSE。这与我们此前发现的 decoder 方向敏感性一致。

## 原始 LPL 在 RAE 上的三个失配

### 1. 层权重退化

原 LPL 利用不同 decoder 层的空间分辨率设置权重。RAE 的 ViT-XL decoder 五个所选 hidden state 全部是 `16 x 16` token grid，因此当前实现只能等权平均。它没有回答哪一层真正最能预测最终图像风险。

### 2. outlier mask 基本无效

三个新增 seed 中，mask 的平均保留率都高于 `99.99997%`。当前 mask 增加了 percentile 和 morphology 计算，却几乎不改变目标。

### 3. 时间权重过于粗糙

硬门控 `t / (1 - t) <= 3` 只让约 `20%` 样本获得 LPL。门内样本一律同权，门外样本一律为零；这没有利用 decoder 风险随时间连续变化的信息。

## 与已有工作的边界

- LPL 已经提出用冻结 autoencoder decoder 的中间特征缓解 latent diffusion 与 decoder 的失配。
- LPL 原文已经做过 decoder 层子集消融：早层可能无益，较深的第三、第四层贡献最大，最后一层增益有限。因此“少用几层”只能是效率基线，不能作为新贡献。
- LPL 原文也已经比较统一层权重和按分辨率加权，并发现后者更好；我们的新问题来自 RAE 的所有 Transformer hidden state 分辨率相同，使原权重依据失效。
- Self-Perceptual Loss 使用 denoiser 自身中间特征改善 diffusion 目标。
- REPA 将 noisy denoiser hidden state 对齐到干净的预训练视觉表征，重点是语义学习和训练效率。
- PixelGen 同时使用局部感知和 DINO 全局语义监督，说明二者互补，但也意味着简单的“LPL + DINO loss”创新性不足。
- Min-SNR 从不同时间步梯度冲突出发重新加权 diffusion loss，因此单纯把硬门控改成常见 SNR 权重也不够新。
- CVPR 2026 的 latent diffusion inversion 工作已经用随机 SVD 和 Hutchinson 估计 decoder pullback metric，并证明样本和 latent 维度的 distortion 不均匀。因此“计算 Jacobian 对角线/主方向”也不是新颖点。
- Geometry-Preserving Encoder/Decoder 从 tokenizer 训练端主动保留数据几何，并分析其对 decoder 优化条件的影响。它进一步说明 latent 几何很重要，但需要重新训练 encoder/decoder；它没有直接解决冻结 RAE tokenizer 后如何让 latent flow 使用 decoder 几何。
- 上述工作把 decoder metric 用于解释和改善隐私攻击；尚未直接回答如何把它变成面向 RAE 生成质量、并且比完整 LPL 更便宜的训练目标。这才是可争取的研究空隙。

因此，最有希望的贡献不是继续堆损失，而是：

> 为高维 RAE latent 构造一个低成本、可校准、随时间变化的 decoder-induced metric 近似，并证明它比原始等权 LPL 更准、更便宜。

## 推荐方法：Decoder Metric Preconditioning

研究优先级固定为：

```text
误差重分配机制验证
-> 单层/去 mask 效率基线
-> 低秩 decoder metric
-> 时间条件与谱值 tempering
-> 最后才考虑额外语义监督
```

### 核心目标

用一个固定的轻量算子 `M_t(z)` 近似 decoder 局部度量：

```text
L_DMP = || M_t(z_0) [z_hat_0 - z_0] ||_2^2
```

它应当近似原 LPL 的有效梯度，但训练时不必每一步反传完整 ViT-XL decoder。

这里的目标是按 decoder 风险加权误差，不是直接乘 `G^{-1}`。后者会反而减弱 decoder 最敏感方向的监督。若 `G` 条件数过大，应使用谱值裁剪或 `s^alpha` tempering，在风险对齐和优化稳定性之间折中。

### 第一版只做固定低秩近似

1. 用少量 ImageNet train latent 做一次离线校准。
2. 对每个 clean latent，在归一化 decoder features 上采样随机 Rademacher 投影 `q`，一次反向得到 `g = J_phi^T q`。
3. 跨图像累计 `g^2` 得到 Hutchinson 对角估计；对堆叠的 `g` 做 randomized SVD 得到全局主方向。这样不需要显式构造 Jacobian。
4. 拟合一个低秩加对角形式：

```text
G ~= U diag(s) U^T + diag(d)
```

5. 用 `trace(G) / dim(z) = 1` 固定整体尺度，使 loss 权重不再随 rank 或校准样本数任意变化。
6. 训练 DiT 时只计算：

```text
e = z_hat_0 - z_0
L_DMP = sum_i d_i e_i^2 + sum_j s_j (u_j^T e)^2
```

不再经过 decoder。

当前 RAE latent 约有 `16 x 16 x 768 = 196,608` 个坐标。一个 fp32 对角度量不到 `1 MiB`，rank-64 basis 约 `48 MiB`；每个样本主要增加一次 `U^T e` 投影，远小于反传 ViT-XL decoder。这个估算说明方案在四张 4090 上有实际可行性，但全局平均 metric 是否足以表示样本相关的 decoder 几何仍必须由 held-out 实验决定。

这条路线直接针对当前 `22%+` 的训练开销，也能明确解释哪些 latent 方向对 decoder 最危险。

### 第二版再加入时间条件

先离散成三个时间区间，各自校准 `G_low/G_mid/G_high`。只有静态版本确实能预测 LPL 和生成收益时，才进一步拟合连续 `G(t)`；不要一开始就加入复杂网络。

## 最小实验顺序

以下所有 metric 拟合、组件选择和阈值选择都只划分 ImageNet train 内部的 calibration/audit 子集。ImageNet validation 和 ADM reference 始终只用于最终报告，避免把感知度量调到评估集上。

### 实验 0：先验证误差是否真的被重新分配

对四个现有 flow/LPL checkpoint 的同一批 held-out 样本、时间和噪声，记 clean latent 预测误差为：

```text
e = z_hat_0 - z_0
```

用 Hutchinson VJP 估计每个 latent 坐标对 decoder feature metric 的敏感度 `diag(J_phi^T J_phi)`，按敏感度分成十组，然后比较：

- 每组的 `e^2` 能量；
- decoder 加权误差；
- flow 与 LPL 分支之间的能量迁移；
- 随机打乱敏感度排序后的负对照。

核心预言是：

> LPL 分支可以有略高的总 latent MSE，但在 decoder 最敏感的坐标或方向上误差更低，并把部分误差转移到低敏感方向。

通过标准：

- 至少三个 checkpoint/seed 在最高敏感组上同向降低误差；
- 降低幅度显著大于随机分组；
- 高敏感组误差变化比总 latent MSE 更能预测 decoder feature error 变化。

若不满足，停止把当前结果解释成 decoder metric 下的误差重分配，先寻找 LPL 的其他作用机制。

### 实验 A：原 LPL 组件审计

固定同一 checkpoint 和 held-out 数据，逐层测量：

- 单层 feature loss 与五层 LPL 的相关性。
- 单层梯度与完整 LPL 梯度的 cosine。
- 单层对最终 decoded LPIPS/像素误差的预测能力。
- 去掉 mask 后 loss、梯度和用时变化。

通过标准：

- 找到 1 到 2 层能保留至少 `90%` 的完整 LPL 梯度方向或风险排序；
- 或明确证明必须保留多层。

这一步是把原论文层消融迁移到 RAE 的必要基线，不作为方法创新。

### 实验 B：低秩 metric 是否能替代 decoder 反传

在未用于校准的 latent 和扰动上比较：

- Euclidean latent MSE；
- diagonal metric；
- rank `8/16/32/64` metric；
- 完整 LPL。
- 随机打乱 diagonal 坐标的负对照；
- 保留相同谱值但随机旋转 basis 的负对照。

核心指标：

- 与完整 LPL 数值的 Spearman 相关；
- 与完整 LPL 对 latent 的梯度 cosine；
- 对 decoded LPIPS/FID proxy 的预测能力；
- 额外训练开销。

通过标准：

- held-out Spearman `>= 0.8`；
- 梯度 cosine `>= 0.7`；
- 训练额外开销 `<= 10%`。

随机负对照必须明显差于真实 metric；否则收益只能解释为增加了一个额外二次损失，不能归因于 decoder 几何。

### 实验 C：500-step 配对训练筛查

比较：

```text
flow
flow + strict LPL
flow + diagonal DMP
flow + low-rank DMP
```

每个分支必须从同一 checkpoint、optimizer 和 RNG 状态开始。先用三个训练 seed、固定 5k 采样，不调终验 seed。

晋级标准：

- DMP 至少保留 strict LPL 的 `80%` FID 改善；
- 墙钟开销不超过 `10%`；
- latent MSE 略差仍可接受，但不能出现持续爆炸或明显 diversity 下降。

若低秩近似能保留原 LPL，再比较对谱值做 `s^alpha` tempering 或上限裁剪。它可能避免少数极端敏感方向主导梯度，并把“廉价近似”升级成真正优于原 LPL 的训练度量。该比较必须在固定校准集上预先选择 `alpha/clip`，不能依据最终 5k FID 反复调参。

### 实验 D：时间权重

只有 C 通过才比较：

- 原始硬门控；
- 三段式风险权重；
- 连续、归一化的风险权重。

权重必须在训练前由 held-out proxy 固定，不能按最终 FID 选择。

## 暂不优先的方向

### 直接加 DINO 语义损失

它可能改善结果，但难以区分收益来自 decoder 对齐还是 REPA/PixelGen 式语义监督。可以作为 DMP 通过后的正交扩展，不应成为第一版核心。

### 直接增大 LPL 权重

当前 LPL 已使梯度范数上升约 `41%`、clip rate 上升到约 `1.5%–2.1%`，而 flow loss 略差。继续增大权重很可能只是扩大目标冲突。

原 LPL 论文还发现 LPL 分支的最佳 EMA momentum 与 baseline 不同。正式论文实验可以同时报告“共享 EMA 的严格配对结果”和“各自调优 EMA 的最佳结果”，但 EMA 调参属于公平性控制，不是方法贡献。

### 一开始做 rollout LPL

短轨迹监督更接近采样，但计算成本更高，也会把“decoder metric 是否正确”与“单步 clean estimate 是否足够”混在一起。应放在低成本 metric 失败后的诊断阶段。

### 同时训练 encoder/decoder

当前四 seed 结果的价值就在于冻结 tokenizer 后仍能观察到稳定的 decoder-diffusion disconnect。立刻联合训练会失去这个干净问题，并显著提高资源需求。

## 研究停止条件

满足任一条件就停止把 DMP 当主线：

1. 低秩/对角 metric 在 held-out 上不能预测完整 LPL 或 decoded image risk。
2. proxy 相关性高，但三 seed 500-step 生成没有一致收益。
3. 必须接近完整 decoder 计算量才能保留 LPL 收益。
4. 收益只能通过明显牺牲 diversity、IS 或普通 flow 稳定性获得。

若 DMP 失败但 strict LPL 的 50k 收益成立，结论仍然有价值：RAE 存在真实 decoder-diffusion disconnect，但这个风险不能被一个全局低秩度量简单压缩。

## 主要文献

- Berrada et al., *Boosting Latent Diffusion with Perceptual Objectives*, ICLR 2025: <https://arxiv.org/abs/2411.04873>
- Lin and Yang, *Diffusion Model with Perceptual Loss*: <https://arxiv.org/abs/2401.00110>
- Yu et al., *Representation Alignment for Generation*, ICLR 2025 Oral: <https://arxiv.org/abs/2410.06940>
- Hang et al., *Efficient Diffusion Training via Min-SNR Weighting Strategy*: <https://arxiv.org/abs/2303.09556>
- Rao et al., *Latent Diffusion Inversion Requires Understanding the Latent Space*, CVPR 2026: <https://arxiv.org/abs/2511.20592>
- Lee et al., *Geometry-Preserving Encoder/Decoder in Latent Generative Models*: <https://arxiv.org/abs/2501.09876>
- Sigillo et al., *Latent Wavelet Diffusion*: <https://arxiv.org/abs/2506.00433>
- Ma et al., *PixelGen: Improving Pixel Diffusion with Perceptual Supervision*: <https://arxiv.org/abs/2602.02493>
