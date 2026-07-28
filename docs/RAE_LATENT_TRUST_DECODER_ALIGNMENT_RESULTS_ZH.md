# RAE Latent Trust 与 Decoder 语义轴对齐结果

> **2026-07 文献审计说明：** 本文记录的数值结果仍有效，但“语义轴”“信任机制”和“接力”等解释应视为探索性假设，
> 不能作为既定机制。RAEv2、iREPA、LV-RAE 及 RAE scaling 结果带来了新的重叠与模型尺度混杂，详见
> [RAE 生成机制文献审计](./RAE_GENERATION_LITERATURE_AUDIT_ZH.md)。

## 一句话结论

本轮实验发现了一个稳定但不能直接变成静态加权方法的机制：

> DINOv2 RAE 中，encoder 跨层更可预测的低方差方向，同时被小型 stage-2 在高噪声阶段优先利用，
> 也被冻结 decoder 赋予更高视觉敏感度；但是某个样本在某个时刻的准确 decoder 修正方向高度依赖当前状态，
> 不能由任何常数 channel metric 近似。

因此现在不应训练新的静态 weighted-MSE、SPC 或 predictability curriculum。真正尚未解决的问题是：能否用明显低于完整
decoder 的代价，得到输入相关的 perceptual correction。

## 1. Frozen decoder rollout 检查

沿用五个 static 小模型、相同 held-out latent、相同噪声、相同 50-step ODE 和三组近等方差方向。每个 seed 使用
16 张图，只在 rollout 终点加载冻结 RAE decoder。图像参考是 `D(E(x))`，不是原始像素，因此不会混入固定的
RAE 重构误差。

原预注册门槛失败：三组中只有两组满足高噪声图像比值高于低噪声图像比值。但失败揭示了更清楚的分解。

| pair | endpoint gain ratio, t=0.30 | endpoint gain ratio, t=0.95 | decoder L1 secant ratio, t=0.30 | decoder L1 secant ratio, t=0.95 |
|---|---:|---:|---:|---:|
| fractional-0 / absolute-2 | 0.760 | 2.633 | 3.013 | 1.083 |
| fractional-2 / absolute-5 | 0.895 | 3.757 | 1.158 | 1.005 |
| fractional-2 / PCA-6 | 0.834 | 4.373 | 1.436 | 1.050 |

这里 `decoder L1 secant = decoded L1 shift / latent endpoint RMS shift`。

三组都满足同一个探索性“接力”模式，而且每项都是 `5/5` seed：

- 低噪声时，高可预测方向在 ODE 中传播得更弱，endpoint gain ratio 小于 1；
- 但 decoder 对该方向的单位终点位移更敏感，secant ratio 大于 1；
- 高噪声时，ODE endpoint gain ratio 变成大于 1，decoder secant 接近 1；
- 最终 decoded L1 与 LPIPS 在三个时刻、三组配对中全部仍大于 1。

因此 end-to-end 视觉 leverage 一直偏向高可预测方向，但主导来源随噪声变化：高噪声主要来自 denoiser dynamics，
低噪声主要来自 decoder geometry。

## 2. 直接 decoder 因果验证

上面的 endpoint 差分可能在 rollout 中旋转，所以又在 32 个未见 clean `E(x)` 上直接施加逐样本等范数扰动：

- 子空间：三组近等方差 pair；
- latent RMS：`0.5%` 和 `1%`；
- 正负两个符号；
- 冻结 decoder、fp32、关闭 TF32；
- 同时量 pixel MSE、L1、AlexNet LPIPS 和四个 decoder hidden 层。

1% 扰动、正负平均的结果：

| pair | pixel MSE gain ratio | L1 secant ratio | LPIPS secant ratio | early hidden ratio | late hidden ratio |
|---|---:|---:|---:|---:|---:|
| fractional-0 / absolute-2 | 4.150 | 2.316 | 3.127 | 4.569 | 3.157 |
| fractional-2 / absolute-5 | 1.472 | 1.174 | 1.576 | 1.732 | 1.729 |
| fractional-2 / PCA-6 | 2.696 | 1.496 | 2.905 | 2.739 | 2.765 |

结论对 `0.5%/1%` 和正负号稳定。样本级 paired bootstrap 中，所有 L1/LPIPS 95% CI 都完全高于 1。
所以 decoder 各向异性是真实的局部性质，不是 rollout 旋转造成的假象。

## 3. 24-block atlas：方差还是跨层可预测性

把直接 decoder probe 扩到 absolute、fractional、PCA 三个家族各 8 个互不重叠 rank-16 block。basis 只在
ImageNet train 1024 图上拟合，predictability 在另一批 256 图上评估；decoder probe 使用更后面的 32 个未见
ImageNet train latent，没有数据泄露。

对 `log(decoder sensitivity)` 回归：

```text
standardized log variance + standardized held-out cross-layer R2
```

| decoder quantity | variance-only R2 | predictability-only R2 | combined R2 |
|---|---:|---:|---:|
| input linear embed | 0.005 | 0.366 | 0.617 |
| pixel MSE gain | 0.175 | 0.667 | 0.670 |
| L1 secant | 0.226 | 0.627 | 0.628 |
| LPIPS secant | 0.114 | 0.705 | 0.732 |
| hidden 1 | 0.177 | 0.819 | 0.830 |
| hidden 2 | 0.169 | 0.860 | 0.878 |
| hidden 3 | 0.182 | 0.859 | 0.873 |
| hidden 4 | 0.156 | 0.830 | 0.851 |

该关系不是 basis 家族标签造成的：去掉任意一个家族后 predictability 系数仍为正；在每个家族内部，predictability
与 decoder 指标的 Spearman 约为 `0.90-1.00`。

decoder 第一层线性投影的方向增益范围只有 `1.47x`，而第一个记录 hidden 层已经达到约 `11.4x`。所以主要的
语义选择性不是 normalization 或 `decoder_embed` 的简单缩放，而是在 decoder Transformer 的早期层中形成并持续保留。

## 4. 为什么不能立刻做静态 weighted loss

24-block atlas 是总体方向敏感性，不代表逐样本 gradient。为避免再次把相关性误读成训练方法，又在从未用于 basis
拟合的 ImageNet-1K validation latent 上做了直接 proxy gate：

- 32 张图计算 loss 排序；
- 8 张图 x 3 个噪声时刻计算完整 decoder LPL gradient；
- predictability metric 只使用 fractional basis 的 held-out R2；
- oracle metric 直接使用前述 decoder atlas 的 hidden sensitivity，属于故意偏向 proxy 的上界诊断。

| proxy | median time Spearman with full LPL | median gradient cosine |
|---|---:|---:|
| latent MSE | 0.066 | 0.013 |
| predictability static metric | 0.065 | 0.021 |
| decoder-atlas oracle static metric | 0.036 | 0.029 |
| decoder prefix 1 | 0.182 | 0.117 |
| decoder prefix 2 | 0.420 | 0.234 |
| decoder prefix 3 | 0.728 | 0.493 |

静态 metric 的门槛是 `Spearman >= 0.80`、`gradient cosine >= 0.70`，predictability 与 oracle 都远未通过。
更重要的是 oracle 也失败，说明问题不是静态权重估计得不够准，而是 full LPL 的修正方向确实依赖样本、token 和时间。

decoder prefix 3 明显更好，但它已执行 decoder 大部分深度，高噪声 gradient cosine 仍只有 `0.229`，不构成一个
足够便宜、足够准确的新方法。

## 5. 理论解释

现有证据支持把 RAE 生成链理解成两个层次：

1. **总体语义轴层次**：跨 encoder 层持续存在的方向更可能包含稳定对象信息；stage-2 在高噪声时优先利用它们，
   decoder 也让它们对图像有更高影响。
2. **逐样本局部相位**：同一个总体重要子空间中，具体哪个 token、哪个符号、哪个组合应该被修正，取决于当前
   `z0_hat` 的位置。常数二次型只能描述第一层，不能描述第二层。

这解释了此前看似矛盾的结果：

- SPC 压低高可预测方向会变差，因为它删掉了高噪声阶段真正被信任的信号；
- decoder atlas 又显示这些方向视觉上重要；
- 但静态 weighted MSE 仍不等于 decoder perceptual loss，因为总体轴重要性不等于逐样本梯度方向。

## 6. 与已有工作的边界

[Boosting Latent Diffusion with Perceptual Objectives](https://arxiv.org/abs/2411.04873) 已在 ICLR 2025 提出用
decoder 中间特征定义 latent perceptual loss，并报告生成收益。因此“给 diffusion 加 decoder hidden loss”不是本文的新方法。

[Latent Diffusion Inversion Requires Understanding the Latent Space](https://arxiv.org/abs/2511.20592) 已从 decoder
pullback metric 讨论 latent 维度的非均匀影响。仅声称 decoder 几何各向异性也不够新。

当前实验可能有独立价值的部分是三者的对齐与断层：

- 跨 encoder 层可预测性可解释 decoder 的总体方向谱；
- 同一量也解释 stage-2 高噪声方向信任；
- 但它无法压缩逐样本 decoder gradient。

这更像一项机制发现。要成为方法论文，必须提出真正输入相关且显著低成本的 correction，并在生成质量上超过
标准 MSE 和完整 LPL 对照。

## 7. 当前研究决定

现在明确禁止：

- 再训练静态 predictability-weighted MSE；
- 调 SPC floor/power/rank；
- 把 decoder atlas oracle 当作可训练 metric；
- 把 prefix-3 当成“便宜 decoder”，因为节省不足且梯度仍不够准。

只保留两个合理后续方向：

1. **机制论文方向**：跨 DINOv2/MAE/SigLIP2 和不同 decoder 重复 atlas，验证“encoder persistence -> denoiser trust ->
   decoder salience”是否具有普遍性。
2. **方法论文方向**：研究输入相关的低秩 Jacobian sketch 或 token-conditioned early-exit predictor；先要求在未见样本上
   达到 full LPL gradient cosine `>=0.70`，再允许一条单 seed 5k 训练。

在达到该门槛前，不再花费生成训练算力。

## 8. 产物

- 五 seed frozen-decode rollout：
  `~/data/eqvae/experiments/rae_spc_multiseed_v1/evaluation/latent_trust_decoder_spotcheck_v1/`
- clean-latent direct decoder secant：
  `~/data/eqvae/experiments/rae_spc_multiseed_v1/evaluation/decoder_subspace_secant_v1/`
- 24-block decoder atlas：
  `~/data/eqvae/experiments/rae_spc_multiseed_v1/evaluation/decoder_subspace_atlas_v1/`
- predictability alignment 回归：
  `~/data/eqvae/experiments/rae_spc_multiseed_v1/evaluation/decoder_predictability_alignment_v1/`
- static/dynamic LPL proxy gate：
  `~/data/eqvae/experiments/rae_spc_multiseed_v1/evaluation/predictability_lpl_proxy_gate_v1/`
