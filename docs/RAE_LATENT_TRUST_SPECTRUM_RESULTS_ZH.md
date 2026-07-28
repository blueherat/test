# RAE Latent Trust Spectrum：从失败的 SPC 到噪声依赖方向信任

## 一句话结论

当前 SPC 训练方法已经被五种子生成实验否定，但失败中提炼出了一个更可靠、也更贴近生成模型的现象：

> RAE stage-2 对 latent 方向的响应不是只由方差决定。低噪声时模型几乎按方向能量响应；高噪声时，
> “该方向能否从 encoder 中间层稳定预测到 final latent”会成为额外且越来越重要的信任信号。

这暂时是一个机制发现，不是一个已经有效的新方法。rollout 与 frozen-decoder 门槛都已完成；结果支持一个
denoiser-to-decoder 的方向 leverage 接力，但否定了静态 predictability-weighted loss。完整后续见
[`RAE_LATENT_TRUST_DECODER_ALIGNMENT_RESULTS_ZH.md`](RAE_LATENT_TRUST_DECODER_ALIGNMENT_RESULTS_ZH.md)。

## 1. 旧路线已经得到什么结论

原 SPC 使用 layer-9 到 final latent 的线性预测，取 predicted covariance 的 top eigenvectors 作为
rank-16 “detail” 子空间，并在训练前 2000 步的高噪声区域压低该子空间。

五个 seed 的 5000 张样本评估给出：

- 平均 FID 变差 `+6.30`，相对恶化 `4.85%`；
- paired bootstrap 95% CI 为 `[+1.29,+12.16]`；
- `0/5` seed 同时改善 FID 与 KID；
- closure 条件也失败。

因此当前 SPC 的 `floor/power/rank` 扫描应停止。它不是训练加速器，也没有改善生成。

## 2. 失败的根因：原 guided basis 几乎就是 top PCA

原准则最大化的是绝对可预测能量：

```text
Cov(predicted final) = W^T Cov(middle) W
```

它没有除以 final 方向自身的方差，因此天然偏向高方差方向。真实 held-out latent 上：

- guided rank-16 捕获 `13.58%` residual energy；
- top-16 PCA 捕获 `14.35%`；
- guided 达到 PCA 能量上界的 `94.65%`；
- guided 与 top-PCA 的平均平方 principal cosine 为 `0.820`。

更直接地说，原先所谓“可预测 detail”主要是“高方差且可预测”的主轴。SPC 在 `t=0.85` 把这些方向的
state SNR 从 `0.1322` 压到 `0.0063`，低于随机方向约三倍。它隐藏的不是最不可靠细节，而是当时最可见的信号。

## 3. 白化后真正分离了方差与可预测性

新诊断改为正则化广义特征问题：

```text
Cov(predicted final) v = lambda Cov(final) v
```

它最大化的是“该方向有多少比例可由中间层预测”，而不是绝对预测能量。所有 basis 和回归只使用原实验的
1024 张 ImageNet train 图像拟合，256 张 ImageNet validation 图像只用于评估，没有 train/validation 泄露。

rank-16 结果：

| basis | validation R2 | residual energy | 每维方差 | 与 top-PCA 重叠 |
|---|---:|---:|---:|---:|
| 原 guided | 0.827 | 13.57% | 4.257 | 0.869 |
| 白化可预测 | **0.931** | 5.25% | 1.647 | 0.198 |
| top-PCA | 0.778 | 13.89% | 4.357 | 1.000 |
| 随机方向均值 | 0.322 | 2.09% | 0.655 | 0.022 |

白化 basis 的 train/validation R2 差只有 `0.0065`，train 前后两半的 subspace overlap 为 `0.926`，
高于原 guided 的 `0.835`。所以它不是小样本过拟合，也确实不同于 PCA。

## 4. stage-2 到底利用哪种方向

使用五个 static stage-2 训练 seed 的 step-5000 online checkpoint，不再训练任何模型。对 128 个 held-out latent，
在每个噪声时刻加入逐样本等 Frobenius norm 的 rank-16 扰动，再测有限差分响应：

```text
directional gain = ||f(x_t + delta,t)-f(x_t,t)||^2 / ||delta||^2
```

对 24 个互不重叠的 rank-16 块，拟合：

```text
log(gain) <- standardized log(direction variance) + standardized validation R2
```

五种子均值如下：

| noise time | 方差 beta | 可预测性 beta | 方差单独 R2 | 联合 R2 |
|---:|---:|---:|---:|---:|
| 0.10 | 1.029 | -0.097 | 0.949 | 0.958 |
| 0.30 | 0.950 | 0.037 | 0.943 | 0.949 |
| 0.50 | 0.903 | 0.130 | 0.954 | 0.966 |
| 0.70 | 0.897 | 0.146 | 0.958 | 0.973 |
| 0.85 | 0.742 | 0.354 | 0.886 | **0.972** |
| 0.95 | 0.542 | 0.558 | 0.734 | **0.947** |

这里 `t` 越接近 1，输入越接近纯噪声。结果呈现连续变化：

- 低噪声阶段，方向方差几乎完全决定模型响应；
- 高噪声阶段，方差作用下降，跨层可预测性作用快速上升；
- `t=0.95` 时两者标准化系数已经接近相等；
- 联合指标在高噪声下比任一单指标都明显更完整。

## 5. 最关键的近等方差交叉

回归相关性仍可能被共线性误导，因此又选了三组每维方差相差不超过约 2% 的子空间直接配对。

最干净的一组：

- 白化 block：方差 `1.647`，validation R2 `0.931`；
- 绝对能量 block：方差 `1.637`，validation R2 `0.591`。

白化 block / 对照 block 的 gain 比值：

| time | gain ratio | 五种子方向 |
|---:|---:|:---:|
| 0.10 | 0.766 | 0/5 大于 1 |
| 0.30 | 0.867 | 0/5 大于 1 |
| 0.50 | 0.912 | 低于 1 |
| 0.70 | 0.915 | 低于 1 |
| 0.85 | **1.495** | 5/5 大于 1 |
| 0.95 | **3.920** | 5/5 大于 1 |

另外两组近等方差配对在 `t=0.85` 的比值分别为 `2.07` 和 `2.12`，在 `t=0.95` 为 `3.48` 和 `3.83`，
也都是 `5/5` seed 同向。

这是真正反常且有研究价值的点：**相同方向能量并不对应相同模型响应；噪声足够高时，跨 encoder 层更稳定的方向会被 stage-2 更强地利用。**

## 6. 当前最合理的通俗解释

在低噪声时，`x_t` 已经包含大量 clean latent 信息。一个方向本身变化越大，模型越需要对它做出大响应，
所以方差主导。

在高噪声时，多数方向都被噪声淹没。此时“方差大”不再保证其中是可靠信号。能从 encoder 中间层一路稳定延续到
final latent 的方向，更像对象结构或稳定表征，stage-2 会把它们当作较可信的少量线索。模型虽然从未看到 layer-9，
但训练数据的 final latent 几何让这种偏好自然出现在速度场里。

暂时把这个随时间变化的规则称为 **latent trust spectrum**：

```text
trust_t(direction)
  = beta_var(t) * log variance
  + beta_pred(t) * cross-layer predictability
```

这个名称只描述现象，不表示已经建立严格概率理论。

## 7. 能说什么，不能说什么

目前可以说：

- 原 SPC 失败的机制已经相当清楚；
- 白化可预测子空间稳定、泛化，并且不同于 PCA；
- stage-2 的方向响应存在可重复的噪声依赖交叉；
- 高噪声下，方差与跨层可预测性是互补描述量。

目前不能说：

- 白化 basis 能改善 FID；
- directional gain 就是无穷小 Jacobian 特征值；
- 高 gain 一定有益，或一定代表误差放大；
- 该规律已跨 RAE encoder、模型尺度和训练长度泛化。

当前 gain 是有限扰动响应；basis 只拟合过一个 1024-image train split；生成模型是小型 stage-2。half-split 和五训练 seed
解决了部分稳定性问题，但没有替代跨 encoder 验证。

## 8. 与已有工作的边界

- [Spectral Forcing](https://arxiv.org/abs/2606.15236) 已经说明可以按噪声时间保留信号频段、隐藏噪声频段；
- [Learning What Matters](https://openreview.net/forum?id=veL6ZAMFmb) 已经研究各向异性的频谱 forward noise；
- [Variational Trajectory Optimization](https://arxiv.org/abs/2602.19512) 覆盖更一般的矩阵值 schedule；
- [Rethinking the Noise Schedule](https://openreview.net/forum?id=ylHLVq0psd) 已从 signal power 与 WSNR 研究 schedule。

因此“各方向使用不同噪声/课程”本身不新。可能的独特点只能是：

1. 方向来自 pretrained representation 的跨层稳定性，而不是固定 DCT 或单纯 PCA；
2. 发现并定量刻画了 variance-dominant 到 predictability-sensitive 的噪声依赖交叉；
3. 该 trust spectrum 能预测真实 rollout 风险，并最终带来生成收益。

第三点尚未验证，所以现在还不到论文方法成立的阶段。

## 9. Teacher-path rollout 因果门槛

该门槛已经在现有五个小模型上完成，没有训练新模型：

1. 取相同 clean latent、noise、label，构造标准路径上的 `x_t`；
2. 在三组近等方差 basis 上加入相同范数扰动；
3. 从 `t={0.95,0.85,0.30}` 用相同 solver 继续积分到 clean endpoint；
4. 比较 endpoint latent error、decoder 后 L1/LPIPS，以及误差沿轨迹的放大曲线；
5. 使用两个扰动幅度，排除单一有限差分尺度偶然性。

实际使用 32 个 held-out latent、五个 stage-2 训练 seed、三种起始时间和两个扰动幅度。24 个 block
均使用逐样本完全相同的输入扰动范数。

### 9.1 一步 gain 能否预测终点 leverage

| teacher-path 起点 | one-step 与 endpoint Spearman | Pearson(log gain) |
|---:|---:|---:|
| 0.30 | 0.665 | 0.662 |
| 0.85 | **0.961** | **0.984** |
| 0.95 | **0.990** | **0.986** |

高噪声下，局部方向响应几乎完整保留到 rollout 终点；低噪声下相关性明显较弱。这说明前面的 gain
不是孤立的一步数值现象，而是能预测生成轨迹对不同 latent 方向的实际 leverage。

### 9.2 近等方差交叉是否保留

最大扰动幅度下，高可预测 block / 低可预测 block 的 endpoint shift-gain 比值：

| pair | t=0.30 | t=0.85 | t=0.95 |
|---|---:|---:|---:|
| fractional-0 / absolute-2 | 0.754 | 0.907 | **2.830** |
| fractional-2 / absolute-5 | 0.889 | **1.561** | **4.011** |
| fractional-2 / PCA-6 | 0.829 | **1.451** | **4.873** |

- `t=0.95` 三组配对均为 `5/5` seed 大于 1；
- `t=0.30` 三组配对均为 `0/5` seed 大于 1；
- 半幅扰动给出相同排序，排除了单一扰动尺度偶然性；
- 第一组 crossover 比另外两组更晚，在 `t=0.85` 仍略低于 1，因此不能声称所有方向共享同一个固定阈值。

### 9.3 高 leverage 是有用信号还是坏的不稳定性

扰动方向取负号，即从 teacher state 中减去对应的 clean-direction signal。高噪声下，移除高可预测方向
不仅造成更大的 endpoint shift，也通常造成更大的 clean endpoint MSE 增量：

- fractional-2 / absolute-5 在 `t=0.85/0.95` 的 clean-MSE 增量差为正，`5/5` seed 同向；
- fractional-2 / PCA-6 同样在 `t=0.85/0.95` 为正，`5/5` seed 同向；
- 第一组在 `t=0.95` 为正且 `5/5` 同向，在 `t=0.85` 尚未 crossover。

因此当前证据更支持：**高噪声下的高可预测方向是模型真正使用的信号锚点，移除它们会伤害恢复；它们不是
应该被压掉的坏捷径。** 这与原 SPC 失败形成闭环解释。

### 9.4 门槛判断

latent 端因果门槛通过，但 frozen decoder 的原预注册时间排序门槛失败：三组近等方差 pair 中只有两组保持
高噪声 decoded ratio 高于低噪声 ratio。进一步分解发现，低噪声排序由 decoder anisotropy 反转，高噪声排序由
rollout dynamics 主导；三组方向在 decoded L1/LPIPS 上始终保持高可预测方向影响更大。

直接 clean-latent decoder probe 和 24-block atlas 证实，跨层 predictability 对 decoder hidden sensitivity 的解释度
达到 `R2=0.82-0.86`，明显高于方向方差。但 predictability 静态 metric 对完整 decoder LPL 的逐样本梯度 cosine
只有 `0.021`，偷看 decoder atlas 的 oracle metric 也只有 `0.029`。因此不再允许反向 trust curriculum 或任何
静态 weighted-MSE 训练。

## 10. 结果位置

- basis 指标与图：`~/data/eqvae/experiments/rae_spc_multiseed_v1/evaluation/predictability_basis_v1/`
- 128-sample 五种子方向实验：`~/data/eqvae/experiments/rae_spc_multiseed_v1/evaluation/predictability_block_sensitivity_n128_v1/`
- 核心图：`gain_explanation_over_time.png`
- 近等方差交叉图：`matched_variance_gain_crossover.png`
- 可复算统计：`gain_explanation_per_seed.csv`、`matched_variance_pairs.csv`
- rollout 因果实验：`~/data/eqvae/experiments/rae_spc_multiseed_v1/evaluation/latent_trust_rollout_v1/`
- rollout 汇总图：`latent_trust_rollout_summary.png`
- frozen decoder 与 24-block atlas：见
  [`RAE_LATENT_TRUST_DECODER_ALIGNMENT_RESULTS_ZH.md`](RAE_LATENT_TRUST_DECODER_ALIGNMENT_RESULTS_ZH.md)
