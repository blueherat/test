# Prior-Decoder 机制的外部模型验证方案

日期：2026-07-22

## 1. 为什么必须做外部验证

当前 `16/64/256d x 5 seeds` 的 Imagenette-64 实验只能支持下面这个有限结论：

> 在我们自己构造的小型两阶段生成系统中，增加 latent 容量会改善真实 latent 的 decoder
> 上限，但也会扩大 learned prior 采样后的图像质量差距。

它不能单独说明：

- 该现象能外推到公开生成模型；
- 问题普遍来自 prior-decoder 接口；
- 高维 latent 天生更难生成；
- 我们观察到的是新现象。

尤其要注意，[DC-AE 1.5](https://arxiv.org/abs/2508.00413) 已明确报告：增加 latent
channel 可以改善重构质量，却会使 diffusion 收敛更慢、gFID 变差。因此我们的容量权衡
首先应被称为“小型受控复现”，而不是新发现。

下一步应使用别人训练好的、规模高于当前 toy、但四张 RTX 4090 可以完成推理的公开模型。
暂时不重新训练大型 encoder、decoder 或 image diffusion prior。

## 2. 候选模型结论

### 2.1 第一优先：DiffAE FFHQ-128

官方资源：

- [论文与项目页](https://diff-ae.github.io/)
- [官方代码与 checkpoint](https://github.com/konpatp/diffae)

结构与当前问题最接近：

```text
image -> semantic encoder -> z_sem (512d)
Gaussian -> latent DDIM -> generated z_sem
(z_sem, pixel noise x_T) -> conditional DDIM decoder -> image
```

优点：

- encoder、learned latent prior 和 stochastic decoder 都有公开实现；
- 官方提供 FFHQ-128、FFHQ-256、Bedroom-128 和 Horse-128 的完整采样 checkpoint；
- 可以固定同一份 decoder noise，只替换 `z_sem`，干净地观察 prior latent 如何改变输出；
- 官方说明 latent DPM 的训练只需一张 2080 Ti，推理远低于本机四张 4090 的上限。

注意：官方名称中的 `72M/130M` 是训练所见样本数，不应误写成模型参数量。代码中 FFHQ
semantic latent 的宽度是 512。

限制：

- FFHQ checkpoint 的训练代码默认使用整个 FFHQ LMDB，不能把同分布 FFHQ audit 宣称为
  严格 held-out 泛化；
- decoder 还有随机子码 `x_T`，必须固定它，否则 latent 差异会和 decoder 随机性混在一起。

这是最适合先落地的外部验证。

### 2.2 第二优先：D2C FFHQ-256

官方资源：

- [论文项目页](https://d2c-model.github.io/)
- [官方代码与 checkpoint](https://github.com/jiamings/d2c)

结构：

```text
image -> NVAE-style encoder -> z
Gaussian -> latent diffusion -> generated z
z -> deterministic decoder logits -> mixture-logistic pixel sample
```

官方 FFHQ-256 配置中的 latent shape 是 `8 x 32 x 32`，即 8192 个连续标量。它比我们
当前的 256d 向量更大，又仍然能在消费级 GPU 上推理。

它的价值是提供一个与 DiffAE 不同的对照：

- DiffAE 是 `512d global semantic code + stochastic decoder`；
- D2C 是 `8192d spatial code + non-iterative decoder`。其卷积 decoder logits 是确定的，
  但官方 `latent_to_image` 会从 mixture-logistic 像素分布采样，因此仍需固定并重复输出 seed。

若两者都出现“raw latent 差异不突出，但 decode 后差距明显”，外部效度会比继续增加自建
toy 的 seed 强得多。

限制：代码基于 PyTorch 1.9/CUDA 10.2，必须使用独立环境；官方仓库主要支持 inference，
不适合在此基础上重新训练大模型。

### 2.3 第三优先：DC-AE f32/f64 + UViT-H

官方资源：

- [官方代码和模型表](https://github.com/mit-han-lab/efficientvit/blob/master/applications/dc_ae/README.md)
- [DC-AE 论文](https://arxiv.org/abs/2410.10733)
- [DC-AE 1.5 论文](https://arxiv.org/abs/2508.00413)

这是更接近真实 ImageNet latent diffusion 的验证：

```text
ImageNet image -> DC-AE encoder -> spatial latent
Gaussian -> UViT-H prior -> generated spatial latent
latent -> deterministic DC-AE decoder -> image
```

可比较的公开配对是：

| tokenizer | 512px latent shape | latent 标量数 | prior |
| --- | ---: | ---: | --- |
| DC-AE-f32c32 | `32 x 16 x 16` | 8192 | UViT-H, 500.87M |
| DC-AE-f64c128 | `128 x 8 x 8` | 8192 | UViT-H, 500.87M |

这组对照尤其有价值：两者 nominal scalar count 相同，prior 架构和参数量也相同，主要改变
的是 latent 的空间/通道组织，而不是简单的“维数更多”。官方表中 f32 与 f64 的 ImageNet
512 生成性能已有差异；DC-AE 1.5 还直接把“重构继续改善而生成变差”归因于 latent 结构
造成的 convergence difficulty。

本机 `/data/shared/imagenet-1k` 已有完整的 294 个 train、14 个 validation、28 个 test
Parquet shard，总计约 156 GB，可以使用 validation 做严格的 empirical-latent audit。

限制：

- UViT-H 是 500.87M 参数，不适合作为第一项集成；
- 该 decoder 是确定性的，只能验证 endpoint mismatch，不能验证 stochastic decoder 的
  时间责任曲线；
- 先做 5k 样本筛查，只有结果有信息量时才做 50k FID。

### 2.4 备选：CompVis Latent Diffusion

[官方 LDM model zoo](https://github.com/CompVis/latent-diffusion) 提供 CelebA-HQ、FFHQ、
LSUN Bedroom、LSUN Church 和 ImageNet 的 autoencoder 与 latent diffusion checkpoint。
它成熟且容易复现，但同一数据集缺少干净的多容量配对，因此更适合作为第三方 endpoint
sanity check，不适合作为容量权衡主实验。

## 3. 暂不优先的模型

### FlowMo

[官方仓库](https://github.com/kylesargent/FlowMo) 已公开 tokenizer checkpoint，但本次核查
发现 FlowMo-Lo 单个 checkpoint 约 7.56 GB，且仓库没有配套发布可直接采样的 latent prior。
它适合重构研究，不适合当前的闭环 prior-decoder 对照。

### DiTo

[项目页](https://yinboc.github.io/dito/) 给出了 ImageNet-256 结果和 162.8M/338.5M/620.9M
三档 diffusion decoder，但截至本次核查没有找到项目官方可运行代码和 checkpoint 链接。

### SWYCC

[论文](https://arxiv.org/abs/2409.02529) 与问题高度相关，但没有找到可直接复用的官方代码
和权重。

### PiD

[PiD](https://github.com/nv-tlabs/PiD) 有完整代码和权重，但 released decoder 面向 2K/4K，
模型和上游 backbone 都明显超过“中等规模”，并且四步蒸馏、超分辨率和 backbone 条件会
引入额外混杂。它应作为最后的高分辨率 stress test，而不是第一项外部验证。

## 4. 统一实验协议

### 4.1 所有模型都测的三路 marginal

```text
Empirical: z = E(x_data)
Prior:     z = learned_prior(noise)
Gaussian:  z = matched Gaussian baseline
```

对每一路同时测：

- latent mean/covariance、SWD、energy distance、C2ST；
- decoder feature mean/covariance、SWD、C2ST；
- 相对同一 reference set、同一特征提取器、同一样本数的 FID/KID；
- `FID(decoded prior, decoded empirical)`，直接衡量接口两侧的输出分布差异。

`End-to-end FID - Oracle FID` 只能作为同设置下的诊断差值。FID 不是可加性距离，不能把
这个差值解释成严格的误差分解。

### 4.2 shuffle 的正确用途

对 marginal FID 来说，empirical latent 只做排列不会改变分布，因此 shuffle FID 没有
信息。shuffle 只用于 paired test：

```text
D(z_i)        与 x_i 的配对质量
D(z_perm(i))  与 x_i 的配对质量
```

它回答 decoder 是否使用样本级 latent，而不是回答 prior marginal 是否匹配。

### 4.3 DiffAE 的额外控制

对同一个样本索引固定 decoder noise `x_T`，只替换：

```text
real z_sem
prior z_sem
matched-Gaussian z_sem
shuffled z_sem
```

同时报告：

- fixed-noise paired LPIPS/identity/feature displacement；
- 多个 `x_T` seed 上的均值和方差；
- real-to-prior 差异相对 decoder 自身随机方差的比例。

否则无法分清是 prior mismatch，还是 stochastic decoder 自己在变化。

## 5. 执行顺序

### Gate A：DiffAE 2k 样本

- FFHQ-128 完整 checkpoint；
- 4 个固定 decoder-noise seed；
- empirical/prior/Gaussian 三路 marginal；
- real/prior/shuffle 的 paired decoder test；
- 先用 KID、SWD、C2ST 和置信区间，不用 2k FID 下强结论。

通过条件：prior 与 empirical 的 latent 差异经过 decoder 后稳定放大，且放大量明显超过
decoder seed 方差。

### Gate B：D2C 5k 样本

- 使用官方 FFHQ-256 checkpoint；
- 复用相同 marginal 和 paired 协议；
- 验证现象是否依赖 stochastic decoder。

通过条件：至少方向上复现 DiffAE 结果，或者给出清楚的 architecture-dependent 反例。

### Gate C：DC-AE ImageNet 5k

- f32c32 与 f64c128；
- 同一个 UViT-H 家族、同样采样步数和 CFG；
- ImageNet validation empirical latent；
- 同时报告官方 rFID/gFID 和本地统一评估。

只有 5k 结果显示稳定差异时，再用四卡生成 50k 样本。

## 6. 验收与停止标准

可以把“小系统现象具有外部效度”写入结论，至少需要：

1. DiffAE 与 D2C/DC-AE 中至少两个独立系统方向一致；
2. 结论在 matched sample count、matched feature extractor 和多个采样 seed 下稳定；
3. decoder 后的差异不能仅由 latent RMS、单一 covariance mismatch 或采样器步数解释；
4. empirical latent 明显优于 prior latent，且 shuffle 证明 decoder 使用样本级信息；
5. 结果同时报告反例，不筛选只支持假设的模型。

出现以下结果时，应停止把它推广为一般 prior-decoder mismatch：

- DiffAE 的差异不超过 decoder 自身随机方差；
- D2C 和 DC-AE 上 raw latent 指标已经充分解释 decoded gap；
- 不同公开系统的方向互相矛盾，且能由架构差异直接解释；
- 只有训练集 empirical latent 好，严格 validation latent 不复现。

## 7. 当前推荐

先集成 **DiffAE FFHQ-128**，再集成 **D2C FFHQ-256**。这两项都比当前 toy 明显更大，
但仍是消费级 GPU 可以控制的实验，并且分别覆盖 stochastic 与 deterministic decoder。

DC-AE f32/f64 放在第三项。它的价值最高但成本也更高，而且外部论文已经提供了强烈的
容量/结构权衡证据；我们本地运行它的目的应是统一协议复核，而不是重新发现论文已经报告
的现象。

## 8. DiffAE Gate A 实际结果

执行日期：2026-07-22。

### 8.1 资产与完整性

- 官方配置：`ffhq128_autoenc_latent`。
- 官方 checkpoint：`last_bak.ckpt`，SHA-256
  `fe952738038549d9528086cb8ed5e0c697b50dc6ca84590703d69e77a21f30f9`。
- 官方 latent 统计：`latent.pkl`，SHA-256
  `fc8ef438454b3a91ef058ce0d218ef0a6085eacc5dc6d441601a01dcd7a66c72`。
- checkpoint `global_step=394531`，恰好对应配置中的约 1.01 亿 latent-DPM 训练样本。
- EMA 分支共有 `686` 个张量、`192,110,467` 个参数；严格检查确认 encoder、
  latent prior、decoder input blocks 和 decoder output blocks 均存在，没有 missing 或
  unexpected key。
- FFHQ-128 共 `70,000` 张 RGB PNG，编号 `00000-69999`，分辨率全部为 128×128。

本实验没有训练任何参数。模型全程 `eval()`、`requires_grad=False`、FP32，关闭 TF32。
Google Drive 中完整生成 checkpoint 的实际文件名就是 `last_bak.ckpt`；它不是未完成下载文件。

### 8.2 协议

- 样本数：2,000。
- 官方评估步数：pixel DDIM `T=10`，latent DDIM `T_latent=10`。官方仓库随附记录在该
  设置下报告 EMA FID `20.6346`；本地不把 2k 指标冒充官方 FID 复现。
- latent 三路：官方 empirical code、官方 learned latent-DPM code、全协方差匹配
  Gaussian code。
- 四个 decoder-noise seed：`30260722-30260725`。
- 每个 seed 内三路使用完全相同的 pixel-noise 随机流；四个 seed 共享同一组 latent。
- 图像指标：官方 FID Inception feature 上的 KID、标准化 SWD、线性 logistic C2ST。
- 真实参考：同一组固定的 2,000 张 FFHQ-128 图像。

### 8.3 Latent 端结果

| source | standardized SWD | mean shift | covariance relative error | effective rank | linear C2ST acc. | linear C2ST AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| learned prior | 0.1650 | 0.0460 | 0.3401 | 114.93 | 0.6500 | 0.6886 |
| matched Gaussian | 0.0417 | 0.0336 | 0.2875 | 151.36 | 0.4950 | 0.4903 |
| empirical reference | 0 | 0 | 0 | 151.66 | - | - |

有限样本下 512 维 covariance error 本身会偏大，因此不应单看该列。更稳定的事实是：
learned prior 的 effective rank 明显下降，且可被线性 C2ST 区分；全协方差 Gaussian 的
effective rank 与 empirical 几乎一致，线性 C2ST 处于随机水平。

### 8.4 Decoder 后结果

四个 decoder seed 的均值与 seed 间标准差：

| source vs real FFHQ | KID | standardized SWD | linear C2ST acc. | linear C2ST AUC |
| --- | ---: | ---: | ---: | ---: |
| empirical decode | 0.02845 ± 0.00089 | 0.23094 ± 0.00546 | 0.97125 ± 0.00188 | 0.99625 ± 0.00075 |
| learned-prior decode | 0.03375 ± 0.00067 | 0.25458 ± 0.01119 | 0.98625 ± 0.00327 | 0.99864 ± 0.00058 |
| matched-Gaussian decode | 0.04991 ± 0.00136 | 0.29595 ± 0.01775 | 0.99089 ± 0.00179 | 0.99959 ± 0.00022 |

learned prior 相对 empirical 的 KID gap 为：

```text
0.00448, 0.00559, 0.00537, 0.00575
mean = 0.00530, seed std = 0.00057, 4/4 为正
```

它相当于 empirical-decode KID 的约 `18.6%`。Gaussian 的平均 KID gap 为 `0.02146`，
约为 empirical-decode KID 的 `75.4%`，同样 `4/4` 为正。

直接比较 decoded empirical 与 decoded prior：

- KID：`0.00592 ± 0.00019`；
- standardized SWD：`0.12358 ± 0.00517`。

同一 latent 只更换 decoder noise 时，跨 seed 分布 KID 为 `-0.00020`，即无偏估计器的
零附近；empirical 的跨 seed SWD 为 `0.02943`。因此 empirical-vs-prior 的直接 SWD
约为 decoder-noise baseline 的 `4.20x`。逐样本 Inception cosine distance 中，更换 latent
来源约为更换 empirical decoder noise 的 `5.05x`。

### 8.5 当前结论

本次外部验证支持以下有限结论：

> 在别人训练好的完整 DiffAE FFHQ-128 系统中，learned latent prior 与 empirical encoder
> latent 之间存在可测量分布差距；该差距在冻结 stochastic decoder 后形成稳定、跨四个
> pixel-noise seed 同向的图像分布质量 gap。

这说明此前自建 Imagenette 系统中的 **prior-decoder interface gap 并非只由我们自己的小模型
实现造成**。但是，它还不证明：

- latent 维数越高就必然越难生成；DiffAE 这里只测了一个 512 维系统；
- decoder 在数学上“放大”了 latent 距离；两侧指标没有共同单位；
- stochastic decoder 是 gap 的原因；需要 D2C 或 DC-AE 的确定性 decoder 对照；
- 2k KID 等价于官方 50k FID。

此外，本地真实参考使用公开的 FFHQ 128 thumbnails，而官方训练管线从 FFHQ-256 LMDB 经
`torchvision.transforms.Resize(128)` 得到输入；两条下采样链可能存在轻微像素差异。因此绝对的
`real-vs-decode` 数值只用于同一参考下的相对比较，不能替代官方 FID。empirical、prior 和
Gaussian 三路共享同一真实参考，这个限制不改变四 seed 的相对 gap 方向。

本实验最有信息量的反常点是 matched Gaussian：它的均值、全协方差、effective rank 和
线性 C2ST 都比 learned prior 更接近 empirical latent，但 decoder 后明显更差。这说明
**二阶统计匹配和线性不可分不足以保证 decoder compatibility**。learned prior 尽管有 rank
收缩，仍明显优于 Gaussian，说明它学习到了重要的非高斯、高阶 latent 支持结构；剩余问题是
它没有完全覆盖 decoder 熟悉的 empirical latent 分布。

因此下一项外部实验优先使用 D2C，复用完全相同的 empirical/prior/Gaussian 协议。该实验
现已完成；D2C 的结果见下一节。需要校正的是，D2C 的卷积 decoder logits 确定，但官方
接口仍会从 mixture-logistic 像素分布采样，并非完全确定性 decoder。

## 9. D2C Gate B 实际结果

### 9.1 实现与完整性

- 官方仓库：`jiamings/d2c`，commit `ae3b0b9957599dab66343b564058c6967271d15c`；
- 官方 FFHQ-256 checkpoint：`model.ckpt`，`674,668,843` bytes，SHA256
  `1dcd0ec51873767c859625b1e907174a147096ef41d1380aafd64d419c5788b9`；
- checkpoint 严格覆盖推理所需的 autoencoder、latent diffusion 和 latent modulation；
- 推理实际使用 `95,200,736` 个参数，latent shape 严格为 `8 x 32 x 32`；
- 模型全部冻结，`eval()`、FP32、TF32 关闭，无 optimizer、无 backward、无训练；
- official prior 使用仓库 README 的 `skip=100` DDIM 采样设置；
- FFHQ-256 图像来自 `merkol/ffhq-256` 的前两个 Parquet shard；
- 取前 2000 张只拟合 Gaussian，再取后 2000 张作 empirical latent 与 real reference，
  两部分严格不重叠；
- 四个输出 seed 为 `30260722/23/24/25`。每个 seed 内三路 latent 使用相同 batch 划分，
  并在每路开始前重置 CPU/CUDA RNG，保证 mixture-logistic 使用相同随机数流。

这里还修正了预注册时的一个架构判断：D2C 不是 image diffusion decoder，但官方
`latent_to_image` 会对 mixture-logistic 输出采样。因此它比 DiffAE 少了迭代式 pixel
diffusion，却仍有较小的最终像素采样随机性，必须保留跨 seed 控制。

### 9.2 Latent 结果

由于 D2C latent 有 8192 维，Gaussian 通过 empirical 数据矩阵因子采样，精确表示拟合集的
样本协方差，不显式构造 `8192 x 8192` 矩阵。协方差误差与有效秩使用 1024 个固定样本的
Gram 恒等式；SWD 和 C2ST 对两路候选复用相同子样本与随机投影。

| candidate | SWD | mean shift | covariance error | effective rank | projected C2ST acc | AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| empirical fit split | 0.0549 | 0.0374 | 0.2772 | 139.57 | 0.4784 | 0.4850 |
| learned prior | 0.1681 | 0.0630 | 0.2993 | 83.78 | 0.5830 | 0.6081 |
| matched Gaussian | 0.0601 | 0.0483 | 0.2997 | 128.65 | 0.5439 | 0.5578 |
| empirical eval reference | 0 | 0 | 0 | 139.86 | - | - |

learned prior 再次出现明显 rank contraction。更重要的是，matched Gaussian 的 SWD 几乎
贴着“两批 empirical latent 自身”的有限样本基线，均值、协方差、有效秩和线性可分性也
明显比 learned prior 更接近 empirical。

### 9.3 Decoder 后结果

2000 样本、四个 output seed 的均值与 seed 标准差如下：

| latent source | real-reference KID | real-reference SWD | C2ST acc | AUC |
| --- | ---: | ---: | ---: | ---: |
| empirical decode | 0.020214 +/- 0.000119 | 0.20025 +/- 0.00089 | 0.97321 +/- 0.00343 | 0.99608 +/- 0.00065 |
| learned-prior decode | 0.031511 +/- 0.000069 | 0.27886 +/- 0.00015 | 0.99411 +/- 0.00036 | 0.99962 +/- 0.00006 |
| matched-Gaussian decode | 0.198526 +/- 0.000055 | 0.70572 +/- 0.00069 | 1.00000 +/- 0 | 1.00000 +/- 0 |

learned prior 相对 empirical 的 KID gap 为：

```text
0.011297 +/- 0.000128，4/4 seed 为正
```

它相当于 empirical-decode KID 的约 `55.9%`。Gaussian 的 KID gap 为
`0.178312 +/- 0.000077`，约为 empirical-decode KID 的 `8.82x`，同样 `4/4` 为正。

直接比较 decoded empirical 与 decoded prior：

- KID：`0.023524 +/- 0.000065`；
- SWD：`0.235933 +/- 0.001219`。

decoded empirical 的跨 seed distribution KID 为 `-0.000239`，无偏估计下可视为零附近；
跨 seed SWD 为 `0.020926`。因此 empirical-vs-prior 的直接 SWD 是输出随机基线的
`11.27x`。Gaussian 对应比例为 `30.66x`。paired feature cosine 也给出同方向控制，但
empirical/prior 样本没有语义配对关系，因此该数值不能解释为逐样本质量指标。

### 9.4 当前结论

D2C Gate B 通过，而且证据强于 DiffAE Gate A：

> 在官方预训练 D2C FFHQ-256 系统中，learned latent prior 与 empirical encoder latent
> 存在稳定分布差异；该差异经过非迭代式 decoder 后形成 4/4 seed 一致的图像质量 gap，
> 远大于 decoder 最终像素采样自身的随机波动。

这说明 prior-decoder interface gap 不是我们自建小模型的 bug，也不是 DiffAE 的迭代式
stochastic decoder 特例。它同时出现在 `512d global latent + diffusion decoder` 和
`8192d spatial latent + convolutional decoder` 两种明显不同的系统中。

最有信息量的反例是 matched Gaussian：它在简单 latent 指标上已经接近 empirical split
的抽样基线，却在 decoder 后严重崩坏。这表明 decoder compatibility 依赖 empirical latent
的非高斯高阶结构或非线性支撑集；匹配均值、协方差、有效秩和线性投影仍远远不够。learned
prior 虽然出现 rank contraction，却比 Gaussian 更会生成合理人脸，说明它确实学到了一部分
关键高阶结构，只是覆盖仍不完整。

仍不能把结果扩大成以下结论：

- 不能说 decoder 在统一度量下“数学放大”了 latent 距离；latent 与 Inception 指标单位不同；
- 不能说 8192 维必然导致 prior 难学；这里只验证了一个固定容量的 D2C；
- 不能把 2k KID 当成官方 50k FID；
- fit/eval 图像虽然互不重叠，但 D2C 原模型用整个 FFHQ 训练，不能宣称模型层面的 held-out；
- 两个外部系统已经支持“可重复机制候选”，但还不足以宣称所有两阶段生成模型都如此。

下一步不必立刻训练新模型。更高价值的是在 D2C 上做等距离的 on-manifold 与 off-manifold
latent 干预，直接定位 decoder 对哪些高阶方向敏感；若还需要第三个外部系统，再使用
DC-AE/ImageNet 的确定性 decoder 做严格 validation 对照。
