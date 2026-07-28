# 紧凑语义 Tokenizer 与生成鲁棒 Decoder 文献调研

> 调研日期：2026-07-20
>
> 研究范围：方向 2（紧凑语义 latent）为主，方向 3（decoder / generated-latent 接口）为辅
>
> 目标设备：4 张 RTX 4090，优先复用公开 checkpoint，避免从零训练大型 tokenizer 或 DiT

## 1. 先给结论

方向 2 值得继续，而且比我们此前强行在 RAE latent 中寻找全局群表示更贴近生成模型。但是，最朴素的版本已经很拥挤：

- 把 DINOv2 等视觉基础模型的高维 patch feature 压到 32/64/96 通道，已有 AlignTok、FAE、PS-VAE、RPiAE、GAE 等多条路线。
- 用语义监督训练普通低维 VAE，已有 VA-VAE、MAETok、VFM-VAE、GAE 等路线。
- 把二维网格进一步压成少量一维 token，甚至单 token，已有 TiTok、FlexTok、RepTok、SemTok 等路线。
- 让 decoder 适应扩散模型产生的、不完全位于真实编码流形上的 latent，已有 RAE noise augmentation、l-DeTok、LV-RAE、GAE 等路线。

因此，论文问题不应再写成“把 RAE 压到 32 通道会不会更好”。这个问题基本已经有人回答。更值得做的是：

> **在固定的小型阶段二生成器预算和端到端采样预算下，语义核心、空间细节与 decoder 生成能力之间应该如何分配容量？**

这是方向 2 和方向 3 的交界处。它不仅问 latent 能压多小，还问压掉的信息由谁补回来，以及总计算是否真的减少。

我的具体建议是先做一个低成本、可证伪的比较：

1. `raw RAE grid`：`16x16x768`，高维、语义强、确定性 decoder。
2. `compact semantic grid`：例如 GAE 的 `16x16x32`，低通道网格。
3. `single semantic token`：RepTok，一个 token，但使用生成式 decoder 恢复细节。
4. `robust conventional latent`：l-DeTok，检查 decoder 鲁棒性本身能贡献多少。

冻结 tokenizer，使用相同数据、近似相同阶段二参数量与训练 FLOPs，比较收敛速度、5K FID/KID、重构上限和**包含 decoder 在内的端到端成本**。只有这个比较出现稳定、可解释的机制后，才值得训练我们自己的 compact tokenizer。

## 2. 方向 2 到底在解决什么

图像 tokenizer 同时承担两个相互冲突的任务：

- 为 reconstruction 保存足够的像素、纹理、位置和颜色信息。
- 为 generation 提供维度较低、分布规整、容易预测的目标。

RAE 选择了前者：直接保留 DINOv2、MAE、SigLIP2 的高维表征，再训练 decoder。原始 RAE 明确指出，这些 latent 通常是高维的，因此 DiT 需要专门的宽 DDT head、训练策略和 noisy decoder 才能有效工作。[RAE 论文](https://arxiv.org/abs/2510.11690)

紧凑语义 tokenizer 的核心想法是：

```text
图像 x
  -> 语义 encoder / teacher feature h
  -> 有损压缩器 A
  -> 紧凑 latent z
  -> decoder 恢复 feature 或图像
```

它与我们此前的可逆 adapter 有本质区别：

- 可逆 adapter 只是换坐标，信息维数不变，也没有消除高维 latent 的自由度。
- 紧凑 tokenizer 是有损压缩。它必须明确舍弃一部分信息，并通过语义先验、decoder 或条件信息补偿。

### 2.1 “压缩”至少有四个不同维度

设 latent 为 `N x C`：

- `N`：token 数量或空间分辨率，例如 `16x16=256`。
- `C`：每个 token 的通道数，例如 768、96、32。
- 分布约束：KL、RMSNorm、VQ、球面约束或其他正则。
- decoder 条件计算：确定性一次解码，还是需要多步 diffusion / flow matching。

只看 `N*C` 会漏掉重要成本。例如 RepTok 的 `N=1` 极小，但它把大量像素细节恢复工作交给了 flow-matching decoder；这不等于细节被一个 token 无损保存，也不等于端到端采样免费。

### 2.2 压缩没有创造信息，只会重新分配信息负担

现有方法主要把被压掉的信息转移到四处之一：

- **语义教师**：DINOv2 等先验决定哪些信息优先保留。
- **强 decoder**：由 decoder 根据紧凑条件生成纹理和局部细节。
- **外部条件**：例如 TexTok 用文本提供语义，使图像 token 更专注于细节。
- **分层残差**：前几个 token/channel 保存结构，后面的保存细节。

所以正确的问题不是“谁的 latent 最小”，而是“在相同总成本与相同任务下，信息被放在哪里最划算”。

## 3. 文献谱系总览

| 路线 | 代表工作 | latent 形态 | 主要机制 | 主要成本转移 |
|---|---|---:|---|---|
| 原始高维 RAE | RAE、RAEv2 | 高维 patch grid | 保留 frozen VFM feature | 阶段二生成器和 decoder |
| 压缩 VFM feature | AlignTok、FAE、PS-VAE、RPiAE | 常见为 `16x16x32~96` | adapter / feature AE / KL | feature decoder、pixel decoder |
| 语义监督低维 AE | VA-VAE、VFM-VAE、GAE | 低通道网格 | VFM 作为 teacher | tokenizer 预训练 |
| 少量或单 token | TiTok、FlexTok、RepTok、SemTok | 1D token 或单 token | 信息排序、生成式 decoder | decoder 推理和大规模 tokenizer 训练 |
| 阶段二容量感知 | CRT、MAETok、DC-AE 1.5 | 离散或连续 | 让表示适配小生成器 | 有意牺牲一部分重构 |
| decoder 鲁棒性 | l-DeTok、LV-RAE、GAE | 多种 latent | corruption/noise/smoothing | decoder 训练与推理 |

这几条路线名称相近，但解决的问题不同。报告余下部分分别讨论。

## 4. 第一组：直接压缩视觉基础模型表征

### 4.1 AlignTok：先对齐，再有限度微调 encoder

[AlignTok](https://arxiv.org/abs/2509.25162) 是 ICLR 2026 工作。它从 DINOv2-L/14 的高维 feature 出发，通过 adapter 得到紧凑 latent，默认使用 32 通道，并训练 CNN decoder。

训练分三阶段：

1. 冻结视觉 encoder，只训练 adapter 和 decoder，建立可重构的语义 latent。
2. 联合优化 encoder、adapter、decoder，同时用语义保持损失约束 latent 不偏离第一阶段的语义结构。
3. 固定 encoder 和 adapter，只精修 decoder。

论文在 ImageNet 256 上报告 64 epochs 的 guided gFID 1.90；其价值在于证明“低维 + 语义”可以比 raw RAE 更快地供 diffusion 学习。

对我们的启发：

- frozen encoder 不是必须永远 frozen；有限度微调可以补回重构细节。
- 但三阶段流程、感知损失和 GAN 训练并不便宜。
- 单纯复现“DINO feature -> 32 channel MLP”缺少新颖性。

### 4.2 FAE：编码端一层足够，整个系统并不只有一层

[FAE / One Layer Is Enough](https://arxiv.org/abs/2512.07829) 是 CVPR 2026 工作，将 frozen 视觉 encoder feature 压成 `16x16x32`。压缩端只使用一个 self-attention layer 和线性层，因此标题强调“一层”。

但完整系统还有：

- 一个 6 层 Transformer feature decoder，用来恢复原始 VFM feature。
- 一个较深的像素 decoder，先在带噪原 feature 上训练，再适应恢复后的 feature。
- 紧凑 latent 上的 KL 约束和 feature reconstruction。

论文在 ImageNet 256 上报告 80 epochs 的 FID 2.08（无 CFG）/1.70（CFG），800 epochs 为 1.48/1.29。

关键判断：

- FAE 有力支持“高维 VFM feature 中存在可供生成使用的低维语义核心”。
- 它也说明压缩器可以小，但 decoder 端不能被忽略。
- 对 4 张 4090 来说，从零复制完整训练不划算；更适合把它当方法设计基线。

### 4.3 PS-VAE：语义压缩与像素重构需要联合平衡

[PS-VAE](https://arxiv.org/abs/2512.17909) 把 frozen DINOv2-B feature 通过 semantic VAE 压到默认 `16x16x96`，先保持语义，再让像素重构梯度有限地更新 encoder。

它强调 raw RAE latent 与 decoder 可解码流形不完全匹配，采用两阶段训练：

1. 训练带 KL 的 semantic encoder/decoder，压缩并恢复 DINO feature；像素 decoder 使用 detached latent。
2. 联合微调，使像素 reconstruction 梯度进入 encoder，同时保留 semantic anchor 和 KL。

论文的通道 sweep 显示，增加 latent 通道不一定持续改善 generation。约 96 通道是其具体 generator、decoder 和训练配置下的经验平衡点，不是普适常数。

对我们的意义：

- 应同时报告 reconstruction、semantic preservation 和 generation，不能用一项代替另外两项。
- “compact semantic latent”不是简单 PCA；压缩器与 decoder 的共同适应非常重要。
- 项目页目前注明代码仍在 legal review，因此短期内不适合作为主复现底座。[PS-VAE 项目页](https://jshilong.github.io/PS-VAE-PAGE/)

### 4.4 RPiAE：允许 encoder 改变，但用 frozen pivot 防止语义漂移

[RPiAE](https://arxiv.org/abs/2603.19206) 不再完全冻结初始化于 DINOv2 的 encoder，而是保留一个 frozen replica 作为语义 pivot。可训练 encoder 在重构目标下改变，pivot 正则防止它丢掉原有语义。

其 variational bridge 也执行低维压缩，并配合较深的 feature decoder 和像素 decoder。论文报告 ImageNet 80 epochs FID 2.25（无 CFG）/1.51（CFG）。

它回答的是一个重要问题：

> frozen VFM 的语义与可重构性冲突时，应该完全冻结，还是允许受约束的微调？

对有限资源实验，pivot 思想可以保留，但完整三阶段大规模训练仍较重。

### 4.5 GAE：低维、语义监督、分布规整和 decoder noise 一起做

[GAE](https://arxiv.org/abs/2603.10365) 是 2026 年 3 月的预印本，目前不能按正式会议论文的成熟度评价。它不是直接把 raw RAE feature 当 latent，而是训练一个紧凑像素 autoencoder，并用 frozen DINOv2-L 的压缩 feature 作为语义监督。

三个关键设计：

- 用专门的 semantic downsampler 构造低维 VFM target。
- 用 RMSNorm/单位超球面式约束替代标准 KL。
- 对 decoder 使用动态噪声，使其适应生成 latent 的偏差。

官方 repo 已发布 32 通道 GAE 和 80/800 epoch LightningDiT checkpoint，并报告 80 epochs 无 CFG gFID 1.82、800 epochs 1.31。[GAE 官方代码](https://github.com/sii-research/GAE)

它与我们拟议的方向 2+3 重叠最强：低维语义 latent 和 decoder robustness 都已经同时出现。因此它应当作为必须比较、最先复现的强基线，而不是被忽略后重新发明。

## 5. 第二组：不是压 raw VFM，而是让普通 tokenizer 获得语义

### 5.1 VA-VAE

[VA-VAE](https://arxiv.org/abs/2501.01423) 对普通 VAE latent 加入视觉 foundation model 对齐，包括逐样本语义和样本间关系。它保留了传统低维 VAE 的生成便利性，同时让 latent 更有语义。

这条路线与 FAE 不同：VFM 是监督信号，不是待压缩的原始 latent。它更容易接入现有 LDM，但语义信息能否完整进入小 bottleneck 取决于训练目标。

### 5.2 MAETok

[MAETok](https://arxiv.org/abs/2502.03444) 用 masked autoencoding 和语义目标训练 tokenizer，强调“重构更好”不等于“生成更好”。它观察到更有语义、更少模式的 token 分布可以让 diffusion 更容易学习。

这与我们此前 RAE 实验的经验一致：latent 的 raw reconstruction 或某个局部几何指标，不能单独预测 gFID。阶段二模型实际看到的分布复杂度才重要。

### 5.3 VFM-VAE 与 VTP

[VFM-VAE](https://arxiv.org/abs/2510.18457) 把 frozen VFM 的多尺度信息融合进 VAE，针对 semantic dynamics 设计约束；它代表“保持低维 VAE 接口，但增强语义”的系统路线。

[VTP](https://arxiv.org/abs/2512.13687) 则把理解、重构和图文对齐放入统一的大规模预训练。它说明长期趋势是在 tokenizer 预训练阶段融合更多任务，但训练规模与数据要求明显超出我们的首轮预算。[VTP 官方代码](https://github.com/MiniMax-AI/VTP)

对我们的现实判断：若目标是低成本验证机制，应先做 frozen-teacher compact AE，不应一开始做统一多任务预训练。

## 6. 第三组：从少量 token 到单 token

### 6.1 TiTok：把二维图像 token 化成一维序列

[TiTok](https://arxiv.org/abs/2406.07550) 使用 ViT tokenizer，把图像压成 32/64/128 个一维离散 token。它的重要贡献是证明图像生成不必始终在二维 latent grid 上进行。

但它没有使用 VFM 语义先验，因此它主要回答“token topology 能否改变”，而不是“如何压缩 DINOv2 feature”。

### 6.2 FlexTok：token 按信息重要性排序

[FlexTok](https://arxiv.org/abs/2502.13967) 通过 nested dropout 训练 1 到 256 个有序离散 token，前部 token 表示语义和粗结构，后部 token 补充细节；其 decoder 使用 rectified flow。

它提供一个关键思想：latent 容量不一定固定，可以是逐步增加的语义到细节序列。但高质量配置的 tokenizer 和 generator 都较大，不适合我们直接从零复现。

### 6.3 RepTok：整张图只生成一个连续语义 token

[RepTok](https://arxiv.org/abs/2510.14630) 是 ICLR 2026 工作。它从 SSL ViT 的 `[CLS]` token 出发，轻微微调 encoder，同时用 cosine anchor 保持原有语义几何。每张图最终只有一个连续 token。

像素恢复由 conditional flow-matching decoder 完成。因此：

- 阶段二 latent generator 可以非常小，甚至使用无 self-attention 的 MLP-Mixer。
- 一个 token 只提供全局条件，局部纹理主要由生成式 decoder 建模。
- 整体系统把计算从“生成一张 latent grid”转移到“条件生成图像细节”。

官方已提供 ImageNet checkpoint 和 reconstruction/generation notebook，适合我们直接做低成本复现。[RepTok 官方代码](https://github.com/CompVis/RepTok)

RepTok 对我们尤其重要，因为它把研究问题暴露得最清楚：

> latent generator 和 decoder 都可以生成不确定性；算力有限时，应该把不确定性放在哪一边？

### 6.4 SemTok 与 semantic-prefix 路线

[SemTok](https://arxiv.org/abs/2603.16373) 使用一维离散语义 token，并用较强的生成/精修 decoder 恢复图像。它进一步支持“语义 token + 生成式 decoder”的趋势，但系统较大、训练长，短期只适合作为概念参考。

[SMAP](https://openreview.net/forum?id=7257Oiv0hZ) 探索 semantic prefix 和可裁剪的后续 token，使 token 按信息顺序排列。其评审状态与成熟度仍需持续跟踪，不宜作为当前结论的唯一依据。

### 6.5 WeTok：离散压缩也在使用生成式 decoder

[WeTok](https://openreview.net/forum?id=QteJJF57yG) 是 ICLR 2026 工作。它用分组 lookup-free quantization 提高离散 token 的容量，并在 decoder 侧引入额外噪声变量进行生成式细节恢复。

它不是 VFM 语义压缩方法，但提供了一个重要对照：强压缩后使用生成式 decoder 并非连续 latent 独有。若我们的结论只在 RepTok 上成立，可能是单个实现的现象；若连续和离散 tokenizer 都出现相同的“generator 变轻、decoder 变重”规律，机制才更可信。

## 7. 第四组：阶段二容量决定什么 latent 才算好

### 7.1 CRT：更差的重构可能带来更好的生成

[When Worse is Better / CRT](https://arxiv.org/abs/2412.16326) 的核心结论是：tokenizer 的最优压缩率取决于阶段二生成器的容量和训练计算。

在阶段二预算不足时，小 codebook 虽然 reconstruction 更差，却可能明显改善 gFID；当阶段二接近充分训练后，大 codebook 才可能反超。论文还通过 causal regularization 有意让 token 更适合自回归建模，即使 rFID 略差。

这是我们路线最重要的理论与实验依据：

- “最佳 latent”不存在脱离 generator budget 的绝对定义。
- 小模型实验不是大模型的劣化替代品，它可能处在不同的最优压缩区间。
- 所有比较必须固定或明确扫描阶段二预算，不能只比较 tokenizer rFID。

### 7.2 DC-AE 1.5：让 channel 按结构到细节排序

[DC-AE 1.5](https://arxiv.org/abs/2508.00413) 通过随机 prefix channel mask，使较早 channel 保留物体结构，后续 channel 补充细节，并让 diffusion 额外学习部分 channel latent。

这说明通道并非必须等价。对我们有两个启发：

- compact latent 可以由“固定 32 通道”改成“可变语义核心 + 可选细节尾部”。
- 但论文显示结构化方法在较高通道数更有优势，不能预设在 16/32 通道同样成立。

[DC-Gen 官方代码](https://github.com/dc-ai-projects/DC-Gen) 已公开，可用于核对实现。

## 8. 方向 3：decoder 为什么是方向 2 绕不过去的一半

### 8.1 真实编码 latent 与生成 latent 不同分布

decoder 通常只在 `z=E(x)` 上训练；生成器产生的 `z_hat` 即使均方误差不大，也可能偏离 decoder 熟悉的流形。于是会出现：

- latent 指标看似不错，但 decode 出现条纹或局部崩坏。
- reconstruction 很好，但 gFID 受少量 latent 偏差放大。
- 压缩越强，decoder 需要补全的信息越多，问题越明显。

这正是我们此前 RAE 条纹、adapter 后 generation 变差等现象背后的共同接口问题之一。

### 8.2 l-DeTok：直接训练 decoder 去掉 latent 扰动

[l-DeTok](https://arxiv.org/abs/2507.15856) 在 tokenizer 训练时对 latent 加入较强 corruption，让 decoder 从受损 latent 恢复干净图像。论文比较 additive Gaussian noise、interpolative noise 和 masking，发现合适的 latent denoising 可同时帮助 diffusion 与 autoregressive generator。

它告诉我们：decoder robustness 不是事后补丁，而可以是 tokenizer 设计的一部分。官方已发布代码与 checkpoint，适合直接作为方向 3 基线。[l-DeTok 官方代码](https://github.com/Jiawei-Yang/DeTok)

### 8.3 LV-RAE：显式区分语义 latent 和低层残差

[LV-RAE](https://arxiv.org/abs/2602.08620) 针对 RAE reconstruction，把 semantic latent 与 low-level residual 结合，并通过 noise fine-tuning 和 inference latent smoothing 改善 decoder 对偏离流形 latent 的敏感性。[LV-RAE 官方代码](https://github.com/modyu-liu/LVRAE)

这条路线与“语义核心 + 可选细节 residual”十分接近，因此任何新方法都必须与它区分：我们若继续，应关注**固定总生成预算下由谁建模 residual**，而不是只提出 residual 本身。

### 8.4 生成式 decoder：DGAE、DiTo、RepTok、FlexTok

[DGAE](https://arxiv.org/abs/2506.09644) 和 [DiTo](https://arxiv.org/abs/2501.18593) 都说明，当 bottleneck 极紧时，确定性 decoder 未必是正确工具；conditional diffusion/flow decoder 可以把一对多的细节恢复显式建模。

这会带来一个真实 trade-off：

- latent generator 更简单、更快。
- decoder 更慢，可能需要多次函数评估。
- 如果只报告 latent generator FLOPs，会误判系统效率。

因此方向 2 的任何实验都必须同时报告 decoder 参数量、吞吐、显存、NFE 和端到端采样时延。

### 8.5 频率与尺度也是 decoder 接口的一部分

[Improving the Diffusability of Autoencoders](https://arxiv.org/abs/2502.14831) 从频谱角度分析 latent，认为过强的高频成分会干扰 diffusion 的 coarse-to-fine 生成过程，并通过 decoder 的 scale-equivariance 正则改善生成。论文在 ICML 2025 发表。

它与我们此前的频域实验有直接联系，但也划定了新颖性边界：仅仅观察 latent 频谱病态或加入频率正则已经不新。更有价值的是研究**紧凑语义 latent 改变了哪些频率负担，以及这些负担是在 generator 还是 decoder 中被处理**。

## 9. 重要近邻：不压 latent 的替代方案

研究方向 2 时，必须回答“为什么不保留 raw RAE，然后改 generator”。以下工作就是替代路线。

### 9.1 REPA：不改 tokenizer，直接教 diffusion 学语义

[REPA](https://arxiv.org/abs/2410.06940) 将 diffusion transformer 的 noisy hidden state 对齐到 frozen 视觉 encoder 的 clean representation。它不压 VAE latent，也不训练 semantic tokenizer，而是降低阶段二模型自己学习语义表征的负担。论文为 ICLR 2025 Oral。

若 compact latent 的优势能被 raw latent + REPA 以相同成本追平，就不能把收益归因于压缩本身。因此 Phase B 至少需要一个轻量 REPA 控制组。

### 9.2 RAEv2 与 Scaling RAE：高维 latent 路线仍在快速变强

[Scaling RAE](https://arxiv.org/abs/2601.16208) 在大规模 text-to-image 上继续使用高维 RAE，并发现某些 ImageNet 阶段的复杂设计在规模扩大后不再必要；这说明 raw RAE 并没有因 compact tokenizer 出现而失效。

[RAEv2 / Improved Baselines with RAEs](https://arxiv.org/abs/2605.18324) 使用最后若干 encoder 层的和改善 reconstruction，并把 RAE 与 REPA 结合。论文报告 80 epochs ImageNet-256 gFID 1.06，但这是其完整大模型配置的自报结果，不能直接外推到我们的小模型。

这两项工作使比较门槛更高：我们至少应比较“raw RAE + 当前最佳小模型训练配方”，不能只比较最早版本的 RAE。

### 9.3 DecQ：给 raw RAE 增加少量细节 query

[DecQ](https://arxiv.org/abs/2605.22777) 从 VFM 中间层提取少量 detail-condensing queries，与原 patch token 一起生成。它用 8 个额外 query 补充 frozen final representation 丢失的细节，论文报告额外计算约 3.9%。

DecQ 与“语义核心 + 细节 residual”非常接近，只是它保留 raw RAE 主体并增加少量细节 token。因此我们的可变容量方案必须将 DecQ 视为直接竞争者，而不是只与固定 32-channel grid 比较。

### 9.4 DC-AE：也可以优先压空间分辨率

[DC-AE](https://arxiv.org/abs/2410.10733) 不是语义 tokenizer，而是用 residual autoencoding 和分阶段适配把空间压缩率提高到 64/128。它提醒我们，降低阶段二成本不只有“压 channel”这一种方式，还可以减少 token 数。

所以实验轴最好同时包含 `N` 和 `C`，否则得出的“低通道最好”可能只是没有比较高空间压缩率。

## 10. 关键论文比较表

下表只用于建立方法关系。论文中的 FID 受 generator、训练 epoch、CFG、采样步数和评测实现影响，不能横向直接排名。

| 工作 | 状态 | 表征来源 | 典型 latent | decoder | 公开实现 | 对 4x4090 的用途 |
|---|---|---|---|---|---|---|
| RAE | 技术报告 | frozen DINO/MAE/SigLIP | 高维 grid | 确定性 ViT | 有 | raw baseline |
| AlignTok | ICLR 2026 | DINOv2-L，可微调 | `16x16x32` | CNN | 未见完整官方代码 | 方法对照 |
| FAE | CVPR 2026 | frozen VFM | `16x16x32` | feature decoder + pixel decoder | 未见官方代码 | 机制参考 |
| PS-VAE | ICML 2026 | DINOv2-B，可微调 | `16x16x96` | feature + pixel decoder | legal review | 方法对照 |
| RPiAE | 预印本 | trainable encoder + frozen pivot | 低维 grid | ViT pixel decoder | 未见官方代码 | pivot 思想 |
| GAE | 预印本 | VFM semantic teacher | 32/64 channel grid | noise-robust AE decoder | 有 checkpoint | 第一优先实测 |
| RepTok | ICLR 2026 | SSL `[CLS]` | 单连续 token | flow-matching decoder | 有 checkpoint | 第一优先实测 |
| l-DeTok | ICLR 2026 | 常规 tokenizer | 常规紧凑 latent | corruption-robust decoder | 有 checkpoint | decoder 对照 |
| DC-AE 1.5 | ICCV 2025 | pixel AE | ordered channels | 确定性 decoder | 有 | 分层容量对照 |
| LV-RAE | 预印本 | RAE + residual | semantic + residual | noise/smoothing | 有 | RAE decoder 对照 |
| RAEv2 | 预印本 | 多层 VFM feature | 高维 grid | 确定性 decoder | 有 | raw latent 强对照 |
| DecQ | 预印本 | RAE + 中间层细节 | grid + 少量 query | 确定性 decoder | 未见完整官方代码 | 细节 residual 对照 |
| WeTok | ICLR 2026 | pixel tokenizer | 分组离散 token | 生成式 decoder | 需核对发布状态 | 离散对照 |

## 11. 哪些说法现在已经不能作为论文新颖性

以下单点本身不足以支撑新的 ICLR 论文：

- “把 DINOv2 的 768/1024 维 feature 压成 32 维”。
- “用一层 attention 或两层 MLP 当 adapter”。
- “在 compact latent 上加 KL”。
- “重构损失和语义保持损失联合训练”。
- “给 decoder latent 加 Gaussian noise”。
- “前几个 channel/token 存语义，后面存细节”。
- “把整张图压成一个 token”。
- “rFID 更低，所以更适合生成”。

这些可以作为组件或 baseline，但不能成为核心 claim。

## 12. 仍可能有价值的研究缺口

### 12.1 最推荐：固定总预算下的容量分配规律

核心问题：

> 对同一数据和同一总训练/推理预算，生成不确定性应由 latent generator 建模，还是由 decoder 建模？

可控变量：

- token 数 `N`：1、16、64、256。
- channel 数 `C`：16、32、64、768。
- decoder 类型：确定性、latent-denoising、少步 flow。
- generator 容量：tiny/small/base。
- 总预算：阶段二 FLOPs + decoder FLOPs/NFE。

潜在论文价值来自一个可复现的 crossover：

- 小 generator 更适合强压缩、强 decoder。
- generator 变大后，保留更多空间细节的 grid latent 反超。
- 最优点由总预算而不是 rFID 决定。

CRT 已在离散 tokenizer 上给出类似容量依赖，因此我们的贡献必须聚焦**连续语义 latent 和 decoder 成本共同计入后的规律**，并最好由此导出一个简单方法。

### 12.2 可变语义核心 + 条件细节 residual

设 latent 为：

```text
z = [z_semantic, z_detail]
```

- 小预算时只生成 `z_semantic`，decoder 随机补细节。
- 中预算时额外生成部分 `z_detail`。
- 大预算时生成完整 spatial residual。

这与 DC-AE 1.5、FlexTok、LV-RAE 接近。只有当我们能证明“按 decoder 风险或阶段二可预测性排序”优于现有 channel/token ordering，才有新颖性。

### 12.3 生成器误差感知的 tokenizer

普通 tokenizer 训练关注 `E(x)` 的 reconstruction；真正部署时 decoder 接收的是 `z_hat`。可让 tokenizer/decoder 关注实际 generator 误差方向：

```text
z_hat = z + delta_generator
D(z_hat) -> x
```

与 l-DeTok 的通用随机 corruption 相比，研究点是：小生成器的误差是否集中在特定语义/空间子空间，针对这些误差训练 decoder 是否能以更少 corruption 获得更好端到端质量。

这与我们仓库已有的 decoder-risk、latent-trust 和 rollout 诊断最自然衔接，也比重新训练全套 tokenizer 更省资源。

## 13. 推荐执行路线

### Phase A：官方 checkpoint 审计，不训练大模型

目标：确认比较接口、成本与指标，预计数天而不是数周。

对象：

- 当前 RAE-DINOv2 baseline。
- GAE 32-channel 官方 autoencoder。
- RepTok 官方 ImageNet checkpoint。
- l-DeTok 官方 tokenizer。

统一数据：ImageNet-1K validation 的固定 1K 样本，另保留 4K 作为最终 audit。官方 tokenizer 使用 ImageNet train 不构成 validation 泄露，但不能用最终 4K 反复选超参数。

统一记录：

- latent shape、dtype、每图字节数。
- reconstruction FID/KID、LPIPS、PSNR。
- 线性 probe 或 kNN 语义保持。
- latent 插值、局部扰动、decoder Jacobian 风险 proxy。
- decoder 参数量、NFE、吞吐、峰值显存和单图 wall time。
- tokenizer encode 时间与阶段二可缓存成本。

Phase A 验收：

- 四个 checkpoint 都能使用各自官方 preprocessing 正确重构。
- 在同一机器上得到可重复的端到端 timing。
- 论文报告的 reconstruction 指标在合理误差内可复核。
- 明确 RepTok 的优势中有多少来自阶段二简化，有多少成本被转移给 decoder。

### Phase B：固定预算的小型阶段二比较

先缓存 compact latent；raw DINOv2-B latent 不建议全量 fp32 缓存。

存储量粗估：

- `16x16x768` fp32：每图约 0.75 MiB，ImageNet 全量接近 1 TB。
- `16x16x32` fp32：每图约 32 KiB，全量约 42 GB。
- compact latent 可保留 fp32；raw latent 使用分片、按需编码或经验证后以 fp16 缓存。

比较两种口径：

1. **同 backbone**：尽量用同一个 tiny DiT/MLP，测 representation 本身的难度。
2. **同总计算的合理模型**：允许单 token 用 MLP-Mixer、grid 用 DiT，但固定训练 FLOPs和参数预算。

最小实验：

- 数据：ImageNet train，固定 3 个 seed。
- generator：约 50M、100M 两档，而不是直接上 675M。
- 训练：先用固定 step 数和样本数，不以 epoch 模糊比较。
- 评测：每个 checkpoint 先做 5K FID/KID；FID 只作筛选，KID 用于小样本不确定性。
- 记录：生成器训练 FLOPs、decoder FLOPs/NFE、总采样时间。

Phase B 进入下一阶段的门槛：

- 至少 3 个 seed 方向一致。
- 固定训练 FLOPs 下，compact 方案的 5K FID/KID 或收敛速度改善至少约 10%。
- 加入 decoder 后，端到端收益仍然存在，或者在质量下降不超过 5% 时达到至少 2 倍速度。
- 不能只凭 train loss、latent MSE 或一组好看的样本晋级。

只有满足门槛的一个配置再做 50K FID。最终希望看到至少约 5% FID 改善，或清楚的质量-速度 Pareto 改善。

### Phase C：只在 Phase B 出现机制后训练自己的小模块

优先原型：

```text
frozen DINOv2-B feature
  -> compact encoder (linear / 1 attention)
  -> 16x16x{16,32,64}
  -> feature decoder
  -> frozen 或轻微适配的 RAE pixel decoder
```

控制实验：

- PCA/SVD 固定投影。
- 线性 `1x1` autoencoder。
- 一层 attention compact AE。
- KL vs RMSNorm。
- clean decoder vs interpolative corruption decoder。

先复用 frozen RAE pixel decoder，避免一开始训练大型 ViT decoder。若 feature reconstruction 好但 pixel decode 差，再只微调 decoder 的小输入桥或最后若干层。

Phase C 的核心验收不是“能重构”，而是：

- 同一 compact latent 在小 generator 下稳定优于 raw RAE。
- 优势不能被简单 PCA 或参数量增加解释。
- decoder corruption 只在真实 generator 误差方向上带来额外收益。
- 结果能导出一句清楚机制，而非组件堆叠。

## 14. 实验公平性与泄露控制

### 14.1 数据划分

- tokenizer 或 adapter 只使用 ImageNet train。
- 从 train 中固定划出一部分用于超参数选择。
- ImageNet validation 只用于预先约定的中间 audit 和最终报告，不根据最终 4K/50K 结果反复选模型。
- 官方 checkpoint 已在 ImageNet train 上训练，评估 validation 合理；需在报告中披露预训练数据。

### 14.2 必须统一或披露的变量

- encoder 输入分辨率、crop、归一化。
- latent scaling/whitening。
- generator 参数量、训练样本数、optimizer 和 learning-rate schedule。
- CFG 与否、采样器、steps/NFE。
- FID implementation 和 reference statistics。
- decoder 推理次数和端到端 wall time。

### 14.3 不能直接横比的数字

- ImageNet class-conditional gFID 与 text-to-image GenEval/DPG。
- guided FID 与 unguided FID。
- 80 epoch 与 800 epoch。
- 不同 DiT 大小或不同采样步数。
- rFID 与某些论文使用的 feature-decoder reconstruction 指标。

## 15. 资源判断

4 张 4090 适合：

- 官方 tokenizer/decoder checkpoint 的统一评测。
- frozen latent 提取和 compact latent 缓存。
- 50M 到约 100M 的小型阶段二模型、多 seed 短训练。
- 小 compact adapter、feature decoder 或 decoder 输入桥训练。

不适合首轮做：

- 从零复现 AlignTok/GAE 的 200 epoch ImageNet tokenizer。
- 直接训练 675M LightningDiT 800 epochs。
- 训练 RepTok 的大型 flow decoder。
- 同时扫描大量 latent 维度、decoder 和 generator 尺度。

算力限制并不一定是劣势。CRT 已提示：小 generator 对 tokenizer 的偏好可能与大 generator 不同。只要严格固定预算，我们研究的正是一个真实而未被充分系统化的问题。

## 16. 停止条件

出现以下情况应停止该分支：

- compact latent 仅降低 reconstruction，而不改善小 generator 的收敛或 FID/KID。
- 表面速度提升在加入 decoder NFE 后完全消失。
- 所有收益都可由更大的 generator 或更多训练 step 等价获得。
- 结果对 seed、preprocessing 或 FID 实现高度敏感。
- 新组件只是 FAE/GAE/l-DeTok 的直接组合，没有新的机制或 Pareto 改善。

反过来，最有价值的结果未必是“compact 永远最好”。如果观察到稳定 crossover，例如小预算下单 token 最好、中预算下 32-channel grid 最好、大预算下 raw RAE 反超，这本身就是清楚、可验证、贴近生成系统的研究结论。

## 17. 推荐阅读顺序

第一轮先建立问题：

1. [RAE](https://arxiv.org/abs/2510.11690)：为什么引入高维 semantic latent。
2. [When Worse is Better / CRT](https://arxiv.org/abs/2412.16326)：为什么 tokenizer 不能脱离阶段二预算评价。
3. [FAE](https://arxiv.org/abs/2512.07829)：为什么 frozen VFM feature 可以有损压成低通道 latent。
4. [RepTok](https://arxiv.org/abs/2510.14630)：压缩到单 token 后，信息和计算去了哪里。
5. [l-DeTok](https://arxiv.org/abs/2507.15856)：为什么 decoder 接口是生成质量的一部分。

第二轮理解竞争方法：

6. [AlignTok](https://arxiv.org/abs/2509.25162)
7. [PS-VAE](https://arxiv.org/abs/2512.17909)
8. [GAE](https://arxiv.org/abs/2603.10365)
9. [MAETok](https://arxiv.org/abs/2502.03444)
10. [DC-AE 1.5](https://arxiv.org/abs/2508.00413)
11. [LV-RAE](https://arxiv.org/abs/2602.08620)

第三轮再看更大系统：

12. [FlexTok](https://arxiv.org/abs/2502.13967)
13. [VFM-VAE](https://arxiv.org/abs/2510.18457)
14. [VTP](https://arxiv.org/abs/2512.13687)
15. [SemTok](https://arxiv.org/abs/2603.16373)

## 18. 最终建议

现在不应立刻设计一个新 compact adapter。先用公开 checkpoint 建立三点事实：

1. 在我们的 4x4090 和小生成器预算下，raw grid、compact grid、single token 的真实收敛差异是什么。
2. 将 decoder 成本算进去后，哪种表示仍然更高效。
3. 生成器实际误差与 decoder 风险是否集中在可识别的 latent 子空间。

若前两点出现稳定 crossover，论文主线可以是“budget-aware semantic latent allocation”；若第三点更强，则主线可以转成“generator-error-aware compact tokenizer/decoder”。二者都比单纯继续压通道更有机会形成一篇机制和方法统一的生成模型论文。

## 19. 主要一手来源

- [Diffusion Transformers with Representation Autoencoders](https://arxiv.org/abs/2510.11690)
- [AlignTok](https://arxiv.org/abs/2509.25162)
- [One Layer Is Enough / FAE](https://arxiv.org/abs/2512.07829)
- [FAE 的 Apple 官方论文页](https://machinelearning.apple.com/research/adapting-pretrained-visual-encoders)
- [PS-VAE](https://arxiv.org/abs/2512.17909)
- [RPiAE](https://arxiv.org/abs/2603.19206)
- [GAE](https://arxiv.org/abs/2603.10365)；[官方代码](https://github.com/sii-research/GAE)
- [RepTok](https://arxiv.org/abs/2510.14630)；[官方代码](https://github.com/CompVis/RepTok)
- [VA-VAE](https://arxiv.org/abs/2501.01423)
- [MAETok](https://arxiv.org/abs/2502.03444)
- [VFM-VAE](https://arxiv.org/abs/2510.18457)
- [TiTok](https://arxiv.org/abs/2406.07550)
- [FlexTok](https://arxiv.org/abs/2502.13967)
- [TexTok](https://arxiv.org/abs/2412.05796)
- [SemTok](https://arxiv.org/abs/2603.16373)
- [WeTok](https://openreview.net/forum?id=QteJJF57yG)
- [When Worse is Better / CRT](https://arxiv.org/abs/2412.16326)
- [REPA](https://arxiv.org/abs/2410.06940)
- [Scaling RAE](https://arxiv.org/abs/2601.16208)
- [RAEv2](https://arxiv.org/abs/2605.18324)
- [DecQ](https://arxiv.org/abs/2605.22777)
- [DC-AE](https://arxiv.org/abs/2410.10733)
- [DC-AE 1.5](https://arxiv.org/abs/2508.00413)；[官方代码](https://github.com/dc-ai-projects/DC-Gen)
- [Improving the Diffusability of Autoencoders](https://arxiv.org/abs/2502.14831)
- [Latent Denoising Makes Good Tokenizers / l-DeTok](https://arxiv.org/abs/2507.15856)；[官方代码](https://github.com/Jiawei-Yang/DeTok)
- [LV-RAE](https://arxiv.org/abs/2602.08620)；[官方代码](https://github.com/modyu-liu/LVRAE)
- [DGAE](https://arxiv.org/abs/2506.09644)
- [DiTo](https://arxiv.org/abs/2501.18593)
- [VTP](https://arxiv.org/abs/2512.13687)；[官方代码](https://github.com/MiniMax-AI/VTP)
