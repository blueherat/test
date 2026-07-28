# RAE 生成机制文献审计与研究重定位

## 结论先行

当前实验中的若干**现象可靠**，但“encoder 跨层可预测方向是 RAE 生成中的统一信任轴”这一**机制解释还不可靠**，
更不足以直接作为论文主线。

最主要的原因不是样本数太少，而是存在三个尚未排除的替代解释：

1. `cross-layer predictability` 可能只是空间结构、线性可分性、数据流形切向方向或 decoder 训练信息分配的代理量；
2. 当前动态结果来自 `DiTDH-S`，主干宽度为 `384`，小于 DINOv2-B latent 通道数 `768`；虽然其 `2048`
   宽 DDT head 正是为缓解该瓶颈而设计，它仍属于原始 RAE 所讨论的低容量敏感区间；
3. decoder 各向异性、离流形敏感和 decoder-aware perceptual loss 已分别被 RAE、LPL、LV-RAE 和 latent
   geometry 工作覆盖，单独重复该现象不构成足够的新颖性。

因此现在应停止继续枚举 loss、权重和 schedule。先完成文献定位，再只做能够区分替代机制的实验。

## 一、必须先读的论文

### A. RAE 主线

| 论文 | 已回答的问题 | 对当前工作的影响 |
|---|---|---|
| [Diffusion Transformers with Representation Autoencoders](https://arxiv.org/abs/2510.11690) | 高维 RAE latent 的训练困难来自模型宽度、维度相关噪声调度和 decoder 的 clean/generated latent 分布差异；提出宽 DDT head、schedule shift 和 noise-augmented decoder。 | 当前小模型虽然有宽 DDT head，但主干仍是 `384 < 768` 的低容量设置，模型尺度是严重混杂；decoder OOD 并非新发现。 |
| [Scaling Text-to-Image Diffusion Transformers with Representation Autoencoders](https://arxiv.org/abs/2601.16208) | 在 0.5B--9.8B 模型上检验 RAE 设计；schedule shift 仍关键，但宽 head 和 decoder noise augmentation 的收益随模型变大而减弱。 | 说明小模型观察到的机制未必能外推到正式大模型；必须做跨尺度验证。 |
| [Improved Baselines with Representation Autoencoders](https://arxiv.org/abs/2605.18324) | RAEv2 聚合最后若干 encoder 层；区分 RAE 的全局语义作用与 REPA 的空间结构作用；80 epoch 达到 gFID 1.06。 | “中间层含有互补信息”和“层间结构影响生成”已进入强 baseline；我们的层间可预测性必须证明不是这两项的重述。 |

### B. 表征对齐与空间结构

| 论文 | 已回答的问题 | 对当前工作的影响 |
|---|---|---|
| [REPA](https://arxiv.org/abs/2410.06940) | 用 clean image encoder 表征监督 noisy DiT 中间层，显著加速生成训练。 | “高噪声阶段依赖 pretrained representation”已有强先验，不能仅凭方向敏感性称为新机制。 |
| [What Matters for Representation Alignment / iREPA](https://arxiv.org/abs/2512.10794) | 在 27 个 encoder 上发现，REPA 收益更受 patch token 空间自相似结构驱动，而非全局线性分类性能；空间归一化和卷积投影进一步改善训练。 | `cross-layer predictability` 可能在间接测量空间组织，不能直接解释为“稳定语义”。 |
| [Representation Entanglement for Generation](https://arxiv.org/abs/2507.01467) | 把全局语义 token 直接并入去噪过程，而不只在训练时做外部对齐。 | 已有工作明确区分全局语义与局部图像 latent；我们的指标需要同样拆分 token mean 与 spatial residual。 |

### C. Tokenizer 的生成友好性

| 论文 | 已回答的问题 | 对当前工作的影响 |
|---|---|---|
| [MAETok](https://arxiv.org/abs/2502.03444) | mask modeling 与语义 target 能让 latent 更可分、更易被 diffusion 建模；重构最好不等于生成最好。 | 层间可预测性也可能只是 discriminative organization 或较少分布模式的代理。 |
| [Reconstruction vs. Generation / VA-VAE](https://arxiv.org/abs/2501.01423) | 高维 latent 改善重构却提高 diffusion 优化难度；用视觉基础模型表征约束 VAE latent，可改善收敛和生成。 | “语义组织让 latent 更易生成”已有直接证据，不能用 predictability 相关性重复声明。 |
| [When Worse is Better](https://arxiv.org/abs/2412.16326) | tokenizer 的最佳压缩率依赖 stage-2 容量；更好的重构在小生成模型上可能反而更难建模。 | 当前 `DiTDH-S` 结果必须视为容量条件下的结果，不能外推为 tokenizer 固有性质。 |
| [Improving the Diffusability of Autoencoders](https://arxiv.org/abs/2502.14831) | 分析 latent DCT 高频能量，提出 decoder scale-equivariance，改善 coarse-to-fine 生成。 | 频域 FxLMS 启发中的“分频优化”已有非常接近的生成文献；仅做频带加权新颖性不足。 |
| [DC-AE 1.5](https://arxiv.org/abs/2508.00413) | 显式组织 channel，使前部通道承载结构、后部承载细节，并对结构通道增加 diffusion 目标。 | “方向重要性不同，应差异化训练”已有直接方法；必须证明我们的方向定义比结构/细节划分多提供了什么。 |
| [Latent Denoising Makes Good Tokenizers](https://arxiv.org/abs/2507.15856) | 直接训练 tokenizer 从强 latent corruption 中恢复 clean image，并在多种 diffusion/AR generator 上稳定改善生成。 | “让 decoder 或 tokenizer 对 latent 噪声更鲁棒”已经是强基线；普通各向同性噪声微调不是新方法。 |
| [Preconditioned Flow Matching](https://arxiv.org/abs/2603.02337) | 从理论和实验上说明中间分布协方差病态会导致不同方向收敛速度失衡，并用可逆预条件改善 FM。 | FxLMS、白化、可逆 flow 预条件这条路线已被直接覆盖，原有频谱 toy 只能作为复现或诊断。 |
| [Both Semantics and Reconstruction Matter / PS-VAE](https://arxiv.org/abs/2512.17909) | 直接把高维 RAE 的生成伪影归因于缺少紧凑正则的 off-manifold latent，并用 96 通道、KL 正则的语义-像素 VAE 改善生成与编辑。 | “RAE 有 off-manifold generation”已被明确提出；循环诊断只能提供更直接的实证或支持新的选择性干预。 |
| [RPiAE](https://arxiv.org/abs/2603.19206) | 微调 representation encoder 保留语义并补重构，再用 variational bridge 压缩 latent，分阶段优化生成易学性与重构。 | “保留语义、补像素、压缩 latent”的 tokenizer 改造也高度拥挤。 |
| [AlignTok](https://arxiv.org/abs/2509.25162) | 冻结 foundation encoder，训练 adapter/decoder，再联合微调并保持语义，最后单独优化 decoder。 | “encoder 后加 adapter、decoder 前补偿”的大方向已高度重叠，而且已被 ICLR 2026 接收。 |
| [VFM-VAE](https://arxiv.org/abs/2510.18457) | 使用 VFM 多层特征、专门 decoder，并以旋转、缩放和噪声下的 SE-CKNNA 诊断表征稳健性。 | 多层 RAE、语义保持和变换稳健性已经被组合研究；群/等变诊断本身不够形成论文。 |

### D. Decoder 几何

| 论文 | 已回答的问题 | 对当前工作的影响 |
|---|---|---|
| [Boosting Latent Diffusion with Perceptual Objectives](https://arxiv.org/abs/2411.04873) | 用 decoder 中间特征构造 LPL；不同 decoder 层贡献不同，只在较高 SNR 使用，FID 改善 6%--20%。 | 完整 decoder hidden loss 已是成熟 baseline；我们的廉价 proxy 必须对它有明确效率或效果优势。 |
| [Improving Reconstruction of Representation Autoencoder / LV-RAE](https://arxiv.org/abs/2602.08620) | 高维 latent 的离流形方向可被 decoder 过度放大；用 latent noise 微调 decoder 并在采样后平滑 latent。 | 我们的 decoder sensitivity atlas 有独立支持，但“离流形敏感”本身已经被直接提出。 |
| [Latent Diffusion Inversion Requires Understanding the Latent Space](https://arxiv.org/abs/2511.20592) | 用 decoder pullback metric 说明 latent 样本和维度具有不同失真与记忆风险。 | “不同 latent 方向的视觉增益不同”已有几何表述；需要研究与生成误差的独特关系，而非只测 Jacobian。 |
| [Geometry-Preserving Encoder/Decoder](https://arxiv.org/abs/2501.09876) | 用成对几何约束研究 encoder/decoder 的弱双 Lipschitz 性和生成优化性质。 | “保持流形几何”本身不是空白；剩余问题必须具体到 RAE 的 off-manifold 生成误差。 |
| [Geometric Decoupling](https://arxiv.org/abs/2604.18804) | 用生成映射 Jacobian、局部尺度和曲率诊断 OOD 生成轨迹的结构不稳定。 | Jacobian 几何诊断也已拥挤；我们的候选方向需要给出 decoder 训练干预及生成收益。 |
| [Navigating the Latent Space Dynamics of Neural Models](https://openreview.net/forum?id=Zunww3FHPU) | 把反复 encode-decode 形成的 latent vector field 用于研究 autoencoder 的吸引子、泛化和记忆。 | `E(D(z))` 循环本身不是新颖点；空缺只能是它是否解释 RAE stage-2 误差和生成质量。 |

## 二、重新解释现有实验

### 1. 可以保留的可靠结论

- 在当前 DINOv2-B RAE decoder 上，等 latent 范数的不同方向会产生显著不同的像素、LPIPS 和 hidden 变化。
- 在当前构造的 24 个子空间中，held-out 跨层线性可预测性与 decoder 局部敏感度高度相关，而且该相关性不只由
  latent 方差解释。
- 常数 channel/basis 权重无法近似逐样本、逐时刻的 decoder perceptual gradient；现有静态 proxy 方案应停止。
- 已训练的 SPC/静态重加权方法在 5 seeds 上恶化 FID/KID，因此不应继续调参挽救。

这些都是当前模型和实验协议下的经验事实。

### 2. 必须降级的结论

- 不能把 `cross-layer predictability` 直接称为“语义轴”。它尚未与 iREPA 的空间结构指标、线性 probing、
  局部流形切向性和 token mean 分量解耦。
- 不能把小 `DiTDH-S` 的高噪声方向响应当成 RAE 的通用生成机制。当前配置主干宽度 `384` 小于 latent
  维度 `768`；`2048` 宽 DDT head 会缓解但不能消除模型尺度混杂。
- 不能把 decoder sensitivity 解释成生成失败的唯一原因。原 RAE 与后续 T2I scaling 结果表明，它的重要性随
  训练阶段和模型尺度改变。
- “denoiser-to-decoder 接力”目前只是对两个归一化响应的描述性分解，不是已经识别出的因果过程。

### 3. 当前真正未解决的问题

文献补齐后，原来的问题仍可研究，但它更像机制分析而非最贴近生成的方法主线：

> 在控制空间结构、语义可分性、方向方差和局部流形切向性之后，encoder 跨层持续性是否仍能独立预测
> diffusion 的方向学习难度与 decoder 的方向增益？这种关系是否随模型尺度发生可重复的转变？

只有答案为“是”，`cross-layer predictability` 才不是已有指标的替身。

PS-VAE 已经明确提出高维 RAE 的 off-manifold generation，因此“发现 RAE 会离流形”也不再是空缺。仍可能有
增量价值、但尚未达到论文 claim 水准的问题被收窄为：

> RAE stage-2 的误差是否主要落在 frozen autoencoder 循环 `F(z)=E(D(z))` 不支持的方向，而 decoder 又是否
> 选择性放大这些误差？若该循环先被验证为局部流形的合理经验代理，再进一步解释为切向/法向失配。

这与 PS-VAE 的低维压缩、各向同性 noise augmentation 的潜在区别在于：候选干预保留 frozen 高维语义 latent，
只约束循环拒绝的样本相关方向。但目前只将它列为待证假设，不能声称 off-manifold、循环算子或流形几何本身
是新颖贡献。详细 claim matrix 与验收门槛见
[RAE 生成研究 Claim Matrix](./RAE_GENERATION_CLAIM_MATRIX_ZH.md)。

## 三、文献后实验顺序

### Phase 0：只做文献和定义审计

1. 逐篇复核上述论文的训练配置、对照组、指标定义和失败案例。
2. 建立 claim matrix：`global semantics / spatial structure / latent spectrum / manifold geometry / decoder robustness / model capacity`。
3. 把本仓库每一条实验结论映射到已有 claim，删除重述，留下真正冲突或空白之处。

验收：每个候选贡献都能明确写出“最接近的两篇论文是什么、它们没有做什么、我们的证据如何区分”。

### Phase 1：三个无需训练的排他性检查

1. **空间结构控制**：在同一组子空间上加入 LDS/空间自相似、token mean 占比和 residual 空间对比度；检验
   predictability 是否仍有独立解释量。
2. **循环一致性控制**：先验证 `F(z)=E(D(z))` 是否更保留真实增强/近邻 secant，而非匹配随机方向；通过后再判断
   stage-2 误差是否被循环拒绝并由 decoder 放大。
3. **模型尺度控制**：对官方可用的 `DiTDH-S` 与更宽 checkpoint 做完全相同的 frozen directional probe；
   不重新训练，只判断小模型排序是否保持。

验收：至少两个控制后 predictability 的独立效应仍稳定，并且方向排序能跨模型尺度复现。否则停止该主线。

### 为什么旧可逆 adapter 路线容易失败

若 adapter `A` 与逆映射 `A^{-1}` 都近似可逆，整个 autoencoder 仍是 `D(A^{-1}(A(E(x))))`。它可以改变坐标条件数，
却不会降低 latent 环境维度、减少 manifold 的余维，或自动让 decoder 适应 stage-2 产生的离支持误差。

这解释了三件事：

- 等变或频谱目标可能让某些 proxy 变好，却不保证生成更容易；
- adapter 的逆映射会把 stage-2 的误差重新送回原 decoder 几何，视觉高增益方向并没有消失；
- PS-VAE 使用的是非可逆的 768 到 96 通道压缩加 KL 正则，解决的问题比可逆换坐标更根本。

因此旧 adapter 的失败更可能是问题设定不充分，而非 epoch、学习率或正则权重没有调好。若未来重新使用 adapter，
它必须提供可验证的降维、分布约束或 decoder robustness，不能只靠可逆坐标变换。

### Phase 2：只允许一个生成实验

只有 Phase 1 通过，才设计一个不等价于 LPL、noise augmentation、REPA/iREPA、DC-AE channel supervision
或 AlignTok 的干预。先在 5k/10k 采样门槛验证，再考虑 50k FID。

验收：

- 与标准 latent MSE、LPL、noise-augmented decoder 和 iREPA/REPA 中至少两个最强相关 baseline 公平比较；
- 主要收益在至少 3 seeds 上成立；
- 同时报训练时间、显存和推理代价；
- 最终必须改善生成指标，而不只是 proxy 或 reconstruction。

## 四、当前研究决定

在完成 Phase 0 前：

- 不继续训练 adapter、decoder 或 DiT；
- 不继续搜索频带权重、方向权重、时间窗口或新的静态 proxy；
- 不把现有结果写成“发现统一 latent trust mechanism”；
- 将现有工作定位为一组有价值的负结果和诊断工具。

目前最诚实的判断是：**现象值得保留，中心解释尚未建立，方法方向高度拥挤。先读清楚再决定，比继续盲试可靠得多。**
