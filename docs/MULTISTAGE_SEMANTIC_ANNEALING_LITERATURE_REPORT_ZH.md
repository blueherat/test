# 多阶段语义退火生成：文献边界、研究价值与低成本验证路线

更新时间：2026-07-20

## 1. 结论先说

你的直觉在概率建模上是成立的：图像生成可以从传统的

```text
高斯噪声 -> 单一 latent -> 图像
```

改写为

```text
高斯噪声 -> 全局语义 -> 空间结构 -> 外观细节 -> 图像
```

也就是逐级减少不确定性，而不是让一个模型一次承担全部生成难度。

但是，经过文献核查，必须很坦率地说：

> **“把两阶段生成改成多阶段，并按语义到细节逐级生成”本身已经不是新想法。**

最接近的工作是 CVPR 2025 的 [Nested Diffusion Models](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Nested_Diffusion_Models_Using_Hierarchical_Latent_Priors_CVPR_2025_paper.pdf)。它已经使用一串 diffusion model，从低维语义表示开始，逐层生成更局部、更高维的 latent，最后生成图像。

此外：

- [Semantic-First Diffusion](https://arxiv.org/abs/2512.04926) 已经明确提出“语义先去噪、纹理后去噪”，通过错开的噪声日程实现语义到细节的时间顺序。
- [ReDi](https://proceedings.neurips.cc/paper_files/paper/2025/hash/186a213d720568b31f9b59c085a23e5a-Abstract-Conference.html) 和 [REG](https://arxiv.org/abs/2507.01467) 已经联合生成高层视觉表示与低层图像 latent。
- [HDAE](https://openaccess.thecvf.com/content/WACV2024/html/Lu_Hierarchical_Diffusion_Autoencoders_and_Disentangled_Image_Manipulation_WACV_2024_paper.html) 已经将不同抽象层级的 encoder 特征注入对应的 diffusion decoder 层，并用于编辑和风格混合。
- [FlexTok](https://arxiv.org/abs/2502.13967)、[Semanticist](https://arxiv.org/abs/2503.08685) 和 [Adaptive Length Image Tokenization](https://arxiv.org/abs/2411.02393) 已经研究了信息按 token 顺序逐步增加、并按图像难度改变信息预算。

所以，原始版本不能作为论文核心贡献。真正还有价值的问题是：

> **怎样自动构造一条“不过难、不过强、不重复”的语义到细节信息阶梯，并证明它在固定总算力下比直接生成更容易训练、更可编辑、更能按需提前停止？**

我认为这个收窄后的问题仍有研究价值，而且与你熟悉的生成模型高度相关。但第一步应做低成本的机制实验，不应直接训练多个大型 RAE diffusion。

## 2. 先把“多阶段”说准确

设最终要生成的图像或 VAE latent 为 `x`，引入三层表示：

- `z_s`：全局语义，例如类别、主体、粗略风格。
- `z_l`：空间结构，例如姿态、布局、部件关系。
- `z_t`：纹理与局部外观。

一个多阶段生成模型可以写为：

```text
p(x, z_t, z_l, z_s)
= p(z_s)
  p(z_l | z_s)
  p(z_t | z_s, z_l)
  p(x | z_s, z_l, z_t)
```

采样顺序为：

```text
z_s ~ p(z_s)
z_l ~ p(z_l | z_s)
z_t ~ p(z_t | z_s, z_l)
x   ~ p(x | z_s, z_l, z_t)
```

这在数学上没有障碍。diffusion、flow、autoregressive model、VAE 都可以参数化其中任一条件分布。

这里不需要把不同维度的 `z_s`、`z_l` 和 `x` 强行放进同一条加噪路径。以条件 diffusion 为例，训练

```text
p(z_l | z_s)
```

时，只在 `z_l` 自己的空间里对 `z_l` 加噪，`z_s` 只是条件；训练

```text
p(x | z_s, z_l, z_t)
```

时，只在 `x` 或 VAE latent 的空间里加噪。每一段都有自己的高斯起点和反向过程。因此，多阶段模型不是“把一个向量连续变形成另一个不同形状的向量”，而是“一串条件分布逐级采样”。

另一种实现是把多个 latent padding/投影成同一个 token 序列，由单一网络联合去噪，再给不同组使用同步或错开的噪声 schedule。ReDi/REG 属于联合生成，Semantic-First Diffusion 属于语义和纹理异步去噪；它们与多个独立 conditional diffusion 是两种不同工程路线。

真正困难的不是“能不能写出这个分解”，而是中间变量是否合适。

一个有用的中间层 `z_k` 必须同时满足两件互相拉扯的事：

1. **可预测**：给定更高层表示后，`z_k` 不能难到与直接生成图像一样。
2. **有增量信息**：它必须为后续生成减少新的不确定性，不能只是重复上一层。

可以用两个直观问题描述：

```text
上一层能不能较容易地产生这一层？
加入这一层后，下一层是否真的变得更容易？
```

如果第一问是否定的，多阶段只是在重新分配同一个困难问题。

如果第二问是否定的，这一层只是额外计算。

如果某一层信息过强，decoder 会绕过其他层，整个系统退化成 autoencoder。Nested Diffusion 明确观察到了这一问题，因此需要用 SVD 降维和高斯噪声手工限制中间特征容量。

## 3. “退火”在这里有三种不同含义

### 3.1 尺度退火

先生成低分辨率，再逐级超分辨率。

代表工作：

- [Cascaded Diffusion Models](https://arxiv.org/abs/2106.15282)
- [Matryoshka Diffusion Models](https://arxiv.org/abs/2310.15111)
- [VAR](https://arxiv.org/abs/2404.02905)
- [MSF](https://arxiv.org/abs/2501.13349)

它们通常实现“低频到高频”或“低分辨率到高分辨率”，但低分辨率不等于高层语义。一张低分辨率图仍可能包含颜色、纹理统计，也可能丢掉小物体语义。

### 3.2 信息量退火

先生成少量、最重要的 token，再不断增加 token 或残差码。

代表工作：

- [VQ-VAE-2](https://arxiv.org/abs/1906.00446)
- [RQ-VAE/RQ-Transformer](https://arxiv.org/abs/2203.01941)
- [FlexTok](https://arxiv.org/abs/2502.13967)
- [Semanticist](https://arxiv.org/abs/2503.08685)
- [Adaptive Length Image Tokenization](https://arxiv.org/abs/2411.02393)

这类工作最接近“逐步增加信息预算”。但 token 的先后顺序未必等于可解释的语义层级。

### 3.3 语义不确定性退火

先决定“是什么”，再决定“在哪里、长什么样”，最后决定纹理细节。

代表工作：

- [DALL-E 2 / unCLIP](https://arxiv.org/abs/2204.06125)
- [RCG](https://arxiv.org/abs/2312.03701)
- [HDAE](https://arxiv.org/abs/2304.11829)
- [Nested Diffusion Models](https://arxiv.org/abs/2412.05984)
- [ReDi](https://arxiv.org/abs/2504.16064)
- [REG](https://arxiv.org/abs/2507.01467)
- [Semantic-First Diffusion](https://arxiv.org/abs/2512.04926)
- [SeFi-Image](https://arxiv.org/abs/2606.22568)

你的想法主要属于第三类，也与第二类有交叉。

## 4. 与核心文献逐项比较

| 工作 | 中间变量 | 生成顺序 | 是否显式语义到细节 | 是否支持分层编辑 | 主要局限 |
|---|---|---|---|---|---|
| VDVAE | 多组随机 latent | 深层层次 prior | 部分，主要是自然涌现 | 弱 | 层级职责不明确，可能 posterior collapse |
| VQ-VAE-2 | top/bottom 离散码 | top 后 bottom | 更偏尺度与全局/局部 | 有限 | 不是现代 VFM 语义空间 |
| Cascaded Diffusion | 多分辨率图像 | 低分辨率后高分辨率 | 否，主要是空间尺度 | 有限 | 级联误差，需要 conditioning augmentation |
| DALL-E 2 | CLIP image embedding + image | 语义 prior 后 decoder | 两级，明确 | 支持语义变化 | 只有一个语义层 |
| RCG | 自监督语义表示 + image | representation 后 image | 两级，明确 | 不是主要目标 | 未细分布局与纹理 |
| HDAE | encoder 多尺度语义 code | 单一 diffusion decoder，多层注入 | 明确 | 强，展示风格混合和属性编辑 | 主要在较受限数据集；不是多个生成 prior 串联 |
| Nested Diffusion | 多层冻结视觉特征 + VAE latent | 多个 diffusion 逐层串联 | **是，和原始想法最接近** | 可固定高层、重采样低层 | 参数随层数增加；容量靠手工 SVD/噪声；存在级联误差风险 |
| ReDi/REG | VFM 语义 + VAE latent | 同一模型联合生成 | 两级但同步 | 可做 representation guidance | 不提供多个可解释阶段 |
| Semantic-First Diffusion | compact semantic latent + texture latent | 同一模型异步去噪 | **是，语义时间上领先纹理** | 不是核心评测 | 主要是两路；语义层级仍由预定义表示决定 |
| FlexTok/Semanticist | 有序 token | token 逐步增加 | 有一定语义先后 | 可通过截断控制信息量 | 不保证每一段对应独立可干预因素 |
| RAE/RAEv2 | VFM feature + learned decoder | 单 latent diffusion 后 decoder | 否 | latent 可编辑性仍未系统解决 | 高维、不同 encoder 层并非天然嵌套 sufficient statistics |

### 4.1 证据成熟度说明

- Nested Diffusion 已被 CVPR 2025 接收，HDAE 已被 WACV 2024 接收，ReDi 已被 NeurIPS 2025 接收。这些是最重要的已同行评审先例。
- Semantic-First Diffusion、SeFi-Image、REG、RAE/RAEv2 等截至本报告日期主要以 arXiv 或技术报告形式公开。报告中涉及它们的性能提升均应理解为作者报告结果，不能视为独立复现结论。
- 但判断论文新颖性时，即使是公开预印本也不能忽略。也就是说，我们可以对其性能主张保持谨慎，同时仍要承认“语义优先异步去噪”这一方法表述已经公开出现。
- 不同论文的 FID 使用的数据、训练轮数、guidance、模型规模和评估实现并不统一，本文不进行横向数字排名，只比较结构与研究问题。

## 5. 哪些论文已经覆盖了原始贡献

### 5.1 “把两阶段改成多阶段”

已被覆盖。

层次 VAE、VQ-VAE-2、级联扩散和 Nested Diffusion 都已经这样做。单纯增加 `z_1, z_2, z_3` 不构成贡献。

### 5.2 “用视觉 encoder 的不同语义层作为阶段”

大部分已被覆盖。

HDAE 使用多层 encoder feature 对应 diffusion decoder 层；Nested Diffusion 用冻结视觉 encoder 构造多层语义 target。2026 年的 [RAEv2](https://arxiv.org/abs/2605.18324) 以及 [DRoRAE](https://arxiv.org/abs/2605.10780) 也开始直接融合视觉 encoder 的多个层。

所以“拿 DINOv2 第 3、6、12 层做三个 latent”本身新颖性很弱。

### 5.3 “先语义，后纹理”

已被直接覆盖。

Semantic-First Diffusion 用两套错开的噪声 schedule，让 semantic latent 比 texture latent 提前去噪。SeFi-Image 又把这一范式扩展到了大规模文生图。

### 5.4 “逐阶段增加信息，可以中途停止”

已有强先例。

FlexTok 支持 1 到 256 个有序 token；Adaptive Length Image Tokenization 会按图像复杂度分配 token 数；Semanticist 强制后续 token 提供互补、递减的信息。

因此，早停或可变长度必须和新的生成目标、编辑能力或算力分配方法结合，不能单独作为贡献。

## 6. 仍然存在的真正缺口

### 6.1 自动决定每一阶段应该保留多少信息

Nested Diffusion 的关键超参数是每层保留多少 SVD 通道、加多少噪声。论文使用逐层的手工或贪心搜索。

这说明它没有真正解决：

```text
哪一级应该拥有哪部分信息？
每一级应当减少多少条件不确定性？
多少容量才不会旁路上层，又足以帮助下层？
```

这是最扎实的研究缺口。

可以将每层价值写成一种“单位计算带来的新增信息”：

```text
value_k = 下游条件难度的下降 / 第 k 层额外计算
```

不必一开始精确估计互信息，可以用 held-out 条件生成 loss、flow matching loss 或小型 predictor 的误差作为代理量。

### 6.2 阶段职责缺乏可证伪定义

很多论文通过插值、风格混合或重采样展示层级，但这不等于每层真的只负责指定因素。

一个更严格的目标是定义 intervention contract：

- 重采样语义层，应允许类别、主体或大结构改变。
- 固定语义层、重采样布局层，类别应基本不变，但姿态和位置可以变。
- 固定前两层、重采样纹理层，主体和布局应保持，颜色、材质和局部细节可以变。

这可以形成一个“阶段干预泄漏矩阵”，定量测量某层变化是否错误地影响了其他层职责。

HDAE 和 Nested Diffusion 有定性展示，但在开放域自然图像、现代 VFM latent、严格干预指标下仍有空间。

### 6.3 训练时真值条件与采样时生成条件不一致

级联系统训练第 `k` 层时通常看到真实的上层 target，但采样时看到的是前一个生成器产生的近似 latent。

这是典型的 rollout gap。Cascaded Diffusion 已证明 conditioning augmentation 对避免误差累积至关重要；Nested Diffusion 也需要对条件加噪并处理训练/推理分布差异。

仍可研究：

- 用生成的 parent latent 混入训练，而不仅是 oracle parent。
- 对 parent latent 做与真实生成误差匹配的扰动。
- 加入跨阶段 rollout consistency。
- 让后级能识别上级不确定性，而不是把单点预测当成真值。

### 6.4 固定阶段数不是最优的

简单图像可能只需要语义层和少量纹理；复杂场景需要额外布局、关系和细节阶段。

自适应 tokenization 已研究“图像需要多少 token”，但“生成链需要走几级、每级投入多少 NFE”仍未被充分解决，尤其是在语义到细节的条件生成链中。

### 6.5 多个独立 diffusion 浪费参数

Nested Diffusion 的五层版本仅增加约 27% GFLOPs，但参数从约 104M 增长到约 523M。计算低是因为高层 latent 很小，不等于存储和训练成本低。

一个共享参数的 stage-conditioned denoiser 可能更适合有限资源：

- 同一个 backbone 处理多个 latent level。
- 用 stage embedding 区分语义、布局、纹理。
- 每个 level 使用不同的输入/输出投影头。
- 可选择同步、异步或顺序 sampling。

不过 Matryoshka Diffusion 已经做了多分辨率共享，因此贡献必须强调“语义残差层级”而不只是共享网络。

## 7. 我认为最有希望的论文命题

建议不要将论文命题写成：

> 我们把 RAE 从两阶段扩展成多阶段。

建议写成：

> **我们研究怎样在固定总算力下构造一条可预测、非冗余、可干预的语义残差阶梯，使图像生成按语义、布局和细节逐级减少条件不确定性。**

更具体的模型可以叫“语义残差阶梯”，核心不是直接拿三层 encoder feature，而是让每一级只表达前面层没有解释掉的部分：

```text
z_sem    = compact_global_feature(x)
z_layout = spatial_feature(x) - predict_spatial(z_sem)
z_detail = vae_latent(x) - predict_texture(z_sem, z_layout)
```

这里的减号不一定是直接逐元素相减，可以是学习出的 residual target。

然后生成：

```text
p(z_sem)
p(z_layout_residual | z_sem)
p(z_detail_residual | z_sem, z_layout)
```

这个设计有三个潜在优点：

1. 后一层被迫提供增量信息，减少重复。
2. 每层条件难度可以单独测量。
3. 可以通过阶段 dropout 与重采样定义明确的编辑接口。

它也有明显风险：RQ-VAE、可扩展压缩和 progressive tokenizer 已经研究残差编码。因此论文必须证明“语义职责、条件生成难度和编辑干预”三者，而不能只证明 reconstruction 更好。

## 8. 为什么不建议立刻在完整 RAE 上做

完整 RAE 会同时引入：

- 高维 DINOv2 latent 的生成难度。
- 大型 decoder 的闭环误差与伪影。
- 多阶段生成器的误差累积。
- encoder 层之间并非天然嵌套的问题。
- 大规模 FID 评估的成本。

如果结果变差，很难判断是层级想法错了、latent target 不合适、某个 stage 没训练好，还是 decoder 放大了误差。

因此第一轮应把最终生成空间固定成成熟的 VAE latent，并把 DINOv2 只当作语义和布局 target。这样仍然是在做生成模型，而且能把 decoder 变量隔离掉。

## 9. 推荐的低成本生死实验

### 9.1 数据与模型

- 数据：ImageNet-1K 的 100 个类子集，或完整 ImageNet 的 128x128/256x256 版本。
- 严格使用官方 train split 训练，validation split 只评估，避免泄露。
- 最终图像空间：冻结的 SD-VAE latent `u`。
- 语义目标：冻结 DINOv2 的全局 token，PCA 到 32 或 64 维。
- 布局目标：冻结 DINOv2 patch token，池化到 4x4 或 8x8，再 PCA 到 16 或 32 通道。
- 图像 decoder：冻结 SD-VAE decoder，不训练 RAE decoder。

### 9.2 四个必须比较的系统

1. `Direct`：直接生成 `u`。
2. `Two-stage`：先生成 `z_sem`，再生成 `u | z_sem`。
3. `Three-stage`：生成 `z_sem -> z_layout -> u`。
4. `Joint-asynchronous`：一个共享模型联合生成 semantic/layout/texture，使用错开的噪声 schedule，作为 SFD 风格基线。

所有系统必须匹配：

- 总参数量，或至少报告参数差异。
- 总 NFE。
- 每张图的总 FLOPs。
- 总训练 step 与看过的样本数。

否则多阶段多花的计算会被误认为结构优势。

### 9.3 先不训练 top-level prior

第一轮可以使用真实 `z_sem`，只比较：

```text
p(u | z_sem)
```

与

```text
p(z_layout | z_sem) p(u | z_sem, z_layout)
```

这直接回答：增加布局中间层是否真的降低最终条件生成难度。

如果使用 oracle `z_sem` 都没有收益，就没有必要训练 `p(z_sem)`。

### 9.4 必须测的四类结果

#### 生成质量

- FID/KID，优先用 KID 做小样本早期筛选。
- Precision/Recall，防止只提升保真度却损失覆盖。
- 相同 wall-clock 与相同 NFE 下的结果。

#### 收敛速度

- 达到相同 KID/FID 阈值所需的训练 step。
- 每阶段 validation flow/diffusion loss。
- 三个 seed 的均值与方差。

#### 阶段职责

固定其他阶段，只重采样一个阶段，测量：

- DINO/分类语义一致性。
- 物体布局或 patch correspondence 变化。
- 颜色直方图与低频结构变化。
- LPIPS 与高频能量变化。

最后形成“被重采样层 x 被改变属性”的矩阵。

#### 级联鲁棒性

- oracle parent 条件下的结果。
- generated parent 条件下的结果。
- 两者差距，即 rollout gap。

### 9.5 训练防旁路措施

- stage dropout：随机丢弃后级或前级条件。
- parent corruption：按真实 parent generator 的误差分布扰动条件。
- capacity budget：限制每层维度或加噪，不允许中间层近乎重构输入。
- unique-use ablation：删除任一层后都应损失它负责的特定属性，而不只是总体 FID 略变。

## 10. 验收标准

第一阶段继续的条件建议设为：

1. 三阶段模型在相同总 NFE/FLOPs 下，相对 compute-matched 两阶段 baseline 的 KID 或 FID proxy 至少改善 10%，且三 seed 方向一致。
2. 或在达到同一质量阈值时，训练 step 至少减少 1.5 倍。
3. generated-parent 相对 oracle-parent 的质量退化不超过 15%。
4. 阶段干预具有单调职责：重采样 detail 时，高层类别保持率至少 95%；重采样 layout 对布局的影响显著大于对类别的影响。
5. 删除任一阶段会损失该阶段独有的指标，证明没有旁路或无效阶段。
6. 优势不能被一个参数量相同的更大 direct model 消除。

立即停止的条件：

- compute-matched direct model 达到相同或更好结果。
- oracle 条件有收益，但 generated parent 一 rollout 就消失。
- 各层重采样产生相同类型的变化，说明没有职责分化。
- 多阶段只改善 reconstruction，不改善生成质量、速度或编辑。
- 结果对 PCA 维度、噪声强度极度敏感，无法形成稳定规律。

## 11. 对 ICLR 价值的客观评分

### 原始版本

```text
“将两阶段 RAE 改成语义到细节的多阶段 diffusion”
```

新颖性：`2/10`

原因是 Nested Diffusion 和 Semantic-First Diffusion 已基本覆盖核心叙事。

### 收窄后的版本

```text
“自动学习可预测、非冗余、可干预的语义残差阶梯，
并在固定总算力下实现自适应阶段生成”
```

潜在价值：`6/10`，若机制与规模实验都强，可提高到 `7/10`。

要达到 ICLR 水准，至少需要：

- 一个清楚的新目标或学习准则，而不只是新架构组合。
- 与 Nested Diffusion、SFD、ReDi、两阶段 RCG 的公平 compute-matched 对照。
- 严格的阶段干预与 rollout-gap 实验。
- 至少两个数据域，不能只有 MNIST/CIFAR。
- 证明收益来自“信息阶梯”，而不是多参数、多 NFE 或更强条件。

## 12. 最终建议

这条路线值得做一次严格、低成本的生死实验，但不值得现在就做完整大规模 RAE 训练。

最推荐的顺序是：

1. 冻结 DINOv2、SD-VAE encoder/decoder。
2. 用真实 semantic parent，先验证加入 layout stage 是否降低 `p(u | condition)` 的生成难度。
3. 通过后，再训练 layout prior，并测 oracle/generated rollout gap。
4. 只有三阶段在 compute-matched 比较中成立，才训练 top semantic prior。
5. 最后再考虑将 SD-VAE endpoint 换成 RAE latent，或把三个独立模型改成共享异步 denoiser。

一句话总结：

> **“多阶段”不是贡献，“语义先于细节”也不再是贡献；真正可能成为贡献的是，自动找到一条每级都可预测、提供独有信息、经得住生成误差且能被单独干预的信息阶梯。**

## 13. 核心参考文献

- Razavi et al., [Generating Diverse High-Fidelity Images with VQ-VAE-2](https://arxiv.org/abs/1906.00446), 2019.
- Child, [Very Deep VAEs Generalize Autoregressive Models and Can Outperform Them on Images](https://arxiv.org/abs/2011.10650), 2020.
- Ho et al., [Cascaded Diffusion Models for High Fidelity Image Generation](https://arxiv.org/abs/2106.15282), 2021.
- Preechakul et al., [Diffusion Autoencoders](https://arxiv.org/abs/2111.15640), CVPR 2022.
- Ramesh et al., [Hierarchical Text-Conditional Image Generation with CLIP Latents](https://arxiv.org/abs/2204.06125), 2022.
- Lee et al., [Autoregressive Image Generation using Residual Quantization](https://arxiv.org/abs/2203.01941), CVPR 2022.
- Lu et al., [Hierarchical Diffusion Autoencoders and Disentangled Image Manipulation](https://arxiv.org/abs/2304.11829), WACV 2024.
- Hudson et al., [SODA: Bottleneck Diffusion Models for Representation Learning](https://arxiv.org/abs/2311.17901), CVPR 2024.
- Gu et al., [Matryoshka Diffusion Models](https://arxiv.org/abs/2310.15111), ICLR 2024.
- Li et al., [Return of Unconditional Generation](https://arxiv.org/abs/2312.03701), NeurIPS 2024 Oral.
- Duggal et al., [Adaptive Length Image Tokenization via Recurrent Allocation](https://arxiv.org/abs/2411.02393), 2024.
- Tian et al., [Visual Autoregressive Modeling](https://arxiv.org/abs/2404.02905), NeurIPS 2024.
- Zhang et al., [Nested Diffusion Models Using Hierarchical Latent Priors](https://arxiv.org/abs/2412.05984), CVPR 2025.
- Bachmann et al., [FlexTok](https://arxiv.org/abs/2502.13967), ICML 2025.
- Wen et al., ["Principal Components" Enable A New Language of Images](https://arxiv.org/abs/2503.08685), ICCV 2025.
- Kouzelis et al., [Boosting Generative Image Modeling via Joint Image-Feature Synthesis](https://arxiv.org/abs/2504.16064), NeurIPS 2025 Spotlight.
- Wu et al., [Representation Entanglement for Generation](https://arxiv.org/abs/2507.01467), 2025.
- Pan et al., [Semantics Lead the Way: Asynchronous Latent Diffusion](https://arxiv.org/abs/2512.04926), 2025.
- Zheng et al., [Diffusion Transformers with Representation Autoencoders](https://arxiv.org/abs/2510.11690), 2025.
- Singh et al., [Improved Baselines with Representation Autoencoders](https://arxiv.org/abs/2605.18324), 2026.
- Feng et al., [SeFi-Image: A Text-to-Image Foundation Model with Semantic-First Diffusion](https://arxiv.org/abs/2606.22568), 2026.
