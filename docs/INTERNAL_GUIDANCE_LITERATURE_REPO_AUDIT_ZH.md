# Internal Guidance 文献、代码与研究路线审计

## 结论先说

当前不应该继续在 RAEv2 上盲目增加 LPL、层选择器或控制器。这个邻域已经很密集：中间层预测、跳层弱模型、注意力扰动、token 扰动、head 选择、动态 guidance scale、时间分段和控制论解释都已经有人研究。

我们现在真正做的事情应当改成：

> 先在一个公开、便宜、可严格复现的系统中，判断 weak-to-strong 方向的局部性质能否预测它对最终生成分布的作用。

这不是在发明一个新 sampler，也不是继续挽救 LPL。它是一个机制判定：如果局部误差指标不能预测最终生成，那么以后就不能再凭单步 MSE、cosine 或方向范数设计 internal guidance；如果存在可重复、跨模型的预测规律，才有资格进一步做方法。

推荐顺序是：

1. 以 NVIDIA 官方 EDM2/AutoGuidance 的 ImageNet-64 模型为第一底座。
2. 先原样复现官方 unguided 与 AutoGuidance 指标。
3. 只给官方 sampler 加观测和时间窗口开关，不重写模型或采样器。
4. 若机制成立，再在 Hugging Face Diffusers 的官方 PAG 实现上做第二次验证。
5. 最后才回到 Internal Guidance 或 RAEv2，判断该机制能否解释双头 guidance。

## 一、我们从哪里开始

最初的问题是：LPL 在旧 RAE 上短程微调后看起来有收益，但在训练更规范、模型更成熟的 RAEv2 上没有复现，甚至 full head 单独采样也会退化。

这迫使我们把两个问题分开：

1. LPL 是否改善了它自身的特征损失。
2. 这种局部改善是否真的能改善整条生成轨迹。

随后，RAEv2 的双头结构暴露出一个更一般的问题。它有一个较浅的 `base` 预测和一个最终的 `full` 预测，官方 internal guidance 使用：

```text
guided = base + s * (full - base)
```

其中 `s=1` 就是只用 `full`；`s>1` 才是沿 `base -> full` 继续外推。官方代码还允许给 IG 和 CFG 分别指定时间区间。

直觉上，`full` 比 `base` 好，并不自动推出继续沿 `full-base` 走会更好。为检查这一点，我们写了局部方向审计，比较：

```text
full - base
```

和监督目标尚未修正的误差方向：

```text
target - full
```

但这里出现了一个关键认识：guidance 本来就会改变生成分布，它不一定以降低单步监督 MSE 为目标。因此，局部方向审计只能描述向量场，不能单独判定最终 FID。

## 二、前面实际做了什么

前面的工作包含两类先导实验，不是一个完整的新方法。

### 1. RAEv2 双头方向审计

目的：判断 `base -> full` 是否继续指向真实训练目标，以及这种关系随噪声时间如何变化。

它回答的是局部问题：

```text
在给定的真实训练配对 (x_t, target) 上，向这个方向走一点会不会降低当前预测误差？
```

它不直接回答：

```text
在模型自己采样产生的 x_t 上，重复使用这个方向是否会改善最终图像分布？
```

### 2. 小型公开 DDPM/PAG 先导实验

为了避免每次都运行大 RAE，我们使用公开的 CIFAR-10 DDPM 和 Diffusers 的 PAG attention processor 做了一个低成本先导实验。结果出现了明显断层：

- 在真实加噪数据上的局部 epsilon MSE 中，正向 PAG 外推主要在高噪声处有利。
- 在三组已完成的 1,000 样本、20 步 DDIM 采样中，低噪声开启 PAG 反而改善了 FID/KID/IS，而高噪声窗口收益很小。
- 在低噪声处，朝 perturbed branch 插值会改善局部 MSE，却明显破坏最终生成。

这个现象值得注意，但目前只能称为先导证据，原因是：

- 每组只有 1,000 张图，FID 方差较大。
- 只完成了三组种子。
- 使用的是很小、较旧的 CIFAR-10 DDPM。
- 局部监督样本和实际采样轨迹不是同一分布。
- guidance 可能有意牺牲逐点预测误差，换取更好的分布质量或更小的模式覆盖。

因此这组结果只负责告诉我们“局部 MSE 不能被默认当成 endpoint 指标”，不负责证明一个新机制。

## 三、已有研究已经做到什么程度

### A. 直接构造 weak-to-strong 方向

**AutoGuidance** 用较小或训练不足的同任务模型作为 weak model，并从 weak prediction 向 strong prediction 外推。它的重要结论是 weak degradation 必须与 strong model 兼容，不是任意坏模型都有效。官方 EDM2 仓库同时提供模型、采样器和 ImageNet-64/512 参考统计。[论文](https://openreview.net/forum?id=bg6fVPVs3s)；[官方仓库](https://github.com/NVlabs/edm2)

**Weak-to-Strong Diffusion (W2SD)** 进一步把 weak-to-strong gap 当作 strong-to-ideal gap 的代理，并用 denoising/inversion reflection 累积修正。官方代码目前主要公开了 SDXL/LoRA 的 weight-difference 路线，并不是一个覆盖论文所有模型配对的通用库。[论文](https://openreview.net/forum?id=tg19FVh3p1)；[官方仓库](https://github.com/xie-lab-ml/Weak-to-Strong-Diffusion-with-Reflection)

所以，“做一个 weak model，然后从 weak 向 strong 外推”本身已经不是创新。

### B. 用网络内部扰动制造 weak model

**PAG** 把部分 self-attention map 替换为 identity，构造结构退化分支，再让生成远离该分支。PAG 已经进入 Diffusers，包含多个 pipeline 和 `pag_scale`、`pag_adaptive_scale` 等接口。[论文](https://arxiv.org/abs/2403.17377)；[Diffusers 实现](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/pag)

**TPG** 在选定 Transformer block 中打乱 token 顺序，并使用：

```text
original + scale * (original - perturbed)
```

官方仓库给出一个完整的 SDXL pipeline，但主要是单文件实现，覆盖范围比 Diffusers PAG 小。[论文](https://arxiv.org/abs/2506.10036)；[官方仓库](https://github.com/TaatiTeam/Token-Perturbation-Guidance)

**STG** 通过跳过时空 Transformer 层构造视频弱模型，避免额外训练。[论文](https://arxiv.org/abs/2411.18664)；[官方仓库](https://github.com/junhahyung/STGuidance)

**HeadHunter/SoftPAG** 已经把扰动粒度推进到单个 attention head，并用 PickScore/CLIPScore 等外部指标迭代选择 head；SoftPAG 还研究了原 attention 与 perturbed attention 之间的连续插值。[论文](https://openreview.net/forum?id=m7zEbdAMsh)；[官方仓库](https://github.com/cvlab-kaist/HeadHunter)

**S²-Guidance** 随机丢弃 Transformer blocks 来采样随机弱子网。这篇工作已被 ICLR 2026 接收，但截至本次审计，官方仓库仍只有 README 和素材，README 仍标注代码即将发布，不能作为当前实现底座。[论文](https://arxiv.org/abs/2508.12880)；[官方仓库现状](https://github.com/AMAP-ML/S2-Guidance)

所以，“逐层当 base”“随机跳层”“搜索最佳层或最佳 head”也已经是拥挤区域。

### C. 直接训练中间预测头或对齐深浅层

**Internal Guidance (IG)** 在中间层接独立输出头，并给中间头和最终头相同的速度目标；采样时对二者外推。官方代码支持 SiT-B/L/XL 训练，但公开预训练权重只有 SiT-XL/2 和 LightningDiT-XL/1，两者在 Hugging Face 上合计约 19 GB。[论文](https://arxiv.org/abs/2512.24176)；[官方仓库](https://github.com/CVL-UESTC/Internal-Guidance)；[官方权重](https://huggingface.co/CVLUESTC/Internal-Guidance/tree/main)

**RAEv2** 把 RAE latent 的 REPA/x-prediction 中间支路重参数化为 base head，并使用同样的 internal guidance 公式。其 base head 不是简单截断网络，而是训练时显式监督的专用 head。[论文](https://arxiv.org/abs/2605.18324)；[官方仓库](https://github.com/nanovisionx/RAEv2)

**DeepFlow** 给多个深度分支加 velocity supervision，并用 VeRA 对齐相邻分支，加快 flow model 训练。[论文](https://arxiv.org/abs/2503.14494)；[官方仓库](https://github.com/innnkyu/DeepFlow)

**LayerSync** 用深层语义表示对齐浅层表示，是训练正则而不是采样 guidance，已在 ICLR 2026 发表。[论文](https://openreview.net/forum?id=4itprlvbRQ)；[官方仓库](https://github.com/EPFL-IMOS/LayerSync)

所以，“给每层加 head”“让 decoder/深层对齐浅层”同样不能只靠形式新颖性成立。

### D. 时间调度、动态强度和控制观点

**Limited Interval Guidance** 已证明固定 guidance 覆盖全轨迹通常不是最优；在其 EDM2/CFG 设置下，高噪声 guidance 有害、低噪声 guidance 大多无用、中间区间最有利。[论文](https://openreview.net/forum?id=nAIhvNy15T)

**Feedback Guidance** 根据当前 conditional signal 的估计可信度动态调整 CFG scale，而不是只用固定时间表。[论文](https://arxiv.org/abs/2506.06085)；[官方仓库](https://github.com/FelixKoulischer/Feedback-Guidance-of-Diffusion-Models)

**Segmented Guidance (SGG)** 在不同采样区间切换 CFG 与 skip-layer guidance，并同时给出 SD3/3.5 推理和 SiT 训练代码。[论文](https://arxiv.org/abs/2603.20584)；[官方仓库](https://github.com/851695e35/SGG)

**Variational Control for Guidance** 已经把扩散 guidance 写成带终端代价和轨迹正则的控制问题。[论文](https://arxiv.org/abs/2502.03686)；[官方仓库](https://github.com/czi-ai/oc-guidance)

**On the Guidance of Flow Matching** 系统区分了 flow matching guidance 与标准 diffusion guidance，并给出一般概率路径上的理论框架。[论文](https://proceedings.mlr.press/v267/feng25s.html)；[官方仓库](https://github.com/AI4Science-WestlakeU/flow_guidance)

所以，“让 scale 随时间变化”“做状态反馈”“用控制理论解释”单独都不足以成为研究贡献。

### E. 为什么局部正确不保证最终正确

**What does guidance do?** 严格说明 guidance 并不简单等价于从预期的幂次倾斜分布采样；在有 score error 时，大 guidance 还可能把样本推离数据支撑。[论文](https://openreview.net/forum?id=AdS3H8SaPi)

**Guided Path Sampling** 从迭代 denoising/inversion 的角度研究了局部误差如何沿轨迹累积，并主张外推可能造成 off-manifold 误差，而插值可使特定迭代过程更稳定。它与我们观察到的“局部指标和 endpoint 指标不一致”相邻，但研究对象主要是带 inversion 的迭代 refinement。[论文](https://arxiv.org/abs/2512.22881)

**ERK-Guid** 直接把 ODE solver 的局部截断误差和 stiffness 当成 guidance 信号，说明“误差传播/数值动力学”本身也已成为研究方向。[论文](https://arxiv.org/abs/2603.03692)；[官方仓库](https://github.com/mlvlab/ERK-Guid)

因此，我们不能把“局部误差会积累”本身当成新发现。真正需要确认的是：对 internal/perturbation weak-to-strong direction，是否存在一个可观测、无需 ground truth 或外部 reward、且能跨样本预测 endpoint 作用的信号。

## 四、官方仓库可复用性审计

| 底座 | 代码/权重状态 | 资源成本 | 适合作什么 | 当前判断 |
|---|---|---:|---|---|
| EDM2/AutoGuidance | 完整；ImageNet-64/512 强弱模型、采样器、参考统计齐全 | ImageNet-64 较低 | 复现 weak-to-strong、时间窗口、局部到终点机制 | 第一选择 |
| Diffusers PAG | 成熟集成；支持多种 pipeline | SD/SDXL 中等 | 第二个、不同 weak 构造的机制验证 | 第二选择 |
| TPG | 完整 SDXL 单文件 pipeline | 中等 | token perturbation 对照 | 可选对照 |
| HeadHunter | 完整，但含大型模型、多 GPU、外部评分搜索 | 高 | 对照已有 layer/head selection | 阅读和比较，不先运行 |
| Internal Guidance | 训练/采样完整；公开权重仅 XL | 高 | 最后验证训练式 intermediate head | 不作为第一站 |
| RAEv2 | 完整且与我们的现象直接相关 | 很高、系统复杂 | 最终迁移和解释 | 最后回归 |
| S²-Guidance | 论文有，官方仓库暂无代码 | 暂不可运行 | 随机 block-drop 对照 | 等官方发布 |
| SGG | SD3/3.5 和 SiT 代码完整 | 中高 | 时间分段与 skip guidance 对照 | 后续候选 |
| W2SD | 公开实现偏 SDXL/LoRA weight difference | 中高 | reflection 对照 | 不作为通用底座 |
| Feedback Guidance | EDM2 分支完整 | 中等 | 状态相关动态 scale 对照 | 可直接复用 |

## 五、为什么第一底座应是 EDM2 ImageNet-64

官方仓库已经预注册了可直接运行的模型配对：

```text
strong: edm2-img64-s-1073741-0.045.pkl
weak:   edm2-img64-xs-0134217-0.110.pkl
scale:  1.70
```

官方报告：

```text
EDM2 ImageNet-64 S unguided FID: 1.58
EDM2 ImageNet-64 S + AutoGuidance FID: 1.01
```

它比当前 CIFAR 先导实验更可靠，因为：

- strong/weak 配对由原论文选择，而不是我们临时制造。
- 有官方 50k 评估脚本与 ImageNet-64 参考统计。
- 官方 sampler 只有一个清晰的 weak/strong 线性组合，易于审计。
- 64×64 和 S/XS 模型使四张 4090 足以做完整配对实验。
- 不含 RAE decoder、双头训练和 latent normalization 等额外混杂。

最重要的是：第一步无需训练任何模型，只运行官方权重。

## 六、下一步究竟探究什么

### 核心问题

> weak-to-strong guidance 的最终收益，能否由局部监督误差、方向对齐、时间位置或轨迹响应稳定地预测？

这里要明确区分三个量：

1. **局部预测质量**：当前 `x_t` 上 strong/weak 对 clean target 的误差。
2. **单步响应**：只在一个时间窗口加 guidance，之后恢复官方 sampler，终点发生什么变化。
3. **完整分布质量**：整条采样使用某种策略后的 FID、FD-DINO、precision/recall 和覆盖度。

如果 1 能稳定预测 2 和 3，才有可能设计低成本 adaptive guidance。

如果 1 与 2/3 系统性脱钩，结论也有价值：internal guidance 不能用 teacher-forced 单步 MSE 来选择方向，必须评价 rollout 或分布响应。

但只有在至少两个官方机制上复现这个脱钩，才有论文级可信度。

## 七、严格执行计划

### Phase 0：原样复现官方结果

不改任何模型和公式，直接运行官方 `edm2-img64-s-fid` 和 `edm2-img64-s-autog-fid`。

验收：

- 使用官方 checkpoint、32-step sampler、guidance 1.70 和官方参考统计。
- 先 5k smoke，确认图像、类别、seed 分片和多卡无重复。
- 再做 50k FID；如果不能接近官方趋势，立即停下检查环境和评估，不进入机制实验。
- 所有新代码只作为官方函数的薄封装；不能复制并悄悄改写 sampler。

### Phase 1：时间窗口与局部量

在同一组 paired seeds 上比较：

```text
unguided
official full-interval AutoGuidance
high-noise only
middle-noise only
low-noise only
single-window impulse
```

同时记录：

- weak/strong clean prediction error；
- `strong-weak` 与 `target-strong` 的 cosine；
- 方向范数和最优局部 scale；
- sampler 的一步状态变化；
- 终点的 Inception/DINO 特征变化。

Phase 1 不训练 controller，也不搜索几十种 schedule。

### Phase 2：检验局部量能否预测终点

对时间窗口和样本做严格 held-out：一部分 seed 只用于提出预测，另一部分只用于验证。

需要回答：

- 局部 MSE gain 的符号是否预测 endpoint quality 的符号？
- cosine、方向范数、噪声时间中谁更稳定？
- 同一个窗口在不同类别和 seed 上是否一致？
- 低 FID 是真实质量提升，还是 precision 上升、recall 下降的重加权？

### Phase 3：第二个官方机制验证

只在 EDM2 得到明确、可重复的现象后，使用 Diffusers 官方 PAG pipeline 验证。优先选择能直接由 Diffusers 加载的公开模型，不重写 attention processor。

这里测试的是不同 weak 构造：

```text
AutoGuidance: 小/欠训练模型作为 weak
PAG: attention perturbation 作为 weak
```

如果两者都出现同样的局部到终点断层，研究问题才具有一般性。

### Phase 4：最后回到 IG/RAEv2

只有前三阶段通过，才问：

- 训练式 base head 是否比人为 perturbation 更可预测？
- RAEv2 的 `base -> full` 是否遵循同样规律？
- LPL 是否改变 full 本身、改变 guidance direction，还是二者都改变？

在此之前不训练新的 RAE adapter、LPL、controller 或多层 head。

## 八、停止条件

遇到任一情况就停止该研究方向：

1. 官方 EDM2 unguided/AutoGuidance 基线无法可靠复现。
2. 时间窗口或 impulse response 对 seed、类别和采样器极不稳定。
3. 局部指标与 endpoint 关系完全可以由 Limited Interval Guidance、Feedback Guidance 或现有 solver-error 方法解释，没有新增可检验命题。
4. 只有在 RAEv2 单一 checkpoint 上出现现象，换到官方 AutoGuidance/PAG 就消失。
5. 最终方法退化成“搜索 layer + 搜索 scale + 搜索时间表”。这既昂贵，也已经被相邻工作覆盖。

## 九、当前不做什么

- 不继续 LPL 大训练。
- 不把未完成的 CIFAR 先导实验写成结论。
- 不自己从头实现 diffusion/flow sampler。
- 不先下载 19 GB Internal Guidance XL 权重。
- 不把“控制系统”当作创新点本身。
- 不再以单步 MSE 下降直接宣称生成质量会改善。

## 十、当前最准确的一句话

我们现在不是在“优化 RAEv2”，而是在审计一个更基础的问题：

> 为什么一个人为或训练得到的弱分支与强分支之差，有时能改善最终生成，而它的单步监督误差、方向对齐甚至符号都可能给出相反提示？

接下来应当在官方 EDM2/AutoGuidance 上复现并测清这个问题。只有这个问题在多个官方系统中成立，我们才考虑提出自适应 internal guidance；否则就应结束，而不是继续堆实验。
