# Internal Guidance、层间动力学与自适应控制研究报告

更新日期：2026-07-31

## 结论先行

我的建议是：**结束 LPL 主线，但可以继续研究 internal guidance；研究问题必须从“怎么把 guidance 做得更复杂”改成“内部方向为什么、何时能够改善最终轨迹”。**

不建议直接把 `base` 从一层扩成所有层，然后学习一组权重。这个朴素版本已经非常接近多类已有工作：中间层输出、随机丢层、选层或选 attention head、深层监督、层间对齐、动态 guidance scale。仅把它们组合起来，创新性和解释性都不够。

值得验证的核心问题是：

> 对同一个采样状态 `(x_t, t)`，不同网络深度产生的“剩余修正方向”是否构成一个低维、状态相关的控制基？我们能否在不知道真实终点的情况下，判断哪个方向会改善完整采样轨迹？

这条问题与现有实验有直接联系：RAEv2 的 `base -> full` 外推有效，而普通 LPL 修改 `full` 后不再稳定有效，说明重要的可能不是任意 feature matching，而是原模型内部已经共同训练并校准好的弱预测与强预测之差。

控制论和微分方程可以提供分析工具，但现在还不能直接把网络层解释成连续时间 ODE。采样时间 `t` 与网络深度 `l` 是两个不同坐标：前者是生成轨迹，后者是一次向量场评估内部的离散计算。更稳妥的建模是：

- sampler 是被控系统；
- 当前 latent `x_t` 是状态；
- 不同层产生的输出差是候选控制方向；
- 每一步是否启用、选哪个方向、用多大强度是控制量；
- 完整 rollout 的终点质量是最终效果。

只有先证明这些候选方向具有稳定、可预测的终点响应，才有资格进一步做反馈控制或状态空间建模。

## 1. 我们已经知道什么

### 1.1 LPL 应当结束

本仓库此前的严格实验已经把 ordinary LPL 的边界测清：

- LPL 在旧 RAE 上可以带来明显改善；
- 在 RAEv2 上，严格复现的 full LPL 没有稳定收益；
- detach、raw、方差归一化、不同外推方式没有把结果变成可复现方法；
- common/contrast-preserving 适配可以消除明显退化，但平均收益约在测量噪声量级；
- 单步 feature 目标改善不等于递归采样轨迹改善。

因此，不能继续通过换归一化、换层、换小权重来延长 LPL。保留它作为一个机制性负结果即可。

### 1.2 RAEv2 的 internal guidance 不是普通的“截断网络”

官方实现中，RAEv2 的 ImageNet DINOv3-L K7 模型使用 `DiTwDDTHeadIG`：

- 主干有 28 个 encoder blocks，随后有 2 个 DDT decoder blocks；
- `base_model_depth=8`；
- 第 8 层隐藏状态通过一个独立训练的 `base_final_layer` 输出 `base`；
- `full` 继续经过剩余 encoder blocks、投影、2 个 decoder blocks 和 final head；
- 训练时 `base` 和 `full` 分别对同一个 flow target 计算损失；
- 推理时使用 `base + scale * (full - base)`，并在指定时间区间启用。

这带来三个重要结论：

1. `base` 的质量来自独立输出头的共同训练，不是任意隐藏层天然可以直接解码。
2. `full - base` 同时包含网络深度差、输出头差和 DDT decoder 路径差。
3. 官方损失只保证二者各自拟合 target，没有直接监督差向量一定指向 `target - full`。

所以，“把每层作为 base”不能简单地复用 final head。否则层间比较会被 head calibration 混淆。

### 1.3 Internal Guidance 原始实现也是固定单层、固定区间、固定比例

[Internal Guidance](https://arxiv.org/abs/2512.24176) 的 LightningDiT 实现会在指定 `encoder_depth` 产生一个单独的中间输出 `zsr`，训练目标为 final MSE 加权 intermediate MSE，采样时计算：

```text
guided = full + (scale - 1) * (full - intermediate)
```

它没有判断当前样本的方向是否可靠，也没有在多个中间方向间选择。RAEv2 继承的是这一基本形式，只是模型结构和目标参数化不同。[RAEv2 论文](https://arxiv.org/abs/2605.18324)与[官方仓库](https://github.com/nanovisionx/RAEv2)提供了可复用的双头实现。

## 2. 文献边界：什么已经有人做了

### 2.1 固定弱模型到强模型的外推已经成熟

| 工作 | 弱预测来自哪里 | 推理控制 | 与本问题的关系 |
|---|---|---|---|
| [AutoGuidance](https://arxiv.org/abs/2406.02507) | 单独训练得更差或更小的模型 | 固定强度外推 | 说明“兼容的退化”比任意坏模型重要 |
| [Weak-to-Strong Diffusion](https://arxiv.org/abs/2502.00473) | weak/strong LoRA 或 CFG 设置 | reflection/zig-zag | 已覆盖一般 weak-to-strong 差分思想 |
| [Internal Guidance](https://arxiv.org/abs/2512.24176) | 一个中间层辅助头 | 固定层、固定时间区间、固定 scale | 最直接的基线 |
| [RAEv2](https://arxiv.org/abs/2605.18324) | 深度 8 的独立 base head | 固定 IG | 当前主实验平台 |
| [SGG](https://arxiv.org/abs/2603.20584) | CFG 与 skip-layer 弱模型 | 分段切换及 schedule | 已覆盖按采样阶段选择 guidance 家族 |

结论：简单提出“更弱的层引导更强的层”没有新颖性。

### 2.2 随机扰动、丢层和选择内部部件也已有密集工作

- [S²-Guidance](https://arxiv.org/abs/2508.12880) 用 stochastic block dropping 构造内部弱模型。
- [Spatiotemporal Skip Guidance](https://arxiv.org/abs/2411.18664) 用跳层构造视频生成的弱分支。
- [Token Perturbation Guidance](https://arxiv.org/abs/2506.10036) 通过尽量保范数的 token shuffle 构造扰动预测。
- [HeadHunter](https://github.com/cvlab-kaist/HeadHunter) 枚举并用 CLIP、PickScore、aesthetic score 选择 attention heads，再对所选 heads 做 perturbation guidance。
- [Focal Guidance](https://arxiv.org/abs/2601.07287) 明确识别 video DiT 中的 semantic-weak layers，并只对这些层恢复条件响应。

结论：随机扰动、找“坏层”、选择 head 或 layer 都不能单独作为贡献。尤其 HeadHunter 已经非常接近“搜索哪些内部部件适合扰动”，只是它依赖外部生成分数，并不是在线预测轨迹响应。

### 2.3 多层监督和层间对齐也不是空白

- [DeepFlow](https://arxiv.org/abs/2503.14494) 在多个深度分支上监督 velocity，并引入 acceleration/refinement 结构。
- [LayerSync](https://arxiv.org/abs/2510.12581) 用深层表示作为 stop-gradient teacher 对齐较浅层表示，并支持 multiple-layer sync。
- [Raptor](https://arxiv.org/abs/2512.19941) 把 ViT 深度看成表征轨迹，报告晚层更新呈低秩吸引结构，并用少数循环 block 近似原网络。

这些工作告诉我们：深度方向可能低秩、层间可能冗余，也可以训练时对齐。但它们没有回答 RAEv2 中多个中间输出方向对完整生成轨迹是否具有可预测的因果效果。

### 2.4 自适应 guidance 和最优控制也已有相邻工作

- [Feedback Guidance](https://arxiv.org/abs/2506.06085) 根据运行中的 posterior statistic 动态改变 CFG scale；方向仍是固定的 conditional-unconditional 差。
- [Variational Control for Guidance](https://arxiv.org/abs/2502.03686) 把 guidance 写成带 terminal cost 和 control cost 的轨迹控制，主要用于 inverse problems。
- [On the Guidance of Flow Matching](https://arxiv.org/abs/2502.02150) 给出 flow guidance 的一般理论框架，是分析 vector-field 修改时的重要约束。
- Adaptive Diffusion Guidance via stochastic optimal control 学习状态相关 guidance scale，但仍主要控制一个既定方向的强度，而不是从多个内部深度方向中辨识控制基。

结论：使用“反馈控制”“最优控制”“状态相关 scale”这些说法本身不新。可能的空白只剩下：**控制方向本身来自单模型内部深度轨迹，并且先通过终点响应辨识，而非靠外部目标暴力搜索。**

## 3. 附件分析中最合理的部分

附件提出的局部判据是正确而重要的。记：

```text
f = full prediction
b = base prediction
z = supervised target
d = f - b
e = f - z
g = f + gamma * d
```

则：

```text
||g-z||^2 - ||f-z||^2
= 2 gamma <e,d> + gamma^2 ||d||^2
```

对正的 `gamma`，要让小幅外推改善误差，需要：

```text
<f-b, z-f> > 0
```

对应的局部最优系数是：

```text
gamma* = <f-b, z-f> / ||f-b||^2
```

这比只看 `base loss > full loss` 强得多。base 更差不代表 `full-base` 指向 target 之后；它可能与剩余误差正交，甚至反向。

若有多个候选方向 `d_1,...,d_m`，令 `D=[d_1,...,d_m]`，局部二次问题为：

```text
min_u ||f + D u - z||^2 + lambda ||u||^2
```

其统计量是：

```text
c_i = <d_i, f-z>
G_ij = <d_i, d_j>
u* = -(G + lambda I)^(-1) c
```

`G` 测量方向冗余，`c` 测量方向是否修正当前残差。这个分解可以成为机制审计的核心。

但附件中“直接学习多层 controller”的部分需要后移，因为训练时可以访问 `z`，采样时无法访问。若一个 controller 只有使用真实 target 才有效，它只是 oracle，不是生成方法。

## 4. 微分方程和状态空间视角怎样用才严谨

### 4.1 不应把 layer index 直接当作采样时间

一次 flow 求值可以写成：

```text
v_theta(x_t, t) = H_L o F_L o ... o F_1(x_t, t)
```

网络深度 `l` 是计算图内部坐标，采样时间 `t` 是外部 ODE 坐标。各层参数不共享，维度语义也可能变化，因此：

- 不能因为 residual block 形如 `h_{l+1}=h_l+Delta_l` 就宣称它是同一个自治 ODE；
- 不能把 `full-base` 自动解释成高阶积分误差；
- 不能直接套稳定性、可控性结论而不做实验辨识。

### 4.2 可以把候选内部差分当作输入通道

在 sampler 的第 `k` 步，合理的离散系统写法是：

```text
x_{k+1} = Phi_k(x_k, v_full(x_k,t_k) + B(x_k,t_k) u_k)
```

其中：

- `B=[d_1,...,d_m]` 是若干内部方向；
- `u_k` 是选择和强度；
- `Phi_k` 是原 sampler step；
- `u_k=0` 必须严格恢复官方 sampler。

这就是一个 data-driven control 问题。首要对象不是神经 controller，而是 impulse response：在单个步骤加入很小、等能量的 `+epsilon d_i` 或 `-epsilon d_i`，然后恢复官方采样，观察最终样本如何变化。

### 4.3 x-pred 参数化存在时间放大问题

RAEv2 训练使用 x-pred，但 sampler 最终需要 velocity。官方转换为：

```text
v = (x_t - x_pred) / max(t, t_eps)
```

因此对 x-pred 添加固定尺度的扰动，会在较小 `t` 处变成约 `1/t` 放大的 velocity 扰动。这会导致：

- 相同 x-pred 差分在不同 `t` 上不等能量；
- 低噪声区域可能非常 stiff；
- 比较各层方向时必须在 velocity 或实际 step displacement 上归一化，而不能只比较 x-pred norm。

这也是固定 IG interval 有意义的原因之一，也是任何自适应方法必须显式处理的因素。

## 5. 真正可能成立的研究假设

### H1：方向可靠性随采样时间和样本状态变化

同一层的 `full-base` 可能在部分 `t` 上与剩余误差正相关，在另一些 `t` 上无效或反向。固定 interval 是数据集平均折中，自适应方法的收益来自避免错误区间，而不是任意增大 guidance。

### H2：多层方向高度冗余，只存在很低维的有效控制基

相邻层的输出差很可能共线。若前三个奇异方向已经解释 90% 以上能量，那么“每层一个系数”没有必要；应当控制低维正交基，并把冗余本身作为内部动力学发现。

### H3：局部 target alignment 不足以预测最终生成质量

此前 LPL 已经显示单步目标与递归轨迹之间存在断层。一个方向即使降低当前 x-pred MSE，也可能经后续非线性放大后恶化终点。因此必须比较：

- local alignment；
- one-step state displacement；
- full-rollout endpoint response。

若三者不一致，这本身就是值得发表的机制结果。

### H4：官方 base 的价值来自共同训练后的 calibration，而不只是“更浅”

若任意冻结 probe 都不如官方 depth-8 base，而只要与 full 共同训练才有效，说明 internal guidance 依赖的是一对协同形成的预测器。这个结论会直接否定“任意层都是可用控制方向”，也解释旧 RAE 与 RAEv2 的差异。

## 6. 推荐实验路线

### Phase A：先审计现有双头，不训练任何参数

平台：RAEv2 官方 ImageNet checkpoint，保持官方 sampler、EMA、步数、CFG/IG 设置。

采集同一批真实 flow 配对上的：

- `full`、`base`、target；
- `||full-target||` 和 `||base-target||`；
- `cos(full-base, target-full)`；
- 局部 `gamma*` 的分布；
- 按 `t` 分桶后的正向比例；
- x-pred、velocity、实际 solver displacement 三个坐标中的 norm。

再做 paired scale sweep：

- `scale in {0, 0.5, 1, 1.25, 1.5, 2}`；
- 固定噪声、标签和 solver；
- 同时评估一步误差与完整 rollout；
- 使用官方 fixed IG 作为必须超过的主基线。

目标：确认官方方向为什么有效，以及局部判据是否能预测最终改善。

### Phase B：冻结 backbone，训练稀疏层 probe

不要一开始每层都放 head。先选 `{4,8,12,16,20,24,28}`，每层使用完全相同结构、相同初始化规则和相同训练预算的 probe：

- backbone、stage-1 tokenizer、decoder 全部冻结；
- 每个 probe 独立拟合同一个 x-pred target；
- probe 不反传到 backbone；
- train/val 图像严格划分；
- 官方 depth-8 head 保持原样，作为 calibrated reference；
- 加一个参数量匹配的 shared-head baseline，检查 head 差异。

需要报告：

- 每个 probe 的 train/val MSE；
- 与官方 full 的 gap；
- 与剩余误差的 alignment；
- 各方向 Gram matrix 与有效秩；
- 每个 `t` 区间的最佳层是否稳定；
- 层深本身能否解释方向质量，还是 probe calibration 才是主要因素。

只有 probe 的 held-out target MSE 达到相近量级后，才能公平比较层方向。

### Phase C：做 impulse-response atlas

对每个候选方向，在某一个 sampler step 施加等 step-displacement 能量的微扰：

```text
v' = v_full + epsilon * normalize(d_i)
```

随后立即恢复官方 sampler，完成剩余 rollout。必须包含 `+epsilon`、`-epsilon`、随机等能量方向和 official IG direction。

记录：

- 终点 RAE latent 位移；
- stage-1 decoder feature 位移；
- DINO/Inception feature 位移；
- 类别置信度与多样性变化；
- 小规模 paired perceptual quality；
- 方向、时间和样本之间的方差分解。

这一步回答的是因果问题：某层方向在某一时刻被注入后，最终发生了什么，而不只是它当时的 loss 是多少。

### Phase D：满足门槛后才做自适应控制

首个 controller 不应是大网络。按复杂度依次比较：

1. official fixed IG；
2. validation 上最佳固定层；
3. 只依赖 `t` 的 layer/scale schedule；
4. 基于 direction norm、角度、base-full disagreement 的小型 ridge/logistic gate；
5. 在低秩方向基上的正则化线性控制；
6. oracle target-aware controller，仅作为上界，绝不能作为可部署方法。

只有 4 或 5 在 held-out state 上超过 2 和 3，才能说明“状态反馈”真的有价值。

## 7. 验收与停止标准

### 7.1 Phase A 通过条件

- official direction 的正向 residual alignment 在主要有效区间显著高于 50%；
- 局部 alignment 或 `gamma*` 与 paired endpoint improvement 有稳定相关性；
- 结论至少在 4/5 seeds 或噪声批次方向一致；
- x-pred 到 velocity 的尺度换算和 interval 效果能解释结果，而不是实现错误。

若 alignment 接近随机，且无法预测 endpoint，立即停止“残差反馈 controller”路线。

### 7.2 Phase B/C 通过条件

- 至少两个非官方层方向在 held-out 数据上具有稳定正向终点响应；
- 多层方向不是全部退化成 official direction 的缩放；或者虽低秩，但存在至少两个功能不同的主方向；
- direction response 在时间与状态上存在可预测结构，而不是纯噪声；
- 简单预测器对 endpoint improvement 的 AUROC 至少约 0.70，或 Spearman 相关至少约 0.5；
- 正负 impulse 产生近似反对称的小扰动响应，说明处于可线性分析范围。

若所有方向都与 official base 共线，研究应收敛为“为什么单一 calibrated direction 有效”，而不是强做多层控制。

### 7.3 方法级通过条件

- 三个及以上 seeds 的 5k 快速评估全部优于 official fixed IG；
- 50k 终验在同一 checkpoint、solver、NFE、EMA 和评估管线上显著改善；
- FID 改善不能以 precision、recall 或类别覆盖明显恶化为代价；
- 方法不依赖真实 target、外部 reward model 在线搜索或额外完整模型 forward；
- 相比最强固定层和最强 `t`-only schedule，仍有稳定增益。

如果只比无 guidance 好，却比不过官方 fixed IG，这不是成功。

## 8. 计算资源建议

这条路线可以控制成本：

- Phase A 只需现有 checkpoint 的 forward 与采样，不训练主模型；
- Phase B 只训练小 probe，四卡按层或数据 shard 并行；
- 不建议缓存所有层的完整 ImageNet hidden states，会产生很大存储开销；优先流式提取选定层；
- 机制筛查先用 1k paired samples，方法比较再用 5k，只有通过后做 50k；
- 不在早期训练 neural controller，也不重新训练 RAEv2 主干。

## 9. 最终建议的论文问题表述

不建议：

> 我们把 Internal Guidance 扩展为多层自适应 guidance。

更扎实的表述是：

> Existing internal guidance extrapolates a fixed shallow-to-deep prediction gap, yet it remains unknown whether this gap is a reliable correction direction along the generative trajectory. We identify the state- and time-dependent endpoint response of internal depth directions, show that their useful component is low-dimensional and calibrated rather than uniformly layer-wise, and derive a feedback rule that activates only causally beneficial directions.

中文含义：

> 现有 internal guidance 固定外推浅层到深层的预测差，但这个差是否在整条生成轨迹上真的是可靠修正方向仍不清楚。我们辨识内部深度方向随状态和时间变化的终点响应，发现真正有用的是低维且经校准的方向，而不是所有层，并据此只启用具有因果收益的方向。

这条表述的核心贡献是“辨识与可预测性”，不是把若干 guidance 技巧拼起来。

## 10. 本地代码审计清单

以下官方仓库已浅克隆到本地 `research_repos/internal_guidance_study/`，未下载权重、数据或依赖，也未执行上游代码。目录已 gitignore，只在本报告记录版本。

| 仓库 | commit | 主要审计目的 |
|---|---:|---|
| `RAEv2` | `8a0d238f8dc3` | 双头 DDT、x-pred、IG 公式和训练损失 |
| `Internal-Guidance` | `d048cc3e1320` | 原始单层辅助头和采样外推 |
| `edm2` | `4bf8162f601b` | AutoGuidance 的独立弱模型外推 |
| `Weak-to-Strong-Diffusion` | `c1c160d63483` | weak/strong reflection 路径 |
| `S2-Guidance` | `560bc457b9aa` | stochastic block dropping；当前仓库只公开 README/资产 |
| `DeepFlow` | `be34b201439c` | 多深度 velocity/acceleration supervision |
| `LayerSync` | `e621ff2b25fe` | shallow-to-deep stop-gradient alignment |
| `Feedback-Guidance` | `cec69babd78c` | posterior-driven dynamic scalar guidance |
| `flow_guidance` | `b47872e9f72c` | flow guidance 理论与实现边界 |
| `oc-guidance` | `1d20e30bf0c1` | terminal/control cost 与轨迹优化 |
| `HeadHunter` | `e14c5cb40eed` | 基于外部分数搜索可扰动 attention heads |
| `Token-Perturbation-Guidance` | `6fb4464a9906` | token 内部扰动弱模型 |
| `STGuidance` | `d9e7be5dadfc` | skip-layer video guidance |
| `SGG` | `1a30588bd483` | segmented CFG/skip-layer guidance |
| `raptor` | `2e6fac2155c1` | 深度轨迹、循环 block 与低秩晚层更新 |

## 11. 下一步唯一推荐动作

先实现 **RAEv2 official dual-head guideability audit**，而不是多层 controller：

1. 复用官方 checkpoint 和 sampler；
2. 按 `t` 记录 full/base/target 的 residual alignment 和局部 `gamma*`；
3. 做同噪声 paired 的单步与完整 rollout scale sweep；
4. 验证 local metric 是否预测 endpoint improvement；
5. 只有通过预设门槛后，再训练稀疏层 frozen probes。

这是当前信息增益最高、成本最低、最不容易自欺的一步。
