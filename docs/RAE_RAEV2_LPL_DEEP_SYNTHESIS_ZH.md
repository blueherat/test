# RAE / RAEv2 上 LPL 探索的深度综合

## 一句话结论

这批实验并不是简单地得到“LPL 在 RAEv2 上失败”。

真正有价值的发现是：

> 旧 RAE 中，decoder-aware 的逐样本修正可以补偿一个尚未被处理的
> 生成质量缺口；RAEv2 的 internal guidance 已经用 weak-to-strong
> 外推处理了相当一部分同类缺口。更根本的是，RAEv2 的官方 guidance
> 会主动牺牲逐样本 clean-latent 和 decoder-feature 误差，却显著改善
> 最终生成分布。因此，paired endpoint error 并不是 guided generation
> 的可靠优化代理。

这比“LPL 权重不合适”或“RAEv2 已经太强”更具体，也更可检验。

## 1. 我们究竟从哪里开始

LPL 的原始问题是 latent diffusion 与冻结 decoder 的脱节：

```text
训练 Stage-2 时只最小化 latent 误差
最终评价却在 decoder 输出的图像空间进行
```

原始 LPL 用冻结 AE decoder 的中间特征比较预测 clean latent 与真实 clean
latent。它仍然是逐样本、逐噪声状态的监督，只是把欧氏 latent 误差换成更靠近
图像空间的误差。

论文公开公式的关键点是：

```text
z_hat = 当前 noisy latent 对应的 clean-latent 估计
L_LPL = decoder_features(z_hat) 与 decoder_features(z) 的归一化平方差
```

原论文在预训练后期加入 LPL，并在 DDPM 与 Flow-OT 上报告收益。其目标使用
预测特征的统计量同时归一化 target 和 prediction，因此 prediction variance
不仅是权重，也参与反向传播。

来源：

- [Boosting Latent Diffusion with Perceptual Objectives](https://arxiv.org/abs/2411.04873)
- [LPL, ICLR 2025 proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/file/204fee94c982a19230c39045aa54f977-Paper-Conference.pdf)

## 2. 旧 RAE 实验实际证明了什么

### 2.1 生成收益是真实的

在冻结 RAE encoder/decoder、只更新 Stage-2 prior 的严格对照中：

- DINOv2-B：三 seed 的 5k 结果均满足 `full LPL < detach < Flow`；
- DINOv2-B 50k ADM FID：`13.5043 -> 12.4540 -> 11.1027`；
- MAE-B、SigLIP2-B：各三 seed 也都满足相同排序；
- 相对同协议官方起点，Flow、detach、full 分别有 `0/9`、`4/9`、`9/9`
  个 seed 改善 FID。

训练没有使用 validation/reference，没有更新 encoder/decoder，Flow/LPL 的
图像、标签、时间、噪声和初始预测均严格配对。这里可以可靠地说：

> RAE-adapted LPL 在旧 RAE 的这组 post-training 设置中确实提供了超出
> 等步数 Flow continuation 的生成收益。

### 2.2 最初的“低敏感方向”解释被反证

我们最初猜测 LPL 会把 latent 误差移到 decoder Jacobian 的低敏感方向。
实际结果相反：

| 指标 | LPL / Flow |
|---|---:|
| latent MSE | 1.0092 |
| clean latent 附近局部 decoder 放大 | 1.1217 |
| 真实误差半径 strict decoder error | 0.8768 |

LPL 的方向在无穷小邻域更敏感，但在真实 Stage-2 误差半径上更好。真实误差
RMS 约为 clean latent RMS 的 `18%--41%`，已经不属于可靠的一阶 Taylor
区域。

因此，LPL 不是在寻找 decoder 的局部零空间，而是在有限距离上改变预测状态。

### 2.3 旧 RAE 中最稳定的可见变化

在同范数真实误差半径上：

```text
Flow decoder-feature variance / clean = 0.8486
LPL  decoder-feature variance / clean = 0.9322
```

同时 centered feature cosine 从约 `0.795` 提高到 `0.804`。这说明旧 Flow
预测有明显的 feature contraction，LPL 同时恢复 feature scale 并略微改善
内容方向。

### 2.4 方差分母确实是主要有效组成，但不是全部

把原 LPL 梯度写成：

\[
\nabla(N/V)=\frac{1}{V}\nabla N-\frac{N}{V^2}\nabla V.
\]

旧 RAE 的 detach 因果消融表明：

- `N / stop_gradient(V)` 本身有稳定收益；
- 允许梯度经过 prediction variance 分母会进一步改善；
- DINOv2-B、MAE-B、SigLIP2-B 中，detach 分别保留 full FID 收益的约
  `39%`、`16%`、`21%`；
- 直接 denominator gradient 在三种 tokenizer 上都占多数，但比例不同。

因此旧 RAE 的 full LPL 不能被简化为普通 feature MSE。它包含一个很强的
反 feature-contraction 更新。

## 3. RAEv2 的 prior 和 internal guidance 到底怎样工作

### 3.1 RAEv2 K7 latent

当前官方模型使用 DINOv3-L 的七个层：

```text
11, 13, 15, 17, 19, 21, 23
```

这些层的 token 按元素相加，latent 形状仍为：

```text
[B, 1024, 16, 16]
```

它不是只用 DINOv3 最后一层，也没有把七层沿 channel 拼接。

### 3.2 Stage-2 的两个输出

`DiTwDDTHeadIG` 有 28 个主干 encoder blocks 和 2 个宽 decoder blocks：

```text
base:
  取第 8 个 block 后的 token
  -> 独立 base_final_layer
  -> clean latent 预测 x_base

full:
  继续通过全部 28 个 blocks
  -> 2 个 wide decoder blocks
  -> final_layer
  -> clean latent 预测 x_full
```

两个头都直接预测 clean latent，而不是旧 RAE 的 velocity。训练时两者都和
同一个 Flow target 比较：

```text
loss = loss_full + loss_base
```

代码先把 x-prediction 转成：

```text
v = (x_t - x_hat) / t
```

再与真实 Flow velocity 比较。因此在 `t` 未触及 clamp 时，它等价于对两个
clean-latent 误差分别施加 `1/t^2` 权重。当前官方 ImageNet 配置没有额外启用独立的
旧式 REPA projector loss；这里的 early x-prediction head 本身就是被重新
解释后的 REPA / intermediate supervision。

这个双头训练存在一个容易忽略的精确恒等式。记：

```text
m = (full + base) / 2
d = full - base
```

两个头使用相同 target `y` 和相同权重，所以：

\[
\|full-y\|^2+\|base-y\|^2
=2\|m-y\|^2+\frac12\|d\|^2.
\]

x-prediction 的 `1/t^2` 会共同乘在等式两边，不改变这个分解。因此官方训练
目标本身在压小 `d`；若两个头都达到同一个 Bayes 最优条件均值，理论上
`d=0`，internal guidance 也会消失。实际有效的 `d` 来自 base/full 的
深度容量差异和有限训练偏差，而不是一个直接以“提高生成质量”为标签监督出的
方向。

采样时 `x_guided=m+1.28d`。这形成一个结构性张力：训练把 `d` 当成两个
预测器尚未消除的差异来惩罚，采样却把它当成质量方向放大。它不说明 internal
guidance 错了，但说明改善单头 paired objective 的 post-training 很可能
改变一个没有被显式稳定的采样坐标。这是旧 RAE 单头与 RAEv2 双头表现相反的
最简结构解释之一。

对应实现：

- `external/RAEv2/src/stage2/models/DDT.py`
- `external/RAEv2/src/stage2/transport/transport.py`
- `external/RAEv2/configs/stage2/training/imagenet-dinov3l-k7.yaml`

### 3.3 采样时的 internal guidance

官方代码使用：

```text
x_guided = x_base + 1.78 * (x_full - x_base)
         = x_full + 0.78 * (x_full - x_base)
```

因此：

- `1.0` 才是直接使用 full；
- `1.78` 是越过 full 再走 `0.78` 倍差值；
- 论文公式中的 `w=0.78` 对应代码配置的 `scale=1.78`。

这个 guidance 在 `t in [0.10, 1.0]` 生效，最后低噪声区间直接使用 full。
guided clean prediction 再转回 velocity，用 100-step Euler ODE 采样。

RAEv2 的时间 shift 为：

```text
sqrt(262144 / 4096) = 8
```

采样网格先从 `1 -> 0` 均匀取点，再经过该 shift。按官方 100-step 配置，
`t >= 0.10` 实际覆盖前 99 次模型评估，仅最后一次评估关闭 IG。因此不能把
当前配置理解为只在一小段高噪声区间使用 guidance；它覆盖了几乎整条生成
轨迹，只避开最后的最低噪声端。

CFG scale 为 `1.0`，所以当前 ImageNet 采样没有启用 CFG。训练中的
`10%` condition dropout 仍然存在，但不是当前结果的 active guidance。

它只需要一次主模型前向：base 是同一个模型中途取出的弱预测，不需要第二个
unconditional 模型，也不需要另训一个 AutoGuidance 弱模型。

对应实现：

- `external/RAEv2/src/utils/guidance_utils.py`
- `external/RAEv2/configs/stage2/sampling/imagenet-dinov3l-k7.yaml`

官方论文报告 K7：

| guidance | 50k gFID |
|---|---:|
| 无 guidance | 1.65 |
| CFG | 1.49 |
| AutoGuidance | 1.14 |
| REPA / internal guidance | 1.06 |

来源：

- [Improved Baselines with Representation Autoencoders](https://arxiv.org/abs/2605.18324)
- [RAEv2 官方项目页](https://raev2.github.io/)
- [Guiding a Diffusion Transformer with the Internal Dynamics of Itself](https://arxiv.org/abs/2512.24176)

## 4. RAEv2 上为什么出现相反结果

### 4.1 strict LPL 的第一层失败：分母捷径

在严格相同的 ImageNet train 数据、官方完整 optimizer/scheduler、global
batch 1024 和相同噪声下，RAEv2 strict LPL 会降低自己的 normalized loss，
但：

- clean-latent error 增加；
- 固定 target variance 的 decoder-feature error 增加；
- prediction feature variance 增加；
- 5k FID 相对 Flow10 从 `10.8278` 退化到 `10.9289`。

切断 denominator gradient 后：

```text
detach 5k FID = 10.8518
```

完全删除方差分母后：

```text
raw 5k FID = 10.8414
```

detach 消除了 strict 约 `76%` 的 FID 伤害，raw 消除了约 `86.5%`。这证明
RAEv2 的 strict 分母梯度确实是额外退化的重要原因。

当前 LPL 也不是在“高噪声段”训练。门控条件为：

```text
t / (1 - t) <= 3
```

等价于 `t <= 0.75`，即相对高信噪比段。按 RAEv2 的 shift=8、100-step
Euler 时间网格，它与 `IG active when t >= 0.1` 的重叠落在采样 update
`73...98`，共 26 步；对应 `t=0.7474...0.1404`。因此当前冲突优先发生在
采样后段，而不是最初从高斯噪声建立粗语义的阶段。这也给后续 segmented
head-swap 提供了明确时间区间，但在完整 `2 x 2` 交换归因前不先扩大实验。

### 4.2 第二层失败：正确优化 paired proxy 仍不等于改善生成

这个结论不能停在“分母写错”：

- detach 确实改善了固定 target-stat 的相对 feature 对齐；
- raw 确实把自己的未见图像 raw feature MSE 降低 `8.65%`；
- target-normalized、symmetric 也能正确降低各自声明的 proxy；
- 这些分支仍没有稳定超过 Flow；
- flow-parallel LPL 与 Flow 的 5k FID 几乎完全相同：
  `10.828043` 对 `10.827822`。

在同一组 16 张未见图像、三个 noise/signal ratio 上取几何平均后，分支结构
更清楚：

| variant | base raw feature | full raw feature | guided raw feature | guided target-normalized |
|---|---:|---:|---:|---:|
| strict vs Flow | -15.74% | -4.77% | +3.03% | +2.28% |
| raw vs Flow | -17.32% | -7.90% | -1.05% | +0.52% |

strict 不是简单地“什么都没优化好”：它分别改善了 base/full 的 raw feature
误差，但两者经过 `base + 1.78(full-base)` 组合后反而变差。这是
internal-guidance 差值代数被扰动的直接证据。raw 则连 guided raw feature
也小幅改善，却仍未改善 5k FID。这又证明即使修复了第一层组合问题，paired
proxy 与 rollout distribution 之间仍有第二层断层。

所以安全的 LPL 分量只是重复 Flow；相对 Flow 真正新增的切向更新，在成熟
RAEv2 上反而伤害生成。

## 5. 最反常也最有价值的新事实

我们从现有 validation raw table 重新计算了官方 RAEv2 的 `full -> guided`
变化。这里没有 LPL，只看官方 internal guidance 自己做了什么。

| noise/signal | latent error | target-normalized feature error | raw feature error | feature variance |
|---:|---:|---:|---:|---:|
| 0.5 | +11.83% | +19.24% | +18.43% | 0.923 -> 0.959 |
| 1.0 | +10.82% | +12.90% | +7.88% | 0.900 -> 0.958 |
| 3.0 | +8.55% | +6.86% | -7.17% | 0.814 -> 0.900 |

`full-base` 与“从 full 指向 paired clean target”的余弦平均约为 `-0.028`：
它几乎正交，甚至轻微朝逐样本 MSE 的错误方向。internal guidance 后的
clean-latent RMS error 平均增加约 `10.5%`。

这里不与“full/base 两个误差的余弦约为 `0.85`”矛盾。设：

```text
e_full = full - clean
e_base = base - clean
d      = full - base = e_full - e_base
```

两个较大的误差向量可以高度同向，但相减后留下的小残差 `d` 可以几乎垂直于
二者。IG 使用的正是这个被共同误差抵消后剩下的差值，而不是 full 或 base 的
主要 MSE 方向。

但官方 50k 结果中，它把 gFID 从 `1.65` 改善到 `1.06`。

这说明：

> Internal guidance 的作用不是让每个 noisy input 对应的 clean-latent
> 条件估计更准确。它故意越过 MSE 较优的 full prediction，牺牲 paired
> reconstruction，换取更高密度、更锐利或更少 outlier 的生成分布。

这个现象可以由条件均值的基本性质解释。固定 `x_t`，若：

```text
mu(x_t) = E[clean | x_t]
```

且 full 已接近 `mu`，那么对任意只依赖 `x_t` 的非零外推 `delta`：

\[
\mathbb E[\|clean-(\mu+\delta)\|^2\mid x_t]
=
\mathbb E[\|clean-\mu\|^2\mid x_t]+\|\delta\|^2.
\]

也就是说，越过条件均值必然增加期望 paired MSE。guidance 仍可能改善 FID，
因为它改变的是 ODE 终点的整体密度、outlier 和 fidelity/diversity 权衡，而
不是恢复当前训练 pair。我们的 `+10.5%` paired error 与 `1.65 -> 1.06`
gFID 并非偶然矛盾，而是这两个目标不等价的直接实证。

这正是我们此前 endpoint LPL 逻辑在 RAEv2 上遇到的根本矛盾：

```text
LPL 问：每个预测是否更接近它配对的 target？
guidance 问：整批生成是否被推向更好的分布区域？
```

两者不是同一个问题。

## 6. 目前最统一的机制解释

可以把旧 RAE 和 RAEv2 放进同一条“回归到均值 / 分布锐化”轴上理解。

### 旧 RAE

- 单头、无 active guidance；
- Flow 的 decoder feature variance 明显收缩；
- LPL 的 prediction-statistics gradient 提供反收缩更新；
- 固定 target-stat 对齐也同步改善；
- 因而 LPL 既补尺度又补内容，FID 改善。

### RAEv2

- base/full 双头形成 weak-to-strong 差值；
- internal guidance 在采样时越过 full；
- 它已经显著恢复 decoder feature variance；
- 它甚至主动让 paired endpoint error 变差来换生成分布变好；
- strict LPL 再次施加 prediction-variance 驱动，容易过校准；
- detach/raw 去掉大部分过校准后，只能回到 Flow 附近，因为 paired
  feature matching 对 guided distribution 没有稳定新增信息。

因此目前最准确的表述不是：

```text
RAEv2 太强，所以 LPL 没用
```

而是：

> 旧 RAE 的 decoder-aware correction 与 RAEv2 的 weak-to-strong
> internal guidance 很可能在承担部分重叠的“反均值收缩 / 质量外推”
> 职责；更根本地，RAEv2 的最优采样输出本来就不是 paired endpoint loss
> 的最优解。

这里仍要保留两个可以被实验区分的竞争解释，不能提前把其中任何一个写成定论：

1. **重复方差恢复**：strict LPL 的 prediction-variance gradient 与
   internal guidance 都在补偿条件回归造成的 feature contraction。旧 RAE
   缺少 active guidance，所以前者有益；RAEv2 已经通过 `1.78` 外推完成主要
   修正，再叠加 strict LPL 容易过量。
2. **差值方向旋转**：即使 feature variance 没有明显过量，LPL 也可能改变
   `full-base` 的方向或跨时间连贯性，从而损坏 internal guidance 使用的质量
   contrast。

二者的可证伪预测不同：

- 若主要是重复方差恢复，降低 IG scale 应系统性恢复 LPL checkpoint 的排序，
  detach/raw 也应随训练长度明显优于 strict；
- 若主要是方向旋转，`2 x 2` head-swap 中替换 LPL contrast 会直接造成退化，
  且单独调小 scale 只能减弱、不能根治；
- 若两者都不成立，就应把差异归因回 full vector field 或 ODE 轨迹交互，而
  不是继续讲“decoder mismatch”。

现有 1k scale sweep 在 `IG=1.0/1.2/1.4/1.6/1.78` 上有明显抽样噪声，尚未
出现足以确认“最佳 scale 系统性左移”的单调证据。因此当前只能说 IG 是退化
放大器，不能说重复方差恢复已经得到唯一确认。

## 7. 哪些是因果证据，哪些仍是推断

### 已经可靠确认

1. 旧 RAE 中 full LPL 跨 tokenizer、跨 seed 改善相对 Flow 的生成结果。
2. 旧 RAE 的直接 denominator gradient 是多数收益的因果组成。
3. RAEv2 strict denominator gradient 是多数额外退化的因果组成。
4. RAEv2 internal guidance 是 LPL 退化的主要放大器；关闭 IG 后，step100
   的相对 FID 伤害缩小约 `74.6%`。
5. detach/raw 正确优化了各自 proxy，但没有可靠超过 Flow。
6. 官方 internal guidance 会显著恶化 paired latent/feature error，同时
   显著改善官方 gFID。
7. 训练无 validation/reference 泄露，encoder/decoder 冻结，分支数据、
   noise、label 和 source state 严格配对。

### 强解释，但还没有被唯一确认

1. LPL 与 internal guidance 共享一部分“反 feature contraction”职责。
2. RAEv2 的失败来自质量外推已经饱和，而不只是模型规模或 DINOv3 latent。
3. `full-base` 的方向变化比两个头各自的 endpoint loss 更能解释 LPL FID。

### 目前不能声称

1. DINOv3、K7、多层 latent、x-prediction、GMuon 或模型规模中的哪一个单独
   造成 LPL 失效。
2. RAEv2 从训练早期开始使用 LPL 也一定失败；当前是从 80 epoch 官方完整
   状态出发的 post-training 结论。
3. “feature variance 越接近 1，FID 就必然越好”。
4. 仅凭 10-step 的细小 FID 差就能判断 detach/raw 的最终排序；50-step
   曲线仍需完成。

## 8. 不做方法缝合后的研究主线

### 核心科学问题：paired-to-distribution inversion

最值得研究的问题不是再设计一种 normalization，而是：

> 为什么 weak-to-strong guidance 明明让逐样本 latent 和 decoder-feature
> 预测更差，却让生成分布更好？什么统计量能够识别这种“有用的偏差”？

这是一条单一问题主线。它不需要把 LPL、REPA、CFG 和其他正则堆在一起。

### 第一候选研究：Useful Error Geometry

RAEv2 中应把以下三个方向分开：

```text
r = clean - full
d = full - base
p = -grad_full decoder_feature_loss(full, clean)
```

这里 `r` 是逐样本回归希望修正的方向，`d` 是采样器实际用于外推的方向，
`p` 是 LPL 想推动 full prediction 的方向。真正需要回答的是：

- `d` 为什么在与 `r` 几乎正交时仍能改善生成分布；
- LPL 是损坏了 full vector field，还是损坏了 `d` 的方向和跨时间一致性；
- `d` 是否主要恢复真实 feature distribution、压低 outlier，而不是降低
  paired MSE；
- detach/raw 接近 Flow，是否因为它们更少扰动 `d`。

如果成立，论文层面的发现会是：

> 在 self-guided generative model 中，弱头的“误差”不是待消除的噪声，而是
> 采样器使用的质量方向；改善两个头的单独预测，不保证改善它们的差值场。

这个问题比“LPL 和 guidance 都在锐化”更严格，因为它要求用方向、轨迹和
交换实验给出因果证据。

现有标量结果已经给出一个进一步线索：strict LPL10 相对 Flow10 只把
`||full-base||` 平均增大约 `0.36%`，detach/raw 则只增大约 `0.04%`；但
strict 的 5k FID 退化远明显于后二者。这说明单纯的 contrast magnitude
不足以解释结果，优先应检查方向和沿 ODE 时间的连贯性。

一个有解释力但尚待验证的模型是“质量切线”：

```text
f_s(x_t) = 模型能力/深度为 s 时的 clean prediction
full - base ≈ Delta_s * partial f_s / partial s
```

base 与 full 是同一模型在两个深度上的预测，因此二者差值可以被看成沿
“模型质量轴”的有限差分。IG 不是沿 paired target 继续下降，而是把 full
向“更强一点的模型可能会给出的结果”外推。这个解释能同时容纳三件事：

1. `d` 不需要指向当前 paired clean target；
2. `d` 仍可能减少生成分布中的 outlier；
3. 同时微调共享主干和两个头，即使改善各自 loss，也可能旋转这条质量切线。

它目前只是一个可证伪假设。若后面的 head-swap 不能把退化归因到 `d`，或
`d` 的跨 checkpoint / 跨时间变化不能预测 FID，就应放弃“质量切线”解释。

当前还出现了一个更具体的“差值场脆弱性”假设。base 与 full 的误差余弦约为
`0.85`，`d=full-base` 是两个高度相关大向量相减后的小残差。参数更新即使
几乎共同地移动两个头，只要留下很小的非共同分量，也可能明显旋转 `d`；
`IG=1.78` 又会把该分量乘以 `0.78` 加回 full。

已有结果与此一致：

- LPL 新增 pixel update 与纯 Flow update 的余弦仅约 `0.05`；
- full-only LPL 相对 Flow 的 FID 差，在 `IG=1.0` 时为 `+0.1668`，到
  `IG=1.78` 时被放大为 `+0.9204`；
- full+base 的额外 pixel update 只有 Flow update 的约 `0.426`，而
  full-only 为约 `0.891`，前者也确实更少退化；
- strict/raw 对 base 的 raw decoder-feature error 改善大于 full，说明
  weak reference 的质量含义可能同时被改变。

最后一条不能单独解释全部结果，因为 full-only 分支也失败。更统一的待验证
机制是：

> decoder-aware post-training 沿 Flow loss 约束较弱的方向移动两个头；
> self-guidance 对两个相关预测做差，使这些微小的 differential updates
> 比 common updates 更容易影响最终采样。

这个假设还能给出一个更精确、可直接做 Hessian-vector product 检验的局部
机制。设对单个头的 decoder-feature 目标为 `phi(u, y)`，同时监督两个头：

\[
L_{\mathrm{aux}}=\phi(m+d/2,y)+\phi(m-d/2,y),
\quad
m=(full+base)/2,\ d=full-base.
\]

当 `d` 足够小时，一阶展开给出：

\[
\nabla_m L_{\mathrm{aux}}\approx 2\nabla\phi(m,y),
\qquad
\nabla_d L_{\mathrm{aux}}\approx \frac12 H_\phi(m,y)d.
\]

因此一次辅助更新对 guidance contrast 的局部作用近似为：

\[
d_{\mathrm{new}}\approx
\left(I-\eta H_\phi(m,y)\right)d.
\]

普通 latent MSE 的 Hessian 近似各向同性，主要均匀缩小 `d`；decoder-feature
loss 的 Hessian 含有 `J_D^T J_D` 及残差曲率，会按 decoder 几何对 `d`
做各向异性的缩放，因而即使 `||d||` 几乎不变，也可能旋转它。这个解释与
此前“LPL 没有把 full-target 误差移向局部低敏感方向”的反证不冲突：
此前研究的是较大的 `full-clean` 有限误差；这里研究的是两个相近预测之差，
局部二阶近似是否成立需要单独检验。

该机制当前不是结论。最小证伪实验应在相同固定状态上比较：

```text
实际差分更新：Delta_d = d_LPL - d_Flow
局部预测更新：-H_phi(m_Flow, y) d_Flow
```

并同时报告二者余弦、`d` 的方向旋转、范数变化和对应 head-swap 结果。若
`Delta_d` 与 Hessian 预测不一致，或方向指标不能预测未见 seed 的生成排序，
就应放弃“decoder 曲率重塑 guidance contrast”的解释。

还必须保留一个现有反例：`guided_common` 在 prediction tensor 上让 full/base
接收相同的辅助梯度，意图优先产生 common-mode update，但其 5k FID 仍从
Flow 的 `10.8278` 退化到 `11.0360`。由于共享参数和两个输出路径不同，这不
保证参数更新后 `d` 完全不变；不过它已经说明 differential fragility 不能
在 head-swap 之前被当作唯一解释。paired decoder update 对 full/common 场
本身有害，仍是同等重要的竞争假设。

### 关键因果实验：交换 full 与 guidance contrast

对 Flow checkpoint 记：

```text
F0(x,t) = full_Flow(x,t)
D0(x,t) = full_Flow(x,t) - base_Flow(x,t)
```

对 LPL checkpoint 记：

```text
F1(x,t) = full_LPL(x,t)
D1(x,t) = full_LPL(x,t) - base_LPL(x,t)
```

在完全相同的初始噪声和 Euler 轨迹设置下，构造四个采样器：

| 强预测 | guidance contrast | guided clean prediction |
|---|---|---|
| Flow | Flow | `F0 + 0.78 D0` |
| LPL | Flow | `F1 + 0.78 D0` |
| Flow | LPL | `F0 + 0.78 D1` |
| LPL | LPL | `F1 + 0.78 D1` |

注意每一步都要在当前状态重新查询对应 checkpoint；不能预先缓存一条模型轨迹
后离线相加。这个 `2 x 2` 设计只改变两个有明确语义的因子，不训练新模型：

- 若 `F0 + 0.78 D1` 退化，而 `F1 + 0.78 D0` 不退化，问题主要在 contrast；
- 若相反，问题主要在 full field；
- 若两项单独都正常、合起来才退化，说明存在轨迹上的非线性交互；
- 若两项都退化，LPL 同时损坏强场和引导差值，不能只修 base head。

这是当前信息增益最高、成本最低的实验。

在查看交换结果前预注册方向预测：full-only LPL10 相对 Flow10 的 1k FID
差值，在 `IG=1.0` 时约为 `+0.1668`，在 `IG=1.78` 时约为 `+0.9204`。
因此若差值场脆弱性是主要机制，应有：

```text
Flow full + LPL contrast
    明显差于
LPL full + Flow contrast
```

若顺序相反，主要问题在 full field；若两个混合角落都接近 Flow、只有纯 LPL
角落差，则优先解释为沿 ODE 轨迹的非线性交互。该预测在混合角落 FID 产生前
写入，后续不根据结果改写。

seed 0 的首轮 1k 交换结果已经否定上述“contrast 主导”预测：

| full source | contrast source | FID | 相对 Flow/Flow |
|---|---|---:|---:|
| Flow | Flow | 40.7326 | 0 |
| LPL | Flow | 41.1419 | +0.4092 |
| Flow | LPL | 40.8909 | +0.1583 |
| LPL | LPL | 41.6531 | +0.9204 |

两个纯角落均与历史样本逐像素相同，四个角落的每 rank noise/label/RNG
fingerprint 完全一致。KID 与 IS 也给出同方向排序。按 FID 的二因素描述性
分解，单独替换 full、单独替换 contrast 和剩余交互项分别约为 `+0.4092`、
`+0.1583`、`+0.3529`；FID 不是线性可加损失，因此这些数值只能用来定位
因子，不能当成严格因果贡献百分比。

结果说明：

- contrast 确实被 LPL 改坏，但它不是首轮中最大的单因素；
- full field 的 paired decoder post-training 本身已经伤害 guided rollout；
- 两者同时来自 LPL checkpoint 时还有很强的轨迹交互；
- differential fragility 只能保留为次级机制，不能再作为当前主解释。

更符合证据的主解释回到 paired-to-distribution inversion：成熟 RAEv2 的
full prediction 已接近 paired conditional estimator，而官方 guidance
故意使用有偏外推改善生成分布；LPL 将 full 拉向 decoder-space paired target，
并不保证改善该有偏 vector field 的分布作用。由于 1k FID 方差较大，这个
反预期结果需要至少一个额外 sample seed 复核后再决定是否做 5k 混合角落。

已有的 4-image endpoint/rollout 审计还显示了第二层放大。该审计使用的是
full+base strict LPL10，不是上面 head-swap 的 full-only checkpoint，因此
只能作为独立机制线索：

- 从 on-data-path 状态只查询一次时，LPL full 的 raw feature error 相对
  Flow 约为 `-0.46%`；
- 递归查询 16 次后，full 的 raw feature error 反而为 `+23.62%`；
- guided 的 raw feature error 从一次查询的 `+4.06%` 增至 16 次的
  `+16.43%`；
- 但同一过程中 LPL 自己的 strict normalized feature error 始终更低，
  16 次时 full/guided 分别约为 `-4.21%/-2.68%`。

这表明 strict LPL 不仅有 paired-vs-distribution 目标错位，还会在自生成状态
上累积“自己的代理继续改善、未归一化 feature 差距继续恶化”的 rollout
错位。后续若扩展轨迹审计，应优先补 raw/detach/full-only，而不是继续只看
单步 validation loss。

### Distributional Decoder Alignment 只保留为诊断

如果最终目标是生成分布，直觉上可以比较：

```text
Flow MSE 继续保留真实 conditional coupling

辅助项只比较：
distribution(decoder_features(z_hat_batch))
与
distribution(decoder_features(z_batch))
```

第一版不需要复杂网络。先比较：

- channel mean；
- channel variance / covariance spectrum；
- sliced one-dimensional projections。

它与 strict LPL 的本质区别是：

- strict LPL 是 paired reconstruction；
- distributional alignment 允许生成输出重新分配到高密度样本；
- 当 guided prediction 的 feature distribution 已经匹配真实分布时，目标会
  自然变小，而不会像 prediction denominator 一样靠继续放大方差获益。

但到 2026 年 4 月，Representation Fréchet Loss 已经直接把 representation
space 的 Fréchet distance 用作生成器 post-training 目标，并明确强调无需
per-sample target。因此，“把 paired decoder loss 换成 batch distribution
loss”本身不再足以成为核心创新。它仍可作为诊断和正对照，用来确认我们看到
的矛盾是否确实来自统计单位，但不应先包装成新方法。

来源：

- [Representation Fréchet Loss for Visual Generation](https://arxiv.org/abs/2604.28190)

### 后备方法：只学习 guidance reference，不改强 prior

若 Useful Error Geometry 成立，可以冻结 full prior 和共享主干，只调整
`base_final_layer`，让弱参考头产生更有用的质量差值。这样：

- `IG=1.0` 的 full sampler 保持逐 tensor 不变；
- 所有效果只来自 guidance reference；
- 不再用 decoder loss扰动已经成熟的 full vector field。

这比继续微调 full prior 有更清楚的因果解释。但它只有在前述方向审计通过后
才值得做，否则只是把同一个无效 proxy 换了参数位置。

### 新颖性边界

以下方向已有很近的工作，不能作为主贡献：

- 泛泛的 weak-to-strong 外推：AutoGuidance、Internal Guidance 和 segmented
  weak-to-strong guidance 已经覆盖；
- 把 guidance 分成平行/正交分量并抑制过强分量：Adaptive Projected Guidance
  已经直接研究；
- 用 representation distribution distance 训练生成器：Representation
  Fréchet Loss 已经直接实现；
- 简单扫描 IG scale、增加一个动态权重或把 LPL 与 IG 相加，都只是工程组合。

当前可能仍有独立价值的窄问题是：

> decoder-space paired objective 与 self-guidance 的 weak-head bias 为什么
> 会发生符号冲突，以及怎样在不破坏 strong field 的前提下识别或学习有用的
> contrast field。

来源：

- [Guiding a Diffusion Model with a Bad Version of Itself](https://arxiv.org/abs/2406.02507)
- [Guiding a Diffusion Transformer with the Internal Dynamics of Itself](https://arxiv.org/abs/2512.24176)
- [Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models](https://arxiv.org/abs/2410.02416)
- [Improving Diffusion Generalization with Weak-to-Strong Segmented Guidance](https://arxiv.org/abs/2603.20584)

### 头脑风暴后的研究优先级

**A 级：Self-guidance differential fragility**

单一主张是：改善 full/base 的 endpoint objective 可能沿主 Flow loss 的弱约束
方向改变二者差值，而 self-guidance 会放大该 differential change。它有：

- 旧 RAE 单头正结果作为自然对照；
- RAEv2 双头反结果；
- `IG=1.0/1.78` 因果消融；
- `2 x 2` head-swap 可直接证伪；
- 不需要先发明新 loss。

若该现象能在 RAEv2 之外至少一个公开 IG/AutoGuidance 模型上复现，才可能形成
一篇以“better predictor, worse guide”为核心的机制论文。

**B 级：Paired-to-distribution inversion**

条件均值恒等式和官方 IG 的 `paired error up / gFID down` 是很强的解释与
评价警告，但 guidance 本来就会改变 fidelity/diversity 和样本密度，因此它
单独不足以构成新贡献。它应作为 A 级主线的理论地基。

**C 级：LPL denominator 是 anti-contraction update**

旧 RAE 和 RAEv2 的 detach 因果消融都支持这一点，但它更像 LPL 的目标函数
剖析。除非能推出跨模型可预测的适用边界，否则只适合作为机制子结论。

**停止条件**

- head-swap 无法把退化稳定归因给 full、contrast 或其交互；
- fixed-state 的 differential metrics 不能预测未见 sample seed 的排序；
- 外部 IG/AutoGuidance 模型不出现同类现象；
- 所谓方法最终只是 scale tuning、projection 或已有 distribution loss。

满足任一核心停止条件时，不再通过增加 loss、adapter 或额外 head 挽救主线。

## 9. 最小、低成本的下一步

### Phase 0：不训练，先做 `2 x 2` 交换

复用已有 checkpoints：

- RAEv2 Flow 与 strict，先做 1k screen；
- 若 contrast/full 归因清楚，再补 detach、raw；
- 每组使用相同 class labels、初始 noise、ODE steps 和 online/EMA 权重；
- 通过后只对必要格子做 5k，避免重复大规模采样。

同时在同一批状态上记录：

```text
Delta_full = full_branch - full_Flow
Delta_base = base_branch - base_Flow
Delta_d    = Delta_full - Delta_base
cos(D, clean - full)
cos(D, -grad decoder_feature_loss)
||D||
decoder feature mean / variance / covariance movement
full 与 contrast 在相邻时间步的方向一致性
```

验收条件：

1. 同一组 paired samples 下，交换实验的输出必须逐 tensor 可复现；
2. 至少一个单因子交换能解释 strict 相对 Flow 的主要 FID 退化；
3. 方向/分布指标在未查看 FID 的 calibration split 上预测该交换结果；
4. 结论在至少三个 sample seeds 上方向一致；
5. 若无法把退化归因给 full 或 contrast，停止发明新 loss，先承认当前机制
   分辨率不足。

### Phase 1：只有机制通过后才做方法

- 若 contrast 是主因：冻结 full/shared trunk，只研究 base reference；
- 若 full field 是主因：LPL 不适合 RAEv2，停止该方法线；
- 若是交互项：研究 trajectory-level objective，而不是 endpoint loss；
- distributional decoder metric 只作为已有思想的诊断对照，不单独主张创新。

若 full field 是主因，但仍希望保留一个不改变 Flow population optimum 的
decoder-aware 候选，可只研究 **decoder pullback preconditioning**。令：

```text
e = predicted_clean - paired_clean
J = decoder feature Jacobian at predicted_clean
g_decoder = J^T W J e
```

其中 `W` 使用离线真实 latent 统计得到的固定正权重，并对 `J/W` stop-gradient。
它满足：

\[
e^\top g_{decoder}=\|W^{1/2}Je\|^2\ge 0.
\]

所以该更新对 latent MSE 至少在一阶上不冲突；若局部度量只依赖 `x_t`/当前
prediction 而不依赖 paired target，它也不把条件均值换成另一个非线性
feature Fréchet mean。它和已经失败的 flow-parallel 消融不同：后者只保留
decoder gradient 在 `e` 上的一维投影，几乎退化成 Flow；这里保留完整的
decoder 敏感度矩阵。

这仍只是后备假设。它需要 JVP+VJP，计算较贵，而且理论上更可能改善优化速度
而不是改变最终最优分布。只有 head-swap 指向 full field、并且低成本
Gauss-Newton 梯度审计显示它提供非平凡但稳定的 PSD 预条件时，才值得做
10-step pilot；否则不启动。

它也不能作为当前论文主创新。到 2026 年，Riemannian Flow Matching、RJF
以及 decoder pullback metric 已经分别研究了 latent manifold、表示编码器的
曲率/Jacobian 误差传播和 decoder 局部失真。我们的 PSD 预条件形式与这些工作
不完全相同，但研究区域已经很拥挤；除非后续出现 self-guidance 特有、现有
几何方法无法解释的结果，否则只把它作为诊断正对照。

相关边界：

- [Learning on the Manifold: Unlocking Standard Diffusion Transformers with Representation Encoders](https://arxiv.org/abs/2602.10099)
- [Latent Diffusion Inversion Requires Understanding the Latent Space](https://arxiv.org/abs/2511.20592)
- [Latent Riemannian Flow Matching for Geometry-Grounded 3D Foundation Models](https://arxiv.org/abs/2607.19120)

## 10. 当前 50-step 训练状态

detach 续训已经从 branch update 10 正确完成到 50：

- global step 连续为 `100091...100130`；
- scheduler 正确恢复到官方末端 `lr=2e-5`；
- Flow/LPL/total loss 和 grad norm 均有限；
- 没有 OOM 或 loss explosion。

update 20、30、40 和 50 均已原子保存；detach 最终 checkpoint 是：

```text
branch-0000050-global-0100130.pt
```

此前沙箱内的 `nvidia-smi` 和 `ps` 无法看到宿主机 GPU/进程，一度被误判为
驱动失联；宿主机检查确认四张 GPU 和原 tmux 训练始终正常。detach 50 保存后，
由于运行中的 wrapper 同时被编辑，旧 shell 在切换 raw 时遇到一次 EOF。该
错误没有影响 detach checkpoint。重新通过 `bash -n` 后，wrapper 已在新 tmux
会话中跳过完成的 detach，并从 raw update 10 正确恢复。

raw update 11 的首步为：

```text
flow=0.890446
raw_lpl=15368.879883
total=0.891697
grad=0.3852
```

raw loss 的绝对值因删除方差归一化而远大于 detach，但其权重只有
`8.1396425e-8`，加权后的总 loss 和梯度量级均正常。wrapper 会每 10 update
保存，继续到 raw update 50。截至本次记录，raw 已到 update 30：

```text
flow=0.888826
raw_lpl=16175.766602
total=0.890143
grad=0.4052
```

后续 raw 已完整到 update 50：

```text
flow=0.886483
raw_lpl=13901.469727
total=0.887614
grad=0.4100
```

`branch-0000050-global-0100130.pt` 已正确保存，未出现数值发散。

独立 tmux 守护任务已经通过 raw50 等待条件和训练配对审计，并启动 5k 采样。
它依次执行：

1. 审计 Flow50、detach50、raw50 的 source/config/global step 和四个 rank
   的最终 RNG state；
2. 确认 detach/raw 使用相同训练样本序列；
3. 用同一组 label、初始 noise、online model、100-step、`IG=1.78` 分别生成
   5k 样本；
4. 逐 tensor 验证同噪声协议，再计算 FID/KID/IS。

此前已验证 Flow30 与 detach30 的训练样本摘要完全相同，raw20 与 detach20
的训练样本摘要也完全相同；Flow50 与 detach50 的四 rank CPU/CUDA/NumPy/
Python RNG state 全部一致。这使当前 50-step 对照不仅“配置相同”，而且训练
随机流也有可审计证据。

raw50 最终审计进一步确认：

- Flow50、detach50、raw50 的 source SHA、config SHA、source/global step
  全部相同；
- 四个 rank 的 CPU/CUDA/NumPy/Python 最终 RNG state 全部一致；
- detach50 与 raw50 的累计 data-index SHA 完全相同；
- 当前同噪声采样使用 online `model`、seed 0、100-step Euler、
  `IG=1.78`、每卡 batch 16；
- 四张 GPU 采样时均为约 98--100% 利用率，单卡显存约 6.8 GiB，没有 OOM
  或进程停滞迹象。

另外已经实现了不训练参数的 `2 x 2` head-swap 采样器，并通过 5 个 toy
测试。它会等本轮 5k 完整评估后再运行 1k screen，避免抢占 GPU。两个纯角落
必须在历史相同 batch/seed 下逐像素复现旧 Flow10/full-only-LPL10 样本；
任一失败就拒绝解释混合角落。

50-step 的同噪声 5k 评估现已完成：

| branch | FID | KID mean | IS |
|---|---:|---:|---:|
| Flow50 | 10.8451 | 0.0001194 | 151.15 |
| detach50 | 10.9339 | 0.0001229 | 149.11 |
| raw50 | 10.8767 | 0.0001176 | 150.49 |

相对 Flow，detach 为 `+0.0888 FID`，raw 为 `+0.0316 FID`。raw 的 KID
略优于 Flow，而 IS 略低，因此不能把这一级别的差异解释为可靠退化，更不能
解释为改善；最稳妥的结论是 raw 在 50 step 后仍回到 Flow 附近。detach 的
FID/IS/KID 三项方向均略差，但单个 5k seed 仍不足以支持很强的效应量结论。

当前 Flow50 的 `10.8451` 与历史曲线中的 `10.6436` 不是同协议数值漂移。
审计确认二者使用同一 checkpoint SHA、seed、per-rank batch 16、100-step、
IG=1.78、标签和首尾 RNG fingerprint；唯一关键差异是历史曲线采样 `ema`，
当前三分支统一采样在线 `model`。本轮使用 online state 是为了让 50 次分支
更新的即时差异不被 `EMA=0.9995` 显著稀释，因此三分支内部比较公平，但不应
把当前绝对 FID 与历史 EMA 曲线直接混合。

这轮结果排除了一个原本合理的希望：去掉 prediction-variance 分母并延长训练
不会自动显露出旧 RAE 那样的隐藏收益。分母梯度解释 strict LPL 的大部分额外
伤害，但 paired decoder-feature alignment 本身没有给成熟 RAEv2 提供稳定的
新增生成收益。

首次自动启动 head-swap 时，`torchrun` 以文件路径执行导致仓库根目录未进入
`sys.path`，在模型加载前触发 `ModuleNotFoundError`。该失败没有生成样本或
占用训练状态。采样器已增加显式仓库根目录 bootstrap，守护命令改为模块启动；
重新通过 5 个单测、`py_compile` 和 shell 语法检查后，任务已在 tmux 中启动。
双 Stage-2 模型加 decoder 每卡约占 10.5 GiB，仍有约 13.5 GiB 余量。
