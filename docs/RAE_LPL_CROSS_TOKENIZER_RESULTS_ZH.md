# RAE LPL 跨 Tokenizer 生成验证

## 研究问题

已有实验表明，在官方 `RAE-DINOv2-B + DiTDH-S` 上，给 latent prior
加入冻结 decoder 中间特征上的 LPL，可以改善严格配对的生成指标。本实验
进一步检验：

> 这个收益是否只属于 DINOv2，还是能跨 RAE 的公开 tokenizer 家族复现？

实验覆盖 RAE 官方公开的三个代表性 encoder 家族：

- DINOv2-B：自监督蒸馏表征。
- MAE-B：masked autoencoder 表征。
- SigLIP2-B：图文监督表征。

MAE-B 和 SigLIP2-B 使用官方公开的 ViT-XL decoder、latent normalization
statistics 和 DiT-XL epoch-80 prior。两个模型的 latent shape 都是
`[768, 16, 16]`，因此它们还构成了相同 prior 架构下的跨 encoder 对照。

## 严格配对协议

- 数据：ImageNet-1K train；validation reference 只用于训练后的生成评价。
- encoder 和 decoder：冻结、`eval()`、确定性前向。
- prior：只更新官方 DiT-XL。
- 数值：四张 RTX 4090，全程 fp32，关闭 TF32。
- 分支：
  - `Flow`：原始 latent velocity MSE。
  - `LPL`：Flow 加冻结 decoder 五层 feature loss。
- 训练：每个 tokenizer 使用 seeds `3407/3408/3409`，各继续训练 500 次
  optimizer update。
- 配对：同 seed 的两个分支具有完全相同的模型参数、新建
  optimizer/scheduler 状态、图像、标签、时间步和噪声。首批 tensor 的
  SHA256 指纹逐项一致。
- LPL 权重：在查看生成结果前，用官方源模型和固定的 256 个 train 样本
  校准，使初始加权 LPL/Flow 为 `0.25`：
  - MAE-B：`0.001310218123057881`
  - SigLIP2-B：`0.0009191518565962038`
- 采样：EMA、Euler 50 steps、CFG 1.0、固定 sampling seed `20260715`、
  类别均衡。

官方 MAE-B/SigLIP2-B prior 只发布 model-only 权重，没有 optimizer
checkpoint。因此本实验不能声称“精确恢复官方 epoch-80 optimizer 状态”；
但同一个 tokenizer 内 Flow/LPL 从相同参数和相同的新 optimizer 状态出发，
因果比较仍然严格。

## 预注册判据

在运行结果前固定：

- 强复现：3/3 seeds 的 5k FID 均改善，且平均 KID 不恶化。
- 弱复现：至少 2/3 seeds 的 FID 改善，平均 FID 改善且平均 KID 不恶化。
- 失败：平均 FID 不改善，或仅 1/3 seed 改善。

只有 MAE-B 和 SigLIP2-B 都至少弱复现，才允许得出“收益不是 DINOv2-B
单一 tokenizer 的偶然现象”。若两者都通过，则使用预先固定的 seed 3409
进行各自 50k ADM 终验。

## MAE-B 的 5k 结果

下面的 5k 指标由 `torch-fidelity` 计算。它们适合严格配对分支内比较，
不能直接当作 RAE 原文的 50k gFID。

| train seed | Flow FID ↓ | LPL FID ↓ | FID 改善 | Flow KID ↓ | LPL KID ↓ | Flow IS ↑ | LPL IS ↑ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3407 | 17.6496 | 16.9699 | 0.6796 | 0.003793 | 0.003492 | 77.5089 | 78.7342 |
| 3408 | 17.7907 | 17.0663 | 0.7243 | 0.003998 | 0.003620 | 78.1699 | 79.7067 |
| 3409 | 17.8310 | 17.0908 | 0.7402 | 0.004009 | 0.003624 | 77.4432 | 78.6336 |
| mean | 17.7571 | 17.0424 | 0.7147 | 0.003933 | 0.003579 | 77.7073 | 79.0248 |

MAE-B 达到强复现：

- FID 3/3 改善，平均绝对改善 `0.7147`，相对改善 `4.02%`。
- KID 3/3 下降，平均下降约 `9.01%`。
- IS 3/3 上升，平均上升约 `1.70%`。

## SigLIP2-B 的 5k 结果

| train seed | Flow FID ↓ | LPL FID ↓ | FID 改善 | Flow KID ↓ | LPL KID ↓ | Flow IS ↑ | LPL IS ↑ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3407 | 13.7162 | 13.3887 | 0.3274 | 0.001419 | 0.001328 | 112.0353 | 113.7402 |
| 3408 | 13.7370 | 13.4065 | 0.3305 | 0.001423 | 0.001337 | 111.8065 | 114.4114 |
| 3409 | 13.6789 | 13.3509 | 0.3280 | 0.001412 | 0.001311 | 112.3505 | 114.0744 |
| mean | 13.7107 | 13.3821 | 0.3287 | 0.001418 | 0.001325 | 112.0641 | 114.0753 |

SigLIP2-B 也达到强复现：

- FID 3/3 改善，平均绝对改善 `0.3287`，相对改善 `2.40%`。
- KID 3/3 下降，平均下降约 `6.53%`。
- IS 3/3 上升，平均上升约 `1.79%`。

## 官方起点的 5k 对照

为判断短程继续训练是否只是在两个分支之间比较，而没有真正超过官方起点，
直接对 DINOv2-B 官方 epoch-14 `DiTDH-S`，以及 MAE-B/SigLIP2-B 官方
epoch-80 `DiT-XL` 做了严格同协议 5k 评价。该评价没有创建 optimizer，
没有执行训练，也没有做 EMA update；官方权重直接进入以下采样协议：

- EMA/model source：未经继续训练的官方 Stage-2 checkpoint；
- Euler 50 steps，CFG 1.0；
- fixed seed `20260715`；
- 4 processes，每 process batch 4；
- fp32，关闭 TF32；
- 与 Flow/detach/full 相同的 reference 和 `torch-fidelity` 配置。

审计发现，旧 DINOv2 官方目录已有 5,008 张 PNG，严格重采样后的 5,008
张 PNG 和 5k NPZ 均与旧结果逐字节一致。旧 MAE/SigLIP2 官方目录各有
5,056 张 PNG，说明它们使用过不同 global batch；严格重采样改为 5,008
张，因此下面不再使用旧 MAE/SigLIP2 官方数字。

下表中的 Flow/detach/full 是各自三个严格配对训练 seed 的均值；官方行是
唯一的零更新固定模型：

| tokenizer | objective | updates | FID ↓ | KID ↓ | IS ↑ | 相对官方 FID | FID 超过官方的 seeds |
|---|---|---:|---:|---:|---:|---:|---:|
| DINOv2-B | 官方原模型 | 0 | 21.6122 | 0.006439 | 71.8754 | 0 | baseline |
| DINOv2-B | Flow | 2000 | 22.1486 | 0.007280 | 69.0761 | +0.5363 | 0/3 |
| DINOv2-B | detach | 2000 | 21.1808 | 0.006659 | 69.9130 | -0.4315 | 3/3 |
| DINOv2-B | full LPL | 2000 | **19.6790** | **0.006006** | 71.7810 | **-1.9332** | **3/3** |
| MAE-B | 官方原模型 | 0 | 17.5508 | 0.003593 | **80.2416** | 0 | baseline |
| MAE-B | Flow | 500 | 17.7571 | 0.003933 | 77.7073 | +0.2063 | 0/3 |
| MAE-B | detach | 500 | 17.6397 | 0.003890 | 78.1980 | +0.0889 | 1/3 |
| MAE-B | full LPL | 500 | **17.0424** | **0.003579** | 79.0248 | **-0.5085** | **3/3** |
| SigLIP2-B | 官方原模型 | 0 | 13.5717 | 0.001379 | **117.8009** | 0 | baseline |
| SigLIP2-B | Flow | 500 | 13.7107 | 0.001418 | 112.0641 | +0.1390 | 0/3 |
| SigLIP2-B | detach | 500 | 13.6402 | 0.001405 | 112.9537 | +0.0685 | 0/3 |
| SigLIP2-B | full LPL | 500 | **13.3821** | **0.001325** | 114.0753 | **-0.1896** | **3/3** |

这里的正负号以 `branch FID - official FID` 定义，因此负数表示超过官方
原模型。结果把两个问题分开了：

1. 三种 tokenizer 的普通 Flow 继续训练均值都比官方起点差，且 `0/9`
   个 Flow seed 在 FID 上超过官方起点。
2. detach 在三种 tokenizer 上都优于同程 Flow，但只有 DINOv2 的 `3/3`
   和 MAE 的 `1/3` 超过官方；MAE、SigLIP2 的 detach 均值仍差于官方。
3. full LPL 的九个 seed 全部在 FID 上超过官方起点，三个 tokenizer 的
   平均 FID 分别改善 `1.9332`、`0.5085`、`0.1896`。
4. full 的平均 KID 也全部超过官方，但 MAE/SigLIP2 的 IS 仍低于官方。

所以，“full LPL 只是在两个同样退化的继续训练分支中相对较好”已经被
严格 FID/KID 基线否定。与此同时，不能声称它全面支配官方模型，因为辅助
IS 并没有在所有 tokenizer 上改善，且这里仍是固定 5k proxy，不是三个
tokenizer 全部完成的官方 50k 绝对对照。

严格官方结果：

`~/data/eqvae/experiments/rae_strict_lpl_official_source_5k/official_source_strict_5k.json`

统一四分支汇总：

`~/data/eqvae/experiments/rae_lpl_detach_cross_tokenizer/official_flow_detach_full_strict_5k_summary.json`

## 计算预算对照

主实验本身已经是严格的等 update、等数据对照：同一 tokenizer 的 Flow 和
LPL 分支都从相同官方权重出发，各训练 500 次 optimizer update，global
batch size 为 16，即每个分支只处理 8,000 个训练样本。这约等于
ImageNet-1K train 的 `0.0062 epoch`，并不是 LPL 额外训练了若干 epoch。

LPL 每次 update 需要额外通过冻结 decoder 计算中间特征和反向梯度，因此
它的单步计算量更大。根据 step 1 到 step 500 的训练日志，LPL 相对 Flow
的墙钟开销在 MAE-B 上约为 `8.7%`，在 SigLIP2-B 上约为 `14.7%`。为排除
“收益只是每步多算了一点”的解释，事后固定 seed 3409，为纯 Flow 分支额外
训练到 600 updates，再用原 5k 采样协议评价。该控制让 Flow 获得 `20%`
更多 updates 和训练样本，略高于两种 tokenizer 的实测 LPL 时间开销。

两个新分支在 step 500 的完整 checkpoint 均与原 Flow-500 精确一致：
模型、EMA、optimizer、scheduler、CPU/CUDA RNG 共 1,759 个张量和 373 个
标量均为零差异。因此 step 501-600 是唯一新增干预。

| tokenizer | branch | updates | FID ↓ | KID ↓ | IS ↑ |
|---|---|---:|---:|---:|---:|
| MAE-B | Flow | 500 | 17.8310 | 0.004009 | 77.4432 |
| MAE-B | Flow，额外预算 | 600 | 17.7512 | 0.004115 | 77.1313 |
| MAE-B | LPL | 500 | **17.0908** | **0.003624** | **78.6336** |
| SigLIP2-B | Flow | 500 | 13.6789 | 0.001412 | 112.3505 |
| SigLIP2-B | Flow，额外预算 | 600 | 13.8835 | 0.001442 | 110.6498 |
| SigLIP2-B | LPL | 500 | **13.3509** | **0.001311** | **114.0744** |

MAE-B 的额外 100 次 Flow update 只让 FID 改善 `0.0798`，同时 KID 和 IS
略微恶化；其 FID 仍比 LPL-500 高 `0.6604`。SigLIP2-B 的 Flow-600 三项
指标都差于 Flow-500，其 FID 也仍比 LPL-500 高 `0.5326`。因此，在这两个
固定 seed 的抽查中，给普通 Flow 超过 LPL 实测额外开销的训练预算，不能
解释 LPL 的生成收益。

这是事后的一 seed、5k 计算预算诊断，不应替代前面的三 seed 配对实验或
后面的 50k 终验。它支持的保守结论是：当前 LPL 增益更像来自 decoder
feature loss 改变了优化方向，而不是简单增加 update 数、数据量或计算时间。

## 跨模型汇总

| RAE tokenizer | prior | seeds | FID 同向 | KID 同向 | IS 同向 | 判定 |
|---|---|---:|---:|---:|---:|---|
| DINOv2-B | DiTDH-S | 4 | 4/4 | 4/4 | 4/4 | 已有强复现和 50k ADM 终验 |
| MAE-B | DiT-XL | 3 | 3/3 | 3/3 | 3/3 | 强复现和 50k ADM 终验 |
| SigLIP2-B | DiT-XL | 3 | 3/3 | 3/3 | 3/3 | 强复现和 50k ADM 终验 |

当前一共得到 10/10 个训练 seed 的 FID、KID、IS 同方向改善。MAE 和
SigLIP2 使用相同 DiT-XL 架构但不同 encoder/decoder，DINOv2 使用另一种
prior 架构。因此，结果已经排除了以下几个简单解释：

1. 只在一个训练 seed 上偶然有效。
2. 只对 DINOv2 latent 有效。
3. 只对 DiTDH-S prior 架构有效。
4. 只改善一个 FID 数字而 KID/IS 反向恶化。

## 训练行为

LPL 并没有同时降低普通 Flow loss。500 次 update 的平均值如下：

| tokenizer | Flow 分支的 Flow loss | LPL 分支的 Flow loss | Flow grad norm | LPL grad norm | LPL clip rate |
|---|---:|---:|---:|---:|---:|
| MAE-B | 0.5839 | 0.5914 | 0.2251 | 0.3040 | 0.40% |
| SigLIP2-B | 0.5591 | 0.5606 | 0.2340 | 0.4983 | 2.13% |

LPL 分支的普通 latent MSE 略高，但生成指标稳定更好。这支持一个保守表述：

> 普通 latent MSE 与冻结 decoder 实际关心的误差并不完全一致；decoder
> feature loss 能给 latent prior 提供额外、可跨 tokenizer 复现的训练信号。

这仍不是具体 Jacobian 机制的充分证明。此前“LPL 把误差转移到 decoder
局部低敏感方向”的强机制说法已被实验反证，不应因生成结果改善而重新启用。

## 50k 终验

预先固定 seed 3409，分别对 MAE-B 和 SigLIP2-B 的 Flow/LPL 分支生成
50,000 张类别均衡样本，并使用官方 ADM/guided-diffusion TensorFlow
Inception 协议评价。

四份 NPZ 均已核验为精确的 `(50000, 256, 256, 3)`、`uint8` 数组。
训练和采样配置均未在 5k 晋级后调整。

首先，使用与 5k 相同的 `torch-fidelity` 实现得到：

| tokenizer | branch | FID ↓ | KID ↓ | IS ↑ |
|---|---|---:|---:|---:|
| MAE-B | Flow | 11.9943 | 0.004107 | 109.3436 |
| MAE-B | LPL | **11.3350** | **0.003792** | **111.8263** |
| SigLIP2-B | Flow | 8.2955 | 0.001454 | 174.5569 |
| SigLIP2-B | LPL | **8.0143** | **0.001375** | **177.2613** |

MAE-B 的 FID 相对改善 `5.50%`，SigLIP2-B 的 FID 相对改善 `3.39%`。
两者的 KID 和 IS 也都同向。

随后，使用 ADM/guided-diffusion 的 TensorFlow Inception graph 和官方
ImageNet 256 reference statistics，在完全相同的四份 NPZ 上得到：

| tokenizer | branch | ADM FID ↓ | sFID ↓ | IS ↑ |
|---|---|---:|---:|---:|
| MAE-B | Flow | 8.6148 | 5.6520 | 109.2572 |
| MAE-B | LPL | **7.9498** | **5.2172** | **111.7411** |
| SigLIP2-B | Flow | 4.9550 | 7.2129 | 174.8867 |
| SigLIP2-B | LPL | **4.6811** | **6.1272** | **177.5565** |

MAE-B 的 LPL 分支：

- ADM FID 绝对改善 `0.6650`，相对改善 `7.72%`。
- sFID 相对改善 `7.69%`。
- IS 相对提升 `2.27%`。

SigLIP2-B 的 LPL 分支：

- ADM FID 绝对改善 `0.2739`，相对改善 `5.53%`。
- sFID 相对改善 `15.05%`。
- IS 相对提升 `1.53%`。

两个新增 tokenizer 的 5k 多 seed、50k `torch-fidelity` 和 50k ADM 三层
证据方向全部一致。终验 seed 是事前固定的 3409，不是根据 5k 排名选择。

## prediction denominator detach 因果消融

为区分原始 prediction LPL 中的两条梯度路径，又增加了
`prediction-detach = N / stopgrad(V)`。它保留 inverse-variance weighting，
仅删除 prediction variance denominator 对模型参数的直接梯度。full 与
detach 使用相同 source、seed、数据流、优化器、训练步数、LPL 权重和采样
协议；当前统一 full 实现还在 DINOv2-B 上逐 tensor bitwise 复现了历史
full checkpoint。

三种 tokenizer 的严格配对 5k FID 均值如下：

| tokenizer | Flow ↓ | detach ↓ | full ↓ | detach 保留的 full 收益 |
|---|---:|---:|---:|---:|
| DINOv2-B | 22.1486 | 21.1808 | 19.6790 | 39.2% |
| MAE-B | 17.7571 | 17.6397 | 17.0424 | 16.4% |
| SigLIP2-B | 13.7107 | 13.6402 | 13.3821 | 21.5% |

每个 tokenizer 都有 3/3 seeds 满足 `full < detach < Flow`，KID 和 IS
方向也一致。DINOv2-B 的预固定 seed-3 50k ADM FID 进一步得到：

| Flow ↓ | detach ↓ | full ↓ | detach 保留的 full 收益 |
|---:|---:|---:|---:|
| 13.5043 | 12.4540 | 11.1027 | 43.7% |

因此，固定 inverse-variance weighting 本身有稳定收益，但 prediction
variance denominator 的直接梯度贡献了 full LPL 收益的多数。这里的百分比
只是嵌套干预 `Flow -> detach -> full` 的 FID contrast ratio；Adam、
gradient clipping 和后续优化轨迹会让两条梯度相互作用，不能把它解释成
可独立相加的机制归因。

完整实现审计、参数更新几何和局限记录在
`docs/RAE_LPL_DETACH_AUDIT_ZH.md`。

补充采样口径：旧采样器在 5k 时会按 global batch 16 向上取整到 5,008
再截断，最终类别计数只有 `0.12%` 的近均衡偏差；所有配对分支使用完全相同
的标签和噪声。50k 可整除 16，类别计数严格均衡。该细节不改变上述分支差值，
但 5k 不应再描述成“每类严格 5 张”。

## 当前结论边界

可以声称：

1. LPL 的短程继续训练收益已跨 RAE 官方三个代表性 tokenizer 家族复现。
2. MAE-B 和 SigLIP2-B 均满足事前固定的强复现判据。
3. 10 个训练 seed、三个生成指标的方向全部一致，因而不是现有证据范围内的
   单 seed 或 DINOv2 特例。
4. MAE-B 和 SigLIP2-B 的预固定 50k ADM FID、sFID、IS 也全部改善。

还不能声称：

1. LPL 对所有 RAE encoder 尺寸、decoder 尺寸、分辨率和完整训练预算都有效。
2. 500 次 update 的改善会在训练 80/800 epoch 后保持相同幅度。
3. 不同 tokenizer 的绝对 5k FID 可以互相排名。
4. 当前实验已经唯一确定了 LPL 生效的微观机制。
