# AutoGuidance、Internal Guidance 与预测目标外推的二维机制实验

## 1. 研究范围

这组实验研究的是通用生成模型中的弱到强外推，不依赖 RAE 或 RAEv2。

核心问题是：

> 当强预测器生成的相邻模态仍然粘连时，沿“弱预测器到强预测器”的方向继续外推，能否把模态分开，同时保留完整分布和全部模态？

当前结论是：

- AutoGuidance（AG）和 Internal Guidance（IG）都能稳定提高局部锐度。
- 中等尺度的 IG 能大幅减少相邻簇之间的桥接，同时保留全部 32 个模态。
- 这种改善不是无代价的真实分布恢复，而是可控的“质量/锐度倾斜”。
- 限制 guidance 的启用区间非常重要；全程使用相同系数会明显扩大分布偏移。
- 预测目标外推（x-prediction 与 v-prediction 之间外推）不是严格的 AutoGuidance，在当前 toy 上也明显更差。

## 2. 原文核对

### 2.1 AutoGuidance

[AutoGuidance](https://arxiv.org/abs/2406.02507) 使用在相同任务、相同数据和相同预测目标上训练的强、弱两个模型：

\[
D_w = D_{weak} + w(D_{strong}-D_{weak}).
\]

弱模型通过减小容量或减少训练步数得到。论文的二维 toy 使用分形高斯混合分布，官方默认系数为 `w=3`。

### 2.2 Internal Guidance

[Internal Guidance（CVPR 2026）](https://arxiv.org/html/2512.24176v2) 在同一个网络的早期层增加辅助输出头：

\[
\mathcal L = \mathcal L_{final} + \lambda\mathcal L_{inter},
\]

采样时使用：

\[
D_w = D_{inter} + w(D_{final}-D_{inter}).
\]

原文的正常系数范围是：

- 二维 toy：`w=2`。
- ImageNet 全程 IG：扫描 `1.3, 1.5, 1.7, 1.9, 2.1, 2.3`，最优约为 `1.9`。
- 去掉低噪声区间后：原文最佳为 `w=2.3, [0.3,1)`。
- LightningDiT 与 CFG 联用的最终设置：IG 系数为 `1.4`。

因此，本报告只把 `w<=2.3` 视为正式范围。更大系数仅用于压力测试，不作为候选方法。

## 3. 已实现内容

### 3.1 官方分形 toy

入口：`experiments/run_guidance_fractal_toy.py`

- AutoGuidance 使用官方 EDM2 toy 代码、官方强弱 checkpoint 和官方 32 步 Heun 采样，属于精确官方复现。
- Internal Guidance 的公开仓库未包含二维 toy 源码，因此依据论文附录重建：4 个隐藏层、宽度 64、第一层后辅助头、4096 次迭代、batch 4096、`lambda=0.5`、`w=2`。
- 该 IG 实现属于“论文规格重建”，不能把其具体数值称为官方数值复现。

官方 AG 结果清楚复现了论文现象：弱模型较散，适度外推将样本推向高密度分支，过强外推则减少分支覆盖。

### 3.2 32 模态螺旋 toy

训练入口：`experiments/train_prediction_target_internal_guidance.py`

统一评估入口：`experiments/evaluate_prediction_target_autoguidance.py`

锐度/分布权衡分析：`experiments/analyze_guidance_sharpness_tradeoff.py`

多种子汇总：`experiments/summarize_internal_guidance_multiseed.py`

固定设置：

- 真实分布：64 维环境空间中的二维弯曲流形，沿螺旋放置 32 个高斯分量。
- 模型：5 层 residual MLP，hidden width 128。
- IG：第一 residual block 后增加中间头，`lambda=0.5`。
- 中间头与最终头预测同一个 clean target；为保持原 toy 的训练权重，两者都使用同一 v-space loss。
- 训练：30,000 step，batch 2048。
- 评估：每个条件 5,000 个样本，100 个采样步。
- 四个独立训练 seed：`20260901` 至 `20260904`。
- 每个条件使用相同初始随机状态；评价参考样本独立于训练样本。
- 训练样本来自解析高斯混合分布的独立随机流，验证集使用固定且独立的随机流，没有训练/验证泄露。

## 4. 指标含义

- `Latent SWD`：完整生成分布与真实分布的 sliced Wasserstein distance，越低越接近真实分布。
- `Bridge rate`：样本落在相邻模态之间低密度区域的比例，越低表示模态分得越开。
- `Peak/valley contrast`：相邻峰值密度与中间谷值密度的对数差，越高表示越锐。
- `Component JSD`：32 个模态的生成权重与真实权重之间的 Jensen-Shannon divergence，越低越好。
- `Occupied components`：至少生成过样本的模态数；低于 32 表示出现模态丢失。

SWD 衡量完整分布，bridge/contrast 衡量局部锐度。不能用其中一个替代另一个。

## 5. AG、IG 与预测目标外推的单种子对照

seed `20260904`：

| 条件 | Latent SWD ↓ | Bridge rate ↓ | Peak/valley contrast ↑ | 保留模态 |
|---|---:|---:|---:|---:|
| 原独立 x-prediction | 0.02790 | 15.68% | 2.07 | 32/32 |
| prediction-target，`w=2` | 0.03756 | 14.56% | 2.53 | 32/32 |
| AG early，`w=3, [0.3,0.7]` | 0.03713 | 4.36% | 5.05 | 32/32 |
| IG final，`w=1` | 0.02528 | 16.76% | 2.00 | 32/32 |
| IG，`w=2.3, [0.3,0.7]` | 0.02893 | 8.34% | 4.22 | 32/32 |

解释：

- prediction-target 外推只有很弱的锐化，却显著恶化完整分布距离。
- AG 锐化最强，但 SWD 和模态权重偏移也更大。
- IG 在该 seed 上提供了更好的锐度/分布折中。
- `IG final w=1` 与 `IG w>1` 是同一模型内的严格采样消融；它们之间的差异才是 IG guidance 的因果效果。
- 原独立 x-prediction 与 IG final 的差异还包含辅助监督改变训练过程的效果，不能归因于采样 guidance。

## 6. 四种子正式结果

以下设置在看其他 seed 结果前固定，系数均位于原文范围内：

| 条件 | Latent SWD ↓ | Bridge rate ↓ | Peak/valley contrast ↑ | Component JSD ↓ | 模态数 |
|---|---:|---:|---:|---:|---:|
| `IG w=1` | 0.02647 ± 0.00151 | 16.79% ± 0.69% | 2.10 ± 0.18 | 0.00239 ± 0.00012 | 32/32 |
| `w=1.9, full` | 0.03173 ± 0.00281 | 10.31% ± 0.66% | 3.80 ± 0.15 | 0.00316 ± 0.00034 | 32/32 |
| `w=1.9, [0.3,0.7]` | 0.02759 ± 0.00169 | 9.89% ± 0.56% | 3.53 ± 0.11 | 0.00269 ± 0.00029 | 32/32 |
| `w=2.3, full` | 0.03432 ± 0.00394 | 9.74% ± 0.96% | 4.58 ± 0.35 | 0.00361 ± 0.00023 | 32/32 |
| `w=2.3, [0.3,1]` | 0.03176 ± 0.00265 | 8.36% ± 0.55% | 4.32 ± 0.54 | 0.00328 ± 0.00015 | 32/32 |
| `w=2.3, [0.3,0.7]` | 0.02817 ± 0.00196 | 8.37% ± 0.47% | 4.10 ± 0.38 | 0.00285 ± 0.00027 | 32/32 |

相对同模型的 `w=1`：

- `w=1.9, [0.3,0.7]`：SWD 增加 4.23%，bridge rate 降低 41.08%，contrast 提高 67.82%。
- `w=2.3, [0.3,0.7]`：SWD 增加 6.44%，bridge rate 降低 50.16%，contrast 提高 95.29%。
- `w=2.3, full`：SWD 增加 29.69%。这说明全程大系数明显不划算。
- 四个 seed 的 bridge rate 全部下降、contrast 全部上升、32 个模态全部保留。
- `w=2.3, [0.3,0.7]` 在 3/4 个 seed 上把 SWD 增幅控制在 10% 内，但没有任何 seed 的 SWD 优于 `w=1`。

因此，当前最严谨的表述是：

> IG 在四个独立 seed 上稳定提高局部模态锐度，并通过 guidance interval 将完整分布偏移限制在较小范围；它没有把生成分布变得比无 guidance 更接近真实分布。

## 7. 为什么看起来像“锐化”

如果中间头和最终头分别近似两个平滑程度不同的分布，其 score 为 `s_inter` 和 `s_final`，则：

\[
s_w=s_{inter}+w(s_{final}-s_{inter})
\]

可形式化为：

\[
s_w=\nabla\log\frac{p_{final}^{w}}{p_{inter}^{w-1}}.
\]

因此它强调的不是“真实概率本身”，而是最终头相对中间头更相信的位置。若中间头把相邻模态抹平、最终头保留了一部分峰谷，外推就会放大这些峰谷，视觉上表现为锐化。

当前实验支持这一解释：

- 中间头的分布明显更模糊。
- 中等 IG 系数稳定减少模态之间的桥。
- 模态没有消失，但 SWD 和 component JSD 略增，说明生成分布被有意推向更尖锐的位置。

prediction-target 外推不具备同样干净的密度比解释，因为 x 和 v 模型使用不同预测目标，二者的误差未必兼容。当前结果也不支持把它直接视为 AutoGuidance。

## 8. 区域性与压力测试

沿 31 个相邻模态间隙逐个检查后发现，IG 的锐化由外向内并不均匀。seed `20260904` 中：

- 真实分布的 inner/middle/outer contrast 为 `0.91 / 4.24 / 10.37`。
- `IG w=2.75, [0.3,0.7]` 为 `0.22 / 3.68 / 11.68`。

外圈已经过度锐化，但最内圈仍然没有恢复到真实峰谷。这说明一个全局系数不能同时校正所有尺度。

额外的 `w=6~12` 压力测试仅用于判断差值方向中是否含有更多内圈信息。最内圈 contrast 最多提高到约 `0.59`，但 SWD、bridge rate 和 component JSD 均明显恶化。该结果说明继续增大系数会进入饱和和失真区域，不是可行方法。

## 9. 当前可靠结论

1. AG 与 IG 在二维多模态数据上确实具有可重复的锐化作用。
2. IG 的共享主干中间头能够提供与最终头兼容的弱预测方向。
3. `w=2.3, [0.3,0.7]` 是当前 toy 上较合理的正式锐化点；`w=1.9` 更保守。
4. guidance interval 不是次要调参，而是控制分布偏移的核心变量。
5. 当前证据不支持“IG 让生成分布更真实”；更准确的说法是“IG 以较小全局分布代价换取显著局部锐度”。
6. 单一全局外推无法完全解决最内圈的粘连，说明弱强差值只编码了部分局部模态信息。

## 10. 下一项机制实验

下一步不应继续扩大系数，而应直接测量 IG 差值方向是否是有用方向。对实际采样轨迹上的状态定义：

\[
g_t=D_{final}(x_t,t)-D_{inter}(x_t,t),
\]

并利用 toy 中可计算的 Bayes posterior clean prediction：

\[
e_t=D_{Bayes}(x_t,t)-D_{final}(x_t,t).
\]

需要按时间和 inner/middle/outer 区域报告：

- `cos(g_t, e_t)`：外推方向是否朝向 Bayes 修正方向。
- `||Proj_{e_t} g_t|| / ||g_t||`：差值中有多少是有用修正。
- tangent/normal 分解：锐化来自沿流形移动还是法向收缩。
- 一步后局部密度、bridge probability 与完整 rollout 的变化。

这一步会回答：为什么 IG 对外圈强、内圈弱，以及 guidance interval 为什么能显著减小 SWD 代价。只有该方向性机制在多个 seed 上成立，才值得设计自适应系数或投影 guidance。
