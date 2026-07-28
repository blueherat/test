# Rollout Checkpoint Selection 前瞻结果：频谱方法线的最终否定

## 最终判断

预注册的两个假设均被前瞻实验否定：

- **H1 失败**：固定的 train-only canonical-DCT moment proxy 不能跨数据集稳定选择
  更好的 checkpoint；
- **H2 失败**：即使 baseline、DCT、PCA 都被公平独立选点，DCT/PCA 仍显著差于
  baseline。

因此，当前“逆方差谱加权 + rollout moment guardrail”方法线证据已足够关闭，不应
继续扩展到 RAE、DiT 或更大数据集。

## 实验完整性

该实验只使用此前未观察的训练 seeds `5--9`：

- 数据集：MNIST、FashionMNIST；
- 每数据集 5 个训练 seed；
- 每个 checkpoint 使用 256 张与 minibatch update 完全不相交的训练图像选点；
- 候选 steps：`200,300,...,1000`；
- 每个最终比较使用 4,096 张测试图像和 3 个独立 rollout seed；
- baseline、DCT、PCA、等谱 random 从同一初始化、同一 batch/noise stream 联合运行；
- update/selection indices overlap 在 10/10 任务中均为 `0`；
- MNIST/FashionMNIST classifier accuracy 分别为 `95.4%/82.6%`；
- checkpoint selection 未读取 test image、label、classifier feature、FID 或 SWD。

合并结果：

```text
$HOME/data/eqvae/experiments/small_image_rollout_checkpoint_selection/
  combined_prospective_20260717_012754/
```

## H1：selection 能否阻止自身晚期漂移

下表为 `selected checkpoint / final checkpoint` 的 feature-FID ratio，越低越好。
预注册 gate 为 mean ratio `<=0.98` 且至少 `3/5` seeds 获胜。

| dataset | model | mean ratio | seed wins | 结论 |
|---|---|---:|---:|---|
| MNIST | baseline | 0.769 | 3/5 | 通过该单元 |
| MNIST | DCT | 0.998 | 2/5 | 失败 |
| MNIST | PCA | 1.017 | 1/5 | 失败 |
| FashionMNIST | baseline | 1.206 | 1/5 | 失败且明显恶化 |
| FashionMNIST | DCT | 1.124 | 3/5 | 均值明显恶化 |
| FashionMNIST | PCA | 1.015 | 1/5 | 失败 |

MNIST baseline 的结果说明 rollout validation 偶尔确实能发现晚期坏 checkpoint，但它
不是跨模型、跨数据集规律。FashionMNIST baseline 被 proxy 选择后反而平均恶化
`20.6%`，直接否定 H1。

## H2：selection 能否救活谱加权

下表为 `structured selected / baseline selected` 的公平 feature-FID ratio。

| dataset | basis | mean ratio | seed wins | 结论 |
|---|---|---:|---:|---|
| MNIST | DCT | 1.238 | 0/5 | 强否定 |
| MNIST | PCA | 1.346 | 0/5 | 强否定 |
| FashionMNIST | DCT | 1.249 | 1/5 | 强否定 |
| FashionMNIST | PCA | 1.422 | 0/5 | 强否定 |

四个预注册单元全部远离 `0.98` gate。结果不依赖某个异常 seed：三个单元是 `0/5`
胜出，剩余 DCT/FashionMNIST 也只有 `1/5`。

## 为什么旧相关性不能转化为 model selection

旧实验中，训练完成后的 endpoint moment difference 与 FID damage 有 Spearman
`0.802`，看起来像一个质量信号。前瞻干预证明它主要是相关症状，而不是充分因果量。

| dataset | selected/final proxy | selected/final FID | proxy-FID Spearman |
|---|---:|---:|---:|
| MNIST | 0.565 | 0.895 | 0.583 |
| FashionMNIST | 0.515 | 1.141 | 0.029 |
| combined | 0.540 | 1.018 | 0.265 |

FashionMNIST 中 proxy 平均改善 `48.5%`，FID 却恶化 `14.1%`，二者几乎零相关。这是
最直接的反证：模型可以显著改善 8-band marginal energy matching，同时让真实生成
分布更差。

语义审计给出一致解释。被 proxy 选中的 checkpoint 在两个数据集、四种路径上均有
更大的 class-entropy gap：

- FashionMNIST gap 增量：baseline `+0.047`、DCT `+0.037`、PCA `+0.073`、random
  `+0.043`；
- MNIST gap 增量：baseline `+0.025`、DCT `+0.010`、PCA `+0.011`、random
  `+0.039`。

也就是说，频带能量更匹配时，类别覆盖反而系统性变差。低维二阶统计没有约束
semantic allocation、cross-band dependency、mode coverage 或完整 feature covariance。

## Gap 的最终答案

此前的问题是：为什么 teacher MSE 和早期轨迹相似，最终生成却发生 seed-dependent
符号反转？现有证据支持的答案是：

> 决定终点质量的是高维、路径依赖的联合分布运输；频谱条件数、单点梯度几何和
> marginal band moments 只能描述其中一个投影。优化路径可以在这些投影继续改善时，
> 沿语义覆盖和跨方向依赖进入更差盆地。

这不是“还差一个更好的频段权重”。当前证据已经否定以下链条：

```text
inverse-variance weighting
-> better spectral optimization
-> better endpoint moments
-> better generation
```

前两步可以局部成立，第三个箭头不成立。

## 研究决策

1. 停止继续调 `gamma`、频段、DCT/PCA basis、moment proxy 或 checkpoint schedule；
2. 不把该方法带到 RAE/DiT；
3. 保留“等谱基底方向决定梯度预算”和“pointwise risk 与 endpoint transport 断层”
   作为可靠机制发现；
4. 不再把 endpoint band moment 称为 generative-friendly predictor；
5. 若开启新方法线，必须直接建模高维联合/语义分布或有效 endpoint objective，那是
   新假设，不能视为当前谱方法的自然补丁。

当前 gap 已通过前瞻反证得到结论，无需继续增加同类小实验。
