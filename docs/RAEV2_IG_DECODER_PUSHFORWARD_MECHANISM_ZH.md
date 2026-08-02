# RAEv2 IG Decoder Pushforward 机制检查

## 结论

本轮没有训练或重新采样模型，而是复用两次独立的 5,000 样本 scale-response
实验，检查 internal guidance（IG）从原始 latent 经过 `E(D(.))` 后发生了什么。

结果排除了两个过于简单的解释：

1. `E(D(.))` 并不是简单地把 IG 偏移整体压缩掉。在 `s=1.78`，roundtrip
   增量范数反而约为 raw 增量的 `1.35x`，方向余弦只有约 `0.52`。
2. IG 没有抵消 autoencoder 的 reconstruction bias。在 held-out classes 上，
   IG 增量与 reconstruction bias 的 whitened cosine 为正。

但 `E(D(.))` 确实强烈削弱了 raw latent 中使真实/生成分布可分的方向：
`s=1.78` 时，该方向在 roundtrip 增量中只保留约 `12%-15%`。与此同时，IG
增量与 `s=1` decoded distribution 的总误差稳定反向。

当前最符合证据的描述是：

> `E(D(.))` 对 IG 偏移进行强烈的各向异性重加权和旋转。它没有缩小全部偏移，
> 但大幅削弱 raw latent 的判别方向，并将另一部分偏移映射成对 full decoded
> distribution error 的修正。

这支持 decoder-mediated ranking reversal，但还不是“decoder 单独导致反转”的因果
证明。

## 实验协议

- 模型：官方冻结的 RAEv2 ImageNet DINOv3-L K7，EMA 权重；
- sampler：官方 100-step ODE，`base + s * (full - base)`；
- scale：`1.0, 1.2, 1.4, 1.6, 1.78, 2.0, 2.2`；
- 两个独立 seed：`20260801`、`20260802`；
- 每个 seed 5,000 样本；不同 scale 使用相同标签和初始噪声；
- 4,000 个 train-class 样本拟合 probe，1,000 个未见 class 样本只用于验证；
- 所有机制统计都由已有 artifact 计算，没有更新模型参数。

## 检查一：配对 IG 增量经过 `E(D(.))` 后如何变化

对相同噪声、相同标签的样本定义：

```text
raw increment       = z_s - z_1
roundtrip increment = E(D(z_s)) - E(D(z_1))
```

另外，在 train classes 上拟合一个区分 `E(x)` 与 `z_s` 的 diagonal-LDA probe，
再检查 raw 判别方向在 roundtrip increment 中保留多少。

| seed | scale | roundtrip/raw norm | raw-roundtrip cosine | raw probe survival |
|---|---:|---:|---:|---:|
| 20260801 | 1.78 | 1.3489 | 0.5204 | 0.1180 |
| 20260802 | 1.78 | 1.3538 | 0.5214 | 0.1461 |

因此，`E(D(.))` 不是简单的 contraction：它增大了总增量范数，同时将增量旋转
约 59 度。真正被削弱的是 raw latent 的 distribution-discriminative component，
而不是全部 IG 增量。

## 检查二：IG 是否抵消 decoder reconstruction bias

在 Inception feature 中定义：

```text
reconstruction bias = mean(D(E(x))) - mean(x)
full decoded error  = mean(D(q_1)) - mean(x)
IG increment        = mean(D(q_s)) - mean(D(q_1))
```

每个 feature 使用真实图像方差做 diagonal whitening。`s=1.78` 的 held-out
结果为：

| seed | IG vs reconstruction bias | IG vs full decoded error | mean-error ratio |
|---|---:|---:|---:|
| 20260801 | +0.1719 | -0.5754 | 0.7348 |
| 20260802 | +0.1254 | -0.6454 | 0.6163 |

`IG vs reconstruction bias` 为正，因此“IG 抵消 autoencoder reconstruction
bias”被当前数据反证。`IG vs full decoded error` 则稳定为负，说明 IG 更像是在
修正 Stage-2 full sampling 已经产生的 decoded distribution error。

这里的 mean-error ratio 只衡量 whitened feature mean；它不是完整 FID，也不能单独
证明全部图像分布变近。

## 检查三：反转是否只来自 Inception

从保存的 `D(z_s)` 图像中提取 `16 x 16` block mean 和 block standard deviation，
得到 1,536 维低频/局部对比度特征。probe 仍只在 train classes 上拟合。

| scale | seed 20260801 AUC separability | seed 20260802 AUC separability |
|---:|---:|---:|
| 1.00 | 0.6189 | 0.5997 |
| 1.40 | 0.5605 | 0.5461 |
| 1.78 | 0.5207 | 0.5098 |
| 2.00 | 0.5134 | 0.5004 |
| 2.20 | 0.5174 | 0.5122 |

越接近 `0.5`，重构图像与生成图像越难被线性区分。两个 seed 都在 `s=2.0`
附近达到最低点，因此反转至少也存在于简单低频像素统计中，不是 Inception 独有。

但低频 probe 的最优点为 `2.0`，此前 FID/roundtrip 的低谷约为 `1.4-2.0`，说明
“精确最优 scale”仍依赖观察空间和指标，不能把 `1.78` 当成唯一机制常数。

## 当前能说和不能说的结论

可以说：

- raw latent distance 的增大主要发生在 `E(D(.))` 不保留的判别方向上；
- decoder/roundtrip map 对不同 latent 方向具有高度不同的响应；
- IG 修正的是 full decoded distribution error，而不是简单补偿 AE 重构偏差；
- decoded ranking reversal 在 DINO roundtrip、Inception 和低频像素特征中都存在。

暂时不能说：

- decoder 单独导致了全部排序反转，因为 roundtrip 指标还包含 encoder `E`；
- IG 后的完整图像分布严格更接近真实分布；
- 当前结果已经提供了可直接训练的 mismatch loss；
- 只根据总范数即可区分 decoder-visible 和 decoder-invisible 方向。

下一项最有判别力的实验是固定所有 `q_s`，只改变 decoder 或 decoder checkpoint，
观察最优 scale 和方向保留率是否系统移动。在此之前，不应据本轮结果重新训练
adapter 或设计新的 perceptual loss。

## 代码与本地结果

- 分析入口：`experiments/analyze_raev2_decoder_pushforward_mechanism.py`
- 单元测试：`tests/test_analyze_raev2_decoder_pushforward_mechanism.py`
- 本地结果目录：
  `/home/zhoushunyu/data/eqvae/experiments/raev2_ig_decoder_pushforward_mechanism/n5000x2_v1`
