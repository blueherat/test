# Transport Risk Atlas 与训练路径反转：结果报告

## 结论

本轮没有得到可在训练前预测 endpoint/FID 伤害的通用分数，因此不进入优化器约束或
RAE 大规模训练。真正的新发现是：

> 等谱加权的平均风险由基底方向决定，但单次训练的收益符号由优化盆地和晚期
> rollout calibration drift 决定。静态 baseline 梯度、final-checkpoint 局部
> directional derivative 和 10% 训练预算的单点代理都不能跨数据集预测该符号。

这把原来的“谱条件数导致慢收敛”假设进一步收紧为一个路径依赖问题，而不是静态
预条件问题。

## 1. Baseline-only Atlas

固定 prospective score 只读取 baseline checkpoint 上的三个量：

```text
allocation_collapse
* gradient_decoupling
* log1p(detail_pressure)
```

实现会主动丢弃 weighted-checkpoint 行；反转全部 endpoint 标签不会改变分数。

30 个已有条件上的回顾性结果：

| 指标 | 结果 | 原 gate |
|---|---:|---:|
| aggregate Spearman | 0.568 | >= 0.70 |
| endpoint-harm ROC-AUC | 0.733 | >= 0.80 |
| 固定阈值符号准确率 | 0.733 | >= 0.80 |
| between-basis Spearman | 1.000 | diagnostic |
| within-basis seed Spearman | -0.027 | diagnostic |
| structured-vs-random pair accuracy | 0.900 | diagnostic |

FashionMNIST Spearman 为 `0.940`，MNIST 只有 `0.068`。Atlas 能解释 DCT/PCA/random
三类基底的平均差异，不能解释同一基底内部的训练 seed。

结果目录：

```text
$HOME/data/eqvae/experiments/transport_risk_atlas/
  retrospective_v1_20260716_234146/
```

## 2. MNIST seed 4 不是抽样噪声

原 1,024-sample 结果中，MNIST seed 4 baseline feature-FID 为 `153.5`，明显差于
其余 seed 的约 `50--67`。为排除一次 rollout 的偶然性，冻结已有 velocity fields，
用 4,096 张测试图和 5 个新 rollout seed 复评：

| training seed | DCT ratio | PCA ratio | random ratio |
|---:|---:|---:|---:|
| 3 | 1.753 +/- 0.050 | 2.261 +/- 0.056 | 0.972 +/- 0.012 |
| 4 | 0.672 +/- 0.011 | 0.789 +/- 0.008 | 1.063 +/- 0.001 |

每个方法在 5/5 rollout seed 上方向一致。seed 4 的 DCT/PCA 改善和 random 恶化都
是真实 checkpoint 行为，不是有限样本 FID 抖动。

结果目录：

```text
$HOME/data/eqvae/experiments/small_image_checkpoint_resample/
  resample_20260716_235259/
```

## 3. 带符号局部 leverage 也失败

在 frozen baseline 上计算普通 MSE update、候选谱加权 update、两者差值和 train-only
differentiable-rollout moment loss 对参数的梯度，再计算 directional derivative。

50-step、64-image pilot 中，raw local derivative 对 seed 3 的 DCT/PCA 主要预测有益，
对 seed 4 则强烈预测有害，与完整训练结果相反。`1e-4` 相对步长的有限差分在代表性
条件上复现 autograd 符号，所以失败不是简单的梯度符号实现错误。

局部导数回答的是“在最终 baseline 附近迈一步会怎样”；原 weighted model 从共同
随机初始化走了 1,000 步，二者不是同一个问题。

结果目录：

```text
$HOME/data/eqvae/experiments/small_image_signed_leverage/
```

## 4. 精确训练路径重放

对 MNIST/FashionMNIST、DCT/PCA/random、5 个 seed 重放完整 paired training，固定
原 batch/noise stream，在 steps `0/25/50/100/200/400/600/800/1000` 上计算独立
train-only rollout moment。每个最终 baseline/weighted 参数 hash 均与原 checkpoint
完全一致。

最反常的路径现象是：

- DCT/PCA 在两个代表 seed 的前 100 步都暂时具有更低 moment loss；
- 约 step 200 后曲线进入不同盆地，早期优势不再决定最终方向；
- MNIST seed 4 baseline band-0 gap 从 step 600 的 `-0.051` 漂到 step 1000 的
  `-0.261`，weighted 则从 `-0.316` 恢复到 `-0.070`；
- MNIST seed 3 最终 baseline/weighted gap 为 `-0.087/-0.203`，方向相反。

探索性地在 MNIST 选择 step 100，band-0 change 与最终 FID damage 的 Spearman 为
`0.855`。在看结果前固定相同 step 和公式到 FashionMNIST，Spearman 只有 `0.067`，
因此该早期预测不泛化。

训练完成后的 endpoint moment difference 在两数据集 DCT/PCA 20 个条件上与 FID
damage 的 Spearman 为 `0.802`，说明 rollout distribution proxy 有解释价值；但它在
100% 训练成本后才可靠，且符号准确率只有 `65%`，不能冒充事前预测器。

结果目录：

```text
$HOME/data/eqvae/experiments/small_image_training_path/
  path_20260717_002529/  # MNIST seeds 3,4
  path_20260717_003018/  # MNIST seeds 0,1,2
  path_20260717_003800/  # FashionMNIST seeds 0--4
```

## 5. 当前机制边界

本轮支持：

1. 等谱基底方向决定平均 gradient allocation 风险；
2. endpoint rollout moment 比 teacher MSE 更接近最终质量；
3. 最终收益符号具有 basin/path dependence；
4. 晚期训练可以在 teacher MSE 没有明显异常时破坏 endpoint calibration。

本轮不支持：

1. 静态 gradient risk 足以预测单次训练；
2. endpoint sensitivity 的无符号范数足以预测收益；
3. final-checkpoint 单步 adjoint 可以外推从头训练；
4. 10% budget 的单点 band-0 指标可以跨数据集排序；
5. 当前证据足以进入 RAE 或提出新的谱优化器。

## 6. 唯一合理的下一道低成本 gate

若继续该方向，只建议测试 **train-only rollout checkpoint selection**：

- 从 step 200 开始每 100 步，在未参与 minibatch update 的训练图像上计算固定的
  canonical-DCT moment proxy；
- baseline 和 weighted 各自独立选择 checkpoint；
- 选择过程不读取 test/FID classifier；
- 最后在完全独立 test indices 和 rollout seeds 上比较 selected 与 final；
- 5 seeds、MNIST 和 FashionMNIST 都必须达到 mean feature-FID ratio `<=0.98`，且
  不允许只修复一个异常 seed。

这项实验已使用未见 seeds `5--9` 前瞻完成，H1 和 H2 均失败。FashionMNIST 的
baseline/DCT/PCA selected-vs-final ratio 为 `1.206/1.124/1.015`；公平选点后
DCT/PCA 相对 baseline 在 MNIST 为 `1.238/1.346`，在 FashionMNIST 为
`1.249/1.422`。当前谱预条件方法线因此关闭。详见
[`ROLLOUT_CHECKPOINT_SELECTION_RESULTS_ZH.md`](ROLLOUT_CHECKPOINT_SELECTION_RESULTS_ZH.md)。
