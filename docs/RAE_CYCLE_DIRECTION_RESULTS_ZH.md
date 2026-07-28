# RAE Cycle-Direction 因果干预结果

## 一句话结论

这条路线的核心假设未通过：`E(clamp(D(z))) - z` 确实是一个稳定、逐样本特异的
cycle-error 下降方向，但它不是把 generated latent 从 decoder 危险区拉回来的方向。
在全部生成路径和全部测试样本上，cycle error 与 D2 早期激活异常呈稳定的反向因果响应。

## 实验范围

- 模型：frozen RAE-DINOv2 encoder 和官方 ViT-XL decoder。
- 生成端点：`static / annealed / reverse / random` 四条路径，共享 noise、label 和 sample index。
- 校准：每路径 128 个样本，索引 `[0, 128)`。
- 独立测试：每路径 128 个样本，索引 `[128, 256)`。
- clean guardrail：128 个未进入上述生成校准/测试的 ImageNet latent。
- 数值：4 卡分布式、fp32、TF32 关闭；encoder/decoder/Inception 全部冻结。

校准只查看 own direction。`alpha=0.025` 是唯一使四条路径的 median Inception
feature cosine 都不低于 `0.98` 的候选，因此在打开 held-out controls 之前被锁定。

## 独立测试结果

| 路径 | own cycle ratio | cycle 下降 | D2 baseline -> own | own 配对 D2 变化 | D2 baseline -> opposite | own feature cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| static | 0.9887 | 1.13% | 1.369 -> 1.587 | +0.217 | 1.369 -> 1.131 | 0.9906 |
| annealed | 0.9872 | 1.28% | 1.921 -> 2.166 | +0.283 | 1.921 -> 1.640 | 0.9881 |
| reverse | 0.9860 | 1.40% | 6.660 -> 6.934 | +0.321 | 6.660 -> 6.305 | 0.9862 |
| random | 0.9878 | 1.22% | 1.456 -> 1.695 | +0.226 | 1.456 -> 1.201 | 0.9900 |

表中的 D2 baseline/condition 是各条件的边际中位数，“配对 D2 变化”是先逐样本相减再取中位数，
因此两者不必完全相减。

配对 bootstrap 95% CI 进一步表明这不是少数异常样本：

- static D2 变化 `+0.217 [0.205, 0.227]`；
- annealed D2 变化 `+0.283 [0.267, 0.290]`；
- reverse D2 变化 `+0.321 [0.301, 0.338]`；
- random D2 变化 `+0.226 [0.215, 0.232]`。

own 在 512/512 个 generated samples 上都使 D2 升高，opposite 在 512/512 个样本上都使
D2 降低。这个符号在 clean guardrail 上也一致：own 使 cycle ratio 降到 `0.9924`，
D2 配对中位变化为 `+0.019 [0.017, 0.021]`，同时 feature cosine 为 `0.99984`。

## 对照与量级

own 方向相比 shuffled/random 的 cycle 改善是稳定的，说明 residual 包含逐样本信息；
但效应量很小。四路径 own cycle ratio 的平均为 `0.98746`，shuffled 为
`0.99575`，random 约为 `1.00003`。own 相对最好对照的优势只有 `0.00829`，
远低于事先规定的 `0.05`。

实际步长只有 latent RMS 的约 `2.6%-3.2%`，但 reverse 的单样本 feature cosine
最低已到 `0.9642`。因此不能通过继续增大 alpha 把一个微小 cycle 改善放大成质量结论。

## 门槛判定

| 预注册检查 | 结果 |
| --- | --- |
| reverse D2 异常降低至少 50% | 失败，反而增加约 4.1% |
| reverse cycle excess 降低至少 25% | 失败，仅 1.6% |
| 至少 3/4 路径 cycle 改善 | 通过，4/4 |
| own 比最好对照低至少 0.05 | 失败，仅 0.00829 |
| opposite 加重 reverse cycle 或 D2 | 通过，cycle 加重 |
| generated median feature cosine 均不低于 0.98 | 通过，最低 0.98625 |
| clean median feature cosine 不低于 0.98 | 通过，0.99984 |

综合 gate 为 **FAIL**，按协议不进入 5k FID/KID/FDD。

## 机制解释

1. `r(z)=E(clamp(D(z)))-z` 可以被解释为指向当前 closure map 输出的方向，但不是已知流形的法向。在小步长下它降低 fixed-point residual 并不意外。
2. 原假设预期 own 同时降低 cycle error 和 D2 异常，opposite 同时加重两者。实验看到了完全稳定的相反符号，因此两个量不在描述同一个“离分布距离”。
3. D2 hidden RMS 可以用来描述内部轨迹，但不能单独当作 latent 健康度或图像质量的单调指标。它与 FID 在四条生成路径之间的排序相关，不等于对 D2 做局部改变就会改善 FID。
4. 因此，不应以 cycle residual 作为 adapter 监督方向，也不应继续把“cycle 更小”当作生成更好的代理目标。

## 研究决策

这一轮已经足以停止当前 cycle-direction 路线，无需再用更大 alpha 或更大样本重复。
它留下的有价值结论是：**RAE generated-latent 的风险不能压缩成一个 cycle distance，
且 decoder 单层边际激活也不是质量目标。**

若继续做低成本机制研究，下一个可验证问题应是“哪些 generated-clean 结构化差异方向真正放大图像级变化”，
而不是继续优化 cycle 或 D2。任何新 proxy 都必须先通过等范数方向干预和小规模真实生成质量验证，再允许进入训练。

## 复现与数据

运行命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  experiments/run_rae_cycle_direction_intervention.py \
  --run-name generated_cal128_test128_clean128_seed20260718
```

完整原始表、分层轨迹、图和 `result.json` 位于：

`~/data/eqvae/experiments/rae_cycle_direction/generated_cal128_test128_clean128_seed20260718/`
