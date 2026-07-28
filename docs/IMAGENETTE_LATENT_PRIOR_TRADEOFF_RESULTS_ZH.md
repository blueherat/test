# Imagenette-64 Latent Prior 与 Decoder 收益权衡：正式结果

## 1. 最终判断

本轮得到的是一个强但边界清楚的正负混合结论：

1. **latent 给 decoder 的收益与使用近似 prior 后的系统代价之间，存在稳定、可测的
   权衡。** 256d 使用真实 latent 的 Oracle FID 比 16d 平均好 `3.39`，但将经验
   latent 分布替换为固定预算 prior 后，modeling gap 平均多 `19.63` FID；两个方向
   都在 `5/5` frozen seed 中一致。
2. **原先预期的中间最优没有出现。** 完整生成在 `5/5` seed 中均为
   `16d < 64d < 256d`，64d 从未最优。平均 End-to-end FID 为
   `103.44 / 106.44 / 118.44`。
3. **“effective rank 或 held-out prior loss 解释 prior 难度”被否定。** 256d 的
   held-out flow MSE 最低，raw latent SWD 也和 16d 相近，却有最大的 decoded gap。
   这说明当前现象不能简单叫作“高维 prior 学不会”。

因此应当区分两句话：

> 得到支持：更丰富的 latent 改善真实-code decoder，但也让完整系统对 prior
> 近似误差更脆弱。

> 没有得到支持：存在 64d 中间最优，且该最优可由 effective rank 或普通 prior
> validation loss 预测。

按照预注册 gate，本轮不进入固定 256 维 rank 干预，也不训练
`decoder loss + prior loss + rank constraint` 的联合方法。当前证据足以确认权衡
现象，但不足以支持原定方法和论文机制。

## 2. 正式规模

- 上游：冻结的 Imagenette-64 `16/64/256d x 5 seeds` encoder 与随机 decoder。
- prior：统一 256 维接口、width 512、6 个 residual blocks，所有容量参数量均为
  `7,096,064`。
- 低维 code 使用固定稠密正交子空间，源高斯和目标 data 位于相同 active subspace。
- 每个 prior：20,000 step、batch 512、fp32、EMA，formal latent sampler 100 NFE。
- 图像评估：完整官方 val 3,925 张，冻结 decoder 50 NFE。
- rollout：Oracle val code、经验 train code、prior code、未经训练 Gaussian code。
- 总计：15 个 prior、300,000 optimizer step、153.6M latent sample draws。
- 标签只用于事后类别覆盖和语义子空间审计，不进入 prior 或 decoder。

## 3. 实现审计

15 个 run 全部通过：

- encoder/decoder 训练前后哈希不变；
- train/val 文件集合无交集，三容量 path hash 完全相同；
- 同一 frozen seed 的三容量 prior 参数量、初始化、batch indices、time 和基础
  256 维 noise 哈希完全相同；
- 正交嵌入往返最大误差不超过 `9.54e-7`；
- 原实验前 1,024 张 Oracle FID 复算误差最大不超过 `1e-4`；
- 全部 loss、权重、latent、features 和 FID 有限；
- 保存 features 独立重算 FID 的误差为 `0`；
- 重新生成 NFE100 latent 后，各 latent 指标最大差为 `1.42e-14`；
- 对最差的 `256d seed2` 独立重采样和重解码，NFE100 FID 与正式值逐位一致。

开发阶段曾有一个 dense-Q FP32 往返误差为 `1.19e-6`，超过预注册 `1e-6` 护栏。
该批在对应 run 训练前停止；实现增加同一子空间内的一次残差修正后，全部正式配置
从零重跑。中止版保留在
`~/data/eqvae/imagenette_latent_prior_tradeoff_aborted_roundtrip_v1/`，不进入结果。

## 4. 主结果

以下是五 seed 均值，括号为 seed 间样本标准差；FID 越低越好。

| latent | Oracle FID | Empirical FID | End-to-end FID | Total gap | Modeling gap | Gaussian FID |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 101.03 (.46) | 100.65 (.24) | **103.44 (.48)** | 2.41 (.48) | 2.79 (.64) | 220.62 (13.74) |
| 64 | 98.25 (1.42) | 96.53 (1.15) | 106.44 (1.37) | 8.19 (.83) | 9.92 (.55) | 214.01 (15.73) |
| 256 | **97.64 (.68)** | **96.02 (.92)** | 118.44 (1.91) | 20.80 (2.12) | 22.42 (1.64) | 211.44 (2.09) |

这里 Empirical rollout 直接从真实 train-code 经验分布抽样，因此
`End-to-end - Empirical` 更接近 prior 本身带来的 decoded modeling gap。

五个 seed 的关键配对差值：

```text
OracleFID_16 - OracleFID_256:
3.39, 4.61, 2.57, 3.98, 2.39

ModelingGap_256 - ModelingGap_16:
21.57, 19.00, 21.27, 18.53, 17.78

EndToEndFID_64 - EndToEndFID_16:
2.87, 3.09, 4.86, 1.49, 2.71
```

这说明不是某个 seed 的偶然 FID 反序：更大 code 的 Oracle 收益稳定，但完整生成
代价增长得更多。

## 5. 预注册 gate

```text
Implementation audit: PASS
Decoder benefit:       PASS (5/5, mean +3.39 FID)
Prior difficulty gap:  PASS (5/5, mean +19.63 FID)
Intermediate optimum:  FAIL (0/5, mean margin -3.00 FID)
Prior vs Gaussian:     PASS (all capacities, all seeds)
Mechanism prediction:  FAIL
Overall positive gate: FAIL
Opposite monotonic gate: FAIL
```

`effective rank / held-out prior loss` 中较好的 LOSO RMSE 为 `2.97`，名义维数基线
为 `1.84`；因此没有达到“至少优于名义维数 5%”的门槛。

## 6. 最反常的机制结果

若“容量越大，prior 越难学”是普通优化意义上的事实，应看到 256d 的 held-out loss
和分布误差更差。实际恰好相反：

| latent | held-out flow MSE | raw latent SWD | real effective rank | generated effective rank |
|---:|---:|---:|---:|---:|
| 16 | .978 | .0299 | 9.93 | 9.81 |
| 64 | .979 | .0594 | 16.44 | 13.05 |
| 256 | **.451** | **.0298** | 20.27 | 20.03 |

256d 的大多数方向很规则，使按维平均的 flow MSE 很低；其全局 covariance、SWD
和 effective rank 也能被 prior 复现。但经过 decoder 后，少量尚未被这些平均指标
捕获的差异产生了很大的图像分布偏差。

所以当前最稳妥的解释是：

> 容量增大不仅给 decoder 更多信息，也提高了 decoder 对 code 分布细微偏差的
> 下游敏感性。当前观测到的是 **decoder-amplified prior mismatch**，而不只是
> prior 的优化困难。

这仍是由结果支持的推断，不是已完成的因果机制。不能把它改写成“effective rank
越大越难生成”。

## 7. NFE 与语义子空间反查

将 latent sampler 从 100 增至 200 NFE 后，平均 FID 改变量为：

```text
16d:  -0.307
64d:  -0.327
256d: -0.301
```

`16 < 64 < 256` 的 end-to-end 排序仍为 `5/5`，所以大 gap 不是 Euler 步数不足。

事后又用 train 标签构造最多 9 维类别判别子空间。nearest-centroid val accuracy
约为 `33% / 38% / 39%`，说明该子空间有语义信息；但 semantic SWD 并未随
modeling gap 增长，普通类别 TV 也没有解释 256d。这个负结果很重要：现有证据
不足以把 mismatch 简化为一个线性类别方向偏差。

## 8. 停止决定

本轮确认了主现象，却没有通过进入方法阶段所需的完整机制 gate：

- 不做预注册中的固定 256 维 rank sweep；
- 不训练 joint prior-aware bottleneck；
- 不把 16d 边界最优事后改写为“中间最优”；
- 不用新的 proxy 替换失败的 effective-rank/flow-loss gate；
- 不扩大容量范围来追求一个更好看的 U 型曲线。

若未来重新立项，应该把新问题明确改成“如何测量或控制 decoder 对 latent
distribution shift 的放大”，并使用新的数据/seed 做独立确认；不能把它作为本轮
预注册结论的延伸。

## 9. 可复核材料

- 预注册：`docs/IMAGENETTE_LATENT_PRIOR_TRADEOFF_PREREG_ZH.md`
- 正式训练：`experiments/imagenette_latent_prior_tradeoff.py`
- 四卡启动：`experiments/run_imagenette_latent_prior_tradeoff_sweep.py`
- 正式汇总：`experiments/summarize_imagenette_latent_prior_tradeoff.py`
- NFE 审计：`experiments/audit_imagenette_latent_prior_tradeoff.py`
- 语义审计：`experiments/analyze_imagenette_prior_semantic_gap.py`
- 外部结果：`~/data/eqvae/imagenette_latent_prior_tradeoff/`
- 汇总表图：`~/data/eqvae/imagenette_latent_prior_tradeoff/comparison_p0/`

仓库内没有写入模型 checkpoint、generated images 或 artifact 目录。
