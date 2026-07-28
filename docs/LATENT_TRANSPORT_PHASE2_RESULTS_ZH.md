# Latent Transport 阶段 2：无训练路径审计

## 结论

阶段 2 的数值控制全部通过，可以进入低成本 toy 因果实验，但目前**不能进入 RAE
stage-2 大训练**。结果证明了两种互相独立的破坏机制：

1. 非正交线性变换只改变 source prior/坐标谱，不弯曲 straight chord；
2. nonlinear adapter 同时引入很大的 prior mismatch 和 chord/pushforward mismatch。

它也推翻了一个过强假设：projected VIV 或局部 kNN velocity ambiguity 不能单独
预测 transport 难度。真实 adapter 越强，Gaussian-straight ambiguity proxy 反而
越低，但 source mismatch、Jacobian directional spread 和 bridge defect 都显著增大。
下一阶段必须直接训练并比较 endpoint 分布，不能继续用 proxy 替代因果结果。

正式 artifact：

```text
~/data/eqvae/artifacts/latent_transport_audit/
  dinov2_imagenet_val_n2048_seed20260718/
```

配置为 ImageNet validation 2,048 张、path subset 512 张、3 个 noise/map repeats、
5 个时间点和 4 张 GPU，全程 fp32、关闭 TF32。运行耗时约 `75.7 s`。

## 数据与泄漏审计

- adapter 训练数据：ImageNet train `0..32767`；
- adapter 旧 external eval：ImageNet validation `0..2047`；
- 本阶段：从 ImageNet validation `2048..49999` 固定随机抽取；
- 实际索引数/唯一数：`2048/2048`；最小索引 `2072`；
- 与旧 external eval 重叠：`False`。

Projected shared-class VIV 使用 1,794 个有重复类别的样本、612 个类别、1,182 个
类内自由度。它是固定 256 维投影与 pooled within-class covariance 下的 VIV，
不是原论文 full-dimensional class-specific VIV。kNN velocity ambiguity 也只是
局部条件方差 proxy，结果文件没有把它命名为 VIV。

## 硬控制

| 检查 | 结果 | 门槛 |
|---|---:|---:|
| identity bridge max | `0` | `<=1e-6` |
| identity chord/push velocity gap | `0` | `<=1e-6` |
| signed orthogonal bridge max | `0` | `<=1e-6` |
| orthogonal prior SW1 / identity sampling floor | `1.062` | `<=1.5` |
| anisotropic linear bridge max | `6.37e-8` | `<=1e-6` |
| condition number vs prior SW1 Spearman | `1.000` | `>=0.8` |
| nonlinear adapter cycle max | `2.83e-7` | `<=1e-5` |
| adapter alpha vs bridge Spearman | `1.000` | `>=0.8` |
| bridge vs velocity-gap Spearman | `0.993` | `>=0.6` |
| 上项 bootstrap 95% CI | `[0.927, 1.000]` | 下界 `>0` |

所有预注册控制通过。关键的线性反例也成立：condition number 从 `1.5` 增至 `8`
时，prior projected SW1 从 `0.0396` 增至 `0.3078`，projected VIV/dim 从 `3.161`
增至 `4.296`，但 bridge 始终约 `6e-8`。因此不能再要求混合 map family 的
`defect-VIV` 单一相关性。

## Nonlinear adapter 强度轴

| alpha | flip-h error | prior SW1 | bridge | velocity gap | Jacobian spread | full kNN recall |
|---:|---:|---:|---:|---:|---:|---:|
| 0 / identity | `0.371` | `0.0369` | `0` | `0` | `1.000` | `1.000` |
| 0.25 | `0.279` | `0.257` | `0.091` | `0.175` | `1.316` | `0.905` |
| 0.50 | `0.201` | `0.633` | `0.200` | `0.419` | `2.490` | `0.827` |
| 0.75 | `0.152` | `1.215` | `0.272` | `0.691` | `4.025` | `0.757` |
| 1.00 | `0.124` | `2.072` | `0.293` | `0.903` | `4.698` | `0.692` |

这里出现清楚的多目标冲突：adapter 的训练目标确实成功，flip-h direct error 下降
约 `66.6%`；可逆性也完全没有问题。但它同时把标准 Gaussian source 推得很远，
弯曲了 conditional path，放大方向依赖 Jacobian，并改变局部邻域。

完整 latent kNN 是每个 rank 的 512 张子集内，在原始 196,608 维上精确计算后再
汇总。低维 sketch kNN 不能用于这个判断：signed orthogonal map 的完整 kNN recall
严格为 `1.0`，固定投影后的 recall 却只有约 `0.071`。后者只是不同随机截面造成的
假破坏，正式表中已明确标成 auxiliary。

## 最反常且最有价值的结果

Projected VIV/dim 在 identity 为 `3.117`，adapter `alpha=1` 仍为 `3.117`；它几乎
看不到 source/path 的巨大变化。更反常的是平均 local velocity ambiguity：

| alpha | Gaussian straight | Matched chord | Pushforward |
|---:|---:|---:|---:|
| 0 | `0.752` | `0.752` | `0.752` |
| 0.25 | `0.748` | `0.749` | `0.750` |
| 0.50 | `0.740` | `0.746` | `0.749` |
| 0.75 | `0.726` | `0.750` | `0.735` |
| 1.00 | `0.703` | `0.761` | `0.699` |

即使 Gaussian-straight 的 source mismatch 最大，它在固定 sketch/Euclidean kNN
定义下反而显得“更可预测”。原因并不神秘：该 ratio 会随坐标尺度、局部距离、
velocity 方差归一化和邻域重排改变；它不是 coordinate-invariant complexity。
这恰好说明只靠 VIV/proxy 选择 latent 或 path 会误判。

## 阶段决策

阶段 2 通过的是“实现与机制可分性”，不是方法有效性。下一阶段只授权低成本、
严格配对的四分支 toy：

```text
Base
Gaussian straight
Matched chord
Pushforward
```

四者最终必须逆变换回同一原始分布比较 endpoint。若 Gaussian-straight 没有形成
至少 10% 的稳定劣势，则该 toy 没有可恢复 gap；若有 gap 但 Pushforward 未在
至少 4/5 seeds 恢复 50%，停止“coordinate-consistent path 自动恢复”的主张。
无论结果如何，在该门槛通过前不进行 CIFAR 或 RAE stage-2 训练。

## 代码

- `experiments/latent_transport_audit_metrics.py`
- `experiments/audit_latent_transport_compatibility.py`
- `tests/test_latent_transport_audit_metrics.py`
- `experiments/latent_transport_paths.py`
