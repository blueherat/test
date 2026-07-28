# RAE 确定性 Decoder 上的严格 LPL 对照实验

## 研究问题

本实验检验一个具体机制：

> latent 速度回归的均匀 MSE，是否低估了确定性 decoder 特别敏感的 latent 误差方向？

实验不是重新训练 RAE encoder/decoder，也不是用 decoder 替代小型 latent 生成模型。两个分支都只继续训练同一个 `DiTDH-S` latent velocity model：

- `flow`：原始 flow matching loss。
- `flow + LPL`：原始 loss 加上冻结 decoder 中间特征上的 Latent Perceptual Loss。

## 严格对照条件

- 数据：ImageNet-1K train；held-out 指标使用 ImageNet-1K validation。
- RAE：DINOv2-with-registers-base encoder，ViT-XL decoder。
- encoder：冻结、`eval()`。
- decoder：冻结、`eval()`，确定性前向；梯度只穿过 decoder 回到预测 latent 和 DiT。
- DiT：两个分支从同一个 checkpoint 和同一个完整 optimizer/scheduler/RNG 状态开始。
- 数值：全程 fp32，关闭 TF32。
- batch、图像、类别、时间、噪声和初始预测：首批 SHA256 指纹逐项相同。
- 采样：EMA、Euler ODE 50 steps、CFG 1.0、相同 seed `20260715`、相同类别序列。

训练、LPL 权重校准和所有优化更新只读取 ImageNet-1K train。ImageNet validation 及其 ADM 统计只用于训练结束后的 held-out 诊断和生成评估，不参与权重校准、checkpoint 更新或 seed 选择。

源 checkpoint：

`/home/zhoushunyu/data/eqvae/stage2_training/finetune_ditdh_s_adapter_from_official_ep5_lr2e5_4gpu/checkpoints/ep-0000000.pt`

其中 190 个模型 tensor 与本地官方 `DiTDH-S_ep14/stage2_model.pt` 逐项相同。实验没有使用此前模型参数已经变化的 `fair_*` step-0 checkpoint。

## LPL 实现

线性 OT 路径为

```text
z_t = (1 - t) z_0 + t epsilon
v = epsilon - z_0
```

由预测速度得到 clean latent：

```text
z_hat_0 = z_t - t v_theta(z_t, t)
```

严格保留论文目标中的关键部分：

1. 只在 `t / (1 - t) <= 3` 的高信噪比样本上计算 LPL。
2. 比较 decoder 相对深度 `20%/40%/60%/80%/100%` 的五层特征，对应 ViT-XL hidden states `(6, 11, 17, 22, 28)`。
3. target 和 prediction 都使用 prediction 分支的 channel mean/std 做 cross-normalization。
4. 使用 percentile 和 morphology outlier mask。
5. target 特征停止梯度；prediction 特征保留穿过冻结 decoder 的输入梯度。

RAE 的五层特征都在 `16 x 16` token grid 上，因此论文按空间分辨率设置的层权重在这里退化为等权。

原论文权重 `3` 不能直接跨 decoder 架构使用：在当前 ViT decoder 上会让 LPL 主导总 loss 并造成梯度爆炸。实验在训练和 FID 评估前，用源 checkpoint 的 256 个固定训练样本一次性校准：

```text
mean flow loss              = 0.424879
raw gated LPL contribution  = 157.377
eligible rate               = 20.3125%
fixed LPL weight            = 0.0006749373
```

该权重使初始加权 LPL 约占 flow loss 的 `25%`，即约占总 loss 的 `20%`。它不是根据最终 FID 调参。

## Held-out 结果

固定 128 张 ImageNet validation 图像、固定噪声，评价 EMA。下面的百分比为 LPL 分支相对同 step 纯 flow 分支的变化。

| step | noise/signal | decoder feature loss | 变化 | clean latent MSE | 变化 |
|---:|---:|---:|---:|---:|---:|
| 500 | 0.5 | 270.512 vs 281.550 | -3.92% | 基本相同 | 约 0% |
| 500 | 1.0 | 476.148 vs 502.896 | -5.32% | 基本相同 | 约 0% |
| 500 | 2.0 | 768.140 vs 815.250 | -5.78% | 基本相同 | 约 0% |
| 500 | 3.0 | 1006.210 vs 1063.220 | -5.36% | 基本相同 | 约 0% |
| 2000 | 0.5 | 255.783 vs 281.050 | -8.99% | 0.057416 vs 0.056870 | +0.96% |
| 2000 | 1.0 | 439.637 vs 501.717 | -12.37% | 0.091471 vs 0.090467 | +1.11% |
| 2000 | 2.0 | 696.430 vs 813.583 | -14.40% | 0.139219 vs 0.137771 | +1.05% |
| 2000 | 3.0 | 914.280 vs 1061.143 | -13.84% | 0.177690 vs 0.176639 | +0.60% |

这说明 LPL 学到的主要不是更低的平均 latent MSE，而是让同等大小附近的 latent 误差对冻结 decoder 更不破坏。

## 5k 生成结果

这些数值由 `torch-fidelity` 在 5,000 张固定样本上计算，只适合两个严格配对分支之间比较，不能和论文中的 50k gFID 或 RAE 官方指标直接横向比较。

补充的官方起点对照直接采样原始
`DiTDH-S_ep14/stage2_model.pt`，没有创建 optimizer、执行训练或更新 EMA。
它使用相同的 fp32、Euler 50 steps、CFG 1.0、seed `20260715` 和类别序列：

| checkpoint | FID ↓ | KID ↓ | IS ↑ |
|---|---:|---:|---:|
| 官方 epoch 14，0 update | 21.6122 | 0.006439 | 71.8754 |
| flow，500 updates | 21.3610 | 0.006237 | 71.8230 |
| flow + LPL，500 updates | **20.1467** | **0.005596** | **73.9966** |
| flow，2000 updates | 21.8395 | 0.006645 | 68.0297 |
| flow + LPL，2000 updates | **19.6806** | **0.005708** | 70.7485 |

LPL-500 的 FID、KID、IS 都超过官方起点；LPL-2000 的 FID/KID 继续
超过官方起点，但 IS 低 `1.57%`。这排除了“LPL 只比继续训练后退化的
Flow 好”这一解释，但仍只是单个固定 5k 样本集，不是 50k 官方起点终验。

首先在 seed 0 内比较训练进程：

| endpoint | branch | FID ↓ | KID ↓ | IS ↑ |
|---:|---|---:|---:|---:|
| 500 | flow | 21.3610 | 0.006237 | 71.8230 |
| 500 | flow + LPL | 20.1467 | 0.005596 | 73.9966 |
| 2000 | flow | 21.8395 | 0.006645 | 68.0297 |
| 2000 | flow + LPL | 19.6806 | 0.005708 | 70.7485 |

500 step 时，LPL 的 FID 改善 `1.2144`，相对改善 `5.68%`；KID 下降约 `10.27%`。

2000 step 时，LPL 的 FID 改善 `2.1589`，相对改善 `9.89%`；KID 下降约 `14.09%`，IS 提升约 `4.00%`。从 500 到 2000 step，纯 flow 分支的 5k FID 略微恶化，而 LPL 分支继续改善，因而当前单 seed 内的效果不是只存在于短暂的早期 checkpoint。

随后从同一个官方 checkpoint 独立训练 seed 1、2、3。每个 seed 内的 flow 和 LPL 分支使用相同图像、标签、时间、噪声和初始预测；三个 seed 的首批逐项 SHA256 指纹均匹配。采样仍使用同一全局 seed 和类别序列：

| train seed | flow FID ↓ | LPL FID ↓ | FID 改善 | 相对改善 | flow KID ↓ | LPL KID ↓ | flow IS ↑ | LPL IS ↑ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 21.8395 | 19.6806 | 2.1589 | 9.89% | 0.006645 | 0.005708 | 68.0297 | 70.7485 |
| 1 | 22.0543 | 19.4700 | 2.5844 | 11.72% | 0.007198 | 0.005834 | 69.6032 | 72.7129 |
| 2 | 21.9830 | 19.5437 | 2.4393 | 11.10% | 0.007111 | 0.005961 | 68.8070 | 72.5984 |
| 3 | 22.4084 | 20.0234 | 2.3850 | 10.64% | 0.007532 | 0.006224 | 68.8182 | 70.0317 |
| mean | 22.0713 | 19.6794 | 2.3919 | 10.84% | 0.007121 | 0.005932 | 68.8145 | 71.5229 |

四个训练 seed 的 FID、KID 和 IS 改善方向均一致。配对 FID 改善为 `2.3919 +/- 0.1766`（sample standard deviation），相对改善为 `10.84% +/- 0.77%`；平均 KID 下降 `16.7%`，平均 IS 上升 `3.93%`。这仍不是论文级最终 FID，但已经排除了“只在一个训练 seed 上偶然有效”这一主要风险。

## 50k 终验：补充快速协议

终验 seed 在查看其 5k 排名之前预先固定为 train seed 3，避免从四个训练 seed 中挑选最好结果。两个分支使用相同的：

- 源 checkpoint、训练初始化和配对训练数据流；
- `50,000` 张类别均衡样本；
- sampling seed `20260715`；
- `50` 步 Euler、EMA、CFG `1.0`；
- fp32，并关闭 TF32。

两份评估 NPZ 均已核验为 `(50000, 256, 256, 3)` 的 `uint8` 数组。使用与 5k 实验相同的 `torch-fidelity` 快速协议得到：

| branch | FID ↓ | KID ↓ | IS ↑ |
|---|---:|---:|---:|
| flow | 16.7410 | 0.007405 | 93.7008 |
| flow + LPL | 14.3156 | 0.006247 | 98.1091 |

LPL 的 FID 改善 `2.4254`，相对改善 `14.49%`；KID 下降约 `15.64%`，IS 提升约 `4.70%`。这再次与四个 5k 训练 seed 的方向一致，但终验主结论仍以 ADM/guided-diffusion TensorFlow Inception 协议为准。

## 50k 终验：官方 ADM 协议

使用 ADM/guided-diffusion 的 TensorFlow Inception graph 和 ImageNet 256 官方 reference statistics，对上面完全相同的两份 50k NPZ 流式计算：

| branch | ADM FID ↓ | sFID ↓ | IS ↑ |
|---|---:|---:|---:|
| flow | 13.5043 | 12.9084 | 93.7710 |
| flow + LPL | 11.1027 | 9.2378 | 98.1204 |

严格配对的 LPL 分支：

- ADM FID 绝对改善 `2.4016`，相对改善 `17.78%`；
- sFID 绝对改善 `3.6706`，相对改善 `28.44%`；
- IS 提升 `4.3494`，相对提升 `4.64%`。

官方 ADM 与 `torch-fidelity` 两套实现、5k 与 50k 两种样本规模、四个独立训练 seed 的方向全部一致。尤其明显的 sFID 改善支持 LPL 主要修正 decoder 敏感的空间和局部结构误差，但它本身还不能证明完整的 Jacobian/decoder metric 机制；后者需要单独的误差方向实验。

这里的绝对 FID 不能直接与 RAE 原文的大模型、完整训练预算、guidance interval 或其他采样配置比较。本终验的因果问题仅是：在同一个小 DiTDH-S、同一个源 checkpoint 和固定额外 2,000 更新下，加入 LPL 是否优于纯 flow。

## 训练稳定性和成本

在 2,000 个配对更新中：

| branch | mean flow loss | mean grad norm | clip rate | wall time |
|---|---:|---:|---:|---:|
| flow | 0.43447 | 0.36066 | 0.10% | 801.9 s |
| flow + LPL | 0.43753 | 0.51638 | 1.85% | 985.3 s |

LPL 分支墙钟开销约增加 `22.9%`。梯度更大且偶尔触发裁剪，但没有持续震荡或数值发散。

新增 seed 1/2/3 也复现了相同训练机制：

| seed | flow 分支平均 flow loss | LPL 分支平均 flow loss | flow 平均梯度范数 | LPL 平均梯度范数 | LPL clip rate |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.43728 | 0.44004 | 0.36646 | 0.52134 | 1.55% |
| 2 | 0.43663 | 0.43935 | 0.36847 | 0.51721 | 1.65% |
| 3 | 0.43648 | 0.43922 | 0.36777 | 0.52337 | 2.05% |

LPL 使普通 flow loss 略差、梯度范数增大，但生成指标稳定改善。这正是“均匀 latent MSE 与 decoder 实际风险不一致”的实验信号，而不是 LPL 同时把所有训练指标都变好。

另有两个 RAE 特有的实现观察：

1. LPL 时间门控平均只选中约 `20%` 样本。
2. 三个新增 seed 的 outlier mask 在被选样本上的平均保留率均高于 `99.99997%`，在当前 RAE ViT decoder 上几乎没有发挥筛除作用。

第二点说明后续可以移除或重新设计 mask 以节省计算，但不能在本次严格复现中途改动。

## 当前结论边界

目前已有直接证据支持：

1. 确定性 RAE decoder 对 latent 误差方向的敏感度并不均匀。
2. 普通 flow MSE 接近，甚至略差时，decoder feature error 仍可显著降低。
3. 500 和 2000 step 的代理指标改善都转化成了固定 5k 生成比较中的 FID/KID 改善。
4. 从 500 到 2000 step，decoder feature error 和生成指标的改善方向一致，且 LPL 相对收益扩大。
5. 2000 step 的 FID/KID/IS 改善已在四个独立训练 seed 上同向复现。
6. 预先固定 seed 3 的 50k 终验中，LPL 的 ADM FID、sFID 和 IS 全部改善；主指标 ADM FID 相对改善 `17.78%`。

目前还不能声称：

1. LPL 已在不同 RAE encoder、decoder、DiT 尺度或完整训练预算上普遍有效。
2. 已复现原 LPL 论文数值；架构、数据规模和训练步数不同。
3. 当前小模型的绝对 ADM FID 可以直接代表 RAE 原文完整配置的 gFID。
4. 当前权重、层选择、时间门控和 mask 已经最优。

多 seed 与官方 50k 终验门槛均已通过。终验预先固定为 seed 3 的 step 2000 checkpoint，没有根据 5k 排名挑选模型；主结论以 ADM FID 为准，使用 10k reference images 的 `torch-fidelity` 数值只作补充。
