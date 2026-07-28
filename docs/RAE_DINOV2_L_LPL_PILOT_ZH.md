# DINOv2-L RAE 上的 Flow/LPL 误差机制 Pilot

## 结论

这个 pilot 支持以下较窄结论：

> 在官方 DINOv2-L RAE decoder 上，LPL 与 Flow 的 latent MSE 基本相同，但两者产生的 latent 预测误差会被 decoder 以明显不同的方式传播。该现象在三个配对训练 seed 上一致。

它还不能证明 LPL 能改善最终生成质量。当前没有官方 DINOv2-L stage-2 prior，本实验使用的是从零训练 250 step 的共同 source prior，再分别继续 100 step，训练程度远低于正式生成模型。

## 模型与协议

- Stage 1：官方 `nyu-visionx/RAE-dinov2-wReg-large-ViTXL-n08`
- Encoder：DINOv2-L/14 with registers，24 层，1024 通道
- Latent：`[B, 1024, 16, 16]`
- Decoder：ViT-XL，28 层
- Stage 2：同一 DiT-DH-S 配置
- 共同 source：Flow objective，从随机初始化训练 250 step
- 分叉：Flow 与 LPL 从同一个完整 step-250 checkpoint 各训练 100 step
- Seed：`3407, 3408, 3409`
- 精度：全程 fp32，关闭 TF32
- LPL 权重：在 source checkpoint 上预先校准为 `4.664743926963856e-4`
- 测试集：8 张未参与训练的 ImageNet-1k validation 图像
- 每张图测试 noise-to-signal ratio：`0.5, 1.0, 2.0, 3.0`
- 总计：每个 seed 32 个观测，三个 seed 共 96 个观测
- 短分叉使用即时 `model` 权重；100 step 的 EMA 只吸收约 4.9% 分支差异，不能用于这个短时 pilot

三个 seed 的首个训练 batch 在图像、标签、时间、噪声、noisy latent、target velocity 和 source prediction 的 SHA256 指纹上均逐项相同，因此 Flow/LPL 差异来自分叉后的训练目标，不是数据或随机数错位。

## 主要结果

下表是三个 seed 几何平均后的 LPL/Flow 比值，小于 1 表示 LPL 更低：

| 误差幅度 | raw feature | strict feature | target-normalized | symmetric-normalized | feature variance |
|---:|---:|---:|---:|---:|---:|
| 0.04 | 0.932 | 0.919 | 0.922 | 0.921 | 1.004 |
| 0.16 | 0.898 | 0.870 | 0.888 | 0.879 | 1.019 |
| 0.32 | 0.873 | 0.807 | 0.856 | 0.833 | 1.056 |
| 0.64 | 0.937 | 0.681 | 0.888 | 0.793 | 1.278 |
| 1.00 | 1.089 | 0.545 | 1.019 | 0.807 | 1.683 |

三个 seed 在完整误差幅度下的 strict feature 比值分别为：

- seed 3407：`0.5441`
- seed 3408：`0.5467`
- seed 3409：`0.5449`

这项结果高度一致。与此同时，完整幅度下 raw feature 比值大于 1，而 prediction feature variance 约为 Flow 的 `1.68x`。因此 raw 指标变差主要混入了预测特征尺度膨胀；经过 cross-normalization 的 strict 指标则显示 LPL 误差方向与 decoder 特征结构更匹配。

## “相同真实 latent，误差从哪里来”

Flow 和 LPL 使用相同的真实 latent `z` 作为监督目标，但它们不是输出同一个预测 latent：

```text
真实 latent：z
Flow 预测：  z_hat_flow
LPL 预测：   z_hat_lpl
```

对应误差是：

```text
e_flow = z_hat_flow - z
e_lpl  = z_hat_lpl  - z
```

本实验比较的正是 `e_flow` 和 `e_lpl` 经过同一个冻结 decoder 后产生的不同响应。共享真实 `z` 是公平配对条件，不代表预测误差相同。

## 不能过度解释的地方

1. source prior 只训练了 250 step，分支只训练了 100 step，不能代表收敛后的 DINOv2-L 生成模型。
2. 官方目前没有发布匹配 DINOv2-L latent 的 stage-2 prior，因此不能做与官方 DINOv2-B checkpoint 等价的成熟模型复现。
3. latent MSE 没有改善，完整幅度下 raw feature loss 也没有改善；结论不是“LPL 全面更优”。
4. 原先用于验证“clean latent 附近局部低敏感方向”的 mechanism gate 仍未通过。当前证据支持 decoder-aware finite-radius alignment，不支持简单的局部 Jacobian 最小特征方向解释。
5. 没有计算 gFID/FID；这个短训练 prior 生成质量不足，强行计算不会回答最终方法是否有效。

## 当前最稳妥的解释

大 DINOv2 上再次出现了同一个核心断层：

> latent 空间的平均误差大小，不能充分决定冻结 decoder 最终如何使用这些误差。

LPL 学到的不是更小的 latent 误差，而是另一种误差组织方式。它在 decoder 的归一化内部表征中明显更接近真实路径，同时会改变预测特征方差。下一步若继续，应把“方向匹配”和“尺度校准”拆成两个目标，而不是只增大 LPL 权重。
