# SiT v800 冻结中间 x 预测头实验

该目录保存“冻结 SiT-S/2 `v800` EMA，在第 8/12 个 block 后接一个独立 `FinalLayer`，仅用 clean-x 目标训练该中间头 50K step”的结果。checkpoint、训练日志和生成样本 NPZ 留在本机数据盘，不写入 Git。

## 实验定义

中间头结构与 Internal Guidance 官方 SiT 实现一致：从第 8 个 block 的 token 接一个独立的 AdaLN `FinalLayer`。本实验只训练新增头，不更新原模型：

```text
x_t        = (1 - t) * noise + t * clean
v_full     = frozen v800 final output
x_internal = trained FinalLayer(block_8_features, t, class)
v_internal = (x_internal - x_t) / max(1 - t, 0.05)
v_gamma    = v_full + gamma * (v_full - v_internal)
```

训练目标是原生 `MSE(x_internal, clean)`；采样前才把 x 预测转换到统一 velocity 空间。分母下限 `0.05` 与仓库中的 JiT-style x 预测协议一致。它避免数据端数值发散，也意味着最后 5% 时间区间的转换并非无截断的 Bayes 等价式。

## 配置与审计

- 模型：SiT-S/2，共 12 个 transformer blocks；
- source：800K checkpoint 的 EMA 权重；
- 中间位置：block 8；
- 新增/可训练参数：`301,840`，source 的 `32,617,760` 个参数全部冻结；
- 数据：ImageNet-100 cached SD-VAE latents；
- 训练：50K step，global batch 256，LR `1e-4`，uniform time，native clean-MSE；
- 采样：Dopri5、FP32/TF32、CFG=1、两张 GPU；
- 指标：同一 noise、labels、ODE、VAE 和 ADM reference 下的配对 FID/sFID/IS，1,000 samples；
- full 条件的样本 NPZ 与此前独立 v800 baseline 的 SHA256 逐字节一致。

## 结果

50K validation：

| 输出 | native x-MSE ↓ | 转换后 v-MSE ↓ |
|---|---:|---:|
| frozen full v | - | 0.793684 |
| internal x，raw | 0.258919 | 1.294079 |
| internal x，EMA | 0.258922 | 1.297204 |

配对 FID-1K：

| 条件 | gamma | FID ↓ | sFID ↓ | IS ↑ | total NFE |
|---|---:|---:|---:|---:|---:|
| full v800 | 0 | 86.9048 | 220.1673 | 29.2452 | 6,714 |
| internal x only | - | 221.7920 | 274.2017 | 6.4770 | 8,802 |
| extrapolation | 0.01 | 86.2399 | 219.8295 | 29.3707 | 6,810 |
| extrapolation | 0.03 | 84.9688 | 219.2164 | 29.7178 | 6,918 |
| extrapolation | 0.05 | 83.7629 | 218.6091 | 29.7622 | 7,182 |
| extrapolation | 0.08 | 82.1534 | 217.8801 | **29.8296** | 8,034 |
| extrapolation | 0.10 | 81.6559 | 217.4374 | 29.6236 | 8,646 |
| extrapolation | 0.12 | 81.2929 | 217.1056 | 29.3605 | 8,988 |
| extrapolation | **0.15** | **81.2270** | 216.7086 | 28.8085 | 9,450 |
| extrapolation | 0.18 | 81.7230 | 216.5961 | 28.2938 | 9,360 |
| extrapolation | 0.20 | 82.3177 | 216.5310 | 27.7294 | 9,414 |
| extrapolation | 0.40 | 96.3421 | 220.2606 | 20.9938 | 9,990 |
| extrapolation | 0.60 | 119.3032 | 220.4493 | 14.8717 | 10,506 |
| extrapolation | 0.80 | 137.1756 | **215.0703** | 13.7944 | 10,920 |
| extrapolation | 1.00 | 161.4859 | 218.3965 | 10.9149 | 11,052 |
| extrapolation | 1.50 | 216.2938 | 231.7206 | 6.6250 | 14,820 |

最佳已测 FID 点 `gamma=0.15` 相对 full v800 改善 `5.6779`，即 `6.53%`。`gamma=0.08` 的 IS 最高；继续增大系数后 FID 和 IS 很快恶化。与相同设置的中间 v 头相比，x 头的有效系数区间更窄、最佳增益更小。

## 文件

| 文件 | 内容 |
|---|---|
| `summary.json` | 配置、目标公式、checkpoint 哈希、冻结/配对审计和完整指标 |
| `training_validation.csv` | 5K 至 50K 的 raw/EMA validation |
| `final_time_bins.csv` | 50K 的五个时间区间指标 |
| `fid1k.csv` | 16 个严格配对条件的 FID、sFID、IS、NFE、显存与样本指纹 |
| `training_validation.png` | 转换后 v-MSE 及 full-internal gap 曲线 |
| `fid1k_sweep.png` | 外推系数的 FID/IS/NFE 曲线 |
| `preview_comparison.png` | 相同 noise/labels 下的 full、internal 与三个外推点 |

以上文件可由下列命令从正式 artifact 重建并重新执行哈希/配对检查：

```bash
python experiments/summarize_imagenet100_sit_frozen_internal_v_head.py \
  --prediction-target clean
```

本机未纳入 Git 的主要产物包括 16 份生成 NPZ，共 `3,145,732,224` bytes，以及 50K head checkpoint `4,865,705` bytes。
