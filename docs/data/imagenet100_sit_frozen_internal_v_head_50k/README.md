# SiT v800 冻结中间 v 头实验

该目录保存“冻结 SiT-S/2 `v800` EMA，在第 8/12 个 block 后接一个独立 `FinalLayer`，仅训练该中间 v 头 50K step”的轻量结果。checkpoint、训练日志和生成样本 NPZ 留在本机数据盘，不写入 Git。

## 实验定义

中间头结构与 Internal Guidance 官方 SiT 实现一致：从第 8 个 block 的 token 接一个独立的 AdaLN `FinalLayer`。本实验只训练新增头，不更新原模型：

```text
v_full     = frozen v800 final output
v_internal = trained FinalLayer(block_8_features, t, class)
v_gamma    = v_full + gamma * (v_full - v_internal)
```

这不是官方联合训练的完整复现。官方实现同时训练主模型与辅助头；本实验是 frozen-backbone probe，用来回答“已有 v800 是否能通过一个很小的内部头获得 IG 收益”。

## 配置与审计

- 模型：SiT-S/2，共 12 个 transformer blocks；
- source：800K checkpoint 的 EMA 权重；
- 中间位置：block 8；
- 新增/可训练参数：`301,840`，source 的 `32,617,760` 个参数全部冻结；
- 数据：ImageNet-100 cached SD-VAE latents；
- 训练：50K step，global batch 256，LR `1e-4`，uniform time，native v-MSE；
- 采样：Dopri5、FP32/TF32、CFG=1、两张 GPU；
- 指标：同一 noise、labels、ODE、VAE 和 ADM reference 下的配对 FID/sFID/IS，1,000 samples；
- full 条件的样本 NPZ 与此前独立 v800 baseline 的 SHA256 逐字节一致。

## 结果

50K validation：

| 输出 | v-MSE ↓ |
|---|---:|
| frozen full v | 0.793684 |
| internal v，raw | 0.880894 |
| internal v，EMA | 0.880931 |

配对 FID-1K 粗扫：

| 条件 | gamma | FID ↓ | sFID ↓ | IS ↑ | total NFE |
|---|---:|---:|---:|---:|---:|
| full v800 | 0 | 86.9045 | 220.1695 | 29.2518 | 6,714 |
| internal only | - | 225.7290 | 276.0204 | 6.4776 | 7,176 |
| extrapolation | 0.01 | 86.0569 | 219.8780 | 29.4492 | 6,846 |
| extrapolation | 0.05 | 83.0555 | 218.6879 | 29.8692 | 6,786 |
| extrapolation | 0.10 | 79.8458 | 217.4677 | 30.1758 | 6,894 |
| extrapolation | 0.20 | 75.9503 | **215.8733** | **30.5545** | 7,434 |
| extrapolation | **0.40** | **74.4506** | 215.9871 | 30.4745 | 8,724 |
| extrapolation | 0.60 | 80.1037 | 220.0168 | 28.7163 | 9,138 |
| extrapolation | 0.80 | 88.6722 | 225.4993 | 24.1756 | 11,244 |
| extrapolation | 1.00 | 101.6090 | 231.3676 | 19.3799 | 12,684 |
| extrapolation | 1.50 | 130.5653 | 240.0816 | 13.1254 | 14,034 |

最佳已测点 `gamma=0.4` 相对 full v800 改善 `12.4539 FID`，即 `14.33%`。本轮按要求在确认成功信号后停止，没有继续细扫或扩大到 5K。

## 文件

| 文件 | 内容 |
|---|---|
| `summary.json` | 配置、官方实现来源、checkpoint 哈希、冻结/配对审计和完整指标 |
| `training_validation.csv` | 5K 至 50K 的 raw/EMA validation |
| `final_time_bins.csv` | 50K 的五个时间区间指标 |
| `fid1k.csv` | 12 个严格配对条件的 FID、sFID、IS、NFE、显存与样本指纹 |
| `training_validation.png` | 中间头收敛及 full-internal gap 曲线 |
| `fid1k_sweep.png` | 外推系数的 FID/IS/NFE 曲线 |
| `preview_comparison.png` | 相同 noise/labels 下的 full、internal 与三个外推点 |

以上文件可由下列命令从正式 artifact 重建并重新执行哈希/配对检查：

```bash
python experiments/summarize_imagenet100_sit_frozen_internal_v_head.py
```

本机未纳入 Git 的主要产物包括 12 份生成 NPZ，共 `2,359,299,168` bytes，以及 50K head checkpoint `4,865,577` bytes。

强头、弱头以及 `strong - weak` 的终点与同状态频率分析另见
[`../imagenet100_sit_internal_head_frequency/README.md`](../imagenet100_sit_internal_head_frequency/README.md)。
