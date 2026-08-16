# SiT v800 冻结中间 epsilon 预测头实验

该目录保存“冻结 SiT-S/2 `v800` EMA，在第 8/12 个 block 后接一个独立 `FinalLayer`，仅用 epsilon 目标训练该中间头 50K step”的结果。checkpoint、训练日志和生成样本 NPZ 留在本机数据盘，不写入 Git。

## 实验定义

中间头结构与 Internal Guidance 官方 SiT 实现一致：从第 8 个 block 的 token 接一个独立的 AdaLN `FinalLayer`。本实验只训练新增头，不更新原模型：

```text
x_t              = (1 - t) * epsilon + t * clean
v_full           = frozen v800 final output
epsilon_internal = trained FinalLayer(block_8_features, t, class)
v_internal       = (x_t - epsilon_internal) / max(t, 0.05)
v_gamma          = v_full + gamma * (v_full - v_internal)
```

训练目标是原生 `MSE(epsilon_internal, epsilon)`；采样前才把 epsilon 预测转换到统一 velocity 空间。分母下限 `0.05` 与 x 预测使用对称的端点稳定处理，避免噪声端的 `1/t` 数值发散。

## 配置与审计

- 模型：SiT-S/2，共 12 个 transformer blocks；
- source：800K checkpoint 的 EMA 权重；
- 中间位置：block 8；
- 新增/可训练参数：`301,840`，source 的 `32,617,760` 个参数全部冻结；
- 数据：ImageNet-100 cached SD-VAE latents；
- 训练：50K step，global batch 256，LR `1e-4`，uniform time，native epsilon-MSE；
- 采样：Dopri5、FP32/TF32、CFG=1、两张 GPU；
- 指标：同一 noise、labels、ODE、VAE 和 ADM reference 下的配对 FID/sFID/IS，1,000 samples；
- 16 个条件的 noise/label fingerprint 各自唯一且共享；
- full 条件的样本 NPZ 与此前 v-head、x-head 的 v800 baseline SHA256 逐字节一致；
- 采样峰值显存 `1,802 MiB/卡`，FID 峰值显存 `6,450 MiB`。

## 结果

50K validation：

| 输出 | native epsilon-MSE ↓ | 转换后 v-MSE ↓ |
|---|---:|---:|
| frozen full v | - | 0.793684 |
| internal epsilon，raw | 0.325422 | 1.153981 |
| internal epsilon，EMA | 0.325526 | 1.160964 |

配对 FID-1K：

| 条件 | gamma | FID ↓ | sFID ↓ | IS ↑ | total NFE |
|---|---:|---:|---:|---:|---:|
| full v800 | 0 | 86.9031 | 220.1730 | 29.2498 | 6,714 |
| internal epsilon only | - | 235.6070 | 287.5244 | 4.0684 | 9,378 |
| extrapolation | 0.01 | 85.8869 | 220.1177 | 29.4522 | 6,720 |
| extrapolation | 0.03 | 84.4909 | 220.0949 | 29.3298 | 6,678 |
| extrapolation | 0.05 | 83.1465 | 220.0824 | 29.2196 | 6,804 |
| extrapolation | 0.08 | 81.4593 | **220.0369** | 29.2384 | 6,960 |
| extrapolation | 0.10 | 80.9707 | 220.0402 | **29.5780** | 7,092 |
| extrapolation | 0.12 | 80.3933 | 220.1004 | 29.4399 | 7,818 |
| extrapolation | 0.15 | 79.8259 | 220.4873 | 29.3960 | 8,100 |
| extrapolation | **0.18** | **79.7120** | 220.8103 | 28.8675 | 8,082 |
| extrapolation | 0.20 | 79.9210 | 220.9963 | 28.3304 | 7,902 |
| extrapolation | 0.40 | 91.6443 | 224.8605 | 23.3229 | 10,044 |
| extrapolation | 0.60 | 112.5502 | 229.1871 | 16.2311 | 10,854 |
| extrapolation | 0.80 | 134.3236 | 235.8209 | 12.7731 | 12,780 |
| extrapolation | 1.00 | 147.9232 | 243.0408 | 10.1532 | 14,172 |
| extrapolation | 1.50 | 168.9997 | 257.5313 | 8.5174 | 16,290 |

最佳已测 FID 点 `gamma=0.18` 相对 full v800 改善 `7.1910`，即 `8.27%`。`gamma=0.10` 同时改善 FID 和 IS；`gamma>=0.4` 后三项生成指标明显退化。

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
  --prediction-target epsilon
```

本机未纳入 Git 的主要产物包括 16 份生成 NPZ，共 `3,145,732,224` bytes，以及 50K head checkpoint `4,865,705` bytes。
