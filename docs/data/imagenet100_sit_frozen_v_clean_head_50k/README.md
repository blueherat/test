# SiT frozen-v800 clean-head 便携数据

该目录保存“冻结 SiT-S/2 `v800` EMA，只训练一个 clean-latent 输出头 50K step”实验的轻量结果。模型 checkpoint、生成样本 NPZ、日志和 FID 临时文件仍位于本机数据盘，不写入 Git。

## 实验定义

```text
v_x     = (x_hat - z_t) / max(1 - t, 0.05)
v_gamma = v800 + gamma * (v800 - v_x)
```

模型总计 `32,617,760` 个参数，只有 clean output linear 的 `6,160` 个参数可训练。source backbone、AdaLN 和原生 velocity output 全部冻结。

## 文件

| 文件 | 内容 |
|---|---|
| `summary.json` | 模型/训练配置、checkpoint 哈希、冻结与配对审计、最终 validation 和完整 FID 行 |
| `training_validation.csv` | 5K 至 50K 的 raw/EMA clean MSE、clean-derived velocity MSE 与 frozen-v MSE |
| `final_time_bins.csv` | 50K validation 的五个时间区间结果 |
| `fid1k.csv` | 11 个严格配对条件的 FID、sFID、IS、NFE、显存峰值与指纹 |
| `training_validation.png` | clean head 训练曲线和转换后 velocity 质量 |
| `fid1k_sweep.png` | 正外推系数的 FID/IS/NFE 曲线 |
| `preview_comparison.png` | 相同 noise/labels 下 `v`、clean、`gamma=.1`、`gamma=1` 的代表性预览 |

所有 CSV/JSON 均由：

```bash
python experiments/summarize_imagenet100_sit_frozen_v_clean_head.py
```

从正式 artifact 重建。脚本会验证 source/head checkpoint SHA256、冻结协议以及 11 个采样条件的 noise/label pairing。

## 协议

- model：SiT-S/2；
- source：`v800` EMA；
- dataset：ImageNet-100 cached SD-VAE latents；
- clean-head training：50K step，global batch 256，LR `1e-4`，Uniform time；
- sampling：Dopri5、FP32/TF32、CFG=1、两张 GPU；
- metric：ADM FID/sFID/IS；
- screening：1,000 samples，sample seed 0。

## 头部结果

| 条件 | `gamma` | FID ↓ | IS ↑ | total NFE |
|---|---:|---:|---:|---:|
| frozen velocity | 0 | **86.9043** | **29.2454** | 6,714 |
| clean head only | - | 148.1405 | 16.0080 | 19,848 |
| extrapolation | 0.01 | 87.2368 | 29.0710 | 6,624 |
| extrapolation | 0.10 | 93.0170 | 26.1721 | 8,028 |
| extrapolation | 0.50 | 157.6062 | 9.6428 | 9,246 |
| extrapolation | 1.00 | 271.5317 | 3.2697 | 11,346 |

所有正 `gamma` 均比 frozen velocity baseline 更差，且 FID 随扫描系数严格单调恶化，因此没有继续做 FID-5K。

## Git 排除项

本机 11 份 `samples_unguided_n1000.npz` 合计 `2,162,690,904` bytes。它们与 head checkpoint、完整日志均未复制进该目录，也不会进入 Git。
