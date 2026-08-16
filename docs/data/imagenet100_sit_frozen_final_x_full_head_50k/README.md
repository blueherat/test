# SiT v800 冻结末层完整 x 预测头实验

## 结论

本实验修正了旧末层 x 头与中间层辅助头在结构和参数量上的不公平：冻结
SiT-S/2 `v800` EMA，在第 12/12 个 transformer block 的最终 hidden state 后接
一个新的完整 AdaLN `FinalLayer`，只用 clean-x 目标训练 50K step。

新头的独立预测能力明显改善，但没有产生可信的外推增益：

- 可训练参数由旧末层线性头的 `6,160` 增至 `301,840`，与 depth8 完整头相同；
- EMA clean-x MSE 为 `0.242859`，转换后 velocity MSE 为 `1.025824`；
- 辅助头单独采样 FID-1K 从旧末层小头的 `148.1405` 改善到 `124.9748`；
- 最佳扫描点 `gamma=0.08` 的 FID 为 `86.8614`，而配对 v800 基线为
  `86.9072`，只改善 `0.0458`；sFID 和 IS 没有同步改善，因此不视为可信收益；
- `gamma>=0.15` 后 FID 持续恶化，`gamma=1.5` 时达到 `144.7727`。

这说明旧末层小头确实受到了结构和参数量限制，但辅助头自身越强并不自动让
`full - auxiliary` 成为更有效的 guidance。相同参数量的 depth8 完整 x 头虽然
单独生成质量更差，却曾在 `gamma=0.15` 得到 `81.2270` 的 FID-1K。

## 实验定义

```text
x_t       = (1 - t) * noise + t * clean
v_full    = frozen v800 final velocity output
x_aux     = new FinalLayer(block_12_hidden, t, class)
v_aux     = (x_aux - x_t) / max(1 - t, 0.05)
v_gamma   = v_full + gamma * (v_full - v_aux)
```

训练时仅优化 `MSE(x_aux, clean)`。source backbone、原始 final layer 和 v 输出全部
冻结；采样时才把 x 预测转换到统一 velocity 空间。

## 配置与公平性

- source：SiT-S/2 800K checkpoint 的 EMA 权重；
- 数据：ImageNet-100 cached SD-VAE latents；
- 训练：50K step，global batch 256，LR `1e-4`，uniform time；
- 新增/可训练参数：`301,840`；source 的 `32,617,760` 个参数全部冻结；
- 采样：Dopri5、FP32/TF32、CFG=1、两张 GPU；
- 评估：1,000 samples，ADM FID/sFID/IS，sample seed 0；
- 所有条件使用完全相同的初始 noise、labels、ODE、VAE 和 reference；
- full 条件的样本 SHA256 与此前 v800 baseline 逐字节一致。

## 三种 x 头对比

| 方案 | 可训练参数 | EMA x-MSE | 转换后 v-MSE | 头单独 FID | 最佳外推 FID |
|---|---:|---:|---:|---:|---:|
| 旧末层小线性头 | 6,160 | 0.252590 | 1.913318 | 148.1405 | 87.2368 |
| depth8 完整 FinalLayer | 301,840 | 0.258922 | 1.297204 | 221.7920 | **81.2270** |
| depth12 完整 FinalLayer | 301,840 | **0.242859** | **1.025824** | **124.9748** | 86.8614 |

这里的 FID 均为 FID-1K。depth12 的 `0.0458` 数值改善远小于该规模评估的波动，
且其他指标不共同支持，因此没有扩大到 FID-5K。

## 文件

| 文件 | 内容 |
|---|---|
| `summary.json` | 配置、checkpoint 哈希、冻结/配对审计和完整结果 |
| `training_validation.csv` | 5K 至 50K 的 raw/EMA validation |
| `final_time_bins.csv` | 50K validation 的五个时间区间 |
| `fid1k.csv` | full、辅助头和 14 个外推系数的全部指标及指纹 |
| `training_validation.png` | 训练收敛和 full-auxiliary gap |
| `fid1k_sweep.png` | FID、IS 和 NFE 随外推系数变化 |
| `preview_comparison.png` | 同 noise/labels 的代表性样本对比 |
| `head_architecture_comparison.csv` | 三种 x 头的统一数值对照 |
| `head_architecture_comparison.png` | 三种 x 头的 MSE、独立 FID 和外推 FID 对照 |

## 运行与重建

训练命令：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
INTERNAL_DEPTH=12 \
PREDICTION_TARGET=clean \
OUTPUT_DIR=/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/\
sit-s-2_v800-ema_frozen-final-x-fullhead-depth12_seed0 \
MAX_STEPS=50000 \
SAVE_EVERY=10000 \
VALIDATION_EVERY=5000 \
GLOBAL_BATCH_SIZE=256 \
bash experiments/run_imagenet100_sit_frozen_internal_v_head_2gpu.sh
```

配对 FID-1K 扫描命令：

```bash
python experiments/run_imagenet100_sit_frozen_internal_v_head_fid1k.py \
  --checkpoint /home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/\
sit-s-2_v800-ema_frozen-final-x-fullhead-depth12_seed0/checkpoints/\
step_00050000.pt \
  --output-root /home/zhoushunyu/data/eqvae/imagenet_sit_flow/\
fid1k_v800_frozen_final_x_fullhead_depth12_step50000_ema \
  --gammas 0.01 0.03 0.05 0.08 0.10 0.12 0.15 0.20 0.30 0.40 0.60 0.80 1.00 1.50 \
  --sampling-cuda-visible-devices 0,1 \
  --fid-cuda-visible-devices 2
```

便携数据和图片可由本机原始 checkpoint、训练日志和采样结果重新生成：

```bash
PYTHONPATH=. python \
  experiments/summarize_imagenet100_sit_frozen_final_x_full_head.py
```

checkpoint、生成样本 NPZ 和日志保留在本机数据盘，不进入 Git。本轮排除的 16 份
生成 NPZ 合计 `3,145,732,224` bytes，50K 辅助头 checkpoint 为 `4,865,705`
bytes。
