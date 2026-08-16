# SiT v800 冻结 hidden-state 外推实验

该目录保存“冻结 SiT-S/2 `v800` EMA，直接外推第 8 层与最后一层 hidden state”的代码审计和配对 FID-1K 结果。实验没有训练任何参数；checkpoint、日志和生成样本 NPZ 留在本机数据盘，不写入 Git。

## 实验定义

SiT-S/2 包含 12 个 transformer block。一次共享 backbone forward 同时取得：

```text
h8  = block 8 后的 token hidden state
h12 = block 12 后、FinalLayer 前的 token hidden state
```

主实验先在 hidden space 外推，再使用冻结 source 的同一个条件 AdaLN `FinalLayer`：

```text
h_gamma = h12 + gamma * (h12 - h8)
v_gamma = FinalLayer(h_gamma, time + class)
```

由于 `FinalLayer` 含有 LayerNorm 和条件 AdaLN，上式不等于先读出速度再外推。因此另做 output-space 对照：

```text
v8      = FinalLayer(h8,  time + class)
v12     = FinalLayer(h12, time + class)
v_gamma = v12 + gamma * (v12 - v8)
```

## 配置与审计

- 模型：SiT-S/2，800K checkpoint 的 EMA 权重；
- source checkpoint SHA256：`b7f7d7318ee4b480fe591bc451c2aceb09efaf4111e57fd9dbefcd1bcfd88caa`；
- 中间位置：block 8/12；
- 可训练参数：`0`；
- 采样：Dopri5、FP32/TF32、CFG=1、两张 GPU；
- 指标：同一 noise、labels、ODE、VAE 和 ADM reference 下的配对 FID/sFID/IS，1,000 samples；
- 23 个正式条件的 noise/label fingerprint 完全相同；
- `gamma=0` 的样本 SHA256 与此前 v800 baseline 逐字节一致；
- 单卡采样峰值显存 `1,824 MiB`，FID 峰值显存 `6,450 MiB`；
- 等价性与回归测试共 `11 passed`。

## 结果

基线和直接中间读出：

| 条件 | FID ↓ | sFID ↓ | IS ↑ |
|---|---:|---:|---:|
| 原始 v800，`FinalLayer(h12)` | **86.9057** | **220.1693** | **29.2499** |
| 直接中间读出，`FinalLayer(h8)` | 237.5248 | 346.6142 | 2.5604 |

Hidden-space 外推：

| gamma | FID ↓ | sFID ↓ | IS ↑ |
|---:|---:|---:|---:|
| 0.005 | **87.1733** | **221.1057** | **29.2906** |
| 0.01 | 87.4206 | 222.0321 | 29.2161 |
| 0.02 | 88.0778 | 223.9555 | 28.9405 |
| 0.03 | 88.8138 | 225.9767 | 28.6246 |
| 0.05 | 90.8543 | 230.0730 | 27.6224 |
| 0.10 | 98.3437 | 241.1050 | 23.1271 |
| 0.20 | 124.5916 | 260.3322 | 16.1997 |
| 0.40 | 195.5815 | 263.5343 | 7.6510 |
| 0.60 | 240.4291 | 267.7704 | 4.3111 |
| 1.00 | 285.0117 | 292.4199 | 2.5998 |
| 1.50 | 315.2926 | 316.4499 | 2.1521 |
| 2.00 | 333.8405 | 331.5418 | 1.9850 |
| 3.00 | 355.4867 | 348.0498 | 1.8405 |

Output-space 对照：

| gamma | FID ↓ | sFID ↓ | IS ↑ |
|---:|---:|---:|---:|
| 0.001 | **86.9447** | **220.7233** | **29.2436** |
| 0.003 | 87.0467 | 221.8323 | 29.2160 |
| 0.005 | 87.0562 | 222.9632 | 29.0930 |
| 0.01 | 87.8445 | 226.0313 | 28.8233 |
| 0.02 | 90.8138 | 232.8672 | 26.7041 |
| 0.03 | 96.5930 | 240.5360 | 23.3286 |
| 0.10 | 211.8760 | 257.4981 | 6.8709 |
| 0.20 | 316.2674 | 301.1123 | 2.5933 |

所有已测正 gamma 的 FID 均差于 baseline。最小差值只有 `0.0390` FID，单独看属于 FID-1K 波动范围；但 FID、sFID 和 IS 随外推强度整体一致退化，没有出现小系数改善窗口。配对预览显示外推先增强锐度，随后产生过曝、轮廓描边和颜色饱和。

## 反向内插补充实验

在相同 `v800` EMA、block 8/12、noise、labels、ODE、VAE 和 ADM reference 下，进一步测试从最终状态向第 8 层状态内插：

```text
hidden: h_alpha = (1-alpha) * h12 + alpha * h8
        v_alpha = FinalLayer(h_alpha, time + class)

output: v_alpha = (1-alpha) * FinalLayer(h12, c)
                + alpha * FinalLayer(h8, c)
```

Hidden-space 的局部结果：

| alpha | FID ↓ | sFID ↓ | IS ↑ |
|---:|---:|---:|---:|
| 0，原始 v800 | 86.9071 | 220.1676 | 29.2484 |
| 0.005 | 86.7217 | 219.2558 | 29.4521 |
| 0.010 | 86.6016 | 218.3744 | 29.7584 |
| 0.0125 | 86.4785 | 217.9469 | 29.6950 |
| 0.0150 | 86.4709 | 217.5111 | 29.8099 |
| **0.0175** | **86.3469** | 217.0360 | 29.8342 |
| 0.0200 | 86.3830 | 216.5615 | 29.9628 |
| 0.0225 | 86.3923 | 216.1263 | 30.0910 |
| 0.0250 | 86.4112 | 215.7024 | 30.1230 |
| 0.0300 | 86.4087 | 214.8188 | 30.2335 |
| 0.0500 | 87.1078 | 211.4321 | 30.6429 |
| 0.1000 | 90.9722 | 205.0725 | 28.7862 |

最佳 FID-1K 为 `alpha=0.0175` 的 `86.3469`，相对 baseline 改善 `0.5602`。该差异较小，`0.0175–0.03` 也彼此接近，因此按要求停止细扫，没有做 FID-5K 放大验证，不将其视为显著质量提升。

Output-space 内插没有改善 FID：最小点 `alpha=0.001` 为 `86.9112`，随后随 alpha 增大而持续恶化。这说明局部正信号只出现在 FinalLayer 之前的 hidden-space 内插，而不是两个速度输出的普通线性平均。

## 方向审计

为区别“原 FinalLayer 直接读取 h8”与“训练过的 depth-8 v-head”，在 32 条从 `t=0` 开始的 v800 baseline rollout 上比较：

```text
g_raw     = v_full - source_FinalLayer(h8)
g_trained = v_full - trained_auxiliary_FinalLayer(h8)
```

| t | cos(g_raw, g_trained) | RMS(g_trained) / RMS(g_raw) |
|---:|---:|---:|
| 0.05 | -0.1463 | 0.0378 |
| 0.10 | 0.0581 | 0.0458 |
| 0.20 | -0.0329 | 0.0629 |
| 0.40 | 0.1952 | 0.1222 |
| 0.60 | 0.3167 | 0.2434 |
| 0.80 | 0.3840 | 0.3759 |
| 0.90 | 0.4092 | 0.3543 |
| 0.95 | 0.4112 | 0.2915 |

因此两个“弱分支”即使单独生成质量都很差，也不是同一个 guidance field。训练辅助头不仅改变尺度，还学习了不同的读出方向；直接 hidden 外推不能替代训练过的中间头。

## 文件

| 文件 | 内容 |
|---|---|
| `summary.json` | 配置、公式、哈希审计、23 个完整指标和方向审计 |
| `fid1k.csv` | 配对 FID/sFID/IS、NFE、显存和样本指纹 |
| `fid1k_sweep.png` | hidden/output 外推的 FID 与 IS 曲线 |
| `hidden_state_gap_audit.csv` | raw 与 trained gap 的逐时间统计 |
| `hidden_state_gap_audit.png` | gap cosine 与 RMS 比例 |
| `preview_comparison.png` | 相同 noise/labels 下的配对预览 |
| `interpolation_summary.json` | 26 个严格配对内插条件、公式、哈希和关键结果 |
| `interpolation_fid1k.csv` | hidden/output 内插的完整 FID/sFID/IS 数据 |
| `interpolation_sweep.png` | 内插全局与局部曲线 |
| `interpolation_preview.png` | baseline、最佳小幅内插和较大内插的配对预览 |

重建方向审计与便携报告：

```bash
CUDA_VISIBLE_DEVICES=0 \
python experiments/analyze_imagenet100_sit_hidden_state_gap.py

python experiments/summarize_imagenet100_sit_hidden_state_extrapolation.py

python experiments/summarize_imagenet100_sit_hidden_state_interpolation.py
```

正式 FID sweep 可由 `experiments/run_imagenet100_sit_hidden_state_extrapolation_fid1k.py` 复现。本机未纳入 Git 的主要产物包括旧外推的 24 份生成样本 NPZ，以及本次内插的 26 份配对样本 NPZ；内插样本共 `5,111,814,864` bytes。
