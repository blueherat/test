# SiT v800 强头、弱头及差值的频率分析

该目录整理冻结 `v800` SiT-S/2 与其第 8/12 个 block 后训练 50K step 的
velocity 弱头频率实验。checkpoint、完整分辨率预览、采样 NPZ 和模型权重均留在
本机数据盘，不写入 Git。

## 比较对象

- 强头：冻结的 SiT-S/2 `v800` EMA 最终 velocity 输出；
- 弱头：同一冻结 backbone 的 block 8 hidden state，经独立 `FinalLayer` 得到的
  velocity 输出；
- 差值：统一定义为 `strong - weak`，即现有外推实验使用的 guidance 方向；
- decoder：两者均使用同一 `stabilityai/sd-vae-ft-mse`。

实验分成两个互补视角。

1. **独立终点 rollout**：复用相同初始噪声、类别、采样器和 seed 的
   `full/internal` 各 1,000 张图，比较最终输出。
2. **相同状态诊断**：先用强头生成 baseline trajectory，在同一个
   `(z_t,t)` 上同时查询强、弱头，再统一换算为
   `x0_hat = z_t + (1-t) * v_hat` 并解码。共 128 个配对样本，
   `t={0.05,0.1,0.2,0.4,0.6,0.8,0.95}`。该诊断将单点 head bias 与
   独立 rollout 漂移分开。

## 频率协议

- 对 BT.709 luminance 做分析；
- 每张图先去空间均值；
- 使用单位 RMS 的二维 Hann window，window 后再次去均值；
- 对正交归一化二维 FFT 的功率做径向统计；
- 径向范围为 `0-0.5 cycles/pixel`；
- low、mid、high 分别为 `[0,0.125)`、`[0.125,0.25)`、
  `[0.25,0.5]`；
- 终点强弱差采用配对 bootstrap 2,000 次给出 95% CI。

## 独立终点结果

| 指标 | 强头 | 弱头 | 弱/强 |
|---|---:|---:|---:|
| low energy fraction | 0.92133 | 0.94928 | 1.0303 |
| mid energy fraction | 0.05032 | 0.03037 | 0.6035 |
| high energy fraction | 0.02835 | 0.02035 | 0.7179 |
| spectral centroid | 0.03867 | 0.03036 | 0.7849 |
| gradient RMS | 0.10797 | 0.09936 | 0.9202 |
| RMS contrast | 0.42410 | 0.44584 | 1.0513 |

弱头终点的频谱质心低约 `21.5%`，high-band fraction 低约 `28.2%`，
gradient RMS 低约 `8.0%`；同时其 very-low-frequency 对比更强，使总体
RMS contrast 高约 `5.1%`。所以弱头不是简单地把整张图统一衰减，而是把更多
能量集中到很低频。完整径向曲线还显示最接近 Nyquist 的少量频率重新上扬，
因此它也不是理想低通滤波器。

终点 `strong - weak` 本身的频带构成为：

| low | mid | high | spectral centroid |
|---:|---:|---:|---:|
| 0.67815 | 0.16117 | 0.16068 | 0.11041 |

差值的质心约为强头的 `2.85` 倍、弱头的 `3.64` 倍，说明它相对两路输出
显著富集边缘和细节；但其 `67.8%` 能量仍在 low band，不能把 guidance
方向概括为纯高通或纯锐化。

## 相同状态结果

同一 `(z_t,t)` 上，弱/强频率指标随时间如下：

| t | high fraction | spectral centroid | gradient RMS |
|---:|---:|---:|---:|
| 0.20 | 0.8636 | 0.9578 | 0.9444 |
| 0.40 | 0.8632 | 0.8703 | 0.7908 |
| 0.60 | 0.7692 | 0.8402 | 0.8528 |
| 0.80 | 0.9079 | 0.9369 | 0.9579 |
| 0.95 | 1.0008 | 0.9912 | 0.9955 |

中段 `t=0.4-0.8` 的低频偏置最清楚；靠近终点时两个 decoded clean
prediction 会重新接近，部分原因是统一换算中的 `(1-t)` 使 velocity 差值的
绝对作用自然收缩。

`strong - weak` 的频率组成则随时间明显迁移：

| t | low | mid | high | centroid | RMS magnitude |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.7804 | 0.0125 | 0.2072 | 0.1003 | 0.0431 |
| 0.20 | 0.8264 | 0.0160 | 0.1576 | 0.0810 | 0.0585 |
| 0.40 | 0.7474 | 0.0564 | 0.1963 | 0.1107 | 0.0684 |
| 0.60 | 0.4881 | 0.1563 | 0.3556 | 0.1896 | 0.0770 |
| 0.80 | 0.3083 | 0.2391 | 0.4526 | 0.2314 | 0.0612 |
| 0.95 | 0.2579 | 0.2399 | 0.5022 | 0.2469 | 0.0123 |

因此这条内部 guidance direction 不是固定频率模板：早期主要包含低频、布局
与大尺度差异，中后期逐步转向边缘和细节；`t=0.95` 虽有最高的 high-band
fraction，但绝对 RMS 已降到 `0.0123`，不能只看比例误判其作用强度。

这些统计与既有 `gamma=0.4` 优于过大外推系数的现象相容，但本实验只建立
频率描述，不能单独证明频率迁移是 FID 改善的因果来源。

## 文件

| 文件 | 内容 |
|---|---|
| `summary.json` | 完整配置、配对审计、终点与同状态统计 |
| `terminal_metrics.csv` | 终点强弱配对指标及 bootstrap CI |
| `terminal_difference_metrics.csv` | 终点 `strong - weak` 指标 |
| `terminal_radial_spectrum.csv` | 终点完整径向功率谱 |
| `same_state_metrics.csv` | 各时刻强弱配对指标 |
| `same_state_difference_metrics.csv` | 各时刻差值指标 |
| `same_state_radial_spectrum.csv` | 各时刻完整径向功率谱 |
| `terminal_frequency_comparison.png` | 终点 PSD、能量比例与弱/强谱比 |
| `same_state_frequency_metrics.png` | 同状态强弱频率指标时序 |
| `same_state_spectral_ratio_heatmap.png` | 同状态弱/强谱比时频图 |
| `same_state_difference_metrics.png` | 差值频带组成与绝对幅度时序 |
| `same_state_difference_spectrum_heatmap.png` | 差值径向谱时频图 |
| `terminal_paired_difference_preview.png` | 终点配对输出与差值的紧凑预览 |
| `same_state_clean_prediction_preview.png` | 同状态 clean prediction 与差值预览 |

完整分析可由以下入口重建：

```bash
python experiments/analyze_imagenet100_sit_internal_head_frequency.py \
  --probe-samples 128 \
  --device cuda:3
```

