# RAE Decoder 噪声坐标几何审计

## 研究问题

本实验检验一个理论上很自然的解释：RAE 的 decoder 在 Stage-1 中接触的是 raw encoder latent 上的各向同性噪声，但 Stage-2 生成模型工作在逐坐标标准化后的 latent 中。两种坐标中的球形方向不同，这个几何错配是否是生成 latent 解码失败的主要原因？

设 raw latent 为 `z`，Stage-2 latent 为

`u = (z - mu) / s`。

官方 Stage-1 代码在标准化之前注入 `delta_z = sigma * epsilon`，而 Stage-2 中的球形扰动对应 `delta_z = s * delta_u`。当 `s` 高度各向异性时，两者覆盖的是不同方向。

## 实验设计

在同一张 clean ImageNet-1K validation 图、同一个 `sigma` 和同一个 Gaussian `epsilon` 上构造三个 paired 条件：

1. `raw_sphere`：Stage-1 官方 raw 坐标球形方向。
2. `stage2_sphere_raw_rms_matched`：Stage-2 `u` 坐标球形方向，并缩放到与条件 1 相同的 raw RMS。这是最关键的公平能量对照。
3. `stage2_sphere_u_rms_matched`：Stage-2 `u` 坐标球形方向，并缩放到与条件 1 相同的 normalized RMS。这是相同 Stage-2 距离的对照。

实验不训练任何参数，冻结 encoder 和 decoder，全程 fp32，关闭 TF32。clean hidden reference 使用 256 张未进入既有训练流的 ImageNet train 图；最终指标使用 512 张 ImageNet validation 图。4 张 GPU 只做样本分片。

官方实现使用 `sigma ~ Uniform(0, 0.8)`。本实验把每个高维 Gaussian 方向精确归一到指定 RMS，以消除有限样本范数波动；在 768 x 16 x 16 维下，这与官方未归一化 Gaussian 的 RMS 差异极小。

## 主要结果

| 条件 | raw delta RMS | normalized delta RMS | image delta RMS | cycle relative RMS | hidden sensitivity |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw sphere | 0.4033 | 0.6487 | 0.0761 | 0.6030 | 4.5304 |
| Stage-2 sphere, raw RMS matched | 0.4033 | 0.4581 | 0.0698 | 0.5439 | 3.8256 |
| Stage-2 sphere, u RMS matched | 0.5711 | 0.6487 | 0.0848 | 0.6130 | 3.4681 |

与 raw sphere 做逐样本 paired 比较：

| 对照 | image delta 比率 | cycle error 比率 | hidden sensitivity 比率 |
| --- | ---: | ---: | ---: |
| Stage-2 sphere, raw RMS matched | 0.881 `[0.871, 0.891]` | 0.909 `[0.905, 0.913]` | 0.874 `[0.866, 0.882]` |
| Stage-2 sphere, u RMS matched | 1.092 `[1.082, 1.102]` | 1.015 `[1.014, 1.017]` | 0.784 `[0.779, 0.790]` |

括号内为 paired bootstrap 95% CI。

这个结果在 `sigma` 的四个区间中没有反转。相同 raw RMS 下，Stage-2 方向的 image delta 比率依次为 `0.719, 0.880, 0.949, 0.972`。

## 结论

原假设被否定。Stage-2 标准化坐标中的球形方向并不比官方 raw 球形方向更危险。在相同 decoder 输入能量下，它反而稳定地更安全。

原因也具有明确的几何解释：

- raw 球形噪声进入 `u` 坐标后变成 `delta_u = delta_z / s`，会放大低方差坐标。
- Stage-2 球形噪声映回 raw 坐标后变成 `delta_z = s * delta_u`，更多能量落在高方差坐标。
- 当前结果说明 decoder 对低方差坐标的扰动更敏感。官方 Stage-1 noising 更像一种偏保守的鲁棒性正则，而不是遗漏了 Stage-2 的危险方向。

更强的反证来自 decoder 内部轨迹。三种各向同性噪声在 D2 的 hidden RMS z-score 分别为 `-2.557, -1.262, -1.713`；此前真实 reverse 生成 latent 在同一 D2 位置达到 `+6.615`。两者不仅幅度不同，偏离方向也相反。在 D14/D21/D28 上也存在明显轨迹差异。因此，真实生成失败不是单纯由 raw/normalized 球形噪声的坐标错配造成的。

## 新发现

decoder 对扰动的 hidden deviation sensitivity 从 D0 的约 `0.9` 增长到 D28 的约 `17`，但后层 hidden RMS 的边际 z-score 又回到 clean 附近。这意味着 decoder 可以在保持每层总体 RMS 正常的同时，持续放大样本级特征方向差异。

这解释了一个此前的断层：层级匹配、CKA 或总体激活幅度可以看起来正常，但生成图像仍然很差。稳定的边际统计不等于稳定的逐样本传输。

三种噪声几何之间的 block gain 差异主要发生在 D0-D3，之后曲线几乎平行。真实生成 latent 却在 D2 出现巨大正偏，进一步说明主异常更可能来自结构化生成误差，例如非零均值偏置、低秩 channel 方向或 token 相关误差，而不是各向同性协方差选择。

## 下一步约束

下一项最有解释力的诊断应保持相同 RMS，但把扰动分解为：

- 全局 channel mean bias；
- PCA 高方差与低方差子空间；
- token-common 与 token-residual 分量；
- 使用真实生成残差估计的低秩方向。

目标不是再比较一种球形噪声，而是找出哪一种结构化方向能复现 reverse latent 的 `D2 +6.6 sigma` 早期异常。若低秩或 token-common 扰动能够复现该轨迹，就能把“生成 latent 离分布”进一步定位到可训练、可正则化的具体子空间。

## 复现

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  experiments/run_rae_decoder_noise_geometry.py \
  --calibration-count 256 \
  --test-count 512 \
  --batch-size 2 \
  --run-name dinov2_cal256_test512_tau08_seed20260718
```

完整表格和图片位于：

`~/data/eqvae/experiments/rae_decoder_noise_geometry/dinov2_cal256_test512_tau08_seed20260718/`
