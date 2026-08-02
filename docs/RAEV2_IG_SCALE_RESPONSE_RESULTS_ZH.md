# RAEv2 Internal Guidance Scale Response 实验记录

## 1. 做了什么

本轮实验使用官方冻结的 RAEv2 ImageNet 模型，测量 internal guidance scale 改变后，
生成结果在三个空间中的分布变化：

1. 原始生成 latent `z_s` 与真实图像 encoder latent `E(x)`；
2. roundtrip latent `E(D(z_s))` 与真实重构 latent `E(D(E(x)))`；
3. 解码图像 `D(z_s)` 与真实图像、真实重构图像及官方 ImageNet FID 统计量。

实验只进行冻结模型采样和评估，没有训练模型，没有反向传播，也没有更新 encoder、
decoder 或 stage-2 参数。

### 1.1 模型与采样配置

- stage-1：RAEv2 `DINOv3-L K7` encoder/decoder；
- stage-2：官方完整 checkpoint 的 EMA 参数；
- 数据：ImageNet-1k train；
- 图像分辨率：`256 x 256`；
- sampler：官方 100-step ODE；
- internal guidance 代码约定：`base + s * (full - base)`；
- `s=1.0`：不做额外 extrapolation 的 full-head 对照；
- 精度：官方推理路径使用 `bf16` autocast，统计量转换为 `fp32/fp64`；
- 设备：4 张 GPU；
- decoder 输出以 clamped `float16` 保存，roundtrip 不经过额外 uint8 量化。

同一个 seed 内，不同 scale 使用完全相同的类别标签和初始噪声。每个 scale 的
rank-0 noise SHA-256 相同，采样、解码和 roundtrip 均使用原子 marker，支持中断恢复。

### 1.2 执行的实验

#### A. 重构 FID 基线审计

使用两个 5,000-image seed 比较：

- 原始 ImageNet 图像与官方 ImageNet FID 统计量；
- `D(E(x))` 重构图像与官方 ImageNet FID 统计量；
- `D(E(x))` 与配对原图；
- 配对原图和重构图的 Inception feature cosine。

#### B. 1,000-image 全尺度筛查

- seed：`20260801`；
- scale：`0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.78, 2.0, 2.2`；
- 每个 scale 使用相同标签和噪声；
- 输出三个空间的 AUC、均值/方差差异、随机投影 MMD/Wasserstein、FID 和 KID。

#### C. 两个 seed 的 5,000-image 验证

- seed：`20260801, 20260802`；
- 每个 seed 5,000 张图；
- scale：`1.0, 1.2, 1.4, 1.6, 1.78, 2.0, 2.2`；
- 每轮均一次运行完成，没有触发自动重试；
- 每轮均生成 26 个完整阶段 marker，manifest 状态为 `complete`。

## 2. 指标含义

- `AUC`：用 held-out 线性分类器区分真实/生成分布，越接近 `0.5` 越难区分；
- `mean_shift_rms`：均值偏移，越低越好；
- `diagonal_variance_relative_l2`：逐维方差相对误差，越低越好；
- `sketch_rbf_mmd_squared` / `sketch_sliced_wasserstein`：固定随机投影后的分布差异，
  越低越好；
- `FID` / `KID`：解码图像 Inception feature 的分布差异，越低越好。

线性分类器使用 class-disjoint train/test split：训练分类器和测试分类器使用不同的
ImageNet 类别，避免分类器只记忆同一类别中的样本差异。

## 3. 实验结果

### 3.1 重构 FID 基线

两个 5,000-image seed 的平均结果：

| 指标 | 结果 |
|---|---:|
| 原图 vs 官方 ImageNet FID 统计量 | 6.862246 |
| 重构图 vs 官方 ImageNet FID 统计量 | 6.861839 |
| 重构图 vs 配对原图 | 2.406357 |
| 配对 Inception feature cosine | 0.968993 |

这说明 5,000-image FID 本身存在明显有限样本基线：即使是真实 ImageNet 子集，相对
官方统计量的 FID 也约为 `6.86`。因此后面的 5k FID 用于同样本量、同一参考统计量下的
scale 相对比较，不能直接与论文报告的 50k FID 比较。

### 3.2 两个 5k seed 的聚合主表

下表为两个 seed 的均值。三个 AUC 都是越接近 `0.5` 越好。

| scale | 原始 latent AUC | Roundtrip AUC | 解码图像 AUC | FID to official |
|---:|---:|---:|---:|---:|
| 1.00 | **0.726795** | 0.763051 | 0.565346 | 7.668611 |
| 1.20 | 0.740299 | 0.714674 | 0.556027 | 7.324354 |
| 1.40 | 0.782249 | 0.674077 | 0.545175 | 7.186506 |
| 1.60 | 0.838152 | 0.644766 | **0.529046** | 7.151325 |
| 1.78 | 0.886640 | **0.637039** | 0.529487 | **7.147290** |
| 2.00 | 0.931649 | 0.643698 | 0.540524 | 7.168786 |
| 2.20 | 0.960460 | 0.666086 | 0.555728 | 7.300946 |

### 3.3 原始 latent 结果

原始生成 latent 与 `E(x)` 的差异随 scale 增大而增大：

- AUC：`0.726795` (`s=1.0`) -> `0.886640` (`s=1.78`)；
- mean shift：`0.039567` (`s=1.0`)；在 `s=1.2` 达到最低 `0.039396`，随后增大到
  `0.052815` (`s=1.78`)；
- `s=2.2` 的 AUC 达到 `0.960460`。

两个 seed 的原始 latent AUC 最低点都为 `s=1.0`：

| seed | s=1.0 | s=1.78 |
|---:|---:|---:|
| 20260801 | 0.716974 | 0.885272 |
| 20260802 | 0.736615 | 0.888009 |

原始 latent 的平均方差比从 `s=1.0` 的约 `0.966` 增长到 `s=1.78` 的约 `0.997`。
internal guidance 使生成 latent 的总体方差更接近真实 latent，但均值和可线性区分的
结构差异同时增大。

### 3.4 Roundtrip latent 结果

经过 `E(D(.))` 后，scale 排序与原始 latent 相反：

- AUC：`0.763051` (`s=1.0`) -> `0.637039` (`s=1.78`)；
- mean shift：`0.054706` (`s=1.0`) -> `0.030788` (`s=1.78`)；
- roundtrip AUC 和 mean shift 在两个 seed 上都由 `s=1.78` 取得最低值；
- `s>1.78` 后 AUC 和 mean shift 再次上升。

| seed | AUC at s=1.0 | AUC at s=1.78 | mean shift at s=1.0 | mean shift at s=1.78 |
|---:|---:|---:|---:|---:|
| 20260801 | 0.765750 | 0.647185 | 0.054903 | 0.030580 |
| 20260802 | 0.760351 | 0.626894 | 0.054509 | 0.030995 |

从 `s=1.0` 到 `s=1.78`，roundtrip mean shift 平均下降约 `43.7%`；AUC 超出随机水平
`0.5` 的部分平均下降约 `47.9%`。

### 3.5 解码图像结果

两个 seed 的平均 FID：

- `s=1.0`：`7.668611`；
- `s=1.6`：`7.151325`；
- `s=1.78`：`7.147290`；
- `s=2.0`：`7.168786`；
- `s=2.2`：`7.300946`。

从 `s=1.0` 到 `s=1.78`，5k FID 平均改善 `0.521321`，相对下降约 `6.8%`。
以真实重构图相对官方统计量的平均 FID `6.861839` 为有限样本基线，guidance 将
`s=1.0` 与该基线之间的额外 FID 差距缩小约 `64.6%`。

每个 seed 的 FID 最低点不同：

| seed | s=1.0 FID | 最低 FID | 对应 scale |
|---:|---:|---:|---:|
| 20260801 | 7.749292 | 7.093245 | 1.78 |
| 20260802 | 7.587930 | 7.148881 | 1.40 |

因此两 seed 平均值在 `s=1.78` 最低，但 `s=1.4-2.0` 是较平坦的低 FID 区间，当前
结果不支持把 `1.78` 解释为精确且唯一的最优点。

## 4. 得到的结果

本轮实验稳定观察到下面的尺度排序反转：

- 在原始 latent 空间中，`s=1.0` 最接近真实 `E(x)`，提高 guidance scale 会使生成
  latent 更容易与真实 latent 区分；
- 经过 decoder 再编码后，`s=1.78` 的 roundtrip latent 最接近真实重构 latent；
- 在解码图像空间中，`s=1.6-1.78` 的 AUC/FID 最好，`s=2.2` 后重新恶化；
- 因此 official internal guidance 没有使 raw endpoint latent 更接近 encoder latent
  分布，但它使经过 autoencoder 映射后的 latent 和最终图像分布更接近相应真实参考。

该结论在两个独立 5k seed 上方向一致。当前结果定位的是 `E(D(.))` 整体映射和最终
图像空间的变化，不能单独把排序反转归因于 decoder 的某一层。

## 5. 文件位置

### 5.1 仓库内代码

- `experiments/run_raev2_reconstruction_fid_audit.py`
- `experiments/run_raev2_scale_response_study.py`
- `experiments/run_raev2_scale_response_pipeline.py`
- `tests/test_run_raev2_reconstruction_fid_audit.py`
- `tests/test_run_raev2_scale_response_study.py`
- `tests/test_run_raev2_scale_response_pipeline.py`

### 5.2 本地实验结果

```text
/home/zhoushunyu/data/eqvae/experiments/raev2_ig_scale_response/
  n1000_seed20260801_v1/
  n5000_seed20260801_scales7_v1/
  n5000_seed20260802_scales7_v1/
```

每个 5k 目录包含：

- `scale_response_metrics.csv`：逐 scale 数值结果；
- `scale_response_curves.png`：单 seed 三空间曲线；
- `image_baseline_metrics.json`：真实图像和重构图像 FID 基线；
- `manifest.json`：实验配置和完整性信息；
- `pipeline_status.json`：执行状态；
- `pipeline.log`：运行日志；
- `latents/`、`roundtrip/`、`decoded/`、`inception/`：评估原始数据。
