# RAEv2 Internal Guidance 固定 Latent 机制审计

## 1. 本轮问题

上一轮已经发现：从 `s=1` 到 `s=1.78` 的配对 endpoint 位移经过

```text
G(z) = E(clamp(D(z)))
```

后，范数约变为原来的 `1.35` 倍，方向余弦约为 `0.52`。这排除了“decoder 只是把 IG 偏移整体压小”这一简单解释。

本轮吸收外部审计建议，继续回答三个更具体的问题：

1. 不同 guidance scale 的 endpoint 是否只是沿同一方向线性移动？
2. `G` 是否像一次 roundtrip 后就基本稳定的斜 retraction？
3. 方向变化是否主要由 decoder 输出 clamp 引起？

全部实验冻结 RAEv2，不训练参数，不重新选择 checkpoint。

## 2. 对外部建议的审计

以下建议合理并已实现：

- 检查实际 scale 轨迹相对固定 endpoint 直线的偏离；
- 检查 `G(z), G^2(z), G^3(z)`，而不是只凭一次 roundtrip 推断 retraction；
- 单独报告 clamp 前后的越界比例、位移范数和方向；
- 把“endpoint 排序反转”与“局部 `full-base` 为什么有效”分成两个问题。

有一项需要修正：latent、像素和特征的维度及坐标系不同，不能直接计算它们之间的方向余弦。因此实现只在同一空间内比较方向：

- decoder pre-clamp 位移与 post-clamp 位移；
- 原 latent 位移与 re-encoded latent 位移；
- 各 observation space 中的实际轨迹位移与固定直线反事实位移。

替换 decoder checkpoint 的建议本轮没有执行，因为本机只有一个公开的 ImageNet DINOv3-L K7 decoder。没有第二个同 tokenizer、可直接替换的 decoder，强行更换会同时改变多个变量。

## 3. 实验一：scale 轨迹是否为直线

固定：

```text
control = z_1
anchor  = z_1.78
```

对每个 scale 构造直线位置：

```text
z_line(s) = z_1 + (s - 1) / (1.78 - 1) * (z_1.78 - z_1)
```

然后比较实际采样 endpoint `z_s` 与 `z_line(s)`。主指标是：

```text
||z_s - z_line(s)|| / ||z_s - z_1||
```

该指标为 `0` 才表示实际 endpoint 落在这条固定直线上。

数据为两组独立 seed，每组 5000 个相同噪声、标签配对样本。

### 3.1 Raw latent 结果

| scale | seed 20260801 | seed 20260802 |
|---:|---:|---:|
| 1.20 | 0.873 | 0.871 |
| 1.40 | 0.712 | 0.711 |
| 1.60 | 0.488 | 0.489 |
| 1.78 | 0.000 | 0.000 |
| 2.00 | 0.551 | 0.555 |
| 2.20 | 0.765 | 0.769 |

两个 seed 几乎重合。scale 轨迹在 raw latent 中已经强烈弯曲，因此改变 `s` 不能理解为沿一个固定 endpoint direction 调整步长。

### 3.2 不同 observation space

以 seed 20260801 为例：

| space | s=1.2 | s=1.4 | s=1.6 | s=2.0 | s=2.2 |
|---|---:|---:|---:|---:|---:|
| raw latent | 0.873 | 0.712 | 0.488 | 0.551 | 0.765 |
| roundtrip latent | 0.890 | 0.761 | 0.572 | 0.688 | 0.942 |
| clamped pixels | 0.891 | 0.744 | 0.531 | 0.619 | 0.858 |
| low-frequency pixels | 0.868 | 0.701 | 0.477 | 0.542 | 0.760 |
| Inception features | 0.887 | 0.760 | 0.580 | 0.695 | 0.946 |

因此 observation map 不是简单地“增加所有弯曲”。低频像素在部分尺度略微拉直轨迹，而 roundtrip latent 和 Inception feature 往往进一步改变方向。更准确的描述是各向异性的方向选择与重参数化。

## 4. 实验二：`G` 是否近似幂等

对 `real`、`s=1`、`s=1.78` 三类 latent 连续计算三次 roundtrip。核心指标为：

```text
||G^k(z) - G^(k-1)(z)|| / ||G(z) - z||
```

如果 `G` 是一次 roundtrip 后快速稳定的 retraction，`k=2` 时该比值应明显接近 `0`。

两组独立 seed 各取 1000 个样本，结果如下：

| condition | power | seed 20260801 median | seed 20260802 median |
|---|---:|---:|---:|
| control `s=1` | 2 | 0.659 | 0.659 |
| control `s=1` | 3 | 0.687 | 0.695 |
| guided `s=1.78` | 2 | 0.616 | 0.611 |
| guided `s=1.78` | 3 | 0.641 | 0.647 |
| real encoder latent | 2 | 0.938 | 0.931 |
| real encoder latent | 3 | 1.005 | 0.996 |

`G` 没有在一次 roundtrip 后快速稳定。尤其真实 encoder latent 的第二、三次变化与第一次同量级。

同一 16 样本分别用 bf16 和 fp32 复算，关键数值几乎一致。例如 control 的 power-2 中位数均约为 `0.707`，所以该现象不是 bf16 累积误差。

结论：当前证据反对把 `G` 称为近似投影或斜 retraction。它更像一个有持续漂移的 autoencoder cycle map。

## 5. 实验三：clamp 是否造成方向旋转

对同一样本的 `s=1 -> s=1.78` decoder 输出，比较 clamp 前后的像素位移。

| metric（power 1） | seed 20260801 | seed 20260802 |
|---|---:|---:|
| post/pre 位移范数 | 0.99837 | 0.99839 |
| pre/post 位移余弦 | 0.99980 | 0.99981 |
| `G` 后/原 latent 位移范数 | 1.38874 | 1.38418 |
| 原 latent/`G` 后位移余弦 | 0.51620 | 0.51944 |

control 与 guided 的平均越界像素比例约为 `0.8%–1.2%`。

clamp 对位移几乎是恒等映射，但完整 `G` 会明显放大并旋转 latent 位移。因此上一轮约 59 度的方向旋转不能归因于 clipping；主要变化发生在 decoder 与 encoder 的组合中。

当前实验仍不能把 decoder 和 encoder 的责任完全分开，因二者输出不在同一坐标空间，不能用一个跨空间余弦强行归因。

## 6. 实验四：固定 endpoint direction 是否足以产生质量改善

将 `z_line(s)` 送入同一个冻结 decoder，与实际递归采样产生的 `z_s` 比较。两条路径使用相同样本数、相同 encoder/decoder、相同官方 ImageNet FID reference。

### 6.1 两组 5k FID

表中为：

```text
FID(fixed chord) - FID(actual recursive trajectory)
```

正数表示固定直线更差。

| scale | seed 20260801 | seed 20260802 |
|---:|---:|---:|
| 1.20 | +0.457 | +0.393 |
| 1.40 | +0.654 | +0.702 |
| 1.60 | +0.013 | +0.034 |
| 1.78 | -0.003 | -0.001 |
| 2.00 | +0.135 | +0.204 |
| 2.20 | +0.360 | +0.467 |

锚点的差异接近零，验证了重新 decode 与原 artifact 一致。所有非锚点尺度上，实际递归轨迹的 FID 都优于固定 endpoint 直线，并在两个 seed 上复现。

### 6.2 C2ST 的边界

低频像素 C2ST 总体也支持实际轨迹更接近 reconstruction distribution。Inception C2ST 在高 scale 的小差异会跨 seed 换序，而 FID 结论稳定。因此不能只拿单个线性探针的高 scale 排名解释图像质量。

## 7. 当前结论

本轮可以可靠地说：

1. **guidance scale 不是固定方向上的线性外推系数。** 改变 scale 会改变整条递归采样轨迹和后续网络输入，endpoint path 在 raw latent 中已经强烈弯曲。
2. **固定 endpoint direction 不足以复现真实 scale trajectory 的生成质量。** 两组 5k FID 均显示，除定义锚点外，固定直线更差。
3. **clamp 不是 raw/roundtrip 排名反转的主要原因。** 它几乎不改变配对像素位移。
4. **`E(clamp(D(z)))` 不是近似一次投影。** 多次 cycle 仍持续产生大幅漂移，斜 retraction 假说未通过最基本的幂等检查。
5. **上一轮的各向异性旋转仍成立，但应归因于完整 autoencoder observation map，而不是简单的 decoder 压缩。**

本轮仍然不能回答：

```text
为什么每个时刻的 full-base 局部方向会让最终 FID 变好？
```

它回答的是 endpoint 排名反转和固定方向反事实，不是 local guidance direction 的来源。

## 8. 下一步建议

decoder-induced ranking reversal 这一支已经有足够机制证据，不建议继续堆更多 endpoint 距离。下一步应转向 IG 动力学，并直接利用本轮发现的“递归轨迹不可替代”。

最小实验是 guidance time-window intervention：

1. 固定同一噪声和标签，以 `s=1` 为 control；
2. 把官方 IG 时间区间切成 4–5 个不重叠窗口；
3. 每次只在一个窗口启用 `full-base`，其余时间保持 control；
4. 再测试两个窗口的组合，并与各自单窗口效果之和比较；
5. 报告 endpoint latent 位移、decoded 5k FID 和窗口间非加性。

验收标准：

- 至少一个时间窗口在两个 seed 上稳定改善 FID；
- 组合窗口的收益显著偏离简单相加，证明 state feedback，而不只是累计固定方向；
- 窗口贡献能解释官方全区间 IG 的主要收益；
- 若所有窗口效果不稳定，则停止“局部控制机制”路线，不上更复杂控制器。

这会直接回答下一层问题：IG 的好处发生在什么时间，以及它依赖单步方向还是后续状态反馈。

## 9. 结果位置

代码：

- `experiments/analyze_raev2_scale_path_geometry.py`
- `experiments/run_raev2_roundtrip_idempotence_audit.py`
- `experiments/run_raev2_fixed_latent_interpolation_audit.py`
- `experiments/analyze_raev2_fixed_latent_interpolation_fid.py`

本地实验产物：

- `~/data/eqvae/experiments/raev2_scale_path_geometry/n5000x2_v1`
- `~/data/eqvae/experiments/raev2_roundtrip_idempotence/n1000_seed20260801_v1`
- `~/data/eqvae/experiments/raev2_roundtrip_idempotence/n1000_seed20260802_v1`
- `~/data/eqvae/experiments/raev2_fixed_latent_interpolation/n5000_seed20260801_v1`
- `~/data/eqvae/experiments/raev2_fixed_latent_interpolation/n5000_seed20260802_v1`

所有大数组和图像均保存在仓库外，没有写入 git。
