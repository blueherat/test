# ImageNet-100 SiT 多尺度 Guidance 实验报告

## 一、结论摘要

本轮完整跑完了 4 个内部 velocity head 的训练、1 份 latent 时频 atlas，以及 99 个完全配对的 FID-1K 条件。最重要的结果是：

1. **静态频谱统计不能单独区分 useful 与 failed guidance。** 多组成功/失败 gap 的平均频谱质心和高频占比几乎重合，因此“某种频谱长相天然有用”的强假设不成立。
2. **gap 的终端作用具有明确的时间和尺度结构。** 对成功的 `depth8_v` 与 `external_v500`，最有效的单元都集中在 `mid × high`；失败的 `depth12_x` 在所有 18 个 time-band 条件中都几乎没有作用。
3. **时间变化的内部深度是本轮最强正结果。** 固定 `gamma=0.4` 时，`depth 4 -> 8 -> 10` 将配对 FID-1K 从 `84.97` 降到 `68.31`，优于任一静态深度、固定 depth-8 gap 和反向 `10 -> 8 -> 4` 调度。
4. **结果不是简单由 gap RMS 决定。** RMS 匹配后，正向调度仍比反向调度好 `7.54` FID；频谱 router 也仍比 anti-router 好 `6.88` FID。
5. **“弱模型只是强模型按频带延迟几步”的版本未通过。** 拟合只得到 low/mid 延迟 1 个 Euler step、high 延迟 0 step；synthetic delayed self-guidance 相对 Euler baseline 最多只改善 `0.94` FID-1K，且 RMS 匹配的大系数明显恶化。
6. **raw unresolved-computation proxy 失败。** 把未训练的中间 token 直接送进最终输出层会造成严重 representation mismatch；即使 RMS 匹配，FID-1K 仍为 `112.59` 至 `115.27`。这否定的是当前 proxy，不足以否定“剩余计算”概念本身。

这些都是**单训练 seed、单采样 seed、每条件 1,000 张的筛查结果**。用户明确取消了本轮 FID-5K，所以当前不能把最优 FID 当成正式方法结论。

## 二、研究对象

strong model 是 ImageNet-100 上训练到 800K step 的 `SiT-S/2` velocity model，推理使用 EMA：

\[
S(z_t,t).
\]

对内部第 \(\ell\) 层，冻结 strong backbone，只训练轻量 velocity readout：

\[
W_\ell(z_t,t).
\]

定义内部差值：

\[
g_\ell(z_t,t)=S(z_t,t)-W_\ell(z_t,t),
\]

并在闭环 ODE 的每次函数求值中使用：

\[
\dot z_t=S(z_t,t)+\gamma g_\ell(z_t,t).
\]

本轮新训练 `depth={4,6,10,12}` 四个 velocity head；每个训练 50K step，global batch 为 256，学习率为 `1e-4`，训练 seed 为 0。已有的 depth-8 `v/x/epsilon` heads、depth-12 x head、最终线性 x head和外部 `v500` weak model用于 atlas 与对照。

### 时间与频带

采样方向为 \(t=0\) 噪声到 \(t=1\) 数据。时间用平滑 partition 划成：

- early：约 \([0,1/3]\)；
- mid：约 \([1/3,2/3]\)；
- late：约 \([2/3,1]\)；
- 边界使用宽度 `0.04` 的 smoothstep 过渡，并保持 partition of unity。

频率在 **SD-VAE latent 的空间网格**上逐 channel 做 2D FFT：

- low：`[0, 0.125)` cycles / latent pixel；
- mid：`[0.125, 0.25)`；
- high：`[0.25, sqrt(0.5)]`。

这里的 low/mid/high 不是解码后像素纹理频率，不能直接解释成图像中的低频轮廓或高频毛发。

## 三、数据可靠性审计

| 检查项 | 结果 |
|---|---:|
| pipeline stages | `104 / 104 complete` |
| 评估条件 | `99` |
| 唯一条件名 | `99` |
| 每条件样本数 | `1,000` |
| noise fingerprint 组数 | `1` |
| label fingerprint 组数 | `1` |
| 覆盖类别数 | `100 / 100` |
| 指标缺失或非有限值 | `0` |
| 残留 sample NPZ | `0` |
| 采样峰值 reserved memory | `1.172 GiB` |

所有 99 个条件使用完全相同的 noise 与 class label fingerprint：

```text
noise: ab8419c7fdfd5b15dacbf4d37a3d567158e4332f25fd94580d3df73bac87e2c2
label: 7c3ae6894e7ebab5c9b6524606f03b6a56b38dccbe472ff40edde26e48654fe6
```

Dopri5 与固定 Euler 使用各自的 baseline。固定 Euler 的 `num_output_points` 没有被误当成 NFE；这里真正运行的是 100-step fixed Euler，总计 12,500 次 batch-level model evaluation。

## 四、主结果

### 4.1 方法级比较

下表的 gain 都相对**相同积分器**的配对 baseline。Dopri5 baseline FID-1K 为 `84.9687`，100-step Euler baseline 为 `85.8222`。

| 条件 | FID-1K ↓ | gain ↑ | sFID ↓ | IS ↑ | total NFE |
|---|---:|---:|---:|---:|---:|
| Dopri5 baseline | 84.9687 | 0.0000 | 222.2481 | 29.5239 | 6,676 |
| depth 4→8→10, native | **68.3064** | **16.6623** | 213.4950 | 33.4044 | 8,662 |
| depth 4→8→10, RMS | 69.5739 | 15.3948 | 214.7147 | 32.6905 | 8,734 |
| spectral router, RMS | 70.3149 | 14.6539 | 216.1114 | 31.7279 | 14,374 |
| full external v800-v500 | 72.0925 | 12.8762 | 219.2522 | **34.7202** | 10,018 |
| static depth 6 | 72.5084 | 12.4603 | 218.3738 | 30.2897 | 8,572 |
| full/static depth 8 | 73.3260 | 11.6427 | 217.7727 | 30.2591 | 8,458 |
| depth 10→8→4, native | 76.8734 | 8.0953 | 218.8061 | 29.0566 | 9,148 |
| spectral anti-router, RMS | 77.1969 | 7.7718 | **211.1041** | 30.4849 | 12,190 |
| full depth-12 x gap | 84.7856 | 0.1831 | 222.6282 | 28.8415 | 7,132 |
| Euler baseline | 85.8222 | 0.0000 | 223.1606 | 28.8335 | 12,500 |
| Euler + depth-8 gap | 73.6076 | 12.2145 | 217.4459 | 30.9876 | 12,500 |
| spectral delay, native, γ=0.8 | 84.8825 | 0.9397 | 218.5488 | 29.2566 | 12,500 |
| raw-depth proxy, best RMS | 112.5927 | -27.6239 | 251.5274 | 19.0687 | 6,166 |

`depth 4→8→10` 的 NFE 比 baseline 高 `29.7%`，采样 wall time 高约 `38.6%`。spectral router 的 NFE 高 `115.3%`，因此它虽然有效，但当前计算效率明显差于固定调度。

指标并非完全同序：FID 与 IS 更偏好正向 depth schedule / external AG，而本表中 sFID 最低的是 anti-router。当前不能宣称某个条件在所有指标上占优。

### 4.2 静态深度与时间调度

静态 head 的 FID-1K：

| depth | 4 | 6 | 8 | 10 | 12 |
|---:|---:|---:|---:|---:|---:|
| FID-1K | 73.0107 | **72.5084** | 73.3184 | 79.5412 | 83.9748 |

静态最优是 depth 6，但正向 `4→8→10` 仍比它低 `4.20` FID。正反调度的因果对照为：

| amplitude | 4→8→10 | 10→8→4 | 差值 |
|---|---:|---:|---:|
| native | **68.3064** | 76.8734 | 8.5670 |
| RMS-matched | **69.5739** | 77.1170 | 7.5431 |

因此，当前正信号不是“选择一个更好的固定 head”，也不能只用不同 head 的 norm 解释。它支持的是：**同一 strong model 的 useful internal discrepancy 随采样时间变化，匹配顺序比固定 readout 更重要。**

这仍不能直接证明 depth 就等于空间尺度，也不能证明 `4→8→10` 是唯一正确 schedule。

### 4.3 成功/失败 gap 的 causal utility map

每个 time-band cell 都执行一次完整闭环采样：

\[
\dot z=S(z,t)+\gamma\mathbf 1_{t\in I}P_b g(z,t).
\]

下面列出 native amplitude 相对 Dopri5 baseline 的 FID-1K gain。

#### depth8_v

| time \ band | low | mid | high |
|---|---:|---:|---:|
| early | 2.69 | 1.39 | 2.34 |
| mid | 1.08 | 2.28 | **8.53** |
| late | 0.22 | 1.42 | **6.65** |

#### external_v500

| time \ band | low | mid | high |
|---|---:|---:|---:|
| early | 2.42 | 1.71 | 1.87 |
| mid | 0.71 | 1.37 | **6.59** |
| late | 0.25 | 0.45 | 1.52 |

#### failed depth12_x

| time \ band | low | mid | high |
|---|---:|---:|---:|
| early | 0.06 | 0.01 | 0.03 |
| mid | 0.03 | 0.06 | 0.12 |
| late | 0.07 | 0.09 | -0.09 |

两个成功 gap 都把最大 utility 放在 `mid × high`，而 failed gap 在所有 cell 中都近似无效。equal-action control 仍保留 `mid × high` 为最大 cell，但优势缩小：

- depth8_v：`8.53 -> 4.33`；
- external_v500：`6.59 -> 2.95`。

这说明 native high-band 的较大幅度贡献了一部分效果，但**幅度不是全部原因**。

结果并不是一个干净的“early-low、late-high”对角线。更准确的描述是：early 阶段多个 band 都有弱到中等作用，主要 utility 集中在 mid/late 的 high latent band。

### 4.4 频带时序

对真实 gap 只保留 `early-low + mid-mid + late-high`，并和逆序 `early-high + mid-mid + late-low` 比较：

| provider / amplitude | coarse→fine gain | fine→coarse gain |
|---|---:|---:|
| depth8_v / native | **10.34** | 4.51 |
| depth8_v / equal action | **8.17** | 4.40 |
| external_v500 / native | **5.67** | 4.00 |
| external_v500 / equal action | **4.83** | 4.42 |
| depth12_x / native | -0.05 | 0.10 |

时序差异在 depth8_v 上最强，在 external_v500 上较弱，failed depth12_x 没有可解释信号。这个结果支持“终端效用依赖时间顺序”，但还不足以声称 AG/IG 共享一个普适的 coarse-to-fine law。

## 五、为什么静态频谱分类失败

atlas 中存在两组几乎一一对应的反例：

| gap | 先验标签 | 平均 centroid | 平均 high fraction |
|---|---|---:|---:|
| depth8_v | useful | 0.3515 | 0.7212 |
| final_linear_x | failed | 0.3477 | 0.7220 |
| depth8_x | useful | 0.3117 | 0.6339 |
| depth12_x | failed | 0.3166 | 0.6453 |
| depth8_epsilon | useful | 0.3797 | 0.7937 |
| raw_final_h8 | failed | 0.3802 | 0.8022 |

因此，无论只看 RMS、频谱质心还是 low/mid/high 能量比例，都找不到一个简单阈值把 useful 与 failed 分开。

这不与 causal map 冲突：

- atlas 问的是“这个 gap 长什么样”；
- causal intervention 问的是“把它在特定时间、特定频带放回闭环 dynamics 后会发生什么”。

当前证据支持后者，否定前者的简单版本。换句话说，频率更适合作为 controller 的测量量，而不是 utility 的静态判别器。

## 六、四个 idea 的实验判定

### Idea 1：Time-varying depth

**通过 FID-1K 筛查。** 正向 `4→8→10` 是 99 个条件中 FID 最低的条件，且 native/RMS 两种幅度下都稳定优于反向调度和所有静态深度。

### Idea 2：Spectral routing

**通过 FID-1K 筛查，但计算开销较高。** router 在 native 与 RMS 两种设置下都优于 anti-router；不过它需要同时计算多个内部 readout，并使 adaptive solver NFE 明显增加。

### Idea 3：Spectrally delayed self-guidance

**当前实现不通过。** band delay fit 没有发现强而分离的多尺度延迟；最优条件相对 Euler baseline 的 gain 仅 `0.94`，远弱于真实 depth-8 gap 的 `12.21`。RMS 匹配在 `gamma=0.4/0.8` 时显著恶化。

### Idea 4：Raw unresolved computation

**当前 proxy 不通过，概念未被干净检验。** 原 SiT final layer 没有被训练来读取 depth-4/8/10 的 raw token。灾难性结果首先证明了 representation mismatch，不能直接归因于“剩余计算差值无用”。

## 七、证据边界

1. 所有结果都是 FID-1K screen，没有 FID-5K、bootstrap CI 或 sampling multi-seed。
2. 新训练的内部 heads 只有一个训练 seed；strong model 也只有一个 checkpoint family。
3. 99 个条件共享 noise/label，减少了横向随机差异，但 FID 本身仍是有限样本分布估计量。
4. 不同方法的 NFE 和额外 head/model forward 不同，当前表不是等 FLOPs 比较。
5. time-band intervention 是有限幅度、闭环、非线性的；9 个 cell 的 gain 不能相加成 full-gap gain。
6. latent FFT 描述 SD-VAE latent 网格，不应直接外推到像素频谱或感知纹理。
7. preview 只用于同 noise/label 的定性核对，不参与方法排序。

## 八、文件位置

便携数据与图：

```text
docs/data/imagenet100_sit_multiscale_guidance_study
```

完整本地实验产物：

```text
/home/zhoushunyu/data/eqvae/imagenet_sit_flow/multiscale_guidance_study_v1
```

Git 包不包含 checkpoint、生成样本 NPZ、99 份独立 preview 或训练日志。
