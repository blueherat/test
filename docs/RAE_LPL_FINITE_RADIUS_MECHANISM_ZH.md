# RAE 上 LPL 的有限半径机制实验

## 结论

现有结果明确否定了一个原先看起来很自然的解释：

> LPL 通过把 stage-2 的预测误差转移到 decoder 局部 Jacobian 的低敏感方向，
> 从而改善生成质量。

实验看到的恰好相反。LPL 误差方向在 clean latent 附近通常比纯 Flow
误差方向更敏感，但当扰动增大到 stage-2 实际产生的误差半径时，LPL
方向又稳定变得更好。

更准确的当前结论是：

> LPL 学到的不是 clean latent 附近的局部低风险方向，而是实际预测误差
> 尺度上的有限距离 decoder 对齐。它尤其能减少深层 decoder feature
> 方差的塌缩，并在较大离流形误差下维持更好的归一化特征匹配。

这是一项稳定的实验发现，但“feature 方差保持是否直接造成 FID 改善”
仍需训练消融才能成为因果机制。

## 实验设计

比较四个严格配对训练 seed 的两个分支：

- 纯 Flow；
- Flow + strict LPL。

每个 seed 内的两个分支来自相同源 checkpoint、相同训练数据流和相同
初始化。评估使用：

- ImageNet-1K validation 的冻结 DINOv2 RAE latent；
- 32 张未参与训练或 LPL 权重校准的图像；
- `noise/signal = 0.5, 1, 2, 3`；
- 四个训练 seed；
- 共 `32 x 4 x 4 = 512` 个配对观测；
- 全程 fp32，关闭 TF32；
- 每个配对观测使用相同的 latent、类别、噪声和时间。

设两个分支的 clean-latent 预测误差为：

```text
e_flow = z0_hat_flow - z0
e_lpl  = z0_hat_lpl  - z0
```

为了排除误差大小的影响，把两个方向缩放到共同半径：

```text
r = min(RMS(e_flow), RMS(e_lpl))
```

然后沿两个方向扫描：

```text
z(alpha) = z0 + alpha * r * unit_RMS(e)
```

其中 `alpha = 0.04, 0.16, 0.32, 0.64, 1.0`。`alpha=1` 是同范数的
真实误差尺度。

另外使用 `0.1%` clean-latent RMS 的对称有限差分估计局部 JVP，并加入：

- 同范数高斯随机方向；
- 打乱 channel 和 token 位置的 LPL 误差方向。

## 核心结果

### 1. 原先的局部度量机制被反证

在 32 图、四 seed 的实验中：

| 指标 | LPL / Flow | LPL 更好的观测比例 |
|---|---:|---:|
| latent MSE | 1.0092 | 0% |
| 1% clean RMS 的 raw 局部放大 | 1.1217 | 15.4% |
| 对称有限差分 raw JVP 放大 | 1.1364 | 15.2% |
| 1% clean RMS 的 strict 局部放大 | 1.0614 | 15.8% |
| 对称有限差分 strict 局部放大 | 1.0750 | 14.5% |
| 真实半径 matched strict LPL | 0.8768 | 100% |

LPL 的普通 latent MSE 略高，局部 decoder 放大率也更高，但在真实半径
上的 strict decoder loss 反而低 `12.3%`，且 `512/512` 个配对观测全部
同向。

因此，LPL 的收益不能解释成“局部 decoder metric 下误差更小”。

### 2. 局部二次近似不足以解释真实误差

实际 stage-2 误差 RMS 是 clean latent RMS 的约 `18%--41%`，中位数约
`28%`。它并不是一个足够小的 Taylor 扰动。

局部二次风险对真实 raw decoder error 的 Spearman 为 `0.799`，普通
latent MSE 为 `0.802`；局部近似没有提供更强解释。对 strict loss 的
Spearman 为 `0.822`，能做风险排序，但仍不能解释为什么 LPL 分支更好。

### 3. 存在稳定的尺度反转

沿同范数误差方向扫描得到：

| 共同真实半径比例 | raw hidden MSE | clean 方差归一化 | 对称归一化 | strict / prediction 方差归一化 |
|---:|---:|---:|---:|---:|
| 0.04 | 1.1106 | 1.0563 | 1.0547 | 1.0531 |
| 0.16 | 1.1031 | 1.0516 | 1.0450 | 1.0383 |
| 0.32 | 1.0806 | 1.0363 | 1.0226 | 1.0087 |
| 0.64 | 1.0302 | 1.0070 | 0.9772 | 0.9461 |
| 1.00 | 0.9869 | 0.9766 | 0.9285 | 0.8768 |

表中都为 `LPL / Flow`，小于 1 才表示 LPL 更好。

512 条轨迹在真实半径时全部是 LPL 更好；其中约 `84%` 在最小扫描半径
`4%` 时仍是 LPL 更差，随后在扫描区间内发生可观察的反转。首次观察到
LPL 更好的半径中位数是共同真实误差半径的 `64%`。这说明局部 JVP 与
实际训练点之间存在真实的非线性断层。

### 4. feature 方差保持解释了主要差异，但不是全部

在真实同范数误差半径上，decoder feature 的几何平均方差相对 clean
feature 为：

```text
Flow: 0.8486
LPL : 0.9322
```

Flow 的预测会使 decoder 内部 feature 方差明显收缩；LPL 仍有收缩，
但更接近 clean feature 的尺度。

三种归一化给出：

```text
clean 方差归一化:      LPL 改善约 2.3%
对称方差归一化:       LPL 改善约 7.2%
prediction 方差归一化: LPL 改善约 12.3%
```

因此，strict LPL 的收益有相当一部分来自更好地保持 prediction feature
方差，但不是单纯利用分母“作弊”：

- LPL 的 prediction 方差仍低于 clean，而不是无界增大；
- 使用固定 clean 方差后，LPL 仍然更好；
- centered channel cosine 也从 Flow 的约 `0.795` 提升到 LPL 的约
  `0.804`。

当前最合理的描述是：LPL 同时改善了有限半径下的 feature 尺度保持和
feature 内容匹配。

### 5. 断层主要出现在 decoder 深层

在局部尺度下，LPL / Flow 的 raw 放大率随 decoder 深度约为：

```text
layer 1: 0.938
layer 2: 0.968
layer 3: 1.036
layer 4: 1.089
layer 5: 1.126
```

前两层的 LPL 方向已经稍好，深层却明显更差。到真实误差半径时，五层
全部变为约 `0.87--0.88`。所以尺度反转不是所有层统一缩放，而主要来自
深层 decoder 的非线性响应。

### 6. 随机方向不是合适的 stage-2 误差对照

同范数随机方向和打乱方向的局部放大只有 LPL 方向的约 `0.59` 和
`0.52`。这说明 stage-2 的误差集中在 decoder 敏感、数据相关的子空间，
并非各向同性噪声。

因此，“把误差旋转成随机低敏感方向”未必是可行目标；那可能同时丢失
stage-2 必须修正的语义或空间信息。

## 数据质量核验

- 幅度扫描 `alpha=1` 的 512 行与独立 matched 实验逐行一致；
- strict loss 最大绝对差为 `0`；
- raw hidden loss 最大差为 `7.6e-6`，属于 fp32 舍入；
- mask-free prediction normalization 与 strict LPL 的最大相对差低于
  `2.4e-7`，说明 outlier mask 不是反转来源；
- 两份主结果表均无 NaN 或 Inf；
- 四个训练 seed 的曲线方向一致。

结果位于：

```text
~/data/eqvae/experiments/rae_lpl_error_geometry/paired_4seed_n32_v1/
~/data/eqvae/experiments/rae_lpl_amplitude_sweep/paired_4seed_n32_v3_norm_decomp/
```

幅度曲线图：

```text
~/data/eqvae/experiments/rae_lpl_amplitude_sweep/
  paired_4seed_n32_v3_norm_decomp/mechanism_curve.png
```

## 当前能说和不能说的结论

可以说：

1. RAE 的 stage-2 误差处在 decoder 的有限距离、非线性区域。
2. LPL 的 FID 改善不是通过降低 clean latent 附近的局部敏感度实现。
3. LPL 在真实误差尺度显著改善 decoder feature 对齐。
4. 保持深层 decoder feature 方差是该改善的重要组成部分。

还不能说：

1. feature 方差保持本身就是 FID 改善的唯一原因；
2. 当前现象已在其他 RAE encoder、decoder 或更大 DiT 上成立；
3. 一个局部低秩 `J^T J` 近似仍能保留 LPL 的生成收益；
4. 当前机制已经足以构成独立论文贡献。

## 下一步：最小因果训练消融

暂时停止原计划中的 local-Jacobian DMP。它试图近似的数据结构正是本次
实验反证的部分。

下一项实验应在现有严格 LPL 训练框架中只替换 feature loss：

1. `strict`：现有 prediction 方差归一化，作为正对照。
2. `target-norm`：只用 clean feature 方差归一化，保留有限距离 feature
   匹配，移除 prediction 方差分母。
3. `symmetric-norm`：使用 clean/prediction 方差平均值。
4. `variance-only`：只匹配各层每个 channel 的 log variance。

所有分支应：

- 从相同 checkpoint 和 RNG 状态开始；
- 使用相同 train 数据流；
- 在相同 256 图 calibration set 上把初始加权贡献校准到 flow loss 的
  25%；
- 同时报告初始梯度范数，避免只匹配 loss 数值；
- 先跑 500 update、3 seed、固定 5k 采样；
- 通过后才跑 2000 update 和四 seed。

因果判据：

- 若 `target-norm` 保留至少 70% 的 strict LPL FID 收益，主要机制是
  有限距离 feature 匹配，而不是 prediction 方差分母。
- 若 `variance-only` 保留至少 50% 的收益，并同步修复 feature 方差
  收缩，则 feature-scale preservation 是强因果机制。
- 若只有 `strict` 有效，说明 prediction-conditioned normalization 与
  非线性 decoder response 的联合作用不可拆。
- 若这些 proxy 改善但 FID 不改善，则当前发现仍只是 decoder feature
  空间里的描述性现象，不能升级成生成机制。

如果消融通过，后续方法不应是 clean latent 附近的静态局部 metric，
而应是依赖时间和误差半径的 finite-radius secant risk surrogate。
