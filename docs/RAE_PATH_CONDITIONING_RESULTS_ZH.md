# RAE layerwise path 逆条件数：探索结果

## 最重要的结论

这一轮真正发现的不是“真实 sampler 轨迹比直线好”，而是：

> time-dependent latent path 会引入子空间特异的 endpoint 逆条件数；它与 decoder
> 对不同子空间的敏感度共同决定生成风险。

前一个 clean-estimate 曲率实验对 `annealed/random/reverse` 使用了错误的线性反演
`z_t-t v_t`，已明确标记为部分无效。修正公式后，事前固定的四个 conditioning
预测全部成立；随后 oracle 实验又推翻了一个过于简单的预测，形成了更细的机制。

## 数学机制

训练路径为：

```text
z_t = (1-t)s(t) + t epsilon
v_t = epsilon - s(t) + (1-t)s'(t)
```

所以：

```text
z_t - t v_t = s(t) - t(1-t)s'(t)
```

若某个子空间的 data coefficient 是 `c(t)`，endpoint 在该观测中的系数是：

```text
k(t) = c(t) - t(1-t)c'(t)
```

速度误差反演为 endpoint 误差时会近似乘上 `1/|k(t)|`。当前 fading schedule
`c(t)=(1-t)^2` 对应：

```text
k_fade(t) = (1+2t)(1-t)^2
```

它在高噪声端趋近 0。

## 实验正确性

- synthetic teacher path 上，修正反演对 static、annealed、reverse 和
  `detail_scale=2.3` 的非正交分解均恢复真实 endpoint，误差容限 `2e-5`。
- 四个实际模型重新生成的 endpoint 相对 RMS 误差为
  `2.31e-7--3.42e-7`。
- 使用相同 seed/noise/label、held-out indices `[128,160)`、32 样本、fp32。
- 第二次 oracle 运行与第一次 conditioning 运行的 1024 条 latent 指标及共同 decoder
  proxy 逐值完全一致。

## 路径条件数证据

### Static：无额外路径病态

semantic 和 basis factor 始终为 1，两个 component error 平滑下降。step 32 的 total
relative error 为 `0.392`，step 40 为 `0.241`。

### Annealed：病态集中在 rank-16 detail

semantic factor 始终为 1，但 detail factor 在高噪声端接近 0：

| step | detail factor | semantic error | basis error |
|---:|---:|---:|---:|
| 8 | 0.0022 | 0.662 | 35.33 |
| 24 | 0.0407 | 0.514 | 3.03 |
| 32 | 0.1169 | 0.407 | 1.11 |
| 40 | 0.3370 | 0.252 | 0.42 |

### Reverse：病态作用在大部分 semantic 维度

reverse 把 fading coefficient 用在 semantic、token mean 和 basis 外的大部分通道。
因此 step 8 semantic error 达 `14.60`，step 32 仍为 `0.712`，step 40 为 `0.523`，
均显著高于 annealed。它必须到采样末段才快速确定语义。

### Random：energy matching 制造了解析过零点

random 使用 `detail_scale=2.568`，basis 的有效观测系数为：

```text
1 - scale + scale * k_fade(t)
```

它在 step 44、`t=0.443` 附近只有 `|k|=0.065`，且前后变号。对应 basis error 从
step 40 的 `2.22` 跳到 `14.48`，step 46 又回落到 `1.33`。这说明以前的
energy-matched random control 并不是一个干净的“随机子空间”对照；缩放引入了额外
的路径奇异点。

## Decoder oracle 的反常结果

最初预测 annealed 的 basis error 最大，所以替换 basis 应比替换 semantic 更有效。
该预测被明确否定。

在 step 32：

| path | corrected endpoint-feature distance | 修 semantic | 修 basis |
|---|---:|---:|---:|
| static | 0.318 | 0.196 | 0.289 |
| random | 0.329 | 0.152 | 0.321 |
| annealed | 0.349 | 0.270 | 0.334 |
| reverse | 0.397 | 0.277 | 0.351 |

除 random 的奇异点外，修 semantic 的逐样本收益在各路径/时间上以
`87.5%--100%` 胜率超过修 basis，单侧配对 Wilcoxon 多为 `p=1e-7--1e-10`。

唯一完全翻转的是 random step 44：32/32 样本均为修 basis 更有效，endpoint-feature
distance 从 `0.348` 降到 `0.186`；修 semantic 为 `0.350`，几乎无作用。

因此 decoder 风险不是只看 latent relative error：

```text
生成影响 ≈ 反演误差幅值 × decoder 方向敏感度
```

通常 decoder 对 semantic 更敏感；低秩 basis error 即使相对值较大，也可能影响较小。
但当路径系数经过零点、误差爆炸到十几倍时，低秩误差仍会成为主导。

## 与已有生成结果的关系

本轮 32 样本 endpoint projected Frechet 再次得到：

```text
static 2.454 < random 2.560 < annealed 2.797 < reverse 3.019
```

它与既有 5k FID：

```text
123.53 < 128.99 < 143.54 < 159.05
```

完全同序。结合 oracle，当前最合理的解释是：

1. static 不引入额外子空间逆病态，因此最好；
2. random 主要扰动 decoder 较不敏感的随机低秩方向，所以总体接近 static，但存在
   一个被 scale 人为制造的局部奇异点；
3. annealed 延迟的是 middle-guided、较有结构的 detail，影响高于随机方向；
4. reverse 延迟 decoder 最敏感且维度最大的 semantic 信息，因此最差。

这里只能称为“与生成排序一致的机制证据”，不能从单 seed 和 32 样本宣称完整因果。

## 值得继续的研究方向

比继续研究 endpoint 差分更有希望的方向是 **decoder-weighted path conditioning**：

1. 为每个子空间显式计算 `k_j(t)=c_j-t(1-t)c'_j`；
2. 设计 coefficient 时约束 `|k_j(t)| >= k_min`，禁止过零和高噪声端过度退化；
3. 用 decoder Jacobian/Gauss-Newton proxy 给子空间风险加权；
4. 或直接使用 endpoint/x0 prediction、按 `k_j(t)` 预条件输出与损失，避免 velocity
   误差被 `1/k_j` 放大。

这与频域 LMS 的启发有直接联系：不能只看每个频带/子空间的误差，还要看它的条件数
和下游系统增益。这里的下游系统就是 RAE decoder。

## 下一步低成本验收

先不训练大模型，离线比较几类新 coefficient：

- 原始 `(1-t)^2`；
- 带 floor 的 `epsilon+(1-epsilon)(1-t)^p`；
- 显式保证 `k(t)>=k_min` 的有理函数路径。

用现有误差和 oracle decoder sensitivity 计算：

```text
Risk = E_t sum_j sensitivity_j * error_j(t)^2 / k_j(t)^2
```

只有候选路径同时满足以下条件才训练 tiny stage-2：

- 全时间、全子空间无过零；
- decoder-weighted risk 相对当前 annealed 至少下降 30%；
- 不把 semantic information 推迟到采样末段；
- 在至少两个 power/floor 设置上结论稳定。

若离线风险没有稳定改善，就停止该方向；若通过，再做 2k-step、3 seed tiny
ImageNet gate，而不是直接投入 800 epoch。

## 结果位置

```text
~/data/eqvae/experiments/rae_path_conditioning/oracle_seed20260718_start128_n32/
```

其中：

- `conditioning_metrics.csv`：逐样本 path-aware component error；
- `decoder_proxy.csv`：corrected 与 component oracle 解码指标；
- `conditioning_errors.png`：路径反演误差曲线；
- `conditioning_decoder_proxy.png`：decoder proxy 曲线；
- `result.json`：配置、模型元数据和预测结果。
