# RAE clean-estimate 轨迹探索：事前预测

> **审计状态：部分无效。** 首轮运行后检查训练路径公式发现，`z_t-t v_t` 只对
> `static` 线性路径是 clean endpoint estimate。`annealed/random/reverse` 使用
> time-dependent data path，必须反演额外的路径系数。因此首轮跨模型 P1--P4 结果
> 只能作为发现异常的探索记录，不能作为 sampler 曲率结论。修正实验见
> `RAE_PATH_CONDITIONING_PREDICTIONS_ZH.md`。

## 问题

前一轮发现：不同 stage-2 模型在相同 noise/label 下产生的两个合法 endpoint 之间，
欧氏直线通常穿过 decoder 更不友好的区域。现在检验一个更贴近生成过程的解释：

> stage-2 sampler 的逐步 clean-latent 估计是否沿一条弯曲但更 decoder-compatible
> 的路径前进，而 endpoint 欧氏弦是有损捷径？

对线性 flow matching，在状态 `z_t` 和速度预测 `v(z_t,t)` 上使用：

```text
z0_hat(t) = z_t - t * v(z_t, t)
```

本实验不直接解码带噪的 `z_t`，只解码 `z0_hat(t)` 与最终生成 endpoint。

## 固定比较

- 模型：现有 `static / random / annealed / reverse` 四个 rank-16 stage-2 模型。
- 每张 GPU 固定一个模型，fp32，Euler 50-step，CFG=1。
- 使用与已保存 endpoint 完全相同的 sampling seed、initial noise、label 和 sample index。
- 主探索样本：indices `[128,160)`，每条路径 32 个。
- 时间索引：`0,8,16,24,32,40,48`，另加最终 endpoint。
- clean reference：独立 ImageNet validation clean latents `[1024,1056)`。

对每个真实估计 `q=z0_hat(t)`：

1. 在首个估计 `a` 到最终 endpoint `b` 的弦上计算逐样本投影进度 `p`；
2. 构造同进度直线点 `l=a+p(b-a)`；
3. 构造与 `q` 逐样本 RMS 完全相同的 `l_rms`。

因此 `q` 与 `l` 的差正交于 endpoint chord；`q` 与 `l_rms` 的差进一步排除了样本
整体 RMS。

## 事前预测

### P1：真实 clean-estimate 轨迹明显弯曲

在中间时间，预计

```text
median ||q-l|| / ||b-a|| >= 0.15
```

至少 3/4 模型满足。若不满足，endpoint 直线断层不能归因于 sampler 路径弯曲。

### P2：真实估计总体向最终 endpoint 前进，但未必严格单调

预计 latent chord progress 的中位数总体由 0 接近 1；允许局部回退。若大量样本长期
落在 `[0,1]` 外，说明单一 endpoint chord 本身不是合适坐标。

### P3：真实估计比同进度欧氏弦更 decoder-compatible

以 independent clean decoded features 为 reference，预计中间时间的 projected
Frechet 满足：

```text
actual < chord
```

至少 3/4 模型、超过一半中间时间成立。

### P4：RMS matching 不能消除真实轨迹优势

预计 `actual < rms_matched_chord` 的方向与 P3 一致，且至少保留 P3 平均优势的 50%。
若 P3 成立但 P4 不成立，说明主要只是半径效应，不支持更深的曲率解释。

## 判读边界

- 这是探索实验，不是论文确认实验；现有小模型 FID 较差，只有一个 sampling seed。
- projected Frechet/SWD 是 32 样本 screening proxy，不是正式 FID。
- 只有 endpoint 数值复现通过后才解释轨迹结果。
- 若 P1 成立但 P3/P4 失败，只能说路径弯曲，不能说这种弯曲有生成价值。
- 若 P3/P4 成立，下一步才使用新 sampling seed 和更大样本做确认。
