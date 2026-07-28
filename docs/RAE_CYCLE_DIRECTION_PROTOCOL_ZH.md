# RAE Cycle-Direction 因果干预协议

## 问题

现有四条生成路径的 `cycle residual` 和 decoder local sensitivity 与 5k FID 排序一致，
但这仍可能只是相关性。本实验不训练模型，只检验逐样本闭环方向

```text
r(z) = E(clamp(D(z), 0, 1)) - z
```

是否能因果地减弱 generated latent 的 decoder 异常。

这里不把 `E(D(z))` 称为严格流形投影。clean latent 的 cycle residual 本来就不为零，
因此实验只判断该方向是否具有样本特异的修正作用。

## 数据与划分

- 复用 Phase-0 四条路径各 256 个 fp32 endpoint。
- 四条路径共享初始 noise、ImageNet label 和 sample index。
- generated calibration：索引 `[0, 128)`，只用于选择一个全局 `alpha`。
- generated test：索引 `[128, 256)`，只使用一次。
- clean hidden reference：256 张 ImageNet train calibration latent，四卡分片累计。
- clean guardrail：此前 closure 未使用的 ImageNet validation logical indices `[256, 384)`。
- frozen RAE-DINOv2 encoder 和官方 ViT-XL decoder；fp32，关闭 TF32。

## 校准

只测试自身 cycle 方向，`alpha in {0.025, 0.05, 0.10, 0.25}`。

候选范围在正式运行前由 4-path x 8-sample 工程 smoke 修正。该 smoke 的 generated
indices `[0, 8)` 完全包含在正式 calibration `[0, 128)` 中，不接触 held-out test。
原候选最小值 `0.25` 对 generated latent 已对应约 `27%-32%` 的相对 RMS 步长，四条
路径均无法通过预先固定的 `0.98` feature-cosine guardrail；因此加入更小步长是量纲
修正，不改变主指标、方向对照或通过门槛。`0.25` 仍保留为大步对照。

每个 alpha 先在每条路径内计算 median cycle-error ratio 与 Inception feature cosine。
只有四条路径的 median feature cosine 都不低于 `0.98` 才合格；在合格值中选择
四条路径 median cycle ratio 均值最小的 alpha。并列时选择较小 alpha。

校准阶段不计算 shuffled、random 或 opposite 的测试结论。

## Held-out 因果对照

在固定 alpha 后，对 generated test 一次性比较：

- `own`：该样本自己的 `r(z)`；
- `shuffled`：另一测试样本的 residual，并匹配当前样本 residual RMS；
- `random`：Gaussian 方向，并匹配当前样本 residual RMS；
- `opposite`：`-r(z)`。

所有方向在乘 alpha 前具有完全相同的逐样本 normalized-latent RMS。实验保存完整 29 层
hidden response、hidden deviation、cycle error、图像变化、clipping 和 Inception feature cosine。

## 层索引纠正

旧 Phase-0 0B 的四个 hidden 指标实际来自 `output.hidden_states[2,4,6,8]`。
其中 reverse 的 `+6.615 sigma` 是 decoder hidden state 2，不是后续叙述误称的 D7。

本实验事先固定 D2 为主异常层，同时保存 D0-D28。不能在看到 held-out 结果后把主层
改成效果最好的层。

## 通过门槛

必须全部满足：

1. reverse D2 median z-score 的绝对值至少降低 50%；
2. reverse 相对 clean baseline 的 cycle excess 至少降低 25%；
3. 四条路径至少三条 cycle error 改善；
4. own 的四路径平均 cycle ratio 至少比 shuffled/random 中较好的一个低 `0.05`；
5. opposite 在 reverse 上加重 cycle error 或 D2 异常；
6. 四条 generated path 的 own median Inception cosine 均不低于 `0.98`；
7. clean own-direction guardrail 的 median Inception cosine 不低于 `0.98`。

通过后才允许生成独立 5k latent 做 FID/KID/FDD。未通过时不能通过重新选 alpha、层、
方向归一化或样本子集挽救结论。
