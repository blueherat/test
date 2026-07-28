# Imagenette-64 Decoder 放大效应诊断：预注册

## 1. 问题与边界

上一轮正式实验已经确认：真实 code 从 16d 增加到 256d 时，Oracle FID 稳定改善；
但用统一预算的 latent prior 替代经验 code 后，decoded modeling gap 稳定增大。与此同时，
256d 的 held-out flow MSE、raw latent SWD 和 effective rank matching 并不差。

本轮只检验一个新的、可证伪的解释：

> prior 在 decoder 真正使用的条件空间中留下了方向性误差；冻结 decoder 对这些方向的
> 响应大于对等幅随机方向的响应，因此较小的 latent 分布误差被放大为较大的图像分布误差。

这不等同于声称“高维 prior 更难训练”，也不训练新 prior、encoder 或 decoder。

## 2. 为什么改在条件空间测量

当前 pixel decoder 先将 `z` 输入 `condition_mlp`，再把每个非零条件中心化并归一化到
固定 RMS。decoder 实际接收的是 192 维条件向量，而不是 raw `z`。因此本轮以
`decoder.condition_embedding(z)` 为正式分析空间；raw latent 指标仅作已有对照。

## 3. 固定实验设计

- 数据与模型：上一轮 Imagenette-64 的 `16/64/256d x 5 seeds` 共 15 个冻结系统。
- prior：读取上一轮每个 run 的正式 EMA checkpoint，使用相同 seed 和 100 NFE 重新采样。
- 样本数：每个 run 默认 256 个 empirical code 和 256 个 prior code。
- 匹配：在单位 RMS、零均值的 192 维条件球面上，用余弦距离进行 Hungarian 一一匹配。
- pixel noise：所有条件分支共享逐样本初始噪声。
- decoder：全程 `eval()`、fp32、冻结，50 步 Euler；禁止重训或选择 checkpoint。
- 固定球面角度：`0.15 rad`。所有干预保持条件均值为 0、RMS 为 1。

对每个 empirical 条件 `e` 构造三种等角方向：

1. `prior_direction`：沿匹配 prior 条件的球面切向方向移动。
2. `empirical_direction`：沿独立 empirical 条件匹配得到的数据切向方向移动。
3. `random_direction`：在 `e` 的球面切空间中采样固定 seed 随机方向。

`empirical_direction` 用于控制真实数据流形上的有限样本变化；`random_direction` 用于控制
普通各向同性扰动。三者输入角度完全相同。

## 4. 正式指标

### 4.1 条件分布差异

- condition SWD；
- condition mean/covariance relative error；
- Hungarian matched mean/median angular distance；
- 固定五折划分的线性与单隐藏层 MLP classifier two-sample accuracy。

classifier 只用于诊断两分布能否被区分，不参与任何模型训练或选择。

### 4.2 Decoder 固定角度响应

对三种方向分别报告：

- 最终图像 paired pixel RMS shift；
- ResNet18 feature RMS shift；
- ResNet18 feature cosine distance；
- 在 base trajectory 的 `t=0.9/0.5/0.1` 处，velocity field RMS shift。

主要方向性量为：

```text
alignment_ratio = response(prior_direction) / response(random_direction)
manifold_ratio  = response(prior_direction) / response(empirical_direction)
```

### 4.3 原始 mismatch 的有限割线响应

额外解码 Hungarian 匹配的 prior endpoint，报告：

```text
endpoint_feature_secant
  = paired ResNet feature shift / matched condition angle
```

它直接描述当前 prior 误差经过冻结 decoder 后的平均放大，但不单独作为因果证据。

## 5. 预注册判断

只有以下条件同时满足，才把 `decoder-amplified prior mismatch` 从“推断”提升为“有受控
干预支持的机制”：

1. `256d` 的 fixed-angle prior-direction feature response 大于 `16d`，至少 4/5 seed
   同向，且五 seed 均值比值不低于 `1.20`。
2. `256d` 的 feature `alignment_ratio > 1.10` 且 `manifold_ratio > 1.10`，两者均
   至少 4/5 seed 成立；即实际 prior 偏差方向不仅比等角随机方向敏感，也比普通
   empirical-to-empirical 数据方向更敏感。
3. 上述趋势在 pixel shift 或至少一个 velocity probe 上同向，不能只出现在单一 ResNet
   特征指标。
4. 一个仅使用 decoder-response 指标的 leave-one-seed-out 线性预测，预测 modeling gap
   的 RMSE 至少比名义维数基线低 10%。该指标预先固定为
   `decoder_weighted_mismatch = matched_angle_mean * fixed_angle_prior_feature_rms / 0.15`，
   不在结果出来后从多个 response 指标中择优。
5. 所有 run 的冻结 hash、采样复现、球面均值/RMS、输入角度和共享噪声检查通过。

若只满足第 1 条而不满足第 2 条，结论限定为“256d decoder 整体条件敏感度更高”，不能说
prior 误差特别对齐高敏感方向。

若第 1 条也失败，则当前 decoder 放大解释被否定；停止沿该机制设计 decoder-aware prior。

### 正式运行前修订记录

`2026-07-22` 的 `16d/seed0/count=16` 隔离 smoke 只用于检查实现，没有进入正式结果。
它显示 prior 方向和独立 empirical 方向都明显比随机切向方向敏感。因此，仅要求
`prior/random > 1.10` 不能区分“prior 特殊对齐”与“数据流形方向普遍敏感”。在任何
`count=256` 正式 run 启动前，第 2 条被加严为同时要求 `prior/empirical > 1.10`；其余
指标、阈值和流程不变。

## 6. 科学限制

- 这是 Imagenette-64 小模型，不直接外推到 RAE/ImageNet。
- 当前 decoder 是条件 flow decoder；这里的 pullback 是固定 pixel noise 下从条件球面到
  最终图像的映射，不等同于确定性 VAE decoder 的几何。
- FID 是 ResNet18 feature FID，不是 ImageNet Inception gFID。
- Hungarian 匹配提供有限样本耦合，不代表真实最优传输的唯一物理配对。
- 本轮若通过，只支持下一步做更严格的 decoder-aware prior 对照，不直接构成新方法。
