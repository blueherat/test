# Imagenette Decoder-Aware Prior 单 Seed 门槛结果

## 结论

在当前 Imagenette-64 两阶段小模型中，把 frozen stochastic decoder 的逐样本
中间响应加入 latent prior 训练，没有优于相同 continuation 的纯 flow 对照。

该结果没有通过进入 decoder adapter 或多 seed 训练的门槛。它只适用于当前
“latent flow prior + conditional pixel-flow decoder”系统，不否定 LPL 在确定性
AE decoder 上的 ICLR 2025 结果。

## 为什么做这个实验

之前已确认：

- 真实 latent 与 learned-prior latent 的 decoded distribution 存在稳定差距；
- decoder 对 latent 扰动的响应不均匀；
- 固定时刻 decoder response 的边际分布差异不能预测最终 modeling gap。

因此本实验没有复活已失败的边际分布匹配，而只测试更窄的问题：

> 在同一 pixel state 和同一 pixel time 下，让 prior 的 predicted-clean latent
> 与真实 latent 产生相同的 frozen decoder 中间响应，是否能改善最终生成？

## 实现

- 源 checkpoint：`d256_seed0_p0`，已训练 20k steps。
- 两分支从完全相同的 EMA prior 权重开始。
- continuation：2k steps，batch 512，fp32。
- `flow`：只使用原 rectified-flow loss。
- `decoder_lpl`：`flow + 0.1 * paired decoder feature loss`。
- LPL 子批量：8；训练 bank：1024 张 Imagenette train 图像。
- prior clean-estimate time：`[0, 0.75]`。
- pixel decoder probe time：`[0.5, 1.0]`。
- decoder layers：`middle/up2/up1/up0`。
- validation bank 只来自 Imagenette val，不与训练 bank 混用。
- 最终质量：1024 张 val、100-step latent ODE、50-step pixel ODE。
- 两分支共享 prior 随机流、采样噪声和 pixel rollout 噪声。
- encoder/decoder SHA256 在训练前后完全一致。

实现入口：

```text
experiments/imagenette_decoder_aware_prior.py
```

## 结果

| 指标 | flow continuation | decoder LPL | LPL - flow |
| --- | ---: | ---: | ---: |
| held-out flow loss | 0.442680 | 0.442694 | +0.000014 |
| held-out decoder LPL | 0.165554 | 0.165312 | -0.000242 |
| end-to-end feature FID | 138.067 | 138.207 | +0.140 |
| Oracle feature FID | 117.044 | 117.044 | 0 |
| modeling gap | 21.022 | 21.163 | +0.140 |
| latent SWD | 0.101093 | 0.101161 | +0.000068 |
| generated effective rank | 19.470 | 19.497 | +0.027 |
| predicted class TV | 0.4932 | 0.4961 | +0.0029 |
| continuation wall time | 36.5 s | 84.9 s | 2.32x |

LPL 确实轻微降低了自己对应的 held-out feature loss，但改善只有约 `0.15%`，
没有转化为 decoded FID、latent distribution 或类别覆盖收益。1024 样本下的
`0.140` FID 差异不足以声称 LPL 有害，但足以说明这里没有可见正信号。

## 正确性检查

1. 精确 velocity 下，`state - t * velocity` 能在 `2e-6` 内恢复 clean latent。
2. 相同 latent condition 的 paired decoder feature loss 精确为零。
3. latent perturbation 后 loss 为正，梯度能回到 condition/prior。
4. frozen decoder 所有参数的 `.grad` 均为 `None`。
5. 真实 checkpoint smoke 覆盖训练、held-out、latent sampling 和 pixel rollout。
6. 新旧相关测试共 `10 passed`。
7. `git diff --check` 通过。

## 与 LPL 原论文的边界

LPL（ICLR 2025）对确定性 AE decoder 比较 `z0` 与 predicted `zhat0` 的多层
decoder features。当前系统的第二阶段 decoder 本身是条件 pixel flow，不存在一次
前向即可得到的确定性 `D(z)`。本实验比较的是相同 pixel path state 下的条件响应，
应称为 `LPL-style path-conditioned loss`，不能称为对原论文的直接复现。

## 研究决定

- 不因该结果启动 decoder adapter。
- 不扩成五 seed；单 seed 正向门槛未通过。
- 不调大权重追逐小幅 FID 波动。
- 保留“decoder 信息进入 prior”作为已覆盖强基线，但当前小系统不支持它是主要
  瓶颈的判断。

若以后回到确定性 RAE/VAE decoder，应首先严格复现原版 LPL；那是另一个实验，
不能用本结果替代。
