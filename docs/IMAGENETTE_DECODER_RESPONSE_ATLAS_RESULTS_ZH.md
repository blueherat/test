# Imagenette-64 Frozen Decoder Response Atlas：正式负结果

## 最终结论

预注册假设被可靠推翻：

> 冻结 decoder 在若干固定时间、固定层上的边际 response distribution discrepancy，
> 不能预测 `16/64/256d x 5 seeds` 的最终 decoded modeling gap，因此当前不能把它
> 作为 decoder-aware prior 的训练目标。

这不是“decoder 没有读取 latent”。paired control 显示 decoder 在高、中噪声强烈
依赖样本级 latent。失败的是更具体的假设：固定时间的边际 response 差异能够代表
整条生成轨迹累计后的输出分布差距。

根据预注册停止规则，不进入 moment repair，不运行 A3-A5，不通过解冻 decoder
挽救结果。

## 实验范围

- 正式 grid：`16/64/256d x seeds 0..4`，共 15 个已训练 prior。
- 正式 atlas：每个 run 使用 256 empirical、256 independent empirical control、
  256 prior latents。
- 统计功效复核：同一 15-run grid 扩到每组 1024 样本。
- decoder：正式 frozen EMA，50-step Euler rollout，fp32。
- probe times：`0.9/0.5/0.1`。
- layers：`condition/down0/down1/down2/middle/up2/up1/up0/velocity`。
- primary：`condition` response、三时间平均 normalized Fréchet/Bures。
- held-out：leave-one-seed-out 与 leave-one-dimension-out。

正式数据位于：

```text
$HOME/data/eqvae/imagenette_latent_prior_tradeoff/decoder_response_atlas_summary
```

1024-sample power audit 位于：

```text
$HOME/data/eqvae/imagenette_decoder_response_atlas_power_audit/
  decoder_response_atlas_power_summary
```

## 正式 256-sample 主结果

没有任何相邻层同时通过两个 held-out protocol：

| layer | LOSO Spearman | LODO Spearman |
| --- | ---: | ---: |
| down0 | -0.557 | -0.721 |
| down1 | -0.779 | -0.682 |
| down2 | 0.200 | -0.743 |
| middle | -0.236 | -0.718 |
| up2 | 0.696 | 0.054 |
| up1 | 0.707 | 0.032 |
| up0 | 0.207 | -0.804 |
| velocity | 0.661 | 0.046 |

up2/up1/velocity 在 leave-seed-out 中看似有一定相关，但完全不能外推到未见 latent
维数。它们主要复述容量内 seed 差异，而不是解释跨容量 modeling gap。

## 1024-sample 功效复核

四倍样本量降低了 real-real floor，确实检出一些微弱差异，但没有挽救预测：

| layer | LOSO Spearman | LODO Spearman |
| --- | ---: | ---: |
| down0 | 0.225 | -0.479 |
| down1 | 0.193 | -0.671 |
| down2 | 0.425 | -0.439 |
| middle | -0.239 | -0.468 |
| up2 | 0.457 | -0.907 |
| up1 | 0.568 | -0.682 |
| up0 | 0.246 | -0.425 |
| velocity | 0.443 | -0.475 |

所有 LODO 相关均为负。最强反例是 up2：256d 的最终 modeling gap 最大，但 response
discrepancy 反而比 64d 小，得到 `-0.907` 的 held-out Spearman。

按容量平均，condition response 的 prior/real-floor ratio 为：

| latent | down0 | middle | up2 | up1 | velocity |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16d | 0.73 | 0.99 | 0.96 | 0.92 | 0.95 |
| 64d | 1.39 | 1.34 | 1.21 | 1.23 | 1.18 |
| 256d | 0.90 | 1.03 | 1.14 | 1.14 | 1.05 |

也就是说，64d 的固定时刻 response mismatch 更明显，但它的正式 modeling gap 约
`9.92`，远低于 256d 的约 `22.42`。这复现并加强了之前的反转：64d 在普通
condition space 更容易被分类器区分，而 256d 在最终 decoded image space 更差。

## Decoder 确实使用 latent

paired forward-path control 在全部 15 个 run 上通过：

| time | mean shuffled / matched MSE | shuffled 更差的 run |
| --- | ---: | ---: |
| 0.9 | 6.867x | 15/15 |
| 0.5 | 3.489x | 15/15 |
| 0.1 | 1.020x | 15/15 |

高、中噪声时错误 latent 明显破坏 velocity prediction。低噪声差异自然缩小，因为
pixel state 已经携带大部分图像信息。因此负结果不能解释为 decoder 忽略 condition
或 condition 接线错误。

## 正确性复核

反结果出现后完成以下审计：

1. `decoder_forward_trace` 与原 decoder forward 在 toy 输入上逐位一致，`atol=0`。
2. 零 latent 的 condition contribution 严格为零。
3. 15/15 frozen decoder SHA256 与正式 summary 一致。
4. 所有 810 个正式 response rows 与 45 个 paired rows 均有限。
5. 构造 covariance shift 的 toy 能被 normalized Fréchet、covariance error 与 SWD
   正确检出。
6. atlas Fréchet 与仓库独立实现的差为 `1.51e-15`。
7. 随机正交坐标变换前后的 Fréchet 差为 `1.76e-15`，identity 为精确零。
8. real-real C2ST AUC 对 `0.5` 的平均绝对偏差在 256/1024 样本下分别为
   `0.0358/0.0154`，统计下限合理。
9. projection seed 从 `48271` 改成 `48272` 后，seed0 的 64d/256d Fréchet ratio
   在 down2/middle/up0/up1/up2/velocity 仍为 `1.22/1.32/1.31/1.20/1.06/1.13`。
10. 新增 partial-grid 汇总边界测试；最终 response-atlas 测试为 `7 passed`。

以上证据排除了 trace 错位、度量公式错误、统计下限错误、256 样本功效不足和单个
随机投影导致的主要假阴性。

## 机制解释

固定时间 response atlas 只观察边际分布：

```text
distribution of R_l(s_t, t, z) at each t
```

最终图像却由整条状态相关轨迹决定：

```text
s_1 -> s_0.98 -> ... -> s_0
```

即使每个固定时间的低阶 response statistics 接近，以下结构仍可能不同：

- response 与当前 state 的联合依赖；
- 同一样本跨时间的误差相关；
- 小误差沿非线性轨迹的累积方向；
- 类别内部 mode、纹理与空间协方差的组合；
- decoder 最终输出特征中的高阶或条件相关多样性。

64d 的误差更“显眼”，所以静态 C2ST 和 response discrepancy 更大，但 decoder
可能容忍它。256d 的误差在单时刻边际上更隐蔽，却在跨时间、state-conditioned
联合结构中累积成更大的最终 covariance/diversity gap。

## 对研究计划的决定

当前提出的 A4：

```text
batch-level fixed-time decoder-response distribution matching
```

很可能优先惩罚 64d，而不是 modeling gap 最大的 256d，因此不是被现有证据支持的
训练目标。LPL-style A3 本身已有工作，且本实验没有给出它能解决当前分布反转的
新机制依据。

按预注册标准：

- moment repair：不授权；
- A0-A5 prior continuation：不授权；
- 五 seed decoder-aware training：不授权；
- decoder adapter 或 full joint training：不授权。

保留下来的可靠发现是：

> 两阶段接口错误不能由静态 latent discrepancy 或固定时刻 decoder-response
> discrepancy 概括。真正造成生成差距的候选是 state-conditioned、跨时间的联合
> transport mismatch。

这一发现可以记录为负机制结果，但当前目标要求在核心训练假设被确认推翻后停止，
所以不在本轮继续发明新的 pathwise loss。
