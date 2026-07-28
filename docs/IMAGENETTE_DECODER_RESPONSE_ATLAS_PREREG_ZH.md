# Imagenette-64 Frozen Decoder Response Atlas 预注册

## 问题

现有 `16/64/256d x 5 seeds` 结果显示，普通 latent/condition 指标不足以解释
decoder 后的 modeling gap。本实验在不训练任何参数的前提下检验：冻结 decoder
内部的条件响应分布差异，能否在未见 seed 和未见 latent 维数上预测该 gap。

只有这一步通过，才允许训练 decoder-aware prior。

## 固定资产

- 数据：`/data/shared/imagenette2-320`。
- run：`16/64/256d x seeds 0..4`，共 15 个正式 prior checkpoint。
- empirical latent：每个 run 的 train latent cache，用固定索引抽取两组互斥的
  256 样本；第一组与 prior 比较，第二组估计同样样本量的 real-real floor。
- prior latent：正式 EMA prior，使用与正式评估相同的 Gaussian seed 规则。
- 每个 run 使用 256 个 empirical/prior latent。
- pixel initial noise 在所有容量和 seed 间固定。
- decoder rollout：50 Euler steps，记录 `t=0.9/0.5/0.1`。
- encoder、decoder、prior 全部 `eval()`、fp32、冻结。

## Response 定义

在各自真实 decoder rollout 的状态 `s_t` 上记录：

```text
raw_l(s_t, t, z) = h_l(s_t, t, z)
condition_l(s_t, t, z) = h_l(s_t, t, z) - h_l(s_t, t, 0)
```

第二项使用同一个 `s_t` 和 `t`，只替换 latent condition，因此隔离 decoder
在当前状态对 latent 的直接响应。记录层为：

```text
condition, down0, down1, down2, middle, up2, up1, up0, velocity
```

空间激活先 adaptive-average-pool 到 `4x4`，velocity pool 到 `8x8`，再用固定
Gaussian projection 投影到最多 128 维。projection seed 在运行前固定，不按结果
调整。

## 分布指标

对 empirical 与 prior response 计算：

- normalized mean error；
- relative covariance Frobenius error；
- normalized Fréchet/Bures distance；
- normalized sliced Wasserstein distance；
- grouped held-out linear C2ST AUC；
- empirical/prior effective rank。

第二组独立 empirical response 与第一组保持相同数量、相同 initial pixel noise，
用于计算 real-vs-real 统计下限。

## Shuffled 控制

生成时 latent 与 pixel noise 本来独立，直接 shuffle 不改变 latent 边际分布，不能
要求 distribution metric 变差。因此 shuffle 只作为 paired forward-path 控制：

```text
s_t = (1-t)x + t epsilon
```

在同一个 validation image state 上比较匹配 `E(x)`、batch-roll shuffled `E(x)`
和 null latent 的 velocity MSE。要求 shuffled latent 的 MSE 明显高于 matched，
用于证明 decoder 确实读取样本级 condition，并验证 condition 接线正确。

## Held-out 预测

每个 layer/representation 的指标先对三个 probe time 取平均，再分别执行：

- leave-one-seed-out，训练 12 个 run，预测该 seed 的 3 个容量；
- leave-one-dimension-out，训练两个容量的 10 个 run，预测剩余容量的 5 个 run。

训练器仅为单变量 ridge linear regression。评价聚合 held-out prediction 与正式
`modeling_gap` 的 Spearman。`latent_dim` 作为 nominal-capacity baseline 同时报告。

## 主门槛

阶段一通过必须同时满足：

1. 15 个 run 完整，checkpoint hash、有限值和 response shape 审计通过。
2. primary representation 为 `condition`，primary metric 为三个时间平均的
   `normalized_frechet`，不根据结果替换。
3. 至少两个相邻 decoder 层同时达到 LOSO Spearman `>= 0.60` 和 LODO Spearman
   `>= 0.60`。
4. 该相邻层对的 empirical-prior normalized Fréchet 平均至少为 real-real floor
   的 `1.5x`。
5. real-real linear C2ST AUC 与 `0.5` 的平均绝对偏差 `<= 0.10`。
6. `t=0.9` 和 `t=0.5` 分别至少 `12/15` run 的 shuffled velocity MSE 高于
   matched，且平均 ratio `>= 1.05`。`t=0.1` 只报告，因为低噪声 state 已包含
   大部分图像信息，decoder 此时可以合理地降低 latent 依赖。

raw response、SWD、covariance error、其他层或单个时间点不能替代主门槛，只用于
解释主结果。

## 决策

- 全部门槛通过：进入无训练 moment repair。
- primary gate 失败：先做两项实现复核，包括 forward trace 与原 decoder 输出逐位
  一致、toy 条件模型上 response metric 可检出构造差异。
- 复核后仍失败：冻结 decoder response 不是可信训练靶标，停止 A3-A5 训练，不用
  解冻 decoder 挽救假设。

## Smoke 后、正式运行前修订记录

`256d seed0, count=32, 10-step` smoke 在任何正式 15-run atlas 产生前完成。它发现
原拆半 real-real floor 的样本量不匹配，并观察到 shuffled/matched velocity MSE
在 `t=0.9/0.5/0.1` 为约 `10.8x/5.2x/1.02x`。因此在查看正式目标指标前做出上述
两项修订。smoke 不进入正式结果，也不能用于选择 layer 或 response metric。

## Formal failure 后的统计功效复核

正式 256-sample atlas 完成后主门槛失败。随后固定检查 `256d seed0` 的 1024-sample
版本，发现 prior/real-floor ratio 在 up1/up2 从约 `0.81/0.92` 上升到
`1.21/1.16`，说明 256 样本会淹没一部分微弱差异，但结果仍低于 `1.5x` 门槛。

因此授权一个全 15-run 的 1024-sample power audit。它保持 projection、probe
times、primary representation、metric、相邻层规则、held-out protocol 和全部阈值
不变。该 audit 只判断 formal failure 是否由功效不足造成，不允许重新选择指标。
