# Teacher MSE 改善但 rollout 变差：机制结论

## 最终判断

目前证据足以确认以下机制：

> 固定频率加权损失优化的是 teacher marginal 上的逐点条件速度误差，但生成质量取决于各时间、各方向对边缘分布输运的因果杠杆。有限容量模型因此学到另一种近似：它在若干低杠杆方向上降低 MSE，却损伤高杠杆的边缘漂移；早期偏差把状态推离 teacher 分布后，partial vector field 在中段对 off-path states 的泛化进一步恶化，误差随 ODE 组合累积。

这不是“频带能量少了一点”或“Euler 步数不够”可以完整解释的现象。更精确的表述是：

1. **高噪声阶段存在 pointwise MSE 与 marginal transport 的目标错配。**
2. **中噪声阶段存在 teacher-forcing / off-path generalization gap。**
3. **两者组合才形成最终生成退化。**

原始的 FxLMS 式解释“频谱加权只是预条件、会更快到达同一有限模型解”已被否定。固定、正定、样本无关的输出权重只在无限函数类的总体最优解层面不改变 `E[v | z_t,t]`；有限网络、共享参数和有限训练会得到不同投影。

## 为什么 MSE 不足以描述输运

对第 `b` 个 DCT 频带，边缘二阶矩满足：

```text
d E[||z_b||^2] / dt = 2 E[<z_b, v_b(z,t)>].
```

频带 velocity MSE 测量 `E[||v_hat_b-v_b||^2]`，但不直接约束状态与速度的交叉项 `E[<z_b,v_hat_b>]`，更不约束模型离开 teacher marginal 后的响应。因此即使 MSE、cosine、prediction slope 都改善，边缘分布仍可能被错误输运。

## 证据链

### 1. MNIST 低成本复现

配置为 8,192 张官方 train、1,024 张独立 test、width 24 小型卷积 velocity field、1,000 次更新、3 个 seed。baseline 与 weighted 严格共享初始化、batch、噪声和时间。

- 低中噪声 teacher MSE 平均改善 `3.62%`。
- 高噪声 teacher MSE 平均恶化 `1.50%`。
- 50 步 rollout 的 latent SWD、decoded pixel SWD、feature SWD、feature FID 分别平均恶化 `9.3%`、`10.9%`、`5.4%`、`16.3%`。
- 增加到 200 Euler 步后 weighted/baseline feature FID 仍为 `1.167`，排除粗糙积分是主要原因。
- 将两个模型逐步校准到相同的 8 个 radial-band energies 后，weighted 仍比同样校准的 baseline 平均差 `12.1%` feature FID，否定“频带总能量就是完整机制”。
- 时间与频带硬拼接显示：MNIST 高噪声损伤的约 `88%–95%` 可由 band 0 输出单独复现；bands 1–7 单独替换基本等同 baseline。

MNIST 因而确认了“高噪声粗结构方向具有超出普通 MSE 的因果杠杆”，但它只是一种简单数据上的特例。

### 2. RAE 时间与频带因果拼接

RAE probe 不训练模型，直接使用三组已有 baseline/partial EMA checkpoint、同一批 64 个 ImageNet validation latents、同一噪声、fp32 和官方 50 步 shifted Euler grid。

高噪声 `t>=0.85`：

- partial 全频带使 summary SWD 平均恶化 `5.49%`。
- 只替换 band 0 恶化 `1.87%`，解释约 `33.8%` 的总差值。
- 只替换 bands 1–7 恶化 `3.62%`，解释约 `66.2%`。

中噪声 `0.30<=t<0.85`：

- partial 全频带使 summary SWD 平均恶化 `11.74%`。
- band 0 只解释约 `16.1%` 的总差值。
- bands 1–7 解释约 `81.1%`，单独造成 `9.52%` 的 SWD 恶化。

这里的“解释比例”是单独 splice 产生的 SWD 差值除以 all-band splice
差值，不是线性方差分解或 Shapley attribution。两个比例接近相加，只能
说明该实验中的交互项较小。

因此，MNIST 的“band 0 单因”不能外推成 RAE 的完整解释。RAE 中更大的问题恰恰来自 teacher velocity MSE 已改善的非零频带。

### 3. RAE 共享状态 band-drift 审计

两个 vector fields 被放到完全相同的状态上比较 `2 E[<z_b,v_b>]`，从而区分模型差异与 state-distribution 差异。

在真实 teacher states 上：

- `t≈0.95`：partial/baseline drift RMSE 为 `2.77x`。
- `t≈0.85`：为 `3.24x`。
- `t≈0.54`：partial 反而更好，为 `0.434x`。
- `t≈0.315`：partial 也更好，为 `0.752x`。

这说明高噪声阶段已经存在 on-path marginal-transport 错配；它不是 rollout 后才出现的。

将两个 fields 放到同一批 **baseline rollout states** 上：

- `t≈0.69`：partial drift RMSE 为 `1.063x`。
- `t≈0.54`：为 `1.121x`。
- `t≈0.315`：为 `1.197x`。
- `t≈0.13`：为 `1.058x`。

也就是说，partial 在 `t≈0.54/0.315` 的 teacher states 上更好，但一进入 baseline rollout states 就发生稳定符号翻转。这是直接的 off-path generalization 证据。

最终 partial trajectory 的 radial energy RMSE 相对 baseline trajectory 在所有报告时间都更大：从 `t≈0.69` 的 `1.091x` 增长到 `t≈0.315` 的 `1.293x`、`t≈0.13` 的 `1.310x`，终点仍为 `1.269x`。

### 4. 一致性与数据质量检查

- 三个 RAE seed 使用完全相同的 64 个 validation indices，且 64 个索引全部唯一。
- frequency-switch、band 表和 shared-state drift 表分别有 84、168 和 1,008 行；没有 null、非有限值或重复主键。
- 新 probe 中的 baseline、high-all 和 mid-all schedule 与旧 time-switch probe 独立实现，summary SWD 最大差异仅 `2.4e-7`。
- DCT band blending、Parseval inner product、energy calibration、Euler 方向和共享 probe 均有独立单元测试。
- 三个训练 seed 的关键方向一致；报告的比例不是由单个 seed 的均值异常驱动。

## 被否定或限制的解释

| 候选解释 | 结论 | 证据 |
|---|---|---|
| 只是 Euler 50 步太少 | 否定 | MNIST 100→200 步已收敛，weighted 差距不消失 |
| radial-band energy deficit 是全部原因 | 否定 | 完全相同的校准能量下 weighted 仍差约 12.1% FID |
| 全局 divergence 单调增大导致收缩 | 否定为主因 | divergence 差异随时间变号，因果时间切换指向高噪声而非低噪声 |
| 高噪声 band 0 是全部原因 | MNIST 成立，RAE 否定 | RAE 中 band 0 只解释高段约 34%、中段约 16% |
| teacher 指标改善只是 train 泄露 | 否定 | 所有 teacher 与输运指标均来自独立 validation split |
| decoder 条纹导致结论 | 否定 | 当前因果拼接与 drift 审计完全在 latent/encoder-stage-2 侧完成 |

## 当前可以声称什么

可以声称：

1. 固定 inverse-variance DCT weighting 在有限 RAE velocity model 中改变了方向和时间上的近似分配。
2. 普通 teacher velocity MSE 与真实生成输运的因果敏感度不匹配。
3. 高噪声存在 on-path marginal drift 错配；中噪声存在由早期轨迹偏移触发的 off-path 失稳。
4. 这两项能够解释“teacher 指标改善而 rollout 分布恶化”的主要实验事实。

暂时不能声称：

1. 已恢复完整 Jacobian 谱或严格证明某个单一 contraction eigenmode。
2. radial second moment 是充分统计量。
3. MNIST proxy 可以替代 ImageNet FID。
4. 该机制自动适用于所有 spectral losses、diffusion parameterizations 或模型架构。

## 下一方法必须满足的条件

继续研究时，不应再调固定 `gamma`。候选方法至少需要：

1. 保留原始 MSE 主目标，并给高噪声粗结构方向设置 no-regression guardrail。
2. 直接约束 batch marginal drift，而不是只约束 velocity amplitude：

```text
L_drift = sum_{t,b} alpha_{t,b}
          || E_batch[<z_t^b, v_hat_t^b>]
             - E_batch[<z_t^b, v_target_t^b>] ||^2.
```

3. 在一到两步 self-generated states 上重复该约束，显式训练 off-path response。
4. 因为纯 radial energy calibration 不充分，还要监控 cross-band covariance、低维特征分布或短 rollout SWD。

新方法先通过 MNIST 3-seed、200-step gate：teacher 高噪声不退化，rollout feature FID 和 latent SWD 同时优于 baseline。只有通过后才允许一次 paired tiny RAE screen。

## 复现入口

- `experiments/mnist_transport_mechanism.py`
- `experiments/rae_frequency_time_switch_probe.py`
- `experiments/rae_band_transport_probe.py`
- MNIST 结果：`$HOME/data/eqvae/experiments/mnist_spectral_rollout_toy/*/mechanism_v1/`
- RAE 频带拼接：`$HOME/data/eqvae/experiments/rae_spectral_tiny/frequency_time_switch/`
- RAE 共享状态漂移：`$HOME/data/eqvae/experiments/rae_spectral_tiny/band_transport/`

以上结论基于三个训练 seed；所有 probe 使用固定验证索引和固定噪声，不写入仓库内的 `outputs/` 或 `artifacts/`。
