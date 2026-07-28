# Teacher MSE 改善但 rollout 变差：机制结论

## 最终判断

目前证据足以确认以下机制：

> 固定频率加权损失在有限共享网络中重分配方向风险。高噪声时，DCT
> 高能量低频方向被低权重，而它与其余频带的参数梯度弱耦合，其他频带的更新
> 无法代偿其下降预算；该方向又具有最大的单频带 endpoint leverage。七个较小的
> 非零频带损伤随后近似累积。早期偏差把状态推离 teacher 分布后，partial vector
> field 在中段对 off-path states 的泛化进一步恶化，误差随 ODE 组合放大。

这不是“频带能量少了一点”或“Euler 步数不够”可以完整解释的现象。更精确的表述是：

1. **高噪声阶段存在 covariance-aligned risk allocation、梯度弱耦合与 endpoint
   leverage 的乘积效应。**
2. **中噪声阶段存在 teacher-forcing / off-path generalization gap。**
3. **前者触发方向性输运偏差，后者沿错误状态分布放大；两者组合形成最终退化。**

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

### 4. 冻结 RAE 梯度几何：为什么高噪声会先错

P20 不训练模型，只在三组 step-10000 tiny RAE baseline/partial EMA 上，对同一批
ImageNet validation latents 求 stage-2 最后 transformer block 与 output head 的
真实 autograd gradient。DCT 和固定 random orthogonal basis 使用完全相同的八个
权重特征值，因此差异只能来自基底与数据/网络梯度几何的对齐。

`t=0.85/0.95` 的三 seed 均值为：

| checkpoint | basis | parameter group | band0/nonzero cosine | allocation multiplier | band0 descent ratio |
|---|---|---|---:|---:|---:|
| baseline | DCT | last block | 0.098 | 4.67x | 0.244 |
| baseline | random | last block | 0.836 | 4.87x | 0.952 |
| baseline | DCT | output head | 0.034 | 4.67x | 0.234 |
| baseline | random | output head | 0.670 | 4.88x | 0.900 |
| partial | DCT | last block | 0.073 | 4.68x | 0.240 |
| partial | random | last block | 0.877 | 4.87x | 0.962 |

DCT 的 band-0 descent ratio 在 baseline/partial、两个高时间、两个参数组和三个
seed 的 `24/24` 条件均 `<0.5`；random 则保持在约 `0.90-0.96`。这排除了
“只是权重谱改变”这一解释：同一权重谱只有在 DCT 风险方向与 shared-parameter
gradient 弱耦合时，才会形成 coarse neglect。该现象同时出现在尚未加权训练的
baseline checkpoint 和已经训练后的 partial checkpoint，因此不是 partial 训练后
偶然形成的表征。

### 5. 逐频带 endpoint 干预：梯度预算能否预测因果损伤

P21 在看到 individual-band 结果前固定预测：band 0 应是最有害单频带，并且
bands 0-4 的 `1 - descent_ratio` 应预测 endpoint 损伤排序。随后只在
`t>=0.85` 把 partial 的一个 DCT band 换入 baseline rollout，得到：

| band | high-time descent ratio | endpoint SWD ratio | mean delta |
|---:|---:|---:|---:|
| 0 | 0.244 | 1.01908 | 0.02542 |
| 1 | 0.511 | 1.00719 | 0.00959 |
| 2 | 0.760 | 1.00645 | 0.00860 |
| 3 | 0.901 | 1.00593 | 0.00790 |
| 4 | 1.022 | 1.00451 | 0.00601 |

band 0 在 `3/3` seed 都排名第一；bands 0-4 的排序 Spearman 在每个 seed 都为
`1.0`。八个 individual deltas 之和与 high-all delta 的相对残差仅为
`-1.03%` 到 `+2.37%`。band 0 占总损伤 `35.8%`，其余七个频带合计占
`64.2%`：所谓“nonzero 贡献 66%”不是某个高频带具有更大杠杆，而是七个较小
效应近似相加。

这组结果把此前最反常的断层变成了可检验机制：**加权目标以及 aggregate
teacher MSE 的改善可由多个低残差方差、高权重方向的收益累计主导，但 endpoint
质量由共享梯度能否覆盖被低权重的高杠杆方向决定。**
P20 的下降预算在独立的 P21 rollout 干预中准确预测排序，因而不再只是相关性。

### 6. 一致性与数据质量检查

- 三个 RAE seed 使用完全相同的 64 个 validation indices，且 64 个索引全部唯一。
- frequency-switch、band 表和 shared-state drift 表分别有 84、168 和 1,008 行；没有 null、非有限值或重复主键。
- 新 probe 中的 baseline、high-all 和 mid-all schedule 与旧 time-switch probe 独立实现，summary SWD 最大差异仅 `2.4e-7`。
- DCT band blending、Parseval inner product、energy calibration、Euler 方向和共享 probe 均有独立单元测试。
- 三个训练 seed 的关键方向一致；报告的比例不是由单个 seed 的均值异常驱动。
- P20 的 aggregate、band、cosine 表分别有 `72/576/4,608` 行；P21 的
  metrics、paired、band 表分别有 `120/120/240` 行。复合主键无重复，数值列无
  null 或非有限值。
- P21 使用的 32 个 validation indices 在三个 seed 间完全一致且全部唯一；旧
  64-sample probe 与新 32-sample probe 的 high-all ratio 仅差 `0.00159`，band-0
  ratio 仅差 `0.00041`。

## 被否定或限制的解释

| 候选解释 | 结论 | 证据 |
|---|---|---|
| 只是 Euler 50 步太少 | 否定 | MNIST 100→200 步已收敛，weighted 差距不消失 |
| radial-band energy deficit 是全部原因 | 否定 | 完全相同的校准能量下 weighted 仍差约 12.1% FID |
| 全局 divergence 单调增大导致收缩 | 否定为主因 | divergence 差异随时间变号，因果时间切换指向高噪声而非低噪声 |
| 高噪声 band 0 是全部原因 | MNIST 成立，RAE 否定 | RAE 中 band 0 只解释高段约 34%、中段约 16% |
| RAE 的 nonzero 66% 说明高频单方向杠杆更大 | 否定 | band 0 是 3/3 seed 最大单 band；nonzero 是七个小效应近似相加 |
| 只要权重谱相同，优化效应就应相同 | 否定 | random 与 DCT 同谱，但 coarse descent 约 0.95 对 0.24 |
| teacher 指标改善只是 train 泄露 | 否定 | 所有 teacher 与输运指标均来自独立 validation split |
| decoder 条纹导致结论 | 否定 | 当前因果拼接与 drift 审计完全在 latent/encoder-stage-2 侧完成 |

## 当前可以声称什么

可以声称：

1. 固定 inverse-variance DCT weighting 在有限 RAE velocity model 中造成与
   数据几何对齐的方向风险重分配，而不是无害预条件。
2. 高噪声 DCT band 0 与其余频带梯度弱耦合，导致有效下降预算塌缩；同谱 random
   对照不存在该现象。
3. 逐 band endpoint 因果杠杆按稳定梯度预算排序，七个非零频带损伤近似累积。
4. 高噪声存在 on-path marginal drift 错配；中噪声存在由早期轨迹偏移触发的
   off-path 失稳。二者解释了“teacher 指标改善而 rollout 分布恶化”的主要事实。

暂时不能声称：

1. 已恢复完整 Jacobian 谱或严格证明某个单一 contraction eigenmode。
2. radial second moment 是充分统计量。
3. MNIST proxy 可以替代 ImageNet FID。
4. 该机制自动适用于所有 spectral losses、diffusion parameterizations 或模型架构。
5. P21 的 32-sample summary SWD 等价于 50k FID，或已经给出成功训练方法。
6. P20 的 8 个 validation latents、一个 random basis 和最后 block/head 足以恢复
   全网络训练过程；它们只构成幅度很大的冻结梯度桥接证据。
7. 梯度预算与 field-splice 损伤的排序一致，意味着训练侧 guardrail 必然改善
   RAE FID；这仍需新的 paired training intervention 验证。

## 下一方法必须满足的条件

P22-P24 已进一步证明：精确保护 baseline transport 可以消除大部分额外退化，
但没有稳定优于 baseline；在 self-generated state 上复用原 paired target 会系统性
恶化。因此继续研究时，不应再调固定 `gamma`，也不应继续该 off-path MSE。候选
方法至少需要：

1. 保留原始 MSE 主目标，并给高噪声粗结构方向设置 no-regression guardrail。
2. 直接约束 batch marginal drift，而不是只约束 velocity amplitude：

```text
L_drift = sum_{t,b} alpha_{t,b}
          || E_batch[<z_t^b, v_hat_t^b>]
             - E_batch[<z_t^b, v_target_t^b>] ||^2.
```

3. self-generated states 只使用可识别的 batch marginal、baseline trust-region 或
   rollout distribution 约束，不能复用原 pair velocity label。
4. 因为纯 radial energy calibration 不充分，还要监控 cross-band covariance、
   低维特征分布或短 rollout SWD，并在独立 validation/test 上验证。

新方法先在 FashionMNIST/MNIST 多 seed 上达到绝对 baseline 至少 `2%` 的 FID
改善，并同时不恶化 latent/feature SWD。P22-P24 未通过，因此当前不允许 paired
tiny RAE screen。详见 `docs/MECHANISM_TO_QUALITY_STUDY_ZH.md`。

## 复现入口

- `experiments/mnist_transport_mechanism.py`
- `experiments/rae_frequency_time_switch_probe.py`
- `experiments/rae_band_transport_probe.py`
- `experiments/rae_frozen_gradient_bridge.py`
- `experiments/run_rae_individual_band_leverage.py`
- `experiments/small_image_residual_adapter.py`
- `experiments/small_image_residual_trust_region.py`
- MNIST 结果：`$HOME/data/eqvae/experiments/mnist_spectral_rollout_toy/*/mechanism_v1/`
- RAE 频带拼接：`$HOME/data/eqvae/experiments/rae_spectral_tiny/frequency_time_switch/`
- RAE 共享状态漂移：`$HOME/data/eqvae/experiments/rae_spectral_tiny/band_transport/`
- RAE 冻结梯度桥接：`$HOME/data/eqvae/experiments/rae_frozen_gradient_bridge/preregistered_20260716_200310/`
- RAE 逐频带 endpoint：`$HOME/data/eqvae/experiments/rae_individual_band_leverage/preregistered_20260716_200958/`
- 机制到质量 gate：`$HOME/data/eqvae/experiments/small_image_residual_adapter/` 和
  `$HOME/data/eqvae/experiments/small_image_residual_trust_region/`

以上结论基于三个训练 seed；所有 probe 使用固定验证索引和固定噪声，不写入仓库内的 `outputs/` 或 `artifacts/`。
