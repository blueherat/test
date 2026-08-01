# RAEv2 Self-Guidance 研究账本

> **路线状态（2026-07-31）：普通 LPL 方法线已停止。** 本文保留完整实验账本，
> 最终决策、文献边界和下一阶段预注册计划见
> [LPL 路线结题审计与下一阶段研究议程](LPL_LINE_CLOSURE_AND_SOLID_RESEARCH_AGENDA_ZH.md)。

## 1. 研究目的

当前研究不再把目标写成“让 LPL 在 RAEv2 上也有效”，而是研究一个更一般的
问题：

> 当生成模型依赖同一网络中 weak/strong 两个预测的差值进行 self-guidance
> 时，后训练如何改善 strong prediction，又不破坏已经有效的 guidance
> 关系？

LPL 是当前最清楚的干预工具，不是必须被保住的最终方法。

## 2. 此前路线的归档

### 2.1 群结构与等变性

- 已完成 RAE/DINOv2/MAE/SigLIP2 的 layerwise、token correspondence、
  Procrustes、generator power 和 mean/residual 诊断。
- 结果支持“存在弱几何响应”，不支持“final latent 中存在干净、全局、
  可组合的群表示”。
- 强行施加等变 adapter 会改变位置编码和 patch embedding 原本形成的表示，
  且没有得到稳定生成收益。
- 状态：保留为表征诊断背景，不是当前生成主线。

### 2.2 latent 平滑度、频谱预条件与 toy

- 多种无需训练的平滑度 proxy 不能稳定预测真实生成质量。
- 频谱机制在 toy 上存在，但从 toy 优化动态迁移到真实 RAE 生成出现断层。
- 状态：机制素材保留，不再扩大真实 RAE 训练。

### 2.3 latent 容量与 prior-decoder 权衡

- 小型两阶段模型显示 latent prior loss 与 decoder 最终质量并不一一对应。
- “中等 latent 容量必然最优”没有得到足够稳定的多 seed 证据。
- 状态：作为两阶段生成背景，不是当前可执行主线。

### 2.4 decoder mismatch 与 LPL

- 旧 RAE 上，严格配对、冻结 encoder/decoder 的 LPL 在 DINOv2、MAE、
  SigLIP2 上跨 seed 改善相对 Flow continuation 的生成指标。
- “LPL 把误差移入 decoder 局部低敏感方向”已被反证。
- 更稳定的现象是：旧 RAE Flow prediction 存在 decoder-feature
  contraction，LPL 的 prediction-variance gradient 提供明显的反收缩更新。
- 状态：这是当前主线的重要正对照。

## 3. RAEv2 已确认事实

1. RAEv2 使用 `full` 与 `base` 两个 clean-latent prediction：

   ```text
   d = full - base
   guided = full + 0.78 * d
   ```

2. 官方 Internal Guidance 在 50k 论文结果中把无引导 gFID
   `1.65` 改善到 `1.06`。
3. 在本地严格同噪声 step100 消融中，关闭 IG 将 LPL 相对 Flow 的 FID
   伤害从 `+3.1025` 缩小到 `+0.7865`，约消除 `74.6%` 的退化。
4. 因此 IG 是 LPL 退化的主要放大器，但不是唯一原因。
5. strict LPL 的 prediction-variance denominator gradient 是额外退化的
   重要来源；detach/raw 去掉该问题后只回到 Flow 附近，没有稳定新增收益。
6. 官方 guidance 会增加 paired clean-latent/decoder-feature error，却改善
   最终生成分布。paired endpoint objective 与 guided distribution objective
   不是同一个目标。
7. seed0 的 `2 x 2` head-swap 没有支持“contrast 是唯一主因”：

   | full source | contrast source | 相对 Flow/Flow FID |
   |---|---|---:|
   | LPL | Flow | +0.4092 |
   | Flow | LPL | +0.1583 |
   | LPL | LPL | +0.9204 |

   full field、contrast 和 rollout interaction 都有贡献。

完整证据和配置见：

- `docs/RAE_RAEV2_LPL_DEEP_SYNTHESIS_ZH.md`
- `docs/RAEV2_LPL_STRICT_CONTINUATION_ZH.md`

## 4. 当前核心假设

RAEv2 的双头训练把 `full` 和 `base` 都拉向同一个 paired target，但采样把
它们的差值当作质量方向放大。这个差值是训练副产物，不是被生成质量直接监督
出的稳定变量。

因此 decoder-aware post-training 可能同时发生两件事：

1. 改变 strong/full vector field；
2. 改变 weak-to-strong contrast，随后被 IG 外推和 ODE rollout 放大。

当前不能声称第二项是唯一机制。更稳妥的研究问题是：

> 在保持原始 contrast function 完全不变时，decoder-aware common correction
> 是否仍然伤害生成？

## 5. 第一项方法实验：Contrast-Preserving Common Adapter

冻结官方 RAEv2 Stage-2，训练一个零初始化的小 adapter：

```text
a = A(x_t, t, full, base)
full' = full + a
base' = base + a
```

于是对任意相同输入状态严格满足：

```text
full' - base' = full - base
```

采样输出为：

```text
guided' = full' + 0.78 * (full' - base')
        = guided + a
```

这个参数化比 `guided_common` 的输出梯度替换更严格。后者只能让一次 backward
时两个 prediction tensor 接收共同梯度；共享主干经过参数更新后不能保证
两个函数发生完全相同的变化。Common Adapter 在函数定义上恒等保持 contrast。

必须同时保留一个限制：

- adapter 会改变 ODE 状态和后续采样轨迹；
- 所以它保持的是任意给定状态上的 contrast function，不保证新旧轨迹访问
  相同状态；
- 最终结论必须来自同噪声完整 rollout。

## 6. 公平对照

所有分支使用同一冻结 source model、同一 adapter 架构、初始化、训练数据、
噪声、时间和更新数：

1. `zero_adapter`：不训练，严格复现官方模型。
2. `flow_common`：只用官方 full+base Flow loss 训练 adapter。
3. `lpl_common`：相同 Flow loss加 decoder-feature LPL。

主实验的 LPL 直接作用于真实采样使用的：

```text
guided' = IG(full, base) + a
```

`full'` 只保留为目标错位消融。原因是当前问题不是再次复现旧 RAE 单头 LPL，
而是检验 decoder-aware common correction 能否改善 RAEv2 实际部署的 guided
vector field。

## 7. 无泄露与实现边界

- 训练只读取 ImageNet train。
- ImageNet validation/ADM reference 只用于训练结束后的指标。
- encoder、Stage-1 decoder、RAEv2 Stage-2 全部冻结。
- optimizer 只包含 adapter 参数。
- Flow 与 LPL 分支使用相同 train index stream。
- checkpoint 单独保存 source SHA、adapter state、optimizer state和数据摘要。
- 不把官方 Stage-2 optimizer continuation 与新 adapter optimizer混为一谈。
- 同噪声采样必须记录每 rank 的 noise、label 和 RNG fingerprint。

## 8. 预注册判断

### 支持 contrast-fragility

若 common adapter 相比原全模型 LPL 显著减少退化，且 `lpl_common` 超过
`flow_common`，说明旧 LPL 的信息仍有价值，主要问题是原微调破坏了
self-guidance 的 weak/strong 参数化。

### 支持 paired-to-distribution conflict

若 contrast 被严格保持后，`lpl_common` 仍不优于 `flow_common`，说明主要
问题不是双头差值被破坏，而是 paired decoder alignment 本身不适合优化成熟
guided distribution。

### 支持 full-field/rollout conflict

若 adapter 在固定状态上降低 LPL/Flow proxy，但完整 rollout 退化，说明断层
发生在递归访问的新状态分布，后续只能研究 rollout-aware objective，不能再
增加 endpoint loss。

## 9. 分级验收

### 静态验收

- zero-init 时 wrapper 输出逐 tensor 等于 source 输出。
- 任意 adapter 参数下，`full'-base'` 与 source `full-base` 在 fp32
  数值容差内一致。
- source model 无梯度，optimizer 参数集合严格等于 adapter 参数集合。
- Flow/LPL 首批数据、latent、noise、time、CFG mask fingerprint 一致。

### 小规模真实验收

- 1 update 和 10 update 无 NaN、无 OOM、loss 与梯度有限。
- `flow_common` 与 `lpl_common` 使用完全相同的数据随机流。
- 1k 同噪声只作方向筛查，不作最终结论。

### 扩大门槛

- 至少两个 5k sample seed 中，`lpl_common` 都不差于 `flow_common`；
- FID、KID、IS 至少两项方向一致；
- contrast preservation 审计通过；
- 通过后才做第三 seed 和一次 50k。

若 5k 两个 seed 都显示 `lpl_common` 更差，先复核代码、checkpoint、采样状态
和指标；确认现象真实后，停止 common-LPL 方法线。

## 10. 当前不作为创新的内容

- 单纯改变 IG scale；
- 简单改变 guidance interval；
- 换一个 intermediate head 位置；
- 增加多个 weak heads；
- 把 LPL、REPA、CFG 和动态权重直接相加。

这些已有紧邻工作或只能构成工程调参。当前可能有研究价值的是
**self-guided model 的 post-training stability**，以及“better paired
predictor can become a worse guide”的可复现实证和约束方法。

## 11. 决策记录

### 2026-07-31

- 建立当前主线。
- 第一实验选择 contrast-preserving common adapter。
- 暂不直接训练 base head，也不先做动态 guidance scale。
- 原因：head-swap 已表明 full field 不能忽略；共同 adapter 是目前能以最少
  新假设严格隔离 contrast 变化的参数化。
- 复核了已有 `guided_common` 全模型实验：它只在一次 backward 中把 full/base
  输出梯度都设为 `0.5`，不能约束参数更新后的函数差值。其非 EMA 5k FID 为
  `11.0360`，差于同计算量 Flow10 的 `10.8278`。因此它是“共同输出梯度不够”
  的反例，不是当前函数级 contrast-preserving adapter 的重复实验。
- 新 adapter 共 `268,480` 个参数；官方 Stage-2 的 `875,216,500` 个参数全部
  冻结。真实 checkpoint 一步训练、保存和恢复已通过。
- 连续两步与一步后恢复到第二步的最大参数差为 `1.46e-11`，EMA 最大差为
  `1.42e-14`，四个 rank 的最终 RNG 状态完全一致。
- BF16 中分别计算 `full+a`、`base+a` 会通过舍入污染差值。采样实现已改为先
  完全复现官方 IG 算术，再只加一次 `a`；zero adapter 与官方 guided 数值
  精确一致，训练中的 contrast 误差约为 `2.38e-7`。
- 1024 个完全相同 ImageNet train 样本的初始梯度审计：

  | objective | gradient norm |
  |---|---:|
  | Flow | `0.00405735` |
  | raw LPL on full | `28799.60` |
  | raw LPL on guided | `10644.49` |

  guided-LPL 与 Flow 梯度余弦为 `0.00644`，几乎正交；与 full-LPL 梯度余弦
  为 `0.95784`。若令 guided-LPL 初始梯度为 Flow 的 `20%`，权重应为
  `7.62338e-8`。
- Flow/LPL 独立训练作业的四个 rank 在首批 image、label、latent、noise、
  time 和 CFG mask 上逐哈希一致。三分支四图同噪声 ODE 冒烟也确认各 rank
  噪声不同、各分支随机流相同且没有重复样本。
- 冻结在线 `model` source 的 10-step 配对实验使用 global batch `1024`、
  AdamW `2e-5`、raw guided-LPL weight `7.62338e-8`：
  - 两分支逐步 Flow loss 最大差 `1.07e-6`；
  - 最终 correction/source RMS 为 Flow `0.1539%`、LPL `0.1565%`；
  - step1 到 step10 的参数更新余弦 `0.9722`，LPL 相对 Flow 的差分范数约为
    Flow 更新的 `23.9%`；
  - contrast 数值误差不超过 `9.54e-7`。
- 1k 同噪声筛查在两个采样 seed 上发生镜像反转：

  | seed | Flow FID | LPL FID | LPL - Flow |
  |---:|---:|---:|---:|
  | 0 | `41.4637` | `41.4745` | `+0.0109` |
  | 1 | `41.3839` | `41.3716` | `-0.0123` |

  这证明 1k 差异由抽样噪声主导，不能选择性报告。
- 三个 5k sampling seed 的 step10 配对结果：

  | seed | Flow FID | LPL FID | LPL - Flow | KID 差值 | IS 差值 |
  |---:|---:|---:|---:|---:|---:|
  | 0 | `10.96996` | `10.95841` | `-0.01155` | `-3.17e-6` | `-0.166` |
  | 1 | `10.78186` | `10.76649` | `-0.01537` | `-0.04e-6` | `-0.015` |
  | 2 | `11.05787` | `11.05597` | `-0.00190` | `-4.46e-6` | `+0.482` |

  FID 差值均值为 `-0.00961 ± 0.00694`，三次方向一致；KID 三次不差，
  IS 方向不一致。这是一个可复现但量级极小的信号，不能称为实用质量提升。
- 逐图同噪声检查显示，小的 vector-field 方向差会被完整 ODE 轨迹放大：
  step10 的 LPL-Flow 输出 RMS 约为 `0.01842`，其相对 Flow-zero 更新方向的
  平均余弦为 `-0.4298`。这说明“参数和单点输出接近”不等于“最终逐图样本
  接近”，但分布指标仍可能几乎相同。
- Flow/LPL 均从 step10 精确续训到 step50，并保存 `20/30/40/50`：
  - resume 后首批 `indices/image/label/latent/noise/time/CFG mask` 哈希全相同；
  - 两支 step11--50 的数据摘要完全相同；
  - 逐步 Flow loss 平均绝对差 `2.93e-6`，最大 `5.48e-6`；
  - contrast 误差始终不超过 `9.54e-7`；
  - step50 correction/source RMS 为 Flow `0.585%`、LPL `0.596%`。
- 参数差没有随训练剂量相对放大。以共同 step1 状态为参照，LPL/Flow 更新
  余弦从 step10 的 `0.9680` 升到 step50 的 `0.9776`；两分支差分范数占
  Flow 更新范数的比例从 `25.6%` 降到 `21.4%`。
- 同一个 1k seed 的 checkpoint sweep：

  | step | Flow FID | LPL FID | LPL - Flow |
  |---:|---:|---:|---:|
  | 20 | `41.42177` | `41.43200` | `+0.01024` |
  | 30 | `41.43402` | `41.45125` | `+0.01723` |
  | 40 | `41.37070` | `41.38510` | `+0.01441` |
  | 50 | `41.37777` | `41.30103` | `-0.07674` |

  因而“LPL 收益随 step 单调放大”不成立。step50 的转折可能是真实的晚期
  效应，也可能是已知很不稳定的 1k FID；必须以配对 5k 复核，不能挑选
  step50 后直接宣称成功。
- step50 的三个 5k seed 没有复现 1k 的大幅转折：

  | seed | Flow FID | LPL FID | LPL - Flow | KID 方向 | IS 方向 |
  |---:|---:|---:|---:|---|---|
  | 0 | `10.91365` | `10.92121` | `+0.00756` | LPL 差 | LPL 好 |
  | 1 | `10.75056` | `10.72680` | `-0.02376` | 近似相同 | LPL 好 |
  | 2 | `10.98238` | `10.96679` | `-0.01559` | LPL 差 | LPL 差 |

  FID 差值均值约为 `-0.0106 ± 0.0162`，与 step10 的 `-0.0096` 同量级，
  方差反而更大；KID 三个 seed 均未改善。从 step10 到 step50，Flow 和
  LPL 自身都能从继续训练中受益，但没有证据表明增加 step 放大了 LPL 的
  相对生成收益。
- 新增固定未见 validation 配对探针。它固定 64 张 ImageNet validation 图、
  noise、time 和 source model，只替换 adapter checkpoint，因此训练日志中
  不同随机 batch 的 LPL 不可比问题被排除。相对同 step Flow：

  | step | noise ratio | raw guided LPL 相对变化 | LPL 更低的样本比例 |
  |---:|---:|---:|---:|
  | 10 | `0.5` | `-0.21%` | `65.6%` |
  | 10 | `1.0` | `-0.27%` | `68.8%` |
  | 10 | `3.0` | `-0.49%` | `85.9%` |
  | 50 | `0.5` | `-0.32%` | `64.1%` |
  | 50 | `1.0` | `-0.75%` | `67.2%` |
  | 50 | `3.0` | `-1.56%` | `87.5%` |

  Flow loss 的相对恶化仅约 `0.00001%--0.0021%`，guided latent 相对误差
  基本不变。由此可以排除“LPL 根本没有优化到自身目标”：它确实越来越能
  降低固定配对上的 decoder-feature loss，尤其在 gate 边界的较高噪声处；
  但该局部改善没有自动转化成完整 sampler 的分布质量。
- 新增 common-adapter recursive endpoint probe，并以 batch `8` 跑了 64 张
  validation 图。它从已知 clean latent/noise 插值状态出发，分别递归
  `1/4/16` 次，因此在引入 self-induced state 的同时仍有真实 clean target。
  step50 LPL 相对 Flow 的 raw feature loss：

  | noise ratio | 1 query | 4 queries | 16 queries |
  |---:|---:|---:|---:|
  | `1.0` | `-0.66%`，显著 | `-0.06%`，不显著 | `+0.48%`，不显著 |
  | `3.0` | `-1.68%`，显著 | `-0.80%`，显著但减弱 | `-0.38%`，不显著 |

  对应的 LPL 更低样本比例从一次查询的 `71.9%/79.7%`，下降到 16 次查询的
  `48.4%/59.4%`。state-path error 和 latent error 的分支差异仍接近零。
  因而更准确的机制不是“进入 rollout 后 LPL 平均值必然立刻反转”，而是：
  **在真实配对上稳定的局部 decoder-feature 优势，随递归状态偏移快速失去
  跨样本一致性，不能形成沿采样轨迹闭合且可累积的改善。**

## 12. 当前机制判断

### 已有证据支持

- 原全模型 LPL 的明显退化至少部分来自 self-guidance 函数被改变；严格保持
  `full-base` 后，大幅退化消失。
- contrast preservation 只消除了一个失败源，并没有让 decoder LPL 自动
  成为有效的生成改进。
- LPL 的 one-step、paired decoder-feature 目标可以在未见 validation 配对
  上泛化，而且随训练继续下降。
- one-step LPL 下降与完整 ODE rollout 的 FID 脱钩。64 图递归探针进一步
  表明，优势会随 self-induced state shift 失去统计稳定性。因此当前主要
  断层位于“训练路径上的局部 clean-latent 预测”到“沿模型自身轨迹可闭合、
  可累积的 decoder compatibility”，而不是 LPL 实现、数据泄露或 optimizer
  没有更新。

### 仍不能宣称

- 不能仅凭 restricted adapter 实验断言旧退化全部由 contrast 变化造成；
  adapter 同时限制了容量、更新幅度和可修改函数族。
- 不能说 LPL 在 RAEv2 上有实用质量提升。step10 三 seed 的 FID 正向仅约
  `0.01`，step50 三 seed 仍约 `0.01` 且 KID 不改善，远小于模型间或正式
  50k 评估尺度。
- 不能把 64 图受控递归直接等同于无条件 sampler 的完整状态分布。它验证了
  局部优势不闭合的机制，但不是正式生成指标。

### 下一条最小研究问题

此前根据 RAEv2 单模型结果，下一项曾暂定为训练短递归
rollout-closure loss。下面两个判别性对照推翻了它作为第一优先级的依据：
旧 RAE 的 LPL 即使没有更准确的递归 latent 轨迹，仍然可以明显改善正式
生成指标。因此不应直接把 rollout error 当成 LPL 成功的必要条件。

## 13. 判别性对照：旧 RAE 递归轨迹与 RAEv2 full-head LPL

### 13.1 对照 A：旧 RAE 的 LPL 是否真的改善递归轨迹

使用已经完成正式生成验证的 DINOv2-B seed 3、update 2000 Flow/LPL EMA
checkpoint，在 64 张固定 ImageNet validation 图上运行与 RAEv2 完全一致的
受控递归探针：

- noise ratio：`1` 和 `3`；
- endpoint query：`1/4/16` 次；
- 每个分支使用相同图像、label、clean latent 和 noise；
- decoder feature 层和原 LPL 保持一致；
- 不训练参数，不使用 validation 结果选择 checkpoint。

结果与“LPL 通过更准确的 latent 轨迹改善生成”的解释相反：

| noise ratio | queries | LPL 相对 Flow latent error | feature variance ratio：Flow -> LPL |
|---:|---:|---:|---:|
| 1 | 1 | `+0.54%` | `0.869 -> 0.951` |
| 1 | 4 | `+1.15%` | `0.905 -> 0.967` |
| 1 | 16 | `+1.22%` | `0.918 -> 0.976` |
| 3 | 1 | `+0.27%` | `0.779 -> 0.878` |
| 3 | 4 | `+1.36%` | `0.844 -> 0.950` |
| 3 | 16 | `+1.57%` | `0.875 -> 0.972` |

六个 latent error 差值的 95% CI 都严格大于零，且 64/64 样本均不是 LPL
更低。raw decoder-feature squared error 在不同噪声和 query 数上有正有负，
所有 95% CI 均跨零，不能解释已知的强 FID 改善。

真正稳定的变化是：

- decoder feature variance ratio 在六个条件下全部显著向 `1` 靠近；
- centered feature cosine 在六个条件下全部提高约 `0.8%--1.5%`；
- 这些变化在递归 16 次后仍然保留。

因此，旧 RAE 的 LPL 收益更符合“修复 prior endpoint 在 decoder feature
空间中的收缩和方向校准”，而不是“让每条 latent 轨迹更接近 clean latent”。
这与此前 full LPL 的 denominator-gradient 会主动增大 feature variance 的
审计结果相互吻合。

完整结果：

```text
~/data/eqvae/experiments/rae_lpl_recursive_rollout/seed3_u2000_n64/
```

### 13.2 对照 B：RAEv2 只对 full head 做 LPL

为排除“训练目标监督 guided 输出，而官方模型的 full head 才是正确对象”
这一解释，训练了保持 `full-base` 完全不变的共同残差 adapter：

```text
full' = full + a
base' = base + a
```

Flow loss 仍同时作用在两个官方输出上，但 decoder LPL 只监督
`full + a`，不监督 guided 组合。审计与训练配置为：

- 官方 DINOv3-L/K7 online `model`，source step `100080`；
- source Stage-2、RAE encoder 和 decoder 全部冻结；
- adapter 参数 `268,480`；
- global batch `1024`，4 卡，AdamW `2e-5`；
- 10 updates，共见到 10,240 张训练图；
- raw full-head LPL 初始梯度范数 `28799.5957`；
- 权重 `2.8176437923896238e-8`，使其初始梯度为 Flow 的 `20%`。

显式重跑的 1024 样本审计与旧 full-head 审计完全一致：

- Flow loss：`0.8966483474`；
- data-index SHA256：
  `7f877f7b3e7e0f7c298ebdf6c4482cf8eacc78ea09ad73f5059cd2fd6bc3a755`；
- LPL gradient-unit SHA256：
  `258d7341a34e1fbbdd7962bd62b68224e0db2f8a2c21da9ed27d954261757d7e`；
- contrast error：`0`。

full-LPL 与 Flow 的首批 image/label/latent/noise/time/CFG mask 逐哈希一致，
10 updates 累计 data-index SHA 也一致。训练中 `full-base` 数值误差始终不
超过 `9.54e-7`，没有 validation/reference 泄露。

固定 64 图配对中，full-LPL 确实降低了采样实际使用的 guided decoder-feature
loss：

| noise ratio | one-query 相对 Flow | 95% CI 是否跨零 |
|---:|---:|---|
| 1 | `-0.24%` | 否 |
| 3 | `-0.53%` | 否 |

但进入递归后，收益快速缩小：

| noise ratio | 1 query | 4 queries | 16 queries |
|---:|---:|---:|---:|
| 1 | `-0.24%` | `-0.10%` | `-0.07%` |
| 3 | `-0.53%` | `-0.31%` | `-0.20%` |

除一次查询以及 noise ratio 3 的四次查询外，其余 CI 均跨零。latent、
state-path 和 endpoint error 均没有稳定改善。把监督对象从 guided 换成 full
没有恢复旧 RAE 的机制。

最后按已有完全相同的 seed 0、100-step、IG `1.78`、4 卡同噪声协议生成
1000 张平衡 ImageNet 样本：

| branch | FID | KID mean | IS |
|---|---:|---:|---:|
| Flow | `41.463665` | `0.00007150` | `56.6316` |
| guided-LPL | `41.474534` | `0.00007245` | `56.5745` |
| full-LPL | `41.462439` | `0.00008148` | `56.6430` |

full-LPL 与 Flow 的 FID 只差 `-0.00123`，远小于已观测到的 1k sampling
波动；KID 反而更差。因此没有达到扩大到 5k 或多 seed 的预注册门槛。

完整结果：

```text
~/data/eqvae/experiments/raev2_common_adapter_full_target_control/
```

### 13.3 两个对照后的机制判断

现有证据已经排除两条直觉解释：

1. 旧 RAE 的 LPL 不是因为 latent rollout 更准确才成功。
2. RAEv2 的 LPL 也不是因为错误监督 guided 而非 full 才失败。

当前最有解释力、但仍需直接验证的假设是：

> 旧 RAE prior 的 endpoint 在 decoder feature 空间中存在明显收缩，full LPL
> 的 denominator-gradient 恰好修复了这种分布校准；RAEv2 的成熟 prior 和
> internal guidance 已经大幅处理了同类校准，剩余的逐样本 feature 对齐既小，
> 也无法提供额外生成收益。

这仍是“最佳支持假设”，不是已经证明的因果机制。下一项最小实验应直接测量：

> 旧 RAE 与 RAEv2 的真实采样 endpoint，相对 clean latent 在各 decoder 层的
> feature mean、variance、covariance coverage 和方向分布究竟差多少？

必须先用无需训练的 endpoint distribution atlas 验证“旧 RAE 有收缩、
RAEv2 已接近校准”。只有该跨模型差异明确存在，并且 LPL 的变化方向确实
指向 clean 分布，才值得设计新的 calibration 方法；否则应停止这条机制线，
承认旧 RAE 的收益尚不能由当前指标解释。

## 14. Endpoint feature distribution atlas 终验

### 14.1 协议与实现审计

统一 atlas 已在旧 RAE DINOv2-B seed 3 和 RAEv2 DINOv3-L/K7 上完成。每个
条件固定使用 64 张 ImageNet validation 图、相同 seed、noise ratio `1/3`、
`1/4/16` 次 endpoint query，并观测 decoder 的 `0.2/0.4/0.6/0.8/1.0`
五个相对深度。

atlas 同时测量两类不能混为一谈的量：

- 单张图内部的空间 feature variance、centered cosine 和 raw MSE；
- 跨 64 张图的 feature mean、covariance、normalized Fréchet 和 sliced
  Wasserstein。各层 feature 先固定池化到 `4x4`，再用同一个固定随机投影
  降到 32 维。

实现通过恒等输入、已知 `0.5x` 收缩、确定性投影、分布式聚合和 metadata
保真测试；RAEv2 的 Flow、guided-LPL、full-LPL step 50 checkpoint 还满足：

- source SHA、source state 和累计 data-index SHA 完全相同；
- 从 step 10 接续后的首批 image/label/latent/noise/time/CFG mask 哈希逐项
  相同；
- source Stage-2、encoder 和 decoder 均冻结；
- `full-base` contrast 数值误差不超过 `9.54e-7`。

### 14.2 旧 RAE：LPL 确实修复了广泛的 endpoint 分布收缩

跨五层和三种 query 数平均：

| metric（越低越好，cosine 除外） | Flow | LPL | 相对变化 |
|---|---:|---:|---:|
| spatial variance log error | `0.051665` | `0.026104` | `-49.5%` |
| projected mean relative error | `0.037525` | `0.027003` | `-28.0%` |
| covariance relative error | `0.077358` | `0.073592` | `-4.9%` |
| normalized Fréchet | `0.005233` | `0.004206` | `-19.6%` |
| normalized SWD | `0.080160` | `0.074634` | `-6.9%` |
| centered cosine | `0.961819` | `0.963182` | 提高 `0.001363` |

逐层看，LPL 把单图空间方差比从 `0.940--0.958` 提高到
`0.964--1.000`，population covariance trace 也整体更接近 `1`。结合上一节
latent RMS 反而稳定变差的结果，可以更有把握地说：旧 RAE 的 LPL 主要改变
的是 decoder 所见 endpoint 的 feature distribution，而不是把 latent 数值
预测变得更准确。

### 14.3 RAEv2：internal guidance 已承担了大部分同类校准

只改变官方 source 模型的采样组合，把 IG 从 `1.0` 改为 `1.78`，不训练任何
参数，得到：

| metric | IG=1.0 | IG=1.78 | 相对变化 |
|---|---:|---:|---:|
| spatial variance log error | `0.060620` | `0.037394` | `-38.3%` |
| projected mean relative error | `0.055508` | `0.035917` | `-35.3%` |
| covariance trace log error | `0.057455` | `0.043417` | `-24.4%` |
| covariance relative error | `0.087856` | `0.085606` | `-2.6%` |
| normalized Fréchet | `0.008477` | `0.006770` | `-20.1%` |
| normalized SWD | `0.095259` | `0.088170` | `-7.4%` |

与此同时，raw MSE 从 `25.12` 上升到 `27.37`，centered cosine 从
`0.94054` 降到 `0.93406`。这说明 internal guidance 不是在做更准确的 paired
regression，而是在调整生成 endpoint 的总体统计；这与旧 RAE 的成功机制
方向一致。

### 14.4 多训到 50 step 后，RAEv2-LPL 暴露出局部与总体校准的分叉

full-LPL 已从 step 10 严格续训到 step 50，并每 10 step 保存 checkpoint。
相对同一 source、IG `1.78`，step 50 的结果为：

| metric | full-LPL 相对 source | guided-LPL 相对 source |
|---|---:|---:|
| spatial variance log error | `-8.25%` | `-5.47%` |
| projected mean relative error | `-22.24%` | `-25.78%` |
| normalized Fréchet | `-4.83%` | `-6.25%` |
| normalized SWD | `-2.48%` | `-2.98%` |
| covariance relative error | `+7.53%` | `+6.54%` |

从 step 10 到 50，full-LPL 的 spatial variance error 又下降 `5.54%`，
mean error 下降 `7.59%`，但 covariance error 上升 `4.88%`，Fréchet 只再
下降 `0.31%`。因此 10 step 不是“训练太少所以看不出作用”：继续训练确实
放大了 LPL 想优化的单图方差和均值效应，同时也放大了 population covariance
形状的错配。

现有 3 seed、每 seed 5k 图的正式 step 50 指标也与此一致：

| branch | FID mean | seed std |
|---|---:|---:|
| Flow50 | `10.8822` | `0.1191` |
| guided-LPL50 | `10.8716` | `0.1275` |

平均差仅 `-0.0106`，step 10 的平均差也只有 `-0.0096`，都远小于 seed
波动。不能宣称有实用生成改善。

### 14.5 当前机制结论与停止条件

现有证据支持一个比“方差越接近 1 越好”更精确的机制：

1. 旧 RAE 存在跨单图空间统计、总体 mean 和 covariance 的广泛 endpoint
   收缩；LPL 恰好沿多个分布指标同时校正，因此可以改善生成。
2. RAEv2 的 internal guidance 已经在不提高 paired 准确度的情况下完成了
   大部分同类校准。
3. RAEv2 上继续施加 normalized LPL，会进一步优化它直接看到的单图方差和
   mean，却不能保持 population covariance；代理目标继续下降，但生成质量
   不再受益。

这仍不能证明 population covariance 是 FID 的唯一因果变量，且两套 decoder
不同，不能直接比较绝对数值。但 IG `1.0 -> 1.78` 是同一模型内的干净对照，
足以排除“RAEv2 根本没有同类 endpoint calibration”这一解释。

因此停止继续延长普通 LPL 训练。若以后重启这条路线，新的方法必须直接约束
batch-level endpoint distribution，并预注册“不能以牺牲 covariance coverage
换取单图方差接近 1”；否则只是重复当前已被终验否定的代理目标。

完整结果：

```text
~/data/eqvae/experiments/endpoint_feature_distribution_atlas/
```
