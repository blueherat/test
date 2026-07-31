# RAEv2 Self-Guidance 研究账本

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

不再继续盲目增加普通 paired LPL step。下一项只问：

> 若直接在短递归产生的 self-induced states 上约束后续 endpoint，能否让
> decoder-feature 优势在 `1 -> 4 -> 16` 次查询中保持，而不是快速衰减？

这是一个由现有反常现象直接推出的 rollout-closure 实验，不是再叠加一个
无关 loss。只有它先改善 64 图 retention，并在至少两个 5k seed 上优于同
预算 Flow rollout control，才值得做更大规模；否则应停止 LPL 方法线，转而
研究 decoder feature metric 本身与生成感知质量的错位。
