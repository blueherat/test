# LPL 路线结题审计与下一阶段研究议程

更新日期：2026-07-31

## 0. 决策摘要

当前应当**停止把普通 LPL 继续做成方法**，但不应删除实验、代码或把旧 RAE
上的正结果说成无效。更准确的决定是：

1. 结束在成熟 RAEv2 上继续扫描 LPL 的权重、训练步数、归一化方式和 guidance
   scale。
2. 保留 LPL 作为一个已经理解得较深的**机制探针**：它可以稳定改变 decoder
   所见的 endpoint feature，但这种局部改变不保证改善最终生成分布。
3. 下一阶段只研究一个问题：

> **能否由模型干预前的 endpoint 分布校准状态，预测 decoder-aware 辅助损失
> 对最终生成质量的作用方向？**

这比“继续改 LPL”更 solid。它直接来自旧 RAE 成功、RAEv2 失败、internal
guidance 自身校准、以及单图统计与总体协方差分叉这四组相互制约的证据。

这条新路线也有严格停止条件：如果在同一架构的多个训练成熟度和 guidance
状态上，干预前的校准状态不能预测干预后的 FID 方向，就停止，不再把当前
跨模型差异包装成机制。

## 1. 从哪里开始，最终查到了什么

最初问题来自两阶段生成的接口错位：latent prior 用 latent MSE/Flow loss
训练，而最终图像由冻结 decoder 产生。LPL 的出发点是让 prior 同时看见
decoder 的中间特征。原始 LPL 工作也正是以“prior 与 decoder 脱节”为动机，
并在多种数据和生成范式上报告了 6%--20% 的 FID 改善
([LPL, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/204fee94c982a19230c39045aa54f977-Abstract-Conference.html))。

本仓库的实验经历了四次结论收缩：

1. **先确认有效性。** 旧 RAE 上 LPL 跨 DINOv2-B、MAE-B、SigLIP2-B 和多个
   seed 有效，不是单一 checkpoint 的偶然结果。
2. **再否定直觉机制。** LPL 没有降低 latent RMS，也没有把误差移入 decoder
   的局部低敏感方向；真实改善发生在有限误差半径后的非线性 decoder feature。
3. **然后发现模型代际分水岭。** 相同思路在 RAEv2 上不再带来有意义提升，
   strict LPL 甚至会显著伤害完整模型。
4. **最后定位到校准层次的分叉。** 旧 RAE 的 LPL 同时修复单图空间方差、
   总体均值和总体协方差；RAEv2 的 internal guidance 已完成大部分同类校准，
   额外 LPL 继续改善单图方差和均值，却开始损害跨样本协方差。

因此，当前最重要的发现不是“LPL 好”或“LPL 坏”，而是：

```text
局部配对目标下降
!= 单图 feature 统计更像 clean
!= 跨样本 endpoint 分布更像 clean
!= 完整采样质量提高
```

旧 RAE 恰好让这些方向大体一致；RAEv2 把它们分开了。

## 2. 证据总表

### 2.1 已经可以当结论的事实

| 事实 | 关键证据 | 结论强度 |
|---|---|---|
| 旧 RAE 上完整 LPL 有真实生成收益 | DINOv2-B 50k ADM FID `13.5043 -> 11.1027`；MAE-B、SigLIP2-B 50k 分别改善 `7.72%/5.53%` | 强 |
| 收益不是多训练几步造成的 | 等步数 Flow、额外计算预算 Flow 和官方起点对照均未解释收益 | 强 |
| 收益不是 latent 预测更准 | 递归探针中 LPL latent RMS 比 Flow 稳定差 `0.27%--1.57%` | 强反证 |
| 收益不是局部低敏感方向 | 同范数局部实验中 LPL 方向反而更敏感 | 强反证 |
| 旧 RAE endpoint 存在广泛 decoder-feature 收缩 | LPL 令空间方差误差 `-49.5%`、均值误差 `-28.0%`、协方差误差 `-4.9%`、Fréchet `-19.6%` | 强描述证据 |
| RAEv2 strict LPL 确实失败 | IG=1 的同噪声 5k：official `11.5240`、Flow100 `11.6942`、LPL100 `12.4807` | 强 |
| RAEv2 失败不只来自 guidance 外推 | `IG=1` 只用 full head 时 LPL 仍比 matched Flow 差 `0.7865` FID | 强反证 |
| full 与 contrast 都会传递伤害 | 2x2 head swap：只换 full 为 `+0.4092`，只换 contrast 为 `+0.1583`，两者都换为 `+0.9204`（1k 筛查） | 中强，1k 只解释机制 |
| common adapter 消除了大退化但没有实用收益 | 3 seed x 5k：step10 平均差 `-0.0096`，step50 平均差 `-0.0106`，远小于约 `0.12` 的 seed 波动 | 强否定实用收益 |
| LPL 在 RAEv2 上确实优化了自己的目标 | 固定 validation 配对上的 raw/normalized feature loss 随训练下降 | 强 |
| internal guidance 本身在做总体校准 | 同一 source 从 IG `1.0 -> 1.78`：空间方差误差 `-38.3%`、均值误差 `-35.3%`、Fréchet `-20.1%`，但 paired MSE 与 cosine 变差 | 强因果证据 |
| RAEv2 多训 LPL 出现统计分叉 | step50 的 mean/Fréchet 改善，但 population covariance error 恶化 `6.5%--7.5%` | 强描述证据 |

### 2.2 仍然不能声称的内容

- 不能说 LPL 普遍无效。旧 RAE 和原论文都存在可靠正结果。
- 不能说 population covariance 是 FID 的唯一原因。atlas 只是低维投影后的
  机制指标，且旧 RAE 与 RAEv2 使用不同 decoder。
- 不能说 RAEv2 失败只由 dual-head/internal guidance 引起。IG=1 对照已经
  证明 full vector field 本身也会被伤害。
- 不能说 common adapter 证明“只要保持 contrast 就安全”。它同时改变了
  可训练参数量、损失归一化和 LPL 剂量。
- 不能用 5k FID 的 `0.01` 级差异宣称提升；这远小于当前 seed 波动。
- 不能把“decoder feature 更像 clean”直接等同于“生成分布更好”。RAEv2
  已经给出反例。

## 3. 为什么应该结束普通 LPL 方法线

### 3.1 不是实现错误、泄漏或冻结边界错误

现有严格审计已覆盖：

- Flow/LPL 首批 image、label、latent、noise、time、CFG mask 哈希一致；
- train 与 ImageNet validation/FID reference 分离；
- encoder、decoder 和 source Stage-2 均冻结；
- online/EMA、checkpoint continuation、采样标签和 RNG fingerprint 有记录；
- common adapter 对 `full-base` 的数值改动不超过 `9.54e-7`；
- LPL 目标在 held-out 配对上确实下降。

所以当前结果的核心不是“LPL 没训到”，而是“训到的东西不是部署时真正需要
改善的东西”。

### 3.2 继续常规扫描不会回答新问题

已经尝试或实质覆盖的自由度包括：full/detach/raw、guided/full 监督、
full/base 两头交换、IG=1/1.78、短训/续训、全模型/common adapter、EMA/online。
继续调权重或训练到更多 step，最多找到一个局部数值，无法解释为什么旧 RAE
成功而 RAEv2 失败，也很难形成可信的新颖性。

### 3.3 “直接改成分布损失”也不能自动成为论文

相关边界已经很拥挤：

- diffusion exposure bias 已被系统定义为训练输入与采样状态不匹配
  ([Elucidating Exposure Bias, ICLR 2024](https://arxiv.org/abs/2308.15321))；
- Distributional Diffusion 已用 proper scoring rule 学习条件后验而不只是均值
  ([ICML 2025](https://proceedings.mlr.press/v267/de-bortoli25b.html))；
- GMFlow 已直接预测动态 Gaussian mixture 而非单一均值
  ([ICML 2025](https://proceedings.mlr.press/v267/chen25cl.html))；
- 感知目标本身已有 LPL 和 self-perceptual diffusion
  ([Diffusion Model with Perceptual Loss](https://arxiv.org/abs/2401.00110))；
- distribution matching distillation 已直接在生成样本分布上优化一阶段模型
  ([DMD, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Yin_One-step_Diffusion_with_Distribution_Matching_Distillation_CVPR_2024_paper.html))；
- guidance 的时间区间和动态强度已有系统研究
  ([Limited Interval Guidance, NeurIPS 2024](https://openreview.net/forum?id=nAIhvNy15T))。

因此，下列方向不建议单独作为主线：再做一种 feature MSE、加入 MMD/SWD、
扫描 guidance schedule、把 LPL 改成 on-policy、或泛泛声称“预测分布优于预测
均值”。它们可以成为对照，但不是当前证据自然推出的独立创新。

## 4. 当前最有价值的机制模型

RAEv2 的部署输出可写成：

```text
u_s(x,t) = full(x,t) + (s - 1) * [full(x,t) - base(x,t)]
```

令后训练造成的变化为 `delta_full` 和 `delta_diff`，则部署场变化为：

```text
delta_u_s = delta_full + (s - 1) * delta_diff
```

这解释了两个现象：

1. 即使 `s=1`，只要 paired auxiliary loss 改坏 `full`，生成仍会退化；这与
   IG=1 实验一致。
2. 当 `s>1`，差值方向的变化会被放大，并通过整个 ODE rollout 与状态变化
   产生非线性交互；这与 2x2 head swap 的额外交互项一致。

但更根本的断层不只在双头。局部训练优化的是：

```text
R_pair(theta) = E ||phi(z_hat_theta(x_t,t)) - phi(z_0)||
```

部署关心的却是从噪声出发、反复查询变化后的向量场后得到的总体分布：

```text
p_theta(z_endpoint) -> decoder -> p_theta(image)
```

`R_pair` 的下降并不决定总体 endpoint distribution 的变化方向。旧 RAE 中，
paired feature 修正恰好同时改善总体均值和 covariance；RAEv2 中，这些方向
不再一致。**这才是所有实验共同指向的核心。**

## 5. 候选研究方向的客观排序

| 方向 | 与现有证据的贴合度 | 新颖性风险 | 资源风险 | 建议 |
|---|---:|---:|---:|---|
| 校准状态能否预测辅助损失收益的正负 | 高 | 中 | 中低 | 主线 |
| guided 生成模型的 deployment-field 后训练稳定性 | 高 | 中 | 中高 | 主线成功后发展方法 |
| 直接 batch-level covariance/MMD/SWD loss | 中 | 高 | 中 | 只作 baseline |
| 更复杂的 LPL 归一化、层权重或 schedule | 低 | 高 | 中 | 停止 |
| 继续扫描 IG scale/区间 | 低 | 高 | 低 | 只作控制变量 |
| decoder Jacobian/pullback metric | 低 | 高 | 高 | 已被局部机制反证，不做主线 |
| 预测完整条件分布而不是均值 | 中 | 高 | 高 | 已有强相关工作，不从这里起步 |

我最推荐的单一研究问题是：

> **同一个 decoder-aware auxiliary loss 的收益符号，是否由干预前的 endpoint
> calibration deficit 决定？Guidance 是否会改变这个 deficit，从而让同一个
> loss 从有益变成无效或有害？**

这里 LPL 只是固定干预，不是论文贡献。旧 RAE 与 RAEv2 是自然形成的两端，
而真正需要补的是中间的连续证据。

## 6. 可执行的研究计划

### Phase 0：正式封存普通 LPL

- 不删除 checkpoint、采样、审计脚本和 endpoint atlas。
- 给普通 LPL 方法线标记 `closed_as_method`。
- 不再启动更多普通 LPL 长训练。
- 后续引用旧结果时必须同时报告 matched Flow、采样 seed 数和 5k/50k 级别。

验收：仓库中有唯一结题文档；旧账本指向本文件；没有运行中的普通 LPL 作业。

### Phase 1：在同一模型内制造“校准成熟度轴”

跨旧 RAE 与 RAEv2 比较仍混入架构和 decoder 差异。第一项新实验必须在同一
模型内做连续控制：

1. 选择 RAEv2 同一 source checkpoint。
2. 使用 `s = 1.0, 1.2, 1.4, 1.6, 1.78`，并可加入 official guidance interval
   的开/关作为第二个独立轴。
3. 每个状态在干预前运行固定 endpoint atlas；不看后续 FID 后再修改指标。
4. 从完全相同权重出发，施加同样预算的 decoder-aware 干预与 matched Flow。
5. 先用 3 seed x 5k 筛查，只对预注册通过的点做 50k。

主指标分三层报告，禁止合成一个事后调权的总分：

- paired：raw feature error、centered cosine；
- within-image：spatial variance log error；
- population：mean error、covariance relative error、Fréchet、SWD；
- 最终：matched `delta FID/KID/precision/recall`。

注意：guidance scale 会直接改变 baseline FID，所以研究对象不是不同 scale 的
绝对 FID，而是每个 scale 内 `auxiliary - matched Flow` 的差值。

### Phase 2：沿训练成熟度做真正的因果检验

仅改变 guidance 可能仍只是采样校准。需要第二条轴：同一架构的早期、中期、
成熟 checkpoint。优先使用官方连续 checkpoint；若没有，再在小型系统上训练，
不直接重训大型 RAEv2。

建议受控系统：ImageNet-64 或 CIFAR-10 上的小型双头 diffusion/flow。保留
确定性 decoder 或直接在 pixel endpoint 建 atlas，训练成本控制在可重复的
3--5 seed 范围内。

每个成熟度只问三件事：

1. 干预前是否存在 mean、variance、covariance 的共同收缩？
2. 固定小剂量干预后，三个统计方向是共同改善还是开始分叉？
3. 干预前的状态能否在不看 FID 的情况下预测 `delta FID` 的正负？

### Phase 3：只有机制通过后才做方法

若 Phase 1--2 成功，方法不应再叫改进 LPL，而应解决 deployment calibration：

- **calibration gate**：只有检测到广泛共同收缩时才开启辅助目标，并在
  population covariance 开始恶化前停止；
- **deployment-field trust region**：直接限制采样实际使用的组合向量场变化，
  分别约束 common 与 differential 分量；
- **generated-state validation**：训练仍可用 paired data，但停止/权重选择必须
  由固定 generated-state endpoint 指标决定。

第一版方法只选其中一个，不把三个模块拼接。优先顺序是 calibration gate，
因为它最贴近现有证据、参数最少，也最容易被严格证伪。

## 7. 预注册验收与停止标准

### 7.1 机制路线继续的最低门槛

必须同时满足：

1. 至少 `8` 个独立 operating points，覆盖两个校准轴，而不只是旧 RAE 与
   RAEv2 两点。
2. 干预前指标由固定 protocol 计算，不能使用测试 FID 反向拟合权重。
3. “广泛共同收缩 -> 干预有益；统计已校准或 covariance 分叉 -> 干预无效/
   有害”的方向至少在 4/5 seed 或两个模型族中重复。
4. 预注册的单指标或判定规则与 matched `delta FID` 的 Spearman 相关绝对值
   至少 `0.6`，bootstrap 置信区间不跨 `0`。
5. leave-one-model/checkpoint-out 后仍能正确预测至少 `75%` 的收益符号。
6. 50k 终验中相对 matched continuation 至少改善 `3%` FID，且 KID、precision、
   recall 不出现明显反向恶化，才称为实用方法收益。

### 7.2 任一出现就停止

- 校准指标只能在看过 FID 后通过调权得到相关性；
- 关系只存在于旧 RAE 与 RAEv2 的跨架构两点比较；
- 5k 有方向但 50k 消失，或效应小于 seed 波动；
- paired/atlas 指标持续改善，但无法预测最终 FID 符号；
- 小模型与 RAEv2 的作用方向系统性相反且无法由预注册变量解释。

这时应把本工作收束成严谨的 negative study，而不是继续发明 loss。

## 8. 一篇 solid 论文需要什么

若目标是 ICLR 级别，当前材料还不够。最少需要四块互相闭合的证据：

1. **现象：** 同一辅助目标在不同校准状态下发生稳定的收益符号反转。
2. **因果：** 在同一架构、同一数据、同一训练轨迹上改变成熟度或 guidance，
   排除 decoder/模型大小混杂。
3. **解释：** 用 paired、within-image、population 三层指标解释何时一致、何时
   分叉，并用 full/contrast 对照解释 guided 部署场的放大机制。
4. **预测或方法：** 干预前就能预测收益符号，或者用一个简单的 gate/trust
   region 在至少两个模型族上稳定避免退化并带来实用收益。

最可能成立的论文主张不是：

> 我们提出了比 LPL 更好的 perceptual loss。

而是：

> **Decoder-aware paired objectives 只有在修复真实 endpoint distribution
> deficit 时才改善两阶段生成；成熟模型和 guidance 会改变这一 deficit，导致
> 局部目标与总体生成质量解耦。我们给出可预测的适用边界。**

这项主张与已有工作有清楚边界：LPL 证明感知目标可能有用；exposure-bias
工作研究训练/采样状态错位；distributional diffusion 与 GMFlow 改变预测分布；
AutoGuidance 研究弱模型差值如何改善生成
([AutoGuidance, NeurIPS 2024](https://arxiv.org/abs/2406.02507))。在本次检索的
主要原始文献中，尚未看到工作系统研究“decoder-aware 后训练收益的符号如何
由干预前 endpoint 校准状态决定”。这是当前最值得验证、但还不能提前宣称
成立的 gap。

## 9. 现有资产如何保留

不要删除下列资产：

- 旧 RAE 正结果与机制审计：`docs/RAE_LPL_RESEARCH_AUDIT_ZH.md`；
- RAEv2 双头、continuation、common adapter 与 atlas：
  `docs/RAEV2_SELF_GUIDANCE_RESEARCH_LEDGER_ZH.md`；
- 全路线解释：`docs/RAE_RAEV2_LPL_DEEP_SYNTHESIS_ZH.md`；
- endpoint atlas 汇总脚本与测试；
- 50k 终验结果和 checkpoint metadata。

本地完整 atlas 位于：

```text
~/data/eqvae/experiments/endpoint_feature_distribution_atlas/
```

LPL 后续只作为固定 probe 出现在 Phase 1--2；其权重、层选择、训练预算全部
冻结，不再作为可调方法搜索空间。这样旧实验不是被丢弃，而是从“候选方法”
转化为研究生成目标错位的一个高质量干预工具。

## 10. 下一项最小行动

下一项实验不是再训练一个 LPL checkpoint，而是先建立一个无需训练的
`calibration operating-point table`：对同一 RAEv2 source 的多个 IG scale，
用同一批 latent、noise 和 endpoint-query protocol 计算完整 atlas，并输出
每个 scale 相对 `IG=1` 的 paired/within-image/population 三层变化。

只有这张表显示出连续且可重复的校准轴，才启动固定 10-step 的 matched
Flow/LPL 干预。若连连续校准轴都不存在，Phase 1 立即停止，可以避免再次进入
昂贵而盲目的长训练。
