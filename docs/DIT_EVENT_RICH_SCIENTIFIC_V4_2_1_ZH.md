# DiT event-rich scientific v4.2.1：收紧 G 的解释边界

## 结论

v4.2.1 在任何真实 event-screen 数据产生前，以不可覆盖方式取代刚冻结但尚未执行的
v4.2。v4.2 原锁完整保留。此次不改方法 v2.2、不改样本、不改标签、不改 B/E 共主
家族，也不改任何原有通过门；只修正一句过强解释，并增加一个纯描述诊断。

需要修正的是：

`G_start = 0.5·1[T_1<∞]·h_1/5 + 0.5·1[T_4<∞]·h_4/8`

只是预先固定的一维启动摘要。`E>G_start` 只能说明 E 的排序能力超过这个特定标量，
不能说明已经排除所有可能的启动日程信息，更不能因果证明“创新与跨尺度方向对齐”是
提升来源。允许的 28 种 `(h_1,h_4)` 组合虽然只在 `(0,8)` 与 `(5,0)` 处发生一次
G 数值碰撞，但一个固定的序数分数仍不是任意非单调、交互式 schedule-only predictor。

因此 v4.2.1 保留 E−G 的 paired cluster-bootstrap 硬门，但把它严格命名为“超越这个
预注册标量 G 的增量”。它仍是未来另行冻结回滚研究的必要条件之一，却不授权更宽的
schedule-adjusted 或因果创新对齐结论。

## 保持不变的硬门

以下逻辑与 v4.2 完全一致：

- B 与 E 仍是唯一两个 Holm 共主候选；
- E 自身仍须通过 v2.2 预模拟审计、768 路径的无标签 mechanics、Stage A、
  class-matched AUC、Holm、固定阈值 TPR/FPR 和越界阳性数；
- E−B 使用 seed `2026082811` 的 100,000 次 paired 128-seed cluster bootstrap，
  是增量主张和未来回滚硬门；
- E−`E_no_state_gate` 使用 seed `2026082812`，是 B-start 机制和未来回滚硬门；
- E−G 使用 seed `2026082813`，只作为“超越预注册标量 G”的未来回滚硬门；
- E−`E_first_hit_full_budget` 使用 seed `2026082814`，仍然只界定 multi-step 相对
  one-shot 的诊断措辞，`required_for_rollback=false`，不能救活 E。

前三个硬比较均要求 observed `DeltaAUC>0` 且 100,000 个 bootstrap replicate 排序后
零基第 4999 个值，即单侧 95% lower bound，严格大于 0。one-shot 比较不进入回滚
conjunction。所有比较都保持完整六类 global-seed block 内的 outcome/score 配对关系。

## 新增：exact-schedule conditional concordance

为了在不夸大结论的前提下看得更细，v4.2.1 预注册一个不参与 GO、Holm 或回滚授权的
描述诊断。它只在 E 已通过全部标签前机制门、Stage A 也允许正式打开 E 产品时运行；
否则记为 `NOT_RUN`。对每条合格的 confirmation 路径定义精确分层：

`s=(class_id, scale1_start, scale4_start)`，

其中每个 `scale_start` 是精确的
`(start_time_index,start_remaining_effective_count)`；未启动必须同时满足
`start_time_index=-1`、`start_remaining_effective_count=0`，并记为 `⊥`。

在每个分层内，只比较 retained blur/soft-fusion clear-bad 与 clean-good 的 E 分数。
令该层阳性和阴性数量为 `n_s+`、`n_s-`，定义按可比较 pair 数微平均的 concordance：

`C = Σ_s Σ_{i∈positive_s,j∈negative_s}[1(E_i>E_j)+0.5·1(E_i=E_j)]
     / Σ_s(n_s+ n_s-)`。

没有同时包含正负样本的层对分子、分母都贡献 0；绝不把空层填成 0.5，也不对 cell
AUC 做 macro average。必须同时报告：

- exact-schedule 可比较 pair 总分母；
- 只按 class 分层时的可比较 pair 总分母；
- 两者之比，即 exact-schedule 的 class-only pair coverage；
- informative exact strata 数；
- informative classes 数；
- 至少向一个 informative stratum 贡献合格 positive 或 negative 样本的不同原始
  global seed 数；mild/disputed 或其他不合格行不得计入；
- exact-stratum 内 E 分数的 pair tie rate。

## 描述 bootstrap 的精确定义

该诊断固定使用 `numpy.default_rng(PCG64(seed=2026082815))`，运行 100,000 次：

1. 每次从 128 个 confirmation global seeds 中有放回抽 128 个完整六类 block；
2. block 内六类、outcome、E 分数和两尺度启动 metadata 始终一起保留；
3. 在重采样数据中重新构造 exact strata 并计算 C；
4. 若 replicate 的可比较 pair 分母为 0，将 C 记为 undefined；不填 0.5、不丢掉后
   重抽，并报告这种 replicate 的比例；
5. 若有 `M>0` 个 defined replicates，将 C 排序，描述性双侧 95% 区间取
   `max(0,ceil(0.025M)-1)` 与 `max(0,ceil(0.975M)-1)` 两个零基位置；若 `M=0`，区间
   报 null。

这里不产生 p-value、GO 或 pass/fail。即使区间很好，也最多支持：

> 在具有可比较正负对的同类别、同精确启动日程子集中，E 存在额外的排序关联。

它不是全 confirmation 总体估计，不是因果检验，也不能证明所有 schedule explanation
都已排除。若 informative strata、classes、seeds 或 pair coverage 很小，结果应直接按
稀疏的描述数据解释。

## 当前状态与执行边界

v4.2.1 冻结前，两种 event-screen 输出前缀下的真实 endpoint、trace、review、score、
embedding 和 label 计数仍为 0。v4.2.1 继续绑定 method v2.2 canonical identity：

`cc4dc5e7c06c25f4d8567a42fb4f0387097a6296c587543830bfeaa4771f6921`

所有 v4.1 或 v4.2 绑定的 endpoint、review、selector、dynamic、evaluator 和 self-test
来源都不能改标签复用，必须另建 v4.2.1 非覆盖锁。新的执行授权 receipt 产生前，
`ready_for_real_sampling=false`；本协议没有真实质量结果，也不执行回滚。
