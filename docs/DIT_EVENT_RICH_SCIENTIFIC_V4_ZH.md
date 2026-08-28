# DiT event-rich scientific v4.1：同一模糊评估总体上的 B / E 检验

## 先说结论

scientific v4.1 在任何真实 event-screen 样本产生前，正式取代了未执行的 v3 B/C
设计以及第一版不可变 v4 锁。第一版 v4 继承了一句“供 candidate C 使用”的旧 rubric
文字，而且 zero-screen 审计只覆盖旧 launcher 前缀；它没有授权或执行采样，但仍作为
审计历史完整保留。v4.1 删除这处语义矛盾，同时覆盖新旧两个输出前缀。v1–v4 都不能
覆盖，也不能被当成 v4.1 执行许可。

v4 只保留两个正式候选：

- `B_persistence`：九个预终止检查点的内部局部模糊严重度均值。它是可预测的内部
  启发式量，不是鞅，也没有 Ville 语义。
- `E_blur_gated_running_max_log`：在 B 状态门和局部 mask 冻结后，对实际实现的
  `Q*/P` 路径似然比进行精确记账，取混合 e-process 的历史最大 log 值。它是
  operational e-process，但不等于理想热流边际密度比。

`C_c3_low_jump` 不参与类别选择、Holm、事件门、阈值、结论或干预；以后即使计算，
也只能明确标成无救援作用的事后诊断。

方法锁固定为：

`facef0f59d1f10cde339440db3bc47dc26ca7fcef012faca01f7f2dfbb31b985`

## 外部评价和内部方法必须分开

盲化视觉标签在 v4.1 中确实有两个实验设计用途：先用 discovery 的模糊/软融合事件数
构造一个六类别的高风险**外部评估总体**，再用独立 anchor 标签决定这个总体是否有
足够事件值得采内部轨迹。这会让最终结论只对“盲评筛出的六类高风险总体”成立，不能
外推成随机 ImageNet 或部署总体的结论。

但视觉标签绝不进入部署方法。B/E 的数值、无标签阈值、状态门、`Q*`、告警和未来
回滚决定都不能读取人工/模型盲评、endpoint 图、FID、Inception、DINO、CLIP、表征
距离或学习质量后验。FID 等若以后计算，也只能在所有内部分析锁定后作为批次级外部
验收，不能反过来选择类别、样本、候选、阈值或干预。

换句话说：盲评在这里定义“考卷考哪类题”和“答案对错”，不参与模型答题过程。

## 精确数据轴

| 阶段 | 固定轴 | 用途 |
|---|---:|---|
| wide discovery | 84 类 × seeds 1000–1011 = 1008 endpoint | 只按保留的 blur/soft-fusion clear-bad 数选择一个六类集合；平手按冻结 roster 顺序 |
| independent anchor | 6 类 × seeds 1012–1035 = 144 endpoint | 只检查模糊事件是否足以继续 |
| label-free calibration | 6 类 × seeds 1100–1119 = 120 trace | 冻结 B 阈值、B 状态门，并检查 E 是否真正打开门且使用到 KL 预算 |
| confirmation | 6 类 × seeds 1200–1327 = 768 trace/endpoint | 在标签和候选产品分别锁定后检验 B/E |

四组 seed 两两不交。endpoint sampler 保留 v3 的 DiT-XL/2、250-step ancestral
DDPM、CFG=4、pair-keyed singleton RNG 和完整 2B transition draw 语义；同一 global
seed 下的不同类别不共享初始或转移噪声。

## 两道“值不值得继续”的门

Anchor GO 恰好是以下三项的交集，不偷偷加入 Wilson 或模型指标：

1. 至少 6 个 blur/soft-fusion clear-bad；
2. 至少 3 个类别各自出现过这种事件；
3. 至少 60 个 clean-good。

Wilson 下界只作描述，不参与 GO，也不冒充 AUC 功效保证。anchor 失败就不采这一池
的 calibration/confirmation 内部轨迹。

在 confirmation 标签接触前，E 还有两道自身门：

1. 冻结方法锁中的 matched-Q oracle 功效门必须保持通过：两尺度最小 anytime power
   为 `0.320155`，预注册下限为 `0.30`，每尺度总 `K=2`；不得重调。
2. 在完整 120 条无标签 calibration 路径上，对每个尺度分别要求：至少 50% 路径曾
   打开 B 状态门；在打开门的路径中，至少 50% 使用的总 K 达到 1.5。

第二道门失败时，E 分支在接触 confirmation 标签前停止，正式 `p=1`，不得打开或
连接 E confirmation 分数。B 分支可以独立继续。

## 正式事件门与统计检验

Stage A 只读已经锁定的外部标签，不读任何候选值。B/E 共用同一事件门：

- 至少 15 个 blur/soft-fusion clear-bad；
- 至少 60 个 clean-good；
- 至少 3 个可比较类别，每类同时至少有一个上述 positive 和一个 clean-good。

失败时两个候选都设 `p=1`，两个分数产品都不打开。

Stage B 的正式统计量是 class-matched、pair-count-weighted、tie-aware ROC AUC。
positive 是保留的 blur/soft-fusion clear-bad，negative 是 clean-good；mild/disputed 和
非模糊 clear-bad 不进入 AUC。100,000 次完整 global-seed block 标签置换同时用于 B
和 E，Holm 家族恰好只有这两个候选。

B 必须同时满足：AUC≥0.75、Holm p<0.05、冻结 B 阈值处 TPR>FPR。

E 必须同时满足：所有前置门和精确性审计、AUC≥0.70、Holm p<0.05、固定
`alpha_e=0.10`（`E>=10`）处 TPR>FPR，以及至少 3 个 blur positive 越过阈值。
这里 `alpha_e` 控制的是实际 P 下总体 anytime 触发预算，不是好图条件 FPR。

## E 不能只靠“自己显著”过关

E 还必须证明它比 B 增加了信息：

`DeltaAUC = AUC(E_blur_gated) - AUC(B_persistence) > 0`

并且预注册的一侧 paired global-seed-block swap permutation `p<0.05`。此外，B gate
机制的固定消融要求：

`AUC(E_blur_gated) - AUC(E_no_state_gate) > 0`。

`E_no_state_gate` 只是消融，不进 Holm，也不能救活失败的 E。

只有 E 自身所有前置门、事件门、正式质量门、相对 B 的增量门以及 gate 消融符号门
全部通过，才允许进入一个**另行冻结、前瞻性**的 evidence-driven rollback 实验。
本 v4 不执行回滚。B 单独通过最多允许 B-specific 启发式探索，不允许声称鞅、Ville、
anytime-valid、似然比或分布扰动保证。

## 当前状态

冻结 v4.1 时，旧前缀 `dit_event_rich_endpoint_screen*` 与新 launcher 前缀
`dit_scientific_v4_endpoint_screen*` 下真实 endpoint/receipt/trace 文件数都为 0；该
事实由锁内 `pre_sampling_zero_audit.json` 固定。继承 rubric 中原来的
`frozen_before_third_pool_images_are_reviewed` 也被改写成明确的历史 lineage 说明；它不
表示任何 v4/v4.1 图片已经存在或被查看。

本科学锁的 `ready_for_real_sampling=false`。只有新的、不可覆盖的 v4.1-compatible
endpoint sampler 锁、盲评/双裁决锁、B/E dynamic evaluator 锁，以及真实独立 reviewer
资格输入都齐全后，才能另建一个执行授权 receipt。旧 v3/v4 源码锁不能授权 v4.1。
