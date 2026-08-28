# DiT event-rich scientific v4.2：固定方法 v2.2 后的确认协议

## 先说结论

scientific v4.2 在任何真实 event-screen 样本产生前，取代未执行的 v4.1。它不改变
外部评估总体、盲评标签、样本轴、B/E 共主检验族、主 AUC、标签置换或 Holm 校正；
只做三类必要修正：

1. 将 E 绑定到不可变的跨尺度路径证据方法 v2.2；
2. 将无标签机制检查放到用独立 calibration 阈值运行的全部 768 条 confirmation
   traces 上，并在打开任何 confirmation 标签或 endpoint review 前完成；
3. 删除不成立的“交换 B/E 分数身份”检验，改用成对 global-seed cluster bootstrap，
   同时加入不读创新的启动时序对照 `G_start`。

v2.2 方法锁的 canonical identity 是：

`cc4dc5e7c06c25f4d8567a42fb4f0387097a6296c587543830bfeaa4771f6921`

v4.1 锁继续原样保留，只是不能授权 v4.2。v4.2 冻结时仍然
`ready_for_real_sampling=false`；所有原来硬绑定 v4.1 的 endpoint、review、dynamic、
selector 和 self-test 产物都必须另建不可覆盖的 v4.2 绑定锁。

## 1. 什么保持不变

以下设计逐字段继承 v4.1，并由 freezer 自检拒绝任何漂移：

- wide discovery：84 类 × seeds `1000..1011`，共 1008 个 endpoint；
- independent anchor：选中的 6 类 × seeds `1012..1035`，共 144 个 endpoint；
- label-free calibration：同 6 类 × seeds `1100..1119`，共 120 条 trace；
- confirmation：同 6 类 × seeds `1200..1327`，共 768 条 trace/endpoint；
- discovery、anchor、calibration、confirmation 的 seed 集合两两不交；
- 同一个冻结 DiT-XL/2、250-step ancestral DDPM、CFG=4、pair-keyed singleton RNG
  及完整 `2B` transition draw 语义；
- 三名独立 reviewer、双裁决、clear-bad/clean-good 定义和 blur/soft-fusion endpoint；
- Anchor GO、Stage-A 事件门、类别选择规则与结论适用范围；
- 共主家族恰好是 `B_persistence` 与 `E_blur_gated_running_max_log`；
- 主统计量仍是 class-matched、pair-count-weighted、tie-aware ROC AUC；
- 主检验仍用 100,000 次完整六类 global-seed block 标签置换；
- Holm family alpha 仍为 0.05，B 的 AUC 门仍为 0.75，E 的 AUC 门仍为 0.70；
- B 的冻结阈值、E 的固定 `alpha_e=0.10`、以及 `TPR>FPR` 报告规则不变。

因此 v4.2 不是另选一个更容易的数据集，也不是看过标签后重写评价标准。它只把错误
的方法绑定和错误的增量检验修正掉。

## 2. v2.2 的 E 到底是什么

对每个尺度 `d∈{1,4}`，B 第一次超过独立 calibration 冻结的阈值、当前白化跨尺度
方向有效、并且从当前起至少还有三个有效检查点时，定义启动时刻 `T_d`。令包含当前点
的剩余有效点数为 `h_d`，启动时冻结

`κ_d = 2 / h_d`。

启动前 `Q_d^*=P`；启动后每个有效点都用跨尺度 score 差的白化单位方向，构造范数为
`sqrt(2κ_d)` 的均值位移。若某个后续当前方向低于固定数值 floor，则复用最近一次有效
单位方向。于是每个已启动尺度在路径上恰好具有

`K_total = Σ_k 0.5||u_{d,k}||² = h_d κ_d = 2`。

一步对数似然比为

`Δℓ_{d,k}=u_{d,k}^T z_k - 0.5||u_{d,k}||²`，

其中 `z_k` 是真实基线转移已经观测到的标准化创新，而 `u_{d,k}` 在抽取该创新前已经
由历史固定。故每个尺度分量以及固定混合

`E_k^mix = 0.5E_{1,k}+0.5E_{4,k}`

都是相对于实际实现基线 `P` 的操作性正鞅。`E_mix≥10` 的 Ville 语义是基线下总体
anytime 触发预算不超过 0.1；它不是 clean-good 条件 FPR，也不等于理想热流的边际
密度比。

方法 v2.2 的 matched-directional `Q*` 审计给出：只条件于一个合格的可预测启动历史，
终点 `log E_d~N(2,4)`。因为另一个分量非负，事件 `E_d≥20` 足以使混合越过 10，故

`P[N(2,4)≥log 20]=0.3092891983>0.30`。

这个数只说明“已经启动的操作性备择本身可检测”，不能乘以启动率后冒充总体质量功效，
更不能冒充坏图 TPR。

## 3. 为什么路径力学门要改

v4.1 在拟合 B 阈值的同一批 120 条 calibration traces 上查看 E gate/K 使用情况。
逐 checkpoint 阈值若在当前创新前确定，in-sample 回放未必破坏 Gaussian LR 的操作性
精确记账；但它没有 fresh-rank 语义，也不能作为确认性的机制 GO 或质量证据。

v4.2 因而先用 seeds `1100..1119` 冻结 B 阈值，再在 768 条尚未揭盲的 confirmation
traces 上运行方法。任何 confirmation 图片、标签、review、外部表征或质量分数打开前，
每个尺度必须同时满足：

- 至少 12 条路径合格启动；
- 已启动路径来自至少 3 个类别；
- 100% 已启动路径覆盖启动时冻结的全部 `h` 个正 KL 步；
- 启动后使用 last-valid direction fallback 的步数比例不超过 1%。

同时固定报告正 KL 步数直方图、KL 最大单步占比、KL participation-ratio 有效步数、
总 KL 分布、fallback 路径数和最长连续 fallback。按构造，完整启动路径应满足总
`K=2`、最大占比 `1/h`、有效步数 `N_eff=h≥3`。这些是实现与可启动性审计，不是
图像质量指标；机制门失败时 E 在揭盲前停止并令确认性 `p=1`，B 可以独立继续。

## 4. 三个对照分别隔离什么

`E_no_state_gate` 去掉 B 越界要求，但保留方向有效性、锁存、固定信息量和混合规则。
主 E 必须在确认集上优于它，才能声称 B-start 机制增加了价值。

`E_first_hit_full_budget` 在同一个可预测首次合格点一次花完 `K=2`，随后恒等。它是
另一个精确、等总预算的 one-shot alternative，并不是“把主 E 在第一步停止”：主 E
第一步只花 `2/h`。它只界定多步相对 one-shot 的诊断性结论，不能救活主 E，也不是
进入回滚的硬门。

`G_start_schedule_diagnostic` 完全不读任何转移创新，只读两尺度是否启动和启动时剩余
长度：

`G_start = 0.5·1[T_1<∞]·h_1/5 + 0.5·1[T_4<∞]·h_4/8`。

分母 5 与 8 是两尺度各自的最大有效检查点数。若 E 不优于 G，那么观察到的区分能力
可能只来自“B 是否、以及多早触发”，而不是路径创新和跨尺度方向的逐步对齐；此时
不能声称路径似然比增量提供了额外信息，也不能进入 evidence-driven rollback。

## 5. 为什么不能再做 score-identity swap

B、E、G 和两个 E 消融是同一路径上的不同函数，不是随机分派且可交换的 treatment。
把两列分数逐样本互换，不能产生增量 AUC 的有效零假设分布。v4.2 删除这种比较，固定
使用 paired global-seed cluster bootstrap：

1. confirmation 有 128 个 global seed，每个 seed 对应完整 6 类的 outcome 与全部
   配对分数；
2. 每次有放回抽取 128 个完整 seed block，block 内六类和各分数的配对关系不动；
3. 对左右两个分数分别重算同一个 class-matched、pair-count-weighted、tie-aware AUC；
4. 计算 `ΔAUC=AUC(left)-AUC(right)`；若某次没有可比较正负对，保守记 `-1`，不丢弃
   也不重抽；
5. 固定运行 100,000 次，排序后取零基第 4999 个值作为单侧 95% lower bound，不插值；
6. 硬门要求 observed `ΔAUC>0` 且 lower bound `>0`。

四个比较及 PCG64 seed 固定为：

| 比较 | seed | 作用 |
|---|---:|---|
| E − B | 2026082811 | E 增量主张与未来回滚硬门 |
| E − `E_no_state_gate` | 2026082812 | B-start 机制主张与未来回滚硬门 |
| E − `G_start` | 2026082813 | 路径 LR 增量主张与未来回滚硬门 |
| E − `E_first_hit_full_budget` | 2026082814 | 多步相对 one-shot 的诊断边界；不是回滚硬门 |

这些 bootstrap 比较不进入 B/E 的两个成员 Holm 家族；原来的主标签置换与 Holm 设计
保持不变。

## 6. 内外边界和停止条件

内部方法只能读取 latent、当前与跨尺度模型输出、pred-xstart 草图、冻结 B 阈值、
转移方差以及随后实际观测到的创新。endpoint 图片、人类或模型标签、FID、Inception、
DINO、CLIP、表征距离和质量后验不得进入 B、启动、方向、阈值、E 或任何干预决定。

外部盲评只在内部产品、机制门和来源身份全部锁死之后，用于判断 B/E 是否真的在清晰
定义的 blur/soft-fusion bad cases 上更高。它是裁判，不是方法的一部分。

只有主 E 依次通过方法 v2.2 的两个预先模拟审计、768 路径机制门、Stage-A 事件门、
E 自身 AUC/Holm/固定阈值门，以及 E−B、E−G、E−no-state-gate 三个 bootstrap 硬门，
才允许另行冻结一个前瞻性回滚实验。本协议不执行回滚，也不证明图像质量已经改善。

## 7. 冻结时的实际状态

v4.2 冻结前对 `dit_event_rich_endpoint_screen*` 与
`dit_scientific_v4_endpoint_screen*` 两类路径做了只读文件系统审计；真实 screen
endpoint、trace、review、score、embedding 或 label 的计数仍为 0。当前只有理论、
模拟 self-test 和不可变协议，没有真实质量结果。

在以下新锁全部完成以前，不得开始真实采样：

- v4.2-bound endpoint sampling source lock；
- v4.2-bound blind-review source lock；
- v4.2 / method-v2.2-bound dynamic、isolated-product、evaluator 与 bootstrap source lock；
- v4.2-bound selector 输出与 scientific self-test；
- 独立 reviewer 资格与 reserve 输入；
- 绑定所有新 identity 的不可覆盖 execution-authorization receipt。

这条限制避免把 v4.1 文件简单改标签后冒充 v4.2 可执行实现。
