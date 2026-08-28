# DiT bad/good 内部方法总账（2026-08-28）

## 0. 本文解决什么问题

本文只记录冻结 DiT-XL/2 ImageNet-256、250 步 ancestral DDPM、CFG=4 这条
bad/good 与选择性修复主线。较早的 RAE/训练研究不在本文混合讨论。

**暂停状态（2026-08-28）：** 本主线已按用户要求停止实验、下载和新候选搜索。本文只作
证据总账，不表示下列未完成试验仍在后台推进。仓库与外部数据的保留边界见
[`DIT_BAD_GOOD_PAUSE_ARCHIVE_2026-08-28_ZH.md`](DIT_BAD_GOOD_PAUSE_ARCHIVE_2026-08-28_ZH.md)。

这里的 `bad` 不是“不完美”，而是相对同一冻结模型的普通采样水平仍明显更差：主要
区域明显模糊或融合、肢体/物体明显错位、重复、断裂、错误附着或拓扑错误。轻微瑕疵、
普通模型质感、合理遮挡/裁切和只是少见的构图不能自动记为 bad。

方法与裁判严格分开：

- 方法只允许读取生成完成前的 sampler 内部量，例如 latent、`pred_xstart`、reverse
  mean/variance、实际 innovation 和事前冻结的反事实模型查询；
- 人工/模型盲评、FID、Inception、DINO、CLIP 只在内部选择封存后评价结果，不得进入
  feature、方向、阈值、前缀筛选、选枝或回滚决定；
- 一个量可以只对某个 subtype 有效。研究目标不是强迫一个分数识别全部 bad case，
  而是在明确适用范围内取得可复现的平均收益并守住伤害门。

## 1. 当前结论总览

| 方法 | 已观察到的最好结果 | 关键反证或限制 | 状态 |
|---|---:|---|---|
| 独立重采后缀 | 机会比较中严格修复 `27/71=38.0%` | 没有可靠内部触发/选枝量 | 动作可行，不能无条件部署 |
| 跨尺度路径 e-process `E` | 操作性 `P/Q*` 似然比为精确正鞅 | clear-bad `1/26` 对 control `2/26`；匹配修复 4 负、0 正、4 平 | 质量语义退役 |
| 中途局部模糊量 `B` | replication AUC `0.9333`；标签敏感性 AUC `0.668--0.848` | 第三池只有 6 个 clear-bad、4 个 blur，正式事件门未开 | 最强 blur/fusion subtype 候选 |
| c3 低结构跃升量 `C` | replication AUC `0.8631` | 标签敏感性 AUC `0.428--0.728`，通道特异 | 副候选/证伪对照 |
| posterior/variance/innovation/跨尺度清单 | 少数旧池事后 AUC 较高 | 独立复核多在随机附近或反向 | 不继续逐项修补 |
| 分支 medoid / high-`O` | 无 | medoid `5/18=27.8%`，随机 `37.5%`；high-`O` AUC `0.286` | `STOP_CURRENT_FORM` |
| `h=10` max-nonconformity | 事后 `10/18=55.6%`，随机 `37.5%` | 揭盲后反向发现、主要集中 class795；prospective 只有采样与内部选择完成，没有可引用的效能结论 | 多分支路线暂停/不作主线 |
| Doob consistency | 历史 discovery AUC `0.7014`、exact `p=0.0436` | leave-class795 AUC `0.5062`，效应集中在少数 blur | 降级为 blur-specific exploratory signal |
| PTCV | 事后 blur subgroup AUC `0.889` | 冻结 expansion AUC `0.4645`、exact `p=0.6060` | 退役，不允许 rescue |
| FPCV dense geometry | seed50 在 step99/149 有可重复非零循环违反 | 只有 label-free 数值 smoke；没有完整 cohort 或 quality join，step199 为零 | 未完成并暂停，不能称 detector |
| Fisher-geodesic guidance | 1,024 图 FID 相对 baseline `-0.152` | 正式 5K：official `7.4938`、Fisher `7.9868`，变差 `+0.4930` | 退役，不调系数/窗口/子集 |
| 官方 Self-Guidance | repo、prompt lock、评估资产和配置已准备 | SD1.4 权重下载未完成；0 张生成图、无 1K/5K FID | baseline 工程暂停 |

## 2. 后缀重采：动作有用，但动作不是检测器

从同一个保存前缀重新抽独立后缀，确实能修复一部分已经出现的模糊/融合。在三位隔离
盲评按旧协议确认“基线确有修复机会”的 71 个比较中，严格成功 27 个，约 `38.0%`。

严格成功同时要求：

1. 基线确有可见 blur/fusion 机会；
2. fresh 后缀明确减轻该缺陷；
3. 类别、身份、物体数、主要姿态和构图保留；
4. fresh 没有引入新的 clear-bad。

这个结果说明某些失败是采样事故，而不是模型能力的必然极限。但失败仍占多数，所以
“总是回滚”不是方法；核心问题仍是生成完成前如何判断前缀和值得保留的后缀。

## 3. 跨尺度路径证据 `E`：定理成立，质量桥断裂

基线一步写为

\[
x_{k+1}=\mu_k+\Sigma_k^{1/2}z_k,\qquad z_k\mid\mathcal F_k\sim\mathcal N(0,I).
\]

把操作性高噪声反事实均值与基线均值之差白化后记为 `u_k`，则

\[
\Delta\ell_k=u_k^\top z_k-\frac12\lVert u_k\rVert^2,
\qquad
E_k=\exp\!\left(\sum_{r\le k}\Delta\ell_r\right).
\]

只要 `u_k` 在抽取当前 `z_k` 前可测，就有

\[
\mathbb E_P[E_{k+1}\mid\mathcal F_k]=E_k.
\]

这项精确性只属于实际实现的两个 Markov chains `P/Q*`，不自动等于理想热流边际比
`p_{v+Delta}/p_v`，也不证明 `Q*` 更偏好的路径视觉上更坏。

主要反证：

- 1,740 条路径中有 26 条 `E>=10`；
- 新盲评的严格 clear-bad 为 alarm `1/26`、matched control `2/26`；
- blur/fusion 为 `9/26` 对 `5/26`，只有弱趋势；
- 匹配 `E+B` 与 `B-only` 的修复 pilot 中，稳定 opportunity path 为 `3/8` 对 `6/8`；
- 八个匹配对的严格修复差为 4 负、0 正、4 平。

因此 `E` 保留为严谨的路径假设记账工具，但退出坏图报警、guidance 和 rollback。失败的
不是鞅恒等式，而是“更像高噪声反事实就更坏/更可修”的经验语义。

## 4. 模糊量 `B`：目前最强的窄 detector 候选

在九个中途检查点，把模型预测的干净 latent `x0hat_{i,k}` 用冻结 VAE 临时解码。对有
内容的局部 tile，计算 Laplacian 二阶边缘能相对 Sobel 一阶边缘能的比例 `q_{i,k,j}`；
模糊边缘可能仍有缓慢的一阶梯度，但缺少陡峭二阶变化。单步严重度与整条路径分数为

\[
B_{i,k}=-\log\!\left(Q_{0.25}\{q_{i,k,j}:j\in\mathcal A_{i,k}\}+\varepsilon\right),
\qquad
B_i=\frac1{|\mathcal K|}\sum_{k\in\mathcal K}B_{i,k}.
\]

冻结 replication 中，大 `B` 预测 clear-bad 的总体 AUC 为 `0.9333`，三类分别
`0.8920/0.9561/1.0000`。固定 10% 操作档在该批抓到 `8/9` bad，同时告警
`23/304` clean-good。第三池因事件过少没有打开正式 score-label join；之后仅作的五种
既有标签口径敏感性分析给出 AUC `0.668--0.848`，方向均一致；最终裁决口径为
`0.6681`，固定 10% 阈值的 TPR/FPR 为 `0.25/0.0644`。这说明方向尚在，但早期
`0.9333` 的强度不能当成稳定复现。

正确定位是：`B` 可能已经足以成为明显 blur/fusion 的 subtype detector，不需要承担
六指、复杂拓扑和所有错位。限制是中途 VAE 解码较贵，连续 `B` 也没有很好匹配可见
修复机会，所以目前仍是表型启发式，不是自动回滚门。

## 5. c3 低跃升量 `C`：便宜但脆弱

对第 4 个 latent channel 的 `pred_xstart` 定义 alpha 补偿空间梯度能 `G_{i,k,3}`，
再取中段最大正跃升

\[
J_i=\max\!\left(0,\max_{k=100}^{148}
  [G_{i,k+1,3}-G_{i,k,3}]
\right),\qquad C_i=-J_i.
\]

直觉是普通轨迹常在中段有一次明显“结构长实”，而模糊/融合轨迹一直拖着不定型。
replication AUC 为 `0.8631`，但五种后续标签口径下只有 `0.428--0.728`，四通道平均
也没有复制；最终裁决口径 AUC 为 `0.4619`，固定 10% 阈值 TPR 为零。它便宜且有
动力学含义，但单通道特异性太强，当前只作副候选和反例控制。

## 6. 大清单中没有站住的量

已经系统检查过 predicted-clean 的幅度、粗糙度、相邻预测变化、状态/重建噪声控制、
reverse learned variance、白化 innovation 能量与空间集中、跨尺度 RMS/夹角/局部化、
原始 KL/LR、normalized posterior 拆分和稀疏组合。

代表性失败包括：

- Candidate-v5 跨尺度联合分数在 600 条轨迹的 AUC `0.5398`，单侧置换 `p=0.2931`；
  固定阈值命中 bad `0/17`，误报 good `23/520`；
- 每维 `log D` 最大正跳 replication AUC `0.2579`；
- tile concentration 波动 `0.4084`；
- effective-count 波动 `0.4740`；
- 标准化全局对齐 `0.5973`；
- 34 路 signed spatial mixture 的 within-prefix running-max AUC 曾为 `0.944`，但在
  64 个新 cross-prefix 上降到 `0.441`，冻结阈值 `0/64` 触发。

这些结果说明“坏轨迹普遍更像高噪声分支、variance 更怪或 innovation 更大”都不是
当前数据支持的统一机制。

## 7. 分支共识首版为什么失败

在同一个前缀上走四条短程 baseline-`P` scout，用共同前缀归一的多尺度
`pred_xstart` 距离衡量条件分支几何。首版预测：

- fresh 分支的 medoid 更可靠；
- 原延续相对 fresh 共识越离群，越像可修复采样事故。

冻结结果与预测相反：

- `h=10` medoid 严格修复 `5/18=27.8%`；同四条 scout 均匀随机为 `37.5%`；
- higher-`O` opportunity AUC 为 `0.286`；把方向翻成 low-`O=0.714` 只是同一失败量的
  代数翻转，不是独立证据。

所以不能继续讲“多数分支形成正确共识、当前坏延续偏离它”。

## 8. 已暂停候选：`h=10` early-horizon max-nonconformity

### 8.1 定义

在 step149 的共同前缀上，用同一冻结 `P` 独立生成四条 fresh scout。第 `m` 条分支完成
`h` 次转移后的 predicted-clean latent 记为 `U_m(h)`。令 `C` 为逐 channel 空间去均值，
`A_s` 为尺度 `s in {1,2,4}` 的 average pooling，共同 `h=0` prediction 为 `R`：

\[
r_s=\operatorname{RMS}(A_sC(R))+10^{-6},
\]

\[
d_h(m,n)=\frac13\sum_{s\in\{1,2,4\}}
\frac{\operatorname{RMS}[A_sC(U_m(h)-U_n(h))]}{r_s},
\]

\[
A_m(h)=\frac13\sum_{n\ne m}d_h(m,n),
\qquad
m^*=\arg\max_m A_m(10).
\]

并列由 GPU 输出前冻结的匿名 A--D slot 打破。该 selector 不读取 attempt0、endpoint、
`B/E/O`、图片、标签、FID 或 embedding。

### 8.2 直觉与边界

事后四个 rank 从中央到离群在 `h=10` 的严格修复数为 `[5,7,5,10]/18`。最离群分支
`10/18=55.6%`，相对随机 `+18.1pp`；但 `h=5` 只 `+6.9pp`，`h=20` 只 `+1.4pp`，
而且非机会样本 clear-bad/preservation 分别比随机差 `3.6pp/1.8pp`。

窄直觉是：若错误结构已经成为当前前缀下最容易延续的局部吸引盆，多数分支会继续
巩固同一错误；十步时最先离开该盆地的 minority branch 可能是有用 escape。

必须主动承认：`A_m` 不是坏图概率、p-value、e-value 或 likelihood ratio；大距离也
可能只是更怪。它不检测哪个 prefix 需要干预，而是每个 prefix 都从四个已计算未来中
选择一个。准确名称应是 `early-horizon outlier branch selector`；在只看 `h=10` 的主
检验通过以前，不能把“transient”或“consensus trap”写成已成立机制。

### 8.3 V1.2 前瞻检验

V1.2 在任何新后缀和外部评价打开前冻结：

- 128 个互不重复 prefix：class795 为 96，class207/602 各 16 个伤害哨兵；
- 固定 step149、四条 fresh `P` scout、`h=10`、上述 max-`A`；
- 512 个新 RNG stream 全部唯一，且与旧 suffix stream 零交集；
- 主比较是同四条 scout 的精确 uniform-policy value，计算预算一致；
- 内部选择产物必须先哈希封存，之后外部盲评才能打开 endpoint；
- class795 主效应为

\[
D_i=\frac14\sum_{m=1}^4L_{i,m}-L_{i,m^*},
\]

  其中 `L` 是封存后得到的外部缺陷严重度除以二。`D>0` 只表示内部策略比同四分支
  均匀随机更会选低缺陷 endpoint，`L` 从未进入方法；
- 窄 GO 至少要求平均 `D>=0.10`、冻结单侧检验通过、bootstrap 下界大于零、物理
  attempt 无异常偏置，并且 clear-bad excess 与 preservation loss 的单侧 95% 上界都
  不超过 5 个百分点。

这项实验只需证明平均选枝收益，不要求识别每个 bad case，也不要求每个 prefix 都获益。

截至 2026-08-28，GPU 执行已完整结束：`128/128` 个 prefix、`640/640` 个 endpoint
全部写出，没有失败、删样或补跑。随后在没有打开任何 prospective PNG 或外部指标的
条件下，由冻结 extractor 封存了 `128` 行内部选择；内部产物 identity 为
`1c2ee25f6ca7b07c6259a0d9abeb176df9e7b26a0413d197417d24a62d24c72e`，二次运行只做
exact validation 并得到同一 identity。

正式盲评代码也在看图前单独冻结。当前有效的是 blind-evaluation lock V1.1：

- lock identity：`42529439647c1647800ff7bdbed587c6e991d816ef7c1909d4ca0fcdc12db7db`；
- protocol identity：`4b72313de34148307add44593089c45e3ad67c8d4e6528e3bb895c4448927440`；
- analyzer SHA-256：`00fb8a7bb6fa556f2d1368a6dfe3c32397ec69c650915375d807bb56154a5224`；
- 任一正式 absolute/pair 票出现 `valid=no` 都由代码强制判为 `INCONCLUSIVE`；
- 三位最终评审均先通过旧冻结 anchor 的 `6/7` 资格门；一位初始候选仅 `5/7`，已在
  正式评审前淘汰，其答案不进入任何结果。

公开盲包 identity 为
`ad98b92621b2aabae2815271f8effc8c70df8f1a35395e809f2dda211f4c81af`：每位评审独立完成
`640` 个 absolute、`512` 个 attempt0-vs-fresh pair 和 `7` 个 qualification。暂停时可确认的
权威状态只有 prospective GPU 输出与内部选择已经封存；本文没有一份通过完整 reliability、
effect、index 和 safety gates 的最终效能结果。用户随后停止多分支方向，所以不得把“盲包已
生成”或“评审流程曾启动”改写为 selector 已验证。相关昂贵输出暂时保留，以便将来只做
审计或在明确恢复后完成原冻结评价。

四条 fresh scouts 条件于共同 prefix 按同一 `P` 独立同分布，`A_m` 又是 permutation-
equivariant 的对称函数；并列由与数组独立的均匀匿名 slot permutation 打破。因此在没有
实现索引偏差时，每个物理 attempt 被选中的边际概率应为 `1/4`。这只提供交换性和代码
完整性检查，不提供质量保证；质量收益仍只能由冻结后的 `D_i` 检验。

正式评估为了得到同四条 scouts 的精确 uniform-policy 反事实，会把四条全部跑到终点。
若以后验证通过，部署不需要四条都跑完：正常前缀已完成前 149 个转移，随后并行走
`4x10` 个短 scout，只把选中的一条继续余下 91 个转移，总计约 `149+40+91=280` 次
branch-transition，相对原 250 步多 30 次，名义 NFE 增幅约 `12%`；四 scout 若可 batch，
墙钟增幅还可能更小。这个计算结论不包含未来可能加入的 prefix trigger 或额外跨尺度查询。

## 9. 后验一致性与有限几何候选

### 9.1 Doob consistency

Doob consistency 检查 learned posterior prediction 在事前固定的小幅未来扰动下是否满足
条件期望的一致性。冻结历史 discovery 的 high-is-worse AUC 为 `0.701449`，独立精确
class-stratified randomization `p=0.043579`；这比简单 motion/energy controls 强。

但原 analyzer 没有完整落实“不得依赖单一类别”的冻结条款。揭盲后的敏感性审计显示，
leave-class795 AUC 只有 `0.506173`，而效应主要来自三张 global-blur bad case。权威重评是
`DOWNGRADE_TO_BLUR_SPECIFIC_EXPLORATORY_SIGNAL`，不是通用 detector，也不是 intervention
trigger。完整结果 identity 为
`6cb1228ef4d3657b2c6f7d33247438d0f0e5ff04b367cdb406856e955c5c8ce9`，独立审计 identity
为 `3cf77bb05db43e2d82d52dfdcbb1c986358e9a08a1077edd6e4d2f04a780c14a`。

### 9.2 PTCV

Projected Tweedie Cone Violation（PTCV）测量 projected posterior-mean Jacobian 偏离对称
半正定锥的程度。generic discovery 的主 AUC `0.629`、`p=0.125`，没有通过冻结主门；
揭盲后 global-blur subgroup 曾出现 AUC `0.889`，因此只允许一次 formula、方向、三类和
endpoint 全部预先固定的 expansion 验证。

该扩展在 9 blur clear-bad 与 304 clean-good 上得到 AUC `0.464547`，精确 class-stratified
单侧 `p=0.606019`，bootstrap 95% 区间约 `[0.219,0.708]`。正式决定是
`STOP_PTCV_BLUR_SPECIFIC_HYPOTHESIS_NO_RESCUE`，结果 identity 为
`f18e8b7c349cceb657fd73f719d5d0437ec29a2cd015f669b11bfd8aa8e8490c`。不得通过删类、
换 checkpoint、换符号或换 blur 定义救回。

### 9.3 FPCV

Finite Posterior Cyclic Violation（FPCV）把理想 Gaussian posterior mean 必须是 convex
gradient 的硬结构写成有限点 assignment：正确输入输出配对应最大化总内积。原冻结
33-point cross-polytope 在 seed50--52、多个 checkpoint/radius 上全部 identity-optimal，
数值上空洞。

随后只做了不看 endpoint、PNG、embedding 或质量标签的 geometry smoke。固定 128 点闭合
多边形能检出原 cross-polytope 会漏掉的微小旋转分量；seed50 在 step99/149 出现完全可重复
的非零违反，重复查询逐元素最大差为 `0`，而 step199 回到零。这只证明 dense geometry
不是恒零量。它尚未冻结唯一 redesign primary、运行完整 240-path cohort 或执行任何
score-label join，因此没有 AUC、TPR/FPR 或干预结论。该线在此状态暂停。

## 10. Fisher-geodesic 与 Self-Guidance baseline

Fisher-geodesic guidance 把 strong/weak denoiser gap 当成一维干净图坐标，用 reverse-mode
directional precision 构造两个 checkpoint 的后验方差，再沿 univariate Gaussian
Fisher--Rao geodesic 外推。1,024 图筛查曾给出 official `40.678659`、FGCE `40.526861`、
time-only null `40.560760`；由于 FGCE 只比 time-only 多改善 `0.033899`，该结果只授权原样
扩到 5K，不能先称有效。

正式 paired 5K 已经给出相反结论：official AutoGuidance FID `7.4938135013`，FGCE
`7.9868453414`，变差 `+0.4930318401`；time-only null 为 `8.0037608086`。FGCE 虽比
time-only 好 `0.016915`，但远不足以抵消相对 official 的系统性均值/协方差错位。正式决定
是 `retire_exact_FGCE_after_fid5k`：不得再调系数、时间窗口、clip、样本子集或与 official
混合来 rescue。FID-5K 仍只是筛查，不能与论文 FID-50K 横比。

Self-Guidance 当前只有官方 repo `843bda799bb5`、冻结的 1K COCO prompt lock、release-
faithful Euler-50 配置、T2IBenchmark COCO-val2014 FID stats 和评估依赖。SD1.4 指定 revision
的权重下载被用户中止且缓存不完整；没有生成任何正式图，也没有 1K/5K FID。因此它是
“可恢复的 baseline 工程快照”，不是已复现结果。

## 11. 暂停边界与若恢复时的纪律

当前没有活动实验。若用户以后明确恢复，顺序最多是：

1. 对明显 blur/fusion 重新设计事件充足、reviewer 先过 anchor 的 `B` 窄确认；
2. 只有固定 `B` 在新数据上证明 `TPR>FPR`，才测试一次性完整独立重启；
3. FPCV 只有在愿意接受一次唯一 score、完整 cohort、无 rescue 的 quality test 时才恢复；
4. Self-Guidance 只有在用户主动恢复权重下载后才继续；
5. `C` 只作便宜副量，`E`、PTCV、Fisher 和多分支主线均不再修补。

下面四类是多分支路线暂停前留下的未验证设计草图，不是当前待办，也不是实验授权。若
未来重开该路线，仍不能在同一外部标签上从 `h=5/20`、min-`A`、类别切片或外部表征中
选赢家；每一类都必须在全新数据上单独预注册。

### 11.1 one-vs-three 暂态逃逸脉冲

当前 max-average distance 无法区分“一条分支脱离另外三条”和“两条对两条分裂”。先定义

\[
I_m(h)=\min_{n\ne m}d_h(m,n)
-\operatorname{median}_{a<b,\,a,b\ne m}d_h(a,b),
\]

再定义

\[
T_m=2I_m(10)-I_m(5)-I_m(20).
\]

只有当第 `m` 条在 `h=10` 与每一条都远、其余三条彼此紧，而且这种一对三结构不是从
`h=5` 一直持续到 `h=20` 时，`T_m` 才大。它在 `h=20` 后选择 `argmax T_m`，用额外十步
计算换取对“暂态”和“一对三”的真正定义。新池中不优于 uniform 或安全恶化即淘汰。

### 11.2 跨尺度依赖的快速松弛

在每条 scout 的内部状态上额外做事前冻结的 shifted-noise model query。令
`U_m^0(h)` 为正常 `pred_xstart`，`U_m^Delta(h)` 为同一规范化状态在更高噪声尺度的预测，
用同一内部距离定义

\[
G_m(h;\Delta)=d_{\rm internal}(U_m^0(h),U_m^\Delta(h)),
\]

\[
C_m=\operatorname{mean}_{\Delta\in\mathcal D}
\frac{G_m(5;\Delta)-G_m(10;\Delta)}{G_m(5;\Delta)+\varepsilon}.
\]

候选选择跨尺度依赖在十步内下降最快的分支。它最贴近原“高噪声支撑”直觉，但只能称为
operational shifted-time sensitivity，不可重新冒充理想边际密度比。不同冻结 `Delta`
方向冲突、无 policy advantage 或只带来过锐化，任一项都证伪。

### 11.3 posterior revision 的先修正后收敛

白化 innovation

\[
z_m(h)=\frac{x_m(h+1)-\mu_m(h)}{\sigma_m(h)}
\]

在正确 sampler 下本来就近似标准高斯，不能把它的 norm 直接当 badness。更合理的是看
单位噪声冲击导致模型的最终结构信念改了多少：

\[
J_m(h)=\frac{d_{\rm internal}[U_m(h+1),U_m(h)]}
{\sigma_m(h)\operatorname{RMS}[z_m(h)]+\varepsilon},
\]

\[
S_m=\frac{\operatorname{median}_{h=1..5}J_m(h)}
{\operatorname{median}_{h=6..10}J_m(h)+\varepsilon}.
\]

高 `S_m` 表示前五步允许较大结构修正，而后五步逐渐收敛；持续被每次 innovation 改写的
分支不会得高分。必须事前锁定 denominator floor，防止用小分母制造假信号。

### 11.4 多尺度结构结晶 / 后期波动

对 predicted-clean latent 定义纯内部多尺度边缘能

\[
e_s(U)=\frac{\operatorname{RMS}[\nabla A_sC(U)]}
{\operatorname{RMS}[A_sC(U)]+\varepsilon},\qquad s\in\{1,2,4\}.
\]

令

\[
G_m=\min_s[e_s(U_m(10))-e_s(U_m(0))],
\]

\[
V_m=\operatorname{median}_{h=6..10}
d_{\rm internal}[U_m(h),U_m(h-1)],
\qquad
K_m=\frac{\max(G_m,0)}{V_m+\varepsilon}.
\]

跨尺度取最小值要求粗、中、细结构都形成，`V_m` 又惩罚后段持续乱跳，因此不会只奖励
单纯高频噪声。它最直接对应 blur/fusion，但 latent edge 与可见 edge 之间仍有语义鸿沟。

每一个都必须先锁公式、因果可用时刻和安全证伪门，再生成独立后缀；外部裁判永远不
反向进入方法。

## 12. 关键复现入口

- V1.2 配置：`experiments/configs/dit_v22_transient_escape_prospective_v1.json`
- V1.2 冻结器：`experiments/freeze_dit_v22_transient_escape_prospective.py`
- V1.2 shard runner：`experiments/run_dit_v22_transient_escape_prospective_shard.py`
- V1.2 纯内部 extractor：`experiments/extract_dit_v22_transient_escape_internal.py`
- V1.2 lock identity：
  `cd8154479f5f6f883ae21d6657a61ec91ff6d2c77f569e18ea589d83517671a9`
- V1.2 内部选择产物 identity：
  `1c2ee25f6ca7b07c6259a0d9abeb176df9e7b26a0413d197417d24a62d24c72e`
- V1.1 外部盲评分析锁 identity：
  `42529439647c1647800ff7bdbed587c6e991d816ef7c1909d4ca0fcdc12db7db`
- `E/B` 修复协议一致结果：
  `f608e77afd72bbb6921720eaf92840c519528697ea0dd1712bac8dd147d6a358`
- branch medoid/high-`O` 冻结结果：
  `00838715cf95869cf4ac788226b05d3a1cf2714f0caa12e17c83b1db7c49e703`
- max-outlier 事后审计：
  `7d0a0094db136c2c6d644733b71b4cd634ad11bf7b9738151771d360c947508a`
- Doob core/runner/analyzer/audit：
  `experiments/extract_dit_v22_doob_consistency_probe.py`、
  `experiments/run_dit_doob_consistency_discovery_probes.py`、
  `experiments/analyze_dit_doob_consistency_discovery.py`、
  `experiments/audit_dit_doob_consistency_discovery_v1.py`
- PTCV core、扩展 runner 与最终 analyzer：
  `experiments/dit_projected_tweedie_cone.py`、
  `experiments/run_dit_projected_tweedie_cone_expansion_probe.py`、
  `experiments/analyze_dit_ptcv_blur_historical_validation.py`
- FPCV core、原 runner 与 label-free geometry smoke：
  `experiments/dit_finite_posterior_cyclic_violation.py`、
  `experiments/run_dit_fpcv_fresh_probe.py`、
  `experiments/smoke_dit_fpcv_probe_geometries.py`
- Self-Guidance 暂停配置与 prompt lock：
  `experiments/configs/self_guidance_sd14_coco1k_fid_screen_v1.json`、
  `experiments/locks/self_guidance_sd14_coco1k_screen_v1`
- Fisher 5K 权威结果（外部研究目录）：
  `/home/zhoushunyu/AItest/ecsg_validation/fid_pivot/FISHER_GEODESIC_FID5K_RESULT.md`
