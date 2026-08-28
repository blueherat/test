# DiT v2.2 内部信号复核：E 退役，分支共识首版未通过

## 结论先说

截至 2026-08-28，当前证据不支持继续把跨尺度 e-process `E` 当作坏图检测器，
也不支持把它当作后缀回滚的“可修复性”触发器。`E` 的鞅记账仍然严格成立，失败的
是它所代表的操作性备择与视觉质量之间的对应关系。

内部模糊量 `B` 比 `E` 更接近可见模糊/软融合表型，但它也只能保留为弱启发式量。
本次试验已经把连续 `B` 强度匹配到很近，两个组最终仍只有 `3/8` 对 `6/8` 条路径
被盲评确认存在可修的模糊。这说明一个标量 `B` 大，并不等于同一种可见缺陷，也不
足以作为自动回滚门。

后缀重采本身不是完全无效：按原协议的严格定义，71 个“基线确有模糊修复机会”的
匿名比较中有 27 个成功，约 `38.0%`。但成功和失败都很多，且当前没有可靠的内部量
知道哪条路径值得重采。因此工程动作不能先于检测量。

后续冻结实验又排除了第一版条件分支故事：`h=10` medoid 的严格修复为
`5/18=27.8%`，低于同四条 scout 均匀随机的 `37.5%`；higher-`O` 预测 repair
opportunity 的 AUC 只有 `0.286`。因此“多数分支形成正确共识、选中央分支”已停止。

揭盲后的完整 rank 审计出现了一个反向线索：`h=10` 最离群 fresh scout 为
`10/18=55.6%`，比同预算随机高 `18.1pp`。但它是看到 medoid 失败后才提出的方向，
`h=5/20` 仅高 `6.9/1.4pp`，而且小样本 safety 两项略差，不能直接称为方法。

当前只允许做一次窄的 new-suffix prospective test：

> 在 step149 的共同前缀上生成四条完全对称的短程 baseline-P scout，只用共同前缀
> 归一的多尺度 predicted-clean latent 距离，在十次转移后选择平均离其他三条最远的
> 分支；内部选择先封存，再由外部盲评比较它与同四条 scout 均匀随机策略的平均质量。

它测试的是 branch-selection utility，不是 bad-prefix detector，也不要求识别全部
bad case。即使通过，也只能称为冻结 sampler/class 范围内的 early-horizon outlier
selector；不能直接声称存在一般的 consensus trap。

FID、Inception、DINO、CLIP 和人工/模型盲评仍然只做内部选择封存后的外部裁判，绝不
进入 selector、触发量、阈值或前缀选择。

## 1. 每个字母是什么意思

- `P`：真实运行的冻结基线采样器；这里是官方 DiT-XL/2、250 步 ancestral DDPM、
  CFG=4。
- `Q*`：人为定义的跨尺度方向备择 Markov chain。星号表示它是操作性备择，不是理想
  热流边际分布。
- `k`：预终点检查或采样转移的时刻索引。
- `x_k`：第 `k` 步抽取当前创新以前的 latent 状态。
- `B_k`：从当前 predicted-clean 草图得到的局部低边缘/软融合持续量；大表示更像模糊
  表型，但不是坏图概率。
- `E_k`：从实际创新相对 `P/Q*` 两条路径假设累计出的 e-value。
- `E_alarm`：`sup_k E_k >= 10`，对应冻结的 `alpha=0.1`。
- `J`：联合事件 `E_alarm AND B_alarm`。它只是 `E_alarm` 的子集，不是新的 e-process。
- `alpha`：Ville 阈值参数；控制的是 `P` 下总体 anytime 触发预算，不是好图条件 FPR。
- `i`：原始生成路径索引。
- `s`：回滚位置；本试验固定为采样步 `109` 或 `149`。
- `m`：后缀尝试索引；`m=0` 是精确重放，`m=1..4` 是全部保留的新随机后缀。
- `O_{i,s,m}`：外部盲评是否确认基线有修复机会，即同一评审同时认为基线非 clean
  且存在 blur/fusion，再取三人中的二票。
- `S_{i,s,m}`：按原协议定义的成功修复事件。

原协议的 `S` 必须同时满足：

1. `O=1`；
2. 至少两位评审各自在同一个匿名比较中判断基线有 blur、fresh 无 blur；
3. 至少两位评审判断类别、身份、物体数、主要姿态和构图保持；
4. fresh 未被至少两位评审判为 `clear_bad`。

这里的 `O` 和 `S` 都是实验后的外部判定，只用于评价内部量，不能进入实际采样方法。

## 2. E 为什么数学正确，质量语义却可以失败

基线一步写成

```text
x_{k+1} = mu_k + sigma_k * z_k,    z_k | F_k ~ N(0, I).
```

`Q*` 只把白化均值沿冻结的跨尺度方向移动 `u_k`，于是同协方差高斯的对数似然比增量为

```text
Delta ell_k = u_k^T z_k - 0.5 ||u_k||^2.
```

只要 `u_k` 在当前 `z_k` 抽取以前由过去信息决定，就有

```text
E_P[exp(Delta ell_k) | F_k] = 1.
```

所以

```text
E_k = exp(sum_{r<=k} Delta ell_r)
```

相对于实际实现的 `P` 是严格正鞅。这证明的是“这条已实现路径更像 `Q*` 还是 `P`”的
序贯记账正确，不证明 `Q*` 所偏好的路径一定是坏图。

v2.2 又把原始跨尺度差归一化成固定信息方向。归一化解决了高维 likelihood-ratio
collapse 的部分工程问题，却主动丢掉了原始差值幅度。因此当前 `E` 更准确的含义是
“实际创新是否持续沿着这个人为跨尺度方向”，而不是“伪影有多严重”。

## 3. 在回滚试验以前，E 已经失去通用坏图检测资格

第三池共 1,740 条类别路径。冻结内部信号的 2×2 计数为：

| 事件 | `B=0` | `B=1` | 合计 |
|---|---:|---:|---:|
| `E=0` | 1,575 | 139 | 1,714 |
| `E=1` | 18 | 8 | 26 |

旧标签只有 4 个主定义阳性，不能支撑稳定 AUC 结论。随后对全部 26 条 `E_alarm`
路径和 26 条同类别、同启动时序对照做重新匿名盲评：

- 严格 clear-bad：alarm `1/26`，control `2/26`；
- blur/soft-fusion：alarm `9/26`，control `5/26`；
- blur 的单侧配对精确 `p=0.0625`，但 clear-bad 方向相反。

因此 `E>=10` 不能作为通用坏图警报。最多还剩一个较窄的可能性：它也许不表示图更
坏，却表示在已经出现 `B` 表型时更容易通过重采后缀修复。本次 pilot 专门检验这个
剩余解释。

## 4. 修复性 pilot 如何设计

选择完全不读取终点图片、旧标签、FID 或任何外部表征：

- 8 条 `J=E_alarm AND B_alarm` 路径；
- 8 条 `B_alarm AND NOT E_alarm` 路径；
- 按类别、完整启动时序和连续 `B` 强度一一匹配；
- case/control 平均 `B` 分别约为 `2.51881/2.52474`；
- 每条路径从 `s=109` 和 `s=149` 各运行一次；
- 每个位置保留精确重放 `m=0` 和全部 4 个 fresh 后缀；
- 共 32 个 job、160 个分支，不做 best-of-N。

正式输出以前已经冻结选择、执行、匿名化、三评审字段和统计规则。最终形成 128 个
“基线—fresh”匿名左右比较。三位彼此隔离的评审看不到 seed、class、role、step、
attempt、`B/E` 或另两人的回答。

关键冻结标识为：

- selection lock：`16acd0bffda207ed73ef78a62909e53997bef68baae66cdffedede1bb207fbd0`；
- execution lock：`c71ac783f2f72b9ec599b20ec7134c0ea1ebad642dbb1155fc7d873cc63d1cb6`；
- review-source lock：`ca06061606ad46719e059298819fc0c66d79a9bcdd0be740635966274ed1bcab`；
- blind delivery：`6f4b4941e0756c38f8e9c51955c0657bab43b69c5e84fda88b82e7ea8db1c503`。

## 5. 必须保留的分析更正

第一版冻结 analyzer 的数据链路和汇总计算本身可复现，但它把“至少两人偏好 fresh”
误当作了“blur/fusion 明确减轻”。它因此多算了 17 个未满足 blur 减轻条件的成功：

- `J`：第一版 14 个，协议一致定义为 10 个；
- `B-only`：第一版 30 个，协议一致定义为 17 个。

第一版结果 ID `055768a7...` 和其后基于该定义的 post-hoc summary `59d94c5b...`
都保留作审计记录，但不能作为协议主结果。协议一致更正结果为：

- result ID：`f608e77afd72bbb6921720eaf92840c519528697ea0dd1712bac8dd147d6a358`；
- analyzer source SHA-256：`64d2f9b8b788509dc55ccb3ed7855af3c81884150ea5182b01fb12fbe7544100`。

这个逐票 resolution 是在揭盲后实现的，所以仍应叫“协议一致 corrigendum”，不能包装
成全新的确认性检验。

## 6. 更正后的结果

### 6.1 冻结机会门失败

| 角色 | 路径数 | 有稳定修复机会的路径 | opportunity 比较数 |
|---|---:|---:|---:|
| `J=E+B` | 8 | 3 | 23/64 |
| `B-only` | 8 | 6 | 48/64 |

协议要求每组至少 4 条 opportunity path。`J` 只有 3 条，因此组间 repairability
比较按冻结规则必须记为 `INCONCLUSIVE`。

一次独立、只看 16 张 attempt-0 原图的事后视觉复核也得到完全相同的 `3/8` 对
`6/8`。差异主要来自狗类：两组都是 3 张狗、5 张滑雪板，但 `B-only` 狗有 `2/3`
出现明显软糊，`J` 狗为 `0/3`。这不是简单类别混杂，也不像成对比较上下文造成的
偶然标签差。

若只算灾难性的 `clear_bad`，独立复核为 `J 2/8`、`B-only 3/8`，差距会缩小。这再次
说明 `B` 更像软表型强度，而不是严重坏图概率。

### 6.2 严格成功率

| 角色 | 全部比较 | opportunity 内 | step 109 | step 149 |
|---|---:|---:|---:|---:|
| `J=E+B` | 10/64 = 15.6% | 10/23 = 43.5% | 5/11 = 45.5% | 5/12 = 41.7% |
| `B-only` | 17/64 = 26.6% | 17/48 = 35.4% | 8/24 = 33.3% | 9/24 = 37.5% |

不能用 opportunity 内 `43.5%-35.4%=+8.1` 个百分点声称 `E` 有帮助，因为它们来自
3 条 `J` 路径和 6 条 `B-only` 路径，并不是同一组可比机会。

### 6.3 一一匹配路径没有出现 E 增量

8 个 `pair_index` 的“`J` 严格成功率减去匹配 `B-only` 严格成功率”为：

```text
[-0.375, 0, 0, -0.125, 0, -0.25, -0.125, 0]
```

即 4 对更低、0 对更高、4 对持平，平均差 `-0.109375`。固定种子的配对 path
bootstrap 描述区间为 `[-0.203125,-0.03125]`，leave-one-pair-out 平均差始终在
`[-0.125,-0.07143]`。这些都是揭盲后的稳定性诊断，不是确认性置信区间或因果效应。

两边都确有机会的匹配对只剩 3 对，其差值为：

```text
[-0.125, 0, -0.25]
```

仍然没有正差，但 `n=3` 太小，只能用于否定“当前数据里已经出现明显 E 优势”，不能
估计一般效应。

### 6.4 回滚深度没有给 E 特异支持

- step 109 的匹配平均差为 `-0.09375`；
- step 149 的匹配平均差为 `-0.125`；
- 两个位置都没有出现稳定的 `J` 优势。

第一版“偏好 fresh”口径下，较深 step 109 对 `B-only` 看起来更好；换回严格 blur
reduction 后，这个现象明显减弱。因此当前不能把回滚深度写成已经成立的工程规律。

## 7. 现在能说什么，不能说什么

可以说：

1. `E` 相对于操作性 `P/Q*` 的鞅恒等式正确。
2. `E>=10` 没有通过通用坏图富集检验。
3. 在本次 B 匹配 pilot 中，冻结机会门失败；匹配结果没有一对显示总体严格修复率的
   `J` 优势。
4. 当前 `E` 不应进入自动 rollback、拒绝或 guidance。
5. `B` 在总体上仍比随机方向更接近 blur phenotype，但连续数值不足以匹配视觉机会，
   更不能等同于严重 bad-case 概率。

不能说：

1. `E` 导致图片变差；`E` 分组是观察性的，而且机会分布失衡。
2. 后缀回滚已经提升总体生成质量；本试验只研究被选中的旧路径。
3. `B-only` 是新方法；它只是对照组。
4. 盲评、FID 或 DINO 可以直接变成在线触发器。
5. Ville 的 `alpha` 是好图 FPR、修复成功率或质量改进保证。

因此科学动作是：保留 `E` 的概率记账结果，停止它当前的质量解释；不再围绕 `E`
调阈值、尺度、预算或回滚深度。

## 8. 条件分支内部量：形式化、首版失败与反向线索

### 8.1 直觉

真正想区分的是两类失败：

- **采样事故**：同一个前缀往后走，多数随机分支能形成相近的正常结构，只有当前延续
  偏离共识。这种情况有机会通过重采修复。
- **模型能力限制**：从同一个前缀重采，多数分支都不稳定，或都形成同一种错误结构。
  继续抽后缀不会稳定解决。

`E` 问的是“创新是否沿跨尺度方向”；它与上面这一区分隔了一层未经验证的语义假设。
条件分支共识直接观察冻结 `P` 自己的条件未来分布，更贴近“可修复性”。

### 8.2 可计算对象

在一个保存的前缀 `X_j` 上，运行 `M+1` 个独立的短程 baseline-P scout。第 `m` 个
scout 走 `h` 次转移后的 predicted-clean latent 记为

```text
X_h^{(m)} in R^(4 x 32 x 32).
```

第一版不学习 `phi`，而是冻结一个纯内部多尺度距离。令：

- `R=X_0`：所有分支共同前缀处的 predicted-clean latent；
- `C(X)`：分别减去 `X` 每个 latent channel 的空间均值，去掉全局 DC 偏移；
- `A_s`：固定 `s x s` average pooling，`s` 取 `{1,2,4}`；
- `RMS`：对全部 channel 与空间位置取均方根；
- `epsilon=10^-6`：只用于避免分母为零。

两条分支在 horizon `h` 的距离定义为

```text
d_h(m,n)
  = (1/3) * sum_{s in {1,2,4}}
      RMS[A_s C(X_h^(m) - X_h^(n))]
      / (RMS[A_s C(R)] + epsilon).
```

它只读取采样器已经生成的 latent；不解码 PNG，不调用 VAE 表征、DINO、CLIP、
Inception 或人工标签。空间中心化避免把无关的 channel 整体漂移当成结构异常，多尺度
池化同时保留局部错位和较大范围融合，共同前缀归一化避免把不同 timestep 的幅度差异
误当成异常。

对当前旧 pilot，四条 fresh 分支集合为 `F={1,2,3,4}`。冻结的发现性量为

```text
D_h^F = median_{a<b, a,b in F} d_h(a,b),

A_{0,h}^F = (1/4) * sum_{r in F} d_h(0,r),

O_{0,h}^F = A_{0,h}^F / (D_h^F + epsilon).
```

- `D_h^F`：fresh 替代未来彼此有多不稳定；
- `A_{0,h}^F`：原 attempt-0 到四条 fresh 未来的平均距离；
- `O_{0,h}^F`：原延续相对 fresh 分支内在分散度的离群比。

这里用 pairwise distance 的中位数定义总体分散 `D`，但单分支 nonconformity 必须用到
其他分支距离的**平均值**，不能用中位数。五分支现有数据的 96 个 job-horizon 组合中，
中位数写法有 77 个结构性并列，而平均值写法没有并列。

若未来有 `K=M+1` 条真正条件可交换的 `P` 后缀，定义对称 nonconformity

```text
A_{m,h} = (1/M) * sum_{r != m} d_h(m,r),

p_{0,h} = (1 + number of scouts m>=1 with A_{m,h} >= A_{0,h}) / (M+1).
```

只要 checkpoint `j` 的选择仅依赖当时已有的 `F_j`，且 current 与 scouts 都在看到任何
后缀以前按同一 `P` 独立生成，上述 rank 才是条件 super-uniform 的。它只证明“当前
分支相对同前缀分支异常”，仍不自动证明图片坏；质量语义必须由之后揭开的外部盲标签
验证。

现有旧 pilot 的 `attempt-0` **不满足这一交换性条件**：16 条原路径是看过完整 `B/E`
未来后筛出的，step 149 的触发甚至包含该步 innovation。因此旧数据里的 `O_0^F` 只能
叫事后描述量，不能叫 conformal `p`。四条 fresh 分支彼此仍可交换，可以计算 fresh-only
rank，但 `K=4` 时最小 `p` 只有 `1/4`，不具备显著报警分辨率。未来要达到
`alpha=0.1/0.05`，分别至少需要 `K=10/20` 条对称分支。

### 8.3 首版提出时为什么值得检验

首版直觉不需要把高噪声方向解释成伪影方向，而是直接比较同一个前缀下真实 baseline-P
条件未来：若原延续孤立而 fresh scouts 形成紧密共识，它可能是可替换的采样事故；若
所有未来都不稳定，则更像模型能力限制。这个直觉在提出时合理，但不是定理，更不是结果。

### 8.4 首版 medoid/high-O 的冻结结果

32 个旧 job 的 label-free extractor 先封存 `h={5,10,20}` 的完整距离矩阵、fresh-only
rank、`D/O` 和选择，之后 evaluator 才打开旧 repair 标签。冻结主预测均未通过：

- `h=10` medoid：`5/18=27.8%`，同四条 scout 均匀随机：`27/72=37.5%`；
- `h=20` medoid 同样为 `5/18`；
- higher attempt-0 `O` 预测 opportunity 的 AUC 为 `0.286`；
- leave-one-path-out AUC 仍只有 `0.167--0.333`。

因此结论是 `STOP_CURRENT_FORM`。low-`O` AUC `0.714` 只是 `1-0.286`，不能当成独立
的新发现。原来的“正确共识在中央”叙事已被当前数据否定。

### 8.5 max-outlier 的反向线索与安全失败

揭盲后才检查的四个 fresh-only rank 成功数为：

```text
h=5 : [7, 7, 5, 8] / 18
h=10: [5, 7, 5, 10] / 18
h=20: [5, 7, 8, 7] / 18
```

`h=10` 最离群分支为 `10/18=55.6%`，比随机高 `18.1pp`。窄解释是：错误结构也许已经
成为当前 prefix 最常见的局部吸引盆，成功分支需要短暂离开多数分支，而不是追随 medoid。

但这只是 post-hoc clue：方向是看完 rank table 才选，`h=5/20` 弱得多，收益主要集中
class795；在非机会样本上 max-outlier clear-bad 为 `14.3%` 对随机 `10.7%`，preservation
为 `85.7%` 对 `87.5%`。它不能被写成已成立的 consensus-trap 或 repair 方法。

### 8.6 V1.2 new-suffix prospective test

V1.2 固定 step149、四条对称 fresh baseline-P scouts 和 `h=10`。对每条分支定义 fresh
平均 nonconformity

```text
A_m(10) = (1/3) * sum_{n != m} d_10(m,n),
```

只选择 `argmax_m A_m(10)`；并列按 GPU 输出前冻结的匿名 slot 解决。96 个 class795
prefix 是唯一 benefit confirmation population，class207/602 各 16 个只作明显伤害
哨兵。主比较不是一次随机实现，而是同四条已计算 scouts 的精确 uniform-policy value。

这项方法不检测 prefix 是否 bad，也不把 `A_m` 当概率或 e-value。它只检验：固定内部
早期几何能否在平均意义上从同预算四条未来中选出更低缺陷的 endpoint。所有选择必须在
打开 PNG、盲评、FID 或表征前封存；主效应、安全门和失败后不得换 horizon/方向的规则
已写入 lock identity
`cd8154479f5f6f883ae21d6657a61ec91ff6d2c77f569e18ea589d83517671a9`。

这个方向仍是冻结主干、无需训练的小模块方法；它把概率对象从人为 `Q*/P` 路径比改成
同一前缀下真实 `P` 条件未来的选枝策略。完整方法总账见
[`DIT_BAD_GOOD_METHOD_LEDGER_2026-08-28_ZH.md`](DIT_BAD_GOOD_METHOD_LEDGER_2026-08-28_ZH.md)。

## 9. 可复现入口

主要源码：

- `experiments/freeze_dit_v22_repairability_pilot.py`
- `experiments/freeze_dit_v22_repairability_execution_sources.py`
- `experiments/intervene_dit_v22_custom_trace_suffix.py`
- `experiments/prepare_dit_v22_repairability_blind_review.py`
- `experiments/analyze_dit_v22_repairability_blind_review.py`
- `experiments/analyze_dit_v22_repairability_protocol_conformance.py`
- `experiments/summarize_dit_v22_repairability_pilot.py`（第一版 success 的 post-hoc
  配对审计，已被协议一致结果取代，不能单独引用）
- `experiments/extract_dit_v22_branch_consensus_label_free.py`
- `experiments/evaluate_dit_v22_branch_consensus_posthoc.py`
- `experiments/audit_dit_v22_branch_consensus_escape_posthoc.py`
- `experiments/freeze_dit_v22_transient_escape_prospective.py`
- `experiments/run_dit_v22_transient_escape_prospective_shard.py`
- `experiments/extract_dit_v22_transient_escape_internal.py`

权威结果：

- 第一版冻结分析：
  `/data/users/zhoushunyu/eqvae/cross_scale_evidence/dit_v22_repairability_pilot_v1_2_blind_review_v1_analysis`
- 协议一致更正分析：
  `/data/users/zhoushunyu/eqvae/cross_scale_evidence/dit_v22_repairability_pilot_v1_2_protocol_conformance_v2`
- 分支共识 label-free 产物：
  `/data/users/zhoushunyu/eqvae/cross_scale_evidence/dit_v22_branch_consensus_label_free_v1`
- medoid/high-O 冻结 evaluator：
  `/data/users/zhoushunyu/eqvae/cross_scale_evidence/dit_v22_branch_consensus_posthoc_evaluation_v1`
- max-outlier 事后审计：
  `/data/users/zhoushunyu/eqvae/cross_scale_evidence/dit_v22_branch_consensus_escape_posthoc_audit_v1`
- V1.2 prospective lock：
  `experiments/locks/dit_v22_transient_escape_prospective_lock_v1_2`
