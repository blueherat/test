# DiT PTCV 冻结发现实验结果（2026-08-28）

## 结论先说

> **2026-08-28 扩展验证更新：PTCV 现已完全停止。** 下文记录的是最初
> `targeted100` 发现实验；随后我们把其中唯一看似可延续的“全局模糊”假设，原样冻结后
> 放到不重叠 seed 的 `expansion_eval360` 上。9 张 `global_blur clear_bad` 对 304 张
> `clean_good` 的主 AUC 只有 `0.464547`，精确 class-stratified 单侧
> `p=0.606019`，且 class 207/602/795 的 AUC 分别为
> `0.3380/0.0965/0.8963`。这不是“门槛设得太严”，而是总体方向接近随机且跨类翻转。
> 因此不再把 PTCV 用于检测、guidance、rollback、rejection 或 1k FID；class 795、
> checkpoint 199、negative-eigen component 等事后切片也不得作为救援。完整扩展结果见
> [第 7 节](#7-扩展历史验证模糊窄假设也被否定)。

Projected Tweedie-cone Violation（PTCV）在当前冻结形式下**不能作为通用 bad-case
检测器**。正式结论为：

```text
STOP_AS_ARTIFACT_DETECTOR_NO_SIGN_AXIS_COMPONENT_RESCUE
```

主分数在 5 个 `clear_bad` 与 69 个 `clean_good` 上的 AUC 为 `0.6289855`，低于事前
要求的 `0.65`；精确的 class-stratified 单侧随机化检验为
`23236/185640 = 0.125167`，也没有达到冻结的 `0.10` 门。删去最早检查点后 AUC 降到
`0.4667`，删去 class 795 后降到 `0.3889`。因此结果既没有通过总体强度门，也没有
通过跨时刻、跨阳性类别的稳健性门。

不过，一个很窄的现象值得在**全新数据**中重新预注册：3 个带 `global_blur` 标记的
clear-bad 都偏高，描述性 AUC 为 `0.8889`、精确 `p=4/2040=0.00196`；两张明显
肢体/物体错位或错误附着图则方向相反，描述性 AUC 只有 `0.2391`。所以当前证据最多
支持“PTCV 可能是早中期全局模糊的内部信号”，不支持把它用于融合、拓扑或错位修复。

## 1. 被检验的量

在类别 `c`、扩散时刻 `t` 和当前 latent 状态 `x` 下，令

- `m_t^c(x)`：冻结 DiT 的 raw class-conditional、unclipped `pred_xstart`；
- `J_t(x)=d m_t^c(x)/d x`：该 clean-prediction map 对输入 latent 的 Jacobian；
- `Q`：事前固定的 16 维 Hadamard-channel × DCT-spatial 正交基；
- `B_t=Q^T J_t Q`：Jacobian 在该低维子空间中的投影；
- `S_t=(B_t+B_t^T)/2`：对称部分；
- `A_t=(B_t-B_t^T)/2`：反对称部分；
- `S_+`：所有对称半正定矩阵组成的锥。

理想高斯去噪后验满足 Tweedie Jacobian 恒等式

\[
J_t(x)=\frac{\alpha_t}{\sigma_t^2}
\operatorname{Cov}(X_0\mid X_t=x,c),
\]

所以它必须对称且半正定。投影矩阵到该锥的精确平方距离为

\[
d_t^2
=\operatorname{dist}_F^2(B_t,\mathbb S_+)
=\lVert A_t\rVert_F^2
+\sum_{\lambda_j(S_t)<0}\lambda_j(S_t)^2.
\]

三个冻结检查点的路径主分数是块对角归一化锥距离

\[
D_i=
\frac{\sum_{k\in\{99,149,199\}}d_{i,k}^2}
{\sum_{k\in\{99,149,199\}}\lVert B_{i,k}\rVert_F^2+10^{-30}},
\]

再用同类别全部 20 条无标签路径的均值和样本标准差转成 `S_i`。`S_i` 越大是冻结的
唯一质量方向。它不是三个时刻等权平均：每个时刻按投影 Jacobian 能量隐式加权。

完整推导、每个坐标变换和 CFG 边界见
[`DIT_PROJECTED_TWEEDIE_CONE_VIOLATION_ZH.md`](DIT_PROJECTED_TWEEDIE_CONE_VIOLATION_ZH.md)。

## 2. 无标签数值审计

正式标签解封以前，20 个 seed × 8 个 class × 3 个 checkpoint 的全部产品已写盘并
哈希封存。数值审计结果为：

| 检查 | 结果 |
|---|---:|
| raw conditional prediction 与原 trace 重放 | `60/60` bitwise exact |
| 从 NPZ 重算 checkpoint/path CSV | 最大误差 `0` |
| small 与 large 半径路径分数 Spearman | `0.998151` |
| small 与 Richardson Spearman | `0.999619` |
| large 与 Richardson Spearman | `0.996944` |
| 每路径最大矩阵差异中位数 | `0.016457` |
| 每路径最大矩阵差异 95% 分位 | `0.038334` |
| 出现负特征值能量的 checkpoint/path | `6/480`、`4/160` |

所有冻结数值门均通过。也就是说，正式失败不能归因于差分半径太小、Richardson 不稳、
矩阵轴写反或 raw conditional/CFG 查询混淆。

三个检查点的 Jacobian 能量份额中位数分别是：

| checkpoint index | internal timestep | 能量份额中位数 |
|---:|---:|---:|
| 99 | 150 | `0.6500` |
| 149 | 100 | `0.2104` |
| 199 | 50 | `0.1411` |

因此主分数天然更偏向 checkpoint 99；后面的 leave-one-checkpoint-out 是必要的机制
审计，不是附加美化。

## 3. 正式质量结果

### 3.1 主结果与固定对照

| 分数，高为 bad | AUC | 精确单侧 p |
|---|---:|---:|
| PTCV 主分数 | `0.628986` | `0.125167` |
| finite-difference gap control | `0.562319` | `0.259605` |
| matrix-energy control | `0.394203` | `0.826885` |
| raw-conditional/CFG gap control | `0.339130` | `0.839135` |
| temporal-change control | `0.278261` | `0.931319` |

PTCV 没有低于任一简单 control，但“比几个失败 control 好”不等于达到检测标准。只有
5 个阳性时，class-stratified bootstrap 95% 区间也非常宽：`0.2639--0.9444`。

### 3.2 五张 clear-bad 的类内位置

| 样本 | 人工缺陷 | 类内 rank / 20 | 标准化分数 |
|---|---|---:|---:|
| `class0602_seed12` | limb/object misalignment | `2/20` | `-1.1833` |
| `class0795_seed15` | global blur | `18/20` | `1.3045` |
| `class0795_seed22` | global blur | `19/20` | `1.7195` |
| `class0207_seed26` | global blur | `13/20` | `0.4362` |
| `class0602_seed27` | misalignment + topology attachment | `9/20` | `-0.3723` |

恰好是 3 张全局模糊图高于类中位数，两张结构错位图低于类中位数。这个分裂比总体
AUC 更重要：它直接否定了“同一个 cone defect 能覆盖所有明显 bad case”的故事。

### 3.3 稳健性反证

| 审计 | AUC |
|---|---:|
| 去掉 checkpoint 99 | `0.466667` |
| 去掉 checkpoint 149 | `0.646377` |
| 去掉 checkpoint 199 | `0.628986` |
| 去掉阳性 class 207 | `0.586364` |
| 去掉阳性 class 602 | `0.883041` |
| 去掉阳性 class 795 | `0.388889` |

信号主要由 checkpoint 99 和 class 795 的两张模糊图支撑；class 602 的结构异常实际
拉低结果。这正是冻结的跨时刻、跨类门没有通过的原因。

## 4. 应该怎样解释，而不应该怎样解释

当前 160 条路径中，cone distance 几乎全部来自 `skew(B)`；负特征值能量只在少数路径
出现，且三张模糊 bad 的负特征值贡献为零或极小。因此观察到的现象更准确地说是：

> 某些全局模糊轨迹在早中期落到 raw conditional denoiser 的高 projected-curl /
> non-conservative 区域。

不能据此说：

- “PSD 违反已经普遍识别坏图”；
- “结构错位来自 posterior unrealizability”；
- “PTCV 已经可以触发 guidance、rollback 或 rejection”；
- “AUC 0.889 已经是独立确认结果”。

最后一点尤其重要：blur subtype 虽然使用了事前存在的 phenotype flags，也事前声明为
descriptive-only，但它仍来自同一 discovery pool。它只能生成下一批的冻结假设。

## 5. 下一步的严格边界

### 5.1 可以继续的窄假设

保持以下对象完全不变，在不重叠的新 seed、事先锁定的全局模糊 endpoint 上验证：

1. 同一 raw conditional mapping；
2. 同一 16 维基、两个差分半径和 Richardson 公式；
3. 同一三个 checkpoint 和块对角能量加权；
4. 同一 higher-is-worse 方向；
5. endpoint 只比较 `global_blur/soft_fusion clear_bad` 与 clean-good；
6. 不使用 skew-only、checkpoint-99-only、class-795-only 或换基救援。

只有新池通过后，才值得检查它相对手工局部模糊量 `B` 是否有增量价值，并再研究
selective resampling。当前不授权任何干预。

### 5.2 结构错位必须换机制

融合、肢体/物体错位和拓扑附着不能沿本次 PTCV 结果继续调参。下一候选应检查一个
不同的必要规律，例如规范热坐标中 learned score/denoiser 跨噪声切片是否满足同一热流
PDE，而不是继续放大单时刻 Jacobian 的 curl。新量仍须先完成无标签数值解析度检查，
再在新标签池上冻结方向和门槛。

## 6. 不可变产物

- frozen config SHA-256：
  `60b359111b734bcbed12b9f9f2e6e1ef748b2328c1a9a150c3b0c41f3c8c7394`
- analyzer 执行时 SHA-256：
  `7c9f10d08c1173d936142f6c0964ba9ba598222541ccf653f1a7f8058d1ce3d3`
- label-free seal identity：
  `e0f301140a9145443df7073a2ff5ba2fbab7c6f6aa07324f1aa27b1da841066b`
- formal result identity：
  `b01d27d88a7ff8822d5279fc2dc2dbade866f6e634539709639d3f6ae417110d`
- formal output：
  `/data/users/zhoushunyu/eqvae/cross_scale_evidence/dit_projected_tweedie_cone_discovery_analysis_v1`

独立审计从 20 个原始 NPZ 重新计算全部 480 个 checkpoint 与 160 条路径，主结果、对照、
LOO 和 subtype 均一致；最大数值差异不超过 `4.3e-14`。本地 seal 能证明文件链一致和
程序内先写无标签分数、后开标签，但它不是外部不可篡改时间戳，所以该结果仍严格称为
discovery，而不是 confirmation。

## 7. 扩展历史验证：模糊窄假设也被否定

### 7.1 冻结设计

扩展实验没有根据新标签调公式，保持 discovery 中的以下对象不变：

- 同一 raw class-conditional、unclipped `pred_xstart`；
- 同一 16 维基、两个半径与 Richardson 重算；
- 同一 checkpoint `99/149/199`；
- 同一块对角能量加权和 `higher-is-worse` 方向；
- 不允许 sign、component、checkpoint、class、basis、radius、endpoint 或组合救援。

数据是与 `targeted100` 不重叠的 seed 130--249、class 207/602/795，共 360 条路径。
冻结标签中有 9 张 `global_blur clear_bad`、304 张 `clean_good`；47 张 mild/disputed
不进入主二分类比较。由于这个扩展池的图像在更早研究中已经被看过，它只能称为
**disjoint-seed historical validation**，不能包装成全新的 prospective confirmation。

### 7.2 正式结果

| 固定检验 | 结果 |
|---|---:|
| PTCV 主 AUC | `0.464547` |
| exact class-stratified one-sided p | `0.606019` |
| bootstrap AUC median | `0.465017` |
| bootstrap 95% interval | `[0.219256, 0.707620]` |
| bad 高于本类中位数 | `5/9` |
| class 207 / 602 / 795 AUC | `0.3380 / 0.0965 / 0.8963` |
| omit class 207 / 602 / 795 AUC | `0.6029 / 0.5303 / 0.3076` |

所有标签解封前的数值门都通过：360 条路径、1080 个 checkpoint 产品完整；small/large
半径路径分数 Spearman 为 `0.998070`，small/Richardson 为 `0.999604`，360 次 raw
conditional checkpoint replay 全部 bitwise exact。因此失败不能归因于差分数值不稳。

固定的简单 control 也说明主量没有独特优势：finite-difference gap 的 AUC 为
`0.519006`，matrix energy 为 `0.410453`，raw-conditional/CFG gap 为 `0.269371`，
temporal change 为 `0.244152`。单独挑 checkpoint 199 (`0.616959`) 或 negative-eigen
component (`0.617325`) 都是看过结果后的切片，且远不足以抵消跨类翻转，故不作救援。

### 7.3 独立复算与最终边界

独立实现重新读取 joined CSV，复算得到：

- 主 AUC 精确为 `0.464546783625731`；
- 全局 midrank、类内固定阳性数的动态规划共有
  `1,440,983,210,716,140` 个合法赋值，单侧尾部概率精确为
  `0.6060188017420034`；
- 所有 per-class、leave-one-class-out、leave-one-checkpoint-out 和固定 control AUC
  与正式 JSON 一致；
- `results.json`、label-free seal 及五个被 seal 的数据文件哈希全部通过。

正式 decision 为：

```text
STOP_PTCV_BLUR_SPECIFIC_HYPOTHESIS_NO_RESCUE
```

值得保留但必须分开的观察是：此前定义的 decoded `pred_xstart` 手工模糊量 `B` 在同一
扩展池上有描述性 AUC `0.942982`，而它与 PTCV 的 Spearman 只有约 `0.062`。这说明
“早期预测图里已经出现可见模糊”仍可能有工程价值，但它不是 PTCV 的理论胜利；若把
`B` 接入实际修正，必须单独定义干预，再用 baseline/method 配对采样与 1k FID 检验。

扩展产物：

- config SHA-256：`f90be2d24adb234bd9dcbba09ef6b92ea568d085628a15fe8721d25db26eb274`
- label-free seal：`68fc50624303b2dd3e0240b9e3cfe644e87ac4a091f37d0381b724001d6cdbe7`
- joined rows SHA-256：`6faed085df37c9e9bbdcb21c058e698e7ba19e7ec01f492f6ebbc159f81a3f1f`
- result identity：`f18e8b7c349cceb657fd73f719d5d0437ec29a2cd015f669b11bfd8aa8e8490c`
- output：`/data/users/zhoushunyu/eqvae/cross_scale_evidence/dit_ptcv_blur_historical_validation_v1`
