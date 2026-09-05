# PFR 机制审计与反例归档

> **同日 RNG 勘误与理论续篇。** 历史 FID-5K 及 lambda sweep 的 nominal
> `seed=0/1` 分别共享 99.84% 与 99.2% 样本，不是独立重复；within-bank paired
> 对比仍有效。query-control 的第二个 bank 实际 seed 为 `1000003`，与 seed0 不重叠，
> 因而 projected 相对 time-only 的两个 RNG-disjoint bank 增量符号复现成立。新的不重叠
> FID-5K ordinary/PFR 为
> `40.7983/37.6459`。推导、RNG 修复与完整口径见
> [`PFR_COUNTERFACTUAL_RESIDUAL_THEORY_ZH.md`](PFR_COUNTERFACTUAL_RESIDUAL_THEORY_ZH.md)。

## 1. 范围与结论边界

本报告只整理截至 2026-09-03 已经完成的 PFR/IG 实验，不提出新的扫参计划，也不把尚未
闭合的解释包装成定论。固定实验对象为 ImageNet-100 SiT-S/2：完整 `v800` EMA 模型记为
`S`，冻结 backbone 后训练 50K 的 depth-4 velocity 弱头记为 `W`。

普通 Internal Guidance 为

\[
G_{\rm IG}(p)=S(p)+\gamma[S(p)-W(p)]
=W(p)+\beta[S(p)-W(p)],\qquad \beta=1+\gamma.
\]

canonical Projected Future Reference (PFR) 在当前点 `p=(z,t)` 上，先把
`beta(S-W)` 投影到 `G_IG` 的正向射线，沿该方向移动一个 `h=1/32` 的小步，同时把查询
时间推进到 `t+h`，得到 `q`。最终场为

\[
\boxed{
G_{\rm PFR}(p)
=G_{\rm IG}(p)-\beta[W(q)-W(p)]
=W(p)+\beta[S(p)-W(q)].
}
\]

当前最可靠的事实不是某个单一理论已经解释了 PFR，而是多个看似自然的解释已被明确
排除，剩余现象被压缩为：**PFR 使用一个时间推进且与候选修正对齐的弱头反事实查询；
时间推进贡献主要收益，空间查询提供可复现的增量，两者缺一时不能复现完整结果。**

## 2. 固定性能锚点

在既有、已提交的 FID-5K 双采样 seed 协议下：

| 条件 | FID-5K |
|---|---:|
| ordinary depth-4 IG | 39.9468 |
| previous oblique condition | 36.5003 |
| canonical PFR | **36.3470** |

这里的“最好”仅指本仓库固定 ImageNet-100/SiT-S/2 协议，不是跨模型或公开 ImageNet
基准的 SOTA 声明。原始双 seed 数据已在
`docs/data/projected_future_reference_ig/fid5k_results.csv` 中提交。

本轮 query control 使用配对 FID-1K。seed 0 的完整对照为：

| 查询 | 定义 | FID-1K |
|---|---|---:|
| ordinary IG | 无反事实查询 | 64.8513 |
| projected | 时间推进 + 正向投影位移 | **61.8592** |
| time-only | 只推进时间 | 62.5067 |
| state-only | 只做投影位移 | 65.3742 |
| anti-projected | 时间推进 + 反向等范数位移 | 62.6378 |
| orthogonal | 时间推进 + 正交等范数位移 | 62.5226 |
| donor | 时间推进 + 其他样本等范数位移 | 62.2997 |

第二个采样 seed 上，projected 仍然优于 time-only、anti、orthogonal 和 donor：分别为
`62.9425`、`63.4923`、`63.9850`、`63.3731` 和 `63.5838`。因此空间方向的作用不是
seed 0 的单次偶然，但它明显小于时间推进本身的作用。

## 3. 反事实修订强度的历史 bank 内响应曲线

定义

\[
G_\lambda=G_{\rm IG}-\beta\lambda[W(q)-W(p)].
\]

两个高度重叠的历史采样 bank 都显示从 `lambda=0` 到 `lambda=1` 的 FID 基本单调改善，超过 1 后
开始回升：

| lambda | seed 0 FID-1K | seed 1 FID-1K |
|---:|---:|---:|
| 0.00 | 64.8487 | 64.7263 |
| 0.25 | 64.1951 | 64.0886 |
| 0.50 | 63.2388 | 63.1730 |
| 0.75 | 62.0627 | 62.0542 |
| 1.00 | **61.8590** | **61.7627** |
| 1.25 | 62.4142 | 62.2786 |
| 1.50 | 62.5468 | 62.4491 |
| 2.00 | 65.7795 | 65.7901 |

这说明 `lambda=1` 在两个高度重叠历史 bank 的响应曲线上都处于谷底；它只提供邻近
样本扰动下的稳定性线索，不是独立 sampling-seed 复现，也不能单独证明某种概率解释。

## 4. 时间响应与空间响应

把 projected query 的弱头修订拆为纯时间响应以及空间增量，并将空间增量相对时间响应
分成平行、正交部分。两个高度重叠历史 nominal bank 的 FID-1K 均值为：

| 条件 | 平均 FID-1K |
|---|---:|
| time-only | 62.9995 |
| temporal + parallel spatial | 62.7157 |
| temporal + orthogonal spatial | 62.6056 |
| full projected | **62.4009** |

因此不能把全部空间收益归结为单一的一维幅值修正，也不能说只有正交分量有效。两个分量
分别加入时都有效，完整空间响应最好。

## 5. 终端分布究竟发生了什么

同一 seed 的 ADM activation 审计把 FID 精确分为均值项和协方差项：

| 条件 | FID | 均值项 | 协方差项 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| ordinary IG | 64.8527 | 7.5545 | 57.2981 | 0.427 | 0.7328 |
| time-only | 62.5076 | 5.4223 | 57.0852 | 0.443 | 0.7044 |
| full PFR | **61.8595** | **5.3587** | **56.5008** | **0.455** | 0.7058 |

PFR 相对 ordinary IG 的 `2.9932` FID 改善中，`2.1958`（73.36%）来自均值项，
`0.7974`（26.64%）来自协方差项。time-only 已取得几乎全部均值项改善；空间响应主要
继续修正协方差。与此同时 precision 上升 `0.028`、recall 下降 `0.027`，所以它更像
质量/覆盖率重新分配，而不是无条件把整个分布变得更正确。

time-only 与 full PFR 的配对 endpoint ADM feature shift cosine 为 `0.955`，也支持二者
共享主要终端作用方向，空间查询提供第二阶增量而非完全不同的生成机制。

## 6. 已被排除或显著削弱的解释

### 6.1 “未来查询是更准确的 Bayes 后验”不成立

在线性 bridge 的 held-out teacher bank 上，所有时间都知道守恒粒子速度
`U=X-E`。平均而言，当前弱头 MSE 为 `0.7510`，projected future query 为 `0.7534`，
time-only 为 `0.7542`；后两者分别恶化约 `0.32%` 和 `0.43%`。PFR 的收益不是因为
`W(q)` 在单步监督意义上更接近真实速度。

### 6.2 “W(q) 是 W(p) 到 S(p) 的条件投影”不成立

若修订 `R=W(q)-W(p)` 是 depth gap `D=S(p)-W(p)` 的正交投影，应满足投影残差恒等式，
相应比例应接近 1。128 样本审计中，projected 和 time-only 的跨时间平均比例只有约
`0.277` 和 `0.196`，且距离 strong 的变化并不符合投影收缩。它不能被解释为弱头沿
strong gap 的 L2 posterior catch-up。

### 6.3 “弱头是单调/PSD 曲率预条件器”不成立

对空间查询位移做 finite secant 审计，空间修订与查询位移的 cosine 从 `t=0.05` 的
`0.411` 很快下降到 `t>=0.35` 的约 `0.015` 以下；positive secant 比例中后期接近
`0.5`。因此弱头响应不是一个稳定的正半定曲率算子，不能用“总在同一能量面上向下走”
解释。

### 6.4 “任意自然信息时钟都等价”不成立

把三种 query horizon 在 `t=0.25` 精确匹配后，raw affine time 的 canonical PFR
FID-1K 约为 `61.86--61.92`；log-SNR clock 为 `64.9416`，additive-SNR clock 为
`66.9064`。因此至少在已测试的两个替代时钟上，PFR 不具有时间重参数化不变性。固定
raw `t` 不是可随意替换的记号。

### 6.5 严格的静态 density-ratio score 解释不成立

小样本 Hutchinson/JVP-VJP 审计中，完整 strong、weak、ordinary IG 与 PFR 场的平均
反对称 Jacobian 能量比例分别约为 `0.00144`、`0.00180`、`0.00457`、`0.00543`；但
depth gap、counterfactual ratio gap 和 PFR revision 分别约为 `0.21265`、`0.21994`、
`0.25324`。主场被共同的近保守分量支配，而真正用于 guidance 的小差分明显非保守。
所以不能把有限网络下的 PFR correction 严格等同为某个静态 log-density ratio 的梯度。
该结论来自 4 个样本、每点 2 个 probe 的机制筛查，绝对数值仍应视为 pilot evidence。

### 6.6 确定性 path-evidence proxy 不能解释 PFR

与 PFR revision 等 RMS 的 path-evidence proxy 得到 FID-1K `72.7186`，反号为
`92.3528`，均显著差于 ordinary IG `64.8509`；PFR 保留与该 proxy 正交的部分为
`63.5992`，只保留平行部分为 `64.3140`。因此当前这一阶确定性 Feynman-Kac/Doob proxy
不是 PFR 的主导方向。

## 7. 当前仍成立的最小机制描述

在弱头自身隐含 score 的固定时间线积分下，projected 位移使弱头 log-density 增加：
跨 10 个时间点平均每维增量为 `+0.00682`，94.53% 的样本为正；反向位移为
`-0.00696`，正比例为 0；正交和 donor 位移均接近 0。这个符号对照非常干净，说明
projected spatial query 的确是一个**与候选修正对齐的 weak-high-density query**。

但 state-only 位移同样有 `+0.00624` 和 94.53% 正比例，却把 FID 从 `64.8513` 恶化到
`65.3742`。因此“查询点在 weak density 下更高”不是充分条件；时间推进是必要组成。

综合现有证据，当前最保守且不自相矛盾的描述是：

\[
\boxed{
\text{PFR 是一个有限时空反事实弱参考：先提高查询的信息时间，}
\text{再用候选对齐的高密度位移选择其空间分支。}
}
\]

这个描述解释了查询是如何被构造和哪些成分具有因果作用，但**尚未形成一个能够从第一
原理推出 FID 改善的闭合定理**。特别是 state-only 反例要求任何后续理论同时解释时间
转移、空间选择与闭环 rollout，而不能只解释其中一个局部统计量。

## 8. 数据索引

本轮紧凑数据位于
`docs/data/pfr_mechanism_audit_20260903/`。其中：

- `candidate_theories/`：DID、scale-space、cross-time、posterior-pressure、reference
  ensemble 与 hierarchy 的 FID 汇总；
- `query_response/`：path-evidence、lambda sweep、query controls、response split 和
  information-clock 对照；
- `diagnostics/`：teacher MSE、projection identity、终端 ADM 分布、conservativity、
  weak secant 与 weak-density line integral。

归档未包含模型权重、生成图片、样本 NPZ、ADM activation 或多 MB 的逐样本表。
