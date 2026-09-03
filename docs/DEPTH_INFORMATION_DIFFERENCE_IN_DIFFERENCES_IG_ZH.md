# Depth-Information Difference-in-Differences Internal Guidance

## 1. 为什么必须重新开始

当前 ITPR/PFR 条件取得了当前协议最好的 FID，但它不满足一个最基本的 population
consistency 要求。若 weak 与 strong 已经处处等于同一个理想场 `F`，普通 IG 严格退化为
`F`；而 ITPR 仍含有

\[
F(p)-F(q),
\]

一般并不为零。因此它的收益可能来自 shared field 的时间漂移，而不是 strong-weak
disagreement。information-time ordering 可以说明 `q` 是一种可定义的查询，却不能说明
该 shared drift 为什么应当作为 guidance。

本轮不把这个缺口藏进叙事，而把它作为新设计必须满足的反例。

## 2. 每一步都天然包含一个 2x2 因子实验

记当前 query 为 `p=(z,t)`，由既定 information intervention 得到的 query 为 `q`。在两个
query 上分别读取 weak prefix 与完整 strong model：

\[
\begin{array}{c|cc}
 & p & q \\
\hline
\text{weak depth} & W_p & W_q \\
\text{strong depth} & S_p & S_q
\end{array}
\]

这里存在两个可控因素：

1. computational depth：`weak -> strong`；
2. information intervention：`p -> q`。

当前深度 refinement 为

\[
g_p=S_p-W_p,
\]

反事实 query 上的深度 refinement 为

\[
g_q=S_q-W_q.
\]

普通 IG 使用 `g_p`。真正新的、同时依赖两个因素的量是 difference-in-differences：

\[
\boxed{
\Omega_{d\times i}
=(S_q-W_q)-(S_p-W_p)
=g_q-g_p.
}
\]

它是 prediction surface 在 depth-information 矩形上的离散混合差分。

## 3. 唯一性：为什么必须是这个矩形差分

考虑四个输出的一般线性组合

\[
L=aS_p+bW_p+cS_q+dW_q.
\]

要求它满足两种不可辨识项消去：

1. 在 `p` 或 `q` 给两头同时加入任意 shared field，`L` 不变；
2. 若 information intervention 对两种 depth 都没有作用，`L=0`。

第一条给出

\[
a+b=0,\qquad c+d=0.
\]

第二条给出

\[
a+c=0,\qquad b+d=0.
\]

这些约束的非零解只有一维。因此除整体尺度外，唯一解就是

\[
(a,b,c,d)=(-1,+1,+1,-1),
\]

即

\[
\boxed{L\propto g_q-g_p.}
\]

这不是从 FID 网格挑选出的方向，而是 2x2 表中唯一的 interaction contrast。它与统计中
difference-in-differences / two-way interaction 的 inclusion-exclusion 结构相同。

## 4. 三条一致性公理

一个由 internal disagreement 驱动的 foresight guidance 至少应满足：

### 4.1 Zero-horizon recovery

`q=p` 时必须严格回到普通 IG。

### 4.2 Diagonal consistency

若 `S=W` 处处成立，额外 guidance 必须严格为零。

### 4.3 Common-mode invariance

若在每个 query 上同时给两头加入相同场

\[
S_r\mapsto S_r+A_r,\qquad W_r\mapsto W_r+A_r,
\quad r\in\{p,q\},
\]

则额外 guidance 不应改变。公共 transport 可以改变基础生成场，但不能被误认成
strong-weak correction。

`Omega` 同时满足三条；`W_p-W_q` 只满足第一条，不满足后两条。

## 5. 由公理推出 aligned refinement transport

普通 IG 的 internal-hierarchy 写法为

\[
G=W_p+\beta g_p,
\qquad \beta=1+\gamma.
\]

不引入新 gain，只把 depth refinement 从当前 query 运输到 information query：

\[
\boxed{
G_{\rm DiD}
=W_p+\beta g_q.
}
\]

等价地，

\[
\boxed{
G_{\rm DiD}
=G+\beta\Omega_{d\times i}
=G+\beta(g_q-g_p).
}
\]

这提供了两个互补解释：

1. `transport`：基础 transport 仍由当前位置的 weak prefix `W_p` 给出，只把 suffix
   refinement 更新为在反事实信息条件下读取的 `g_q`；
2. `interaction`：在普通 IG 上只加入 depth 与 information 两个因素的混合效应，纯时间
   漂移和纯深度主效应都被 inclusion-exclusion 消去。

与当前 ITPR 的跨角点

\[
W_p+\beta(S_p-W_q)
\]

不同，新式子始终在同一个 query 上比较 `S_q-W_q`，不会把 `S_p` 与 `W_q` 的公共漂移
混进 disagreement。

## 6. Query 只负责施加 intervention

为把理论变化限制在“读什么”而不是“如何造 query”，先沿用已经锁定的 projected query：

\[
G=W_p+\beta(S_p-W_p),
\qquad C=\beta(S_p-W_p),
\]

\[
a^*=\left[\frac{\langle C,G\rangle}{\|G\|^2}\right]_+,
\qquad
q=(z+h a^*G,t+h).
\]

这样当前 ITPR 与新 DiD 条件拥有完全相同的 `p/q`，唯一变化是跨角点 weak revision
是否被替换为同点 depth interaction。

## 7. 预注册的因果对照

固定 checkpoint、普通 IG schedule、`h=1/32`、projected query、noise/labels 与评估流程，
先比较：

1. `cross-corner`：当前 ITPR，`W_p+beta*(S_p-W_q)`；
2. `aligned weak-base`：理论主条件，`W_p+beta*(S_q-W_q)`；
3. `aligned strong-base`：`S_p+gamma*(S_q-W_q)`；
4. `anti-interaction`：`W_p+beta*(2*g_p-g_q)`；
5. `time-only aligned`：不做空间 intervention，只在 `(z,t+h)` 读取 `g_q`。

在读取结果前给出判据：

- 若 `aligned weak-base` 最好或至少追平 cross-corner，说明 common-mode-invariant mixed
  difference 同时获得理论与质量支持；
- 若 `anti-interaction` 优于正向，则 interaction 的符号推导失败；
- 若 cross-corner 仍显著最好，则 shared weak drift 是当前质量收益不可删除的一部分，
  DiD 理论只能作为一个合理性约束，不能作为现有最佳方法的质量解释；
- 若所有 aligned 条件均退化，则停止把 information query 包装成 disagreement guidance。

FID 不是定理的前提。上述排序是理论的可证伪预测，而不是定义。

## 8. FID-1K 结果：该理论预测被否定

固定 ImageNet-100 SiT-S/2 `v800`、同一个 depth-4 weak head、相同 noise/labels、
`h=1/32` 和 projected query 后，四个预注册条件得到：

| 条件 | FID-1K | sFID | IS |
|---|---:|---:|---:|
| `weak_gap_transport_time_only` | 68.7665 | 217.3625 | 34.5091 |
| `weak_gap_transport_projected` | 68.2483 | 216.3792 | 33.8345 |
| `strong_gap_transport_projected` | **66.5900** | 211.6714 | 35.1405 |
| `weak_gap_antitransport_projected` | 66.8832 | 207.7589 | **36.2024** |

同协议普通 depth-4 IG 的 FID-1K 约为 `64.85`，cross-corner ITPR/PFR 约为
`61.9`。四个 common-mode-invariant 条件全部明显更差，因此没有扩到第二个 sampling
seed 或 FID-5K。

这组结果否定了本报告第 5 节的质量假设：虽然 `g_q-g_p` 是唯一满足三条一致性公理的
线性交互项，但这些公理并不是高质量感知 guidance 应满足的充分设计原则。尤其，现有最佳
方法中被该理论视作 nuisance 并删除的 shared weak-field evolution，实际上是收益不可删除
的主要部分。

因此必须严格区分：

1. **代数结论仍成立**：矩形差分是唯一消除 query-wise common mode 的线性交互项；
2. **质量解释不成立**：生成质量并不要求额外场在 `S=W` 的 population diagonal 上归零；
3. **后续边界**：若一个方法在 `S=W` 时仍非零，就必须明确把它解释为有意改变目标分布
   的 guidance（例如有限强度密度比/温度倾斜），不能继续称为有限模型误差修正。

原始数据位于：

`/data/users/zhoushunyu/eqvae/imagenet_sit_flow/depth_information_did_ig_v1/fid1k_seed0/summary/all_conditions.csv`。
