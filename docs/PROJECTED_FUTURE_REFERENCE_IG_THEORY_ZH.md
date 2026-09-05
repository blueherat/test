# Projected Future-Reference Internal Guidance

> **2026-09-03 理论与 RNG 勘误。** “reference staleness / 更匹配未来状态”现已被
> characteristic-query 与 teacher-MSE 实验否定，不再是推荐解释；推荐主线是
> counterfactual weak-reference residualization 的 terminal-risk analysis，包括
> boundary、schedule-jump 与 cross-Jacobian 分解，见
> [`PFR_COUNTERFACTUAL_RESIDUAL_THEORY_ZH.md`](PFR_COUNTERFACTUAL_RESIDUAL_THEORY_ZH.md)。
> 本文历史 FID-5K `seed=0/1` 又共享 4992/5000 个样本，故 paired 方法差有效，但
> “两个独立 seed”无效；新的不重叠 bank 得到 ordinary/PFR `40.7983/37.6459`。

## 1. 研究目标

这条方法不再从经验式

\[
G+\eta[W(z,t)-W(z+\alpha hG,t+h)]
\]

出发调 `alpha/eta/H`，而是从 internal head 的层级结构出发，只保留一个
lookahead 时间尺度。核心直觉是：

> 普通 IG 使用的是施加 refinement 之前的弱参考。先把 refinement 以最小失真的
> 方式写入一个前瞻 latent，再让弱头读取它，便得到与即将访问状态相匹配的未来参考。

这不是 CFG 式的 conditional/unconditional fixed point，也不要求 strong 与 weak
最终相等。它校正的是 **reference staleness**。

更高层的一句话原则是：

> **一个会改变 latent 的 internal refinement，不应继续相对改变前的 internal
> reference 来定义；它应相对“该 refinement 已经写入 latent 后”读到的 reference
> 自洽地定义。**

因此方法不是给普通 IG 再加一项经验 force，而是在近似求解一个 internal refinement
的自洽方程。

## 2. 层级残差是 canonical decomposition

记当前 latent/time 为 `(z,t)`，完整模型 velocity 为

\[
S=S(z,t),
\]

depth-4 internal head 为

\[
W=W(z,t),
\]

并定义 suffix refinement

\[
\Delta=S-W.
\]

普通 IG 为

\[
G=S+\gamma\Delta.
\]

代数上它也可以精确写成

\[
\boxed{G=W+\beta\Delta,\qquad \beta=1+\gamma.}
\]

对于两个任意独立模型，这两个写法只是不同 anchor；但 internal head 有额外结构：
`W` 是较浅计算得到的 base estimate，`S-W` 是完整 suffix 相对该 base 的 refinement。
因此这里把第二种写法作为 **hierarchical residual assumption**，而不是宣称它由纯代数
唯一决定。

这一假设有可证伪预测。若从 strong-anchor 写法

\[
G=S+\gamma(S-W)
\]

构造同样的前瞻 innovation，效果应当不同。FID-1K 实验确实得到：

| anchor | FID-1K |
|---|---:|
| hierarchical weak base，系数 `1+gamma` | **61.9244** |
| strong base，系数 `gamma` | 63.5911 |

所以当前数据支持 internal hierarchy 的定向 residual 解释；它不自动推广到任意
AutoGuidance strong/weak pair。

## 3. 用最小空间干预构造 counterfactual latent

层级 refinement 对 velocity 的贡献为

\[
C=\beta\Delta.
\]

直接用 `z+hC` 查询未来弱头会构造一条不一定与当前生成方向相容的虚拟路径。当前实际
部署的场是 `G`，所以只允许在它张成的正向 tangent ray 中编码 calibration：

\[
\mathcal K_G=\{aG:a\ge0\}.
\]

选择与 `C` 最近的 forward-ray 空间表示：

\[
\boxed{
a^*=\arg\min_{a\ge0}\|aG-C\|_2^2
=
\left[
\frac{\langle C,G\rangle}{\|G\|_2^2}
\right]_+.
}
\]

这是闭式凸投影，不是待扫描的系数。每个样本得到自己的 `a*`。前瞻查询点为

\[
\boxed{
q_h(z,t)=\left(z+h a^*G,\ t+h\right).
}
\]

它不能被称为真实 characteristic future：空间只推进 `h*a*G`，而 information time
推进完整的 `h`。它是在当前 forward direction 内以最小空间失真表示 refinement，随后
让 weak head 在更高信息量的 diffusion time 读取这个 counterfactual latent。

不做这一空间投影、直接使用 `z+hC` 的 FID-1K 为 `62.6703`；闭式投影后为
`61.9244`。这支持的是 minimum spatial intervention，而不是某个固定 `alpha`。

## 4. 未来参考，而不是额外 force

让弱头读取前瞻状态：

\[
W^+=W(q_h(z,t)).
\]

然后只把普通 IG 中的 base reference 换成这个 anticipated reference：

\[
\boxed{
G_{\mathrm{PFR}}
=W+\beta(S-W^+).
}
\]

它与普通 IG 的关系是精确恒等式：

\[
\boxed{
G_{\mathrm{PFR}}
=G+\beta(W-W^+).
}
\]

因此没有额外的 `eta`。未来项的系数由原 IG 的 residual coefficient `beta` 唯一确定。

该构造有三个基本性质：

1. `h=0` 时 `W+=W`，严格退化为普通 IG；
2. 若弱参考在该虚拟前瞻下不变，也严格退化为普通 IG；
3. 它使用有限 horizon 的真实网络查询，不依赖小扰动 Taylor 近似。

若只为解释局部极限，在光滑条件下有

\[
W^+
=W+h\left(\partial_tW+a^*J_zW\,G\right)+O(h^2),
\]

所以

\[
G_{\mathrm{PFR}}
=G-\beta h
\left(\partial_tW+a^*J_zW\,G\right)+O(h^2).
\]

这说明旧的 material-derivative 观察只是该有限前瞻构造的局部展开，不是方法本体。

## 5. 自洽 refinement 方程

固定当前 `(z,t)`，并记普通 IG 的场、refinement 与正向 tangent ray 为

\[
G_0=W+\beta(S-W),\qquad
C_0=\beta(S-W),\qquad
\mathcal K_{G_0}=\{aG_0:a\ge0\}.
\]

令 `Pi` 表示到这个固定闭凸集合的欧氏投影。一个自洽的 internal calibration 应满足

\[
\boxed{
C^\star
=
\mathcal T_h(C^\star)
:=
\beta\left[
S(z,t)-W\left(z+h\Pi_{\mathcal K_{G_0}}C^\star,t+h\right)
\right].
}
\]

它表达的不是“两个头最后应该相等”，而是：

1. calibration `C*` 会把 latent 推到一个前瞻状态；
2. 弱参考必须在这个状态重新读取；
3. 重新读取后定义出的 calibration 仍应等于原来的 `C*`。

普通 IG 的 `C0` 忽略了第 2 步，是这个方程的零次显式猜测。PFR-IG 正好执行一次
Picard 更新：

\[
\boxed{
C_1=\mathcal T_h(C_0),\qquad
G_{\mathrm{PFR}}=W+C_1.
}
\]

把 `C0` 代入后，`Pi(C0)=a*G0`，所以这与上一节的实际算法完全相同；这里没有为了
讲 fixed point 而更改方法。

这个结构还有一个简单但严格的局部保证。假设固定时间 `t+h` 的弱头关于 latent 是
`L_W`-Lipschitz。由于到闭凸集合的欧氏投影是 1-Lipschitz，

\[
\|\mathcal T_h(C_1)-\mathcal T_h(C_2)\|
\le
\beta hL_W\|C_1-C_2\|.
\]

因此当

\[
\boxed{q:=\beta hL_W<1}
\]

时，`T_h` 是收缩映射：自洽 calibration 唯一存在，并且一次 PFR 更新满足

\[
\boxed{
\|C_1-C^\star\|
\le q\|C_0-C^\star\|.
}
\]

这个定理不声称更接近 `C*` 就必然降低 FID；它严格说明的是算法为何不是任意调项：
PFR 是普通 IG 朝其自洽 internal refinement 解迈出的一次有方向、有误差收缩含义的
更新。实际网络是否落在收缩区间仍需用迭代 residual/JVP 实验验证。

### 5.1 与仓库既有 implicit-reference 实验的关系

这个 fixed-point 外壳本身不是本轮新发现。此前 FWR-IG 已研究过 strong-anchor 方程

\[
V^*=S+\gamma[S-W(z+hV^*,t+h)],
\]

也给出过同类 Lipschitz 收缩结论；进一步 Picard 到 `K=2/3` 反而比一次 query 更差。
因此不能把“收敛到 reference fixed point”本身当成质量原理，也不能把本节包装成新的
fixed-point 理论。

PFR 相对既有 FWR 的实质变化只有三项：

1. 由 internal hierarchy 选择 weak-base residual，而不是 strong anchor；
2. correction 系数由精确恒等式固定为 `beta=1+gamma`；
3. calibration 先投影到当前 forward ray，再构造 information-time counterfactual。

本轮结果支持的是这三个结构选择组成的方法，而不是“隐式求解越充分，生成质量越好”。

## 6. 更直观的交换图解释

普通 IG 默认下面两件事近似可交换：

1. 在当前 state 上计算 weak-to-strong refinement；
2. 让 latent 沿生成过程进入一个稍后的、更高 SNR 的状态。

但 weak reference 本身会随 state/time 改变。因此“先 refinement，再前进”和“前进后
重新读取 weak reference”并不相同。差值

\[
\mathcal E_h[W]
=W(z,t)-W(q_h(z,t))
\]

就是有限的 reference-transport defect。PFR-IG 使用

\[
G_{\mathrm{PFR}}=G+\beta\mathcal E_h[W]
\]

补上该 defect。

一句话版本是：

> **普通 IG 对一个即将过期的 weak reference 做外推；PFR-IG 先把 refinement 放进
> latent，让 weak head 看见它，再相对这个 anticipated reference 做同一次外推。**

这与 Foresight Fixed Point 的共同 taste 是“信息必须进入 latent 后再由模型读取”，但
质量语义不同：FSG 寻找 conditional/unconditional consistency；这里利用 internal
hierarchy 修正 weak reference lag，不要求两个头相等。

## 7. 必要的因果对照

当前 FID-1K 同 bank 结果为：

| 条件 | FID-1K |
|---|---:|
| depth-4 IG | 64.8509 |
| calibration 直接写入 `z+hC` | 62.6703 |
| calibration 投影到 forward tangent ray | **61.9244** |
| 上式再去掉相对 `S-W` 的平行 revision | 61.9695 |
| 相对 `S` 投影 revision | 61.9604 |
| 相对 `G` 投影 revision | 62.0712 |
| strong-anchor、零 guidance 极限严格成立的版本 | 63.5911 |
| 关闭静态 IG，只保留 weak future mechanism | 81.0983 |

纯 strong 的配对 FID-1K 约为 `84.97`。因此 future weak mechanism 单独有正作用，但
远不足以解释完整结果；它与静态 IG 的层级 refinement 组合后才得到主要增益。

输出端正交投影的三种 axis 差异很小，当前不能把某一条投影轴宣称为独有机制。更简洁
的 full revision 版本目前还是主要方法；输出投影只保留为相关工作/机制对照。

正式 FID-5K 结果进一步支持删除输出投影：

| sampling seed | 旧 oblique 最优 | 输出正交 innovation | PFR-IG full revision |
|---:|---:|---:|---:|
| 0 | 36.504419 | 36.441546 | **36.350961** |
| 1 | 36.496253 | 36.440152 | **36.343097** |
| 均值 | 36.500336 | 36.440849 | **36.347029** |

PFR-IG 相对旧最优降低 `0.153307` FID；两个高度重叠的历史 nominal bank 数值只差
`0.007864`，但这不是独立重复。相对 depth-4 IG 的 `39.946776`，均值降低
`3.599747`。

其他 ADM 指标并非全线同向：

| 方法 | sFID-5K | IS |
|---|---:|---:|
| depth-4 IG | 69.925170 | 36.852156 |
| 旧 oblique 最优 | **69.064928** | 40.021557 |
| PFR-IG | 70.220809 | **40.669031** |

所以当前最准确的说法是：PFR-IG 是本仓库该协议下新的 **FID/IS 最优**，但不是
sFID 最优。FID 与 IS 的增益不能被包装成所有感知/覆盖指标都改善。

## 8. Horizon 的边界

当前连续 Dopri5 实现仍需要一个 lookahead cap `H`。为避免把一个细扫小数包装成理论，
正式候选固定使用

\[
H=1/32,
\]

把它解释为 32-step 名义离散化中的一个 future time quantum。邻近
`H=7/256` 的两 seed FID-5K 为 `36.5546/36.5508`，而 `H=1/32` 的正交版本为
`36.4415/36.4402`；最终 full-revision 版本进一步达到
`36.3510/36.3431`。这说明 canonical 值没有因放弃微调而吃亏。

更完整的离散 sampler 应直接使用当前 solver step

\[
h_k=t_{k+1}-t_k,
\]

从而完全删除 horizon 超参数。当前正式结果尚未验证该版本，所以不能提前声称方法已经
无任何时间尺度选择。

## 9. 与 APG 的边界

APG 将 CFG update 相对 conditional prediction 分成平行/正交分量，主要用于抑制高
guidance scale 的过饱和。这里的核心操作是：

1. 把 internal refinement 投影到 forward tangent ray，以构造 future query；
2. 用 future weak reference 替换 stale weak reference。

输出端相对 gap/strong/guided field 的正交化只是审计对照，而且三者 FID-1K 基本持平，
不应作为主要 novelty。

## 10. 当前结论边界

可以声称：

1. 三参数经验 FMD 可以被一个由层级 residual 与闭式投影导出的单 horizon 方法替代；
2. 该方法在当前 ImageNet-100 SiT-S/2 setup 上形成新的内部 FID/IS-5K 最优；
3. weak-base/strong-base、无静态 IG、无 tangent projection 等反证与理论预测方向一致。

不能声称：

1. reference-transport defect 必然是真实数据分布的下降方向；
2. 一个训练 seed、两个 sampling seed 等于跨模型 SOTA；
3. FID 改善自动意味着 sFID、precision/recall 或人类质量全部改善；
4. `H=1/32` 已由定理唯一确定；
5. 该层级 weak-base 语义可无条件推广到两个独立 AutoGuidance 模型。
