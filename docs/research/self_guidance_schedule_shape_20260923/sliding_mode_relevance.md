# 滑模控制与当前 signed self-guidance：哪些结构真的相通

日期：2026-09-23。仅理论与原始文献核查；未修改训练代码，未运行 GPU。

当前模型写成

\[
\dot X_t=S(X_t,t)+a(t)B(X_t,t),\qquad B=S-W,
\]

其中时间从噪声走向数据，当前被优化的是所有样本共享的标量时间增益。整个速度场依赖状态，但增益没有根据单个样本的状态在线调整。我们的训练目标是终点分布，而不是一条已知参考轨迹。

**判断：滑模有值得借鉴的局部控制结构，但两次平滑反号不能证明已有滑模或二阶振荡机制。更有价值的问题是：对什么可观测量，当前受限方向还有多少控制能力？**

## 1. 对当前 actuator，滑模公式可以精确写出来

选一个明确的、光滑的标量面 \(s(x,t)=0\)，定义

\[
A_s=\partial_t s+\nabla s^\top S,\qquad
b_s=L_Bs=\nabla s^\top B.
\]

沿实际轨迹有恒等式

\[
\dot s=A_s+b_s a.
\]

在 \(b_s\ne0\) 的区域，输出 \(s\) 对输入 \(a\) 的 relative degree 为 1。保持在面上的 equivalent control 是

\[
a_{\rm eq}=-A_s/b_s.
\]

若允许**样本级状态反馈**，指定 reaching law \(\dot s=-k\operatorname{sign}(s)\)，可以取

\[
\boxed{a(x,t)=-\frac{A_s+k\operatorname{sign}(s)}{b_s}.}
\]

于是 \(V=s^2/2\) 满足 \(\dot V=-k|s|\)。在解存在、该区域内分母不退化且无饱和的条件下，到达时间不超过 \(|s_0|/k\)。面上要使用相应的滑动解/等效控制解释，而不是把 \(\operatorname{sign}(0)=0\) 当作完整的不连续系统理论。这是经典滑模构造，不是生成模型新定理。[Utkin, 1977](https://doi.org/10.1109/TAC.1977.1101446)

若实际输出动力学再有扰动 \(d_s\)，且 \(|d_s|\le\Delta<k\)，则 \(\dot V\le-(k-\Delta)|s|\)。保证针对该输出的扰动界，不能直接扩大为任意生成误差的分布鲁棒性保证。

### 小分母意味着什么

当 \(|a|\le a_{\max}\)，即时可实现的输出速度为

\[
\dot s\in[A_s-a_{\max}|b_s|,\ A_s+a_{\max}|b_s|].
\]

在 \(s>0\) 时至少要有 \(A_s-a_{\max}|b_s|<0\)，在 \(s<0\) 时至少要有 \(A_s+a_{\max}|b_s|>0\)，才能朝面运动。想保持固定到达速度，还需要相应的严格裕量。

因此 \(B\to0\)，或者 \(B\) 与 \(\nabla s\) 趋于正交，都可能使 \(b_s\to0\)。除非分子同步缩小，否则上述公式会要求发散增益；在 \(b_s=0\) 时，输入对该输出**即时一阶失效**。这不等于整个非线性系统永远不可控，但 relative-degree-1 公式已失效，不能用加大 gain 掩盖。

若末端 \(b_s\sim1-t\)，固定强度 reaching law 会要求 \(a\sim(1-t)^{-1}\)。这是末端大系数的一种机制，不是当前曲线已经验证的成因。若 \(A_s\) 和 reaching 强度也同步消失，增益可以有界。真实 CFG 的某些光滑桥满足末端差方向衰减；内部 \(S-W\) 必须另测，不能直接套用。

## 2. “二阶”“高阶滑模”“terminal”是三个不同问题

若输出是 \(e(x,t)\)，常见二阶跟踪面 \(s=\dot e+\lambda e\) 需要核查输入相对阶。对我们的一阶受限系统，通常

\[
\dot e=\partial_t e+\nabla e^\top S+a\nabla e^\top B
\]

已经含有输入。因此 \(s\) 本身通常含 \(a\)，再求 \(\dot s\) 就含 \(\dot a\)。要把它作为标准状态滑模面，需建立合适的扩展系统，例如把 \(a\) 纳入状态、以 \(\dot a\) 为新控制，或证明输出确实具有更高 relative degree。不能只对 ODE 再求一次导数，就声称存在独立的质量—阻尼二阶动力学。

高阶滑模约束的是 \(s,\dot s,\ldots\) 的消失阶数；它并不要求原系统就是机械二阶系统。它可以降低实际控制的抖振，但有输入增益、相对阶、扰动正则性等条件，并不自动提供终点分布优化或省掉反传。[Levant, 2003](https://cris.tau.ac.il/en/publications/higher-order-sliding-modes-differentiation-and-output-feedback-co/)

Terminal sliding mode 中的 terminal 指特定误差动力学的有限时间到达性质，不能与本任务的 terminal distribution objective 混同。

Chattering 指为维持滑动而发生的高频状态驱动切换。当前平滑 schedule 的两次反号既不等于 chattering，也不是存在滑动面的证据。将 sign 换成 saturation/tanh 可形成有限厚度边界层；通常要把严格有限时间滑动结论改为邻域或渐近结论。

## 3. 最大的缺口是面，而不是控制律

可以形式上设 \(s(x,t)=\ell(x,t)-c(t)\)，但必须解释 \(\ell\) 的意义。当前干净图像判别器并不自动成为所有噪声时刻的质量函数。构造有效的时间条件 value/feature，或经过 clean prediction 使用它，都有估计误差与额外局部导数成本。

更本质地，把所有样本推向同一个 feature level 不保证正确的数据分布。若大量初始状态在有限时间被压到一个低维面，甚至可能与希望保持的分布多样性冲突。\(s\to0\)、\(B\to0\)、conditional 与 unconditional 预测接近，都不能单独认证真实感与分布覆盖。

对向量面 \(s\in\mathbb R^m\)，当前标量输入的即时作用是一个 \(m\times1\) 向量；不能泛化地独立满足所有坐标的 reaching law。即使只取标量面，不同样本也可能需要相反的增益。一个共享 \(a(t)\) 不能同时实现这些反馈要求。

## 4. 已有非常直接的 SMC-CFG prior

[CFG-Ctrl, CVPR 2026，最新版 v2](https://arxiv.org/html/2603.03281v2) 的算法 1 已使用

\[
e=v_c-v_u,\quad s\approx\Delta e+\lambda e_{\rm prev},\qquad
v_{\rm new}=v_u+w[e-k\operatorname{sign}(s)].
\]

sign 是逐坐标向量，因此一般改变 guidance 方向，不能写成我们当前的单个增益乘 \(S-W\)。论文报告了 SD3.5、Flux、Qwen-Image 的实验。因此“用滑模改 CFG”本身已有明确先例；我们能研究的是受限 actuator、实际终点分布目标及其可实现性。

以下是**独立数学审查**，用于限制能从该文继承的定理，不否定其经验结果：

- §3.4 从矩阵最小奇异值下界直接推出逐坐标 sign 控制的下降方向，条件不充分。\(\Gamma=-I\) 有正的最小奇异值，却使 \(s^\top\Gamma[-k\operatorname{sign}s]=k\|s\|_1>0\)。需要控制方向条件，或正确的增益逆/方向补偿。
- 附录 §6.3 使用更强的正向 dominance 假设：\(\Gamma=w\nabla e=wI+\Delta\Gamma\)，\(\|\Delta\Gamma\|\le\rho\)，\(w>\rho\sqrt d\)。在该定义下，条件要求 \(\|\nabla e-I\|<1/\sqrt d\)；增大 \(w\) 不会自动使它成立，因为偏差也乘 \(w\)。
- 对其一阶输入系统，\(s=\dot e+\lambda e\) 的严格求导通常含控制导数。需要额外论证才能得到论文使用的 \(\dot s=\Phi+\Gamma\Delta e\) 形式。其带历史差分的离散算法更不能直接视为我们连续 scalar 系统的精确稳定性定理。

一个最简单的核对是 \(e(x)=x\)、\(\dot x=w(x+u)\)：有 \(s=(w+\lambda)x+wu\)，所以 \(\dot s=(w+\lambda)w(x+u)+w\dot u\)。最后一项不能一般丢弃。

## 5. 更贴近现有 schedule 的路线：控制分布矩，并量化反馈的价值

下面是我们针对当前系统的推导，不依赖前述论文。对实际 rollout 分布 \(q_t\)，选一个具有明确含义的可观测量

\[
M(t)=\mathbb E_{q_t}\phi_t(X),\quad
s_q(t)=M(t)-m_*(t),\quad b(x,t)=\nabla\phi_t(x)^\top B(x,t).
\]

在可交换积分与微分的条件下，精确地有

\[
\dot s_q=A_q+\mathbb E[a(X,t)b(X,t)],\quad
A_q=\mathbb E[\partial_t\phi+\nabla\phi^\top S]-\dot m_*.
\]

若坚持共享增益，\(\dot s_q=A_q+a(t)\mathbb E b\)。于是可以对**群体矩误差**设计 reaching law；这是群体反馈，不是每个样本都到达同一滑模面。只约束一个矩，当然还远未匹配整个分布。

更有信息量的是一个经典 Cauchy/Riesz 投影结论。假设想实现瞬时修正 \(\mathbb E[ab]=\delta\)，并最小化 \(\mathbb E[g a^2]/2\)，其中 \(g(x,t)>0\)。

| 增益 family | 最优增益 | 最小代价 |
|---|---|---|
| 共享 \(a(t)\)，且 \(\mathbb E b\ne0\) | \(\delta/\mathbb E b\) | \(\delta^2\mathbb E g/[2(\mathbb E b)^2]\) |
| 状态反馈 \(a(x,t)\)，且 \(\mathbb E[b^2/g]>0\) | \(\delta b/[g\mathbb E(b^2/g)]\) | \(\delta^2/[2\mathbb E(b^2/g)]\) |

由 \((\mathbb E b)^2\le\mathbb E g\,\mathbb E(b^2/g)\)，状态反馈的最优代价不高于共享 schedule。尤其当 \(\mathbb E b=0\) 而 \(\mathbb E(b^2/g)>0\)，**共享 schedule 对这个矩失去一阶控制力，样本级 gain 仍然可控。** 失败原因是样本响应相互抵消，而不一定是 \(B\) 太小。

这提供可检验诊断：比较时间曲线 \(\mathbb E b\) 与 \(\mathbb E b^2\)，而不是只看 scale 的正负。若要解释真实速度能量，取 \(g=\|B\|^2\)，并在 \(B=0\) 处以零贡献处理；直接取 \(g=1\) 则只是系数代价，会依赖 \(B\) 的归一化。

这是已有线性约束最小二乘在本任务的应用，不报新理论。它提示一个具体研究缺口：**先证明共享时间增益受到样本响应抵消限制，再研究小型状态反馈是否解除该限制。** 增益公式只需要当前局部可观测量，但选择 \(\phi_t\) 和目标矩路径是否真的改善终点目标仍是核心；用有限 batch 估计也不能直接继承无噪声有限时间保证。

## 6. Barrier 更适合明确的约束，而不是替代分布学习

已有 [Constricting Barrier Functions，最新版 v3](https://arxiv.org/html/2602.21429v3) 对生成采样施加逐步收紧的约束，通过状态反馈 QP 作最小修正。其离散保证针对精确一步 barrier 条件；线性化 QP 与精确条件需要区分。该先例解决指定集合约束，不认证终点分布质量。

对我们的 scalar family，若指定 \(h(x,t)\ge0\)，连续时间条件是

\[
\partial_t h+\nabla h^\top S+a\nabla h^\top B\ge-\alpha(h).
\]

可将 nominal schedule 投影到这一标量不等式与增益限幅的交集；该交集可能为空。这是比“直接加 sign”更透明的可行性检查，但仍需要有效的约束函数。实际 Heun sampler 的保证还需核查 \(h(F_i^a(x),t_{i+1})\)，不能仅凭连续条件给出。

当前最有价值的下一步是选择少量可解释观测量，估计其受限 Lie derivative、跨样本抵消与末端退化，再决定是否值得扩展增益 family。仅凭 +−+ 曲线给系统命名为滑模、振子或有限时间稳定系统，理论信息仍然不足。
