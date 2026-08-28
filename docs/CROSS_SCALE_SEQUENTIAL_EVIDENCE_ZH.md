# 跨尺度序贯路径证据：理论边界与复现实验协议

## 当前结论

这份文档现在是已经完成前瞻检验的理论与实验历史，不再是仓库唯一主线。
有限步路径似然比的鞅结论仍然成立，但冻结的高噪声质量备择在 64 条独立
cross-prefix 轨迹上没有触发，并且发现集中的排序优势没有复现，因此该具体
候选已经退役。当前主线改为更宽的 bad/good 轨迹指标发现，见
[`BAD_GOOD_TRAJECTORY_METRICS_ZH.md`](BAD_GOOD_TRAJECTORY_METRICS_ZH.md)。除非新数据
独立支持某个跨尺度量，否则本文的跨尺度量只作为候选指标和失败对照，不再
被默认解释为伪影机制。

主假设不是“跨尺度差分本身能识别所有坏图”，而是下面这个可证伪命题：

> 对冻结的随机生成器 (P)，把更高噪声或更弱预测定义成同一历史上的
> 可预测备择转移核 (Q)。实现级路径似然比 (E_k=dQ_{0:k}/dP_{0:k})
> 是非负鞅。若坏样本在 (E_k) 的 anytime-valid 越界事件中显著富集，
> 才允许用完整重启或受限回滚改变这些轨迹。

其中最后一句是实验假设，不是前两句的数学推论。实现级 (Q/P) 比值即使
完全校准，也只说明轨迹更像预先写下的 (Q)；只有当 (Q) 近似“坏图条件下
实际会走的路径分布”时，它才有质量检测功效。

路线评级为 **Conditional GO**。理论给出校准的干预预算，不能独自保证图像
质量；质量结论需要分别通过“找得准”和“修得好”两道实验门槛。

## 一、对象、字母和严格结论

### 1. 连续热流只是动机

令：

- (v)：规范化的前向热扩散方差；(v) 越大，噪声越强；
- (p_v(x))：数据在噪声尺度 (v) 下的密度；
- Δ：固定的正尺度偏移；
- (q_v(x)=p_{v+Δ}(x))：更高噪声的反事实密度；
- (H_v(x)=q_v(x)/p_v(x))：单点跨尺度密度比；
- (s_v(x)=\nabla_x\log p_v(x))：尺度 (v) 的 score；
- θ_v(x)=\nabla_x\log H_v(x)=s_{v+Δ}(x)-s_v(x))：密度比对状态的敏感方向；
- τ：反向生成时钟，(v_τ=v_{\max}-τ)；
- (Y_τ)：沿 (p_v) 运行的反向热扩散状态；
- (B_τ)：标准 Brownian motion。

若 (p_v,q_v>0)、足够光滑且都满足

\[
\partial_v p_v=\frac12\Delta_xp_v,
\]

则

\[
\partial_vH_v=\frac12\Delta_xH_v+s_v^\top\nabla_xH_v.
\]

沿反向过程

\[
dY_τ=s_{v_τ}(Y_τ)dτ+dB_τ
\]

应用 Itô 公式得到

\[
dH=H\,θ^\top dB_τ,
\qquad
d\log H=θ^\top dB_τ-\frac12\lVertθ\rVert^2dτ.
\]

因此 (H) 在适当可积性条件下是真鞅；一般至少是非负局部鞅，从而是
非负超鞅。这里：

- (H) 是累计证据；
- θ 是证据对当前状态的方向导数；
- θ 的平方范数是 log-likelihood ratio 的二次变差速率；
- (\frac12\lVertθ\rVert^2) 才是局部 KL 漂移率；
- 它们都不是普适的“伪影分数”。

如果 (P,Q) 从不同初始边缘开始，路径比要包含初始密度比。如果实现让二者
共用同一个高斯初态，应使用转移核的离散 likelihood ratio，不能直接把终点
(H) 当成完整路径的 Radon–Nikodym 导数。

### 2. 实际论文主定理应放在离散随机采样器上

令：

- (x_k)：第 (k) 个生成状态；
- \(\mathcal F_k\)：看到 (x_k) 为止的全部历史；
- (P_k=\mathcal N(\mu_k,\Sigma_k))：冻结 baseline 的实际一步转移核；
- (Q_k=\mathcal N(\widetilde\mu_k,\Sigma_k))：在同一历史上、观察下一状态前
  计算的高噪声弱备择；
- \(\delta_k=\widetilde\mu_k-\mu_k\)：两核的均值差；
- ℓ_k：累计 log-likelihood ratio；
- (E_k=\exp(\ell_k))：e-process，也就是路径似然比财富；
- α：允许 baseline 轨迹被触发的总预算；
- τ_α：首次达到 (E_k\ge1/α) 的停止时刻。

为避免特殊字体混淆，均值差直接写成

\[
\delta_k=\widetilde\mu_k-\mu_k.
\]

观测 (x_{k+1}) 后，精确增量为

\[
\Delta\ell_k
=\log\frac{q_k(x_{k+1}\mid\mathcal F_k)}
             {p_k(x_{k+1}\mid\mathcal F_k)}
=\delta_k^\top\Sigma_k^{-1}(x_{k+1}-\mu_k)
-\frac12\delta_k^\top\Sigma_k^{-1}\delta_k.
\]

只要 μ_k、\(\widetilde\mu_k\)、Σ_k 都在观察 (x_{k+1}) 前由
\(\mathcal F_k\) 决定，且 (Q_k\ll P_k)，就有

\[
\mathbb E_P[E_{k+1}\mid\mathcal F_k]=E_k.
\]

必须把两种“精确”分开：连续热流中的 \(H_v=p_{v+\Delta}/p_v\) 是理想边际
密度比；这里的 \(E_k\) 是两个**实际实现的操作性 Markov kernels** 之间的记账
精确。learned、discretized sampler 中后者一般不等于前者，也不会自动继承
Self-Guidance 的尺度语义。共用初始噪声分布时 \(E_0=1\)；若 (P,Q) 的初始
分布不同，必须把初始 Radon--Nikodym 比纳入 \(E_0\)。

这相对于“由代码当步保存的 \(\mu,\Sigma\) 所定义的两个 Gaussian 核”精确成立，
不要求网络 score 正确，也不要求 CFG 很小。这里和通常数值 SDE 文献一样，把
这些浮点参数视作实数；若把最终 FP32 加法舍入本身也当成离散随机核，连续
Gaussian 密度不再是机器状态的字面概率质量。实现直接使用实际抽到的 innovation
计算 noise-form LR，避免由 `x_next-mu` 反推噪声的消减误差，但论文不得把这层
数值约定夸成对浮点格点核的测度论恒等式。简式要求两核共享协方差；若协方差
不同，必须使用包含 log-determinant 的完整 Gaussian likelihood ratio。

在 (P) 下，条件均值为

\[
\mathbb E_P[\Delta\ell_k\mid\mathcal F_k]
=-\frac12\delta_k^\top\Sigma_k^{-1}\delta_k.
\]

所以高维实验要监控每步可预测 KL 预算

\[
\kappa_k=\frac12\delta_k^\top\Sigma_k^{-1}\delta_k.
\]

否则备择均值差可能让 likelihood ratio 在一两步内数值退化。备择的任何
KL-budget 缩放也必须在观察下一状态前固定。

这里最大的工程风险不是校准错误，而是**校准正确却没有检测功效**。记

\[
D_k=\delta_k^\top\Sigma_k^{-1}\delta_k=2\kappa_k.
\]

同协方差 Gaussian 核下，在 (P) 中条件精确地有

\[
\Delta\ell_k\mid\mathcal F_k\sim
\mathcal N\!\left(-\frac12D_k,D_k\right),
\qquad
\mathbb E_P\!\left[\left.
\left(\frac{E_{k+1}}{E_k}\right)^2\right|\mathcal F_k\right]
=e^{D_k}.
\]

因此典型路径的 log evidence 会按 \(-\tfrac12\sum_kD_k\) 下沉，而二阶矩每步
乘上 \(e^{D_k}\)；期望为 1 可能完全由极罕见巨权重维持。若以可预测规则强制

\[
K_{\rm total}=\sum_k\kappa_k\le K_*
\]

逐路径成立，则

\[
Z_k=E_k^2\exp\!\left(-\sum_{i<k}D_i\right)
\]

是非负鞅，并有

\[
\mathbb E_P[E_T^2]\le e^{2K_*}.
\]

对固定权重 mixture，若每个分量都满足同一总预算，则由平方函数的凸性也有相同
的二阶矩上界。这给出比“每步 cap”更直接的 collapse 控制。旧的每检查步
\(\kappa=0.2\) 保留为已经锁定的 reference 配置，不因 smoke 结果事后改写；
但它在约 30 个 active checkpoint 上允许累计 \(K\) 接近 6，对应极宽松的二阶矩
上界 \(e^{12}\)。新的探索性分支预先固定
\(K_*\in\{0.5,1.0\}\)，用事先声明的 checkpoint allowance 将总预算分配到
整条路径；任何剩余预算和 tempering 都只能依赖当前历史，不能看下一步
innovation。它们分别把单分量终点二阶矩上界压到 \(e^1\) 和 \(e^2\)。这两个
预算仍只是 discovery 候选，不能在 confirmation 中择优。

首轮实现采用最简单、可复算的分配：对尺度分量 \(d\)，若预定有 \(m_d\) 个
non-identity checkpoint，则每步 allowance 固定为 \(b_k=K_*/m_d\)，并在观察
innovation 前取

\[
\gamma_k=\min\!\left(1,\sqrt{b_k/K_k^{(0)}}\right).
\]

当 \(K_k^{(0)}=0\) 时约定 \(\gamma_k=1\)。
未用完的 allowance 不向后挪用，因而逐路径必有
\(\sum_kK_k\le K_*\)。每条路径同时报告 raw/capped 单步 \(K_k\)、累计 \(K\)、
log-E 分位数、cap saturation、二阶矩诊断和跨路径 ESS；只看
\(\mathbb E[E]\) 是否接近 1 不足以判断功效。

### 3. 真正的质量证据是什么

先把终点质量目标直接写进概率空间。令：

- (K)：采样的最后一步；
- (B\in\mathcal F_K)：预先定义的终点坏事件，例如明显的模糊、融合或
  肢体/物体拓扑错位；
- (a=P(B))：冻结 baseline 下这种坏事件的发生率，并假定 (0<a<1)；
- (h_k=P(B\mid\mathcal F_k))：走到第 (k) 步后，这条 prefix 最终变坏的
  条件风险；
- (M_k^B=h_k/a)：相对初始坏图率归一化的质量风险证据。

令 (\mathcal F_{-1}=\{\varnothing,\Omega\}) 表示抽取任何 sampler 随机性之前的
平凡滤过，并定义 (h_{-1}=a)、(M_{-1}^B=1)。对 (k\ge0) 使用上面的 (h_k,M_k^B)。

由条件期望的塔式法则，

\[
\mathbb E_P[h_{k+1}\mid\mathcal F_k]=h_k,
\qquad \mathbb E_P[M_k^B]=1,
\qquad M_{-1}^B=1.
\]

若 (\mathcal F_0) 已经包含随机初态 (x_0)，一般只有
(\mathbb E[M_0^B]=1)，不能写成 (M_0^B=1)。

所以 (M_k^B) 是有界非负 (P)-鞅。对任意 (A\in\mathcal F_k)，

\[
P(A\mid B)
=\frac{P(A\cap B)}{a}
=\frac{\mathbb E_P[1_Ah_k]}{a}
=\mathbb E_P[1_AM_k^B].
\]

因此严格地说，

\[
M_k^B
=\left.\frac{dP(\cdot\mid B)}{dP}\right|_{\mathcal F_k},
\qquad
M_K^B=\frac{1_B}{a}.
\]

这给出了理论上真正该逼近的对象：不是任意跨尺度差分，也不是任意轨迹
不稳定性，而是“终点失败条件路径律”相对普通路径律的 likelihood ratio。
实际无法在采样中直接计算 (P(B\mid\mathcal F_k))，因为 (B) 需要看到终图，
所以必须构造可计算的代理备择 (Q)。高噪声跨尺度 (Q) 是否合理的可证伪条件
正是：它的路径比是否在独立样本中逼近或至少排序 (M^B)。实现级精确校准
不能替代这一步功效验证。

这个推导还暴露了 α 的一个硬约束。因为 (0\le h_k\le1)，

\[
0\le M_k^B\le\frac1a.
\]

若 α<a，则 Ville 阈值 (1/\alpha) 甚至高于理想质量鞅的最大值 (1/a)，
连完美的 (M^B) 都不可能越界。于是很小的 α 只能针对发生率不超过该预算的
更稀有灾难子事件；它不可能同时以高召回率清除一个比 α 常见得多的宽泛
“轻微瑕疵”类别。这与“α 控制总体干预率，而不是好图条件误报率”的结论一致。

固定 prefix (H_J=h_J) 的后缀实验必须在条件律
(P(\cdot\mid H_J=h_J)) 下重新定义
(a(h_J)=P(B\mid H_J=h_J)) 和质量鞅；该条件质量鞅从 1 开始。上面的
(\alpha<a) 不可越界结论也必须使用当前实验总体（全路径或固定 prefix 条件总体）
对应的 prevalence，不能把二者混用。

若失败包含多个机制 (B_m)，则理想备择也通常不是一条方向，而是多个
(P(\cdot\mid B_m)) 的固定先验 mixture。发现阶段可以学习或设计多个低维、
可预测代理核；确认阶段必须冻结其方向、尺度、起始时刻、局部区域和 mixture
权重。看完当前 innovation 后再选择能让证据最大的方向，不再具有声明的
一步 likelihood-ratio 校准。
只有当 (B_m) 构成 (B) 的互斥分解且 mixture 权重取
(P(B_m\mid B)) 时，这个 mixture 才精确等于 (P(\cdot\mid B))；否则它只是一个
固定、仍可校准但必须另验质量功效的代理备择。

在 baseline 的标准化 Gaussian innovation
(Z_{k+1}\mid\mathcal F_k\sim\mathcal N(0,I)) 坐标中，给定采样前即可观察的
低维特征 (f_k)，若把代理 (Q) 限制为单位协方差 Gaussian 均值偏移族，则
使坏路径条件期望 log-LR 最大的 moment-matching 方向为

\[
\lambda_k^B(f)
=\mathbb E_P[Z_{k+1}\mid f_k=f,B].
\]

它解释了应该怎样比较理论和数据：如果学到的坏路径方向与跨尺度 score-gap
方向对齐，支持原来的高噪声机制；若近似正交或符号相反，则跨尺度差异可能只是
一个状态特征，而不是合适的干预核。方向可在独立 discovery 数据上学习并冻结；
冻结后构成的 Gaussian exponential likelihood ratio 仍可校准，但其质量功效
必须在新 seed 上检验。

若 discovery 显示同一种跨尺度轴上的正、负两个方向都可能对应失败，也不能在
看到 innovation 后逐步选择符号。一个仍然精确、但语义更弱的预注册备择是：在
后缀开始前以固定先验选择一次符号，并在整条路径保持该符号。令 (u_k) 是
(\mathcal F_k)-可预测的 whitened 均值偏移，则

\[
L_k^+=u_k^\top Z_{k+1}-\frac12\lVert u_k\rVert^2,
\qquad
L_k^-=-u_k^\top Z_{k+1}-\frac12\lVert u_k\rVert^2,
\]

以及

\[
E_n^{\pm}
=\frac12\exp\!\left(\sum_{k<n}L_k^+\right)
 +\frac12\exp\!\left(\sum_{k<n}L_k^-\right)
\]

仍是正 (P)-鞅。这里的 mixture 是**路径级固定符号**；它不同于每一步重新抽取
符号所产生的逐步 cosh 乘积，二者对应不同的 (Q)。这个双侧备择只能说明
innovation 沿预先指定的跨尺度轴异常一致，不能再解释成路径一定“更像高噪声”。
它是否与坏图有关仍须独立验证。

change-point 也必须在看轨迹前混合。令起点集合 (\mathcal J) 与权重在 suffix
entry 前固定；对 (j\in\mathcal J)、(s\in\{-1,+1\})，定义

\[
E_n^{(j,s)}=
\begin{cases}
1,&n\le j,\\[2mm]
\displaystyle\exp\!\left[
\sum_{k=j}^{n-1}\left(su_k^\top Z_{k+1}-\frac12\lVert u_k\rVert^2\right)
\right],&n>j.
\end{cases}
\]

若这里使用均匀先验，则

\[
E_n^{\mathrm{cp},\pm}
=\frac1{2|\mathcal J|}\sum_{j\in\mathcal J}
 \sum_{s\in\{-1,+1\}}E_n^{(j,s)}
\]

仍是正鞅；尚未启动的未来分量必须保持 (E=1)。它不是看完轨迹后挑最佳起点。

### 4. Ville、mixture 和 α 的准确含义

若 (E_0=1)，Ville 不等式给出

\[
\Pr_P\left(\sup_kE_k\ge\frac1\alpha\right)\le\alpha.
\]

固定非负权重 \(\pi_j\) 且 \(\sum_j\pi_j=1\) 时，不同尺度的

\[
E_k^{\mathrm{mix}}=\sum_j\pi_jE_k^{(j)}
\]

仍是 e-process。权重不能在看过确认集轨迹后随意修改。
主触发必须使用这个预先固定的 mixture。若改成“任一分量越过
\(1/\alpha\) 就触发”，每个分量各自的 Ville 界不能自动合并成总概率
\(\alpha\)；这种规则必须预先分配检验预算或另作多重性校正。

α 控制的是 baseline (P) 下所有轨迹的总触发率，而不是语义上的“正常图
误报率”。若坏图比例是 (a)，触发对坏图和好图的条件概率分别为 TPR 和
FPR，则

\[
a\,\mathrm{TPR}+(1-a)\,\mathrm{FPR}\le\alpha.
\]

特别地，当 \(a>0\) 时

\[
\mathrm{TPR}\le\min\!\left(1,\frac{\alpha}{a}\right).
\]

例如模型相对坏图率为 10%、\(\alpha=1\%\) 时，即使 FPR 为 0，也至多触发其中
10% 的坏图。强干预保证天然限制了可获得的总体质量变化。

因此论文中应称它为 **baseline intervention budget** 或 **alarm-rate
guarantee**。

### 5. 拒绝与回滚的两个不同质量条件

纯拒绝触发样本后，剩余坏图率为

\[
a'
=\frac{a(1-\mathrm{TPR})}
       {a(1-\mathrm{TPR})+(1-a)(1-\mathrm{FPR})}
=\frac{a(1-\mathrm{TPR})}
       {1-a\,\mathrm{TPR}-(1-a)\mathrm{FPR}}.
\]

当 (0<a<1) 时，(a'<a\) 当且仅当 \(\mathrm{TPR}>\mathrm{FPR}\)。这只说明
越界事件“找得准”。

对回滚算法，令 (T) 为触发事件，(r=P(T))，
(b_T=P(\mathrm{bad}\mid T))，\(\rho_{\rm bad}\) 为触发后新样本的坏图率。若未触发样本
保持不变，则

\[
a_{\mathrm{alg}}=a-r(b_T-\rho_{\rm bad}).
\]

回滚改善质量的额外必要条件是

\[
\rho_{\rm bad}<b_T.
\]

而且绝对坏图率改善有一个容易忽略的上限：

\[
0\le a-a_{\rm alg}=r(b_T-\rho_{\rm bad})\le r\le\alpha_e.
\]

所以 \(\alpha_e=0.05\) 的方法即使触发集选择和修复都完美，模型相对坏图率最多
下降 5 个百分点。ADM64 的**绝对缺陷率**接近 1 并不意味着模型相对 bad
prevalence 接近 1；后者必须用冻结 anchor 另行估计。若模型相对主终点仍是高
基率，它适合验证机制，却不适合作为“选择性清除罕见灾难”的最终展示；强模型上
的低基率、重缺陷子集才是更有意义的质量实验。增大 \(\alpha_e\) 可以扩大潜在
收益，但会同步放宽干预率、计算和分布扰动预算，不能免费获得。

因此必须分开报告：

1. 越界富集：\(\mathrm{TPR}>\mathrm{FPR}\)；
2. 回滚修复：\(\rho_{\rm bad}<b_T\)。

### 6. 计算与分布保证的修正版

若每次从初始噪声独立重采完整轨迹，直到一次不越界，令单次触发率
(r\le\alpha)，则尝试次数 (N) 是几何分布：

\[
\mathbb E[N]=\frac1{1-r}\le\frac1{1-\alpha}.
\]

这个结论不能原样套到任意固定 prefix 的后缀重试。某个危险 prefix 的条件触发率
可能接近 1。证据决定回滚深度可以给出另一个严格界，但必须区分“已经观察到的
失败后缀”和“回滚后用全新随机数抽取的后缀”。

令

\[
B=\frac1\alpha,
\qquad 0<\rho_{\rm rb}<1,
\]

并在首次越界 \(\tau=\inf\{k:E_k\ge B\}\) 后回看，选择最近的预定 checkpoint

\[
J=\max\{j<\tau:E_j\le \rho_{\rm rb}B\}.
\]

这个 \(J\) 是在看见越界后选出的 last-exit time，一般**不是停止时刻**。因此不能
对那条已经失败的原后缀声称
\(P(\sup_{k\ge J}E_k\ge B\mid\mathcal F_J)\le\rho_{\rm rb}\)；选择 \(J\) 这件事
本身已经泄露了后续会越界。

正确的定理针对独立新后缀。保留 \(J\) 以前的完整 sampler 状态，丢弃造成越界的
后缀，并从 baseline (P) 的真实条件转移用全新随机数生成 \(X'_{J+1:T}\)。从原有
证据状态继续计算

\[
E'_k=E_J\prod_{i=J}^{k-1}
\frac{Q_i(X'_{i+1}\mid\mathcal F'_i)}
     {P_i(X'_{i+1}\mid\mathcal F'_i)}.
\]

对每一个可达的固定 prefix \(h_J\)，条件似然比仍是从 1 开始的非负鞅，所以

\[
P^{\rm fresh}_{h_J}\!\left(\sup_{k\ge J}E'_k\ge B\right)
\le\frac{E_J}{B}\le\rho_{\rm rb}.
\]

这是逐 prefix 的 kernel-wise Ville 界。正因为它约束的是选择完成后才生成的
独立新后缀，\(J\) 可以是从已丢弃失败路径回看得到的最近安全 checkpoint；不能把
它错误表述成对 last-exit stopping time 使用条件 Ville。

若每次失败后都再次选择满足 \(E_J\le\rho_{\rm rb}B\) 的安全 checkpoint，且下一
后缀在给定当前全部历史后仍用独立的 baseline-P 随机性生成，则第 \(m\) 个新后缀
失败事件 \(F_m\) 满足

\[
P(F_m\mid\mathcal G_{m-1})\le\rho_{\rm rb}.
\]

不要求各次无条件 IID，也有

\[
P(N_{\rm fresh}>n)\le\rho_{\rm rb}^{n},
\qquad
\mathbb E[N_{\rm fresh}]\le\frac1{1-\rho_{\rm rb}}.
\]

这里 \(N_{\rm fresh}\) 只数触发以后新抽的后缀，并包含最终成功的一次；最初那条
已知失败、随后被丢弃的后缀不在几何界内。失败重抽次数的期望至多为
\(\rho_{\rm rb}/(1-\rho_{\rm rb})\)。若共享初始噪声使 \(E_0=1\)，取
\(\rho_{\rm rb}\ge\alpha\) 可保证初始 checkpoint 总是合格；否则必须显式给出
无安全 checkpoint 时的完整重启规则及较松的 \(\max(\rho_{\rm rb},\alpha)\) 界。
结合首次 baseline 越界率不超过 \(\alpha\)，每个初始样本新增 fresh-suffix
抽样数的无条件期望不超过
\(\alpha/(1-\rho_{\rm rb})\)。这是后缀次数界，不是等长 NFE 界；实际计算量
还必须按每次的回滚深度加权报告。

这个结论还要求：

- 每个 \(Q_k/P_k\) 在抽取下一状态前可预测、规范化且 \(Q_k\ll P_k\)；
- 重采 suffix 逐步使用原 baseline (P)，不能同时加入未计入核的 guidance；
- 保存完整的 Markov/solver 增广状态，并使用新的独立随机数，不能恢复并复用
  原失败 suffix 的 RNG；
- mixture 必须保存每个分量在 \(J\) 的 \(E_J^{(d)}\) 并从这些值继续，不能只保存
  mixture 标量后把各分量重置成 1；
- 算法在有限 horizon 内有可达安全 checkpoint，并对数值 underflow/overflow
  使用可审计的 log-domain 实现。

若算法在首条 baseline 轨迹不触发时逐 bit 保持原输出，只在触发时可能改变
输出，并保证最终终止，则由耦合有

\[
\operatorname{TV}(P_{\mathrm{terminal}},P_{\mathrm{algorithm}})
\le P_P(T)\le\alpha.
\]

回滚输出一般不等于 (P(\cdot\mid T^c))；只有完整拒绝采样才是条件分布。

## 二、相关论文与 baseline 选择

### 第一复现：CFG-Rejection / Diffusion Sampling Path Tells More

这是最接近本路线“路径信号提前暴露坏样本”的直接 baseline：

- 论文：<https://arxiv.org/abs/2505.23343>
- 官方代码：<https://github.com/WSX20003/CFG-Rejection>
- 主模型代码：<https://github.com/NVlabs/edm2>
- ImageNet 主网：`edm2-img512-s-2147483-0.025.pkl`
- 无条件弱网：`edm2-img512-xs-uncond-2147483-0.025.pkl`
- CFG：1.4；Heun：32 步；默认 `S_churn=0`；
- 论文设置：50 个固定 ImageNet 类、总计 10k 样本；
- 论文 bad case：类错配、主体过小或被背景淹没、肢体/结构畸变、语义焦点不清。

官方复现材料存在四个必须显式记录的问题：

1. 论文写“50 类共 10k”，官方 notebook 对单个类设置 `seeds=0..9999`；若对
   50 类全部运行会得到 500k。第一轮按论文文字采用 50 类 × 200 seeds = 10k，
   500k 仅作为可选的 notebook-scale 扩展。
2. 论文公式是 score 差的平方累计；官方 notebook 实际记录 EDM2 去噪预测
   (D_x-D_{\rm ref}) 的逐通道空间范数，并对最早 5 步和通道取均值。
   EDM2 中 score 差还应包含 (1/\sigma^2) 因子。复现必须同时保存：
   `official_notebook_metric`、去噪差 ASD、schedule-dependent score-gap energy，
   不能混称。最后一项含 \(1/\sigma^4\) 因子，不是跨 schedule 的严格不变量。
3. 官方 notebook 在 FP32 中计算空间范数。首版 runner 先转 FP64 再求范数，
   定义相同但不是位级数值复刻；该运行永久标为 discovery-v1。新 runner
   将官方 FP32 量和高精度诊断分开保存。
4. 官方评估 notebook 用未排序的 `os.listdir()` 读取 `.pth`，后续又丢失
   filename/seed，可能把路径分数与终点质量错配。本仓库必须以
   `(class_id, seed)` 键连接；在验证文件顺序前，不声称复刻官方表格的配对管线。

该 sampler 在给定初始噪声后是确定性的，能复现 ASD baseline，但不能直接
承载非退化 Gaussian 路径 likelihood ratio。

### 第一严格 e-process 验证：ADM ancestral DDPM

- 官方代码与权重：<https://github.com/openai/guided-diffusion>
- 首先使用 ImageNet-64 class-conditional diffusion；通过后升到 ImageNet-256；
- `p_mean_variance` 暴露实际 mean、variance、log-variance；
- `p_sample` 是每步 Gaussian ancestral kernel；
- 构造 (Q) 时固定使用 (P) 的 learned variance，只替换备择均值；
- 跳过方差为零的最终步。

它不是为了替代 EDM2 复现，而是第一块最不含糊的定理实验台。

#### ADM 中唯一的 primary 高噪声备择

ADM 是 variance-preserving (VP) 坐标，不能把不同 timestep 的 raw 状态直接
当成热方程中的同一个 (x)。令：

- \(\bar\alpha_t\)：原始 1,000-step cosine schedule 在时间 (t) 的累计
  signal power；
- \(a_t=\sqrt{\bar\alpha_t}\)、\(b_t=\sqrt{1-\bar\alpha_t}\)；
- \(x_t=a_tx_0+b_t\epsilon\)：ADM 的 raw VP 状态；
- \(z_t=x_t/a_t=x_0+\sqrt{\nu_t}\epsilon\)：规范化热坐标；
- \(\nu_t=(1-\bar\alpha_t)/\bar\alpha_t\)：真正满足加性热流的方差时间；
- (c)：ImageNet 类别；
- \(\epsilon_\theta(x,t,c)\)：class-conditional ADM 的 epsilon 预测，即网络
  输出的前三个通道；variance head 的后三通道不进入 score；
- \(u_t(x;c)=-\epsilon_\theta(x,t,c)/b_t\)：raw VP 坐标中的条件 score。

对固定正热偏移 \(\Delta\nu\)，在原始离散 timestep 中选择最接近

\[
\nu_{t^+}=\nu_t+\Delta\nu
\]

且满足 \(t^+\ge t\) 的 (t^+)。令

\[
\rho_t=\frac{a_{t^+}}{a_t}.
\]

同一个规范化位置 \(z=x_t/a_t\) 在高噪声网络坐标中对应的输入不是 (x_t)，
而是

\[
x_t^+=a_{t^+}z=\rho_tx_t.
\]

所以当前 (x_t) 坐标下正确的跨尺度概率比梯度为

\[
\boxed{
\theta_t(x_t)
=\rho_tu_{t^+}(\rho_tx_t;c)-u_t(x_t;c)
}.
\]

它严格对应规范化热密度族 (r_\nu) 的

\[
\nabla_{x_t}\log
\frac{r_{\nu_{t^+}}(x_t/a_t\mid c)}
     {r_{\nu_t}(x_t/a_t\mid c)}.
\]

固定 log-SNR 距离对应的是 \(\nu^+=e^d\nu\)，不是原推导中的
\(\nu^+=\nu+\Delta\nu\)；它只保留为工程 ablation。把同一个 raw (x_t)
直接送给 (t^+) 比较的也是两个不同 (z)，不能作为 primary。

实际 baseline (P) 仍完全保留官方 classifier guidance。令：

- \(\mu_t^P\)：经过 clipped-x0 posterior 和 noisy-classifier mean shift 后的
  官方实际均值；
- \(V_t\)：当前模型 variance head 给出的逐像素正方差；
- \(\Sigma_t=\operatorname{diag}(V_t)\)：实际 (P) 的协方差；
- \(K_t^{(0)}=\tfrac12\theta_t^\top\Sigma_t\theta_t\)：未经缩放的条件 KL；
- κ：在观察本步 Gaussian innovation 前冻结的每检查步 KL 上限；
- \(\gamma_t=\min(1,\sqrt{\kappa/K_t^{(0)}})\)，当
  \(K_t^{(0)}=0\) 时取 1；
- \(\delta_t=\gamma_t\Sigma_t\theta_t\)：最终备择均值差。

primary 定义为

\[
\boxed{
Q_t=\mathcal N(\mu_t^P+\delta_t,\Sigma_t)
}.
\]

classifier gradient 只留在实际 \(\mu_t^P\) 中，不进入 primary 的 θ。原因是
官方模型本身已经 class conditional，再叠加同类别 classifier gradient 后形成的
跨时间向量族一般不再是同一条件密度的热半群。把 classifier 也跨尺度移动只作为
`full-guided` ablation，不能冒充热概率比。

这个 (Q) 也不是“把官方 sampler 的 timestep 改成 (t^+)”：clipped-x0 和
respaced posterior 意味着 \(\Sigma_t\theta_t\) 一般不等于精确替换 denoiser 后的
posterior mean 差。正确表述是：**用规范化热概率比的 score 梯度，对实际官方
Gaussian 核做同协方差指数倾斜**。似然比相对于该实现核精确；高噪声语义仍要
靠坐标 toy 和真实数据验证。

在 (P) 下，本步有

\[
\Delta\ell_t\mid\mathcal F_t
\sim\mathcal N(-K_t,2K_t),
\qquad K_t=\gamma_t^2K_t^{(0)}\le\kappa.
\]

最高噪声边界没有合法更高尺度时、未检查的步、以及最终确定性的 `t=0` 步都令
(Q=P)、\(\Delta\ell=0\)。shifted-x0 插入当前 posterior 可以作工程 ablation；
直接调用 shifted `p_mean_variance` 会同时改变目标时间跨度、posterior 系数和
learned variance，因此拒绝使用。

### 强模型扩展：DiT-XL/2 ancestral DDPM

- 官方代码与权重：<https://github.com/facebookresearch/DiT>
- 本仓库 `models/diffusion/` 已包含同源 GaussianDiffusion；
- `train_gen/sample_ddp.py --ddpm True` 已提供 250-step DDPM 入口；
- ImageNet-256、公开 DiT-XL/2 权重和本地 ADM-FID 环境适合作为强主结果。

### 只做碰撞边界的工作

- Self-Guidance：<https://github.com/maple-research-lab/Self-Guidance>。核心比较对象，
  但 SD3/FLUX 默认 flow/ODE 不适合作为第一版 Gaussian LR 实验台。
- Feynman–Kac Correctors：<https://github.com/martaskrt/fkc-diffusion>。覆盖路径权重、
  SMC/resampling；我们不能声称首次使用路径比。其发布的图像脚本使用
  `edm2-img512-xs-guid-fid`：EDM2-XS conditional/unconditional (`0.045`)、
  CFG 1.4、64 步、`S_churn=40`、particle batch 8，并计划生成 80k。这里 batch
  同时是重采样粒子集合，不能为了吞吐任意改变。该配置要作为后续路径方法
  baseline 单独复现，不能拿 CFG-Rejection 的 EDM2-S/32-step 结果代替。
- RNE：<https://proceedings.iclr.cc/paper_files/paper/2026/hash/7a99ad21706dec5b28f9ad715e12197f-Abstract-Conference.html>。
  覆盖 diffusion path-density ratio 的推断时控制。

可守的新颖性必须同时包括：高噪声分支作为伪影内部备择、形成前的尺度证据
越界、anytime-valid 触发预算，以及选择性完整重启/受限回滚。

## 三、按顺序执行的实验

### R0：来源冻结和下载

1. 保存论文、官方仓库 URL、Git commit、权重 URL、字节数和 SHA-256。
2. 大文件只写入 `$EQVAE_DATA_ROOT/baselines` 和
   `$EQVAE_DATA_ROOT/cross_scale_evidence`。
3. 每次运行保存模型 hash、类、seed、采样参数、软件版本和 GPU 信息。

### R1：原样复现 CFG-Rejection

1. smoke：golden retriever (207)、giant panda (388)、strawberry (949)，每类
   seeds 0–7；验证像素可重现、信号有限、两种 ASD 定义都被保存。
   这些 seed 已被看过，永久归入 discovery。
2. pilot：论文明确展示过定性失败或密度异常的 10 类，每类
   seeds 0–99，共 1,000 张；类 ID 为 `289,336,405,437,520,562,681,701,900,936`。
   同时盲看随机样本及所有信号的对称上下尾部。论文称
   `hummingbird`，但标准 ImageNet 中 hummingbird 是 94，而其发布的 50 类列表
   包含的 95 是 `jacamar`。已核对这是官方 notebook 唯一的类别映射错误。
   当前 1,000 张 pilot 不含两者；语义正确的 hummingbird 补充组使用 94，
   代码忠实对照使用 95 并显式标为 jacamar。
3. paper-10k：官方 50 类，每类 seeds 0–199，共 10k。它与 discovery
   seed 重叠，因此只是 replication 集，不是 confirmation。类 95 不进入
   语义主汇总，只单独报告代码忠实结果。
4. confirmation-10k：只在终点标签、备择核、尺度、checkpoint、mixture、
   KL budget 和阈值全部锁定后，生成从未看过的 seeds 10,000–10,199。
   锁定前不生成 tail grid、不看图、不计算排名。切分单位是 seed，
   不是 `(class, seed)`，避免同一初始噪声跨 split。
5. 复现 AvgkNN/LOF、PickScore、AES、HPSv2 与 ASD 的排序关系；同时加入
   ImageNet top-1/top-5 类一致性。
6. 报告论文公式量与 notebook 实际量的结果差异。`tau5` 只表示最前
   5 个 solver iterations，不是物理时间；`score_asd_full` 含
   \(1/\sigma^4\) 因子，是 schedule-dependent diagnostic，不是跨 schedule 不变量。

本阶段不运行我们的干预，只确认 baseline 和 bad cases 确实存在。

### R2：冻结 bad-case 标签协议

终点标签不得看路径证据。v2 的 severity 和 flags 继续记录绝对缺陷，但
`severity >= 2` 不再等同于正式 bad；低分辨率生成器的典型样本本来就可能普遍
含有可定位缺陷。主二元终点改为
`model_relative_catastrophic_bad`：两名标注者先独立锁定单图绝对检查，再把
候选图与五张冻结的同权重、同 sampler、同分辨率、同类别 baseline-P 典型
anchor 作盲成对比较。每名标注者须至少 4/5 次判候选更差、其中至少 3/5 次判
“明显更差”，并指出一个 `material` 或 `primary_structure_failure`，才得到
reviewer-level 阳性。所有临时阳性和跨阈值分歧由第三名不看任何路径分数的
标注者仲裁；另抽查至少 10% 双阴性。先报一致率，再计算 TPR/FPR。

anchor 只从 seed 不重叠、未按 ASD/evidence/尾部选择的 baseline P 建立，且一套
冻结 reference 同时用于 baseline 和方法臂，不能各自按本臂中位数重新归一化。
快速 pilot 使用 12 类 × 每类 32 张 reference，另取每类 20 个新的配对 seed；
正式 discovery 使用 50 类 × 每类至少 40 张 reference，不稳定的类扩至 80 张；
confirmation 使用 50 类 × 50 个未查看配对 seed。完整字段、anchor 稳定性规则、
审核员校准和仲裁阈值见 `docs/CFG_REJECTION_VISUAL_AUDIT_ZH.md`。

ADM runner 的随机流只由 public seed 决定；不同类复用同一个 seed 就会复用整条
Gaussian innovation。除专门检查“同 innovation、不同类”的配对诊断块外，正式
reference、discovery 和 confirmation 必须给每个 `(class, replicate)` 分配唯一
public seed，而不是把 20 或 50 个 seed 与所有类做笛卡尔积。这样每类仍有预定的
重复数，同时避免名义 2,500 图实际只有 50 条独立创新流。若历史复现实验已经
复用了 seed，则推断必须按 seed 聚类，不能把图像数当 IID 样本数。

自动标签只作 secondary endpoints，包括：

- 条件类别 top-1/top-5 是否匹配及置信度；
- DINOv2 类内 kNN 和 LOF；
- PickScore、AES、HPSv2；
- 后续文本模型上的 GenEval/DPG 类别指标。

人工标签至少分为：类错配、结构/肢体错误、主体缺失或过小、背景吞没、纹理
伪影、正常但罕见细节。绝对缺陷字段用于缺陷负担和机制分层，不决定正式主
标签。人工标注界面隐藏 seed、ASD、我们的 (E_k) 和采样方法臂。

seed 按上述互斥范围固定分成 discovery / validation / confirmation。尺度偏移、检查点、
mixture 权重和 KL budget 只能用 discovery/validation 选择；confirmation 只运行
一次。除上下尾部展示外，统计检验必须覆盖完整预注册样本，不能先看图挑成功例。
确认集只保留一个 primary detector 配置和一个 primary bad 终点；其他比较
明确标为 secondary，并使用 Holm 校正或同等的家族错误率控制。

### R3：先诊断，不干预

在 ADM-64 ancestral sampler 上：

1. 完全冻结 baseline (P) 和随机 seed；
2. 使用上节唯一的规范化热坐标 score-tilt (Q)。第一轮 discovery screen 固定
   \(\Delta\nu\in\{0.01,0.1,1,10\}\)，等权组成 mixture；固定检查内部
   timesteps `249,241,...,1`，其他步 (Q=P)。每个分量第一轮使用
   \(\kappa=0.2\)，同时保存未截断 KL 和 cap saturation；这些只是 discovery
   reference 配置，已有 smoke 结果不得因未越界而事后更改。另行预注册
   \(K_{\rm total}\in\{0.5,1.0\}\) 的累计预算探索分支，以事先冻结的 checkpoint
   allowance 使每个分量的 \(\sum_k\kappa_k\) 不超过总预算。validation 后必须
   锁定一个 primary mixture 和一个预算，confirmation 不得重选；
3. 保持 (P) 的 Σ_k，FP64 累计 log (E_k)；
4. 对每个候选尺度报告每步 KL budget、标准化 innovation、逐步
   Gaussian LR 单元测试和 α 越界率。单类报告可给 exact binomial 区间；跨类
   pooled 结果因为相同 seed 复用同一创新流，必须按 seed 整块做 cluster
   bootstrap，不能套 IID 二项区间。\(E_P[E_k]\) 的样本均值只作重尾诊断，
   必须同时报告 tail/ESS，不得以“是否接近 1”作为单一生死门；
5. 在 confirmation 集只检验坏图是否在越界事件富集。

对照至少包括：CFG-Rejection ASD、瞬时 \(\lVertθ\rVert\)、累计
\(\lVertθ\rVert^2\)、随机 checkpoint 分数、等计算 best-of-N。

必须同时通过：

- 数值/校准门：无 NaN/Inf，经验触发率与 Ville 界相容；
- 检测门：held-out 的 TPR 显著大于 FPR，且不是单个类驱动；
- 提前门：有用富集在终点形成前出现，而非只有最后一步才可见。

失败则停止，不实现回滚。

同一 seed 在不同类上复用同一初始噪声，所以样本不能当作 IID。推断使用
class-stratified 效应、按 seed/class 的 cluster 或 hierarchical bootstrap，并做
leave-one-class-out 检查。汇总 global Spearman 只能作描述性数字。

primary 标签记为
\(B=1\iff\texttt{model_relative_catastrophic_bad}\)，其补集只能叫
`not-model-relative-catastrophic-bad`，不能偷换成 clean；因此主表中的条件报警率
也必须按这个模型相对终点命名。v2 的 `absolute_clear_defect`、`possible_bad`
和严格 `clean` 只作预先声明的敏感性分析，不能替代主终点。
e-process 的报警预算记作 \(\alpha_e=0.05\)，富集检验的显著性水平记作
\(\alpha_{test}=0.05\)，二者不是同一个概率。

### R2.5：先过“模型相对 + 可修复”双门，再把样本当作方法案例

“终图有瑕疵”不等于“这是采样时可选择性清除的 bad case”。主案例先后通过
两道互不替代的门：

1. **模型相对门**：在同权重、同 sampler、同 CFG、同分辨率和同类别下，候选
   必须明显低于冻结的同类典型样本。模型普遍存在的普通模糊或风格限制不能因为
   绝对上不完美就全部标成 bad；
2. **语义保持的可修复门**：从缺陷形成前的同一 prefix 重新采样时，必须能在不
   改变主体类别、对象数量、主要姿态或构图的前提下，降低原来的严重缺陷。若
   “修好局部连接”只能靠换成另一个对象、增加主体或整体换构图，该样本至多说明
   模型存在另一个可行模态，不能证明原 failure 是可被局部 sampling intervention
   修正的轨迹事故。

可修复性门使用与 evidence 完全隔离的 oracle diagnostics：先冻结候选、mask、
prefix 和重采次数，保留所有后缀，盲评“原缺陷是否消失”和“语义/对象数/主要
构图是否保持”。它不参与 detector 的 TPR/FPR，也不能把 human-selected best-of-N
当作方法输出。候选只有在这两道门之后，才进入 R3 的“证据能否在终点前找到它”
检验。

工程上优先寻找三类错误：明显的局部或主体模糊、身体/物体/部件融合，以及肢体
或主要物体的错位连接。轻微纹理问题和局部植物叶片朝向等高语义纠缠细节只保留
为机制诊断，除非它们能稳定通过上述语义保持修复门。

最小 discovery 先固定 50 类 × 20 个全新且逐 `(class, replicate)` 唯一的 seed
= 1,000 条独立创新流 baseline 轨迹。confirmation 至少 50 类 × 50 个同样唯一、
从未查看的 seed = 2,500 个配对单元；每个单元保存
baseline 和方法终点，最多 5,000 张图，并在生成/查看前冻结 detector、early
cutoff、标签终点、seed/class 清单、纳入概率和分析代码 hash。主效应为

\[
D=\Pr(T=1\mid B=1)-\Pr(T=1\mid B=0),
\]

以按 seed 整块 bootstrap 的单侧 95% 下界大于 0 为统计门，同时要求点估计达到
预先冻结的实际意义门槛。若 discovery 的 alarm 率太低而使 confirmation 预期
alarm 少于 50，应该扩大生成池或停止，不能靠降低审图标准制造功效。提前门在
discovery 后冻结一个 cutoff；当前候选是首次越界时 `internal_t >= 65`，即至少
还剩 64 个反向 transition。

### R4：干预

先做完整独立重启，因为它拥有最直接的计算界和条件接受分布。随后测试一次
checkpoint 回滚；只有一次回滚确实改善后，才启用上文严格定义的
“最近安全 checkpoint + 独立 fresh suffix”证据决定深度协议。不能把已失败
last-exit 后缀误计为满足条件 Ville 的一次随机试验。报告：

- 触发率 (r)、TPR、FPR、触发集坏图率 (b_T)；
- 修复后坏图率 \(\rho_{\rm bad}\)，并直接检验
  \(\rho_{\rm bad}<b_T\)；
- FID/FD-DINO/KID、precision/recall、类别准确率、人类盲评；
- NFE、wall time、显存、完整尝试数和额外后缀成本；
- 相同计算预算下的 random restart、best-of-N、ASD-Rejection。

只有 ADM 通过后，才在同一 EDM2-S 权重上实现并验证一个明确的 reverse-SDE
采样协议；不能把结果标成原论文的 Heun baseline。最后再扩展到本地 DiT-XL/2
DDPM 和文本到图像模型。

## 四、停止条件

满足任一项即停止该实现分支：

- (Q) 不能写成观察下一状态前确定的规范化转移核；
- sampler 为确定性 ODE 且没有另行定义合法随机核；
- calibration 在独立 Monte Carlo 中失败；
- 越界在 confirmation 集没有坏样本富集；
- 富集完全由类 ID、CFG 大小或可预测 KL 能量解释；
- 完整重启/单次回滚在等算力对照下不改善质量；
- 改善只来自删掉多样性，precision 上升但 recall 或类覆盖显著坍塌。

这一协议把最容易自欺的三个步骤隔开：先复现别人的信号，再确认我们的证据
能找坏样本，最后才测试它是否真的能修坏样本。

## 五、当前执行状态（2026-08-26）

已经冻结并下载：

- CFG-Rejection commit `82dd2a50effd9c120f6c7b84ff1653e0efb5fc25`；
- EDM2 commit `4bf8162f601bcc09472ce8a32dd0cbe8889dc8fc`；
- FKC-Diffusion commit `aa6f5ed4a0ebb91329d4cd5823cc7e77c5e196e6`；
- Self-Guidance commit `843bda799bb531ccf8d98164d8ccf1a3b8ce3f6d`；
- RNE commit `13ab898536101f3c9a5405ab452cd85ad581b96e`；
- DiT commit `ed81ce2229091fd4ecc9a223645f95cf379d582b`；
- OpenAI guided-diffusion commit `22e0df8183507e13a7813f8d38d51b072ca1e67c`；
- EDM2-S 主网：560,565,890 bytes，SHA-256
  `aa40d3ee46f3db358df37bc4387ce90020df593e4839c1ee056203d448357239`；
- EDM2-XS 无条件弱网：248,541,796 bytes，SHA-256
  `42d0ca453ba0dc1efc6e8b0c6c21063b423ffa12630b8a786eb51eabef9fb442`；
- ADM ImageNet-64 diffusion：1,183,736,577 bytes，SHA-256
  `a18558f9a2499615a3ff9759ad12299690ad36ee3378c395adbb94855e2b634f`；
- ADM ImageNet-64 noisy classifier：261,889,658 bytes，SHA-256
  `d5c4c240e4f0d36460f58520c2803a11490db7e5540e42d2ad75f0cf75bb3586`；
- FKC 图像实验的 EDM2-XS conditional (`0.045`)：249,566,482 bytes，SHA-256
  `27a6c6eaf697b68a74f9c7b72e82f91c2e898d22f629b4546c053865cfe3da68`；
- FKC 图像实验的 EDM2-XS unconditional (`0.045`)：248,541,796 bytes，SHA-256
  `2ea8fffdf0e32d68da3b4050e77c3f9defb1a50a2c9a4a845eb8f927355dea08`。

首个 4-GPU smoke 已完成：3 类 × 8 seeds = 24 张，32-step Heun，CFG 1.4，
每 rank model batch 2。24/24 图像和路径信号均成功写出，采样循环 wall time
13.93 秒。相同 batch、class 388、seeds 0/4 与 EDM2 官方原始脚本输出的 PNG
SHA-256 完全一致，说明信号埋点没有改变 baseline 采样。

这个 smoke 太小，因此不支持质量结论。首次根据缩略网格写下的
“肉眼没有明确坏图”是错误判断：原分辨率逐图复查后，可见多例错位、
肢体/物体粘连、背景侵入、局部模糊和液化纹理。完整纠正性记录与之后的
审图规约见 `docs/CFG_REJECTION_VISUAL_AUDIT_ZH.md`。

这个 smoke 已经显示
“量的定义”确实会改变排序：官方前 5 步 notebook 量与前 5 步 score-ASD 的
全局 Spearman 为 0.987，但与完整 32 步 score-ASD 仅为 0.200；三个类内的后者
分别为 0.238、-0.024、0.643。这个观察只能作为扩大 pilot 的理由，不能从
24 张图推断哪种量更能预测质量。

运行产物位于
`$EQVAE_DATA_ROOT/cross_scale_evidence/cfg_rejection_edm2/smoke/`，包括固定
manifest、逐图路径信号、完整图像、汇总 CSV/JSON，以及每个类、每个量同时
展示低尾和高尾的对称检查网格。

第一个 discovery pilot 也已完成：10 类 × 100 seeds = 1,000 张，4 张
RTX 4090 的采样循环用时 282.75 秒，1,000/1,000 图像和信号齐全。这次
运行的 notebook norm 使用 FP64，因此它标记为 discovery-v1，不与后续
FP32 官方复现混合。v2 已改为 FP32 norm，但它仍在 GPU/FP64 中做最后
扁平平均；与官方 CPU/FP32 “先通道、再前五步”的最大差异约
\(2.64\times10^{-6}\)，但 1,000 张的完整排名和 10% tail 集合都没有改变。
v3 已保持 FP32 buffer、FP32 Euler/prime 相加，并在 CPU 上使用官方的
两次嵌套 mean；24/24 个 smoke scalar 与从原始 gap 按 notebook 顺序重算的
FP32 值逐 bit 相等，图像也 24/24 与前版 SHA-256 相同。

用完全独立、本地冻结的 torchvision ConvNeXt-Tiny ImageNet-1K V1 做类别
一致性弱标签，整体 top-1 为 92.6% (926/1,000)。在每类内各取
official notebook 量的 10% 低尾/高尾时，类别错误率为 5%/2%。这只能
说明低尾稍微富集类别不一致，不能说它能检测结构伪影。实际原图
已出现明确反例：

- class 520 / seed 6 中婴儿五官和手部异常，但分类器对 crib 置信度
  0.925，且它位于官方信号低尾；
- class 289 / seed 33 的雪豹脸部/口部和背景纹理明显变形，还有乱码字形，
  但它在官方信号高尾且分类正确；
- class 562 / seed 94 的人体脸、手臂和腿部严重崩坏，喷泉只是背景，
  但它仍在官方信号高尾。

所以当前可靠结论是：CFG-Rejection 量更像“主体是否典型、清楚并占据画面”，
对类错配/主体缺失可能有用，但分类器和官方信号都会漏掉“语义正确但
结构崩坏”的 bad case。这正是主人工终点必须独立存在的原因。

论文文字口径的 v3 paper-10k 复现已完成：官方 50 类 × seeds 0–199，
10,000/10,000 图像与信号齐全，batch 32，4 张 RTX 4090 采样用时
986.66 秒。独立 ConvNeXt-Tiny 的整体 top-1 为 91.32% (9,132/10,000)。
每类内的 10% 低/高尾对比为：

- 官方 early-5 notebook 量：分类错误率 9.8% / 8.1%，低尾多 1.7 个百分点；
- full denoiser-gap energy：15.7% / 5.6%，低尾多 10.1 个百分点。

按 class 等权、按 seed 聚类的探索性 bootstrap 中，官方量的低减高错误率
95% 描述区间为 [-0.76, 4.09] 个百分点；full denoiser gap 为
[7.77, 12.43] 个百分点。这些都是已暴露的 discovery/replication 结果，
只是 semantic proxy，不是结构伪影的 confirmation。原图再次证实两个尾都有
明显结构坏例：例如 class 701 / seed 175 在低尾出现人体/织物大面积融合，
而 class 562 / seed 94 和 class 405 / seed 87 在高尾仍分别有人体崩坏和
飞艇结构异常。

ADM64 的官方随机采样入口已经先以原始脚本跑通 4 张。随后新增固定 class/seed
的严格 baseline runner：官方 250-step ancestral DDPM、classifier scale 1、
249 个随机反向步和一个确定性终止步；相同 seed 在不同类上复用完全相同的初始
及逐步 Gaussian innovations，网络调用固定 batch 1，避免 batch grouping 改变
像素。首个 3 类 × 2 seeds smoke 已完成 6/6，wall time 215.59 秒，断点重跑
完整通过 PNG 像素 hash、runner hash 和 manifest 校验。另用官方原始
`p_sample_loop`、相同 CUDA RNG stream 和初始噪声独立重跑 class 207 / seed 0，
最终 12,288 个 RGB 值逐 bit 相同，最大绝对差为 0。

这 6 张仍只作环境与视觉规约 smoke，不能做质量率估计。首次把两张狗图描述为
“相对连贯”的口径仍然过松；nearest 8x 与 smooth 8x 再次逐区复看后，6/6 都能
定位到问题：狗图有颈肩/躯干连接异常、前肢缺失或融成毛团，panda 有身体与
树枝/背景粘连及结构缺失，草莓有果实互融、果实—叶柄连接错误、局部液化与重复
纹理。这只是已看过来源的探索性绝对缺陷纠正，不能当作盲标质量率；它们均不
满足严格 clean，但没有冻结的同类 baseline anchor，故 6/6 的
`relative_bad=not_evaluated`，不得写成“6/6 都低于 ADM64 的通常水平”。正式
结论必须走 v3 的绝对检查、冻结 anchor 比较、双人复核和仲裁。

规范化 VP-to-heat 坐标的解析 Gaussian toy 已加入
`experiments/adm64_path_evidence.py`：修正后的 score pullback 与直接热密度比
梯度最大误差为 \(5.6\times10^{-17}\)，在三模态 Gaussian mixture 上为
\(2.2\times10^{-16}\)；错误的“同 raw x 比较”在同一测试中的最大误差为
0.543。同协方差 Gaussian LR 与直接 `Normal.log_prob` 的最大误差为
\(1.0\times10^{-15}\)，低 KL Monte Carlo 的 \(E_P[E]\) 为 0.99964。

observe-only ADM64 runner 的 schema v2 已完成。它直接使用实际 P 转移保存的
FP32 标准差 \(\sigma\)，提升到 FP64 后构造
\(\sigma\theta\)，避免先平方再开方的舍入错位；每步同时保存
\(K_{raw}\) 和 \(R_{raw}=(\sigma\theta)^\top\epsilon\)，所以任意**事先声明**
的 cap 都能由
\(\gamma R_{raw}-\gamma^2K_{raw}\) 独立复算。真实 class 207 / seed 0 上，
无埋点 P 与加入 32 个检查点、106 次 shifted-U-Net 评估后的最终 float32 tensor
逐 bit 相同，最大绝对差 0。

第一轮 \(\kappa=0.2\) 的 3 类 × 2 seeds observe-only smoke 已完成 6/6，全部 PNG
与独立 baseline 逐像素相同，intervention 为 0。四个 shift 的非 identity
checkpoint 数分别为 21/26/29/30；每条路径的 cap saturation 数稳定为
9/20/29/30，说明较大两个 shift 几乎全程需要 temper。6 条路径的 mixture
running-max log-E 为 0.804–1.456，均未达到 \(\log20=2.996\)。这在 6 个样本上
是 **0/6 越界**；完全不能估计触发率，也不能判定检测失败，尤其不能再把六张
绝对有缺陷的图预设成六张模型相对 bad。它只说明数值、路径不变性、schema 和
断点复核已通过。与此同时，每步 \(\kappa=0.2\) 在约 30 个 active checkpoint
上允许过大的累计信息量，存在典型 log-E 下沉、二阶矩由罕见权重支撑的高维
collapse 风险。因此该配置保留为锁定 reference；下一轮另测预注册的
\(K_{\rm total}=0.5\) 和 1.0，但不删除或重解释旧结果。相同 seed 跨三个类的
log-E 明显同向，再次证明正式推断必须按 seed 聚类。

为了尽快验证工程闭环，另做了一次与正式证据完全隔离的 post-hoc mechanics
实验：只保留已看过 smoke 中的单尺度 `Delta nu=1`，取 `alpha=0.25`，越界后固定
向前回退一个预定 checkpoint，并各抽一次 intervention suffix 与同 checkpoint
random-control suffix。6/6 按预期触发：seed 0 在内部 `t=193`，恢复
`x_201`；seed 1 在 `t=217`，恢复 `x_225`。原始分支 6/6 与冻结 baseline 像素
一致，rollback state、触发 innovation、分支 RNG 和 draw count 均通过重入校验。
但全局、早期后缀重采大幅改变构图，逐图可见改善与退化并存；两个 suffix 又是
同条件分布的独立 P 重采。因此它只证明 mechanics 正确，不证明质量改善、触发
优于随机 trigger，也不能引用 evidence-determined rollback 的条件 Ville 界。

随后生成了 3 类 × 20 张同配置 baseline 参考图。它直接纠正了 smoke 的错误
定位：原先 seed 0/1 的六张大多处在 ADM64 的常见质量带，继续保持
`relative_bad=not_evaluated`，不再作为方法成败样本。看任何路径证据之前冻结了
三个单审核者 discovery 候选及同 innovation 跨类对照：class 949/seed 104、
class 388/seed 105、class 207/seed 116。选择记录在
`experiments/annotations/adm64_relative_bad_preselection_v1.json`；它只用于快速
检查局部证据是否跟类特异缺陷走，不能计算 TPR 或充当正式双人相对标签。

FKC 图像代码也已逐行审计。发布路径是 EDM2-XS conditional/unconditional
`0.045`、CFG 1.4、64-step Heun、`S_churn=40`、8 粒子；40/64 使 gamma 在所有
64 步饱和到 \(\sqrt2-1\)，63 个非末步都做组内中心化、clip 与系统重采样，
最终每组只保存第 1 槽，所以 80k 指 particle trajectories 而不是 80k PNG，
实际是 10k 输出。官方 `--class` 被局部变量遮蔽而无效，CPU 重采样又使用未设
seed 的全局 RNG，故原 CLI seed 不足以复现输出。它应作为 FKC prior-art
baseline 严格复现并额外记录 resampling seed；其“高斯 churn 后再经非线性
Heun”存储转移不是显式同协方差 Gaussian，因此不能取代 ADM64 作为本方法第一
个 exact-LR 实验台。真实最小路径已经跑通：CFG 保存 8 张，FKC 的 8 粒子组
保存 1 张；显式设 CPU resampling seed `20260826` 后独立运行两次，输出 PNG
逐 byte 相同，SHA-256 为
`40972b566d61c9fd8e9eb53fc55a60770bf4c09757e2f8db9825de585d324657`。非盲严审
中 8 张 CFG 均观察到 severity 2/3 的绝对缺陷；FKC 单图虽整体更连贯，仍观察到
severity 1 的绝对缺陷。它们都没有同配置冻结 anchor，故
`relative_bad=not_evaluated`。此外，因
额外 RNG 消耗和重采样 ancestry 未记录，不能把它与 CFG seed 0 当成严格配对
质量实验。新增的 fail-closed wrapper 保留发布代码的随机类、第二次无效类别抽取、
63 次重采样及只保存 slot 0 的行为；在相同显式 resampling seed 下，CFG 的 8 张
和 FKC 的 1 张均与直接调用发布代码的输出逐 byte 一致。重复命令只做来源、权重、
manifest 与 PNG hash 校验，不会悄悄重采样。wrapper 是唯一包含额外 CPU RNG
契约的精确入口；发布 CLI 本身没有对应的 seed 参数。

### DiT 后缀分叉的纠正性发现（2026-08-27）

官方 DiT-XL/2 ImageNet-256 的 class 207 / seed 2 路径已在内部时刻
`225,180,120,60` 保存 prefix；每个 checkpoint 保留一次 exact replay 和四次
独立 baseline-P fresh suffix，共 20 个 endpoint records、17 张唯一终图；四个
exact replay record 是同一张 baseline。这只是同一个已查看 prefix 家族，
所有下面的比较都是 post-hoc discovery，不能报告成样本率、TPR/FPR 或泛化结果。

视觉终点首先纠正了旧版“保持原姿态/构图才算修复”的错误口径。终点绝对质量
(B) 与 prefix 保持度 (C) 是两个不同变量：换姿态、对象数或背景可以降低
(C)，但只要结构连贯就不应因此被标成坏图。v2 记录在
`experiments/annotations/dit_imagenet256_seed2_suffix_quality_review_v2.json`，
旧 v1 只保留作审计历史。

尾巴也不能按“能否解释为尾巴”二分。对 `t=60` 的五张近构图终图，审核分别看
尾根与骨盆的连接、由根到尖的粗细变化、以及边界毛流是否连续。attempt 004
在本组相对最自然：连接清楚、连续变细，而且有金毛尾巴应有的蓬松羽状轮廓；attempt 002
接近自然但仍偏宽；replay attempt 000 的横向结构可以是尾巴，却宽、平、弱渐细；
attempt 001 短而钝，近似桨状；attempt 003 的尾尖突变成细丝。后四种差异说明
“语义可命名”不等于“局部形态自然”。与此同时，attempt 003 的明确坏点不是
尾巴本身，而是左后腿形成无合理脚掌的 U 形悬环。为了避免阈值污染，001、002
和 replay 都不放进主好/坏二元 matched pair；第一轮只比较 003（明确坏）与
004（本组最好）。

这里的“本组最好/最自然”不是说局部完全无瑕疵；004 的相对优势主要是毛流更
连续、轮廓更蓬松。二元 `good` 只表示没有明显低于该模型平均采样水平的结构
失败，不表示解剖或每个局部形态完美。

冻结 VAE 对保存的 `pred_xstart` 逐时刻解码后，失败并没有共同的“一次大突变”
形态：

- `t=180/attempt002` 的管状、融合躯干伴随较强的全局预测跳变，属于 burst/fusion
  候选；
- `t=120/attempt001` 的成人犬后肢连接较早固定，后续主要继续增加纹理而没有
  修正连接；
- `t=60/attempt003` 的 U 形腿也从共同的含混状态逐渐定型，并非最后一两步突然
  生成。相反，本组相对更自然的 attempt 004 在后段仍发生较强的局部修形与
  毛发细化。

因此这个 discovery 反例反驳了“大变化必然就是坏图”的确定性强版本。更合理的
机制至少分为两类：一类是过强突变/融合，另一类是错误拓扑过早锁定并在后续只
做表面细化。后者如果要形成特征，必须把“仍有未解决的跨尺度分歧” (U_{k,r})
与“实际纠正运动” (A_{k,r}) 结合，例如探索

\[
S_{k,r}=\frac{U_{k,r}}{\varepsilon_0+A_{k,r}},
\]

其中 (r) 是预先固定的空间 tile，(U_{k,r}) 是当前历史可计算的多尺度预测
分歧，(A_{k,r}) 是此前已经观察到的局部修形量，
(\varepsilon_0>0) 只用于稳定分母。这个 (S) 目前只是风险特征；只有当它在
看到下一步 Gaussian innovation 前决定下一步 (Q) 时，相应的 operational
likelihood ratio 才保持可预测性。不能用本步变化倒过来下注本步噪声。

离线、未调参的轨迹指标也支持“异质机制而非统一突变”。严格排除四条 fresh
branch 共用的 suffix-entry 行后，三个明确坏分支没有任何一个预声明标量都一致
异常：`t=180` 坏分支的最大 `pred_xstart` jump 比最佳其他分支高 32.2%，速度变化
高 6.8%，局部 tile 集中度高 3.7%；`t=120` 坏分支主要是 full-field
change-point log-e 高 25.4%；`t=60` 坏分支只在 centered learned-log-variance
change 上高 1.88%，而 jump、velocity 和两个方向性 e-process 都靠后。特别是
`t=60` 的 hot-tile change-point 最大 log-e，正常 attempt 004 为 2.0335，坏的
attempt 003 仅为 0.1188；把“更多局部细化”直接当报警会优先拒绝更好的结果。
修正前 state-drift 的三个 rank-1 结论来自共享入口最大值及微小差异，已撤回；
修正脚本会报告 tie 数、是否唯一最大及相对 margin。

终点代理同样给出边界。ResNet18 和 ConvNeXt-Tiny 对 20/20 endpoint records
（17 张唯一图）的 top-1 都是 class 207，且明确好/坏分支的类别置信范围重叠；
类别正确不能发现局部拓扑错误。
在事后固定的 `t=60` 尾部框中，attempt 004 的 Sobel、Laplacian 和 band-pass
六个高频描述量均为五图第一，暖色轮廓复杂度第一、方向熵第二，与“羽状毛发更
自然”的观察一致。但这些框和指标是在看图后定义的，而且高频也会奖励噪声，
所以只能证明该视觉差异可被量化，不能当通用 detector。

随后完成的 shifted-DiT exact replay 使用 (\Delta\nu=0.25,1,4)、全局加
(4\times4) tile、共享 baseline 方差以及每个分量整段 (K\le0.5) 的可预测预算；
保存的实际 innovation 与所有 likelihood-ratio 项均可从原始张量独立重建。
这个 discovery 观察不支持统一的单侧高噪声 detector：`t=60` 的坏分支主要与高噪声方向同号，
而 `t=120`、`t=180` 的坏分支主要由反号分量解释；普通正向 running maximum
也会把边界样本或好样本排得更高。

一个事后发现但形式上仍精确的候选是路径级 (\pm) mixture：固定
(\Delta\nu=0.25)，对 `global + 16 row-major tiles` 等权，再对整条路径保持
固定的 (+\theta/-\theta) 两个符号等权。其终点 log-e 在 `t=60,120,180`
各自四个 fresh suffix
中都把明确坏分支排第一；对应值约为 `0.127, 0.170, 0.086`。但 `t=225` 的
四张明确好图中也有一张达到 `0.161`，所以不存在跨 checkpoint 的分离阈值。
固定 (\Delta\nu=0.25) 的 change-point (\pm) anytime running maximum 也在三个
discovery 组内把坏分支排第一，但只有约 `0.099, 0.050, 0.162` log-e；若要让
三者都越过 Ville 阈值，必须取 (\alpha\gtrsim0.951)，完全不具备“少量干预”的
意义。排序线索不等于可部署的 e-value 功效。

局部分量中，在单个 (\Delta\nu=0.25)、`tile_12`、(+\theta) 分量的 anytime
running maximum 里，事后挑出的 `t=60/attempt003` 达到 `2.123363` log-e，
而 attempt 004 的最大值只是初始值 `0`。把格子叠回终图后又发现：`tile_12`
名义上对应左下雪地区域 (y=192:256,x=0:64)，不覆盖尾巴，也不覆盖 U 形异常腿
主体；尾巴主要在 `tile_4/5`，异常腿主体更接近 `tile_13`。因此这个结果不能称为
尾巴或后腿定位；它可能来自 VAE 感受野
外溢，也可能只是同一 prefix 上的随机相关。它随后被固定为黑盒候选并进入新
suffix 盲测；结果见下一节。

确认方案把三个问题分开：终点是否有明确结构失败、后腿拓扑是否连贯、尾巴
是否具有自然根部连接/连续渐细/蓬松毛流。它在同一保存的 `t=60` prefix 上生成
全新的独立 baseline-P suffix，采样时只记录冻结的 `tile_12` 候选而不干预，并在
揭示分数前完成盲标；这只能检验该 prefix 内的可重复性。冻结规则要求：有富集才
能转到未看过的 seed/prefix，若没有则立即淘汰 `tile_12`，且不能回到同一批图
继续挑 tile。下一节记录了这项方案的实际失败结果。更长期的主目标仍是逼近
failure-conditioned innovation 方向；跨尺度差分现在只应被视为候选方向字典，
而不是已经成立的高噪声伪影定理。

### 同一 `x60` 的前瞻盲测结果（2026-08-27）

冻结方案随后在完全相同的 class-207/seed-2 `x60` 上生成了 32 条新的、互相独立
的 baseline-P 后缀。四个八分支 shard 全部保留，采样时不做任何干预；图片包删除
了 branch index、随机流和证据元数据。两位评审在看不到任何 evidence/alarm/rank
的条件下独立标注，再由第三位评审逐张查看原图和放大局部作保守仲裁，最后才锁定
完整 32 行标注并一次性解封证据。

这个过程首先暴露了标签本身的边界。两位初评对后肢主终点只在 9/32 张上完全
一致（Cohen's kappa 约 0.056）：一位把大量侧视抬足、遮挡和投影重叠判成融合，
另一位只判真正清楚的闭环。仲裁严格采用“必须明显低于本池普通水平且不能由姿态
解释”的标准，得到 2 张 `clear_failure`、18 张 `not_clear_failure`、12 张
`uncertain`；整体结构终点为 3 张 clear bad 和 29 张非 clear bad。因为明确主事件
少于预注册的 3 张门槛，本轮在统计上是 **event-limited/inconclusive**，不能估计
稳定的检测率。

尾巴始终与后肢主终点分开。仲裁得到 9 张 natural、12 张 odd、8 张 malformed、
3 张因裁切不可评分。在 29 张可评分尾巴中，根部连接只有 1 张轻度可疑、没有明确
断裂；相反，粗细/体积渐变有 18 张非零（其中 6 张明确缺陷），末端收束有 17 张
非零（其中 7 张明确缺陷），另有 11 张桨状、8 张短钝、3 张突然变成细丝。毛流
本身大多尚可：20/29 为自然，只有 2 张明确坏。这精确说明了为什么“横向结构可以
解释成尾巴”仍不足以判正常：一张图可以有蓬松毛流和连贯根部，却仍在体积渐变或
末端上略怪；相对最自然也不等于完美。

冻结的主候选是单个 `+theta/tile_12` 分量、`Delta-nu=0.25`、整段
`K=0.5`、边界 `log(5)`。结果是 32/32 **零报警**：两张明确后肢失败为 0/2，
18 张明确非失败为 0/18；整体结构 clear bad 为 0/3，其他图为 0/29。因此
`TPR=FPR=0`，没有任何可供回滚的绝对越界。按照冻结决策规则，`tile_12` 的
质量解释应当淘汰，不能回到这 32 张继续改 tile、符号、尺度、预算或阈值。事件数
不足使我们不能断言总体中不存在任何关联，更不能否定所有局部跨尺度证据；但按
预注册规则，这个具体候选在冻结阈值下已经 failed-to-pass，必须退休。

预先冻结、只作描述性分析的 34 路路径混合给出另一个更窄的线索。它对
`global + 16 tiles` 的整条路径正负方向等权；其 **running maximum** 对两张
明确失败的盲排 AUC 为 0.944，失败平均降序名次 2.5，18 张非失败为 11.39；但
terminal 值的 AUC 只有 0.611，平均名次分别为 8.5 和 10.72。这与“异常轨迹曾
短暂出现跨尺度证据、随后终点证据回落”相容，也比固定 `tile_12` 更符合异变/
突变的观察。不过它只有 2 个正例、排除了 12 个不确定样本、所有图共享一个
`x60`，且没有进行推断检验或证明达到可用的 Ville 边界。固定 mixture 本身是
e-process，`max_k E_k` 的越界可用 Ville 控制；但 running-maximum 数值本身不能
被另称为一个新的 e-value。高 AUC 只是排序线索，不能写成 detector 成功，更不能
据此在本池挑新的 cutoff。

理论因此再收紧一步：目前最可能的瓶颈不是似然比鞅是否精确，而是备择方向的
**功效与覆盖**。单位置、单符号的高噪声备择过窄；对位置和符号作事先混合可能
覆盖不同的瞬时失败方向，但混合稀释又可能让绝对 e-value 永远达不到低干预率所需
边界。下一次有效实验必须在全新 seed/prefix 上，把这一个固定 34 路 mixture 的
running-boundary 预先提升为主候选，保留固定 `Delta-nu/K/alpha`，并在证据解封前
完成全量盲标。需要同时报告绝对越界和坏图富集；只复现 AUC 排序而没有有用越界，
仍然不能进入回滚采样。

### 跨 prefix 的正式盲测结果（2026-08-27）

上述最后一个候选随后在全新 prefix 上接受了真正的前瞻检验。协议在 GPU 执行前
固定 64 条 class-207 baseline-P 轨迹、`t=60..0` 观察窗、
`Delta-nu=0.25`、每分量总 `K<=0.5`、`global + 16 tiles` 的 34 个整路径
正负分量、均匀权重和唯一门槛 `E_mix>=5`。8 个 shard 均保持 baseline-P
采样不变并通过逐转移、创新、CFG、LR、VAE decode 与 PNG 重建。两份独立视觉
评审封存后，第三位评审对 64/64 个分歧或不确定行都查看 native 与 enlarged 图，
锁定共识后才消费唯一一次 aggregate-only 解封。

主标签为 4 张 `clear_overall_structural_bad`、59 张 `not_clear`、1 张
`uncertain`。固定候选在 64/64 条路径上都没有越过 5：TP=`0/4`、FP=`0/59`，
uncertain 也是 `0/1`，所以 TPR=FPR=0，单侧 Fisher `p=1`。零越界分支优先于
事件数不足分支，正式结果是 **`frozen_threshold_failed_to_pass`**；同时
`N_clear_bad=4<5`，故附带 `event_limited=true`。这个候选必须退休，不能用同池
降低阈值、翻转方向、改权重或挑 tile 来挽救。

更重要的是，同一 prefix 上的排序线索没有复现。固定 mixture 的 running-maximum
AUC 从 discovery 的 0.944 降为跨-prefix 的 **0.441**；坏图的 running maximum
均值反而较低（0.116 对 0.152）。terminal AUC 为 **0.169**，坏图终点 log-e
均值为 -0.118，非坏图为 0.023。这反对“明显坏图在整条路径上更像冻结的高噪声
备择”这一具体语义，不只是说明 Ville 门槛太保守。

冻结的轨迹形状面板只留下一个尚未确认的反向线索：
`running_max-terminal` 的 drawdown 在 4 对 59 的比较中 AUC 为 0.805，均值
0.234 对 0.130；但最大单步正跳 AUC 只有 0.610，累计正变化 AUC 只有 0.377，
running maximum 本身也不高。因而严谨表述只能是“坏图可能以相对其早期峰值更低
的 mixture 值结束”，不能写成“坏图发生了已验证的证据突变后回落”。4 个正例、
11 个相关特征、没有推断检验，也禁止用最大 AUC 救回 primary。

尾巴仍只作视觉描述。64 张中仅 7 张能可靠评分：1 张 natural、5 张 odd、1 张
malformed，57 张因裁切、遮挡或身份不清而不可评分。在 7 张可评分图中，末端收束
有 5 张轻度异常，渐细/体积有 3 张轻度和 1 张明确异常，根部有 2 张轻度异常，
另有 2 张桨状；毛流为 6 张正常、1 张明确异常。这再次确认：横向结构可解释成尾巴
与尾巴自然是两个问题；蓬松毛流只支持 F 项，不能覆盖 R/T/D 的缺陷。

#### 为什么零越界还暴露了事前功效缺口

记单个分量的累计条件 KL 为

\[
K_j=\sum_k \frac12\delta_{k,j}^\top\Sigma_k^{-1}\delta_{k,j}\le 0.5.
\]

在完全匹配的操作性备择 `Q_j` 下，该分量累计 log-LR 的名义分布为均值
`+K_j`、方差 `2K_j`。34 路均匀 mixture 若主要只有一个分量匹配，该分量要独自
支撑 `E_mix>=5`，量级上需达到

\[
\log E_j \gtrsim \log(5\times34)=5.14.
\]

而 `K_j=0.5` 时其均值仅 0.5、标准差仅 1。其他相关分量会改变精确功效，但这个
量级计算已经说明：`K` cap、34 路稀释与 `alpha=0.2` 的组合在执行前没有证明
“即使数据真来自所写 Q，也有合理触发率”。以后任何 e-process 候选在看真实质量
标签前，必须先通过 **Q-side power gate**：直接从其操作性备择核模拟，预先报告
matched-Q 越界功效、baseline-P 实际触发率、混合权重惩罚和数值稳定性。校准正确
不能代替备择功效。

#### 理论保留什么，放弃什么

保留的是有限步 likelihood-ratio 鞅、Ville 总干预算和受限回滚的条件证明；它们
没有被这次数据反驳。放弃的是当前 `higher-noise fixed-path 34-way mixture = bad
trajectory` 的经验命题，以及由它直接进入回滚的计划。当前评级因此从泛化的
**Conditional GO** 收紧为：**定理保留，固定质量备择 NO-GO，禁止立即干预**。

drawdown 不能直接当 e-value。不过若以后只允许一次低成本形式化尝试，可以把
“短暂出现、换位置/换符号、随后消失”的备择写成预先固定的 hidden-state
likelihood ratio。令 `L_{k,j}=Q_{k,j}/P_k` 是第 j 个分量的本步 LR，`A` 是在看本步
创新前固定的 row-stochastic 状态转移矩阵，`w_{k,j}` 是各备择状态的财富，则

\[
\widetilde w_{k,j}=\sum_iw_{k,i}A_{ij},\qquad
w_{k+1,j}=\widetilde w_{k,j}L_{k,j},\qquad
E_k^{switch}=\sum_jw_{k,j}.
\]

因为 `E_P[L_{k,j}|F_k]=1` 且每行 `A` 的和为 1，
`E_P[E_{k+1}^{switch}|F_k]=E_k^{switch}`。加入 `L_{k,0}=1` 的 P/inactive 状态，
便可严格表示证据的开始、停止和位置/符号切换，而不是事后对 drawdown 设阈值。
它仍只是由本轮反向线索产生的新备择：必须先做 Q-side power gate，再在全新类别、
prefix 和盲标池中一次性检验；若仍无绝对越界或 running score 不超过随机排序，
整个内部跨尺度质量检测方向应停止，转向公开 Self-Guidance/FID baseline 或外部质量
模型，不再围绕同一批 DiT 图继续找统计量。
