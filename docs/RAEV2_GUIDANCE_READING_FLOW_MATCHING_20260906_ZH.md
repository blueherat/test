# Flow Matching 原论文：局部配对桥的训练依据与有限实现

日期：2026-09-06。本次只新增精读 Lipman、Chen、Ben-Hamu、Nickel、Le 的 **Flow Matching for Generative Modeling**，不扩展其他论文，不修改当前冻结实验、阅读总表或研究状态。核查时总表已有 48 篇，包括 *On the Guidance of Flow Matching*，但没有这篇基础论文。

**保留的正面结构是：先指定相邻时刻的真实配对，再用它的固定残差监督整个辅助时间内的条件速度；训练无需模拟辅助 ODE。** 这支持当前有限步桥的学习入口，并解释为何必须观察非零辅助时间。它不提供有限 CNN 的拟合保证，也没有证明 conditional regression 的估计噪声会自动减小。本文给出非独立配对所需的本地推广，以及现有 midpoint 在一个解析模型中捕捉 covariance pressure 的有限结果。

一手版本为 [arXiv:2210.02747v2](https://arxiv.org/abs/2210.02747v2)，2023-02-08；[ICLR 2023 官方节目](https://iclr.cc/virtual/2023/papers.html)核实会议身份。下载的 28 页 PDF 页眉是 Preprint；OpenReview PDF 返回 HTTP 403，故不声称已核对独立的会议终稿。阅读覆盖正文 §2–6、Appendix A 三个定理证明、B 的连续性方程、E.2 训练与评估设置，并检查 §6.2 的 midpoint 实验；未复现作者模型或采样结果。PDF 第 14 页关键证明已渲染复核。固定原文、提取文本、关键页面与下载记录位于 `experiments/raev2_guidance_restart_20260906/reading_flow_matching_v1` 数据归档目录。

## 1. 原论文给出了什么

原文 Theorem 1 从 conditional density / velocity 出发，以 posterior 权重混合速度，得到边缘路径的连续性方程。Theorem 2 证明 conditional 与 marginal 两个平方损失只相差一个不含网络参数的常数，因此期望参数梯度相同。Appendix A 明确依赖积分／微分交换、可积性与梯度支配条件；正文还写定正密度。这里的等价性适用于任意固定参数化类，**不等于该类包含正确速度，也不等于训练会达到正确速度**。[原文 §3、Appendix A](https://arxiv.org/html/2210.02747v2#S3)

Theorem 3 的 Gaussian affine 公式是选定 affine flow 的速度；不是声称给定 density path 只有一个速度。§4.1 的 conditional OT 使用独立的 noise 与 data，逐 conditional Gaussian 的路径为直线；作者明确区分 conditional OT 与 marginal OT。§6.2 的低 NFE 实验使用 midpoint，支持路径选择可能改善实际数值效率；没有给当前两次辅助查询的误差定理。ImageNet-32 的 FM-Diffusion / FM-OT 分别报告 FID `6.37 / 5.02`、adaptive NFE `193 / 122`，评估为 50K；这是重新训练模型的历史比较，不能直接作为 RAEv2 guidance 的质量或总成本结论。[原文 §4.1、§6、Appendix E.2](https://arxiv.org/html/2210.02747v2#S4.SS1)

## 2. 非独立 teacher pair：本地推广，而非直接照用正密度定理

固定 `(t,s,c)`。按照[有限步结构](RAEV2_ACTUAL_ROLLOUT_FINITE_TRANSPORT_20260906_ZH.md)，令

\[
Y=T_{t\to s}(Z_t),\quad W=Z_s,\quad R=W-Y,
\qquad U_\tau=Y+\tau R.
\]

`(Y,W)` 的联合分布记为 π，二者使用相同的真实 X 和 bridge noise。它们无需独立。假定 `Eπ||R||<∞`，对任意紧支撑光滑测试函数 φ，支配收敛给出

\[
\frac d{d\tau}\mathbb E_\pi\phi(U_\tau)
=\mathbb E_\pi[\nabla\phi(U_\tau)\cdot R]
=\mathbb E[\nabla\phi(U_\tau)\cdot w_\tau(U_\tau)],
\quad w_\tau(u)=\mathbb E[R\mid U_\tau=u].
\tag{1}
\]

因此 `hτ=Law(Uτ)` 与 w 满足弱连续性方程，端点为 `h0=T#p_t`、`h1=p_s`。这个证明不需要 conditional density、密度严格为正、Y 可逆地决定原 Z，或 endpoint 独立性。它也是对原论文混合证明所用思想的直接推广。若把 `(Y,W)` 当 condition，则 conditional path 是点质量而非原文写定的正密度 Gaussian；因此必须补上式 (1)，不能仅改符号后引用原 Theorem 1。

式 (1) 只证明路径和速度的弱关系。从这里到由同一个 deterministic ODE map S 实现 `S#h0=h1`，仍需速度的适定性及相应输运唯一性；例如适当的时空连续性、空间局部 Lipschitz 和防止有限时间逃逸的增长控制。终点可能奇异时另需极限陈述。当前近邻配对的温和性值得检验，但并未由原论文或式 (1) 自动证明。

若再有 `E||R||²<∞`，对不读入原 X、ε、Y 或 W 的辅助模型 `vθ(Uτ,τ,t,s,c)`，平方损失严格分解为

\[
\mathcal L_{\rm pair}(\theta)
=\underbrace{\mathbb E\|v_\theta(U_\tau)-w_\tau(U_\tau)\|^2}
_{\mathcal L_{\rm marginal}(\theta)}
+\underbrace{\mathbb E\operatorname{tr}\operatorname{Cov}(R\mid U_\tau,\tau,t,s,c)}_C.
\tag{2}
\]

条件期望的正交性使交叉项为零。C 不含 θ；在梯度交换可行时，两个损失的期望参数梯度相同。这个本地 Hilbert 空间证明不要求光滑密度，明确支持当前非独立 pair。它同时说明：改变配对 π 可以保留两个端点，却改变整个中间 h、w、不可约误差 C 和学习难度。

## 3. 论文的简化直觉如何落到估计噪声

**固定 pair 的 R 沿 τ 不变。** 采样不同 τ 只需要做 `Y+τR`，不需要再运行原生模型或模拟辅助 ODE。这是可直接保留的计算结构：缓存一次真实 teacher 前向，可以覆盖多个辅助时刻。多个 τ 仍共享同一个 X、ε 与 R，并不是多个独立 teacher；验证或不确定性统计不能把它们当作独立数据量。

无 floor 的 Euler 下，`R=β(X−G(Z_t))`、`β=1−s/t`，因此

\[
\mathbb E\|R\|^2=\beta^2\mathbb E\|X-G(Z_t)\|^2,
\quad
C\le\mathbb E\|R\|^2.
\tag{3}
\]

相邻 pair 保留了共同的 X、ε，使 displacement energy 随局部步幅缩小。若将两个端点独立重配，通常会失去这个小量；但式 (3) 只是噪声能量的上界，**不是所有条件化下的方差排序，更不是信噪比保证**。有 floor 或有限精度时，标签必须继续使用实际 `W−T(Z_t)`，不能拿简式覆盖它。

对一次 stochastic parameter gradient，记 `Jθ(U)=∂vθ(U)/∂θ`，则在给定 `(U,τ,t,s,c)` 时，由随机 target 产生的条件 covariance 为

\[
\operatorname{Cov}(g\mid U,\tau,t,s,c)
=4J_\theta(U)^T\operatorname{Cov}(R\mid U,\tau,t,s,c)J_\theta(U).
\tag{4}
\]

原论文的期望 gradient equality 不会消除式 (4)。其直线、固定方向的 regression target 直觉和较快训练实验，不能替代降低这个 covariance 的证明。本文没有从论文推出适用于当前 finite native G 的额外降方差算法，不建议为此更改冻结训练。

在 Bayes 情形，`E[R|Y]=0`，故 `w0=0`；起点的整个 target energy 都可能是不可约噪声。一般光滑局部展开中，covariance pressure 对 w 的响应为 `O(β²)`，其 squared signal 可以为 `O(β⁴)`，同时 target noise 为 `O(β²)`。因此局部 residual 很小不代表回归容易。非 Bayes G 另有一阶条件均值偏差，不能把这个 oracle 阶数套到全部实际记录上。

式 (2) 给出一个实际可解释的配对评价量：同一独立 pair 上相对零场的风险改善可写成

\[
\Delta=\mathbb E[2v_\theta(U)\cdot R-\|v_\theta(U)\|^2]
=\mathbb E[\|w(U)\|^2-\|v_\theta(U)-w(U)\|^2].
\tag{5}
\]

它显式消去两臂共享的 `||R||²`；比要求原始 CFM loss 接近零更合适。式 (5) 的样本估计仍有不确定性，仅衡量 teacher 路径上的可学习速度，不能冒充实际 rollout KL 或 FID 保证。若把 R 除以只依赖 `(t,s)` 的尺度 κ 做训练，部署正确乘回 κ 后 oracle 条件速度不变，但损失变成 `κ⁻²` 加权的 marginal loss；共享有限网络的拟合侧重会改变，不能称作完全相同的优化问题。

## 4. 现有 midpoint 能捕捉什么

原文没有保证有限 CNN 可表达 w，也没有保证一次 midpoint 就能完成整个 `τ∈[0,1]` 校准。CFM 的随机梯度无偏是给定 population 目标下的性质；有限 teacher bank 的训练直接对应经验配对目标，泛化需要独立样本证据。训练 loss 的下降、ODE 解的逼近和完整 RAEv2 终点质量是三个不同对象。

不过，当前单步 explicit midpoint 有明确的正面机制解释。沿已出现的标量 Bayes Gaussian pair，设

\[
Y\sim N(\mu,A),\quad R\sim N(0,B),\quad Y\perp R,
\quad w_\tau(u)=\frac{\tau B}{A+\tau^2B}(u-\mu),
\qquad A>0,\ B\ge0.
\]

起点 Euler 使用 `w0=0`，完全不改变状态。现有 midpoint 则使用

\[
k_1=w_0(y)=0,\quad k_2=w_{1/2}(y+k_1/2),
\qquad
S_{\rm mid}(y)=y+k_2
=\mu+\frac{4A+3B}{4A+B}(y-\mu).
\tag{6}
\]

完整 oracle flow 的伸缩系数为 `sqrt((A+B)/A)`，midpoint 后的方差差有精确有限公式

\[
(A+B)-\operatorname{Var}(S_{\rm mid}(Y))
=\frac{B^3}{(4A+B)^2}.
\tag{7}
\]

式 (7) 由展开两个二次多项式直接得到；若 `B/A` 小，其差是三阶于 `B/A`，而不是起点 Euler 留下的整个 B。这说明两次查询确实能够读到 covariance pressure 的后续响应，支持当前结构的机制试验。该结论是**本地 Gaussian 推导**，不是原论文定理，也不是实际 CNN 的数值或 FID 保证；若末端 B/A 不小，不能继续使用小量解释。

实际 finite native G 的均值误差、有限模型学习误差、辅助数值误差与 actual-q 转移仍需分别观察。当前冻结 pair、网络和 midpoint 可按固定预算验证这些问题；原论文并没有导出新的 gain、原时间窗口或通过调参选择的 schedule，本次也不引入它们。

## 5. 归档与本次产出边界

原文 PDF SHA-256：`5eeb39ba516396924aba4787452f9d0abdee88467a4d0c264d9f66cad0c5ee14`。

归档路径：`/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/reading_flow_matching_v1`，其中 `manifest.json` 记录文件、来源与摘要值。保存 28 页 PDF、HTML、arXiv 版本页、ICLR 会议来源、提取全文及关键 proof 页；下载失败单独记录，不把错误页面当成会议 PDF。

本次只完成一篇阅读及当前结构的数学解释，没有运行 GPU、训练、实现新 arm、改动冻结实验或取得新增 FID。正面产出是非独立 pair 的合法监督目标、不可约噪声的明确对象，以及 midpoint 保留后续 covariance 响应的有限解析解释。
