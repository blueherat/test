# RAEv2 proximal 校准的 channel 共享结构审阅

**状态：原逐坐标 diagonal 审计早期 heldout 失败后提出的结构修订，不是预先注册的原始方案。** 原 100 步审计继续保留。新的 channel 共享规则如进入开发验证，应在所有 100 个时间步统一应用，不搜索 guidance 强度、不选择时间窗口、不使用 FID 选择结构。原 bank B 已参与提出修订，因此复用 B 只能作为开发验证，不能称为这条修订的全新独立确认。

本文只做理论审阅，未修改已冻结审计代码、未运行 GPU。基本离散方法与 toy 见 [原理论记录](RAEV2_PROXIMAL_ERROR_PROJECTION_20260906_ZH.md)。原运行的早期报告显示：步骤 0–11 的 heldout 风险增益均为负，步骤 1–11 的逐步近似区间也为负；training J 正、heldout J 负。它支持“高维矩拟合过拟合”的假设，尚不是全部 100 步的最终结论，也不能单靠这些数值判定真实总体斜率为零。

## 1. 共享的是函数空间，不是虚构独立样本

对单个时间步，y=z_s 为真实下一状态，r=T(z_t)−y。用 j 表示 channel、p 表示其 P=16×16 个位置。原逐坐标锥有每个 (j,p) 独立的平移及斜率；现在统一限制为

\[
h_{j,p}(y)=c_j+a_j(y_{j,p}-m_j),\qquad a_j\ge0.
\]

这里 m_j=E[P⁻¹∑ₚy_{j,p}]，c_j=E[P⁻¹∑ₚr_{j,p}]。因 c_j 自由，中心的取法只是重参数化，不改变这个共享 affine 函数空间。每个时间步从 262144 对自由平移/斜率降为 1024 对，维度由 latent 的 channel 结构直接决定，没有待选的 rank、频率截止点或正则化强度。

在原始欧氏目标 E∑ⱼ,ₚ(rⱼ,ₚ−hⱼ,ₚ)² 下，闭式解是

\[
V_j=\mathbb E\frac1P\sum_p(y_{j,p}-m_j)^2,\qquad
M_j=\mathbb E\frac1P\sum_p(r_{j,p}-c_j)(y_{j,p}-m_j),
\]

\[
a_j=\max(M_j/V_j,0),\qquad
S_{j,p}(z)=\frac{T_{j,p}(z)-c_j+a_jm_j}{1+a_j}.
\]

V_j=0 时定义 a_j=0。上述推导不要求不同位置独立，也不要求真实图像空间平稳或网络对平移严格等变：欧氏平方范数本来就是各坐标平方之和，共享参数的 normal equation 只需把这些项相加。空间相关性影响估计方差，不能把独立样本数写成 n×P。这个结构是对估计空间的限制，不能包装为已经证明的真实图像对称性。

## 2. 原始训练充分统计足以精确转换

原 `DiagonalFit` 已存每位置的训练均值 μʸⱼ,ₚ、μʳⱼ,ₚ、总体分母方差 Vⱼ,ₚ 与协方差 Cⱼ,ₚ，所有位置样本数相同。定义位置平均符号 ⟨·⟩ₚ，则

\[
m_j=\langle\mu^y_{j,p}\rangle_p,\qquad
c_j=\langle\mu^r_{j,p}\rangle_p,
\]

\[
V_j=\left\langle V_{j,p}+(\mu^y_{j,p}-m_j)^2\right\rangle_p,
\]

\[
M_j=\left\langle C_{j,p}+
(\mu^r_{j,p}-c_j)(\mu^y_{j,p}-m_j)\right\rangle_p.
\]

再计算 a_j=max(M_j/V_j,0)。这是同一训练集上共享锥最小二乘解的**准确充分统计转换**，无需重新运行训练图像的模型 forward。不能直接平均原 aⱼ,ₚ；也不能只平均原 V/C 而遗漏位置均值之间的贡献。拟合完成后需在 heldout 新计算实际 proximal 误差，原 diagonal 的 heldout 聚合指标不足以恢复共享修正的效果。

## 3. 哪些原保证保留，哪些没有增强

共享函数构成原 Hilbert 空间里的闭凸锥。因此总体投影仍有

\[
\mathbb E\|r-h(y)\|^2=\mathbb E\|r\|^2-\mathbb E\|h(y)\|^2,
\qquad S(z_t)-y=(r-h(y))/(1+a).
\]

prox 每个 channel 的 Jacobian 块为 (1+a_j)⁻¹I，故非扩张与原有限步 W₂ 递推上界的机制均保留。锥分解和 proximal 非扩张的经典依据是 [Moreau 原文 §4.b、§5.b](https://www.numdam.org/article/BSMF_1965__93__273_0.pdf)。

Bayes oracle 也保持：若 T(z_t,label)=E[y|z_t,label]，则每个坐标都有 E rⱼ,ₚ=0，且

\[
\mathbb E[r_{j,p}y_{j,p}]
=-\mathbb E\operatorname{Var}(y_{j,p}\mid z_t,label)\le0.
\]

所以 c_j=0、M_j≤0，a_j=0。不同位置的真实均值、空间相关、不同类别均值均不破坏这个论证。它是总体结论；样本里正斜率比例约一半不能据此判断模型有真实正斜率，边界附近的估计噪声经 max(·,0) 本身就会产生许多正值。

**Gaussian 保证需改成 channel 总距离，而非每个坐标均不增。** 例如两个独立坐标目标均为 N(0,1)，真实 coupling y=z，T=diag(2,1)z。共享 a=(1+0)/2=.5，S=diag(4/3,2/3)z；第二个本来准确的坐标会被压缩，但总距离下降。

对独立 Gaussian 坐标，令 u 拼接各位置输出标准差、去 channel 平均后的输出位置均值，v 同样拼接目标量。共享 proximal 可写成 S=λ(T−μᵀ_pool)+μʸ_pool；若有非零斜率，则 λ=V_y/Cov_pool(T,y)。任意实际 coupling 都满足 Cov_pool(T,y)≤uᵀv，而 V_y=‖v‖²（采用相同的求和/平均约定）。Cauchy–Schwarz 给出

\[
\lambda\ge\frac{\|v\|^2}{u^\top v}
\ge\frac{u^\top v}{\|u\|^2}=\lambda_{W_2}^*.
\]

所以从 λ=1 移到该 λ 只靠近沿此射线的最优 W₂ 系数；channel 平均均值误差同时被消除，故总 channel 的真实输入单步 W₂² 不增。若输入/目标跨 channel 也独立，则可求和得到总 W₂² 结论。相关、非 Gaussian 的联合分布不能直接用这个可分离 Gaussian 证明，仍有原有限步传播上界机制。共享还可能漏掉位置特异的偏差，例如 T=diag(1.1,0)z 时共享 a=0。

## 4. 有限样本收益可以怎样准确表述

先单独看 translation。令 μ=E r，训练均值为 μ̂，且其估计噪声协方差为 Σ̄。令 Π 为把各 channel 内位置平均后广播的正交投影。对独立测试样本，拟合共享 translation Πμ̂ 的预期风险是

\[
R_{shared}=\mathbb E\|r\|^2-\|\Pi\mu\|^2+
\operatorname{tr}(\Pi\bar\Sigma).
\]

与逐坐标 μ̂ 相比，风险差为

\[
R_{shared}-R_{full}
=\|(I-\Pi)\mu\|^2-\operatorname{tr}((I-\Pi)\bar\Sigma).
\]

这给出明确可证伪机制：若被去掉的位置特异方向主要是估计噪声，共享改善有限样本风险；若那里是真实系统偏差，共享引入 approximation error。无需空间独立即可成立；它只分析 translation 的统计贡献，不冒充整套 nonlinear rollout 或 FID 的定理。含非负斜率的完整方案仍应直接审计 heldout D_k。

没有一种非平凡有限样本均值校正可以对所有总体都保证优于“完全不校正”：真实 μ=0 时，baseline 的均值估计风险已经是零，任何非零随机估计都会加风险。James–Stein 结果保证在指定 Gaussian 噪声模型下胜过样本均值，而非对所有 μ 胜过零校正；真实高维、相关、异方差 latent 不能直接套用单位协方差公式。见 [James–Stein 1961 原文 §2](https://sta721-f24.github.io/website/reading/James-Stein-1961.pdf)。因此不把 SURE/shrinkage 再引入为这一轮的自动“修好”按钮；依据同一风险估计选择方案后，该最小风险估计还会产生选择乐观偏差，见 [Tibshirani–Rosset 原始分析](https://arxiv.org/abs/1612.09415)。

## 5. One-per-class 银行的总体与不确定性

这次一类一图对应的总体是 **uniform-class mixture**：p(X,label=c)=K⁻¹p(X|c)，K=1000，另配独立 Gaussian 噪声。global/shared 校正最小化这个混合总体的平均风险，不是每一类别的风险，也没有估计每类各自的校正。某一类别变差不与总体风险下降矛盾。

若每类图像按目标 p(X|c) 随机选取，训练均值 μ̂=K⁻¹∑꜀r_c 无偏，其方差是

\[
\operatorname{Cov}(\hat\mu)=K^{-2}\sum_c\operatorname{Cov}(r\mid c).
\]

固定类别覆盖不产生 between-class 抽样波动。对固定校正的任一标量 heldout 指标 D，令其类均值 μ_c、类内方差 σ_c²；一类一图时，普通跨图 sample-variance/K 的期望为

\[
\operatorname{Var}(\bar D)+
\frac{1}{K(K-1)}\sum_c(\mu_c-\bar\mu)^2.
\]

因此常规跨图 SE 在这个随机分层模型下平均偏保守，却不是精确的分层置信区间；每类仅一张，不能从本 bank 单独估计全部类内方差。跨时间共享 X/E，应继续将整张图的所有时间记录放在同一统计单元内。若图像实际是固定顺序挑选、或 train/heldout 是同一有限类样本中不放回选出的两张，须使用对应抽样解释；仅“不重叠”不能自行证明 iid 或对目标总体无偏。本审阅未重新认证旧 bank 的原始选图随机性。

这也是为什么已有 bank B 的开发表现不能替代独立新图像确认：B 的早期失败已经影响了共享结构选择，即使没有看 FID，方法选择依赖仍存在。

## 6. 均值平移的 rollout 边界，以及暂不执行的零平移候选

一个准确的反例：真实 p_t=p_s=N(0,1)，teacher coupling y=z；模型 T(z)=z+b。总体 translation c=b 让 teacher 风险从 b² 降至零，Lip 不变。但若当前实际 rollout q_t=N(−b,1)，原 T#q_t=p_s 已经正确，减去 c 后实际 W₂²=b²。共享只能降低拟合方差，不能排除这种 p/q 偏差抵消；原文的跨步反例也仍适用。latent 均值变好本身更不等于 Inception 特征均值或 FID 变好。

**以下只是理论候选，本轮不执行、不作为共享方案的可切换开关。** 若结构目标是保护已经正确的实际输出均值，可以移除自由 translation，选固定真实下一状态均值向量 m_s=E y，限制

\[
h_{j,p}(y)=a_j(y_{j,p}-m_{s,j,p}),\quad a_j\ge0,
\]

\[
a_j=\max\left(\frac{\mathbb E\sum_p r_{j,p}(y_{j,p}-m_{s,j,p})}
{\mathbb E\sum_p(y_{j,p}-m_{s,j,p})^2},0\right),\quad
S=m_s+\frac{T-m_s}{1+a}.
\]

该锥的投影/非扩张/oracle 不变性质仍成立；与共享 affine 方案不同，中心必须保留真实各位置均值。对同一**任意 incoming q**，有

\[
\mathbb E_q S-m_s=\frac{\mathbb E_q T-m_s}{1+a}.
\]

所以这个单次 correction 不会推偏已经正确的实际输出均值，并使其均值距离收缩。channel 分别取不同 a 时仍不保证 covariance、整体 W₂ 或递归终点 FID。若用估计中心 m̂_s，额外均值偏移为 a/(1+a)⊙(m̂_s−m_s)，因此还要审计中心估计；“零平移”不意味着完全没有统计误差。bridge 的 m_s=(1−s)E X，可共用数据均值而不拟合 100 个独立噪声均值。

### 单一全局斜率可得到任意分布的实际单步 W₂ 保证（仅理论候选）

若进一步规定所有坐标共用唯一 a≥0，并固定真实目标均值 m=E Y，令 X=T(z_t)、Y=y，定义

\[
A=\mathbb E\|X-m\|^2,\quad B=\mathbb E\|Y-m\|^2,\quad
C=\mathbb E[(X-m)^\top(Y-m)].
\]

零平移锥投影得到 a=max((C−B)/B,0)，S=m+λ(X−m)，其中 C>B 时 λ=B/C，否则 λ=1。这里也不引入可调系数。令

\[
C_*:=\sup_{\pi\in\Pi(\mathcal L(X),\mathcal L(Y))}
\mathbb E_\pi[(X-m)^\top(Y-m)].
\]

由于实际校准 coupling 只是候选之一，C≤C_*≤√(AB)。正比例缩放不会改变最优 coupling 的选择，故准确地有

\[
W_2^2(\mathcal L(m+\lambda(X-m)),\mathcal L(Y))
=\lambda^2A+B-2\lambda C_*.
\]

若 C≤B，映射不变；若 C>B，则

\[
0<\lambda_*:=C_*/A\le B/C_*\le B/C=\lambda<1.
\]

因此从 1 缩到 λ 不会越过这个实际 W₂ 二次式的最小点，**对任意有限二阶矩分布，真实输入的单步实际 W₂ 不增**。B=0 的退化目标可定义 a=0 保持不变。这不是局部 MSE 的替代解释，而是直接的最优 coupling 推导；也不依赖 Gaussian 或像素独立。

该定理针对用于定义 A/B/C 的同一个输入分布（当前为真实 z_t）；换成 rollout q_t 后不自动成立，除非对应 cross moment 也是在该 q_t 与目标之间定义、校准的。有限样本估计、中心误差、递归分布变化及最终 FID 仍有未解决之处。这个结果仅作为后续理论线索记录，未改变当前 channel 共享修订，也不是新增 GPU 实验队列。
