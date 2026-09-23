# 保持 reference 边缘的可训练控制：泄漏界、有限检验盲点与低秩实现

2026-09-22。理论补充。所有分布均指连续 ODE 的实际 rollout 边缘；数值求解误差需要额外计入。本文没有运行图像实验。

## 1. 一个不需要显式 score 的精确泄漏量

固定 reference 场 W，其边缘为 p_t。给 W 添加 u_t，所得实际边缘记为 p_t^u。假设密度正且足够光滑、二阶矩有限、边界通量消失，并满足下述 Hodge 分解与 ODE 解唯一性所需条件。

在 H_t=L²(p_t;R^d) 中，令 G_t 为光滑梯度场的闭包，定义

\[
R_{p_t}(u_t)=\sup_{\psi:\ E_{p_t}|\nabla\psi|^2\leq1}
|E_{p_t}[u_t\cdot\nabla\psi]|
=\|P_{G_t}u_t\|_{L^2(p_t)}.
\]

这是加权负 Sobolev 型连续性方程残差，也是 u 的分布可见部分的能量范数。写 u=h+r，其中 r=P_Gu，h 与所有梯度正交。于是 div(p_t h_t)=0，而 p_t 同时满足速度 W 与 W+h 的连续性方程。

**有限时间漂移界。** 若修改后场 b_t=W_t+u_t 的空间 Lipschitz 常数为 L_t，且相关流存在，则同初始分布下

\[
W_2(p_T^u,p_T)
\leq\int_0^T e^{\int_t^T L_sds}R_{p_t}(u_t)dt.
\tag{1}
\]

证明：令 X 随 W+h 演化，因此 Law(X_t)=p_t；Y 随 b 演化并令 Y_0=X_0。两者差满足

\[
\dot Y-\dot X=b(Y)-b(X)+r(X).
\]

对二阶均方差使用 Minkowski 与 Grönwall 即得 (1)。如果 Hodge 投影只以闭包意义存在，需要用正则逼近或连续性方程叠加原理来补足流表示；这不是任意粗糙场下无条件成立的断言。

这个界控制的是**有限修改后的真实 reference 分布**，而非只控制修改瞬间的密度导数。大 Lipschitz 常数可能使界数值上很松，所以形式保证不等于现有图像网络可直接取得有用证书。

其 score-free 对偶形式为

\[
R_p(u)^2=\sup_\psi\{2E_p[u\cdot\nabla\psi]-E_p|\nabla\psi|^2\}.
\tag{2}
\]

给 endpoint 目标加上 λ 倍 (2)，得到只需要冻结 W 的实际 rollout 样本及 test-network 输入梯度的 saddle 训练问题。它无需计算 div(u)，也无需先把 W 当成其实际边缘 score。

## 2. 有限 test 函数给的是下界，不能直接作为保分布证书

对有限测试 φ_1,...,φ_m，设

\[
c_j=E_p[u\cdot\nabla\phi_j],\qquad
G_{ij}=E_p[\nabla\phi_i\cdot\nabla\phi_j].
\]

该测试空间测得的范数为 R_M²=cᵀG†c≤R_p²。Gram 归一化使结果对重复/线性重标测试不敏感；简单求和 c_j²不具备这个性质。

遗漏部分为 r_⊥=P_Gu−P_{span{∇φ_j}}u，并有

\[
R_p^2=R_M^2+\|r_\perp\|_{L^2(p)}^2.
\]

因此只有在能控制遗漏部分时，小 R_M 才能提供上界；否则必须称为有限观测下的近似约束。增加独立测试和实际 weak rollout 两样本比较可以发现违例，但也不能自动变成高维全分布证明。

**有界光滑反例。** 取 p=N(0,1)、W=0，以及

\[
u(x)=\sin x-\tfrac12e^{3/2}\sin 2x.
\]

在参考 p 上，E[u]=0、E[xu]=0，所以所有均值/方差弱约束在每个时间均为零。但真实修改后的过程具有

\[
\left.\frac d{dt}E[X_t^4]\right|_{t=0}
=4E_p[x^3u(x)]=12e^{-1/2}\approx7.27837.
\]

在一维正态与零边界通量条件下不存在非零有限能量 gauge，故此例真正 R_p=||u||≈1.26804。这里 u 全局有界且 Lipschitz；失败不是由 ODE 爆炸导致。

**minibatch 陷阱。** 若 g=u·∇φ，直接平方 minibatch 均值会得到

\[
E[(\bar g_B)^2]=(E[g])^2+\operatorname{Var}(g)/|B|.
\]

即使真实 gauge 使 E[g]=0，也会被额外的方差项压小。两独立 minibatch 均值乘积或 U-statistic

\[
\frac{(\sum_i g_i)^2-\sum_i g_i^2}{N(N-1)}
\]

具有无偏性；对偶 saddle 形式 (2) 在固定 critic 下也可用无偏单样本期望。有限样本 critic 仍有过拟合与优化不充分问题。

## 3. 近似 score 的误差只通过一个方向性漏项进入

令 Aᵀ=−A，score 估计为 ŝ=s_p+e，构造

\[
u=\operatorname{div}A+A\hat s
=u_{\rm exact}+Ae.
\]

则逐点恒等式为

\[
\operatorname{div}(pu)=\operatorname{div}(pAe),
\qquad
R_p(u)=\|P_G(Ae)\|\leq\|Ae\|\leq\|A\|_{op,\infty}\|e\|.
\tag{3}
\]

把 (3) 代入 (1) 即得 reference 漂移上界。完整 score 误差并不是最紧量；真正相关的是 Ae 中的加权梯度部分。若另有数值/网络近似项 ξ，同理增加 ||P_Gξ||≤||ξ||。

**低秩而不要求完整 score。** 取 A=B C Bᵀ，其中 B∈R^{d×k} 列正交，Cᵀ=−C，B/C 可依赖 t/条件但不依赖 x。只需学习

\[
g_*(x)=B^T\nabla\log p(x),\qquad u=B C g(x).
\]

可对真实 reference rollout 样本使用投影 score matching：

\[
J(g)=E_p\left[\tfrac12\|g(x)\|^2+\sum_{i=1}^k b_i\cdot\nabla_xg_i(x)\right],
\]

积分分部给出 J(g)−J(g_*)=½E_p||g−g_*||²，所以 R_p(u)≤||C||op||g−g_*||。这属于已有 sliced/projected score-matching 思路在本控制问题中的使用，不应把估计目标本身当新方法。[Sliced Score Matching](https://proceedings.mlr.press/v115/song20a.html)

一个关键限制：g_* 不是 Y=BᵀX 的低维**边缘** score。补齐正交坐标 Z 后，它是 ∇_y log p(y|z)，一般需要输入整个 x；将其换成 ∇_y log p_Y(y) 会改变 Y/Z 的依赖关系，从而破坏 full weak 边缘。小输出维数不等于只需低维边缘数据。

## 4. 把 guided 收益与 weak 泄漏贡献分开

在当前 baseline guided 场 v=S+a(S−W) 下，边缘记 q_t。固定终点泛函的一阶测试 f，令 λ_t 为沿 v 的 backward costate，终值 λ_T=f。对 W→W+εu，有

\[
\delta J=-\int_0^T a(t)E_{q_t}[\nabla\lambda_t\cdot u_t]dt.
\]

使用相对于 reference p_t 的 u=h+r 分解，则 δJ=δJ_gauge+δJ_leak。令 ρ_t=q_t/p_t，则

\[
|\delta J_{\rm leak}|
\leq\int |a(t)|\sqrt{E_{p_t}[\rho_t^2|\nabla\lambda_t|^2]}\ R_{p_t}(u_t)dt.
\tag{4}
\]

若 ρ_t≤M_t，可放松为

\[
|\delta J_{\rm leak}|
\leq\int |a(t)|\sqrt{M_t}\|\nabla\lambda_t\|_{L^2(q_t)}R_{p_t}(u_t)dt.
\tag{5}
\]

因而若总一阶改善量严格超过泄漏贡献的有效上界，才能据此保证非零 gauge 贡献。由有限 test 求得 R_M 不能替代右侧 R_p。对实际 GAN，此式适用于固定当前判别器的局部生成器目标；不能直接写成有限步 GAN 收敛或 FID 保证。

密度比大时，微小 reference 漂移也可能主导 guided 收益。(4) 明确揭示了为什么“weak FID 没变，但 guided FID 变好”还不足以证明保边缘机制。

## 5. 文献边界与一个不应依赖的近邻论断

- [NGIF](https://arxiv.org/html/2605.25107v1) 已用样本弱连续性方程、随机 Fourier tests 与可选 gauge 正则学习非梯度 population dynamics。故“无需 score 的 weak test 约束 + 在 gauge 内选择”已有直接近邻。本处需贡献的是 reference/guided 双动力学目标、误差归因与可运行的优化设计。
- [Gauge freedom, ICLR 2024](https://arxiv.org/html/2402.03845v1) 已有加权散度条件与相应分解；不能重复当作新定理。
- [Nonreversible Langevin samplers](https://arxiv.org/abs/1506.04934) 已研究固定目标不变时选择非可逆扰动以降低时间均值方差，并分析目标依赖性。它优化的量与有限时间 strong/reference 组合终点不同。[后续 splitting 工作](https://arxiv.org/abs/1701.04247) 表明离散偏差与非可逆收益必须一起核算。
- [Incompressible optimal transport](https://arxiv.org/abs/2504.01109) 是带控制能量的不可压缩 mixing 近邻。其启发是保测度运动的受限可达性；不应把“旋转/搅拌能有用”单独作为 novelty。
- [EDDY](https://arxiv.org/html/2605.06553v1) 的单分布反对称 Stein 恒等式可安全使用。其 Claim 2 后“保单粒子边缘但改变联合分布”的表述，在 iid 初始、独立噪声、光滑唯一解与其写出的逐粒子 marginal-Stein 构造下需要谨慎：设 P_t=∏_i p_t(x_i)，则允许 A_i 依赖所有粒子的情况下仍有 div_i(P_t u_i)=P_{t,-i}div_i(p_t u_i)=0。因此整个 product path 是修改后联合 FPE 的解，唯一性给出联合分布仍为 P_t。该推导意味着不能直接依赖其精确粒子构造作为“精确保边缘且改变 iid 群体多样性”的已证实例；本文的单 reference 控制理论完全不依赖这个联合论断。数值近似、score 误差或不同噪声耦合应另行分析。

本轮最有用的推进是将“近似保边缘”变成有条件可量化的命题：(1) 控制真实 weak 漂移，(4) 控制 guided 收益可被泄漏解释的比例，并明确有限测试只能检出一部分违例。
