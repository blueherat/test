# 最后一轮文献核查：从“不变”到 CFG，尚缺什么

核查截止 **2026-09-13**。只读原始文献与已有记录；本轮没有 GPU、实验或新扫描。以下限定为六篇关键原始来源，重点判定可实施缺口，不声称穷尽所有论文。

**结论：本轮没有找到值得再启动实验的实质新方法。** 最有力的两个候选分别是“按反演路径修正 CFG 的实际分布”和“消除分支 gauge 对 CFG 组合的干扰”。前者已有直接的解析分布与 Feynman–Kac 先例；后者不仅有能量蒸馏先例，仓库也已经完整推导过同一个交叉项和合法联合 Gaussian 反例。免训练局部改写并非被证明不可能，但目前没有同时满足独立目标、可估计修正、公平成本的方案。

## 1. 必须先明确哪一个对象保持不变

用户原现象的接口是 \(x\mapsto F(x;\text{保持原图})\)：原图是任务条件，理想输出应满足该任务约定。当前纯类别条件 SiT 只有 \(v(x_t,t,c)\)，没有独立的原图条件。令完整生成流为 \(G_c\)，再用自己的精确逆定义 \(G_cG_c^{-1}x\)，其恒等性只验证可逆性，不能区分生成质量。换成另一个公共逆会引入参考模型和求解器偏差；此前的真实 copy 研究已经遇到该问题，见[公共逆与 CFG copy 审查](cfg_copy_invariant_after_refiner.md)。

能迁移到纯 FM 的真实数学对象包括：指定分布的连续性方程、已知目标的 Markov 不变分布、规范 score 的保守性。但这些对象都不自动要求同一张图在随机变换后逐像素不动，也不自动把模型自洽转成真实图像质量。

## 2. 反演路径上的 CFG 分布修正：直接先例已经存在

Jiang、Ma 的 **Analytic Distribution of Classifier-Free Guidance for Schedule Design**，初稿 2026-07-22，当前核查 **v2，2026-08-06**，不是未来论文。其扩散时钟 \(s\) 从数据走向噪声，与仓库 FM 的 dataward \(t\) 相反。设 \(p_s,q_s\) 满足同一 forward Fokker–Planck 方程，扩散系数为 \(\sigma(s)\)，精确 scores 为 \(s_p,s_q\)，常数 CFG 权重为 \(w\)。v2 **Theorem 4.2 / Eq.16** 给出实际 deterministic CFG 终点密度：

\[
\widehat\rho_{s_0}(x)
=C\,p_{s_0}(x)^wq_{s_0}(x)^{1-w}
\exp\!\left[-\frac{w(w-1)}2\int_{s_0}^{T}
\sigma(s)^2\|s_p(X_s^x)-s_q(X_s^x)\|^2\,ds\right].
\]

这里 \(X^x\) 沿 guided probability-flow ODE 从给定图像位置反走至噪声；它正是“用反演路径计算几何混合与实际 CFG 分布的偏差”的直接先例。公式依赖精确 scores、共同 forward 路径、正确终端初始化、正则性及积分存在；不能把 SiT learned velocity gap 的平方直接叫作实际模型密度的无偏修正。v2 Theorem 4.3 的变权重情形还有 \(w'\log(p/q)\) 项；阶跃 cutoff 需分段处理边界项。旧仓库读过 v1，定理编号不同。[原文 v2](https://arxiv.org/html/2607.19725v2)

**Feynman–Kac Correctors in Diffusion: Annealing, Guidance, and Product of Experts**，初稿 2025-03-04，核查 v2，2025-06-08，进一步提供了目标分布的加权粒子实现。其基本形式为

\[
\partial_t\rho=-\nabla\cdot(v\rho)+\tfrac12\sigma^2\Delta\rho
+(V-\mathbb E_\rho V)\rho.
\]

粒子运动之外还必须累计权重、必要时重采样；§3.2 与附录 D.3 处理几何组合/引导。删掉权重过程通常就删掉了分布修正；单粒子归一化权重恒为 1，也无法产生选择效果。这提供有原则的目标采样，但增加粒子成本与权重退化问题，既不保证逐图恒等，也不保证所选几何混合比真实条件分布有更好 FID。[原文 v2](https://arxiv.org/html/2503.02819v2)

**判断：** CFG 的瞬时 score 是几何混合 score，不代表整条非平稳 CFG 流的边缘就是该几何混合。把上述校正称为新的“反演不变性 guidance”会撞到直接先例；指定该目标也仍需说明为何服务于质量。

## 3. 闭环 work 为零是合法约束，但没有给出新的质量修正

### 3.1 约束到底成立在哪里

仓库的线性路径是 \(x_t=tX+(1-t)\epsilon\)。在规范条件期望速度、相同物理时间和欧氏坐标下，

\[
g^*(x,t)=v_c^*(x,t)-v_u^*(x,t)
=\frac{1-t}{t}\nabla_x\log p_t(c\mid x),\qquad 0<t<1.
\]

因此固定 \(t\) 的闭合空间路径满足 \(\oint g^*\cdot dx=0\)。沿采样时间前进再反演组成的时空循环不能直接套用这个结论：它还涉及时间变化和不同向量场的非交换性。这一固定时间推导已经见于[旧 FSG / curl 审查](../../FSG_RELEASE_CURL_AND_APG_REVIEW_20260911_ZH.md)。

**On Investigating the Conservative Property of Score-Based Generative Models**，ICML 2023，核查 arXiv **2209.12753v3，2023-06-04**，已有训练正则

\[
L_{\rm QC}=\tfrac12\mathbb E\|J_s-J_s^{\mathsf T}\|_F^2,
\]

并以随机迹估计降低成本。它研究训练期降低非保守性；并未证明冻结模型逐点去 curl 会提升质量。该论文也早已被仓库阅读，不能把上述约束重新命名为新发现。[原文](https://arxiv.org/html/2209.12753v3)

**Diffusion Models Observe Only Gradients: A Geometric Perspective on Score Matching Errors**，初稿 2026-06-04，核查 **v2，2026-06-28**，把误差分为真实密度 \(p_t^*\) 下的 weighted-Hodge 分量：

\[
e=\Pi_{p_t^*}e+r,\qquad \nabla\cdot(p_t^*r)=0.
\]

其结论允许很大的无散误差而不改变指定边缘；观测分布误差取决于有权重的梯度分量。这里的权重、边界条件和目标路径不可省略，SDE 的 KL 界也不能直接变成 SiT 有限步 ODE 的质量界。仓库此前已读 v1，并做过真实路径的有限势函数修正；其首次 1K 结果仅相对降低 FID 约 0.083%，后续规模审计状态须读原记录，不能把它重新发明成未尝试路线。[原文 v2](https://arxiv.org/html/2606.06179v2)；[已有实现与限定](../../RAEV2_OBSERVABLE_ERROR_GUIDANCE_20260906_ZH.md)

### 3.2 最强反驳也必须保留：分支各自无害不代表 CFG 后无害

固定时刻令 \(s_i=\nabla\log p_i\)、\(\nabla\cdot(p_i r_i)=0\)，并假设 \(\pi_w\propto p_u^{1-w}p_c^w\) 可归一化。则

\[
\frac{\nabla\cdot\{\pi_w[(1-w)r_u+wr_c]\}}{\pi_w}
=w(1-w)(r_u-r_c)\cdot(s_c-s_u).
\]

它说明分支 gauge 可以在组合后变成可见误差，故不能用“无散部分不影响分布”一口否决 CFG 分支修正。**但该完整 boxed 公式、联合 Gaussian 反例和离散类别反例已经出现在旧文档 §184–250 附近**，本轮没有新增机制。[已有完整推导](../../FSG_RELEASE_CURL_AND_APG_REVIEW_20260911_ZH.md)

如果只把混合场按 \(\pi_w\) 投影，投影前后加权散度相同，因而不会消除上式的可见项。要消除理想化的分支 gauge，需要分别按 \(p_u,p_c\) 投影；这要求获得对应密度的全局加权投影，且无法消除本来就在梯度方向上的错误。

### 3.3 最接近的实质先例是能量蒸馏，不能照搬其全部表述

Thornton 等的 **Composition and Control with Distilled Energy Diffusion Models and Sequential Monte Carlo**，arXiv 2025-02-18，AISTATS 2025。§3.2 Eq.7 给出

\[
\min_u\ \mathbb E_{p_t}\|\nabla u-v\|^2,
\]

Eq.9 用 pretrained score 训练能量梯度，§4 再用于 composition / SMC。**先把预训练分支变成保守能量，再组合和控制**已有实质相近先例；它需要训练，论文没有实现我们的 class-conditional SiT 反演小循环。[会议原文](https://proceedings.mlr.press/v258/thornton25a.html)；[可定位公式全文](https://arxiv.org/html/2502.12786v1)

严谨边界：Eq.8 右式印刷遗漏了应有的 \(\nabla\)；附录 A.1 开头写普通 \(\nabla\cdot r=0\)，但保持加权密度需要 \(\nabla\cdot(p_t r)=0\)。从 Eq.7 的一阶最优条件可得到后者。正文先假设原场的实际流边缘就是投影权重 \(p_t\)，才讨论边缘保持；Eq.9 使用真实图加噪分布，而有误差教师的实际生成边缘未必相同，所以不能将其后续措辞扩张成无条件保持定理。这里仅认定方法原则相近，不认定全部理论前提或工程实现相同。

### 3.4 径向锚定没有填上这个缺口

本轮考虑的局部构造，固定时间并取与当前求导变量独立的锚点 \(a\)，令 \(r=x-a\)：

\[
\phi_a(x)=\int_0^1g(a+sr)\cdot r\,ds,\quad
\nabla\phi_a(x)=\int_0^1\{g(a+sr)+sJ_g(a+sr)^{\mathsf T}r\}\,ds.
\]

若 \(g\) 本来就是梯度，它在适当星形区域上恢复 \(g\)；但一般情况下这是按指定路径构造势，**不是**所需的密度加权正交投影。固定锚点位置会影响结果；若每点换成依赖 \(x\) 的反演锚点又停止其梯度，则得到的实际向量场不再由同一个全局势保证保守。完整求导还要支付路径/Jacobian 成本。它没有从模型内部制造真实 score 的监督信号，也没有证明删掉的部分是上式有害的分支项。未检到同构局部实现，不能据此声称新颖或值得投入。

## 4. 反演后作分布保持变换：目标是真的，但提升目标不能免费得到

**Markovian Flow Matching: Accelerating MCMC with Continuous Normalizing Flows**，Cabezas、Sharrock、Nemeth，NeurIPS 2024，§3.1 已使用 inverse-flow → latent proposal → forward-flow，并用目标密度和流 Jacobian 完成 Metropolis 校正；§3.2 结合 MALA 与 FM 训练。目标是可逐点评估的未归一化密度，既不是凭类别标签就可获得，也不是单张图恒等。[会议原文，§3 Eq.6–10](https://proceedings.neurips.cc/paper_files/paper/2024/file/bcd11db0b26d8fc2266b91d3ff982ed1-Paper-Conference.pdf)

对任意可逆 \(G\)，目标的 pullback 为

\[
\pi_Z(z)=\pi_X(G(z))|\det DG(z)|,
\quad
\nabla_z\log\pi_Z=DG(z)^{\mathsf T}\nabla_x\log\pi_X(G(z))
+\nabla_z\log|\det DG(z)|.
\]

如果目标只是模型自己的 \(P_G=G_\#\nu\)，且 \(K\) 保持 \(\nu\)，则共轭核 \(GKG^{-1}\) 保持 \(P_G\)。这是确切的不变量，但原本从 \(P_G\) 采出的整个人口分布不会因此改善；图像可以变化，分布不变。若目标改为真实图像分布，需要独立密度/数据估计和相应接受率，不再是模型自己的 no-op 约束。仓库也早有该共轭核方向及完整正反路径成本讨论，见[五个机制的旧审查](../../FIVE_MECHANISM_IDEAS_20260912_ZH.md)。

跨条件 \(T_{b\leftarrow a}=G_bG_a^{-1}\) 的往返与共享坐标路径一致性同样可由代数自动成立，无法为质量定标；另要求每对条件之间的最小改动 transport 同时路径无关，也不是理想模型必须满足的普遍性质。

## 5. 与本仓库各支线的边界及结束判断

| 已考虑的方向 | 实际约束对象 | 本轮判断 |
|---|---|---|
| 自己的精确 ODE 逆、反复 copy | 可逆映射/数值解的一致性 | 可以诊断求解误差，不能独立定标图像质量 |
| Z / W2SD / FSG 的强前弱逆及其精化 | 不同场的组合动力学 | 算子与时钟审查已有记录；改精细逆不自动增加数据目标 |
| CFG-CTRL 的滑模量、gap 或历史控制 | 所选控制变量 | 不是真实图必须固定的对象，不从本轮再转向控制分解/调度 |
| CFG 路径分布校正 | 指定目标密度/路径权重 | 解析公式、FK 粒子校正已有直接先例 |
| 分支保守化后 CFG | 分支加权连续性与组合交叉项 | 机制已在仓库，能量蒸馏有实质先例；局部近似尚无可靠收益依据 |
| 反演空间的分布保持核 | 已知目标的不变分布 | 自目标不会改善人口分布；真实目标需要额外信息与成本 |

Z / W2SD / FSG 的原始论文、官方代码时间约定及已试配置沿用[既有反演 prior-art 核查](../cfg_inversion_20260913/fsg_w2sd_prior_art.md)和[Z 接续审查](../cfg_inversion_20260913/z_sampling_handoff.md)，本轮不重述为新论文，也不将未读取的最新实验结果写成负结果。

**停止依据是本轮缺少实质新、可实施的候选，不是证明所有“不变性”研究无价值。** 当前仍缺少：既能被纯类别条件 SiT 低成本观测，又由真实数据目标决定，并能指出应当怎样修正而非只测到不自洽的量。继续把旧的循环、curl 或路径权重换名，无法补足这一缺口。因此不建议追加 GPU 扫描；保留上述机制和假设，结束这一轮。
