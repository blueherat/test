# 从 SSG 出发，怎样为 CFG / IG 设计参考模型的训练目标

2026-09-14。本文回答的问题是：参考模型一定要学习强模型的生成分布吗？高斯平滑是否值得做？有没有更接近“放大模型误差”的训练目标？本轮完成文献核对、目标推导和实施方案，新增验证仅为代数检查，**尚未运行下述新目标的图像质量实验**。

我建议保留两个研究候选：**保留原分布成分的平滑参考**，以及**增加生成过量部分权重的参考**。纯高斯平滑作为必要对照。前者便于先做真实模型实验；后者更接近我们原先关心的误差分布，但密度比估计是实质难点。两者均可使用已经固定的内部读出结构，采样时不增加独立主干。是否有效应由同预算生成比较决定。

**SSG 解决了什么，仍留下什么。** SSG 冻结像素扩散主干，在中间层接可训练 adapter，用强模型生成的图像重新加噪训练，再外推最终与中间预测。公开实验生成训练图像时使用 CFG，所以这个训练来源包含 guidance、求解器与步数的选择，并非只由 checkpoint 决定。它的实验支持自身样本是有效的训练来源；并未证明这是最优参考分布，也未证明内部参考恰是某个高斯卷积。它主要使用额外 Transformer block 与输出头，不能把我们的小 MLP 实验当作论文配置的完整复现。[^1]

用生成数据训练负参考，SIMS 已有先例；跨噪声构造平滑参考，Self-Guidance 已有明确动机。因此，“自己生成数据训 weak”和“用更平滑的分布作参考”本身都不能作为新贡献。[^2][^3] 有价值的剩余问题是：**构造哪个参考分布，会让强减弱主要响应希望纠正的偏差？怎样用一个正确、稳定的训练目标实现它？**

以下 (P) 表示真实数据总体分布，(G) 表示某个固定强模型与固定采样器实际产生的终点分布，(B) 表示选定的参考训练来源，可以取 (P) 或 (G)。类别默认固定；涉及多类别时逐类定义，再保持相同类别先验。小写表示密度。时间采用从噪声到数据的线性路径

\[
Z_t=tX+(1-t)\epsilon,
\qquad \epsilon\sim\mathcal N(0,I),\qquad 0<t<1.
\]

给定合法参考分布 (Q)，理想 clean 预测、速度与 score 的关系为

\[
D_Q(z,t)=\mathbb E_Q[X\mid Z_t=z],\quad
v_Q(z,t)=\frac{D_Q(z,t)-z}{1-t},\quad
s_Q(z,t)=\frac{tD_Q(z,t)-z}{(1-t)^2}.
\tag{1}
\]

因此同一时间、同一参数化下，预测的线性外推与 score 外推对应。但有限神经网络的预测场，通常不等于其实际终点分布重新加噪后的精确场。下文的分布恒等式必须与这项近似分开。

**首先排除一种容易误判的“改 loss”。** 对完整输入 (I=(z,t,c))，若只乘一个依赖 (I) 的正权重，

\[
\mathcal L(f)=\mathbb E[\lambda(I)\|f(I)-Y\|^2],
\qquad \lambda(I)>0,
\]

则无限容量总体最优解仍为 (f^*(I)=\mathbb E[Y\mid I])。把欧氏距离改成依赖 (I) 的确定正定二次型，也仍是同一个条件均值。时间权重、SNR 权重、固定频率权重可以改变有限模型的拟合侧重，但一般不能单凭它们推出“weak 学会了一个不同的理想分布”。TIW-DSM 对时间相关密度比加权也指出：只加权而缺少相应 score 修正，不会得到目标分布的正确 score。[^4]

内部读出只接收特征 (H) 时，严格结论需把可测性条件改成“权重只依赖 (H)”。如果权重还包含 (H) 无法恢复的信息，它会改变有限信息下的条件均值。不能把上述完整输入定理直接当作真实 IG 读出的无效性证明。

真正能定义不同参考分布的一条直接路径，是给**干净训练端点**加权。令 (w(X)>0)，

\[
q_0(x)=\frac{b_0(x)w(x)}{\mathbb E_B[w(X)]}.
\]

则原生加权 FM 目标

\[
\mathcal L_w(W)=
\mathbb E_{X\sim B,t,\epsilon}
\left[w(X)\left\|W(H(Z_t,t,c))-(X-\epsilon)\right\|^2\right]
\tag{2}
\]

正好是在 (Q) 的端点分布上训练。这里权重与不可观测的 clean target 相关，所以不会在给定 (z) 后约掉。权重应预先确定并停止梯度；不是根据当前采样状态随时给 guidance 乘系数。

已有文献覆盖重要性加权、密度比修正和直接 score difference 学习。DLSM 与后来的 Sobolev 正则研究也提醒：分类器概率拟合得好，不意味着对它求导就能得到可靠的引导方向。[^4][^5][^6] 本文不把这些基础技术重新宣称为新发明。

**高斯平滑可以严格实现，但要同时改输入与监督目标。** 选择

\[
X\sim B,\qquad Y=X+\tau\xi,\qquad
\xi,\epsilon\overset{\mathrm{iid}}\sim\mathcal N(0,I).
\]

令 weak 学习 (Q_0=B*\mathcal N(0,\tau^2I))，正确训练对是

\[
Z_t=tY+(1-t)\epsilon,
\qquad \text{clean target}=Y,
\qquad \text{velocity target}=Y-\epsilon.
\tag{3}
\]

这里是对**整个图像或 latent 向量的分布做卷积**。对图像做空间 Gaussian blur 是 (X\mapsto AX) 的推前分布，属于另一种操作；对扰动输入的 score 取平均又对应对 log density 的平滑，三者不能互换。

由独立高斯相加，

\[
q_t=b_t*\mathcal N(0,t^2\tau^2I).
\tag{4}
\]

若只是额外扰动输入，仍以原始 (X) 为 clean target，再用原来的 score 转换公式，就不是式 (4) 的正确参考。记

\[
d_t^2=(1-t)^2+t^2\tau^2,
\]

错误做法的总体 clean 预测为 ((z+d_t^2s_Q)/t)，转换后得到的是

\[
s_{\rm wrong}=\frac{d_t^2}{(1-t)^2}s_Q.
\tag{5}
\]

它混入了一项显式的时间相关缩放。这种实现即便 FID 变化，也不能直接归因于“训练出了高斯平滑的弱分布”。像素端点若加高斯后再裁剪到有效像素范围，也改变了卷积核；可以另立假设，但不能沿用精确高斯公式。

训练还可以更干净：解析地积分掉额外端点噪声，采用 Rao–Blackwell 化的监督。只抽一个标准高斯 \(\eta\)，构造

\[
\begin{aligned}
Z_t&=tX+d_t\eta,\\
Y_{\rm RB}&=X+\frac{t\tau^2}{d_t}\eta,\\
V_{\rm RB}&=X+\frac{t\tau^2-(1-t)}{d_t}\eta.
\end{aligned}
\tag{6}
\]

用 (Y_{\rm RB}) 训练 clean 读出，或用 (V_{\rm RB}) 训练 velocity 读出，具有与式 (3) 相同的总体最优目标。它们是原随机监督给定 (X,Z_t,t) 的条件期望，减少监督噪声；不是教师预测蒸馏，也不需要额外强模型调用。对于 JiT，继续使用原生 velocity MSE 对应的 \(1/\max(1-t,0.05)^2\) 权重；SiT 直接回归速度目标。该恒等式不意味着两种有限步优化轨迹相同。

这个构造与 SG 的重合必须明确。令

\[
\rho_t=t+d_t,\qquad \widetilde t=t/\rho_t,
\]

对于维度 (D)，

\[
q_t(z)=\rho_t^{-D}b_{\widetilde t}(z/\rho_t),\qquad
s_Q(z,t)=\rho_t^{-1}s_B(z/\rho_t,\widetilde t).
\tag{7}
\]

所以纯高斯参考在理想场层面，可以通过跨噪声查询及正确重标度得到。Self-Guidance 已覆盖这一思路的主要动机；其具体实现和任意线性 FM 坐标中的精确式 (7) 不能不经核对就视为相同。[^3] 我们若训练一个共享内部读出来实现它，值得检验的是有限容量下的效果及省掉额外主干查询的实际收益，不能只凭换成训练 loss 就认领新的分布理论。

**第一个候选：保留原分布成分的平滑参考。** 我更愿意检验

\[
R=B*\mathcal N(0,\tau^2I),\qquad
\boxed{Q_\beta=(1-\beta)B+\beta R},\qquad 0<\beta<1.
\tag{8}
\]

实现很简单：以概率 \(1-\beta\) 使用普通训练对，以概率 \(\beta\) 使用式 (6) 的平滑训练对；一次随机选择作用于整张图。训练网络不接收这个来源标记。两种样本走同一个 weak，形成一个混合分布模型；不是在同一 (z) 上线性平均两个已训练头。

由于前向加噪是线性算子，每个时刻都精确满足

\[
q_{\beta,t}=(1-\beta)b_t+\beta k_t,
\]

其中 \(k_t\) 为平滑分量 (R) 加噪后的密度。对混合密度求 log 梯度可得

\[
\boxed{
s_B-s_{Q_\beta}
=\frac{\beta k_t}{(1-\beta)b_t+\beta k_t}(s_B-s_R).
}
\tag{9}
\]

这说明 reference 学到的差值具有一个由全局分布定义的门控：在原分布远强于平滑分量的区域，修正较弱；在平滑分量占据较多质量的区域，保留较强修正。实现不需要在采样时估计式 (9) 的比例，weak 的混合分布训练已经隐式包含它。

还有一个有用的性质：

\[
\frac{b_t(z)}{q_{\beta,t}(z)}\le\frac1{1-\beta},
\qquad
\log\frac{b_t(z)}{q_{\beta,t}(z)}\le-\log(1-\beta).
\tag{10}
\]

当 \(\beta=1/2\) 时，正向 log-ratio 势的上限是 \(\log2\)。这是相对密度比的经典有界性，RuLSIF 等工作已经研究；候选贡献应是把它作为内部负参考的训练目标并验证作用，而非这个不等式。[^7] **势函数有上界不等于梯度有上界**，也不是数值稳定性或 FID 保证。

它一般不是单个高斯平滑。平滑噪声的特征函数为

\[
(1-\beta)+\beta e^{-\tau^2\|k\|^2/2},
\]

不能等同于某个固定方差高斯的指数形式。不过在 \(\tau\to0\) 的一阶近似下，它相当于较弱的高斯平滑，因此**必须与纯平滑参考的较小 guidance 强度比较**，否则很可能只是在重新调强度。这里也没有证明所有模型错误都来自模糊：若强模型已准确，锐化仍会改变正确分布；若它有尖锐假峰，也可能强化假峰。Selective Underfitting 的结果进一步限制了“凡是平滑都应该反向消除”的说法。[^8]

该候选的可失败假设是：比纯卷积更保留原分布的参考，能减少对已经合理区域的额外锐化，从而在同一头、同一采样预算下改善质量。若只有减小强度的对照有效，就不能把收益归为混合目标。

**第二个候选：让参考多学习“模型生成过量的部分”。** 这条承接此前的[共同误差分布推导](AG_IG_SHARED_CONTAMINATION_DERIVATION_20260912_ZH.md)与[结构化 mismatch 分析](AG_IG_STRUCTURED_MISMATCH_20260912_ZH.md)，比 Gaussian 更接近我们原先的误差分布研究。它不需要假设

\[
G=(1-\varepsilon)P+\varepsilon E
\]

在全空间严格成立。对具有共同支配测度的总体密度，可以直接定义

\[
m(x)=\min\{g(x),p(x)\},\qquad
e_+(x)=[g(x)-p(x)]_+,\qquad
e_-(x)=[p(x)-g(x)]_+.
\tag{11}
\]

于是

\[
g=m+e_+,\qquad p=m+e_-,\qquad
\delta=\int e_+=\int e_-=\operatorname{TV}(P,G).
\tag{12}
\]

这是重叠质量与剩余质量的标准分解，不是关于网络误差方向的额外假设。共同部分未必等于整个理想分布；模型缺少的部分 (e_-) 也明确保留了下来。(e_+) 指统计上的过量生成，可能包括过度重复的正常图像类型，不能直接等同于逐张视觉上“坏图”。

构造一个始终为正的参考：

\[
\boxed{
q_\lambda(x)
=\frac{g(x)+\lambda[g(x)-p(x)]_+}{1+\lambda\delta}
=\frac{m(x)+(1+\lambda)e_+(x)}{1+\lambda\delta},
}\qquad \lambda>0.
\tag{13}
\]

这相当于保持共同部分相对不动，把**生成过量的质量**增加 (1+\lambda) 倍后归一化。它有三个与均匀学习自身样本不同的性质：

- 参考有明确的目标偏差；它不是仅凭有限容量或少训练步数偶然变弱。
- 若 (G=P)，则 (Q_\lambda=G)。在强场也与该分布自洽、weak 精确的理想极限下，额外引导消失。纯 Gaussian 平滑通常不具备这个固定点。
- 构造没有消除生成不足的部分。抑制过量与补回缺失是不同任务，不能声称式 (13) 自动恢复完整 (P)。

从强模型样本 (X\sim G) 出发，它正好对应正权重

\[
\boxed{
w_\lambda(x)
=1+\lambda\left[1-\frac{p(x)}{g(x)}\right]_+,
\qquad 1\le w_\lambda(x)\le1+\lambda.
}
\tag{14}
\]

把它代入式 (2) 就得到具体训练 loss。选 \(\lambda=1\) 时，权重落在 \([1,2]\)。它不需要负 loss，不需要 hard rejection，也不要求先把少量异常质量归一化成一个独立训练集。

对比一个容易想到、但不适合直接实现的目标：

\[
\mathcal L_{\rm signed}=(1+\lambda)\mathcal L_G-\lambda\mathcal L_P.
\]

它形式上对应 \((1+\lambda)g-\lambda p\)。如果共同污染假设及正性条件确实成立，能写出漂亮的误差放大解释；一般情况下这个密度会在某些区域为负，平方 loss 的局部二次项甚至可以向负无穷下降。正则化或裁剪会改变目标，不能把它们称为不改变含义的修补。式 (13) 是一个明确的正分布替代。

**权重怎么估计，额外成本在哪里。** 在类别先验一致的真实／生成端点数据上，以各半来源训练一次二分类器。若 \(\eta(x)=\Pr(\mathrm{generated}\mid x)\) 精确，则

\[
\eta=\frac{g}{p+g},\qquad
\frac p g=\frac{1-\eta}{\eta},\qquad
w_\lambda=1+\lambda\left[\frac{2\eta-1}{\eta}\right]_+.
\tag{15}
\]

只在 \(\eta>1/2\) 时增加权重，因此实际使用分母的区域满足 \(\eta>1/2\)，不会因为 \(\eta\to0\) 产生巨大权重。实施时对另一折图像离线计算权重，固定后只训练一个原有尺寸的内部读出。**采样阶段没有分类器调用、分类器梯度或在线密度比系数。** 训练分类器、生成数据和提取端点特征均应计入离线成本。

重要性加权和判别器密度比估计都有大量先例。Discriminator Guidance 已用真实／生成差异改进扩散采样；TIW-DSM 和 Discriminator-Weighted Diffusion 已研究以密度比调整扩散训练。[^4][^9][^10] 这里待检验的具体变化是：把可解释的过量质量**加重到负参考的训练目标里**，用原内部读出实现，而不在每一步通过外部分类器求梯度。尚未确认这一具体构造的首创性。

这条路最难的地方不是代数。高维图像中真实与生成分布可能非常容易分开；有限真实数据的原子经验分布与连续生成分布甚至可以互相奇异，此时式 (14) 会几乎处处成为同一个常数，退回均匀自身样本训练。分类器记住训练集、识别不同 JPEG／resize／编码流程，也会导致同样问题。

所以式 (11)—(15) 讨论的是总体分布，不能直接把有限训练集的原子密度当作 (p)。实际必须采用相同预处理、留出预测与来源匹配，并检查权重是否退化。如果用固定图像表征做分类，则精确识别的是**表征分布**的过量部分；诱导出的 (G(x)w(f(x))/Z) 仍是合法目标，但不能再宣称它等于完整像素空间的式 (13)。它也不能纠正该表征看不见的细节错误。当前建议将其排在可直接构造的混合平滑之后，而不是先投入大规模分类器调优。

**加噪后还能成立什么。** 对任意固定端点正权重，有一个完整路径恒等式：

\[
q_t(z)=\frac{g_t(z)M_t(z)}{Z_w},\qquad
M_t(z)=\mathbb E_G[w(X)\mid Z_t=z],\qquad
Z_w=\mathbb E_G[w(X)].
\]

因此

\[
\boxed{s_G(z,t)-s_Q(z,t)=-\nabla_z\log M_t(z).}
\tag{16}
\]

它抑制的是那些在模型的加噪后验下更可能对应“过量生成端点”的状态。端点权重通过后验传播到中间时间，与 energy-guided diffusion 中条件期望的结构相同；CEP 等工作已研究相应的中间引导问题。[^11]

但正部算子与加噪不交换：

\[
\mathcal F_t[(g_0-p_0)_+]
\ne[\mathcal F_tg_0-\mathcal F_tp_0]_+
\]

一般成立。因此不能把式 (13) 中的端点 (p,g) 直接换成每个时刻的 (p_t,g_t)，更不能声称这套训练每一步都精确沿 (s_P-s_G) 修正。仅在端点、密度光滑且远离 (p=g) 的边界，才有

\[
s_G-s_Q=
\begin{cases}
\displaystyle\frac{\lambda(p/g)}{1+\lambda(1-p/g)}(s_P-s_G),&g>p,\\[6pt]
0,&g<p.
\end{cases}
\tag{17}
\]

式 (17) 给出目标选择的方向动机；式 (16) 才是整个加噪族的正确解释。还有一项限制：当 (p/g\to0) 时，式 (17) 的系数也趋于零，有界权重并不为极端偏离区域提供任意强的修正。有限 strong 还带有场与生成分布的自洽误差，内部 weak 也有拟合误差。因此本文没有从这些式子推出最终采样分布精确恢复或 FID 必然改善。

**与我们已有结果对齐。** 以下是同轮对照的 FID，数字仅用于排除重复方案，不跨行比较绝对质量：

|仓库已经做过的路线|同轮结果|本次处理|
|---|---|---|
|直接训练 guided MSE，SiT IG，1K|直接目标 64.7947；同预算原生目标 64.8497；ADG 64.6906|不换名称重提反射 target|
|四次误差训练 weak，SiT IG，1K|quartic 67.6047；square 66.8430|不继续扫误差指数|
|原尺寸头学习 strong 生成数据，SiT IG，1K|生成数据 FM 65.1729；真实数据 FM 65.0208；ADG 64.6900|均匀自身样本是对照，不当作新 idea|
|RAEv2 Context MLP 20K，独立 5K|MLP 7.1223；native 6.9208|更低训练 loss 不能单独支持引导改进|

来源为[直接目标结果](SIT_GUIDED_OBJECTIVE_RESULTS_20260912_ZH.md)、[Bayes risk 结果](IG_BAYES_RISK_RESEARCH_20260912_ZH.md)、[参考 loss 对照](GUIDANCE_REFERENCE_LOSS_RESEARCH_20260912_ZH.md)、[RAEv2 5K](RAEV2_CONTEXT_5K_RESULTS_20260914_ZH.md)。这些有限配置的负结果不否定所有相近训练目标，也不是 SSG 完整 adapter 的复现结论。

MLP 在 SiT/JiT 的独立 5K 正结果可以使它成为控制容量与计算预算的载体；此次研究变化应该只来自训练目标，不能把头的表达能力再次包装成核心解释。另一个值得参考的训练工作 SGG 会修改强模型的训练目标，并使用 stop-gradient 弱信号；它没有直接解决这里“冻结强模型后如何定义参考分布”的选择问题。[^12]

**CFG 与 IG 应怎样分别落地。** IG 先做上述逐类参考目标：冻结整个强主干，以同一深度、同一结构、同一初始化、同一特征归一化统计的 readout 做比较，沿用既有 guidance 系数与窗口。采样只在原强主干经过中间层时读取新头，替代原 weak 读出；额外开销应是小头本身，仍需实测。

CFG 可以保持 conditional/null 两次主干查询的结构，冻结 conditional，训练 null 读出学习类边际的混合平滑分布，或类边际的过量加权分布。但此时 null 已不是与 conditional 完全相同数据分布的精确无条件边际，原 CFG 的纯类别后验解释会改变。应称为参考先验改造的 CFG 变体，并单独与原 CFG、APG 及同预算 null 续训比较。不能因为 IG 改善就推断 CFG 改善，也不建议把 CFG 实验和 IG+CFG 的额外组合同时展开。

**最小真实模型实验方案。** 这是可执行前的研究规格，尚未创建训练队列或冻结运行请求。优先用 SiT-S/2 的廉价环境筛选目标，再把通过者迁移 JiT；RAEv2 留作后续泛化检验。

第一轮使用已有 SiT 强模型无引导生成的连续 latent bank 与对应真实 bank，各 2K、每类 20 个；这只是小数据筛选，不能标成 SSG 的百万样本复现。真实、生成来源的差异必须保留为控制。四个 head 均采用已验证的同一 Context 结构、相同初始化及冻结归一化统计，以匹配的原生训练配置训练固定 3K，使用最终 EMA：

|头|训练端点与监督|目的|
|---|---|---|
|A|真实数据，原生 FM|同预算真实目标控制|
|B|固定 strong 生成数据，原生 FM|均匀自身样本控制|
|C|同一生成 bank，式 (6)|纯 Gaussian 基准|
|D|同一生成 bank，普通对与式 (6) 各半|混合参考候选|

先验固定 \(\beta=1/2\)，Gaussian 标准差设为生成训练 bank 全局每维中心 RMS 的一半。这是可解释的初始尺度约定，**不是理论推出的最优值**；本轮仅用这一尺度，不按 FID 回调。SiT 用速度目标，时间分布、batch、优化器等继承 Context 训练配置。原 bank 的生成模型、采样器与参考训练状态需记录为同一个明确的 (G)。

质量比较保持 Heun64、既有 IG 额外系数 \(\alpha=0.8\) 与原窗口，使用新 1K 配对噪声及类别。固定比较 A、B、C、D、C 的半强度、已有最佳 Context、原 IG。这里半强度是排除“混合只等价于减小 guidance”的唯一额外系数对照。若 D 未胜过全部相关强对照，则结束这个固定构造。若出现至少 1 FID 的实际筛选优势且 IS 不明显下降，再冻结 D 与最强对照做独立 5K；1 FID 是资源推进标准，不是显著性结论。之后 JiT 保持构造不变迁移，不为每个模型展开宽度、步数、强度与平滑尺度网格。

第二轮再评估式 (14)，只增加两个头：过量权重、同类别内打乱的同一组权重；复用相同均匀生成数据头 B。离线端点分类器采取两折留出预测，\(\lambda=1\)，固定权重后训练，按整个训练 bank 的类别内平均权重归一化，不用每个 mini-batch 随机归一化。若来源处理不一致或权重基本常数，先记录目标退化，不能把一条不可识别的权重方案直接投入大规模训练。质量门槛与独立确认沿用第一轮，分类准确率不作为方法有效性的替代指标。

两轮都报告实际头训练时间、离线样本／权重准备成本、主干与额外 prefix 调用数、完整采样延迟、FID、IS 和固定配对图。它们回答的是新训练目标能否产生真实收益；后验可视化、toy 模式图、局部 score 指标均不作为晋级理由。

**本轮复核范围。** [代数检查脚本](../experiments/weak_reference_loss_20260914/check_identities.py)核对了 Gaussian 监督积分、FM 重标度、混合 score、过量分布归一化及端点 score 关系，并给出正部不与加噪交换、signed density 可能为负的反例。[检查记录](data/weak_reference_loss_20260914/algebra_checks.json)全部通过，恒等式误差约 \(10^{-14}\)，积分归一化误差约 \(5\times10^{-12}\)。这验证公式及其边界，没有提供新图像质量证据。

我的判断是：**下一步最有价值的变化，是让参考的训练目标明确规定“哪些质量应该被增加”，再观察强减弱是否因此更有效。** 混合平滑提供一个直接可采样的起点；过量质量加权提供一个更贴近模型误差的起点。若真实生成比较不支持，就保留数学事实并放下当前方法，不靠继续降低 weak 自身的 loss 来维持解释。

**来源。** 文献核对截至 2026-09-14。以下是原始论文或作者代码；本文候选目标、具体组合与实验规格是研究推导，尚无新颖性或生成收益的确认。

[^1]: Zixuan Fu 等，[A Frozen Pixel-Space Diffusion Model Can Guide Itself with Its Own Samples](https://arxiv.org/html/2607.29122v1)，§3–4、附录实现；[作者代码](https://github.com/zfu006/SSG)，实现核对 commit `34d760f6ac1dc51f11386408556c8483b3f3f9da`。
[^2]: Sina Alemohammad 等，[Self-Improving Diffusion Models with Synthetic Data](https://arxiv.org/html/2408.16333v1)，SIMS，合成数据训练负参考。
[^3]: Tiancheng Li 等，[Self-Guidance: Boosting Flow and Diffusion Generation on Their Own](https://arxiv.org/html/2412.05827)，§IV；跨噪声参考及 heat equation 动机。该思路早于 SSG。
[^4]: Yeongmin Kim 等，[Training Unbiased Diffusion Models From Biased Dataset](https://arxiv.org/html/2403.01189)，§3、附录 A.5；特别是只按 noisy state 加权与正确 score 修正的区别。
[^5]: Chen-Hao Chao 等，[Denoising Likelihood Score Matching for Conditional Score-based Data Generation](https://arxiv.org/abs/2203.14206)，ICLR 2022；[作者方法说明及公式](https://chen-hao-chao.github.io/dlsm/)。
[^6]: Chenghan Xie、Jose Blanchet、Renyuan Xu，[Sobolev Regularized Score Difference Estimation in Diffusion Models](https://arxiv.org/html/2608.18237v2)，§1–2；分类后求导的统计误差与 Sobolev 正则。本文不采用采样时分类器求导。
[^7]: Makoto Yamada 等，[Relative Density-Ratio Estimation for Robust Distribution Comparison](https://arxiv.org/abs/1106.4729)，2011 预印本、Neural Computation 2013；相对密度比与有界分母构造。
[^8]: Kiwhan Song 等，[Selective Underfitting in Diffusion Models](https://arxiv.org/abs/2510.01378)，训练监督区域、采样外推区域与泛化的关系。
[^9]: Dongjun Kim 等，[Refining Generative Process with Discriminator Guidance in Score-based Diffusion Models](https://proceedings.mlr.press/v202/kim23i.html)，ICML 2023；由真实／生成样本估计 score 修正。
[^10]: Seonghyun Ban、Heesan Kong、Kee-Eung Kim，[Data Augmentation with Diffusion for Open-Set Semi-Supervised Learning](https://openreview.net/pdf/9ccac95da6d454260441b74551f011a2b4e7224c.pdf)，NeurIPS 2024，§4、Algorithm 1；[作者实现](https://github.com/khtks/DWD)。Discriminator-Weighted Diffusion 的端点重要性加权训练用于限定加权技术本身的新颖性。
[^11]: Cheng Lu 等，[Contrastive Energy Prediction for Exact Energy-Guided Diffusion Sampling in Offline Reinforcement Learning](https://proceedings.mlr.press/v202/lu23d.html)，ICML 2023；端点能量诱导的中间引导问题。
[^12]: Liangyu Yuan 等，[Improving Diffusion Generalization with Weak-to-Strong Segmented Guidance](https://arxiv.org/html/2603.20584)，§4.3、附录 D；弱信号参与强模型训练，与本轮冻结强模型后更改参考目标的区别。
