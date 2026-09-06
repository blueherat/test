# RAEv2：从实际分布与 flow 配对理解 guidance

日期：2026-09-06。本轮阅读下面三篇原文的方法、主要理论及实验，并核对所列相关附录。本文没有新训练或 FID 结果。理论的假设、论文的方法选择、本文对 RAEv2 的推论分开记录。

**Analytic Distribution of Classifier-Free Guidance for Schedule Design，Jiang / Ma，2026 预印本。** 已读 [v1 原文](https://arxiv.org/html/2607.19725v1) §4–6、Appendix A.1–A.2。最有价值的是精确的轨迹分布公式：当 p、q 是同一前向扩散的正则边缘、使用 exact scores、乘积可归一化且 ODE 良定时，确定性 CFG 的终点不只是 `p^ω q^(1−ω)`，还乘上沿 guided characteristic 积分的因子：

\[
\exp\!\left[-\tfrac12\omega(\omega-1)
\int_{t_0}^{T}g(s)^2\|s_p(X_s)-s_q(X_s)\|^2ds\right].
\]

其证明关键是乘积参考边缘满足含源汇项的 Fokker–Planck 方程，而 ODE 只做质量输运。`t₀>0`、初始参考分布、尾部条件都不可省略。p/q 不一定非要条件/无条件，但必须确实对应所假设的扩散边缘。

论文的三因子 schedule 另加入信号权重和低噪声误差抑制；分布定理没有唯一推出这个最优 profile。SD1.5 的 1K、8-NFE 主表中，最佳常数 CFG 的 FID 67.67，最佳所提 schedule 为 67.87；收益主要在过强 guidance 区域。应借鉴其“局部场对应何种整条轨迹分布”的推导，不据此启动手工调度或声称超过强基线。[原文 §5–6、Table 2](https://arxiv.org/html/2607.19725v1#S6)

**对 RAEv2 的推论。** native F/B 是同类条件的有限场，已有非保守性证据；不能把它们代入公式后将 `||F−B||²` 解释成已知的质量惩罚。更一般地，若任意正则参考密度为 ρ_s，实际速度为 b_s，定义连续性残差

\[
\mathcal R_s=\partial_s\rho_s+\nabla\cdot(\rho_s b_s),
\]

则实际输运密度 q_s 沿 b_s 的 characteristic 满足

\[
\frac d{ds}\log\frac{q_s}{\rho_s}(Z_s)
=-\frac{\mathcal R_s}{\rho_s}(Z_s).
\]

这是本文独立写出的连续性恒等式，不要求 b 是 score。积分时仍须保留初始密度比；消除残差结合相同初始分布、输运适定性及终点极限，才可恢复参考边缘。它解释了为什么应追踪沿轨迹累积的分布残差；但它不自动给出有限模型中可估计的 `ρ_s` 或质量目标。旧 observable potential 已以真实 bridge ρ 为参考学习修正此类残差，有限实现并未消除残差，故把这条恒等式重新命名不会形成新方案。

**Emergence of Distortions in High-Dimensional Guided Diffusion Models，Ventura et al.，2026 预印本。** 已读 [v1 原文](https://arxiv.org/html/2602.00716v1) §3–5，重点核对 §4.1 的谱公式与 §4.2 的极限假设。它在 exact-score 条件下仍得到 guidance 导致的均值外推和类内协方差收缩，说明分布偏置不只来自网络估计误差。

联合 Gaussian 的闭式结果假设条件与无条件协方差可交换。Gaussian mixture 分析则区分类数随维数指数/次指数增长的高维极限，后者在其分离和初始化条件下可恢复条件目标。不能把 RAE 的 1000 类和 latent 维数直接代入并声称实际采样无失真。论文后续的负 guidance 窗口含可调斜率、截距，且主要是 Gaussian 机制分析，不是同成本 ImageNet 新结果。[原文 §4–5](https://arxiv.org/html/2602.00716v1#S4)

**对 RAEv2 的推论。** “类内方差收缩”必须先说明在哪种表示下、相对哪个目标。仓库两个完整 5K bank 已显示：IG 在原 latent 中扩张类内方差，在解码 Inception 中收缩类内方差，后者还修复了 Full 的过度离散。因此直接以负 guidance 或加噪“恢复多样性”不成立；高维 Gaussian 结论提供分辨机制的坐标，而没有指定本模型该采取哪一个符号。

**On the Guidance of Flow Matching，2025。** 已读 [v3 原文](https://arxiv.org/html/2502.02150v3) §3–5、Appendix A.2–A.5 的相关推导。对于端点配对 z=(x₀,x₁)，引导到 `p′(x₁)∝p(x₁)e^(−J(x₁))` 的精确场是

\[
g_s(x)=\mathbb E\!\left[
\left(\frac{\mathcal P e^{-J(X_1)}}{Z_s(x)}-1\right)
v_{s|z}(x)\mid X_s=x\right],
\qquad
\mathcal P=\frac{\pi'(x_0\mid x_1)}{\pi(x_0\mid x_1)}.
\]

`𝒫=1` 对独立配对精确成立，对强耦合不是普遍恒等式。Gaussian/Tweedie-Jacobian 和能量梯度方法是额外结构及近似下的化简。正文训练部分同时学习条件 normalizer 与 guidance；条件回归梯度恒等式不保证两个有限网络训练成功。

实验涵盖合成数据、D4RL 和 CelebA-HQ 逆问题。论文明确报告全局 Monte Carlo guidance 在高维图像上受不可承受的样本需求限制；图像逆问题的收益也不等于无额外观测的 class-conditional 生成提升。[原文 §3–5](https://arxiv.org/html/2502.02150v3#S3)

**对 RAEv2 的建设性启发。** 可先选择有明确定义的目标端点分布，再导出合法的桥与校正标签；学习一个轻量辅助场有理论入口。关键是保留正确的两端边缘。以任意生成终点与独立 Gaussian 重新配对，可以构造新的合法 bridge，但这个 bridge 的理想速度并不等于原模型实际 rollout 速度。原模型只能作为控制先验，不能冒充该 bridge 的 exact field。

给定已明确定义、有限二阶矩的目标分布 ν_y，独立取 `ε∼N(0,I)`、`X∼ν_y`，令 `Z_s=(1−s)ε+sX`、`U=X−ε`，以及在该固定 bridge 测度下平方可积的冻结基场 b。则对足够大的辅助向量场类，

\[
\arg\min_u\;\mathbb E\|b_s(Z_s)+u_s(Z_s)-U\|^2
=\mathbb E[U\mid Z_s]-b_s.
\]

加到原 b 上会恢复这个 bridge 的理想边缘速度，无须假设原 b 是 exact score。若限定梯度闭包，则只得到相应加权投影；对有限神经网络、离散 Euler 和分布质量的保证仍需分别建立。这一推导同时表明：当 ν 仍是原 encoder 数据分布，它就回到已失败有限求解器的总体目标；换用 guidance matching 一词不会绕过失败。

**由阅读引出的精确反例：部分校正可能破坏跨时刻的误差抵消。** 这不是某篇论文的实验结论，也不是允许部署手工窗口；它用于辨别“桥残差更小”到底保证什么。取独立 `X,ε∼N(0,1)`，则上述真实 bridge 的方差 `σ_s²=(1−s)²+s²`、速度 `v*=σ′_s z/σ_s`。定义冻结场

\[
b_s(z)=v_s^\star(z)+\sigma_s k_s,\qquad
k_s=\begin{cases}1,&s<1/2,\\-1,&s\ge1/2.\end{cases}
\]

其实际终点仍为 `N(0,1)`，因为 `∫₀¹k_s ds=0`。若一个有限梯度校正器只消除前半程的平移误差，bridge 上积分平方误差从 `2/3` 降为 `1/3`，但终点变成 `N(−1/2,1)`，终点 `W₂²` 从 0 升为 `1/4`。这是直接求解线性 ODE 的解析结果；coupling MSE 也下降，因为条件不可约项不变。它没有证明 RAEv2 已发生同一机制，但指出必须检验的联系：局部纠错是否破坏原轨迹中有益的误差抵消。

**三篇共同改变的研究选择。**

1. 不再把“更完整地消除 `X−G`”当作自动成立的质量目标。先识别需要保留的引导偏置和具体待修复的分布误差；`X−F` 虽可保持旧偏置，也需要说明为何该偏置值得保留。
2. 若采用学习式质量 guidance，必须把目标端点分布、桥边缘和真实 rollout 区分开；若使用配对，明确检查配对后的目标权重。能量或分类器本身不是这些条件的替代。
3. 理论的价值应体现为少量有方向预测的检验、由结构决定的控制，而非给系数/窗口搜索补解释。上述三篇尚未为 RAEv2 导出通过全部条件的新实现，不能据此宣称已达 5%。

相关阅读：[强弱模型与监督目标](RAEV2_GUIDANCE_READING_STRONG_WEAK_20260906_ZH.md)、[分类器视角与 predictor-corrector](RAEV2_GUIDANCE_READING_GEOMETRY_20260906_ZH.md)。
