# ICG / TSG：条件参考的分布意义与随机扰动的机制

日期：2026-09-06。阅读 [No Training, No Problem: Rethinking Classifier-Free Guidance for Diffusion Models](https://studios.disneyresearch.com/app/uploads/2025/04/No-Training-No-Problem-Rethinking-Diffusion-Guidance-for-Diffusion-Models-Paper.pdf)，ICLR 2025 正式稿。实际范围：正文 §§3–8、附录 A–G 的推导、实验和伪代码；未逐图评价 H 的定性样本。另核对 [arXiv v2](https://arxiv.org/html/2407.02687v2) 中相应理论段落，未宣称两个版本逐字相同。材料归档于 `reading_independent_condition_guidance_v1`。作者页面、正文链接及有限公开检索未找到可确认的官方代码库；此处的代码阅读仅指正式稿 Algorithms 1–2、Figures 11–12，不能称为实现复现。

本文有真实的正面实验：DiT-XL/2 的 CFG/ICG FID 为 5.56/5.50，EDM2-XS 为 3.36/3.35；TSG100 与 unguided200 为 6.39/12.94。DiT 使用 10K，EDM 系列使用 50K。它提供了同 denoiser 调用数的较强 TSG 对照，但没有 RAEv2 的完整墙钟比较。ICG 仍选 guidance 强度；TSG 还选 embedding 噪声幅度、幂指数、层数和时间区间。其价值是扩展参考分支的构造，不是已经给出本任务的免调参解法。

## 独立推导：条件平均究竟是哪种 score

以下用理想、可微、正密度的有限类别模型说明结构，不假设真实 RAEv2 已满足它。固定时刻，令 `s_y(z)=∇ log p(z|y)`。准确的混合分布恒等式是

\[
s(z)=\sum_y p(y\mid z)s_y(z).
\]

取与 z 无关的类别分布 q，则

\[
\bar s_q(z)=\sum_y q_y s_y(z)
=\nabla\log g_q(z),\qquad
g_q(z)\propto\prod_y p(z\mid y)^{q_y}.
\]

这是几何平均密度的 score；通常不同于类别先验混合密度。原文附录 A 的 KL 梯度恒等式与此一致。推理时独立抽一个标签，不会使原训练联合分布中的类别与图像变成独立，也不会把已学到的条件函数替换成另一联合分布下的条件函数。网络容量增加本身不会消掉这种参考分布差异。

对固定 q 的理想平均 guidance，得到

\[
s_c+\omega(s_c-\bar s_q)
=\nabla\log\{p(z\mid c)^{1+\omega}/g_q(z)^\omega\}.
\]

这是一个准确的**瞬时**对比解释。括号内密度仍需可归一化；不同噪声时刻的这些密度也未必来自同一端点分布的加噪路径。因此它既不是“随机标签无偏估计 unconditional”，也不是实际 guided ODE 终点分布的公式。

一个可解例子使差别更具体。两类等先验，均值 `±m`、共同方差 v，且 q 均匀。则几何参考是 `N(0,v)`，`s_+−bar_s=m/v` 是常向量；真正 CFG 对比为 `2m p(−|z)/v`，随后验变化。取 `m=2,v=2,z=2`，随机类别平均 score 是 −1，真正混合 score 约为 −0.03597242。两者差异在无限精确模型里仍存在。该例只否定分布等同，不否定论文有限网络的质量结果。

它也带来一个有用的新问题：某个弱参考有益，究竟因为它近似 marginal，还是因为它在特定方向上定义了更有效的对比？前者与后者的实验解释应分开。对 RAEv2，另一个随机类别的 Full 输出需要另一次条件化 backbone 调用，不能算作当前同类 Base/Full 的免费共享分支。直接沿用 1.78 也未由上述分布关系导出合理强度。

## 独立推导：随机 embedding 更新与 Langevin 的差别

令 e 为 embedding，扰动 `δe` 均值零、协方差 Σ，弱支为 `D(e+δe)`。局部展开得到

\[
G-D\simeq-\omega J_e\delta e
-\tfrac{\omega}{2}H_e[\delta e,\delta e].
\]

因此线性随机部分均值零，其协方差为 `ω² J_e Σ J_eᵀ`；二阶平均响应由 embedding Hessian 的收缩决定。沿 Euler 步长 h 加到状态后，随机增量协方差为 `h² ω² J_eΣJ_eᵀ`。固定扰动尺度下，这不是非零扩散系数的 Langevin 连续极限；若强行令扰动随 `h^−1/2` 增长，又会改变局部展开条件和平均漂移。一般状态相关扩散还需相容的漂移项，不能仅由“高斯随机增量”推出目标密度保持。

实际论文扰动的是高维 embedding，通常不沿标量时间曲线的切向，故其效应也不能直接用 `∂D/∂t` 代表。这不排除有限步随机扰动有益，但需要区分平均结构响应、协方差及曲率偏置，不能将三者统称为纠错噪声。

## 当前取舍

保留两点启发：**弱参考的分布意义可与 unconditional 不同；随机弱支的平均对比与方差是两个独立机制。** 本文没有确定可部署的新方向、自然增益或时间规律，不据此追加随机标签、时间 embedding 强度或层窗口扫描。当前冻结的反射 1K 实验保持原样；这份阅读没有新增 GPU、训练、图像生成或 FID 计算。
