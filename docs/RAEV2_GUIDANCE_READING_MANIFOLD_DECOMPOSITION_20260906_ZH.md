# RAEv2 基础阅读：流形 score 的有限噪声分解与曲率耦合

最值得保留的机制是：**法向位置会改变切向后验不确定性，后者经曲率产生正常的法向后验均值偏置。** 因而不能把完整 score 叫作法向，也不能把论文的 `on-support score` 改称切向；将有限噪声 clean 预测直接压到流形上，可能删除正确的 Bayes 去噪结构。这提供了由几何与条件矩导出设计的入口，但当前还没有可由这篇论文直接部署的 RAEv2 guidance。

## 来源、版本与实际读取范围

Zixuan Zhang、Kaixuan Huang、Tuo Zhao、Mengdi Wang、Minshuo Chen，*Diffusion Model for Manifold Data: Score Decomposition, Curvature, and Statistical Complexity*，本次严格读 [arXiv:2603.20645v1](https://arxiv.org/abs/2603.20645v1)，提交时间 2026-03-21。arXiv 页面另列 2026-04-28 的 v2；**未读 v2，不将 v1 结论或问题自动归于 v2，也未确认会议正式录用版本**。

已下载 [v1 PDF](https://arxiv.org/pdf/2603.20645v1)、[v1 TeX](https://arxiv.org/src/2603.20645v1) 和版本页，归档 `R/reading_manifold_decomposition_v1`（R 为 restart_20260906 数据根）。PDF 86 页、2,414,168 bytes，SHA `2d55549b97307e713df55a38f06278d6742daf11cc87249ff51d112615515ab8`；TeX 包 1,917,430 bytes，SHA `c12834ad4dc2e4685e04dc728890db4eb46b26e30931a2f034afb1f340608002`。PDF 标题页日期为 2026-03-24，TeX 使用 `\today`；版本身份以 v1 链接及文件 SHA 为准。

通读正文 §1–6；精读 Lemma 3.2 与 A.2、Theorem 4.4 对应 B.1–B.3 的构造及误差汇总、Theorem 5.1 与 C.1；另读 A.1、关键 cross-term 引理和 E.1/E.2。独立 agent 另核 C.1。未逐行复核全部 D–F 辅助网络／多项式引理，未追读全部外部引用，故不是整篇 86 页证明的无缺口认证。论文没有新增图像 FID、guidance 参数或同成本性能实验。本轮无 GPU、模型查询、训练、新采样或 FID；不更新索引。

## 1. Lemma 3.2 是精确分解，不是“score 只沿法向”

论文用 OU forward：`X_t=α_t X_0+√h_t ε`，`α_t=e^(−t/2)`，`h_t=1−e^(−t)`。固定正 t，在到缩放流形 `M_t=α_t M` 的距离严格小于 `α_t τ` 的 tube 内，最近点 q 唯一。记 `n=x−q`、`m=E[X_0|X_t=x]`，则

\[
\boxed{\nabla\log p_t(x)
=-\frac{n}{h_t}+\frac{\alpha_t m-q}{h_t}
=s_\perp+s_M.}
\]

证明先微分 Gaussian 卷积得到 `(α_t m−x)/h_t`，再加减 q；这对满足 tube 条件的**任意固定正噪声**成立。所谓 small-noise regime 用于保证真实 forward 状态以高概率进入 tube、以及后续有效近似，并非恒等式额外要求 t→0。

`n` 与 q 处的切空间正交，但 `α_t m−q` 一般不在切空间，因为流形上随机点的均值未必仍在流形。因此真正的正交分量应写成

\[
P_Ts=\frac{P_T(\alpha_t m-q)}{h_t},\qquad
P_Ns=-\frac n{h_t}+\frac{P_N(\alpha_t m-q)}{h_t}.
\]

大噪声 Lemma 3.1 则用 partition of unity 将 Gaussian 卷积拆成局部图的混合，权重是相应图的后验责任概率；它不是手选最近图。局部 `on-support` 与局部正交项同样未必正交。线性／仿射支撑才有完全分离的特例。

进一步令任一缩放 clean 点为 `a=α_t x_0`：

\[
E_1=\|q-a\|^2,\qquad E_2=2\langle n,q-a\rangle,
\]
\[
p_t(x)=(2\pi h_t)^{-D/2}e^{-\|n\|^2/(2h_t)}
\int e^{-(E_1+E_2)/(2h_t)}dP_{\rm data}(x_0).
\]

距离平方的梯度在 tube 内为 `∇_x(||x−q||²/2)=n`，所以第二个积分的 log-gradient 正好是 `s_M`。**q 随 x 变化，求梯度时不能任意 stop-gradient。** 即使在 `x=q` 时 E2 的数值为零，其法向导数也一般不为零；把 E2 删掉再求梯度并不等价。

v1 A.2 在定义 g 的单个显示公式里将指数写成 `−E1+E2`；上一行密度分解、正文 Lemma 3.2 和 B.1 实际使用的是 `−(E1+E2)`。本笔记使用从平方展开直接得到的后者，并明确这是对 v1 符号不一致的辨认。

## 2. 曲率如何把法向位置和切向后验连接起来

下列是独立的局部解释，**不是将 Lemma 3.2 包装成它没有陈述的渐近定理**。需额外局部足够光滑、正密度、tube 内唯一稳定极小点与后验集中，才能控制余项。

在缩放流形 q 处选正交切向坐标 u，用第二基本形式 II 写局部图：

\[
a(u)=q+u+\tfrac12\mathrm{II}_q(u,u)+O(\|u\|^3).
\]

因 `n⊥u`，

\[
E_1+E_2
=\|u\|^2-\langle n,\mathrm{II}_q(u,u)\rangle
+O(\|n\|\|u\|^3+\|u\|^4).
\]

定义 `S_n` 为满足 `uᵀS_nu=<n,II(u,u)>` 的 shape operator，则局部 Gaussian 后验的主精度是 `I−S_n`，而不是 I；在它正定并满足上述局部渐近条件时，切向后验协方差主项为

\[
\operatorname{Cov}(u\mid x)\simeq h_t(I-S_n)^{-1}.
\]

后验均值的法向偏移相应包含

\[
P_N(\alpha_t m-q)
=\tfrac12\mathrm{II}_q:\mathbb E[uu^\top\mid x]
+\text{局部余项／非局部后验贡献}.
\]

这里是**原始二阶矩**，不能不加条件就替换成协方差。系数 1/2 来自嵌入图的 Taylor 展开，不是 guidance gain。切向密度及体积元也进入后验权重；法向位置、曲率和密度不能通过删掉一个方向普遍解耦。若固定 n≠0，`−<n,II(u,u)>` 就是二阶项，不能藏进 `O(||u||³)`。

论文用 reach 控制交互项，例如局部证明得到 `|E2|≤4||n||E1/(α_t τ)`，再在高概率小噪声域控制其 Taylor 截断误差。它保留 E2 的幂，而不是证明应删 E2。Reach 不仅反映曲率，还会受不同流形部分彼此靠近的影响，不能直接当单一局部曲率数值。

## 3. 一个精确圆例：有限噪声 mean 与 projection 不同

以下无数值实验，是对分解的解析核对。令 X 均匀分布在半径 R 的圆上，`Y=X+√h ε`；对 `x=r e_r`、`r>0`，设 `k=Rr/h`，`A(k)=I_1(k)/I_0(k)`，其中 `I_j` 是由角度积分定义的修正 Bessel 函数。直接对角度密度积分有

\[
p_h(x)=\frac{e^{-(r^2+R^2)/(2h)}}{2\pi h}I_0(k),\qquad
m_h(x)=R A(k)e_r,
\]
\[
s(x)=\frac{-r+RA(k)}h e_r,
\quad q=R e_r,
\quad s_M=\frac{R(A(k)-1)}h e_r.
\]

有限 h 下 `0<A(k)<1`，故即使 x 在圆上，posterior mean 仍在圆内；此时 `s_perp=0`，而 `s_M` 是非零的**纯法向**项。固定 r、h→0，`A(k)=1−1/(2k)+O(k⁻²)`，从而 `s_M→−e_r/(2r)`；在 r=R 时是圆的平均曲率向量的一半（采用曲率向量为切向基上 II 的迹，不另除 d 的约定）。

Jacobian 也可精确微分。它的径向、切向特征值分别为

\[
\lambda_N=\frac{R^2}{h}A'(k)\to0,\qquad
\lambda_T=\frac Rr A(k)\to\frac Rr.
\]

因此固定 off-manifold 点的极限是 `J_m→J_Π=(R/r)P_T`，**不是 P_T**。圆内 r<R 时切向响应甚至大于 1。只有 x 在流形上、或法向偏移也趋零时，才恢复该圆例的 `J_m→P_T`。这与局部 `I−S_n` 因子一致：圆的 `S_n=−(r−R)/R`，故 `(I−S_n)⁻¹=R/r`。这个例子同时排除了“近似 score 的整个 tube 内 posterior Jacobian 都趋向正交切投影”的过强说法。

## 4. Theorem 4.4/5.1 提供什么，不提供什么

两者要求：紧致光滑 d 维嵌入流形、正 reach 下界 τ、额外指数图覆盖正则性，以及有严格正上下界的 β-Hölder 流形密度。这里是无边界型平滑局部几何设置；不能默认为包含任意分层、硬边界或自交数据支撑。

| 原文结果 | 读到的证明路线 | 对固定 RAE checkpoint 的边界 |
|---|---|---|
| 4.4：存在 ReLU 近似器，L²(P_t) 平方误差主阶 `D^(2γ+d+2)ε²/h_t`，`γ=ceil[β log(1/ε)/(log(1/ε)+β logτ)]` | B.1/B.2 分别做局部图／指数函数／嵌入的多项式近似，构造乘法、投影、图选择及比值网络；控制 tube 外尾部；B.3 合并两种噪声区间 | 是容量与存在性结果；所构造网络包含未知 geometry／density 信息，不是现有 E、D 或 Full 的认证 |
| 5.1：作者宣称训练集期望下、时间平均 L²(P_t) 误差主阶 `D^(2γ+d+2)n^(−2β/(d+2β))` | C.1 依赖 4.4 的近似偏差，再用中心化 excess loss 的 Bernstein 方差界、covering number 与全局 ERM，平衡 `ε²` 和 `ε^(−d/β)/n` | 非逐图／逐时刻界，无固定 checkpoint 优化误差、SGD 收敛或有限 noise/time 积分误差保证；证明含隐藏对数 |

5.1 中 `γ=ceil[β/(1+(d+2β)logτ/log n)]`，并需足够大的 `n>τ^(−(d+2β))`。其 `t0` 是噪声时间截断，不是训练迭代 early stopping。积分

\[
\frac1{T-t_0}\int_{t_0}^T\frac{dt}{1-e^{-t}}
=\frac{\log(e^T-1)-\log(e^{t_0}-1)}{T-t_0}
\]

在 `t0=n^(−c)`、`T=c log n` 这样的配对下为常数量级，因此不自动带来 n^c 的幂损失；仅写 `T=O(log n)` 上界不能任意套用此消去。

其时间切换是为证明两个近似域之间有重叠而构造的分片线性函数，阈值依赖 ε、τ、维数和图的常数。它有明确近似理论用途，**不是论文替当前推理 checkpoint 决定了一个免估计的 guidance window**。§5.2 的分布论述涉及连续 reverse SDE 的 Wasserstein 距离；不是 RAE 的有限步 Euler、FID 或公平成本比较。

需要保留的 v1 证明问题：

1. B.2.3（PDF p.47）给 `s3` 的分母下界时，先限制积分为 `||x−α_t x0||≤√h_t`，再声称其至少为常数倍 `(√h_t/α_t)^d`。但其分析域 `K_t(ε)` 允许 `dist(x,α_t M)>√h_t`，此积分域可为空；固定 t、ε→0 时更不能获得写出的与 x 无关的正下界。缺少额外局部条件或法向衰减因子。
2. C.1 将原网络类的 ERM 缩至有 `C_R/√h_t` 输出界的类，未充分说明近似器的存在为何允许原 ERM 同样选择；Lemma C.1 的 C.6 又用流形密度下界把给定 x0 的条件期望直接控制为边缘期望，这一步按现有论证不成立，后续还将相关界用于未裁剪的真 score。

因此，这里把统计率记为**作者的主张与已读证明路线，尚未由我们独立确认整套证明成立**。这些问题不推翻可直接从 Gaussian 卷积重建的 Lemma 3.2，也不抹掉“保留曲率交互、利用 intrinsic dimension”的有用结构；本轮没有修补该论文或声称证明定理为假。

## 5. 能导向什么机制，以及 RAE 的入口边界

可保留一个具体方向：**把正确的法向 posterior mean 偏置视为切向后验二阶矩经 II 的必然结果，而不是统一的 off-support 错误。** 若未来能获得可信的局部曲率、条件矩和误差控制，其耦合式会决定修正的结构及 1/2 系数，不需要先手选 gain。但当前真实 latent 流形的 q、II、reach 和后验矩未被识别；AE cycle 不能因此充当最近点映射，已有的平坦仿射约束也不给出未知非线性曲率。

另一条精确条件关系：若两个目标具有同一支撑流形与同一 Gaussian 噪声，score 差中的共同 `−n/h` 会自然相消，但剩余的 `s_M` 差仍可含法向曲率效应。系数和为 1 的线性 score mixing 已保留这个共同项；这不能单独导出新的 guidance，也不能把实际 Full/Base 自动视为两个已知精确目标 score。

对 RAE bridge，可在固定时刻令 `α=1−t_RAE`、`h=t_RAE²` 复用卷积恒等式，但不能直接移植 OU 动力学／切换阈值。对应小噪声 tube 需要 roughly `√(D) t_RAE≪(1−t_RAE)τ`；这约束的是原 forward 分布，不能保证受 IG 驱动的实际 rollout 同样落在 tube。`t_RAE=1` 时 α=0，缩放流形的正 reach tube 退化，局部投影分析不能作为全轨迹初始保证。

尤其不要由纯噪声时 `score=−z` 推出 clean guidance 必须为零：`s_t=[(1−t)M_t−z]/t²` 在 t=1 已丢失 M_t 信息，clean head 的边界后验均值仍可依赖条件。score 与 clean 参数化在该边界不可直接反解。

本轮不据这些代理结构提出新增性能臂，不改冻结 sampler，不拟合曲率阈值或时间窗口。5% 以上公平成本 FID 改善仍须真正的机制设计与直接质量实验闭合；这篇阅读给出的进展是把“正常的曲率偏置”与“模型误差”严格区分开。
