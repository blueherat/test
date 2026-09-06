# 从 guidance 的局部近似追到输出偏差：DPS 反应项阅读

阅读对象：Delgadino 等，*Diffusion-Based Posterior Sampling: A Feynman-Kac Analysis of Bias and Stability*，arXiv:2605.06538v1，2026-05-07，[原文](https://arxiv.org/html/2605.06538v1)。26页 PDF SHA `586ee8b5e1f8fc2fadf3281ae6b7a6a785a029e2573217d14e420f2ed222923e`。已读正文 §§1–5、附录 A–H 的文本与公式，重点独立核算 A、D.1–D.4；本地渲染第11、16页确认公式。没有复现作者图像实验、取得官方实现或核读全部引用文献。原文、文本、页面及本笔记归档于 `R/reading_dps_feynman_kac_v1`。

收获是一个明确对象：**近似 guidance 的路径反应项怎样改变最终样本密度**。这连接局部近似与分布偏差，比单独压低预测 MSE 更接近我们的设计问题。论文部分公式有实质不一致，以下保留可独立推导的机制及边界，不将其全部结论作为已验证保证。

## 1. 用概率路径定位误差

设真实 prior 的 OU 加噪路径满足 `∂_t p=Δp+div(xp)`，t 为前向噪声时间。对固定干净端点 reward R，真正目标是 `π_0∝exp(R)p_0`。DPS 以 posterior mean `m_t=E[X_0|X_t=x]` 代替整个 posterior，构造 `h_t=exp(R(m_t))`、`π_t=h_t p_t/Z_t`。

这个路径在 t=0 与 t→∞ 有合适端点，有限 T 通常仍需处理初始密度比；它一般不是目标的 OU 加噪路径，真正的加噪权重为 `E[exp(R(X_0))|X_t]`。论文 D.2 的乘积微分给出

\[
\partial_t\pi_t=L^*\pi_t+(c_t-\partial_t\log Z_t)\pi_t,
\qquad
c_t=\frac{\partial_t h_t-\Delta h_t-(x+2\nabla\log p_t)\cdot\nabla h_t}{h_t}.
\]

满足积分及尾部条件时 `∂_t log Z_t=E_{π_t}c_t`，所以真正相关的是反应项的空间变化；仅减少其常数部分不改变归一化分布。反向使用 `x+2∇logπ` 的普通扩散采样会遗漏该反应项。D.4 特定情形的密度比 PDE 可独立推出 OU Feynman–Kac 表达式，并保留有限T的初始密度比；这不需要沿用附录A错误的一般公式。

此机制也说明，改了引导漂移以后必须重新计算其路径偏差。不是先算原路径的一个标量，再假设任意降低这个标量的干预都改善终点。

## 2. 后验不确定性怎样进入反应项

令 `a_t=e^t−e^(−t)`，真实 posterior mean 满足 `J_m=Σ_t/a_t`，其中 `Σ_t=Cov(X_0|X_t)`。其 backward Kolmogorov 关系使一阶微分项抵消，留下

\[
c_t=-\frac{1}{a_t^2}
\left[\operatorname{tr}(\Sigma_t H_R(m_t)\Sigma_t)
 +\|\Sigma_t\nabla R(m_t)\|^2\right].
\]

系数由噪声过程导出，不是 guidance 强度扫描。它保留 **不确定性方向与 reward 曲率／梯度的耦合**。只看 `trΣ` 或总 uncertainty 会丢失方向和符号：H_R 可以为负，两个贡献可以相消。

一个独立解析例子进一步说明这一点。取一维 stationary OU prior `p_0=N(0,1)`，令 `a=e^(−t)`，则 `m_t=ax`、`Σ_t=1−a²` 与 x 无关。取 `R(u)=−λu²/2`、λ>0，直接代入得

\[
c_t(x)=\lambda a^2-\lambda^2 a^4x^2,
\quad
c_t-\mathbb E_{\pi_t}c_t
=\lambda^2a^4\left[\frac{1}{1+\lambda a^2}-x^2\right].
\]

反应项空间变化非零，但 `∇trΣ_t=0`。因此论文所讨论的 uncertainty-trace drift 在此精确例子完全没有修正作用；不能将降低总 posterior variance 当成一般偏差消除定理。论文 §4 的实际 drift intensity r 本身也仍是超参数。

## 3. 独立扩展：有限 denoiser 多出来的项

以上抵消依赖真实 posterior mean。对任意光滑有限预测 `m_t:R^D→R^k`，取 `J_m∈R^{k×D}`，定义

\[
b_t=x+2\nabla\log p_t,
\qquad
\mathcal Dm_t=\partial_t m_t-\Delta m_t-J_m b_t.
\]

直接对 `h=exp(R(m))` 求导，得到同一路径反应项的精确表达式

\[
\boxed{c_t=\nabla R(m_t)\cdot\mathcal Dm_t
-\operatorname{tr}(J_m^T H_R(m_t)J_m)
-\|J_m^T\nabla R(m_t)\|^2.}
\]

这个式子不要求 J 对称，也不先把它认作 posterior covariance。另一位协作者独立核对了乘积法则、转置和前向时间方向。若 R 显式依赖时间，还要加 `∂_tR(m_t,t)`；归一化仍需减 `∂_tlog Z_t`。

它指出未来可以追问的具体机制：有限预测的跨噪声动力学误差 `Dm`，与 oracle Jensen／曲率偏差是否同量级、是否沿当前 guidance 目标产生同向作用。不能直接删掉第一项，再把神经网络 Jacobian 插进 oracle 公式。当 prior score 本身也只近似时，还会引入相应误差；这里没有声称已获得 RAE 的实际密度或可部署修正。

## 4. 原文中需修正的公式与结论边界

以下按v1原文，不猜测后续版本或未取得的代码如何实现。

- **Eq.8 与 Eq.44 不一致。** Eq.44 将 `a_t^−2` 乘到 trace 和 gradient-square 两项，符合上面的直接链式求导；主文Eq.8与附录随后标作“equivalently”的式子却只给 trace 乘该因子。D.3 中间展示行的 trace 也有正负号不一致，最终Eq.44的共同负号才与乘积法则相容。
- **附录A Lemma2 的一般密度比方程不成立。** 若 `∂π=Δπ−div(vπ)+fπ`、prime同理，令g=π'/π，直接乘积展开应得

\[
\partial_tg=\Delta g+(2\nabla\log\pi-v')\cdot\nabla g
+[\operatorname{div}(v-v')+(v-v')\cdot\nabla\log\pi+f'-f]g.
\]

原文漂移多了v，反应项还缺少 score 交叉项并有符号问题。最简单的检验是共同 OU drift `v=v'=−x`、π=N(0,1)、π'=N(me^(−t),1)：真实g方程漂移为−x，原文给−2x。一般时变 Feynman–Kac 的辅助 drift 还需按终点时间反转；原文只反转 potential，没有同步反转 drift。D.4 的专门 OU 比值推导恰好用到了正确的−x，并且该辅助漂移不显含时间，所以仍可独立保留这一特例。

- **附录B.2.1 固定off-manifold的展开漏掉曲率。** `x−Π_Mx=n` 固定非零时，`||x−φ(u)||²` 中含二阶项 `−〈n,II(u,u)〉`，不能装入 `O(||u||³)`。因此整个 tube 内 `J_m→P_T` 的结论过强。半径R圆上均匀数据、固定观测半径r，posterior mean 的切向导数极限为 `R/r`，与最近点投影的导数相同，只有r→R才趋向1。这个边界与[流形分解阅读](RAEV2_GUIDANCE_READING_MANIFOLD_DECOMPOSITION_20260906_ZH.md)的保留曲率结构一致。
- **时变 reward 反应项并非简单线性乘 schedule。** 若 `logh=η(t)R(m_t)`，其未中心化反应为 `η'R−η tr(J_m^T H_R J_m)−η²||J_m^T∇R||²`，精确m的其余项才抵消。论文早停推导里把整个旧reaction乘η，会漏掉平方项的η²。停止后分布经真实 posterior kernel 运输的原则可保留，具体v1系数不能照抄。
- **DDPM时间换算与稳定性需保留局部范围。** AppendixF精确式为 `Δt=−log(1−β)/2`，其一阶展开应为β/2；随后用了β/4，故具体数值schedule不能作为已核尺度。对固定正步长下的 `sign(x)` 显式更新，通常出现两点往复，确实展示一种不光滑 guidance 机制；但恰好落零、减小步长族、约束Jacobian退化等情形不支持“一切步长、所有轨迹必然振荡”的无条件说法。本文不据此设计手工早停。

## 5. 实验究竟支持什么

论文的正面实证是机制展示：四分量Gaussian mixture、二次reward下，20条OU路径估计的相对权重图与50万DPS样本的模式偏差相对照；MNIST的classifier reward下观察到相邻增量振荡，最后100步停止guidance后该时段振荡消失，但reward也不再向零推进。MNIST使用guidance常数0.1和历史50步子空间显示。作者报告单H100上几分钟；没有本任务所需的 RAE/ImageNet FID 或同总成本5%比较。

这些结果支持“需要解释累积分布偏差”和“数值振荡可能来自guidance”的研究方向，不构成目前RAE发生同一机制的证据。本文没有新增GPU、模型查询、训练、采样或FID，也没有据理论公式对完整图像重加权、筛选或拒绝。

## 6. 与仓库旧路线的关系

仓库[semigroup一致性推导](SEMIGROUP_CONSISTENT_GUIDANCE_THEORY_ZH.md)早已从端点power tilt推到Jensen修正与Feynman–Kac value；其[RAE轨迹归档](RAEV2_GUIDANCE_EXPLORATION_ARCHIVE_20260905_ZH.md)也记录了权重退化、有限双头的score语义缺口与value失败。此次不是发现一种全新的FKC方法，不重启粒子权重、旧value网络或gap平方梯度。

本次更具体的新增认识是：真实 posterior 的曲率耦合和有限 denoiser 的动力学残差必须分开；反应项的空间变化才产生归一化偏差，且任何干预都会改变被分析的路径。下一项设计若使用这个入口，需要识别实际可获得的目标与误差项，不能仅拿到一个理论名字便启动训练。当前公平成本至少5%的研究目标仍未完成。

独立复核：另一位协作者逐项重建Gaussian反应项、一般密度比PDE、共同OU反例及η²项，全部相符；有限m展开另经两位协作者独立核算。其有限T边界措辞建议已纳入上文。该复核是解析计算，没有数值toy、模型或GPU调用。
