# 双向 inverse-copy 配对：一阶偏好消去与二阶 Bayes 修复

日期：2026-09-13。仅理论审查；未改数据脚本、未新增 toy 或 GPU 实验。

**裁决：值得与单侧方案并列一个小变体。** 等权正反变换使训练目标严格不偏好 S/W 的角色，消去一般近恒等变换的一阶逆偏移；留下的二阶信号具有各向异性条件均值修复形式，但不等于纯 score、真实密度恢复或投影。固定 real/pair 总权重、τ 分布、架构和训练更新数，才是可解释比较。

## 1. 精确 Bayes 回归式

固定类别，真实数据有正的光滑密度 p。令 d 是光滑向量场，C+=exp(hd)、C−=exp(−hd)=(C+)⁻¹，h 为变换幅度。以概率 π 输入真实 X，另外以 (1−π)/2 分别输入 C±(X)，所有监督目标均为 X。这里的符号只用于离线配对，不作为 P 的输入。

记推前密度

\[
q_+(y)=p(C_-(y))|\det DC_-(y)|,\qquad
q_-(y)=p(C_+(y))|\det DC_+(y)|.
\]

忽略额外幂等正则，无限容量平方风险最优解为

\[
P^*(y)-y=
\frac{\frac{1-\pi}{2}
\{q_+(y)[C_-(y)-y]+q_-(y)[C_+(y)-y]\}}
{\pi p(y)+\frac{1-\pi}{2}[q_+(y)+q_-(y)]}.
\]

不能直接把两条逆位移等权平均：观察到同一个 y 后，两种来源的后验权重是 q±，需要包含体积 Jacobian 和真实密度。

## 2. 对称自治流的二阶项

设 Jd=Dd，b=div(pd)/p=div d+d·∇log p，各量在 y 处取值。局部展开为

\[
C_\pm(y)=y\pm hd+\frac{h^2}{2}J_d d+O(h^3),\qquad
q_\pm=p\mp h\,\mathrm{div}(pd)+O(h^2).
\]

代入得到

\[
\boxed{
P^*(y)-y
=(1-\pi)h^2\left[
(\mathrm{div}d+d^\top\nabla\log p)d+\frac12J_d d
\right]+O(h^4).
}
\]

O(h⁴) 需要足够高阶光滑性；因为该精确 exp(±hd) 混合关于 h 是偶函数，三阶项消失。固定 h 下则应使用上一节精确表达。

若 A=ddᵀ，可等价写为

\[
(1-\pi)h^2
\left[A\nabla\log p+\mathrm{div}A-\frac12J_d d\right].
\]

其中 div A 按矩阵行散度定义，div A=Jd·d+d div d。只有常向量 d 等特殊情形才只剩 A∇log p；一般不能直接命名为 score。

对应正向人工扰动具有二阶平均位移 h²Jd·d/2 和协方差 h²ddᵀ。修复器是在同时补偿状态相关扰动、体积变化和后验来源概率，不是仅沿“模型差方向的负号”移动。

## 3. 非自治、非交换的真实 copy 配对

现在 h=1−τ，终点固定为物理时间 1，定义

\[
C_h=\Phi_W^{1\leftarrow1-h}\circ\Phi_S^{1-h\leftarrow1}.
\]

所有下式速度和导数都在 (x,1) 取值。令 d=W−S，则

\[
C_h(x)=x+hd+h^2e+O(h^3),
\]

\[
e=\frac12[
-\partial_t d+J_S S+J_W W-2J_W S].
\]

或者

\[
e=\frac12J_d d+
\frac12(J_S W-J_W S-\partial_t d).
\]

这包含非自治时间导数和非交换项，并不等于 exp(hd)。但其真逆总满足

\[
C_h^{-1}(x)=x-hd+h^2(J_d d-e)+O(h^3).
\]

因此，等权混合 C_h 与 C_h⁻¹ 时，e 在领先 Bayes 修复项中抵消，仍得到

\[
P^*(y)-y=(1-\pi)h^2
[(\mathrm{div}d+d^\top\nabla\log p)d+\tfrac12J_d d]
+O(h^3).
\]

一般只可写 O(h³)，因为非自治 C_h 家族不必满足 C_-h=C_h⁻¹。多个 τ 的小区间混合对领先项按 h² 加权；当前 h=.25/.5 并不自动处于该渐近范围。

比渐近展开更强的一点是：对每个固定 τ，交换 S/W 会把 Cτ 与 Cτ⁻¹ 对调。若符号等权且数据/损失不向学生暴露符号，整个训练目标在此交换下严格不变。这排除了标签意义上的单侧参考偏好，但没有排除共同模型偏差。

## 4. Gaussian FM 端点会使阶数进一步退化

对 independent Gaussian FM，精确速度在内部满足

\[
v(z,t)=z/t+(1-t)\nabla\log p_t(z)/t.
\]

在正则端点极限下 v(z,1)=z。如果 S/W 都严格满足这一端点恒等，则 d(·,1)=0，JS=JW=I，前述一阶项消失：

\[
C_h=I-\frac{h^2}{2}\partial_t(W-S)|_{t=1}+O(h^3).
\]

此时实际变换幅度是 O(h²)，对称 Bayes 修复领先信号可降到 O(h⁴)。当前神经 checkpoint 未硬编码 v(z,1)=z，不可默认其端点差为零；也不能把一般 O(h²) 修复公式当成当前两个有限区间的实测主导项。

## 5. 明确反例：保持密度的双向旋转仍导致收缩

令 p=N(0,I₂)，d(x)=Jx，J 为二维 90° 旋转生成矩阵。C± 是角度 ±h 的旋转，二者均严格保持 p，所以 q+=q−=p。Bayes 解却是

\[
P^*(y)=[\pi+(1-\pi)\cos h]y.
\]

它会收缩方差，尽管输入分布本来已经完全正确。二阶公式中 div(pd)=0，剩下 Jd·d/2=−y/2，恰好复现这一收缩。

因此双向化不证明 density restoration。它去掉的是符号方向的一阶偏移；来源不确定时，平方误差条件均值仍可能平均合法变化、损失细节和多样性。

同样，P=I+h²f 时 P²−P=h²f+O(h⁴)；过强的幂等正则可能直接压小刚出现的修复信号。双向与单侧应保持幂等项设置相同，首轮可先比较仅 real＋pair 的主体目标。

## 6. 值得比较的范围

变体仅改变配对分布：

\[
L_{\rm pair}^{\rm sym}
=\mathbb E_{x,\tau}\frac12[
\|P(C_\tau x)-x\|^2+
\|P(C_\tau^{-1}x)-x\|^2].
\]

不能把新增两对直接叠加到原损失上而使 pair 总权重翻倍，否则连 π 都发生了变化。已确认 repair 对全部配对取平均、real/pair 总权重不变；两版均使用相同 P、400 fit/100 holdout 与 1500 次训练更新，而非简单匹配 epoch。P 不接收 sign/τ。

反向配对可共享 W 从 1 逆到 .5 的轨迹，S 分别从 .75/.5 回 1，额外 160 次单分支查询/真实图；推理成本不变。以负步长 Heun 求 ODE 逆只是近似真逆，因此正反符号误差还需按既有数值精度控制，不能把离散不对称误认成保留下来的统计信号。

此理论针对光滑总体风险。有限 500 图上的高容量插值可以记住各配对，未必学习上述条件均值；因此 holdout 和独立生成评估仍是核心。若双向版比单侧更保留真实图、同时改善实际生成，才支持单侧 bias 是有害成分；若只减小修复幅度或模糊图像，不构成质量机制证据。

## 7. 与 denoising autoencoder 文献相邻，但没有平稳分布保证

Alain–Bengio 的 JMLR 2014 工作证明，在相应正则化和小噪声条件下，重建残差反映数据 score；本文的状态相关、秩一变换有额外散度和曲率项，不能直接套用各向同性 Gaussian 噪声的公式。[原文](https://www.jmlr.org/papers/v15/alain14a.html)

Generalized DAE 的 Theorem 1 讨论交替使用 corruption kernel 与学习到的完整条件分布进行采样；其结论需要条件估计一致性和 Markov 链遍历性等条件。本文部署的是确定性条件均值近似 P，既未从完整后验采样，也未运行该交替链，因此一次 P 或反复 P 都不继承数据分布平稳性保证。[Generalized DAE 原文 §2.3](https://proceedings.neurips.cc/paper/5023-generalized-denoising-auto-encoders-as-generative-models.pdf)

本变体仍是 [反演配对修复器理由](inverse_copy_refiner_rationale.md) 的一个有限消融，理论新颖性与图像质量收益都尚未确立。
