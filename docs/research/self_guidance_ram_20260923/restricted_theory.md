# RAM 用于受限 internal guidance 的理论审计

日期：2026-09-23。本文针对用户贴文中的直接替换
`v_theta = S + a (S - W_phi)`，独立推导其边界；不修改训练、不启动实验。
全部时间统一为 `t=0` noise、`t=1` data。平方损失使用 `1/2` 系数，若使用原文无 `1/2` 的写法，下面所有训练梯度乘 2。

## 1. 原论文承认的近似，与本文审计范围

[RAM 原论文](https://arxiv.org/html/2605.10759v1) 的第 4 节明确说明：Eq.14 丢弃了 path-cost correction；endpoint 重新加噪与受控 SDE 的 joint law 在中途通常不同；Eq.15 的 velocity plug-in 在中途也是近似。Eq.17 是便宜的 consistency update，不能直接视为任意受限 ODE 终点目标的无偏梯度。第 4.1 节进一步讨论其 fixed point 与精确 KL optimum 的差异。

这里不反驳这些原文结论。审计的是更强的移植说法：把 arbitrary guided field 直接塞进 Eq.17，是否自动保留当前完整反传所优化的目标及其 stationary points。答案是否定的；下面给出可分离的误差项和解析反例。

## 2. 两种状态分布与一种速度缺陷

设当前真实 ODE 为

\[
\dot x_t=v_\theta(x_t,t),\qquad x_0\sim p_Z,
\]

其真实中间分布为 \(\mu_t^\theta\)，终点为 \(q_\theta\)。局部 RAM 数据则由

\[
X\sim q_\theta,\quad Z\sim p_Z\text{ independently},\quad
Y_t=(1-t)Z+tX,\quad D=X-Z
\]

构造。把 \(Y_t\) 的密度记为 \(\rho_t^\theta\)，其 canonical FM velocity 为

\[
\bar v_\theta(y,t)=\mathbb E[D\mid Y_t=y].
\]

需要区分：

1. **状态分布误差**：\(\mu_t^\theta\ne\rho_t^\theta\)。真实 guided rollout 与直线重插值可以走过不同区域。
2. **速度表示缺陷**：\(d_\theta=v_\theta-\bar v_\theta\)。即使两种密度相同，速度也可以相差一个保该密度的无散度场。

第二个缺陷不由第一个缺陷为零而消失。我们之前研究的 density-preserving gauge 正是这一自由度的例子。

## 3. RAM 局部半梯度的精确分解

令 reference 为 \(S\)，\(u=v_\theta-S\)，reward 为固定函数 \(R(X)\)。RAM 的 stop-target 是

\[
\widehat v=S+R(X)(D-v_\theta).
\]

端点采样、重插值、target 均 stop-gradient。定义

\[
m(y,t)=\mathbb E[R(X)\mid Y_t=y],
\]

\[
c(y,t)=\mathbb E[(R-m)(D-\bar v_\theta)\mid Y_t=y].
\]

则

\[
\mathbb E[R(D-v_\theta)\mid Y_t=y]
=c-m d_\theta.
\]

因此实际局部更新向量为

\[
\boxed{
g_{\rm RAM}(R)
=\mathbb E_{t,Y\sim\rho_t^\theta}
\left[(\partial_\theta v_\theta)^\top
\left(u-c+m d_\theta\right)\right].
}
\]

此式不需要 RAM 的近似条件，是对所写 stop-gradient 算法的直接条件期望恒等式。它分出 reference pull、reward 与插值方向的条件协方差，以及由非 canonical velocity 带来的 reward-dependent correction。

参数梯度必须包含局部产品的两部分：

\[
\partial_\phi v=-a\,\partial_\phi W_\phi,
\qquad
\partial_\psi v=(S-W_\phi)\,\partial_\psi a.
\]

但不包含 \(X\)、\(Y\) 或 target 对参数的导数。局部实现可以便宜且正确实现上述半梯度；这与它是否等于原终点目标的梯度是两个问题。

## 4. 常数 reward 平移不自动成为合法 baseline

对任意常数 \(C\)，

\[
\boxed{
g_{\rm RAM}(R+C)-g_{\rm RAM}(R)
=C\,\mathbb E[(\partial_\theta v_\theta)^\top d_\theta].
}
\]

而 \(\mathbb E[R(X)]-\beta\operatorname{KL}(q_\theta\|q_{\rm ref})\)
在 \(R\mapsto R+C\) 下只增加与参数无关的常数，其真梯度完全不变。

所以 reward centering 只有在

\[
\mathbb E[(\partial_\theta v_\theta)^\top d_\theta]=0
\]

时才对当前参数切空间不改变期望更新。\(v_\theta=\bar v_\theta\) 是充分条件，但不必是必要条件。只在某个初始化点满足该条件，不能保证受限投影更新后继续满足。

这个缺陷可以直接测量，不必估计密度或训练一个完整 canonical teacher：

\[
k_\theta=
\mathbb E[(\partial_\theta v_\theta)^\top(v_\theta-D)]
\]

恰好是把当前端点重插值、做普通半平方 FM loss 得到的 stop-sampling 梯度。
由此可比较 reward 平移前后的局部梯度，检查大小、方向及与精确终点梯度的关系。

对于 batch/group 中的自身均值，另有有限样本效应。令
\(G_i=(\partial_\theta v_i)^\top(D_i-v_i)\)，假设组内独立同分布且类别条件固定，则

\[
\mathbb E[(R_i-\bar R)G_i]
=\left(1-\frac1N\right)
\left(\mathbb E[RG]-\mathbb E[R]\mathbb E[G]\right).
\]

即使 \(\mathbb E[G]=0\)，也会把 reward 项相对 reference pull 缩小为 \(1-1/N\)。leave-one-out 均值可以去除这个系数，但不能去除 \(\mathbb E[G]\ne0\) 时的缺陷。再除以随机组内标准差，会进一步改变 reward 权重；不能仅以“降低方差”解释全部效果。

## 5. 解析反例：全部 marginals 完全正确，RAM 仍区别对待 gauge

取二维标准 Gaussian 为 noise 与目标。令

\[
s(t)=(1-t)^2+t^2,\qquad b(t)=\frac{2t-1}{s(t)},
\quad J=\begin{pmatrix}0&-1\\1&0\end{pmatrix}.
\]

reference 为

\[
S(x,t)=b(t)x.
\]

它是 independent Gaussian endpoint/noise 直线插值的精确 Bayes FM 场。其 ODE 解为 \(x_t=\sqrt{s(t)}z\)，终点为原来的标准 Gaussian。

考虑

\[
v_\omega(x,t)=b(t)x+\omega Jx.
\]

其精确流为

\[
x_t=\sqrt{s(t)}\exp(\omega tJ)z.
\]

所以所有 \(\omega\) 具有相同终点，且全部实际中间密度均等于
\(\rho_t=\mathcal N(0,s(t)I)\)。状态分布误差严格为零，
但 \(d_\omega=\omega Jx\ne0\)。

取常数 reward \(R\equiv C\)。对所有 \(\omega\)，任意终点奖励和终点 KL 的目标均不变；RAM 的局部梯度却为

\[
\begin{aligned}
g_\omega(C)
&=(1+C)\omega\int_0^1\mathbb E_{\rho_t}\|Jx\|^2dt\\
&=\boxed{\frac43(1+C)\omega},
\end{aligned}
\]

其中 \(\int_0^1s(t)dt=2/3\)。加常数可使该更新加速、消失（\(C=-1\)）或反向（\(C<-1\)）。

这没有说明 RAM 必须保留无用旋转。它说明 RAM 选择了特定的 velocity representative，这一选择不能自动等价于只关心 endpoint 的目标，也不能对任意 reward 平移保持不变。

## 6. 更强反例：受限族包含 reference，局部更新仍可离开终点全局最优

固定 \(\omega\ne0\)，定义一参数族

\[
v_\theta(x,t)
=\left[b(t)+\theta(\theta-1)\right]x+\omega\theta Jx.
\]

该族在 \(\theta=0\) 包含精确 reference。
令 \(h(\theta)=\theta(\theta-1)\)，则终点为

\[
q_\theta=\mathcal N(0,e^{2h(\theta)}I).
\]

对固定常数 reward，终点 KL 为

\[
\operatorname{KL}(q_\theta\|\mathcal N(0,I))
=e^{2h(\theta)}-1-2h(\theta).
\]

\(\theta=0\) 与 \(\theta=1\) 都是终点目标的全局最优。后一处实际场保留旋转。

在 \(\theta=1\)，\(\partial_\theta v=x+\omega Jx\)，RAM 梯度为

\[
\boxed{g_{\rm RAM}(C;\theta=1)=\frac43(1+C)\omega^2.}
\]

例如 \(C=0\)，一次任意足够小的非零梯度更新都会令 \(h(\theta)\ne0\)，从而使原终点 KL 从零变成正数。这一现象并非 reference 不在模型族中；它来自受限参数把 endpoint-neutral 方向与 endpoint-changing 方向耦合在一起。

也可以严格嵌入 internal-guidance 形式：取 \(a=1\)、\(W_\theta=2S-v_\theta\)，即可得到 \(v_\theta=S+(S-W_\theta)\)。

该反例不反驳带路径能量的原始 stochastic-control 目标：旋转增加路径代价，原目标本来可以惩罚它。它反驳的是“原终点目标、路径目标、RAM 受限局部回归三者自动等价”的说法。

## 7. 受限函数族中的投影与度量差异

当前真实终点 loss \(\mathcal J(\theta)=\mathbb E\ell(x_1)\) 的连续伴随梯度是

\[
\nabla_\theta\mathcal J
=\mathbb E\int_0^1
(\partial_\theta v_\theta(x_t,t))^\top\lambda_t\,dt,
\]

其中 \(-\dot\lambda_t=(\partial_xv_\theta)^\top\lambda_t\)，
\(\lambda_1=\nabla\ell(x_1)\)。当前实际离散 Heun 的精确反传对应离散版本。

RAM 则用重插值分布 \(\rho_t\) 和局部残差 \(u-c+md\)。它同时改变了采样测度与未来反馈。

在无限函数空间中，正时间权重常常不改变一个处处为零的 fixed-point 方程；在受限族中，stationarity 只有

\[
\mathbb E[w(t)(\partial_\theta v)^\top\operatorname{residual}]=0,
\]

权重会改变不同时间、状态间的投影折衷。最简单的两个时间点共用一个参数 \(\theta\)，target 为 \(c_1,c_2\)，加权平方回归最优为
\((w_1c_1+w_2c_2)/(w_1+w_2)\)。更换权重就更换了受限最优。

若以 \(L^2(\rho)\) 投影理解更新，其切空间 Gram 矩阵是

\[
M_\theta=\mathbb E_\rho[(\partial_\theta v)^\top\partial_\theta v].
\]

它不是从场扰动到终点变化的传播算子所诱导的度量。某些路径扰动对终点完全无效，却有正的局部 FM 范数。因此“局部 velocity regression 更便宜”是真实优点；“它保留了当前受限终点最优问题”需要额外证明。

## 8. 可保留的研究价值与可检验用法

RAM 仍非常值得做低成本候选更新。这里的理论指向三种具体用途：

1. **作为新 surrogate 明确评估。** 保留它的局部计算优势，同时承认 reference anchoring、reward normalization 与时间权重会改变受限优化问题。比较最终质量和实际 wall-clock，而不只比较训练步数。
2. **测量缺陷而非凭感觉调 reward。** 同一批 endpoint 上估计 \(k_\theta\)、RAM 与精确 endpoint 梯度的夹角、两者残差。再改变 reward 常数和归一化方式，区分数值尺度效应与目标变化。
3. **作为精确梯度的控制变量。** 用 RAM 得到便宜向量 \(\widetilde g\)，以概率 \(p\) 算同参数、同 endpoint batch、同固定 D 下的精确 \(g\)，构造

\[
\widehat g=\widetilde g+\frac Ip(g-\widetilde g),
\quad I\sim\mathrm{Bernoulli}(p).
\]

条件于当前 batch 和局部更新，\(\mathbb E_I\widehat g=g\)。这条用途不要求 RAM 自身是原目标的梯度，也不要求 canonical self-consistency。真正需要检验的是 \(\|g-\widetilde g\|\) 是否足够小，以及节省的计算能否抵消校正方差。

其中第三条直接连接已有完整反传实现：便宜局部回归提供预测，精确反传提供校正。它比仅依据 RAM 论文的无限函数空间固定点来宣称直接替代，更适合当前特殊的 weak-head-plus-scale 受限模型。
