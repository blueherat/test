# RAEv2：真实下一状态误差的凸 proximal 校正

**状态：理论与 CPU toy 已通过；可进入真实、独立 heldout calibration 前提审计。尚无本方法的 RAEv2 FID 结果。** 本方案不新增 guidance 强度或时间窗口；所有系数由真实 coupling 的矩估计得到，完整使用冻结的官方 100 步与官方 IG。它保证有限步误差传播上界改善，在明确的 Gaussian 子类还有实际单步 W₂ 保证；一般非线性网络的实际终点 W₂/FID 仍需实验验证。

后续结构修订及数据复用边界另见 [channel 共享校准审阅](RAEV2_PROXIMAL_SHARED_CALIBRATION_20260906_ZH.md)；它因原 diagonal 早期 heldout 失败提出，不属于本记录的原始实验。

## 1. 设计对象是下一状态的可观测误差

设真实数据 latent 为 X，独立噪声 E，bridge 为 z_t=(1−t)X+tE。对官方 dataward Euler 步 t→s，记 Δ=t−s>0，U=X−E，冻结的 baseline 映射为

\[
T(z)=z+\Delta v(z,t),\qquad y=z_s=z_t+\Delta U,\qquad r=T(z_t)-y.
\]

官方未触发 clamp 时 v=(G−z_t)/t，其中 G 是已含冻结 IG 的 clean prediction，故 r=(Δ/t)(G−X)。当前 shifted Euler100 的最末模型评估时间约 .074766>.05，全部评估点满足此条件。实现仍宜直接计算 r=T(z_t)−z_s；若以后改变 grid，clamp 区域不能套用未 clamp 的简式。

在**真实下一状态 y** 上拟合最小的完整逐坐标凸二次势梯度锥：

\[
\mathcal C=\{h(y)=c+a\odot(y-m):c\in\mathbb R^d,\ a\ge0\},\qquad m=\mathbb E y.
\]

最小化 E‖r−h(y)‖² 的总体解逐坐标为

\[
c=\mathbb E r,\qquad
a_i=\max\!\left\{\frac{\operatorname{Cov}(r_i,y_i)}{\operatorname{Var}(y_i)},0\right\}.
\]

方差为零的坐标设 a_i=0，不引入分母 regularizer。所有期望取真实配对数据分布；无需假设 latent 为 unit Gaussian，也无需估计完整 d×d 矩阵。c、a、m 是估计量，并非“没有参数”；无新增的是待搜索的 guidance 幅度、窗口及其时间调度。

令 ψ(y)=cᵀy+½∑ᵢaᵢ(yᵢ−mᵢ)²，部署

\[
\boxed{\ S=\operatorname{prox}_{\psi}\circ T,\qquad
S(z)=\frac{T(z)-c+a\odot m}{1+a}\ }
\]

除法逐坐标进行。这里没有自由的 proximal 步长：拟合对象 r 已是完整离散步误差，系数 1 由下面的误差恒等式决定。校准需要真实样本上的模型预测；采样无需增加模型 forward。

## 2. 有限步保证与终点分布上界

锥投影的正交关系给出

\[
\mathbb E\|r-h(y)\|^2
=\mathbb E\|r\|^2-\mathbb E\|h(y)\|^2.
\]

它可直接由上述闭式矩公式验证，也是 Moreau 锥分解的具体实例。proximal 非扩张与锥分解是经典结果，见 [Moreau 1965，§4.b、§5.b](https://www.numdam.org/article/BSMF_1965__93__273_0.pdf)；本记录的贡献范围是将这些结构连接到冻结采样器的真实离散误差，不能声称发明了新凸分析定理。

逐样本有更强的准确恒等式：

\[
S(z_t)-y=\frac{r-h(y)}{1+a}.
\]

因此单步真实 coupling defect 满足

\[
\delta_S^2:=\mathbb E\|S(z_t)-y\|^2
\le\delta_T^2-\mathbb E\|h(y)\|^2\le\delta_T^2,
\quad\delta_T^2=\mathbb E\|r\|^2.
\]

同时 Lip(proxψ)=maxᵢ(1+aᵢ)⁻¹≤1，所以 Lip(S)≤Lip(T)。这直接作用于有限 Euler 映射，避免仅由 drift 的 one-sided Lipschitz 改善却无法控制 Euler 二阶项的缺口。

设第 k 步真实边缘为 p_k，采样边缘为 q_k。对任何有有限二阶矩的真实相邻 coupling，通过拼接 coupling、三角不等式与 Lipschitz 性，有

\[
W_2(q_{k+1},p_{k+1})
\le L_k W_2(q_k,p_k)+\delta_k.
\]

以相同 B₀=W₂(q₀,p₀) 开始递推 Bₖ₊₁=LₖBₖ+δₖ，校正前后传播常数与局部 defect 均不增，故 **完整 100 步的这一个 W₂ 上界不增**。不需假定 rollout 恰好位于真实 p_k；p/q 差异由传播项承接。需要存在适用的 Lipschitz 上界，真实网络的全局数值界可能很松；本方法并未从 heldout 样本测出一个有效的全局 L。以模型误差与正规性连接确定性流的分布误差，可参见 [Benton、Deligiannidis、Doucet 的原始工作](https://arxiv.org/abs/2305.16860)。上面的离散递推是这里直接证明的版本。

有类别条件时，需保持真实与采样类别先验相同，按同标签耦合并使用对标签统一的 Lipschitz 上界；可先界定标签配对的平均平方传输误差，再界定无条件 W₂。不能把类别比例不同的数据银行当作同一个 p_k。

## 3. 为什么拟合下一状态：不修正不可约条件噪声

若 baseline 为真正的 Bayes velocity v*=E[U|z_t]，则 T(z_t)=E[y|z_t]。于是

\[
\mathbb E r=0,\qquad
\operatorname{Cov}(r,y)
=-\Delta^2\mathbb E\operatorname{Cov}(U\mid z_t)\preceq0.
\]

每个对角协方差非正，故 a=0,c=0，S=T。这个结论不要求 Gaussian，说明总体校正不会把不可约 coupling 残差当作可去除的正向膨胀。带条件标签时将 z_t 的条件域增广为 (z_t,label)，结论仍成立。有限校准样本不具备此恒等保证，必须检查泛化。

## 4. Gaussian 子类的实际边缘保证

考虑标量联合 Gaussian (z_t,y)，方差为 C_t,C_s>0，交叉协方差 C_ts>0，baseline 为 T(z)=kz+d，k≥0。总体拟合并校正后均值精确等于 E y，而线性增益为

\[
k_{new}=\min\{k,k_{cap}\},\qquad
k_{cap}=C_s/C_{ts}.
\]

真实 Gaussian 最优传输增益为 k*=√(C_s/C_t)。Cauchy–Schwarz 给出

\[
k_{oracle}=C_{ts}/C_t\ \le\ k^*\ \le\ k_{cap}.
\]

所以校正只降低超过 k_cap、已经超出最优增益的部分；其余方差保持原样，均值误差消失。由一维 Gaussian W₂ 公式立即有

\[
W_2^2(S_\#p_t,p_s)\le W_2^2(T_\#p_t,p_s).
\]

这正是“保留条件不确定性间隙”的结构：conditional-mean Euler 常有方差不足，而本方案不会继续收缩它。对独立 Gaussian 坐标和逐坐标非负 affine 增益可求和推广。它保证**从真实 p_t 出发的单步实际边缘距离**，不自动覆盖 q_t≠p_t 的递归采样，也不自动覆盖相关 Gaussian 的非对角映射或一般非线性网络。

## 5. 有限样本与真实模型准入

用固定 calibration bank 估计全部 100 步的 c,a,m，再完全冻结。在独立 heldout 图像上逐样本计算

\[
D_k=\|T_k(z_k)-y_k\|^2-\|S_k(z_k)-y_k\|^2,
\quad
J_k=2h_k(y_k)^\top r_k-\|h_k(y_k)\|^2.
\]

对**任何冻结的有限样本拟合**，a≥0 仍保证非扩张，并且逐样本 D_k≥J_k；但 E J_k≥0 或 E D_k≥0 不由训练集投影自动推出。heldout D_k 是实际 coupling 风险改善的无偏估计，不需要知道条件均值。统计单位是独立图像，不能将像素、latent 坐标或同图的 100 个时间点当作独立样本。

若固定拟合后 D_k 的方差有已知上界 V_k，n 个独立 heldout 样本下，Chebyshev 与 union bound 给出：以至少 1−α 的概率，所有 K=100 步同时满足

\[
\mathbb E D_k\ge\bar D_k-\sqrt{K V_k/(n\alpha)}.
\]

这是一个保守但严格的有限样本例子。仅把未知 V_k 换成样本方差并不能保留这个定理；配对 bootstrap/标准误可作为现实审计证据，需要明确其统计假设。α 是置信水平，不是采样方法的强度参数。逐步总体非负足以继承上述 bound 比较；只看到 100 步未加权总和为正，并不足以声称它成立。

真实前提审计应回答：拟合在新图像是否降低 D_k？改善是否超过估计噪声？全步证书是否一致？若只在校准集有效，应记录有限样本失败，而不是挑选正时间段或回调强度。逐坐标拟合有高维统计风险，真实审计本身是必要的一关。

旧 IG 的 clean MSE 变差仍能改善 FID，已说明有益生成偏差可能被误差校正削弱。本方案增加了非扩张与有限步传播控制，Gaussian 子类还有实际分布定理，因此具备机制准入依据；仍须明确比较冻结官方 IG 的最终生成结果，不能以校准证书取代 FID。

## 6. CPU toy 与明确失败边界

代码：[audit_raev2_proximal_projection_toy.py](../experiments/audit_raev2_proximal_projection_toy.py)。完整结果：[toy_audit.json](data/raev2_proximal_error_projection_20260906/toy_audit.json)。使用数学上的官方 shift=8、100 步 grid；代码中测试场的常数固定，没有搜索 guidance 强度或窗口。

| 固定二维实验 | 原映射终点 W₂² | 校正终点 W₂² | 原 W₂ 上界 | 校正 W₂ 上界 |
|---|---:|---:|---:|---:|
| 非零均值、不同方差 Gaussian bridge，含正向增益与 bias 误差 | 0.917623 | 0.002908 | 3.495206 | 2.751612 |
| 同一 Gaussian 的 Bayes oracle | 0.002697 | 0.002697 | 2.654965 | 2.654965 |
| 平稳独立 Gaussian，非线性 drift z+sin(z) | 11.384376 | 0.071193 | 6.875604 | 0.413325 |
| 旋转/拉伸跨步抵消反例 | 0 | 0.028455 | 12.761413 | 11.925362 |

表中实际距离为 **W₂²**，上界为 **W₂**，不可直接跨列比较。Gaussian 使用准确矩递推，非线性独立坐标利用单调 quantile coupling；2048 与 1024 点 Gaussian 积分的 W₂² 最大差为 1.36×10⁻¹²。还检查了 Gaussian 单步 oracle、最优传输增益、不确定性间隙内部与超过 cap 四个由定理定义的边界，及 Gaussian bridge 全部 100 步的真实输入边缘不增。

最后一行有意保留失败。目标为平稳 N(0,I₂)，令

\[
A=\begin{pmatrix}\log2&-\omega\\\omega&-\log2\end{pmatrix},
\quad\omega=\sqrt{(\log2)^2+4\pi^2},\quad T_k=e^{A\Delta_k}.
\]

它也可写为 Euler 映射，v_k(z)=(T_k−I)z/Δ_k，且所有 T_k 的乘积=eᴬ=I，原终点恰好正确。每步拟合并作 proximal 校正后，单步真实 coupling 风险与 Lipschitz 常数仍全部不增，但破坏跨步抵消，实际终点变差。故完整 bound 机制成立，不意味着一般实际距离单调，更不意味着真实 RAEv2 的 5% FID 目标已经完成。

复现：

```bash
python experiments/audit_raev2_proximal_projection_toy.py --output docs/data/raev2_proximal_error_projection_20260906/toy_audit.json
```
