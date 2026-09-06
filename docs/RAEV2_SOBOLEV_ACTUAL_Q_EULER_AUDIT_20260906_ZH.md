# Actual-q 矩反馈与有限 Euler：独立理论审计

**裁决：实际 q 反馈在有限、非闭合特征下确实可以不同于旧真实 bridge 拟合；有限 Euler 的最小修正也可由下一步矩约束决定，无需自由时间 gain。** 这两点已用可解反例和 CPU toy 验证。但它们尚不支持 RAE 训练：当前缺少“哪个可达的实际分布缺口应当纠正”的证据，有限步非线性可行性与小样本反馈也未解决。

本笔记独立于[原阅读文档](RAEV2_GUIDANCE_READING_SOBOLEV_20260906_ZH.md)，没有修改原文档。以下是独立推导；不是 [Sobolev Descent](https://proceedings.mlr.press/v89/mroueh19a/mroueh19a.pdf) 已证明的 diffusion 算法。本次只运行有界 CPU 数值积分，不使用 RAE、decoder、GPU 或图像指标。

## 何时与旧解相同，何时不同

仍用 dataward 时间 s、真实 bridge 速度标签 U、baseline b，取 `J=[∇Φ₁,…,∇Φ_m]∈R^{d×m}`。记

\[
\begin{aligned}
D_q&=E_q[J^TJ],\quad \delta=E_p\Phi-E_q\Phi,\\
r_q&=E_{\rm real}[\partial_s\Phi+J^TU]
       -E_q[\partial_s\Phi+J^Tb],\\
u_q&=JD_q^+r_q.
\end{aligned}
\]

若 r 在值域中且连续流适定，`δdot=r_q−D_qD_q⁺r_q=0`。旧解为 `a_p=D_p⁺r_p`、`r_p=E_real[Jᵀ(U−b)]`。两者在 `q=p` 时相同，因此同一初始分布下，初始向量场相同。

有两种足够强的全程等价情形：旧解已满足完整真实路径连续性方程且解唯一，故始终 `q=p`；或有限矩动力学真正闭合。后者的一个充分条件是 `∂sΦ+Jᵀb` 及 `JᵀJ` 的每个分量都在 `span{1,Φ₁,…,Φ_m}` 中。此时只要 `E_qΦ=E_pΦ`，就有 `D_q=D_p`、`r_q=r_p`。线性/Gaussian 例子很容易落入这种闭合情形，不能用它们证明反馈带来了新信息。

一般非线性 b 不满足闭合。有限矩匹配允许 `q≠p`，从而两个测度下的 Jacobian Gram 和平均矩速度不同。旧解在实际 q 上遗漏的矩导数为

\[
r_q-D_qa_p=(r_p-D_pa_p)+(r_q-r_p)-(D_q-D_p)a_p.
\]

若旧有限解精确，第一项为零。**新增反馈信息可被直接测为后两项的净和**；只测 `q≠p`、某个 AUC 或一个非零 Gram 差，均不足以说明实际净缺口有量级。

## 最小非线性反例：同一真实 Gaussian bridge、两个矩

取独立 `E,X~N(0,1)`，真实 bridge 为

\[
Z_s^p=(1-s)E+sX,\quad \sigma_s^2=(1-s)^2+s^2,\quad
v_s^\star(z)=\frac{\sigma'_s}{\sigma_s}z.
\]

这是与 RAE 时间方向相同的独立 Gaussian 线性 bridge。使用显含时间的两个特征及一个全局光滑 baseline：

\[
x=z/\sigma_s,\quad \Phi_s(z)=(x,x^2/2),\quad
b_s(z)=\frac{\sigma'_s}{\sigma_s}z+\sigma_s\cos x.
\]

在标准化坐标中目标恒为 `N(0,1)`，baseline 额外 drift 是 `cos x`。令 `c=exp(−1/2)`。旧真实 bridge 解在物理坐标为 `u_old=−σ_s c`，因此旧实际连续流满足 `dx/ds=cos x−c`。

若实际-q 反馈保持标准化均值 0、方差 1，其同一两势空间中的修正为

\[
u_q(z)=\sigma_s(a_0+a_1x),\quad
a_0=-E_q\cos x,\qquad a_1=-E_q[x\cos x].
\]

这正是 `JD_q⁺r_q`，不是额外设计的 controller。两者初始都有 `(a₀,a₁)=(−c,0)`。但 Gaussian 积分给出

\[
\beta=\frac{1-e^{-2}}2=0.4323323584,\qquad
a'_{1,q}(0)=-\beta,
\]

而旧 `a₁` 恒为 0。相应地

\[
\operatorname{Var}_{q^{old}_s}(x)=1+\beta s^2+O(s^3),\quad
\operatorname{Var}_{q^{feedback}_s}(x)=1.
\]

两条实际分布都立即偏离目标 Gaussian，因为

\[
\left.\frac{d}{ds}E_q[x^3]\right|_{s=0}=-3c=-1.819591979\ne0.
\]

因此这里有完整的因果顺序：有限残差场先改变未受约束的形状；非闭合的 `x cos x` 平均值随后改变；实际反馈与旧拟合分开。两个矩精确保持并不等于目标分布正确。这个反例也没有使用不可控的多项式爆炸场或非配对的假 bridge。

## 有限 Euler 的自然约束，与连续公式的区别

真实一步为 `Y=z+h b_s(z)`，h 来自既定 solver 网格。一个直接对应的有限问题是

\[
\min_d\tfrac12E_q\|d(z)\|^2
\quad\text{s.t.}\quad E_q\Phi_{s+h}(Y+d(z))=E_{p_{s+h}}\Phi_{s+h}.
\]

这里 `d=h u`；固定 h 下最小化位移能量与速度能量的解相同。目标是下一真实边缘的指定矩，修正量由约束决定，没有另设反馈 gain。但“硬匹配全部所选矩”本身是需要质量依据的设计选择，不是 Sobolev 定理强迫采用的终点质量目标。

一般 Φ 非线性时，这是非凸约束问题。最优性条件形如 `d(z)=J_{s+h}(Y+d(z))η`，Jacobian 在修正后位置计算，不能用当前 `D_q` 一次线性求解冒充精确有限步解。它也不保证 d 是修正前 z 的梯度场。若必须保留 `u=J_s(z)a`，应改为在 a 上最小化 `½aᵀD_qa` 并施加同一非线性矩约束；此时可能根本不可行。

局部展开揭示一个关键限制：

\[
hD_qa\simeq\delta_s+h r_s,\qquad a\simeq D_q^+(r_s+\delta_s/h).
\]

所以一步硬修复自然带来 `δ/h`，并不是凭空消除了反馈强度问题。若当前已有固定误差，h 越小速度修正越大；若每步独立估计目标矩，`O(N^{-1/2})` 的统计抖动也可能被 `1/h` 放大。固定、平滑的真实 bank 与持续推进的同一粒子 cohort 能避免每步独立重抽造成的部分抖动，但其精确匹配仍只是经验矩，不是总体证书。

## toy 中同一势空间的精确有限步解

当前标准化粒子满足均值 0、方差 1。记

\[
k=\sigma'_s/\sigma_s,\quad \tau=\sigma_{s+h}/\sigma_s,\quad
m_b=E_q\cos x,\quad \kappa=E_q[x\cos x],\quad
v_b=\operatorname{Var}_q(\cos x).
\]

限制修正仍为 `u=σ_s(a₀+a₁x)`，下一标准化状态为

\[
x_{next}=\tau^{-1}\{A x+h(\cos x-m_b)\},\quad A=1+h(k+a_1).
\]

均值约束固定 `a₀=−m_b`；方差约束给出

\[
A=-h\kappa\pm\sqrt{\tau^2-h^2(v_b-\kappa^2)},\qquad
a_1=(A-1)/h-k.
\]

当判别式非负且 `1+h(k+κ)>0` 时，正根使 `σ_s²(a₀²+a₁²)` 最小；这由二根到零系数的距离比较决定，不由图像指标选分支。若条件变化应直接比较可行根的能量；判别式负时，这个有限势空间无解，不能 clip 后继续宣称精确匹配。`h→0` 时有 `a₁→−κ`，回到当前-q 连续解。

作为参照，若允许任意 d，一维均值/方差约束有全局最小解

\[
Z_{next}=m_*+\frac{\sigma_*}{\sigma_Y}(Y-EY),\quad \sigma_Y>0.
\]

证明来自 `Cov(Z_next,Y)≤σ_*σ_Y`；最小 `E d²=(m_*−EY)²+(σ_*−σ_Y)²`。这一解对 Euler proposal 缩放，通常含 b 的非线性部分，不能与上面的“同一有限势空间”解混为一谈。将均值/方差投影迁移到 RAE 还会与[旧 proximal 校准](RAEV2_PROXIMAL_ERROR_PROJECTION_20260906_ZH.md)高度重合，不能仅靠新理论名称重启旧方向。

## 已运行 CPU toy：实际物理坐标 Euler

用 Gauss–Hermite 节点推进整个分布，而非随机采样；这是总体积分的数值近似，不是 exact population 的替代证明。所有方法在物理 z 中执行 Euler，然后记录 `x=z/σ_s`。不能先归一化连续方程再把归一化 Euler 当作同一个数值方法。

均匀网格、终点 `s=1`，192 个节点：

| 步数 | 旧 Ep 解：方差 | 当前-q 瞬时解：方差 | 同势空间有限解：方差 |
|---:|---:|---:|---:|
| 32 | 1.31548608 | 0.92743172 | 1（舍入内） |
| 128 | 1.39789971 | 0.98133748 | 1（舍入内） |
| 512 | 1.41938725 | 0.99530091 | 1（舍入内） |

旧解的偏差没有随网格加密消失；当前-q 瞬时解的离散方差误差约一阶减小；有限解每一步满足这两个矩。256 节点复核 512 步时，两种节点数的一至四阶矩最大差：旧解 `7.05e−9`，其余方法均 `<1.75e−10`；有限同势解的逐步矩误差 `<4.22e−15`。

另使用仓库[网格公式](../experiments/run_raev2_distribution_auc.py) `t=8v/(1+7v)`、`s=1−t`，100 步，最大 `h=0.07476635514`，256 节点：

| 方法 | 终点均值 | 方差 | 三阶中心矩（目标为 0） |
|---|---:|---:|---:|
| 旧 Ep 瞬时解 + Euler | −0.00656611 | 1.31332927 | −2.19160257 |
| 当前-q 瞬时解 + Euler | 舍入内 0 | 0.93248881 | −1.35783263 |
| 同势空间有限步最小解 | 舍入内 0 | 1 | −1.49860567 |
| 不限制势空间的最小位移 | 舍入内 0 | 1 | −1.49247647 |

有限同势解逐步均值/方差误差 `<8.89e−16`，最小判别式 `0.9833960174`，正根最小化条件的最小余量 `0.9930316209`。**矩全程精确而终点三阶矩显著错误，是本 toy 最重要的失败边界。** 它证明机制可实现，不证明这组特征值得用于生成质量。

初次 192 节点、24 条轨迹的积分窗口 wall `0.18149 s`、CPU `2.97270 s`；shift8 复核窗口 wall `0.02310 s`、CPU `1.46935 s`，均不含 import。CPU 计时包含数值库线程累计时间；另一次节点数复核未单独计时。这些不是 RAE 成本预测。

## 可重跑的最小规格

下列代码重跑上面的 shift8 表；依赖 NumPy，所有系数由矩约束决定。`finite_span` 的两个标量矩约束与输入势空间始终相同。

```python
import math
import numpy as np
from numpy.polynomial.hermite import hermgauss

n, w = hermgauss(256)
x0, w = np.sqrt(2) * n, w / np.sqrt(np.pi)
v = np.linspace(1., 0., 101)
grid = 1 - 8*v/(1 + 7*v)

for mode in ('old', 'feedback', 'finite_span', 'finite_free'):
    z = x0.copy()
    for s, s1 in zip(grid[:-1], grid[1:]):
        h = s1 - s
        sig = math.sqrt((1-s)**2 + s*s)
        sig1 = math.sqrt((1-s1)**2 + s1*s1)
        k = (2*s-1)/sig**2
        x, b = z/sig, np.cos(z/sig)
        mb, xb = w@b, w@(x*b)
        Y = z + h*(k*z + sig*b)
        if mode == 'old':
            z = Y - h*sig*math.exp(-.5)
        elif mode == 'feedback':
            D = np.array([[1., w@x], [w@x, w@(x*x)]])
            a = np.linalg.solve(D, -np.array([mb, xb]))
            z = Y + h*sig*(a[0] + a[1]*x)
        elif mode == 'finite_span':
            assert abs(w@x) < 1e-12 and abs(w@(x*x)-1) < 1e-12
            vb = w@((b-mb)**2)
            disc = (sig1/sig)**2 - h*h*(vb-xb*xb)
            assert disc > 0 and 1+h*(k+xb) > 0
            a1 = (-h*xb + math.sqrt(disc)-1)/h - k
            z = Y + h*sig*(-mb + a1*x)
        else:
            m = w@Y
            z = sig1*(Y-m)/math.sqrt(w@((Y-m)**2))
    m = w@z
    print(mode, m, w@((z-m)**2), w@((z-m)**3))
```

## 下一步值得测的事实，以及停止条件

第一个新事实应是**一个已具备生成缺陷含义的固定 Φ，其实际反馈净缺口 `(r_q−r_p)−(D_q−D_p)a_p` 在独立实际轨迹上有可靠量级**。这项量能区分“旧空间本身没拟好”和“因实际 q 已偏离而需要新反馈”；零或极小结果便否定该特征族的新增反馈价值。不能从只含真实 bridge 的旧 `R²/RC/c²` bank 重建这个量。

随后才值得测固定真实 Euler proposal 的矩缺口 `E_p,nextΦ−E_qΦ(Y)`、有限修正可行性与所需能量，检查拟合数据与独立当前 q 的符号/量级是否一致。若缺口主要是有限样本抖动、Jacobian 近零、修复量过大或实际状态梯度成本不可接受，就应停止该结构，而非引入窗口、削峰系数或改用最有利时刻。

推到 RAE 仍有三个具体障碍：固定特征的误差是否损害终点质量；每类样本稀疏时如何可靠估计当前条件矩；非线性有限约束是否能以可接受成本求解。若 Φ 含 backbone/decoder，Gram 和有限步 Jacobian 需完整输入导数。粒子 cohort 的统计与同步、目标 bank、求解及全部模型调用都进入公平成本；固定 cohort 的精确经验矩不等于独立生成分布的总体矩。

因此本轮实际完成的是有解释力的 CPU 反例与有限步求解审计。继续训练 RAE 尚无依据；应先获得上述实际反馈净缺口及误差性质证据，最终质量标准仍是公平成本下至少 5% FID 改善，而非矩误差归零。
