# 从 RAM 缺陷到概率流的保持：normalized ITM 的可证扩展

2026-09-23。本笔记只研究理论和 CPU Gaussian 例子，不修改训练或启动 GPU。承接 [RAM 审计](../../classifier_guidance/RAM_INTERNAL_GUIDANCE_REVIEW_20260923_ZH.md)。

**主要结果不是再提出一种 reward-weighted regression，而是发现：如果旧场已经实现所用桥的边缘密度，那么 normalized implicit Tilt Matching 会准确搬运旧场的保分布概率流，无须先抹掉它。此前看似 canonical defect 的部分，在另一种锚点和目标下可以成为必须保留的结构。**

这不证明当前 SiT/JiT 满足桥边缘一致性，也不把新 surrogate 等同当前离散 Heun 的终点梯度。

## 1. 文献边界：应先承认已有理论

[Tilt Matching，2025-12](https://arxiv.org/html/2512.21829v1) 已提出：沿指数倾斜参数演化 canonical velocity 的条件协方差 ODE，以及把指数权的所有阶修正写成 implicit regression 的 ITM。其 Eq. 20 是下文目标的直接算法来源；weighted FM、条件协方差、homotopy 和 implicit fixed point 都不是本项目的新发明。

[Newton Matching，2026-09](https://arxiv.org/pdf/2609.05727) 在 §8.7（PDF 第 85–86 页）明确讨论 noncanonical anchor，以及用准确的条件 reward baseline 消除其影响；近似 baseline 的误差正好等于 baseline error 乘 canonicality defect。它的 §10.3 讨论 RAM 的 critical-point inconsistency；附录 A 建立 density 到 canonical field 的映射和 canonical projection。

因此，“条件中心化解决 canonical defect”及“先在分布空间更新、再回归速度场”已经有非常直接的先例。本轮值得保留的推导是下面的**概率流保持解释、常数奖励下受限小头的精确静止性质，以及对不一致桥的残差传递公式**。在已核验的 Tilt Matching §3 和 Newton Matching §8.7、§10.1 中，没有找到以下 inverse-density transport 的明确表述；这只是本次核查范围内未发现，不能据此宣称首次提出。

## 2. 设定：同一密度路径可以有不同速度场

采用仓库方向，噪声 t=0，终点 t=1。旧终点 X∼q，独立 Z∼N(0,I)，定义

\[
Y_t=(1-t)Z+tX,\qquad U=X-Z,
\qquad \rho_t=\operatorname{Law}(Y_t),
\qquad m_t(y)=\mathbb E[U\mid Y_t=y].
\]

m 实现桥密度 ρ 的连续性方程：

\[
\partial_t\rho+\nabla\cdot(\rho m)=0.
\]

设旧网络场为 v=m+d。先作一个**实质性条件**：v 的实际 ODE 边缘也恰为 ρ。于是

\[
\nabla\cdot(\rho d)=0.
\tag{1}
\]

d 可以非零，例如 Gaussian 上的旋转。保持密度的是概率流 ρd，而不是 d 在普通欧氏意义下必为 divergence-free。Eq. (1) 比 v=m 弱，允许完整的保密度速度自由度。

给定固定终点 reward R，目标终点是一次 KL-proximal 倾斜

\[
q^+(x)=q(x)\,\bar w(x),\qquad
\bar w(x)=\frac{e^{\eta R(x)}}{Z_\eta},\quad
Z_\eta=\mathbb E_qe^{\eta R(X)}.
\]

记

\[
h_t(y)=\mathbb E[\bar w(X)\mid Y_t=y],\qquad
\rho_t^+(y)=\rho_t(y)h_t(y),
\]

\[
m_t^+(y)=\frac{\mathbb E[\bar w(X)U\mid Y_t=y]}{h_t(y)}.
\]

默认密度正、边界通量衰减、条件矩存在，场足够光滑并且相应 ODE 和连续性方程适定。若 h 太小导致速度失控，下面的代数恒等式本身不足以保证全局采样器适定。

## 3. 一个简单但有用的命题：运输概率流，无须 canonicalization

定义

\[
\boxed{v_t^+=m_t^+ + d_t/h_t.}
\tag{2}
\]

那么

\[
\rho_t^+(v_t^+-m_t^+)=\rho_t d_t.
\tag{3}
\]

旧的保密度概率流完全保留。因此

\[
\partial_t\rho_t^++\nabla\cdot(\rho_t^+v_t^+)
=\nabla\cdot(\rho_t d_t)=0.
\]

在上述适定条件下，v+ 从相同 Gaussian prior 出发，准确生成 q+。不必要求旧场 canonical。

**为什么不能简单取 m+ + d？** 因为

\[
\nabla\cdot(\rho_t^+d_t)
=\rho_t d_t\cdot\nabla h_t,
\]

一般非零。保密度自由度随密度改变而改变；保持旧速度旋转与保持旧概率流旋转不是同一件事。

Eq. (2) 可视为两个连续性方程解空间之间的一个仿射映射：对任意平滑正密度路径 ρ、ρ+，若 m、m+ 是各自任一特解，则 v↦m+ +(ρ/ρ+)(v−m) 把实现 ρ 的场映到实现 ρ+ 的场。这里没有依赖 Gaussian 以外的特殊性质；Gaussian 独立桥只是让算法中的条件期望可采样。

但这个一般恒等式不是任意 actual rollout 的现成实现：仍需两条密度路径和相应特解。直接把实际路径代入，不会自动获得可计算的密度比和新路径特解。

## 4. normalized ITM 恰好隐式实现这个映射

固定旧场 v，训练新场 f，使用 stopped target

\[
\boxed{
L(f)=\frac12\mathbb E\left\|
f(Y_t)-\operatorname{sg}\{v(Y_t)+(\bar w(X)-1)[U-f(Y_t)]\}
\right\|^2.
}
\tag{4}
\]

所有旧采样、权重和 target 都停止梯度。人口层面的条件正常方程为

\[
f-v-\mathbb E[(\bar w-1)(U-f)\mid Y_t]=0.
\]

整理得

\[
h f=v+\mathbb E[\bar wU\mid Y_t]-m
=h m^++d,
\]

故解正好是 Eq. (2)。它在实现上不需要单独估计 m、m+、h 或 d；这些量仅用于解释人口极限。

这就是已有 ITM 的指数权目标加全局正规化，锚点为旧 guided 场，而非强模型 S。不能把此处“旧场”的锚点替换为任意 reference 而保留上述结论。

未正规化的原始 ITM 也有密度层面的扩展：若 w=e^{ηR}、H=E[w|Y]，则解是 m+ +d/H，满足 ρ+(d/H)=ρd/Zη。时间无关的全局缩放仍保留 divergence-free 性，因此也能实现相同目标密度。但奖励常数会改变所选速度代表；正规化形式则消掉了这层不必要变化。

## 5. 对受限 weak head 的一个实在好性质

若 R 是常数，则每个样本的正规化权重都是 1，Eq. (4) 退化为回归旧场 v。

因此只要新模型初始化复制旧参数，在**任意受限函数族内**，每个样本的半梯度都是零：无需先拟合 canonical m，也无需精确估计条件均值。

这直接避免上一轮 Gaussian counterexample 中“已经在终点最优，仍因消旋转而损伤终点”的那一种更新。它不是所有有限容量问题的解答，但确实修复了一个明确失败模式。

奖励增加常数 C 时，\(\bar w\) 逐点不变，整个经验更新也不变。用同 batch 的 exp-reward 均值作正规化，同样具有这一**代数平移不变性**；不过随机 self-normalization 会引入统计偏差和样本耦合，不能当作人口 Z 已知。

对当前 class-conditional 生成，以上分布和期望应理解为给定类别 c，尤其正规化常数是 Zη,c=E[e^{ηR}|c]。这样任何只依赖类别的 reward 常数也不会更新该条件生成场。跨类别使用单一 batch 正规化会给各类保密度概率流施加不同常数缩放；虽然理想终点的类内倾斜仍可相同，但失去“类内常数 reward 零更新”的保证。实际 batch 类别稀疏时，Zc 的估计本身需要单独设计。

对固定旧 anchor，Eq. (4) 的期望半梯度还有简单解释。设 f=fθ、Jθ=∂θf，则

\[
g(\theta)=\mathbb E_{t,Y\sim\rho_t}
[J_\theta^\top h_t(f_\theta-v^+)]
=\mathbb E_{t,Y\sim\rho_t^+}
[J_\theta^\top(f_\theta-v^+)].
\tag{5}
\]

即它正好优化在**目标桥密度**上的速度拟合误差。这里“半梯度”仅指实现中停止 target 导数；该期望向量本身是一个固定平方拟合目标的准确梯度。

当函数族受限，人口驻点只保证 target-density-weighted 的切空间投影，不能保证终点 KL 或现有 Heun reward 的参数驻点。若 ideal v+ 实现 ρ+，且拟合场 f 在 x 上的 Lipschitz 常数为 L(t)，标准同步耦合/Gronwall 给出

\[
W_2(q_f,q^+)
\le\int_0^1 e^{\int_t^1L(s)ds}
\left(\mathbb E_{\rho_t^+}\|f_t-v_t^+\|^2\right)^{1/2}dt.
\tag{6}
\]

这说明拟合误差确有终点意义；但大 Lipschitz 常数、tail 中很小的 h、或小头无法表达 v+ 都会削弱这个界。保持概率流不保证是受限模型族里最好拟合的代表，这仍是一个可研究的选择问题。

## 6. 小步极限重新解释“defect 项”

令 A=R−EqR。小 η 时，\(\bar w=1+\eta A+O(\eta^2)\)。Eq. (2) 的一阶变化是

\[
\boxed{
\partial_\eta v^+\big|_0
=\operatorname{Cov}(R,U\mid Y_t)
-[\mathbb E(R\mid Y_t)-\mathbb ER]d_t.
}
\tag{7}
\]

第二项不是在这个目标下需要删除的误差：它恰好运输旧概率流。把 reward 进行准确的 state-conditional centering，会只剩 Cov 项，相当于保持 d 的速度值，通常破坏新密度的连续性方程。

这不推翻此前 RAM 分解。两者锚点和目标不同：RAM 以 S 为固定 reference、使用线性 reward 并讨论其强场正则目标；此处以旧 v 为 proximal anchor，讨论正规化指数倾斜和整个密度路径。不能看到相同的乘积项，就在两种目标下赋予同一种解释。

若沿 tilt 参数 λ 持续演化，准确的族是

\[
v_{\lambda,t}=m_{\lambda,t}
+\frac{\rho_{0,t}}{\rho_{\lambda,t}}d_{0,t},
\]

它与 canonical Tilt Matching 有相同终点倾斜，但保留一个运输后的概率流代表。全局正规化后的 centered explicit update 是这个族的一阶 Euler 近似。

## 7. 真正不能跳过的条件：actual rollout 与桥的密度路径

当前 v 由自己的 ODE 产生 q，并不代表它沿路实现 ρ=Law((1−t)Z+tX)。定义桥连续性残差

\[
e_t=\partial_t\rho_t+\nabla\cdot(\rho_tv_t)
=\nabla\cdot(\rho_td_t).
\]

即使 e≠0，normalized ITM 的人口解仍是 Eq. (2)，但现在

\[
\boxed{
\partial_t\rho_t^++\nabla\cdot(\rho_t^+v_t^+)=e_t.
}
\tag{8}
\]

因此它**准确保持这个连续性残差，既不自动校正也不必直接放大它**。未正规化形式的残差为 e/Zη。

旧场与旧桥具有相同终点，只说明这个残差对旧动力学的累积结果在终点可能抵消；换成新动力学之后，抵消未必继续。因此仅验证终点分布一致，无法使用 Eq. (2) 的准确生成保证。

这把剩余障碍定位得比“v 是否 canonical”更细：

1. d≠0 但 div(ρd)=0 的保密度部分，ITM 可以正确运输。
2. div(ρd)≠0 的桥路径不一致部分，ITM 只会把它带到新桥。
3. restricted-head projection 和有限步 Heun 还有各自的误差。

这三层不应合并为一个“canonical defect 大小”。尤其大 ||d|| 不必意味着密度层面的错误大。

## 8. 条件中心化与 cross-fitting：有用，但其位置应清楚

设 b̂(Y) 预测条件 reward 均值、m̂(Y) 预测条件 U 均值。对独立评估样本，

\[
\mathbb E[(R-\hat b)(U-\hat m)\mid Y]
=\operatorname{Cov}(R,U\mid Y)
+(\mu_R-\hat b)(m-\hat m).
\]

cross-fitting 能避免同样本拟合带来的依赖偏差，但不能令 nuisance error 自动为零。这个乘积误差结构在 Newton Matching §8.7 的 m̂=v 特例中已有明确表述。

它适合估计 canonical covariance、构造便宜梯度 surrogate。它不会自动得到当前确定性 sampler 的 reward 梯度，也不会自动保留 Eq. (3) 的旧概率流。小 scalar reward predictor 估计的是解析重加噪桥的条件均值，不是实际 ODE 的未来价值函数；二者数据联合分布不同。

## 9. CPU 核验

脚本：[tilt_flux_audit.py](../../../experiments/theory_self_guidance_20260922/tilt_flux_audit.py)。结果：[tilt_flux_audit.json](tilt_flux_audit.json)。

旧终点 N(0,I2)，旧场为标准桥 canonical field 加 ωJx、ω=.8。取 R(x)=x2²/4，目标终点为 N(0,diag(1,2))，正规化常数 Z=√2。

脚本检查：implicit normal equation、ρ+d/h=ρd 的点态概率流恒等式、有限差分的 weighted divergence，以及独立 ODE 数值采样。对照 m+ +d 具有非零 weighted divergence，其终点 covariance 偏离目标；正确场 m+ +d/h 达到目标二阶矩的数值精度。还检查正规化权重对 reward 常数平移不变、常数 reward 时 restricted-head 半梯度严格为零。

实际结果：implicit residual 最大 1.78e−15，概率流误差 2.78e−17，有限差分 weighted-divergence 残差 4.73e−11；32768 个 scrambled Sobol 初始点经 ODE 生成的二阶矩为 [[1.000044, .000209], [.000209, 1.999757]]。直接保留旧旋转的对照场的准确 covariance ODE 给出 [[1.179465, −.360070], [−.360070, 1.805607]]。正确场 RK4 从 256 步加到 512 步的固定 1024 个种子终点最大差 2.14e−10；二阶矩剩余误差主要是积分求期望的有限样本误差。

这只是低维理论核验，不是当前图像模型的训练收益证明。更直接贴近现有终点损失的 actual-rollout Stein feedback 由另一份研究分析；Newton Matching §8.3.1 Eq. 60 的沿流 density-ratio identity 已是其连续时间敏感度恒等式的近邻，不能把基础恒等式本身声称全新。

## 10. 对方法路线的具体含义

若决定接受“终点 KL-proximal 倾斜”作为新目标，normalized ITM 是比简单 RAM 更值得比较的有原则候选：固定旧 guided anchor、指数权正规化、局部小头回归，常数 reward 时不会强迫模型做无关 canonicalization。

若希望严格保留当前 Heun GAN reward 参数目标，则本笔记不提供替代证明；actual-rollout sensitivity 或之前的随机精确反传校正更直接。

值得继续研究的空间是：能否只识别/修正 div(ρd) 对当前可训练切空间的影响，同时保留较大的无害概率流，并选择适合小头容量的速度代表。这比“一律消掉 d”更贴近受限 internal guidance 的结构。
