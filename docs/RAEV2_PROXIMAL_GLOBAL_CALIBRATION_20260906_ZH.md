# RAEv2：全局零锚点 proximal 校准与实际单步 W₂ 保证

**状态：独立 C 单步机制审计通过，但冻结配对 FID-1K 未改善。** 官方基线 `38.3977874968`，本方法 `38.4083632899`，相对降低为 `−0.02754%`；IS `57.3993 → 57.2912`。5% 目标未达到，不扩大或放大本方法系数。该选择在任何 bank C 的 stage-2 结果或候选 FID 出现前作出，依据是更直接的分布保证与每步仅一个标量的统计结构，不是依据候选之间的真实生成指标排序。

完整 C 审计保留全部 100 步：39 个活动步的主 H 证书均通过固定 100 步 Bonferroni 正态近似下界检查，61 个非活动步逐样本保持精确恒等。所有活动步的 D 同时下界为正，而该信号没有转化为终点 FID。两分支 `seed=202609066`、每批 8、1000 类各一张、完整 100 步与官方 IG，噪声、标签、checkpoint、decoder、像素量化和评价器均核验一致；无额外模型前向。

结果：[配对比较 JSON](data/raev2_guidance_restart_20260906/global_fid_comparison.json)、[独立机制确认](data/raev2_guidance_restart_20260906/global_independent_confirmation.json)。完整外部产物位于 `raev2_guidance_restart_20260906/proximal_seed202609066/` 与 `proximal_global_c_seed202609065/`。该阴性结果不否定已证明的单步定理，但否定“本次微小 latent 校正已经带来实用 RAEv2 FID 收益”。

## 1. 选择过程与冻结范围

原逐坐标 diagonal 方案完成全部 100 步审计：23 步平均 heldout D 为正、77 步为负，74 步在 Bonferroni 校正后为负；未运行 FID。其训练证书为正而多数新图像风险恶化，支持拟合高维 offset/斜率的统计风险假设，不能解读为原总体投影定理错误。

[channel 共享结构](RAEV2_PROXIMAL_SHARED_CALIBRATION_20260906_ZH.md) 是看到早期 diagonal 失败后提出的第一条修订。随后理论审阅得到下述更强结构：所有坐标只用一个非负斜率，完全去掉 translation 与待估计中心，直接保证任意分布的真实输入单步 W₂ 不增。因此选择 global zero-anchor 为唯一新的主候选，channel 方案保留理论，不执行其 bank C/FID 比较。bank B 已用于设计修订，不能作为本候选的全新独立确认。

所有官方 100 步统一使用相同的闭式规则，官方 IG=1.78 与其 [.1,1] 区间保持冻结。不新增 guidance 幅度、时间窗口、rank 或阈值搜索；数据估计的逐步标量由同一个目标推导，不是手工时间调度。未激活的时间步严格恒等，不要求其 heldout D 显著为正。

## 2. 只拟合一个未中心化 cross moment

在每个官方 Euler 步 t→s，令 Z=z_t，Y=z_s 为真实 coupled bridge 状态，X=T(Z) 为冻结 baseline 的下一状态预测。这里 X 专指本节预测下一状态，不是 clean data latent。

定义

\[
A=\mathbb E\|X\|^2,\qquad B=\mathbb E\|Y\|^2,\qquad
C=\mathbb E[X^\top Y].
\]

在闭凸锥 h(y)=ay、a≥0 中投影 r=X−Y，得到

\[
a=\left(\frac{C}{B}-1\right)_+,\qquad
\boxed{\ S(Z)=\lambda T(Z),\quad
\lambda=\begin{cases}B/C,&C>B>0,\\1,&\text{otherwise}.\end{cases}\ }
\]

这是 ψ(y)=½a‖y‖² 的 proximal 映射，λ=(1+a)⁻¹。B=0 时锥在真实目标上退化，定义恒等保持安全。原点是固定锚点；定理**不要求**真实 latent 均值为零、单位方差或 Gaussian，也无需估计高维均值。a 是一步完整残差的投影系数，没有额外可调 proximal 强度。

原 diagonal 训练充分统计可直接恢复它。若每坐标已有 target 均值 μʸ_i、residual 均值 μʳ_i、target 方差 V_i 和 residual/target 协方差 K_i，则

\[
B=\sum_i(V_i+(\mu^y_i)^2),\quad
N:=C-B=\sum_i(K_i+\mu^r_i\mu^y_i),\quad
a=(N/B)_+.
\]

求和或整体除以维度必须对所有矩一致；两者给出相同 λ。无需重新训练、无需额外采样模型 forward。不能以只含 centered covariance 的统计替代 N：这里刻意采用零锚点的未中心化 inner product。

## 3. 任意分布的实际单步 W₂ 定理

假设 X,Y 有有限二阶矩。令

\[
M=\sup_{\pi\in\Pi(\mathcal L(X),\mathcal L(Y))}
\mathbb E_\pi[X^\top Y].
\]

实际校准 pairing 是边缘 coupling 之一，所以 M≥C；Cauchy–Schwarz 给出 M²≤AB。对任意 λ>0，缩放是可逆的，因此最优 coupling 的 cross moment 与 λ 无关，准确地有

\[
W_2^2(\mathcal L(\lambda X),\mathcal L(Y))
=\lambda^2A+B-2\lambda M.
\]

若 C≤B，λ=1 恒等。若 C>B>0，则 M≥C>B 且 A≥M²/B>M，故

\[
0<\lambda_*:=M/A\le B/M\le B/C=\lambda<1.
\]

λ_* 是上述实际 W₂ 二次式沿缩放射线的最小点。所选 λ 从 1 向最优点移动而不越过它，因此

\[
\boxed{W_2( S_\#p_t,p_s)\le W_2(T_\#p_t,p_s).}
\]

这里保证的直接就是生成下一状态的**边缘分布距离**，不是把局部 MSE 下降称作质量改善。无需 Gaussian、坐标独立、最优 teacher pairing 或保守 score。证明也说明它为何较保守：实际 pairing 的 C 只提供最大 cross moment M 的下界，不知道 M 时不能冒进缩到射线最优系数。

Bayes oracle 同样保持：若 X=E[Y|Z,label]，则 C=E‖X‖²=A≤B，从而 λ=1。这排除了不可约条件不确定性导致的总体收缩。

与此同时，凸锥投影使真实 coupling 的平方误差不增，Lip(S)=λ Lip(T)≤Lip(T)，因此原 [有限步传播上界](RAEV2_PROXIMAL_ERROR_PROJECTION_20260906_ZH.md) 也保留。也可直接以边缘 defect δ_k=W₂(S_k#p_k,p_{k+1}) 递推，W₂(q_{k+1},p_{k+1})≤Lip(S_k)W₂(q_k,p_k)+δ_k；当前定理保证这个更直接的 δ_k 不增。proximal 非扩张及锥分解是经典结构，见 [Moreau 原文](https://www.numdam.org/article/BSMF_1965__93__273_0.pdf)；上述最优 coupling 的缩放证明是本记录直接展开的特化论证，不声称凸分析或 Wasserstein 理论本身的新颖性。

## 4. 有限拟合的直接分布证书

从有限训练 bank 拟合 λ 后，非扩张仍严格成立，但总体 C/B 的比较需要新数据验证。对任意冻结 λ∈(0,1)，一个直接支持实际 W₂ 结论的充分条件是

\[
\boxed{H_{stay}:=\lambda^2\mathbb E\|T(Z)\|^2-\mathbb E\|Y\|^2\ge0.}
\]

因为 H_stay≥0 推出 λ≥√(B/A)≥M/A=λ_*，同样不会越过实际 W₂ 的射线最优点。总体 λ=B/C 在激活时由 C²≤AB 自动满足 H_stay≥0。

一个更弱、同样充分的有限拟合条件为

\[
\boxed{H_{sec}:=(1+\lambda)^2 A-4B\ge0.}
\]

由 M≤√(AB)，实际距离之差满足

\[
\Delta W_2^2=(1-\lambda)[2M-(1+\lambda)A]
\le(1-\lambda)[2\sqrt{AB}-(1+\lambda)A]\le0.
\]

H_sec 允许越过射线最小点，只要求修正后的距离不高于 λ=1 的原距离；H_stay 保证不越过。两者均是直接分布证书，H_sec 可作为独立 heldout 准入的优先证书，H_stay 作附加诊断。它们不会改变拟合 λ 的规则。

heldout 可逐图记录 H_stay,i=λ²‖T(Z_i)‖²−‖Y_i‖²、H_sec,i=(1+λ)²‖T(Z_i)‖²−4‖Y_i‖²，以及原 D_i=‖T(Z_i)−Y_i‖²−‖λT(Z_i)−Y_i‖²。固定 λ 后它们均为相应总体量的无偏样本均值（服从实际 bank 抽样设计）；使用新图像评估其误差与置信区间，不能把坐标数当作独立样本数。未激活的 λ=1 是恒等映射，无需 H≥0，也无需 D>0。

H_sec 总体非负也蕴含所有实际 coupling 的总体 D≥0，因为 D=(1−λ)[(1+λ)A−2C]，且 C≤√(AB)。所以不能声称总体 H_sec 正而总体 D 负；有限样本的均值/区间可能因估计波动而显示不同强度，J 则不必非负。原 diagonal 的 D 失败与这一新 scalar 证书应分别保留。

该 H 条件仅用边缘二阶矩：如果用实际 rollout 的 T#q_t 能量验证它，同一个**固定 λ** 也能支持对应 q_t 的单次分布保证。但这里没有运行这一验证，也没有加入在线重选 λ 的采样规则。当前真实前提审计的输入仍是真实 corrupted latent。

class-balanced 一类一图定义 uniform-class mixture；global λ 对应该混合总体，不保证每类均单调。若需保留标签，证明也适用于相同类别先验下的 Σ_cπ_cW₂²(p(·|c),q(·|c))：把 M 定义为同标签 coupling 的最大 cross moment，所有类别共用同一个 λ，二次式仍成立。这支持条件模型的传播论证，而非假设忽略标签后的最优 coupling 自动与模型条件相容。原 bank 的选图与分层不确定性边界见共享审阅。bank C 的新审计用于检验已冻结主候选；任何结构再选择都会改变后续独立确认的解释。

## 5. CPU toy：直接距离、oracle 与失败边界

代码：[audit_raev2_global_proximal_w2_toy.py](../experiments/audit_raev2_global_proximal_w2_toy.py)。结果：[global_w2_toy_audit.json](data/raev2_proximal_error_projection_20260906/global_w2_toy_audit.json)。所有测试场固定，不搜索方法强度或窗口；动态 Gaussian 使用数学上的官方 shift=8、100 步 grid。

| 实验 | 原实际 W₂² | global 后实际 W₂² | 附加验证 |
|---|---:|---:|---|
| 非零均值非 Gaussian 离散分布、非最优 pairing | 61.25 | 0.274552 | 线性规划解离散最优传输，C<M |
| 非零均值、不同方差 Gaussian bridge100 | 0.917623 | 0.028955 | 100 个真实输入单步边缘全部不增 |
| 同一 Gaussian 的 Bayes oracle100 | 0.002697 | 0.002697 | 每一步 λ 严格为 1 |
| 非零均值：T=2Y−1，Y∼N(1,1) | 1 | 0.222222 | 均值误差可增，而总距离下降 |
| 两步抵消：T₁=2I，T₂=I/2，目标均为 N(0,I₂) | 0 | 0.5 | 单步定理成立，实际终点仍可变差 |

Gaussian bridge 的完整 W₂ 递推上界为 3.495206→2.712108；注意这是 W₂ 上界，表中实际距离是 W₂²。非 Gaussian 例子的目标均值为 (2.3,1.2)，实际 pairing C=37.658，小于最优 M=41.6；λ=0.353179，确实不越过射线最优 λ*=0.317194。

最后一行的 boundary 是必要限制：raw 两个映射互相抵消，终点正确；修正把第一步变成 I、第二步仍为 I/2，终点反而变差。每个真实输入的单步 W₂ 保证都成立，完整上界也从 √2 降为 1/√2。另一个直接的 q/p 反例是：teacher p_t=N(0,I₂) 校准 T=2I 得 λ=1/2，但实际 incoming q=N(0,I₂/4) 时 raw 输出已经正确，再缩放会把 W₂² 从 0 增至 .5。

因此本机制满足理论支撑的准入，有比局部 MSE 更直接的分布保证，也消除了原高维 translation 拟合；它仍不承诺所有递归 rollout 的终点距离单调，更不代表 RAEv2 的约 5% FID 目标已实现。latent W₂ 与经 decoder/Inception 计算的 FID 也不是同一个对象，最后的生成目标必须保留真实实验验证。

复现：

```bash
python experiments/audit_raev2_global_proximal_w2_toy.py --output docs/data/raev2_proximal_error_projection_20260906/global_w2_toy_audit.json
```
