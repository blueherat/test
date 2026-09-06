# RAEv2：实际 rollout 的能量球投影

**状态：理论与数值核验通过，冻结配对 FID-1K 未改善。** 官方基线 `38.3977874968`，能量球 `38.4086041620`，相对降低 `−0.02817%`；IS `57.3993 → 57.4165`。目标未达到。完整 100 步中 51 步自然激活，最小 FP32 缩放系数 `0.9904050`；没有据此回调 B、时间段或收缩强度。

本方案从实际 rollout 的能量估计出发，所有 100 步统一使用同一公式、整个固定 N=1000 cohort 共用一个缩放系数；不能按每个计算 batch=8 单独归一化，也不搜索幅度、时间窗口或粒子数来争取 FID。首次运行前，真实 GPU 的 16 图官方模式与原 sampler 的 endpoint 和像素逐位相同。正式 `seed=202609066` 的全部 1000 个初始噪声、标签、模型/decoder/评价协议均与基线核验一致；仍为每图 100 次模型前向。

结果：[配对比较 JSON](data/raev2_guidance_restart_20260906/energy_ball_fid_comparison.json)。完整产物在 `raev2_guidance_restart_20260906/energy_ball_seed202609066/`。与独立的 global A-fit 候选均无实用 FID 收益，因此不继续搜索这两个径向结构的手工变体。

## 1. 仓库先例：有近邻，但不是同一干预

最接近的先例不是 9 月 5 日 radial IG，而是 [mnist_transport_mechanism.py](../experiments/mnist_transport_mechanism.py)：`rollout_states` 在每步 Euler 后，对整个实际 state bank 计算 8 个 DCT band 的能量；`calibrate_band_energy` 用 √(target/current) 双向匹配各频带，目标也来自 clean moments 的 bridge 公式。[结果记录](TEACHER_ROLLOUT_MECHANISM_ZH.md)显示，完全匹配频带能量后，weighted 模型仍比同样校准的 baseline 差约 12.1% feature FID。这排除了“能量是模型质量差距全部原因”，没有证明所有单向能量约束都无效。不能把沿 rollout 做矩校准重新包装成首次提出。

本方案的结构差异是：**一个全局二阶矩球、只收缩、直接作用于下一状态的实际分布**。去掉扩张是定理成立的关键；去掉多个独立 band 则使保证直接对应完整 latent 的 W₂。

[9 月 5 日 RAEv2 radial/retraction](RAEV2_RADIUS_DIRECTION_PLAN_20260905_ZH.md)改的是每个 token 的 clean prediction，参考半径来自同状态 full head；它的微小正信号在独立 bank 反向。`run_raev2_decoder_variance_intervention.py` 则在 stage-2 endpoint 已固定后改变 decoder hidden 的空间方差。这两条都没有使用实际下一状态分布的全局目标能量球。

`analyze_raev2_projection_hierarchy.py` 也报告过将 B+s(F−B) 的 teacher clean 二阶矩匹配到真实 clean target 的根，并明确不建议直接部署：它会混入不可约条件方差。这里无需构造该外推根，不放大条件均值的方差不足。

## 2. 总体规则不需要真实输入假设

在一步 t→s 中，实际 incoming 分布为 q_t，冻结 Euler 映射为 T_t，记 μ=T_t#q_t，目标边缘为 ν=p_s。定义 A=E_μ‖Z‖²，B≥E_ν‖Y‖²，令

\[
\lambda(\mu)=\min\{1,\sqrt{B/A}\},\qquad
P_B\mu=(\lambda(\mu)\operatorname{Id})_\#\mu.
\]

A=0 时定义 λ=1；B=0<A 时投影为原点。不假设 latent 零均值、Gaussian、球面支撑或坐标独立。B 是总体二阶矩上限，不是每张图的硬半径上限；所有样本共用同一个 λ，个体之间的半径比例保持。

在承载任何 coupling (Z,Y) 的 Hilbert 空间 L² 中，Z↦λZ 正是闭球 {U:E‖U‖²≤B} 的度量投影。因此

\[
\mathbb E\|\lambda Z-Y\|^2\le\mathbb E\|Z-Y\|^2
\quad\text{当 }\mathbb E\|Y\|^2\le B.
\]

取最优 coupling 可得实际分布保证

\[
\boxed{W_2(P_B(T_t\#q_t),p_s)\le W_2(T_t\#q_t,p_s).}
\]

这里 q_t 可以是已经偏离训练 bridge 的实际 rollout 分布。这比仅在 p_t 上校准的 [B/C proximal 候选](RAEV2_PROXIMAL_GLOBAL_CALIBRATION_20260906_ZH.md)少了真实输入的限制；比较二者总体作用于同一 μ 时，B/C 激活规则的 λ≥√(B/A)，因而本规则沿同一射线走得更远，但仍不会超过安全边界。

也可直接写 M=sup_coupling E[ZᵀY]≤√(AB)，W₂²(λ#μ,ν)=λ²A+B_ν−2λM。A>B 时，λ=√(B/A)≥M/A；从 1 缩到 λ 不会越过沿射线的最小点。

## 3. 它还是 Wasserstein 中的非扩张 law map

对任意 μ、η，选取最优 coupling Z∼μ、V∼η。两个缩放系数只取决于各自边缘的 L² 范数；在同一 coupling 空间内应用闭球投影的非扩张性，得到

\[
\boxed{W_2(P_B\mu,P_B\eta)\le W_2(\mu,\eta).}
\]

该结论不是把所有 Wasserstein 投影都假设为非扩张；这是本能量球的 Hilbert lift 直接证明。闭凸集投影/proximal 非扩张是经典结果，见 [Moreau 原文 §4–5](https://www.numdam.org/article/BSMF_1965__93__273_0.pdf)。若 μ 的能量超过 B，P_Bμ 到 μ 的 W₂ 距离恰为 √A−√B，达到由三角不等式与 δ₀ 导出的球距离下界。

因此总体分布递推 q_{k+1}=P_{B_{k+1}}(T_k#q_k) 的稳定常数不超过原 T_k 的 Lipschitz 常数。它仍不保证与整条未修正 baseline 比较时终点 W₂/FID 单调：原来的两步 2I→I/2 抵消反例依然成立。这里提高的是每次实际校正的分布保证与稳定性，不是假定未来网络不会利用已有偏差抵消。

## 4. 目标能量的来源与固定数值

实现以每坐标平均能量表示。若 d=1024×16×16，clean normalized latent 为 X₀，独立 ε∼N(0,I_d)，则

\[
b_s:=d^{-1}\mathbb E\|(1-s)X_0+s\varepsilon\|^2
=(1-s)^2m_2+s^2,\qquad m_2=d^{-1}\mathbb E\|X_0\|^2.
\]

交叉项为零来自独立、零均值噪声，不要求 clean 均值为零。计算 B=d b_s 或统一使用规范化范数 ‖·‖/√d 均等价；不得混用两种归一化。

本轮使用原 A bank 1000 个真实 clean normalized latents 的 FP64 empirical m₂，固定值为 **0.9977984298211652**。因此 b_s=(1−s)²×0.9977984298211652+s²。这精确定义了代理目标：从这 1000 个 clean latent 的 class-balanced empirical 分布抽取 X₀，再混入独立标准 Gaussian。对这个代理分布，b_s 是准确总体矩，不是带有限 ε 噪声的单次 target bank 估计。

独立 C 清洁矩审计报告 m₂=**0.9960813034350342**，A/C 差约 **+0.0017171263861310**，所报告 pooled SE=**0.00570590562620335**；差异不显著，不能称 A 矩已被证明为真实总体上界。C 只用于矩不确定性审计，不据此回调、混合或重新选择目标 b_s。

对真实目标 ν，即使使用的 B 偏小，也有准确的单项误差界

\[
W_2(P_B\mu,\nu)
\le W_2(\mu,\nu)+
\left(\sqrt{\mathbb E_\nu\|Y\|^2}-\sqrt B\right)_+.
\]

证明是插入 P_Bν，再用 law 非扩张与 W₂(P_Bν,ν)=(√B_ν−√B)_+。目标矩估计偏大仍安全，只是修正较弱；偏小的代价由半径亏损控制。每坐标规范化的 W₂/√d 使用 (√b_true−√b_A)_+；数据矩不确定性对 b_s 的贡献为 (1−s)²(m₂,true−m₂,A)。这保留了真实总体与 empirical 代理目标的区别，不通过 C 来追逐更有利的数值。

## 5. 有限 N=1000：必须把整个 cohort 当作一个粒子系统

每一步先对当前全部 N 个粒子计算 T(zᵢ)，累加

\[
a_N=\frac{1}{Nd}\sum_{i=1}^N\|T(z^i)\|^2,\qquad
\lambda_N=\min(1,\sqrt{b_s/a_N}).
\]

所有 N 个预测下一状态乘同一个 λ_N，然后进入下一步。后续能量必须来自已经修正过的递归粒子状态，不能在未修正 baseline 轨迹上先测完一条曲线再冒称实际 q_t 校正。

固定 N=1000、每类一个粒子；计算 microbatch=8 只用于分批模型 forward。需要保存全部下一状态或使用分片存储，在完整 cohort 的 FP64 总能量确定后统一应用 λ_N。每个 microbatch 单独求 λ 会产生另一套依赖分组的算法，不能混称同方法。整个 cohort 的重排、分片和求和应遵循固定记录；模型 BF16 本身的 batch 数值差异仍应按原冻结协议控制。

λ_N 是整个 R^{Nd} 粒子向量向半径 √(Nd b_s) 球的投影，因而对粒子 RMS 距离非扩张。更强地，对每个已经实现的 cohort，经验分布 μ_N 的二阶矩是准确的，所以

\[
W_2(P_B\mu_N,\nu)\le W_2(\mu_N,\nu)
\]

对任意能量不超过 B 的真实或代理 ν 逐 cohort 成立；不需要把 ν 也离散化，不需要先假定经验分布已经逼近 ν。

**但这个结论不等于单粒子无条件边缘的总体质量改善。** 取本来已经准确的 ν=(δ₀+δ₂)/2，B=2，N=2 iid 粒子。cohort (2,2) 被缩为 (√2,√2)，其他三个可能 cohort 不变。每个经验分布到 ν 的 W₂ 都不增，但合并所有随机 cohort 后，单粒子输出 law 偏离 ν，W₂² 从 0 增至 0.0857864。粒子间依赖和有限 N 的单边截断偏差确实存在，不能因经验定理而隐去。

N 是平均场数值近似的分辨率，会改变有限粒子 law；microbatch 则是计算分块。二者不是同一概念。这里 N=1000 由固定筛查规模与完整类别覆盖确定，不比较不同 N 的 FID 来择优。后续独立确认应以新的完整 cohort 为重复单位，不把同一 interacting cohort 的粒子当作完全 iid 来声称高精度置信区间。

## 6. 有限粒子误差可以只依赖能量标量的估计

一般高维经验 W₂ 的收敛很慢，见 [Fournier–Guillin 原始论文 Theorem 1](https://arxiv.org/pdf/1312.2128)。因此不以 d=262144、N=1000 声称整个经验分布已接近总体分布。

本机制只通过一个能量矩耦合粒子，可作更具体的有限时间 strong-coupling 论证。设理想平均场粒子 \bar zᵢ_k 用同样初始随机数，但每步使用准确总体能量；它们独立演化。若 T_k 的 Lipschitz 常数为 L_k，令

\[
D_k=\left(\mathbb E\frac1N\sum_i
\|z^{i,N}_k-\bar z^i_k\|_d^2\right)^{1/2},\quad
A_k=\mathbb E\|T_k(\bar z_k)\|_d^2,
\quad V_k=\operatorname{Var}(\|T_k(\bar z_k)\|_d^2),
\]

其中 ‖·‖_d=‖·‖/√d。在 iid 版本且四阶矩有限时，有限粒子球投影非扩张与

\[
|\lambda(a)-\lambda(A)|\sqrt a\le|\sqrt a-\sqrt A|
\]

给出

\[
D_{k+1}\le L_kD_k+\sqrt{V_k/(NA_k)}\quad(A_k>0).
\]

原理是先比较实际粒子与理想 iid 粒子的**同一个有限 N 投影**，再把后者的样本能量换成准确 A_k。样本能量方差为 V_k/N；初始 D₀=0，有限 100 步展开得到形式上的 O(N⁻¹ᐟ²) RMS 误差。one-per-class 的独立分层粒子将 V_k/N 换成 N⁻²∑ᵢV_{k,i}，不把类间均值差当成类别抽样噪声。

这是该标量相互作用的直接离散推导，不是无条件引用某个 McKean–Vlasov 定理。常数仍受真实场的 L_k、能量四阶矩和时间累计影响；它不证明 N=1000 已足够，也不消除上面的有限粒子反例。若另外使用独立固定 calibration particles 先递归估 λ 再冻结生产曲线，可避免生产 batch 分组依赖，但生产 q 的矩保证仍只近似成立；本轮没有把这条方案加入可切换实验分支。

## 7. Tiny CPU 审计

代码：[audit_raev2_energy_ball_toy.py](../experiments/audit_raev2_energy_ball_toy.py)。结果：[energy_ball_toy_audit.json](data/raev2_proximal_error_projection_20260906/energy_ball_toy_audit.json)。用小离散分布的最优传输线性规划计算 W₂，不使用 Gaussian 近似。

| 检查 | 结果 |
|---|---|
| 非零均值非 Gaussian actual law | W₂² 61.25→0.104934 |
| 两个不同分布经过相同能量球投影 | W₂² 46.55→3.135674，验证 law 非扩张 |
| 全部 1000 粒子一次求和 vs microbatch=8 累加一次总能量 | λ 差为 0 |
| 错误地按分组分别归一化，原 cohort 已等于目标 | W₂² 0→0.171573 |
| N=2 全枚举：经验距离均不增但单粒子总体失真 | W₂² 0→0.0857864 |
| 真实 B=2，误用 B=1，原 law 已正确 | W₂²=0.171573，恰等于半径亏损平方 |

全部检查 passed。没有通过调参产生的 toy 胜出，也没有把真实图像 FID 替换成 latent W₂。能量只约束一维统计结构，仓库先例已经说明它不足以完整决定生成质量；本次的价值在于一个可核验、可部署到实际 rollout 的分布校正机制，最终约 5% image FID 目标仍须真实实验回答。
