# RAEv2：由可观测连续性误差导出的最小 guidance

> **后续评估更正（2026-09-06）**：下文保留首次 1K 的原规则、结果和当时停止决定。有限样本论文与历史特征复核说明，1K≥5% 不是最终目标的必要条件；现冻结原权重补做配对 5K。新执行状态以[规模审计协议](RAEV2_OBSERVABLE_POTENTIAL_SCALE_AUDIT_20260906_ZH.md)为准。旧结果未改变，尚无公平成本达标结论。

**状态：理论推导与一个固定有限实现已完成；该实现配对 1K 筛查失败，已停止。** 固定训练和全时刻验证见[求解器记录](RAEV2_OBSERVABLE_POTENTIAL_SOLVER_20260906_ZH.md)，FID 从 `37.5627037` 到 `37.5316644`，仅相对降低 `0.082633%`，见[筛查记录](RAEV2_OBSERVABLE_POTENTIAL_SCREEN_20260906_ZH.md)。下文先记录理论对象，再保留有限求解与其边界，不能把精确连续保证当成该有限实现已经达标。当前目标仍是 [理论—机制—设计—公平成本下至少 5% FID 改善](RAEV2_GUIDANCE_GOAL_20260906_ZH.md)。辅助模型和训练允许使用，所有成本须计入比较。

## 1. 要区分的是分布误差，而不是普通 curl

为避免采样时间方向混淆，本文使用 dataward 时间 \(s\in[0,1]\)：

\[
Z_s=(1-s)E+sX,\qquad U=X-E,\qquad p_s=\operatorname{Law}(Z_s),
\]

其中 X 是真实 encoder latent，E 为独立标准 Gaussian。条件类别保留，真实与采样使用同一类别先验；下文省略类别记号。原 RAEv2 时间为 \(t=1-s\)。冻结 baseline 的实际 dataward drift 记为 \(v_s\)，可包含官方 IG；无需假设它是某个密度的 score，或是自身终点分布的 bridge Bayes denoiser。

真实边缘的一条合法速度为

\[
v_s^\star(z)=\mathbb E[U\mid Z_s=z],\qquad
\partial_s p_s+\nabla\!\cdot(p_s v_s^\star)=0.
\]

在 \(L^2(p_s;\mathbb R^d)\) 中，令 \(\mathcal G_s\) 为光滑梯度场的闭包，\(\Pi_s\) 为正交投影。误差 \(e_s=v_s-v_s^\star\) 分解为

\[
e_s=\Pi_s e_s+(I-\Pi_s)e_s,
\qquad \nabla\!\cdot\big(p_s(I-\Pi_s)e_s\big)=0.
\]

连续性残差只取决于第一项：

\[
\partial_s p_s+\nabla\!\cdot(p_s v_s)=\nabla\!\cdot(p_s\Pi_s e_s).
\]

因此总 MSE 可以包含不改变指定边缘路径的运动。这里的“不可观测”严格指相对于真实 \(p_s\) 的连续性方程；不是说该向量场在任意输入分布、有限 Euler 步或 decoder 后都无影响。

这与普通去 curl 不同。例如对 \(p=\mathcal N(0,I)\)，反对称矩阵 A 产生的旋转 Az 满足 \(\nabla\cdot(pAz)=0\)，但常量平移虽然 curl 为零，却会移动 Gaussian 均值。欧氏 curl 或未加权投影不能决定这一区别。

这一几何对象属于已有 weighted-Hodge / Wasserstein tangent 理论。[Diffusion Models Observe Only Gradients, 2606.06179，§3–4](https://arxiv.org/html/2606.06179v1) 将其用于 diffusion 误差分析与 critic 诊断。本文不声称提出新分解或新投影定理。

## 2. 连续性方程唯一决定最小修正

固定真实路径 \(p_s\)，求

\[
\min_{u_s}\ \frac12\mathbb E_{p_s}\|u_s\|^2
\quad\text{s.t.}\quad
-\nabla\!\cdot(p_su_s)=\partial_sp_s+\nabla\!\cdot(p_sv_s).
\]

若 \(v_s,v_s^\star\in L^2(p_s)\)，则可行集非空且是闭 affine 集，唯一最小范数解为

\[
\boxed{u_s^\star=-\Pi_s e_s=\Pi_s(v_s^\star-v_s).}
\]

在相应 Sobolev/Poisson 适定条件下，它可写成 \(u_s^\star=\nabla\phi_s\)，势函数只确定到加常数。梯度闭包中的唯一向量场结论与势函数的具体正则性应区分。该修正保留原场的加权无散部分：

\[
v_s+u_s^\star=v_s^\star+(I-\Pi_s)e_s.
\]

**若从同一个 \(p_0=\mathcal N(0,I)\) 开始、修正后的连续性方程解唯一，并存在所需的终点极限，则实际连续采样的边缘就是 \(p_s\)。** 这是全路径分布结论，系数自然为 1。它不是对未知路径同时优化的 endpoint OT，也不是最小化总速度动能；所最小化的是在指定真实路径上添加到 baseline 的修正动能。

在 \(s<1\) 的加噪区间可先讨论光滑密度；RAE clean 分布含精确 affine 约束，不能直接假设 \(s=1\) 仍有正 Lebesgue 密度。终点极限与数值离散需单独处理。原论文的 **SDE KL 界不能直接搬到 RAEv2 的确定性 100 步 Euler**；本节保证的是上述精确连续性方程和最小修正，不保证有限实现或 FID 单调改善。

## 3. 可用真实数据学习，但仍是残差 conditional flow matching

对不显含时间的光滑测试函数 \(\psi\)，弱式为

\[
\mathbb E_{p_s}[\nabla\psi\cdot u_s]
=\partial_s\mathbb E_{p_s}\psi-\mathbb E_{p_s}[\nabla\psi\cdot v_s]
=\mathbb E[\nabla\psi(Z_s)\cdot(U-v_s(Z_s))].
\]

由此得到势函数的总体目标

\[
\begin{aligned}
L_s(\phi)
&=\tfrac12\mathbb E\|\nabla\phi(Z_s)\|^2
-\mathbb E[(U-v_s(Z_s))\cdot\nabla\phi(Z_s)]\\
&=\tfrac12\mathbb E\|v_s(Z_s)+\nabla\phi(Z_s)-U\|^2
-\tfrac12\mathbb E\|v_s(Z_s)-U\|^2.
\end{aligned}
\]

这说明：coupling target 提供无偏弱残差，梯度场结构来自最小修正原理；它不是任意 gate 配终点 loss。但监督信息与 residual conditional flow matching 相同，并未绕开高维条件均值估计。使用某个 X 构造 U 不表示每条生成轨迹必须最终对应这个 X。

若测试势显含 s，对 \(\mathbb E[\phi(Z_s,s)]\) 求全导数时还须减去 \(\mathbb E[\partial_s\phi]\)。直接使用上面的 coupling 形式可避免遗漏此项。

若 baseline 与真实速度本身都属于梯度闭包，完整修正会恢复 \(v_s^\star\)；一般情况下保留 baseline 的无散运动。因此它可以作为冻结 backbone 的独立 guidance 学习，但不能预先声称辅助问题比学习 denoiser 更简单。总体理论不决定有限势空间、时间拟合方式、数据量与优化预算；已完成的一次固定实现及其失败筛查记录见本文后续与配套实验文档。这些选择会影响有限解，不能以总体唯一性代替实现依据。额外训练、势函数输入梯度和推理开销均须记录。

该公式只使用真实 \(p_s\)，并未使用实际 rollout \(q_s\) 作反馈。把权重换为冻结 baseline 的 \(q_s\)，或从已经偏离 \(p_s\) 的中途状态接入，不能直接继承第 2 节的同初始分布结论。[Stein transport](https://arxiv.org/abs/2409.01464) 也研究指定路径的变分速度，但其 RKHS 几何、kernel 与 regularization 不是这里自动获得的有限实现。

## 4. 与旧校准的关系及有限求解的可否证条件

旧 [proximal 校准](RAEV2_PROXIMAL_ERROR_PROJECTION_20260906_ZH.md) 在有限的凸二次势结构中拟合下一状态 coupling 误差，再部署非扩张映射；它不等于本节的完整 Poisson 解。旧 residual DSM/flow regression 与本方案共享条件监督信息，不能把换成弱式称为一种全新的误差来源。若再次限制为线性或二次势，就回到低维矩校准的表达范围，必须正视已有泛化失败。

对冻结有限修正 \(\hat u_s\)，独立数据上的剩余弱残差为

\[
A_s(\psi)=\mathbb E[\nabla\psi(Z_s)\cdot(U-v_s(Z_s)-\hat u_s(Z_s))].
\]

找到可靠非零的独立 witness，可否定“该修正已解掉完整连续性残差”。若 \(\hat u_s\in\mathcal G_s\)，总体上还有

\[
\sup_\psi\{2A_s(\psi)-\mathbb E\|\nabla\psi\|^2\}
=\|u_s^\star-\hat u_s\|_{L^2(p_s)}^2.
\]

有限 critic/test space 只给这个 supremum 的下界；未找到 witness 不构成完整残差很小的证书。用于选择势函数的样本不能同时充当独立确认。后续若推进，需同时检验独立弱残差、真实 rollout、离散误差与图像指标；局部回归目标下降或几个矩吻合不足以替代这些条件。

## 5. 用于解释 IG 与 MSE 反向的机制测量

这里只定义诊断，不预注册新 guidance 系数、时间窗口或 schedule。设 Full drift 为 \(v_F\)，原生差分 drift 为 d，\(v_\gamma=v_F+\gamma d\)。在原时间 \(t=1-s\) 且未触发输出到速度的 clamp 时，\(d=(F-B)/t\)。官方 IG 开关及其实际部署定义必须保留。

令

\[
u_F=\Pi_s(v_s^\star-v_F),\qquad k_s=\Pi_s d.
\]

则可观测误差为

\[
O_s(\gamma)=\|u_F-\gamma k_s\|_{L^2(p_s)}^2,
\qquad
\gamma_{\rm obs}=\frac{\langle u_F,k_s\rangle_{p_s}}{\|k_s\|_{p_s}^2}
\quad (\|k_s\|>0).
\]

若 \(k_s=0\)，此准则不能辨识系数。有限估计得到的近零分母也必须报告，不能加任意稳定器后包装为解析 guidance。

总速度误差还包含

\[
\|(I-\Pi_s)(v_F-v_s^\star)+\gamma(I-\Pi_s)d\|_{p_s}^2.
\]

因此 IG 可以降低 O、同时因无散项增大而提高总 MSE。这是可建立的机制，不是当前 RAEv2 已确认的归因：需要分别学习/估计 \(u_F\) 与 \(\Pi_s d\)，在独立样本上验证方向与 O 的变化。两项都由有限势函数拟合时，还须检查投影逼近误差。

一个解析例子说明这不是空的可能性：对平稳 \(p=\mathcal N(0,I)\)，取 \(v_F=-b\)、\(d=b+Az\)，A 反对称且 \(\mathbb E\|AZ\|^2>\|b\|^2\)。\(\gamma=1\) 后速度为 Az，完整流保持 p，而原速度导致均值平移；总速度 MSE 却变大。这个例子只验证机制可存在，不证明 RAEv2 的 IG 依靠同一机制，更不把 \(\gamma_{\rm obs}(s)\) 自动变成部署调度。

## 6. Decoder 目标与最终质量边界

完整恢复真实 encoder latent law 后，图像目标为 \(D_\#E_\#p_{\rm image}\)，即重构分布，而非自动等于真实图像分布。它解决了“冻结 baseline 并非自身 endpoint bridge”这一前提缺口；decoder 推送造成的目标差异仍须单独验证。

[已有两组 5K 对照](RAEV2_IG_SCALE_RESPONSE_RESULTS_ZH.md) 中，重构图 FID 均值为 6.861839，官方 IG 1.78 为 7.147290，相对差约 3.99%。这只是现有有限样本、固定协议的对照：**不是人口 FID 下界、不是筛查收益硬上限，也不单独构成否决依据。** 不得把 5K 的有限样本数值直接外推到 1K 或 50K，也不得减去真实子集 FID 后改写用户的 5% 标准。

最终仍需在公平总成本下，以相同噪声、类别和评价协议做配对筛查及独立种子确认；理论的精确分布结论不能替代这一结果。当前有限求解器的首次配对 1K 筛查已失败：官方 FID 37.5627037、候选 37.5316644，相对降幅仅约 0.083%。按事先固定的必要条件停止该候选，未运行更高成本对照或独立种子；无质量达标结论。

## 7. 已完成的可解 Gaussian 数值审计

[CPU 脚本](../experiments/audit_raev2_observable_error_gaussian.py) 使用一个预先固定的二维、非各向同性 Gaussian bridge。其真实均值与协方差已知，baseline 误差由固定对称线性项、固定均值偏差及 \(\Omega\Sigma_s^{-1}(z-m_s)\) 构成，其中 \(\Omega^T=-\Omega\)。最后一项满足加权无散条件；它的普通 Jacobian 却不必反对称。

对线性误差矩阵 E，最小梯度修正的对称矩阵 S 满足

\[
S\Sigma_s+\Sigma_s S=E\Sigma_s+\Sigma_s E^T.
\]

脚本独立检查这条 Sylvester 方程、加权无散条件、能量正交分解，以及连续动力学的终点均值和协方差。普通 `Sym(E)` 修正作为固定反例保留；100/200/400 步 shift-8 Euler 只作数值收敛检查，不选择方法系数或时间调度。

| 固定场 | 连续终点 W₂² | Euler-100 W₂² | Euler-200 W₂² | Euler-400 W₂² |
|---|---:|---:|---:|---:|
| oracle 概率流 | 0（舍入内） | 0.00261494 | 0.00066623 | 0.00016819 |
| baseline | 0.12255085 | 0.12347924 | 0.12288450 | 0.12268521 |
| 加权最小修正 | 4.44e-16 | 0.00030733 | 0.00007616 | 0.00001895 |
| 普通对称 Jacobian 修正 | 0.39154449 | 0.38111642 | 0.38641400 | 0.38899853 |
| oracle 加纯无散误差 | 0（舍入内） | 0.00030733 | 0.00007616 | 0.00001895 |

这个反例验证了加权几何的必要性：直接按普通 Jacobian 对称部分纠正，可以删掉本来保持目标分布的运动。它也显示连续边缘相同的场可以具有不同的 Euler 误差；不能把连续恒等式改写为离散恒等式。

完整固定矩阵、逐时刻代数检查、未截断 W₂² 舍入值及源文件 SHA 见[机器归档](data/raev2_guidance_restart_20260906/observable_error_gaussian_audit.json)。共执行两次 CPU：v1 完成原数学检查，v2 增加均值自动断言、计时和源码快照，所有数学结果逐值相同；v2 主程序耗时 0.56125 秒、CPU 0.56094 秒，均不含 import。v1 未单独记录程序耗时。两次均无 RAEv2、decoder 或图像特征模型调用，无训练、图像采样或 FID。

## 8. 有限实现的准入边界

标量势函数网络可以作为上述变分问题的数值近似：输出必须通过真实输入梯度形成 \(u=\nabla\phi\)，部署系数由方程固定为 1。网络宽度、优化器与固定预算是需要事先记录的数值实现选择，不要求定理导出每个宽度；它们不能改变修正的定义或成为广泛调参。

若势函数使用 F(z) 或 backbone 中间特征，输入梯度必须包含其完整链式导数。baseline v 仅用于数据上的固定回归目标时则可以 detach，因为它不依赖势函数参数。零初始化、输入梯度、参数反传和完整推理成本需独立核对；模型冻结不等于所有输入链路都可以停止梯度。

现已固定一个 608,000 参数的直接输入标量势网络，以 \(c=\nabla_z\Phi\)、\(u=c/t\) 实现上述变分问题，不使用 backbone 特征或新的部署系数。9 项 CPU 检查与真实 RAEv2 GPU pilot 已通过；官方原生 BF16 混合、零修正 Euler 及参数二阶梯度均已核对。一次固定 Adam 训练（2,048 次、batch 32、lr 1e-4、无调度与 checkpoint 挑选）已经完成，训练相位耗时 546.36 秒。

在与训练源图互斥的 1K 图上检查全部 100 个时间点，加权 coupling MSE 降低 1.0968%；前 17 个时间点的平均 gain 为负，其余 83 个为正。加权剩余弱残差依然显著非零，不能声称完整 Poisson 问题已求解。具体结构、数据与成本见[有限求解器协议](RAEV2_OBSERVABLE_POTENTIAL_SOLVER_20260906_ZH.md)。随后真实图像筛查仅约 0.083% FID 降幅，未达到用户 5% 目标，固定候选已经终止；该结果不能通过改窗口、系数或挑选种子补救。

这个有限结构还有明确的表达限制：第一层为逐 token 的 \(W:\mathbb R^{1024}\to\mathbb R^{128}\)，后续网络只依赖 \((Wz_p+b)_p\)，因此每个 token 的即时修正恒属于同一个 \(\operatorname{row}(W)\)，并对各 token 的 \(\ker W\) 输入扰动不变。训练可以改变 W 的行空间，却不能在该次固定 forward 内产生行空间外的修正；后续 backbone 状态反馈仍可把轨迹影响传播到其它方向。此限制不等于“捕获 128/1024 的误差”，也不是 FID 上限。它说明有限求解器的逼近条件不能从连续定理中省略。
