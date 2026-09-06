# Sobolev Descent 原文核对：从实际采样分布学习输运方向

**结论：可迁移的结构是用当前实际粒子分布的 Jacobian Gram，把两样本的特征均值缺口变成输入梯度场；它不要求 RAEv2 是 exact score。** 这与旧真实 bridge 上的 `U−b` 残差势函数有不同的监督信息。但其严格局部结论是指定特征 MMD 的下降贡献，不是完整 RAE 轨迹、KL 或 FID 的下降保证。当前证据尚不足以选择一组值得纠正的 RAE 特征并启动训练。

本次仅阅读、代数核对和仓库来源复查，没有实现、训练或 GPU 实验。

原文为 Mroueh、Sercu、Raj，*Sobolev Descent*，AISTATS 2019，PMLR 89:2976–2985。[正式页面](https://proceedings.mlr.press/v89/mroueh19a.html)、[正文](https://proceedings.mlr.press/v89/mroueh19a/mroueh19a.pdf)、[完整附录](https://proceedings.mlr.press/v89/mroueh19a/mroueh19a-supp.pdf)。实际阅读范围：正文 §2–5；附录 A、B/B.1、C.2、D、E、F、算法 1/2 与 H。以下将原文结论与本文重新推导、迁移判断分别标明。

## 原文证据的边界

正文 Theorem 1 是固定目标下的 MMD² 一阶变化；§3.1 与附录 B.1 声称在无停滞假设 A 下连续流收敛。有限特征只匹配相应均值；分布识别还需要合适的 characteristic kernel。算法 1 每次粒子更新后重算当前均值和 Gram。神经版本用近似 critic，不能直接继承精确有限维求解的结论。[正文 §2–3、5](https://proceedings.mlr.press/v89/mroueh19a/mroueh19a.pdf)

最强机制证据是正则化改变 Gaussian/形状输运路径；不是现代生成质量基准。颜色实验是 RGB 三维点云，EMD/Sinkhorn 仅用 4K/6K 子样本；CIFAR 实验只针对 5K 张 truck，FID 使用 Inception 192 维特征，图横轴到 10K 次粒子更新，未提供足以作本任务公平成本比较的图像生成强基线。附录形状神经实验为 800 次粒子更新，每次 10 次 critic 更新，初次 50 次。[正文 §5、图 4/8](https://proceedings.mlr.press/v89/mroueh19a/mroueh19a.pdf)、[附录 G、H.3](https://proceedings.mlr.press/v89/mroueh19a/mroueh19a-supp.pdf)

## 重新推导：两样本缺口为什么能给出方向

以下固定一个采样时刻，把额外输运时间记为 τ，避免与 RAE 原时间 t 混淆。设目标为 p，当前状态分布为 q；两者只需样本。选定可微特征 `Φ: R^d→R^m`，采用原文列梯度约定

\[
J(z)=[\nabla\Phi_1(z),\ldots,\nabla\Phi_m(z)]\in\mathbb R^{d\times m},\quad
\delta=\mathbb E_p\Phi-\mathbb E_q\Phi,\quad
D_q=\mathbb E_q[J^TJ].
\]

对势函数 `φ_a=aᵀΦ`，二次变分目标及其正则项成为

\[
\begin{aligned}
\mathcal L(a)
&=\tfrac12\mathbb E_q\|\nabla\phi_a\|^2
 -(\mathbb E_p\phi_a-\mathbb E_q\phi_a)+\tfrac\lambda2\|a\|^2\\
&=\tfrac12a^T(D_q+\lambda I)a-a^T\delta,\\
a_*&=(D_q+\lambda I)^{-1}\delta,\qquad u(z)=J(z)a_*.
\end{aligned}
\]

这里使用未归一化的平方 discrepancy witness；把它再除以 discrepancy 会改变输运速度。`D_q` 是**输入梯度的 Gram**，不是 `Cov_q(Φ)`，也不是普通特征 kernel 矩阵。

一阶条件对每个测试特征给出

\[
\mathbb E_q[\nabla\Phi_j\cdot u]+\lambda a_{*,j}=\delta_j.
\]

因此 `z↦z+εu(z)` 的特征均值变化为 `εD_qa_*+O(ε²)`。当 λ=0 且可解时，它精确实现当前有限特征缺口的一阶填补；λ>0 留下 `λa_*`。无限函数空间、适当边界和可解性下，对所有测试函数成立的弱式才对应 `−div(q∇φ)=p−q`。有限特征正则解不等于这个完整 Poisson 方程。

令 `MMD_Φ²(p,q)=||δ||²`。直接对均值求导，无需任何 score：

\[
\left.\frac{d}{d\epsilon}\operatorname{MMD}_\Phi^2(p,(I+\epsilon u)_\#q)\right|_0
=-2\delta^TD_q(D_q+\lambda I)^{-1}\delta
=-2\sum_j\frac{\rho_j}{\rho_j+\lambda}\delta_j^2\le0,
\]

其中 `ρ_j` 是 `D_q` 的特征值，`δ_j` 是相应坐标。这也等于原文的 `−2(MMD²−λS²)`。这里下降的是所选有限特征的 MMD；它既不推导 KL 下降，也不解全路径最小动能 OT，更不能由此推出 FID。全 Sobolev 空间无正则的理想极限甚至给出密度混合 `q_τ=(1−e^(−τ))p+e^(−τ)q_0`，并不自动产生期望的平滑形态输运。[附录 D、F](https://proceedings.mlr.press/v89/mroueh19a/mroueh19a-supp.pdf)

## 收敛、奇异性与有限步：不能省略的条件

原文假设 A 是 `δ≠0 ⇒ δ∉Null(D_q)`，而非仅仅 `D_q+λI` 可逆。若特征在 q 支持上饱和，可能 `δ≠0` 但 `Jδ=0`，正则化虽然给出有限系数，却没有任何一阶输运进展。对固定总体 q 的常量特征，缺口本来为零；小样本误差、近饱和和近共线特征则会使估计与解敏感。

**证明核查。** 附录 B.1 从“非负单调函数有极限”直接转为在某个有限 `τ₀` 达到极限。这一步本身不成立；不能把这段论证当作非紧 RAE 状态空间的充分收敛证明。这不等于否定其局部恒等式，也不证明紧空间下定理为假：在紧/紧致可控的状态族、连续的均值和 Gram、流存在且无停滞时，可另用极限点论证。一个可自行检查的更强充分条件是沿轨迹 `D_q≽ρ_min I`、`ρ_min>0`，此时

\[
\operatorname{MMD}^2(p,q_\tau)
\le e^{-2\rho_{\min}\tau/(\rho_{\min}+\lambda)}\operatorname{MMD}^2(p,q_0).
\]

这类统一可控性远强于在一个 bank 上 Cholesky 成功；有限特征 MMD 为零也只代表该组矩匹配。

**有限步的自行推导。** 记 `F(ε)=½MMD²(p,(I+εu)#q)`，`K=δᵀD_qa_*`。若沿该段 `F''≤L_line`，则

\[
F(\epsilon)-F(0)\le-\epsilon K+\tfrac12L_{\rm line}\epsilon^2.
\]

所以只有 `K>0` 且 `0<ε<2K/L_line` 等条件才能控制这一步；原文一阶导数不能保证任意有限 ε。流映射可逆还涉及势函数 Hessian，安全的充分条件可用 `ε sup||∇²φ||op<1`。不能从一次正的有限差分或训练 MMD 下降声称得到了全局界。

**估计与刷新。** 经验解为 `â=(D̂_q+λI)^−1δ̂`。它在拟合 bank 上的代数下降量非负，属于同库拟合性质；独立当前 q 上必须检查 `δ_currentᵀD_current â>0` 及不确定性。冻结旧 q 的均值、Gram 或 critic 后，这个量可能反号。算法 1 在原粒子推进后重新拟合正是关键；不必每步重新抽初始噪声，但必须统计已经移动的当前粒子。用另一个 reference cohort 估计时，还需说明其规律与正在生成的 cohort 一致。沿同一初始噪声跨时间的记录不能当作独立 SEM 样本。

## 与旧 observable potential 的实质差异

令 dataward 时间 `s=1−t`，真实 bridge 为 `Z_s=(1−s)E+sX`、`U=X−E`，真实边缘为 `p_s`，冻结 RAE baseline drift 为 b。既有推导的有限特征解与新结构可直接比较：

| 对象 | 旧真实 bridge 势函数 | 当前分布的 Sobolev 输运 |
|---|---|---|
| 二次项测度 | `p_s` | 实际生成状态 `q_s` |
| Gram | `E_p[JᵀJ]` | `E_q[JᵀJ]` |
| 右端 | `E_real[Jᵀ(U−b(Z_s))]` | `E_pΦ−E_qΦ` |
| 误差信息 | 真实路径的速度/连续性残差 | 已经积累在实际状态中的特征分布缺口 |
| 所需监督 | 真实 coupling 的 U | 两组不配对样本即可 |
| 保证适用范围 | 指定真实路径的梯度投影 | 固定目标的局部特征输运 |

旧目标是残差 flow matching 的梯度场限制，详见[旧推导](RAEV2_OBSERVABLE_ERROR_GUIDANCE_20260906_ZH.md)。把其 `p_s` 权重改为 `q_s` 并继续使用原 `U−b` 标签不会自动得到新右端；实际生成状态没有一个天然配对真实 U。两样本目标增加的是实际分布反馈，不能称为把旧网络换了一个 loss 名字。

一个自行构造的局部例子：固定 `p=N(0,1)`、正确静态 baseline `b=0`，当前粒子因先前误差为 `q=N(m,1)`。旧真实路径速度残差为零，不能诊断既成偏移；线性特征的两样本解则有 `D_q=1, δ=−m, u=−m/(1+λ)`，会修正均值。这个计算只需有限矩，不声称线性无界特征满足原文全部有界假设，也不证明 RAE 的遗漏属于均值平移。

现有补空间径向 witness 和自身幅度 witness 都是在真实 bridge 上测得，不能充当 `q_s` 的两样本缺口证据；[已完成诊断](RAEV2_POTENTIAL_OMITTED_WITNESS_20260906_ZH.md) 的小有限下界也不是完整误差上界。若新 Φ 仍只包含旧均值、径向或同一低秩输入特征，则存在明显重复风险。

## 移动目标下的 RAE 迁移：一个具体可证伪的启发

有用的设计变化是：**先分别估计“当前积累缺口”和“当前 baseline 继续制造缺口的速度”，再判断哪个需要修正。** 这样不会把所有两样本差异都当作生成错误，也无需假设 F/B 中任一头是自身生成边缘的 score。

以下是本文推导，不是原文对 diffusion 的定理。先让 Φ 不显含 s。对实际 ODE `dZ/ds=b_s(Z)+J(Z)a_s`，有

\[
\begin{aligned}
\delta_s&=\mathbb E_{p_s}\Phi-\mathbb E_{q_s}\Phi,\\
r_s&=\mathbb E_{\text{real bridge}}[J(Z_s)^TU]
       -\mathbb E_{q_s}[J(Z)^Tb_s(Z)],\\
\partial_s\delta_s&=r_s-D_{q_s}a_s,\\
\tfrac12\partial_s\|\delta_s\|^2&=\delta_s^Tr_s-\delta_s^TD_{q_s}a_s.
\end{aligned}
\]

第一项记录目标运动和 baseline 输运，第二项才是 Sobolev 修正带来的变化。纯 `a_s=(D_q+λI)^−1δ_s` 只使第二项非正；第一项可能占主导。若 Φ 显含 s，还必须在 r 中增加 `E_p∂sΦ−E_q∂sΦ`。原 t 方向、velocity 与 clean correction 的单位也必须统一：当前 wrapper 的 clean 修正 c 对应 dataward `u=c/t`，不能把上述 u 原样加在 clean 输出上。

可进一步区分两个**研究对象**：解 `D_qa=δ` 是追赶当前缺口；解 `D_qa=r` 是匹配当前目标矩速度。后一条有一个无需手工反馈率的精确条件结构。允许 Φ 显含时间，用本文的 `d×m` Jacobian 约定，定义

\[
r_s=\mathbb E_{\rm real}[\partial_s\Phi_s(Z_s)+J_s(Z_s)^TU]
-\mathbb E_{q_s}[\partial_s\Phi_s(Z)+J_s(Z)^Tb_s(Z)].
\]

考虑当前 q 上的有限矩约束问题

\[
\min_{u\in L^2(q_s)}\tfrac12\mathbb E_{q_s}\|u\|^2
\quad\text{s.t.}\quad\mathbb E_{q_s}[J_s^Tu]=r_s.
\]

若 `r_s∈Range(D_q)`，其唯一最小范数向量场为

\[
u_s(z)=J_s(z)D_{q_s}^{+}r_s,\qquad \partial_s\delta_s=0.
\]

证明只需 L²(q) 中的正交分解：约束的 Riesz 表示向量是各 `∇Φ_j`；与它们正交的 u 分量不改变约束却增加能量，因此最优场在它们张成的空间内，系数由 Gram 伪逆给出。零空间可能造成系数不唯一，但不造成 L²(q) 向量场不唯一。只要上述条件沿反馈后的实际 q 持续成立、连续流存在且交换求导合法，从 `δ_0=0` 开始便始终保持这些矩匹配；这不要求基模型 exact score。

**与旧方法的准确交点：** 仅当 `q_s=p_s` 时，显含时间项相消且 `r_s=E_p[J_sᵀ(U−b)]`，Gram 也回到 `D_p`。离轨以后，其生成侧使用实际 q 的 b 和 J，因而与旧真实 bridge 残差不同。若 r 不在 Gram 值域中，则连这些矩导数也无法完全匹配；使用正则化会留下 `λ(D_q+λI)^−1r` 的导数残差。有限样本、退化值域和当前 q 刷新都不能省略。

精确矩导数匹配不会主动消除已有初始或离散偏移，因为此时 `δdot=0`。将它与缺口追赶项相加需要额外反馈率和新推导，论文没有给出一个适用于 RAE 的唯一系数。这是本文独立的条件性扩展，不是 Sobolev 原文已证明的 diffusion 算法，也不据此生成 gate、调度或训练候选。

采用这一结构前，至少需要以下具体证据链：

1. **确认对象。** 在实际官方 IG rollout 的 `q_s` 上，预先固定、类别条件一致的一组可微 Φ 显示独立可复现的非零 `δ_s`；不是生成终点重新加噪的 `bar_q_s`，也不是不说明映射的 predicted-clean 分布。图像/类别/噪声分组和时间配对保持完整。
2. **确认是应纠正的误差。** 找到同一特征缺口与明确生成缺陷的联系，例如类内结构/覆盖缺失，并以固定方向的有限轨迹干预检查修复是否改善独立的终点质量证据。只有 AUC、训练 MMD 或残差下降不足够；不能先扫特征与时间再只展示有利部分。
3. **确认方向可达。** 缺口在 Jacobian Gram 的可靠非零谱空间中；独立当前样本上的 `δᵀDâ` 为正且有实际量级，同时估计 `δᵀr`。若前者很小、后者抵消，便否定“该特征族提供有效误差方向”，不用扩大网络掩盖它。
4. **确认真实有限步与成本。** 使用事前固定的有限采样规则、包含状态反馈的完整 rollout，先验证方向和离散误差，再以相同噪声/类别和总成本做配对质量筛查、独立确认。最终仍须达到公平成本下至少 5% FID 改善；MMD 不是替代目标。

如果最终把向量场投影到同 class 的 `F−B`，必须重新推导和测量投影后的方向；一般 `u=Ja_*` 不等于 `g(F−B)`，原来的负导数不会自动保留。当前 F/B 同类条件有利于类别一致的两样本比较，但既不认证 score，也不认证原生差分包含所需输运方向。

## 旧 AUC 证据的精确范围

本次核对了[实验代码](../experiments/run_raev2_distribution_auc.py)与原始运行归档。当前源码 `SamplerStateRecorder.__call__` 在调用模型前保存 sampler 输入 x；`_branch_states` 用官方 sample_fn 从同一初始噪声推进；真实分支用 encoder latent 构造 `(1−actual_t)*clean+actual_t*noise`；`score_states` 直接对状态 `flatten(1)` 后作线性打分。历史 commit `c99c2d6` 的 v1 对应代码为第 472、834、366 行。故代码对象是实际 rollout 状态，对照为真实 bridge；不涉及生成终点再加噪、预测 clean 或 decoder 特征。

归档根目录 `/home/zhoushunyu/data/eqvae/experiments/raev2_distribution_auc` 下，两份 `n5000_seed20260801_v1/manifest.json`、`n5000_seed20260802_v1/manifest.json` 记录每次 5000 样本，800/200 类拆分为 4000/1000 配对训练/验证样本，分类器为全 latent diagonal-LDA；对应 `auc_results.csv` 在 `t=0.1983471074` 给出 full 的 AUC `0.658191/0.679597`、IG 的 `0.775077/0.790383`。v1 manifest 没有运行源码 SHA，因此本次核对不声称历史执行代码已有逐字节来源认证。

这支持**该指定探针的可分辨性增加**；不等于任意分布距离都更大，也不证明该偏差有益的因果机制。已记录的 IG 图像 FID 收益来自另一[配对尺度研究](RAEV2_IG_SCALE_RESPONSE_RESULTS_ZH.md)，不能由这两个观察直接推导“偏移导致收益”。因此 AUC 不能单独批准或否定 Sobolev；它只提醒需要上述第二项误差性质证据。

## 可计算性与总成本

对 M 个当前粒子，`D̂=(1/M)ΣJ_iᵀJ_i` 的秩至多 `min(m,Md)`。显式存所有 Jacobian 成本为 `O(Mdm)`，直接 Gram 乘法为 `O(Mdm²)`，小矩阵分解为 `O(m³)`。可以逐粒子累积，或用 `D̂v=(1/M)ΣJ_iᵀ(J_iv)` 做矩阵无关求解：先对 `vᵀΦ` 求输入梯度，再做 Φ 的方向导数。必须把方向向量按线性算子定义处理，避免误把额外 Hessian 项算入 Gram。

这些表达说明可计算，不说明便宜。若 Φ 包含 frozen backbone/decoder，输入 Jacobian 必须穿过整条链；冻结参数不允许随意 detach。推理需付势函数输入梯度、当前 cohort 统计/同步和求解开销；在线更新 critic 还需计入训练成本。离线固定 critic 或蒸馏可以降低部署成本，但必须重新检查当前 q 的一致性和实际下降余量。

λ 控制条件数与有限样本误差，需要事前可说明的数值/统计规则；它不是论文自动决定的 guidance 强度。本文不提供手工时间窗、schedule、以 FID 调 λ 或选图。所有粒子在采样轨迹内获得连续状态更新并全部输出；原文的粒子输运不需要终端拒绝、重采样选择或 best-of-N。低维解析特征虽便宜，如果只重复旧径向/均值修正，仍缺少采用理由。

当前可交付的是这一结构与证据要求。它扩展了理论来源和可测误差对象，尚未解决“哪些 RAE 偏差值得纠正”以及公平成本下 5% 质量收益两项关键问题。
