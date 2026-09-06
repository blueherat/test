# Moser有限密度源：理论成立，当前可估计实现未闭合

本文件完成最后轮次中的第4轮。结论是：**直接指定有限密度差确实避开了“把IG residual covariance全部补回”的错误目标，但现有有限样本/有限场并未因此获得可部署保证。** 保留这一理论入口，不通过临时扩基、正则或gain搜索继续训练；本轮后收束。

## 一手来源与阅读范围

Rozen、Grover、Nickel、Lipman，*Moser Flow: Divergence-based Generative Modeling on Manifolds*，NeurIPS2021。[正式正文](https://proceedings.neurips.cc/paper_files/paper/2021/file/93a27b0bd99bac3e68a440b48aa421ab-Paper.pdf)、[正式补充](https://proceedings.neurips.cc/paper/2021/file/93a27b0bd99bac3e68a440b48aa421ab-Supplemental.pdf)。阅读正文§2–3的假设、定理1/2及其推导，§4–5的实现和实验设定；逐项检查补充A的连续性证明与B的定理2证明。没有重新证明通用逼近定理3，也没有执行作者模型。

论文在紧致、连通、可定向、无边界流形上，以正密度间的混合路径构造flux/密度速度，保证精确流的端点输运。模型用已知source密度减去网络散度表示目标；总体损失通过正部似然和负部惩罚约束密度。λ≥1且正密度下界条件支持总体极小值，有限训练还需ε、网络及积分近似。正文展示曲面/低维密度建模与训练效率，不是ImageNet FID证据；§5.2实际比较λ=1/2/10/100，不能称部署参数由定理唯一决定。

补充A为flux采用与正文相反的符号定义，同时速度也换号，二者相容。定理2证明中的正负部代数须按正文 `μbar=μplus−μminus` 理解；总体论证可由质量归一化直接重建。这里保留正确结论，不把排版符号当作机制失败。

## 从理论导出的新入口及其与旧路线的关系

令q⁻为原生IG一步作用于真实p_t后的分布，目标为p_s。设

\[
\rho_\tau=(1-\tau)q^-+\tau p_s,\qquad
\operatorname{div}j=q^--p_s,\qquad v_\tau=j/\rho_\tau.
\]

连续性方程直接给出准确端点。在满足正则性的理想情况下，校正S作用于原生一步T后有 `S#T#p_t=p_s`，所以对实际输入q_t的KL通过共同映射不放大；可逆时仍只是等号。它与[已有有限输运保证](RAEV2_ACTUAL_ROLLOUT_FINITE_TRANSPORT_20260906_ZH.md)相容。

这里插值的是**密度**，而配对桥插值随机变量Y/W；两条路径一般不同。只要求边缘source，不要求E[W−Y|Y]=0。因此它正确避开第2轮漏掉均值漂移和重复IG方差补偿的问题。但未知高维q⁻的密度值、可微正性和flux解并不由现有样本自动提供；native BF16数值状态也不是光滑密度定理的直接实例。

一个本地推导的、只用两样本的替代为

\[
J_\tau(u)=\tfrac12\mathbb E_{\rho_\tau}\|\nabla u\|^2
 -\{\mathbb E_{p_s}u-\mathbb E_{q^-}u\}.
\]

一阶变分给 `Eρ[∇u·∇h]=E_p h−E_q h`。在适当函数空间、有界逆算子和边界条件下，它识别最小动能梯度场；这是弱Poisson问题，**不是原论文的显式密度似然loss**。有限基u=θ·φ时成为 `A_τθ=b`，`A=Eρ[JφJφᵀ]`、`b=E_pφ−E_qφ`。如果A正定有唯一有限投影，但只对选定测试函数成立。

这与[旧Sobolev研究](RAEV2_GUIDANCE_READING_SOBOLEV_20260906_ZH.md)有实质重叠。新检查的问题是能否通过有限密度路径自动闭合旧实现的缺口；不能把重新解一个Gram或扩大特征说成已获得新的质量机制。

## 固定裁决的结果

[预先协议](RAEV2_MOSER_FINITE_SOURCE_PROTOCOL_20260906_ZH.md)与[CPU脚本](../experiments/audit_raev2_moser_finite_source.py)只使用圆S¹的已知正密度，避免用不符合原假设的分布反驳Moser定理。q均匀，p=(1+0.5cosx)/(2π)。

| 检验 | 准确目标/解析值 | 实测结果 |
|---|---|---|
| 准确Moser速度端点E cosx | 0.25 | 0.249999999999762 |
| 准确Moser速度端点E cos2x | 0 | 2.70e−12 |
| 准确流的不变量最大误差 | 0 | 2.30e−10 |
| 单cos势Galerkin端点E cosx | 实际解tanh(.25)，目标0.25 | 0.244918662404095 |
| 单cos势Galerkin端点E cos2x | 实际解tanh(.25)²，目标0 | 0.059985151192853 |

准确速度为 `−.5sinx/(1+.5τcosx)`，不变量 `x+.5τsinx=x_initial`。单cos势在规定ρ_τ上的Gram恒为.5，source=.25，最优系数恒为.5，故其投影方程**每个τ都精确成立**。但是实际速度变为−.5sinx，它产生的分布不等于规定混合路径，终点新增明显的第二Fourier矩。当前路径上的投影关系不能代替对实际移动分布的反馈。

另一个严格区分总体和经验的例子：仅有源点0、目标点π，同一cos势的经验梯度Gram为0、source为−2，因此 `J(k)=2k` 没有下界。数值记录k=0/−1/−10/−100得到0/−2/−20/−200；数学结论来自无界k，不是由四个点外推。经验Dirac不满足原Moser正密度假设，恰好说明不能将总体定理直接转成任意经验训练保证。一般A半正定时，若b含null(A)分量则同样无界；若b在range(A)，仍需控制可识别方向和泛化误差。

固定1024中点、DOP853容差如协议；准确/投影两流分别89/62次CPU RHS，runner wall0.008691s、CPU0.008689s。它们是解析数值验证，没有RAEv2前向、训练、解码或FID。论文下载/阅读、实现和独立复核成本另记，不与此runner时间混为一谈。

## 最后准入决定

正确的全密度source得到正确的目标，这是保留的结构。失败的是“有限两样本弱目标已自然给出可部署校正”这一实现桥梁。原论文需要已知source密度和正性控制；弱形式需要有依据的函数空间、梯度Gram估计与实际路径误差控制。当前仓库没有补足这些量的新证据。返回既有受限势空间只是旧路线近邻；临时扩基、平滑带宽或正则扫描与本次约束不相容。

因此不新增RAEv2训练/1K，不宣称所有Moser方法不可能，也不将toy端点精确运输当作FID改善。第4轮到此裁决，原公平成本≥5%目标仍未实现。

原文与来源：`/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/reading_moser_v1`；固定数值输出：同根目录 `moser_finite_source_v1`。论文PDF保留外部原址，紧凑manifest与数值证据纳入最终Git归档。
