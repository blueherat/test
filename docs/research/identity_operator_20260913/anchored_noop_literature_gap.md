# 原图锚定的反演保真：三篇最接近的方法与尚缺的迁移

2026-09-13。有界原始文献核查；未运行 GPU、实验或修改代码。先读了 [iCD 等旧核查](../cfg_inversion_20260913/shared_inversion_calibration.md)、[IGN/SIGN 与 FM 适配](fm_fixedpoint_construction.md)、[RF 旧核查](../cfg_inversion_20260913/rf_boundary_quality_followup.md)及[修复器实际结果](inverse_copy_refiner_results_audit.md)，不重新扩展综述。

**结论：确实存在无需训练、把原图持续作为参考的反演保真操作；但下面三篇都没有提供“仅凭复制稳定性，改善无外部图像条件的 CFG 生成”的论证或实证。** 最接近的新证据是 Direct Inversion 的逐步参考误差补偿，以及 ProxEdit 的 inversion-trajectory guidance。它们解决“如何保留这张图”，并未解决“什么变化会让自产图更好”。RF-Inversion 另有随机采样器，但那不是由逐图 no-op 误差导出的质量改进。

## 1. RF-Inversion：原图作为解析终端条件

[Semantic Image Inversion and Editing using Rectified Stochastic Differential Equations](https://arxiv.org/html/2410.10792v1#S3.SS5)，首版 2024-10-14，ICLR 2025。关键位置为 §3.3 Eq.8、§3.5 Eq.15–17、Algorithm 2。

统一生成时间 t=0 噪声、t=1 图像。源 latent x 始终保留；反演时另固定 Gaussian 端点 ξ。用反演时钟 s，机制为

\[
\frac{dy_s}{ds}=(1-\gamma)[-v(y_s,1-s)]+\gamma\frac{\xi-y_s}{1-s},\quad y_0=x;
\qquad
\frac{dz_t}{dt}=(1-\eta)v(z_t,t,c)+\eta\frac{x-z_t}{1-t},\quad z_0=y_1.
\]

原图通过第二式持续进入采样器，不需要模型原生图像 tokens。**η=1 的端点恒等来自解析控制器，而不是生成模型能力**；其端点取极限，实践有截断，有限终端罚项的解还含 1/λ 分母修正。γ=η=0 则回到同场精确逆的恒等。非零中间值是保真与编辑之间的取舍；原算法反演还指定 ξ，不能称为全程无新噪声。

论文 §3.5 / Appendix A.4 确实另给 η=0 的 RF 随机采样器，并有 Figure 26 定性对照。其理论对象是相应 RF 的边缘演化，依赖模型/score 与路径的相容性；它不是从 no-op 得到更优数据分布，也不证明任意高 CFG 场可照搬后更好。源图控制与这个随机采样支线须分开。

## 2. Direct Inversion：保留整条原图轨迹，补偿 source 分支

[Direct Inversion: Boosting Diffusion-based Editing with 3 Lines of Code](https://arxiv.org/html/2310.01506v2#S4.SS2)，首版 2023-10-02，核对 v2 2023-10-19；§4.2、Algorithm 1。

设 r_k 是已存的原图 DDIM inversion 轨迹，Ψ_k 是朝图像方向的一步更新。其 source 纠偏可概括为

\[
o_{k+1}=r_{k+1}-\Psi_k(r_k),\qquad
z^{\rm src}_{k+1}=\Psi_k(z^{\rm src}_k)+o_{k+1}.
\]

若 source 状态从 r_k 开始且用相同更新算子，补偿后就等于 r_{k+1}；原图信息不是仅存在于初始逆噪声中。论文让 source / target 承担不同职责，**只把 source offset 加回 source 分量，target 不直接加这个纠偏**，再依赖编辑方法的信息交换。不能将 source 的精确轨迹复现误报成“最终 target 图同 prompt 必定逐像素恒等”。其效果评价是图像编辑、结构与背景保持，没有独立 from-prior CFG 改善实验。把参考误差直接加回并不创造一个跨图通用质量方向。

## 3. ProxEdit：向 inversion 轨迹靠拢，不把同场降强度当重建保证

[Improving Tuning-Free Real Image Editing with Proximal Guidance](https://arxiv.org/html/2306.05414v3#S3.SS2)，首版 2023-06-08，核对 v3 2023-07-06；正式出版名为 ProxEdit，WACV 2024。这里采用 v3 的 §3.2、Algorithm 1，避免沿用 v2 仅回拉 predicted-clean 的版本。

先用 source prompt、CFG w=1 反演并保存 z*。目标与 source 的预测差被阈值化；完成一步更新后，在预计无需编辑的 mask M 内执行

\[
\tilde z_{t-1}\leftarrow\tilde z_{t-1}
-\eta M\odot(\tilde z_{t-1}-z^*_{t-1}).
\]

η=1 是 mask 内完整替换。source=target 时，预测差为零只能使引导退化为普通 source DDIM；**这仍不消除离散 inversion/reconstruction 误差**，所以才另引入上述参考轨迹纠偏。论文 Remark 3.1 也明确区分“精确跟踪 DDIM reconstruction 轨迹”和“跟踪原 inversion 轨迹”。这是一种无需拟合参数的原图保真机制，所用参考和 mask 仍属于编辑任务，不是无图像条件 CFG 的新目标。

## 4. 可以移植什么，不能据此启动什么

三篇给出的直接技术收获是：**反演得到初态之后，不必丢掉原图；可以把原图或它的轨迹作为贯穿重建的参考。** 当前 SiT 的 sampler 层也能表达此接口。以已计算的原图 inverse 轨迹 r(t) 为例，在正确物理时间推进一步后做

\[
a_{k+1}=\Psi^{\rm CFG}_k(z_k),\qquad
z_{k+1}=a_{k+1}-\rho M_k\odot(a_{k+1}-r(t_{k+1})).
\]

这是上述已知参考跟踪机制的 FM 记法，**不是新采样建议或新实验候选**。ρ 是状态参考的回拉系数，非 CFG 权重；纠偏量是 latent 单位。ρ=1、M=1 时终点直接等于所存原图，强弱模型都能通过，不能拿这种成功给生成能力排名。只在部分时间/区域回拉，则保真仍需实测。

若每轮重新反演上一轮图片，并把那张图作为本轮锚点，这可以定义真实的重复 image→image 操作；但当前图片有伪影时，精确保真也会保留伪影。若去掉原图锚点，就失去这三篇提供的保真依据；若用自己的输出代替锚点，尚无外部纠错目标；若改用公共模型输出，又回到已发现的参考偏差。上述三篇没有补上这个缺口。

因此本轮**不支持再次扩大 inverse-copy refiner 或采样扫描**。现有反演工具足以构造、检查“保持当前图片”，却没有给出免训练的、数据选择性的“保留正确内容并纠正自产错误”原则。新的 CFG 改善声明至少还需要独立于回环自洽的目标依据，以及独立生成评估；修复器这次四个 FID 均未改善，也不能用重建微降替代这一证据。
