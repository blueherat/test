# 真实图反演先验：有界验证协议

2026-09-13。前一项 inverse-copy refiner 的四个后处理臂没有 FID 收益。当前候选使用此前理论留下的另一项必要条件：真实图能够重建之外，反演后的整体编码分布也要与部署时的初始分布相容。**它新增真实数据密度拟合，不把任意往返位移重新命名为零编辑错误，也不是多模态复制操作的等价实现。**

用户本轮追加：先完成手上工作；若没有效果，从 Z-Sampling 出发优化。因此本候选只做以下有界筛查，无收益就转向 Z-Sampling，不再扩展终端修复器、公共参考或高斯先验参数搜索。

## 构造与依据

固定 G 为 S800、CFG extra=1.25、cutoff=.75、64 步 Heun 的实际生成映射，物理时间 0→1。通过逐步求解这个离散映射的逆获得 E；不能将较细 ODE 反演直接视为 G 的精确逆。FP32、关闭 TF32，两 Heun stage 都采用该区间正向左端点的 guidance 开关。

若 G 真正可逆，Q_c=E#P_c，则 G#Q_c=P_c。对同一个 G 的两个候选先验 q1/q2，换元式中的 Jacobian 项抵消：

\[
-\log p_{G\#q_1}(x)+\log p_{G\#q_2}(x)
=-\log q_1(E(x))+\log q_2(E(x)).
\]

所以 heldout 真实逆编码上的 NLL 差具有生成密度意义，无需额外算散度；不能跨不同 CFG 强度、不同 G 使用此抵消，也不能从 NLL 改善推出 FID 改善。当前 E 为数值近似，预检仅提供其精度证据，不证明全局可逆性或精确密度等式。

[Inverse Noise Correction](https://arxiv.org/html/2510.02692v3#S4) 已覆盖真实数据逆噪声再学习的基本原理。本次是低容量高斯适配，部署增加零次 SiT 查询，不能声称发明逆噪声学习；Gaussian ex-post density 也有更早先例。[原文边界](research/identity_operator_20260913/inverse_prior_gaussian_screen.md)。

## 预先固定的数据与拟合

仅从 ImageNet100 **训练**缓存固定 3000 张原图，100 类各 30 张；每类 20 fit、10 holdout，以 source ID 分隔。seed=2026091471。每图 VAE posterior 采样一次后冻结，无反演中新增噪声；不加载 FID validation 图做拟合。

拟合每类均值收缩到全局均值：μcκ=μglobal+κ(μraw,c−μglobal)。给定这些均值，以所有 fit source 的残差平方平均拟合共享的逐坐标方差。κ∈{0,.25,.5,1}。从标准高斯到该 Gaussian 的仿射插值为

\[
A_{\rho,\kappa}(\epsilon,c)
=\rho\mu_{c,\kappa}
+[1+\rho(\sigma_\kappa-1)]\odot\epsilon,
\quad \rho\in\{0,.25,.5,.75,1\}.
\]

20 个名义组合中 ρ=0 去重为同一个恒等，总共 17 个独立候选。只按 1000 heldout source 的 Gaussian NLL 选择；共享逐坐标方差、每类均值仍是低阶近似，均值收缩用于限制每类 20 图下的过拟合。独立统计单位是原图，不是 4096 个坐标。

保留全局 diagonal 子族 κ=0，以及零均值 isotropic 方差作为控制；后者区分有益结构与简单 noise temperature。选择也包含恒等，并去重实际相同的赢家。若全部选择恒等，则无需重复采样来确认完全相同的算法。

## 反演与生成评估

先在 8 个预定类别检查原生 CFG64 前向逐位一致、已知噪声前像恢复，以及真实图重建。逐步 inverse Picard 上限 16、残差 RMS 阈值 1e−5；达到迭代上限仍未通过不得静默纳入 bank。预检合格后才构造其余逆编码，保存每批成本、残差、哈希；失败保留为失败，不能只统计成功图。

通过拟合后，冻结 prior，在新 balanced 1K seed=2026091473 上运行 CFG+prior，并以相同精度重跑 CFG/APG。CFG 复查 extra=1/1.25/1.5，APG 复查 extra=1.5/2/2.5，均 64 Heun、224 次分支调用/图；既有大范围调参结果作为这些中心值的依据。候选只校正初始噪声，所有额外仿射计算计入时间，离线反演和拟合成本另列。

FID 不用于回调 prior κ、ρ 或增加新拟合族。若仅优于 CFG、仍输给 APG 或只与 isotropic 控制相当，不称为完成目标。明确有竞争力的结果才进入新的独立 5K；没有质量证据则结束当前候选并开始用户授权的 Z-Sampling 方向。

源码：[反演 bank](../experiments/cfg_inverse_prior_20260913/bank.py)、[拟合](../experiments/cfg_inverse_prior_20260913/fit.py)、[采样接口](../experiments/cfg_inverse_prior_20260913/sampler.py)、[成对评估入口](../experiments/cfg_inverse_prior_20260913/run.py)。

## 实际执行结果（2026-09-13）

普通 Picard 预检失败后，只增加一次 Anderson 数值修复；同8例预检及收紧检查通过，但扩大真实数据时 row142 再次未收敛，随即停止剩余分片。共保存1360/3000个有效源，未形成完整 bank、未做真实数据 NLL 选择或 FID 评估；不能将求解器覆盖失败写成方法质量被否定。全部记录成本为1,016,512次分支图像查询，包含失败与中断。

完整计数、数值误差和停止边界见[实际结果报告](CFG_INVERSE_PRIOR_RESULTS_20260913_ZH.md)。本候选至此结束，后续转向用户已授权的 Z-Sampling。
