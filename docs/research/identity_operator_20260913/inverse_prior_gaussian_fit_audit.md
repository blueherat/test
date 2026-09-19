# 固定 CFG 的高斯逆先验：数学与 CPU 实现审查

2026-09-13。审查 fit.py、sampler.py、run.py 及其直接调用的 baseline / runner；真实反演 bank 的完整审核由另一审阅者承担。本次只运行 CPU 合成数据，没有读取或修改真实实验输出。接续[有界验证协议](../../CFG_INVERSE_PRIOR_SCREEN_20260913_ZH.md)与[原文审计](inverse_prior_gaussian_screen.md)。

## 1. 数据目标成立，但不是新的 no-op 算子

冻结类别 c 的实际生成映射 G_c，包括 checkpoint、CFG、cutoff、Heun64 和精度规则。若 G_c 可逆且 E_c=G_c^{-1}，真实数据逆分布 Q_c=(E_c)#P_c 满足 G_c#Q_c=P_c。自重建 G_c E_c(x)=x 本身仍然是恒等契约；真正额外的监督来自独立真实图像的**聚合逆分布**，不是把自回环误差改名为质量。

对任意候选先验 q，若相应密度存在且 KL 有定义，

\[
D_{KL}(P_c\Vert G_c\#q_c)=D_{KL}(Q_c\Vert q_c).
\]

因此同一固定 G 下两个候选的 image-latent NLL 差，等于它们在 E_c(x) 上的 NLL 差，Jacobian 抵消。此结论不允许跨不同 CFG 强度比较，不是有损 VAE 解码后 RGB 的精确似然，也不证明更低 FID。[INC §4](https://arxiv.org/html/2510.02692v3#S4) 已覆盖逆噪声学习的核心原则。

数值 Picard 回收实际 Heun 步提供局部近似逆的证据，不证明 G 全局可逆。令部署 Gaussian 为 N(μ,D)，相对原 N(0,I) 的 log-density ratio 为 ℓ。编码误差 δ 的影响恰为

\[
\ell(z+\delta)-\ell(z)
=\delta^\top[(I-D^{-1})z+D^{-1}\mu]
+\tfrac12\delta^\top(I-D^{-1})\delta.
\]

所以小重建 MSE 单独不够认证密度选择准确；逆编码的精化稳定性更直接。全局 Lipschitz 常数 L 若存在，近似逆还满足

\[
W_2(G\#q,P)\le L W_2(q,E\#P)
+\sqrt{\mathbb E_P\|G(E(X))-X\|^2},
\]

但当前实验没有可用的全局 L，也不能据此报告质量保证。

## 2. 当前拟合的准确含义

2000 fit、1000 selection holdout，100 类分别 20 / 10 张。给定 κ，μ_cκ=μ_global+κ(μ_raw,c−μ_global)，共享逐坐标方差 N^{-1}Σ_i(z_i−μ_{c_i,κ})² 是**给定这些均值**的 Gaussian MLE。总体 Gaussian cross entropy 对应 forward-KL 目标；经验原子分布对连续 Gaussian 的 KL 本身不可当作有限训练目标，代码实际计算的是样本 NLL。

固定协方差时，均值收缩可由向全局均值的 Mahalanobis ridge 惩罚推得 κ=n/(n+λ)；n=20 时 κ=.25/.5/1 对应 λ=60/20/0。随后重新估计方差的整个流程，不应直接称为这一惩罚下的联合 MLE。每类均值仍有 4096 坐标、只有 20 张 fit 图，是低阶统计族而非少参数模型，必须靠收缩与独立留出选择限制过拟合。

A(ε,c)=ρμ_cκ+[1+ρ(σ_κ−1)]⊙ε 是从标准 Gaussian 到拟合 Gaussian 的精确 W2 位移测地线；ρ 路径的每一点不是 Gaussian KL 最优解。κ、ρ 只在预先固定网格上按留出 NLL 选择，ρ=0 保留原采样器。零均值 isotropic MLE σ²=(ND)^{-1}Σ_i||z_i||² 是必要的 noise-temperature 对照。

留出数据用于选择，所以赢家的留出改进带有选择偏差；每个 source 的 NLL 差可以配对，不能把 4096 坐标当独立图计算标准误。NLL 改善只能说明此低阶 prior 与逆编码更相容；能否超过调优 CFG、APG 和 isotropic 对照仍需独立生成实验。

## 3. 实现裁决与实际验证

未发现拟合、采样方向、时间约定或 FID 反馈拟合错误。候选参数只读取 fit 子集；留出集只打分和选择。采样只将原 Gaussian 作一次仿射，然后调用同一个 CFG64，保持 224 次分支调用。configs、prior 文件和相关源码在 runner 中冻结；sampler 额外验证 prior SHA 与绑定的 CFG 配置。新增 CFG / APG 邻近强度只作为生成对照，没有跨 G 迁移 prior 或比较逆噪声似然。

唯一修正是部署去重：原实现只按 grid index 合并赢家，现在按实际 FP32 means/std 合并，包括不同 index / family 变成同一仿射，以及转换到 FP32 后实际回到 identity 的情形。全部 NLL 网格行与选择元数据保留，不重复生成相同映射。

CPU 结果：3000×4096 Gaussian、只改变 holdout、严格 identity、跨 family 相同非恒等映射、FP32 舍入后 identity，共五个 fixture。拟合参数、NLL、按 source 计算的 SE 与独立计算最大误差均为 0；改变 holdout 不改变任何候选参数。实际 baseline Heun64 配 CPU 假场时，零仿射与 native 逐位相同，非平凡仿射与直接传入变换噪声也逐位相同，均 224 calls。错误 prior 哈希、错误绑定强度和覆盖已有 fit 均被拒绝。

复现：OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python /tmp/cfg_inverse_prior_audit_20260913/audit.py。

结果：[audit_results.json](/tmp/cfg_inverse_prior_audit_20260913/fixtures_9cewebp_/audit_results.json)。修正后 fit.py SHA256：4dc99ec204f6eda26125397f089f91fe79df77174a34a76944dc62d258f7dea0。sampler.py 与 run.py 未修改。CPU 假场只验证实现语义，真实模型数值逆与实际图像收益仍由当前预检和生成实验判定。
