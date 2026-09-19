# 真实逆噪声的高斯先验：INC 覆盖范围与一个有界筛查

2026-09-13。只读文献与仓库；未运行 GPU、实验或修改代码。接续[旧 INC / RAE 审计](../cfg_transport_search_20260913/prior_and_hybrid_audit.md)，不重启 RAE/PFR 分支。

**判断：可以做有明确数据目标的低成本拟合筛查，但应称为 INC 思想下的高斯先验校准，不是免数据拟合的新 CFG 原理。** 前提是先获得与实际 tuned CFG64 对应的可靠真实图逆噪声。拟合与部署便宜，不表示离线反演银行便宜；当前结果也尚不能承诺生成收益。

## 1. INC 已覆盖基本原理，未见高斯仿射版本实证

[Fine-Tuning Diffusion Models via Intermediate Distribution Shaping，2510.02692v3](https://arxiv.org/html/2510.02692v3#S4)，ICLR 2026；首版 2025-10-03，v3 2026-03-03。§4 的 INC 对**实际离散生成映射** T 反演真实数据，学习逆噪声分布，再接冻结 T；Lemma 4.2 给出 pushforward 恢复数据及可逆映射下 TV/KL 不变。Euler 离散逆的唯一性条件为 hL<1，不是任意模型、步长都满足。

§5.3 的实际校正器是另训的 16M flow，原模型为 65M、无条件像素空间 flow。相同原模型 200 步下，增加 200 步校正器后 CelebA-HQ FID 13.39→8.02、LSUN-Church 8.63→7.26；推理 FLOPs 从 1.374→1.806×10¹²。不是零成本仿射校正。[Table 5](https://arxiv.org/html/2510.02692v3#S5.SS3)

D.8 Theorem D.2 用逆噪声相对 Gaussian 的 **KL** 控制一个加权积分的 velocity 差，支持学习较简单校正器的动机；它不证明逆分布的误差主要在均值/协方差。G.1 明确每个主模型步数另训校正器，LSUN 还因反演不稳加入 σ=10⁻³ 图像扰动。该扰动不能无说明沿用到本题的原图 no-op 协议。[D.8 / G](https://arxiv.org/html/2510.02692v3#A4.SS8)

本次未在该文找到条件 Gaussian、协方差仿射适配器或当前 SiT CFG64 的实验。因此这可能是压缩实现的实证问题，不能把 inverse-prior learning 本身声称为新贡献。

## 2. 高斯拟合本身也有明确先例

- [Optimizing the Latent Space of Generative Networks，ICML 2018，§4.3](https://proceedings.mlr.press/v80/bojanowski18a/bojanowski18a.pdf)：GLO 对训练得到的图像 latent codes 拟合单个全协方差 Gaussian，再通过生成器采样。其 codes 与生成器联合优化，并非冻结 FM 的 ODE 逆。
- [From Variational to Deterministic Autoencoders，ICLR 2020，§4](https://arxiv.org/html/1903.12436v3#S4)：对真实图 encoder outputs 做 ex-post density estimation，明确比较全协方差 Gaussian 和 10-component GMM，也用于已有 VAE/ WAE。采样时无需重训 decoder。这里的 RAE 是该篇 regularized autoencoder，**不是重启仓库 RAEv2 项目**。

所以“真实图编码→高斯拟合→冻结生成器”不是新原则；本题差别仅在编码器是**同一实际 CFG 离散映射的逆**。

## 3. 为什么这个目标比单独 copy loss 更实在

以下是变量替换公式的直接应用。固定类别 c 与完整实际采样映射 G_c，包括 checkpoint、CFG 强度、cutoff、64 步 Heun 和精度规则。若 G_c 是可微可逆映射，令

\[
Q_c=(G_c^{-1})_\#P_{\rm data,c},\qquad q_\phi(\cdot|c)=\mathcal N(\mu_c,\Sigma_c),
\quad p_\phi=(G_c)_\#q_\phi.
\]

对原图 latent x，z=G_c^{-1}(x)，候选先验与标准 Gaussian ν 的 NLL 差为

\[
\boxed{-\log p_\phi(x|c)+\log p_0(x|c)
=-\log q_\phi(z|c)+\log\nu(z).}
\]

Jacobian 项因 **G_c 固定**而抵消，不必计算散度。对真实逆噪声做留出 Gaussian NLL 检查，因而有对应的生成密度目标；这不是各模型自身往返均为零的空检验。总体 Gaussian 最大似然拟合在包含 ν 的族内不会增加 forward KL，但有限样本拟合没有此保证，NLL 改善也不保证 FID 改善。此等式仅适用于 latent 分布，不是有损 VAE 后 RGB 像素的精确似然；不同 CFG/模型的 G 之间也不能忽略不同 Jacobian 后比较噪声 NLL。

**采样方向是 ε~N(0,I)→μ+Lε→G_c，LLᵀ=Σ；不是把真实逆噪声白化后仍声称学到了 Q。**

## 4. 只建议一个低参数 family 作为快速准入检查

当前 z∈R^(4×1024)。若真实银行仍只有每类几张，不拟合每类 4096 维自由均值和协方差：即使真分布就是 N(0,I)，n_c=4 的经验均值也有 E||μ̂_c||²=4096/4=1024，协方差秩最多为 3，足以制造严重的估计假象。

可预先固定较小的 family：

\[
q_\phi(z|c)=\prod_{u=1}^{1024}\mathcal N(z_{:,u};m_c,\Sigma),\qquad
z_{:,u}=m_c+\operatorname{chol}(\Sigma)\epsilon_{:,u}.
\]

m_c 只有每类 4 个广播 channel 均值，Σ 为共享 4×4 协方差；共 410 个自由参数。它只检验条件通道偏移/相关这一小部分结构。原标准 Gaussian 对应 m=0、Σ=I。若使用向此恒等先验的收缩，收缩量只由预留逆噪声选择，保留恒等为合法结果；不用 FID 扫描收缩参数。该统计族是本次工程建议，并非上述论文实验。

部署仅一次小矩阵仿射变换，**额外 generator NFE=0**，仍完整执行相同 CFG64。离线反演、拟合和最终评估成本另记。每图算完整 log-density 差，再按 source ID 汇总留出均值/不确定性；不能把同图 1024 位置当成 1024 张独立图来给置信区间。若这种受限 family 留出无收益，只否定其可捕获的低阶结构，不否定完整 INC。

## 5. 三项必要限制

1. **模型与离散逆必须配套。** 不能使用先前 S/W/XL 公共逆银行代替 tuned CFG64 的逆；也不能将细网格反向 ODE 直接标为 Heun64 的精确逆。Heun 的一步逆须针对完整两次评估映射求前像，两个 stage 都复用该步 accepted-left 的 CFG/cutoff。先做已知初始 Gaussian 的 G64→inverse 回收，再检查真实图 roundtrip 和逆噪声统计/留出分数随求解精化的稳定性；小 endpoint residual 单独不足以认证逆噪声正确。APG 带历史，不属于这个 stateless CFG 筛查。
2. **按真实原图 source ID 分 fit / selection holdout，排除 FID reference 数据。** 所有裁剪、VAE posterior 样本和 inverse 版本同源归组；VAE 编码规则与原 latent 数据目标一致。若这些图也用于训练 SiT，应说明留出仅针对先验拟合，并非 backbone 未见数据。最终 paired 图像测试用新噪声、冻结类标签；已有 1K 选择 bank 只能筛查，不能当独立确认。
3. **控制记忆与估计偏差。** 不从逆噪声银行逐条抽取代码冒充新生成；不拟合近奇异的每类 full covariance。使用满秩、收缩的低参数连续 prior，检查生成结果与 fit 图的近邻并报告多样性。拟合收益应大于反演精度变化造成的读数变化；若只剩数值差或训练集 NLL 下降，就不扩大生成实验。

**可执行裁决：先验证固定 CFG64 的逆标签，再做这一个低参数 Gaussian 的留出密度检查；只有通过，才值得冻结一个仿射先验跑配对生成。** 这补充的是“正确逆噪声的总体分布”条件，不是把已失败的 terminal copy refiner 换名。当前没有新增 bank、参数拟合或生成结果。
