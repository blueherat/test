# 两个空间子空间上的 Gaussian 密度比 guidance

2026-09-07，下一候选的固定推导；尚未运行 FID。动机来自仓库已完成的 [两个 5K bank 的能量审计](RAEV2_SPECTRAL_ENERGY_AUDIT_20260906_ZH.md)：IG 的空间平均（DC）二阶能量偏低、其正交补（AC）偏高，两者相消让总能量看似正确。AC 是全部非恒定空间分量，不能简称高频。本设计不是浅层/深层频率对应，也不从 FID 学分频系数。

令 P 为每通道 16×16 空间上的常量正交投影。仅保留两个统计约束 E||PX||²、E||(I−P)X||²；最大熵分布是零均值 Gaussian，两个子空间的各坐标方差记 λ_DC、λ_AC。真实数据与官方生成各有一组 λ，均由已存在的样本直接求矩，不拟合时间曲线：

    λ_DC = 256 E[spatial_mean(X)²]
    λ_AC = 256/255 E[(X−spatial_mean(X))²].

这里是原始二阶矩约束下的零均值近似，不冒充已去均值的 covariance。真实训练 bank 为5000图；生成 bank 是旧 seed20260801/20260802、各5000图、固定 IG1.78/interval[.1,1]/100steps/EMA，仅读取这一预定分支，不读取其他 scale 的 FID 或用它们估计 λ。旧生成 B4、不同噪声及 FP16终点缓存与当前B8的差异保留，不能声称它精确等于当前实际 q₀；两个旧 seed 已在历史分析使用，不作为新独立确认。

在 z=aX+tε、a=1−t 下，两种 Gaussian 的边缘方差为 Vp=a²λp+t² 和 Vq=a²λq+t²。它们的 score 差为 (1/Vq−1/Vp)z。等价的 clean posterior mean 差有无奇异公式：

    c(t;λp,λq) z = a t²(λp−λq)/(Vp Vq) z.

新 clean 预测定义为

    G_new = G_native + c_DC(t) Pz + c_AC(t) (I−P)z.

若原模型恰为 Gaussian q 的 Bayes denoiser，这一修正逐点把它变成 Gaussian p 的 Bayes denoiser；不需要原场可积性推断。实际 RAEv2 的 conditional、多模态、有限 IG 显然不满足这一 Gaussian 身份，所以迁移是显式 surrogate correction，质量需评测。它并不把 oracle 恒等式包装成 RAEv2 的 FID 定理。[Discriminator Guidance](https://proceedings.mlr.press/v202/kim23i.html) 提供密度比修正的相关机制；这里的两个 Gaussian 是有意受限的矩模型，不使用其一般正确 score 假设证明本地性能。

此设计的四个 λ 是两组样本矩，唯一强度固定为1，没有可搜索的几十段参数。整个时间形状由线性 Gaussian channel 决定，在 t=1 和t=0处修正均为0。DC的大方差自动把影响放到较高噪声阶段，AC的影响较晚；不手工选窗口。只需每张图的空间均值和逐元素运算，无新模型调用。

预定先通过 Gaussian posterior replacement 与完整小矩阵 score→clean 的独立测试，再8图数值检查、同一 paired seed202609071 1K。若方向有希望则冻结转入独立5K；若失败，不调两个子空间强度或把 AC 继续切成频段。本候选尚未进入当前正在运行的 S² 5K，二者实验保持各自冻结身份。
