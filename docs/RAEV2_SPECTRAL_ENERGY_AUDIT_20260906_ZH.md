# RAEv2 固定空间 DC/AC 二阶能量审计（2026-09-06）

**历史 IG 的总体能量接近 real，掩盖了空间 DC 缺少与 AC 过量。** 在两套完整 5K 中，IG 的总二阶能量/real 分别为 **1.00136、0.99664**，但 DC 能量比为 **0.94074、0.93276**，AC 能量比为 **1.05829、1.05678**。两个分量的误差方向相反，总量相互抵消。相对 real 的完整 1024 通道残差向量在两 bank 中也同向；排除两 bank 重叠的 21 个源图身份后，描述性结果基本不变。

这是一个可供后续理论解释的分布事实，**不是协方差估计、本征谱、完整频率谱或质量改进证据**。本次没有设计或启动 guidance、拟合 gain、选择通道、调整采样、训练或计算 FID。

计算前已冻结 [request.json](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/spectral_energy_audit_v1/request.json)，SHA256 为 `d7a4860fe131a5075ccfc3ec9a9294747359f3b6af56f4223e54b69c66a03f17`。输入沿用 [feasibility](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/spectral_covariance_feasibility_v1.json) 中两套 `n5000_seed{20260801,20260802}_scales7_v1/latents/`，仅使用 `real`、`scale_s1p000000`、`scale_s1p780000`。每臂四个 FP16 `[1250,1024,16,16]` shard，按 `id=rank+4×local_row` 还原，标签为 `id%1000`，全部 1000 类每类五张。

这里 real 是原图的归一化 encoder latent `E(x)`；历史 scale1 是原 runner 的 `B+1*(F−B)` 分支，不能保证 BF16 算术中逐位等于纯 Full F。历史采用 B4、TF32 开启、CPU rank-seeded noise、BF16 autocast，终点再存为 FP16；real encoder 也在 BF16 autocast 下运行。这些历史值不能与当前 TF32 关闭、不同 noise bank 或 FP32 encoder reference 视作相同协议。

对每张 latent 的每个通道，用全部 `16×16` 空间位置定义

\[
\mu_c=\frac1{256}\sum_{h,w}X_{c,h,w},\qquad
D_c=\mu_c^2,\qquad
A_c=\frac1{256}\sum_{h,w}(X_{c,h,w}-\mu_c)^2.
\]

对应固定正交投影 `P=11ᵀ/256` 与 `I−P`；`D_c+A_c=mean_HW(X_c²)`。**AC 包含全部非 DC 空间分量，不能简称为高频。** 这里仅在每张图内部做空间分解，没有减去数据集均值或类别均值；真实均值可以非零，因此这些量是原始二阶能量，不是去中心协方差。

全部 FP16 原值提升至 FP64，空间均值和两遍 AC 归约均用 FP64。每个视图先计算每类、每通道的逐图能量均值，再对全部 1000 类等权平均。记该全局通道向量为 `E_bank,arm,component`，残差固定为 `R=E_generated−E_real`。跨 bank cosine 使用全部 1024 维原始残差，不归一化或挑选通道；符号一致率为 `mean(sign(R_A)==sign(R_B))`。两个零会计为一致，但本次所有这些残差向量均没有零坐标。

主分析完整保留两侧各 5000 个 ID。唯一事先指定的身份敏感性视图，是把重叠的 21 个 `source_rows` 对应 ID 从两侧 real 及各自两个生成分支同时排除；各 bank 剩余 4979 张，其中 21 类各四张、979 类各五张，仍保留全部类别并以各类剩余 `n_c` 算均值。这是统计去重，**不是生成后选图**。没有根据图像、能量或结果排除样本。

下表是各分量跨 1024 通道的能量总和相对 real 的比值。总能量是 DC+AC 原始二阶能量，不是总体方差 trace。

| 视图 | seed | 分支 | DC / real | AC / real | (DC+AC) / real |
|---|---:|---|---:|---:|---:|
| 完整 5K | 20260801 | 历史 scale1 | 1.01373 | 0.92729 | 0.96915 |
| 完整 5K | 20260801 | IG 1.78 | 0.94074 | 1.05829 | 1.00136 |
| 完整 5K | 20260802 | 历史 scale1 | 1.00878 | 0.92601 | 0.96615 |
| 完整 5K | 20260802 | IG 1.78 | 0.93276 | 1.05678 | 0.99664 |
| 去重身份 | 20260801 | 历史 scale1 | 1.01354 | 0.92719 | 0.96902 |
| 去重身份 | 20260801 | IG 1.78 | 0.94061 | 1.05825 | 1.00127 |
| 去重身份 | 20260802 | 历史 scale1 | 1.00892 | 0.92604 | 0.96624 |
| 去重身份 | 20260802 | IG 1.78 | 0.93282 | 1.05686 | 0.99669 |

完整残差向量的跨 bank 描述如下。Cosine 对残差幅度敏感，不能单独解释为每个通道都同向，因此同时保留无阈值的符号一致率。

| 视图 | 分支 | 分量 | 残差 cosine | 符号一致率 |
|---|---|---|---:|---:|
| 完整 5K | 历史 scale1 | DC | 0.579420 | 75.0000% |
| 完整 5K | 历史 scale1 | AC | 0.996903 | 99.9023% |
| 完整 5K | IG 1.78 | DC | 0.874470 | 94.1406% |
| 完整 5K | IG 1.78 | AC | 0.997536 | 99.3164% |
| 去重身份 | 历史 scale1 | DC | 0.580529 | 75.0977% |
| 去重身份 | 历史 scale1 | AC | 0.996913 | 99.9023% |
| 去重身份 | IG 1.78 | DC | 0.874235 | 94.3359% |
| 去重身份 | IG 1.78 | AC | 0.997545 | 99.2188% |

完整主视图中，IG 的 DC 残差在两个 bank 的 **962/1024** 个通道上都为负；AC 残差在 **842/1024** 个通道上都为正、175 个上都为负。去重后对应 DC 双负为 964 个，AC 双正 841 个、双负 175 个。这里报告全部预设通道的符号计数，没有据此挑出待干预通道。

这组结果说明：仅用总体能量接近来判断 latent 分布已经校准，会漏掉固定空间分量中可重复的误差结构。它尚未说明这些差异来自怎样的网络误差、何种 guidance 动力学，或修正它们是否改善解码后的分布。尤其不能从二阶能量直接推出一个正确的 gain、状态反馈或终点变换。两个 bank 均被历史研究使用过，去重只消除了已知共享源图，**不能证明严格统计独立**；此次不提供 CI、显著性或独立确认结论。

全部产物在 [spectral_energy_audit_v1](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/spectral_energy_audit_v1)。[summary.json](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/spectral_energy_audit_v1/summary.json) 保存全部比值、符号计数、向量范数、24 个原始文件完整 SHA 与耗时；[seed20260801_energies.npz](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/spectral_energy_audit_v1/seed20260801_energies.npz) 和 [seed20260802_energies.npz](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/spectral_energy_audit_v1/seed20260802_energies.npz) 均保留：

- `class_energy[view,arm,class,component,channel]` 与全部逐类生成-minus-real `class_residual`；
- `global_equal_class_energy`、`global_equal_class_residual`，以及轴名称与全部 1024 通道编号；
- 全部原始 IDs、labels、source rows、去重 keep mask 和两个视图的逐类样本数。

没有构造大协方差矩阵。一次顺序读取 **15,728,640,000 字节（14.6484 GiB）** 原始 FP16 payload，同时计算完整 NPY SHA，未再遍历一轮大库哈希。全部 **30,720,000 个 image-channel** 的 `DC+AC=mean(X²)` 恒等式通过，最大绝对/相对误差为 **2.49e−14 / 8.68e−15**。冻结耗时 0.015 秒；主计算 **55.054 秒 wall、54.120 秒 CPU**。模型调用、GPU 时间均为零。

另一位代理独立复核了冻结协议和保存的统计产物，见 [spectral_energy_stats_review_v1/review.json](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/spectral_energy_stats_review_v1/review.json)。用逐类 Kahan 求和和 Python `fsum` 重算全局均值、cosine 与总量比，并检查身份、逐类 residual、去重计数和未受影响的 979 类不变：**94 项核对、28,455,008 个值**通过，最大差 **2.27e−13**，约 1.509 秒 CPU。复核读取约 328 MB 统计产物，未重读原始 latent；因此它是独立的协议与汇总复核，**不是独立重新提取或独立样本确认**。
