# RAEv2 噪声端点：零 velocity 是否更准确

**现有缓存可以完成一个无需重造噪声的无偏替代检验；结果强烈反对“当前起始 velocity 比零更差”的解释。** 固定 t=1、全部 1000 类，官方场与零场的每维风险差估计为 **−1.18912893 ± 0.00501374 描述性 class SEM**。这里解析积分了零场风险，不能称为原噪声下的逐图配对差。没有 GPU、噪声生成、模型调用或 FID。

本轮检验由[CFG-Zero⋆ 阅读](RAEV2_GUIDANCE_READING_PROJECTION_ZERO_20260906_ZH.md)提出，只问起始 velocity 的局部精度，不判断后续采样质量或模型整体是否收敛。[原论文 §4.2](https://arxiv.org/html/2503.18886v2#S4.SS2)的 zero-init 动机是欠拟合时预测 velocity 可能比零更差，不是精确初速度必为零。

**边界和现有协议相符。** 已核对原验证 runner、request 和补空间审计缓存：`z=(1−t)X+tε`，step000 的 t 精确为 1；ε 由独立 CUDA Generator、seed `202609094`、B32 的 `randn` 生成。因此在该端点 z=ε，与类内 X 独立。每类恰好一个原验证样本，X 是原 FP32 encoder 产生、随后保存为 FP16 的标准化 latent，读取时转 FP32。维数 `D=1024×16×16=262144`。

G 是 EMA step100080 的 native BF16 `B+1.78(F−B)`，随后转 FP32；t=1 位于官方区间 `[0.1,1]` 内。缓存 residual 是 **X−G，尚未加入势函数校正**。这份旧验证以 B32 运行，不能冒称重新执行或逐位重现现场 B8 sampler；模型身份来自原归档 SHA，未重新载入模型。

以数据方向时间 `s=1−t` 表示，在 t=1 时

\[
b_G=G-z,\qquad b^*=\mu_c-z,
\quad \mu_c=\mathbb E[X\mid c].
\]

零 velocity 对应的 clean prediction 是 z，**不是 clean prediction 为零**。目标风险差为

\[
\Delta_0=\frac1D\mathbb E\big[\|G-\mu_c\|^2-\|z-\mu_c\|^2\big].
\]

由于 `G(z,c)` 与 X 条件独立、ε 为标准 Gaussian，

\[
\mathbb E_\epsilon\frac{\|\epsilon-X\|^2}{D}
=1+\frac{\|X\|^2}{D},
\qquad
\Delta_0=\mathbb E\left[r^2-1-\frac{\|X\|^2}{D}\right],
\quad r^2=\frac{\|G-X\|^2}{D}.
\]

两个风险中类内真实方差项相消，不需要将五图类原型当作真实 μ。所用的是模型规定的 Gaussian 期望；浮点伪随机采样和 residual 计算仍有数值近似。本轮既未用 CPU 模仿 CUDA 噪声，也未从另一 bank 拿噪声或误差来拼配。

**原始逐噪声配对版本缺少信息，解析替代足够。** `observable_potential_validation_v1/shard0/step000.npz` 保存逐图 r²；`potential_omitted_witness_v1/shard0/step000.npz` 保存相同身份、相同噪声重放的 FP64 residual energy，以及旧 FP32 r² 的逐位 parity。两者都没有完整 z 或 `||z−X||²`。`potential_clean_bank_fp32_v1/validation/` 的 X 与 metadata 完整且 SHA 匹配。故可以计算上述解析替代。此前 predicted-clean interaction 是 t≈0.4103/0.1983/0.1404 的 decoded 特征，既不是 t=1 也不是所需 raw latent residual，本轮只读其 request 以确认范围，没有混用其数值。

**计算前冻结。** [request.json](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/noise_endpoint_zero_witness_v1/request.json) 的 SHA256 为 `ea2d899f21f4415270fca03589e543bf01bc44cbde6a297e01802171ca749184`，先写入并 fsync，再计算均值、X 范数及风险差。固定全部原 1K，label 取 metadata，不能用 `ID mod 1000` 替代；没有选类、时间扫描或 seed 搜索。主 r² 使用补空间缓存中 FP32 residual 的 FP64 平方／归约；X 范数由原 FP16 值直接升 FP64 计算。

| 每维量 | 全部 1000 类均值 | 描述性 class SEM |
|---|---:|---:|
| 官方 G 的 `r²` | 0.81115743 | 0.00449088 |
| `||X||²/D` | 1.00028636 | 0.00586699 |
| 解析零场风险 `1+||X||²/D` | 2.00028636 | 0.00586699 |
| 差值 `r²−1−||X||²/D` | **−1.18912893** | **0.00501374** |

差值的描述性 `mean ± 1.96 SEM` 为 `[−1.19895587, −1.17930199]`。1000 个保留类别各自的估计值均负，范围 `[−1.96016929, −0.65884926]`；这不等于已经证明每个类别的总体风险都为负。使用旧 FP32 归约 r²，均值为 −1.18912895；新旧 r² 的逐图最大差仅 `1.277×10⁻⁷`。结果没有接近数值舍入或描述性区间的符号边界。

这是**只对零场一项作解析积分**的替代估计。它与原始逐噪声配对差有相同的目标期望，但样本值不同；由于与 r² 的协方差改变，不能宣称方差必然下降。每类只有一个 source 和一次噪声、只有一个固定 cohort，SEM 只描述这些固定类别的异质性，不是完整的条件抽样置信保证。

在此具体边界和协议下，“zero-init 通过更准确的初始 velocity 起作用”的前提不受支持。它没有排除其他 checkpoint、其他时刻或其他离散动力学下 zero-init 的经验作用；也没有证明当前场已在所有意义上训练充分。但不能再援引这条被本地证据反对的初始误差解释，直接启动手工 K 或时间窗口搜索。即使将来局部风险下降，仍需另证真实终点质量与公平成本下的 FID。

完整来源和全部逐图／逐类量保存在 [noise_endpoint_zero_witness_v1](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/noise_endpoint_zero_witness_v1)，包括 `cohort_identity.npz`、`by_image_and_class.npz`、`summary.json`。准备／计算分别耗 **0.760 / 2.258 秒 wall、1.066 / 2.567 秒 CPU**。另一条不导入 producer 的逐图 FP64 平方均值计算，核对全部 1000 个身份、r² parity、X 范数及差值；最大绝对差 **7.99×10⁻¹⁵**，17 项 SHA 检查通过，另耗 **1.697 秒 wall、2.007 秒 CPU**。这是另一数值实现的自复核，未冒称独立人员评审。三阶段均为零 GPU／模型／FID／噪声调用，计时不含最后 JSON 写入、解释器退出和父 shell 启动。
