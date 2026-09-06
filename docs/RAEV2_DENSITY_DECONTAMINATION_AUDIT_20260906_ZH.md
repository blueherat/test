# RAEv2 线性密度去污染准入审计

日期：2026-09-06。**当前固定系数 mixture 解释与 marginal-ratio 估计器未通过准入；不进入 coupled guidance 或 1K，不用温度、偏置、clamp 或 gain cap 补救。** 两个独立探针在 797 个共同可测条目中给出 82 个共同违反协方差必要条件的条目，覆盖全部 8 个样本。数据不能区分混合假说失效与密度比估计误差，也没有 FID 结果。

**理论前提。** 固定沿用官方 $w=1.78$，假设

\[
p_F=\frac1w p_\star+\frac{w-1}{w}p_B,
\qquad p_\star=w p_F-(w-1)p_B\ge0.
\]

这相当于假设 Full 含 **43.8202% 的 Base 污染**，不是从数据估计出的比例。共同 Gaussian noising 与加法可交换，因此若 clean 分布满足该关系，各 noisy marginal 也满足。记 $\ell=\log(p_B/p_F)$、$u=((w-1)/w)e^\ell$，则正性要求

\[
\ell<L=\log\frac{w}{w-1}=0.8250747236,
\quad\beta=\frac{u}{1-u},\quad G=F+\beta(F-B).
\]

若 $F,B$ 是桥 $z_t=(1-t)X+t\epsilon$ 下相应分布的精确 posterior mean，$G$ 精确恢复目标 posterior mean。在纯噪声端 $\ell=0$，$\beta=0.78$；此后系数取决于状态密度比。没有 latent 维度归一化、density temperature 或有限 gain cap；所有内积和 trace 按全坐标求和。

混合恢复借鉴 [FBG](https://arxiv.org/html/2506.06085v2)。任意轨迹上搬运 marginal density 已见 [SuperDiff Proposition 5 / Eq. 10](https://arxiv.org/html/2412.17762) 与 [Density Guidance Eq. 4 / Appendix B](https://arxiv.org/html/2502.05807)，**不是本次提出的新原理**。由 $d\log p_i(z_t)/dt=-\mathrm{div}v_i+s_i^\top(v_G-v_i)$ 相减，代入 $v_i=(z-i)/t$、$s_i=((1-t)i-z)/t^2$，得到

\[
\frac{d\ell}{dt}=\frac{D^\top[z-(1-t)(B+F-G)]}{t^3}
-\frac{\mathrm{div}D}{t},\qquad D=F-B.
\]

这是固定 $p_F,p_B$ 的 transport，不能直接解释为实际 guided 分布的 log-density；神经网络换算出的 score 也不自动等于该网络 CNF 所诱导密度的 score。

**后验协方差的必要条件。** 精确 mixture 满足

\[
C_F=(1-u)C_\star+uC_B+\frac{u}{1-u}DD^\top.
\]

由 $J_i=\frac{1-t}{t^2}C_i$ 与 $C_\star\succeq0$，任意方向 $v$ 必须有

\[
M(v)=v^\top(J_F-uJ_B)v
-\frac{1-t}{t^2}\frac{u}{1-u}(D^\top v)^2\ge0.
\]

本次测 detached gap 方向；VJP 的 $v^\top J^\top v=v^\top Jv$ 可检验此必要不等式。通过一个方向不足以证明 PSD。两个 trace 探针各自产生 $\ell,u$ 和 margin，因此 directional VJP 的直接计算并没有消除密度比的不确定性。

**实际运行与结果。** 权威结果为 `density_transport_audit_seed202609074_v2`：官方 EMA checkpoint step 100080、DINOv3-L K7、8 类各 1 个噪声，shift 8 的 100 步 Euler，共 800 个样本—时刻条目。保持原生 IG 公式及 `[0.1,1]` 区间，模型 FP32、无 autocast、关闭 TF32；没有声称与 BF16 生产轨迹 bitwise 相同。每步两个独立、新鲜 Rademacher 探针各计算一次 gap trace VJP，另计算 Full/Base gap-direction VJP。FP64 冻结系数积分沿实际 IG 轨迹累积两份 $\ell$，**均不反馈采样**。最后 $t\to0$ 只推进原生轨迹，不在奇异零端点积分 density。

| 检查 | 探针 0 | 探针 1 |
|---|---:|---:|
| 满足估计正性、可测 covariance 的条目 | 799 / 800 | 797 / 800 |
| covariance margin < 0 | 88 | 88 |
| 估计正性违例 | 1 | 3 |
| 最大 query $\ell$ | 0.840351 | 0.967533 |

共同可测 **797** 条，其中共同负例 **82 / 797 = 10.2886%**；这是相关轨迹条目的描述，不能当作总体失败概率。8 个样本的共同负例次数为 `[7,9,10,19,4,13,13,7]`，位于 step index **1–22**。82 个共同负例的两份 margin 共 164 个值，范围 **−832.548169 至 −0.288756790**，最接近零者仍为负。正性违例均发生于 sample 3：探针 0 在 step 15，探针 1 在 step 13–15。

前 20 个 query 时刻（step 0–19，共 160 条）两份 $\ell$ 的绝对差均值为 **0.259549483**，最大 **1.349413362**。最后 query $t=0.074766353$ 的 $\ell$ 范围为 **−1,000,893.231589 至 −553,754.432713**。这些巨大负数来自未校准 transport 估计，**不是已确立的真实 density ratio**。两个探针共享模型、轨迹和离散方案，共同负例不能排除共同系统误差。完整统计见[机器归档](data/raev2_guidance_restart_20260906/density_decontamination_audit.json)。

**v2 数值修订。** 初版 `density_transport_audit_seed202609074` 保留。v2 用实际 $(D^\top v)^2$，不以理想单位方向下的 $\|D\|^2$ 替代，并加入导数诊断和 margin finite 检查。实际 $|\|v\|^2-1|$ 最大为 $5.047\times10^{-10}$。两版 noise/probe seeds、初始噪声 SHA 相同；逐行浮点诊断有细小差异，未声称全列 parity。单探针 88 / 88 负例、1 / 3 正性违例及裁决不变。本归档从 v2 CSV 独立重算聚合并核对 summary SHA，所有非缺失数值有限；正性失败的 covariance 字段保持缺失，没有修补。

**CPU 验证范围。** 本会话由实现子代理执行的 8 项单元测试通过，验证隐式原方程、正性域、$w=1$、零系数无解、巨大 uncapped gain、不可表示根显式失败、独立 quadrature、近邻时刻稳定性及全坐标求和；归档过程未重跑。标量方案解 $\ell_{new}+B\beta(\ell_{new})=\ell_{old}+A$，在 $B>0$ 时由单调性得到唯一域内根，使用 FP64 barrier 距离坐标求解。这是冻结系数离散方程的保证，不是实际模型的 mixture 保证。

真实文件 `density_decontamination_toy_seed202609073.json` 人工构造满足假说的 Gaussian mixture。2D 上恢复 posterior mean 最大误差 **2.665e−15**，任意轨迹 log-ratio 导数最大误差 **2.742e−14**。1D 的 128 个确定性 Gaussian 分位点中，transport 终点分位 RMSE 随 100 / 400 / 1600 步从 **0.02226046 → 0.00583724 → 0.00149356** 下降，query ratio 最大误差从 **0.399653 → 0.198257 → 0.069350** 下降；对应最大 $\beta$ 为 **8.90 / 11.23 / 12.35**，未 cap。它只验证构造分布上的代数和离散收敛，不验证 RAEv2 污染比例，也不是 FID 证据。机器归档保存实际 toy 文件 SHA 和其输出的源码 hashes。

**成本与否定边界。** v2 耗时 **62.28395 秒**，100 次 B8 forward、400 次 B8 VJP，峰值 **14.44745 GiB**；初版另耗 **61.83440 秒**。两次合计 124.11835 秒、200 次 B8 forward、800 次 B8 VJP。异构前向／反向调用不能直接计为相同 NFE；这些是诊断成本，没有同成本生成收益。最终比较必须包含当前官方 100 步 IG 1.78；若候选增加计算量，还须提供使用可比计算预算的可靠基线，并相对该基线达到至少 5% 的 FID 降低。

本次只有 8 类、单噪声 bank，时间条目高度相关；FP32 不消除 trace 方差、score 与密度不一致、低噪声 $t^{-3}$ 放大、冻结系数或 Euler 离散误差。没有 coupled candidate、训练、解码、选样或 FID。裁决是**当前混合解释及估计实现缺少继续投入 1K 的必要证据**，不证明所有线性 correction 不可能，也不宣称已测得 FID 失败或提升。

数据根目录：`/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/`。代码：[准入审计](../experiments/audit_raev2_density_transport.py)、[标量代数与求根](../experiments/raev2_density_decontamination.py)、[构造 toy](../experiments/audit_raev2_density_decontamination_toy.py)。机器归档直接提取真实 CSV/request/summary/toy，保存源文件 SHA256 及运行时 config/checkpoint/code hashes；未重新下载或修改来源。
