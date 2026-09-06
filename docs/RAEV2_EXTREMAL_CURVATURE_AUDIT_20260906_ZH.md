# RAEv2 极值对称 Jacobian 曲率审计

日期：2026-09-06。**40/40 个冻结状态上找到 Full 的正 Rayleigh 方向，同一状态的随机方向全部为负。** 这补足了[旧三方向审计](RAEV2_NORMAL_CURVATURE_AUDIT_20260906_ZH.md)的探针盲点：旧 gap tangent、random tangent、known normal 方向的负值仍成立，但不能据此排除其他正方向。本次没有生成候选、计算新 FID，也没有证明正方向应被抑制。

以下统计由实际完成的 CSV/JSON 独立在 CPU 重算；[机器归档](data/raev2_guidance_restart_20260906/extremal_curvature_audit.json)保留全部 40 行、120 个 Krylov 检查点、两个运行的 request/summary、逐组聚合与源产物 SHA256。归档阶段没有重跑 GPU 或新增测试。

**冻结协议与被测对象。** 模型为官方 DINOv3-L K7、EMA step 100080。复用旧 normal-noise audit 保存的 sample IDs / labels 0–3，共 4 类各一张，teacher 与官方 IG 1.78 rollout 两域；固定 solver step indices `[47,67,84,92,97]`，对应下表实际时刻。每个状态 shape 为 `[1,1024,16,16]`，全 latent 维度 262144；没有限定到 gap 或已知切空间。Teacher 是已知 clean 的 Gaussian bridge，rollout 是旧官方生成轨迹状态，两者不构成两个独立数据 bank。

参数、状态和 JVP/VJP 为 FP32，无 autocast/TF32，使用 math SDPA；Krylov basis 与 reduction 为 FP64。每个样本采用 seed `202609081+sample_id` 的独立 Gaussian 起向量，并跨时间、域复用。Full 固定做 24 次对称 Krylov 迭代，记录 8/16/24；双遍重正交后在投影矩阵中求最大代数 Ritz 值，最终方向另作一次直接 JVP/VJP 复核。

对单位向量 $v$，定义

\[
H_F=(1-t)\frac{J_F+J_F^T}{2}-I,\qquad
\rho_F=v^TH_Fv.
\]

表中 $v$ 是 **Full 的最终 Ritz 方向**；Base 和 IG 均在同一个 $v$ 上计算，不是它们各自的极值。所测时刻均在官方 guidance 窗口 `[0.1,1]` 内，故
$H_{IG}=H_F+0.78(H_F-H_B)$，$\rho_{IG}=\rho_F+0.78(\rho_F-\rho_B)$。

对真实 posterior mean，Gaussian bridge 恒等式给出 $t^2\nabla^2\log p_t=(1-t)J_F-I$，推导与适用条件见旧审计。本次测量的是实际网络的对称 Jacobian，**没有假设网络是精确 posterior 或某个真实密度的 conservative score**。即使精确模型也可能有合法多峰结构的正曲率；正值不等于模型错误，更不证明真实密度的 saddle。

**全部域与时刻结果。** 以下是每组 4 个样本的算术均值。`abs gap cos` 为 $|\langle v,F-B\rangle|/\|F-B\|$，取绝对值只去除特征向量符号不定性。

| 域 | step | 实际 t | Full ρ | Base ρ | IG ρ | random ρ | abs gap cos |
|---|---:|---:|---:|---:|---:|---:|---:|
| teacher | 47 | 0.900212 | 2.758752 | 0.734927 | 4.337336 | -0.996436 | 0.059829 |
| teacher | 67 | 0.797583 | 4.340482 | 0.885628 | 7.035268 | -0.988031 | 0.043811 |
| teacher | 84 | 0.603774 | 1.746206 | 0.193121 | 2.957612 | -0.949796 | 0.005570 |
| teacher | 92 | 0.410256 | 0.805601 | 0.051280 | 1.393972 | -0.840779 | 0.005126 |
| teacher | 97 | 0.198347 | 0.276432 | -0.005020 | 0.495965 | -0.491653 | 0.006437 |
| rollout | 47 | 0.900212 | 1.662477 | 0.124918 | 2.861772 | -0.996017 | 0.064282 |
| rollout | 67 | 0.797583 | 2.095081 | 0.372891 | 3.438390 | -0.986328 | 0.013203 |
| rollout | 84 | 0.603774 | 1.003970 | 0.049438 | 1.748504 | -0.942636 | 0.006749 |
| rollout | 92 | 0.410256 | 0.529485 | -0.023541 | 0.960844 | -0.825842 | 0.007053 |
| rollout | 97 | 0.198347 | 0.222953 | 0.000680 | 0.396326 | -0.479263 | 0.004661 |

Full 直接 ρ 的范围为 **0.213765 至 10.209675**；random ρ 的范围为 -0.997087 至 -0.417598。在同一 Full 方向上，IG ρ 全部为正且 40/40 大于 Full。绝对 gap cosine 最大为 0.155510；**这只描述与一个已找到方向的关系，不能推出 gap 在整个 positive subspace 中的投影小**。

**非对称分量与直接数值残差。** 本表 `skew/H` 精确定义为
$\|(1-t)(J_F-J_F^T)v/2\|/\|H_Fv\|$，分母含 $-I$ 位移；它不是整个 Jacobian 的 skew/symmetric 矩阵范数比。先逐样本求比，再取均值。直接残差为 $\|H_Fv-\rho_Fv\|$，由最终独立 JVP/VJP 得到。

| 域 | 实际 t | skew/H 均值 | 直接残差均值 | 直接残差最大 |
|---|---:|---:|---:|---:|
| teacher | 0.900212 | 0.530685 | 3.182e-06 | 4.090e-06 |
| teacher | 0.797583 | 0.451668 | 1.287e-03 | 5.115e-03 |
| teacher | 0.603774 | 0.499705 | 1.405e-05 | 4.989e-05 |
| teacher | 0.410256 | 0.616566 | 8.903e-04 | 3.497e-03 |
| teacher | 0.198347 | 0.959262 | 1.169e-02 | 1.709e-02 |
| rollout | 0.900212 | 0.605973 | 2.587e-06 | 4.524e-06 |
| rollout | 0.797583 | 0.510295 | 2.584e-06 | 3.223e-06 |
| rollout | 0.603774 | 0.540002 | 5.300e-03 | 2.093e-02 |
| rollout | 0.410256 | 0.743426 | 1.843e-03 | 5.816e-03 |
| rollout | 0.198347 | 1.371652 | 1.084e-02 | 1.497e-02 |

独立直接 ρ 与缓存 Ritz 值的最大绝对差为 5.615e-06；$|v^T(J_Fv-J_F^Tv)|$ 最大为 8.264e-05。已知 normal 子空间中的 $v$ 能量最大为 9.064e-06，这里只是方向描述。FP32 重新查询与旧 BF16 Full 输出的 RMS 差范围为 0.010356–0.014771；微分结论属于本次 FP32 映射。

**8/16/24 次迭代收敛检查。** Ritz 列为组内均值，增量列为逐样本 24−16 增量的最大值，残差列为 24 次迭代的缓存相对残差最大值。

| 域 | 实际 t | K8 Ritz | K16 Ritz | K24 Ritz | max Δ24−16 | max relative residual K24 |
|---|---:|---:|---:|---:|---:|---:|
| teacher | 0.900212 | 2.649377 | 2.758753 | 2.758753 | 7.914e-07 | 3.979e-07 |
| teacher | 0.797583 | 4.317280 | 4.329987 | 4.340481 | 4.151e-02 | 2.604e-03 |
| teacher | 0.603774 | 1.732371 | 1.745700 | 1.746206 | 2.025e-03 | 4.137e-05 |
| teacher | 0.410256 | 0.743655 | 0.796918 | 0.805601 | 3.469e-02 | 5.509e-03 |
| teacher | 0.198347 | 0.187671 | 0.269549 | 0.276432 | 1.564e-02 | 6.742e-02 |
| rollout | 0.900212 | 1.633311 | 1.662466 | 1.662477 | 4.239e-05 | 1.986e-06 |
| rollout | 0.797583 | 2.001312 | 2.095080 | 2.095081 | 2.754e-06 | 1.177e-06 |
| rollout | 0.603774 | 0.935114 | 1.003100 | 1.003969 | 2.388e-03 | 2.562e-02 |
| rollout | 0.410256 | 0.464031 | 0.527656 | 0.529485 | 5.994e-03 | 1.196e-02 |
| rollout | 0.198347 | 0.142306 | 0.209301 | 0.222953 | 2.979e-02 | 6.758e-02 |

最后时刻仍有约 6.8% 的相对残差，不能概括为全部充分收敛。在精确算术下，最大 Ritz 值是全空间最大特征值的**下界，而非上界**。正直接 Rayleigh 已提供一个正方向的数值见证；一个起点、有限 Krylov 子空间及小残差都不构成“找到全局最大值”、全谱或整个正特征子空间的证明。

**有限差分：全部预定步长均保留。** Formal 的首个固定状态为 teacher、sample 0、step 47、$t=0.9002123475$，最终 K24 单位方向。令 $h=\sqrt d\,\mathrm{RMS}=512\,\mathrm{RMS}$，

\[
f_d=\frac{F(z+hv)-F(z-hv)}{2h},\quad
\mathrm{relative\ error}=\frac{\|f_d-J_Fv\|}{\|J_Fv\|},\quad
\rho_{FD}=(1-t)v^Tf_d-1.
\]

这里误差检查的是 **Full JVP**，不是对称化后的 $H_Fv$。

| Formal perturbation RMS | h | JVP relative error | JVP cosine | FD ρ | autograd ρ |
|---:|---:|---:|---:|---:|---:|
| 0.001 | 0.512 | 0.380109219 | 0.975947812 | 2.484482 | 4.303033 |
| 0.0003 | 0.1536 | 0.091972695 | 0.998715999 | 3.895551 | 4.303033 |
| 0.0001 | 0.0512 | 0.012621559 | 0.999975767 | 4.248770 | 4.303033 |
| 3e-05 | 0.01536 | 0.001215458 | 0.999999668 | 4.298444 | 4.303033 |

最小扰动 RMS `3e-5` 的相对误差为 **0.001215458（0.121546%）**，与其余较粗步长结果一起归档；这是一状态、一方向的核对，不能替代其余 39 个状态的差分验证。

先行 smoke 使用 rollout、sample 0、step 47 的 K8 方向，与 formal 首个 teacher K24 差分对象不同：

| Smoke perturbation RMS | h | JVP relative error | JVP cosine | FD ρ | autograd ρ |
|---:|---:|---:|---:|---:|---:|
| 0.001 | 0.512 | 0.266072660 | 0.971450054 | 1.032556 | 1.386072 |
| 0.0003 | 0.1536 | 0.103136818 | 0.994920253 | 1.437524 | 1.386072 |

Smoke 的约 26.6% / 10.3% 误差促使 formal 加入两个更小扰动，并保留所有原步长；这是数值核对的 refinement，没有调整 guidance 方法、时间系数或生成质量目标。

**局部正曲率与生成方向的区分。** 对向数据运动的时钟 $\tau=1-t$，漂移 $b=(F-z)/t$ 满足

\[
\mathrm{sym}(J_b)=\frac{H_F+tI}{(1-t)t},\qquad
v^T\mathrm{sym}(J_b)v=\frac{\rho_F+t}{(1-t)t}.
\]

因此瞬时距离增长阈值是 $\rho_F>-t$，不是 $\rho_F>0$。本次正 ρ 是更强的局部条件，但不会自动给出有限时间轨迹不稳定、错误累积或抑制该方向的终点质量收益。

**成本、产物与裁决边界。** Formal 完成 40 行，运行 wall time **579.148 秒**，计数为 model forward **1088**、JVP **1040**、VJP **1080**，峰值显存 **4855322624 bytes（4.522 GiB）**。JVP 调用内部也计入 model forward，三种计数不能相加称作互不重叠的 NFE。Smoke 完成 1 行、21.901 秒，forward/JVP/VJP 为 15/10/11；两次 wall time 合计 **601.049 秒**。这是机制诊断成本，不是新采样方法的生成成本。今后任何候选仍须和当前官方 100-step IG 1.78 及额外计算量匹配的基线比较。

4 类、旧冻结状态、复用起向量和多个相关时刻只支持局部数值结论，不能推断总体频率或独立确认。结果修正的是“旧探针没有发现正方向”的证据范围，**尚未导出新 guidance，不放行仅凭正曲率的抑制设计或 1K 质量主张**；没有新的采样、解码、训练或 FID，也不能宣称某一候选的 FID 失败。

Formal 原始产物：[summary](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/extremal_curvature_seed202609081/summary.json)、[40 行 CSV](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/extremal_curvature_seed202609081/extremal_curvature.csv)、[Krylov history](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/extremal_curvature_seed202609081/krylov_history.json)、[FD](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/extremal_curvature_seed202609081/finite_difference.json)、[首状态张量](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/extremal_curvature_seed202609081/first_state_numerics.pt)。Smoke：[summary](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/extremal_curvature_smoke_seed202609081/summary.json)、[FD](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/extremal_curvature_smoke_seed202609081/finite_difference.json)。实现：[审计运行器](../experiments/audit_raev2_extremal_curvature.py)、[对称 Krylov](../experiments/raev2_symmetric_krylov.py)。

上述实际结果文件、request、运行器快照和进度文件的 SHA256 在机器归档中重新计算。较大的模型和旧输入状态沿用原 request 已记录的 SHA256，并明确标注未在本次归档重复全文件哈希。
