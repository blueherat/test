# RAEv2 已知法向噪声与方向曲率审计

日期：2026-09-06。裁决：**法向泄漏与已测修正量太小，不放行简单投影或两次查询镜像平均的 1K；未观察到正方向曲率，不放行抑制正曲率的 1K。** 这是机制筛查裁决，不是 FID 失败，也没有达到研究目标。

**后续测量补充：** [极值曲率审计](RAEV2_EXTREMAL_CURVATURE_AUDIT_20260906_ZH.md) 在固定的 40 个状态上均检出正 Rayleigh 方向，而对应随机方向均为负。本文原三个方向的阴性测量保持有效，但不能再以它们支持“没有正曲率方向”的解释。新测量仍未识别需要修正的质量误差，不自动放行正曲率 guidance。

数值全部由两次真实运行的 CSV/summary/request 聚合；[机器归档](data/raev2_guidance_restart_20260906/normal_curvature_audit.json)保存源文件完整 SHA256、聚合定义、逐时刻曲率、模型身份与计算成本。10 个状态文件的 SHA256 已重新计算并与两次运行记录交叉匹配。

模型固定为官方 DINOv3-L K7、EMA step 100080；轨迹使用 shifted Euler 100 步、shift 8、IG 1.78、区间 `[0.1,1]`，Full/Base 来自同一次模型调用。模型参数与状态 FP32，原轨迹 forward 使用 BF16 autocast/TF32。所有镜像查询只作诊断，原轨迹更新仍使用原始输出，没有解码、拟合或 FID。

**已知仿射结构。** `vision_encoder.py` 的 DINOv3 默认去掉最终 LayerNorm affine；K7 特征由各层归一化 patch token 均值加上最后选定层的空间均值构成。二者逐 token 的 channel sum 都为零，因此原始 encoder token $a_j$ 在精确算术下满足 $\mathbf1^Ta_j=0$。RAE 再作逐位置标准化 $X_j=(a_j-\mu_j)/\sigma_j$，其中 $\sigma_j=\sqrt{\mathrm{var}_j+10^{-5}}$，故

\[
\sigma_j^T X_j=-\mathbf1^T\mu_j,\quad
u_j=\sigma_j/\|\sigma_j\|,\quad
c_j=-\frac{\mathbf1^T\mu_j}{\|\sigma_j\|}\nu_j.
\]

记 $P_N$ 为逐位置投影到 $\nu_j$ 的正交投影，干净 latent 属于 $H=\{x:P_N(x-c)=0\}$。这只是一条逐 token 仿射约束；K7 层均值及附加空间均值不保持单层单位范数，不能据此再假定球面结构。源码：[encoder](../external/RAEv2/src/encoders/vision_encoder.py)、[RAE 标准化](../external/RAEv2/src/stage1/rae.py)。

**能够保证的对象。** 对 $X\in H$，投影 $\Pi_H f=f-P_N(f-c)$ 满足逐样本恒等式

\[
\|f-X\|^2-\|\Pi_H f-X\|^2=\|P_N(f-c)\|^2.
\]

对 teacher channel $Z_t=(1-t)X+t\epsilon$、独立标准 Gaussian $\epsilon$，反射
$R_tz=z-2P_N(z-(1-t)c)$ 保持 $(X,Z_t)$ 的联合分布。令 $\bar f(z)=[f(z)+f(R_tz)]/2$，则平方展开给出

\[
E\|f(Z_t)-X\|^2-E\|\bar f(Z_t)-X\|^2
=\tfrac14E\|f(Z_t)-f(R_tZ_t)\|^2\ge0.
\]

因此投影和对称平均具有 population teacher-MSE 保证，是已知的正交投影／对称平均结构；它们不保证有限样本差值处处为正，不保证实际 rollout 的输入分布保持该反射对称，也不保证终点 FID 改善。rollout 表内的 reference error 仅表示与共享标签、初始噪声的原始 clean 样本之距离，不能称为 posterior risk。

**法向测量。** 32 类各一张，独立 C clean bank 的 IDs 0–31；100 个 solver 时刻、两域，共 6400 行。干净缓存的仿射法向能量占比均值为 **4.185e-11**，最大 5.499e-11，支持上述结构在 FP16 存储误差内成立。Full−Base 的 pooled 法向能量占比为 teacher **9.456e-06**、rollout **7.988e-06**。

下表能量比先加总分子、分母再相除；相对收益也由加总 reference squared error 计算。100 个 solver 步等权计数，是审计汇总，不是连续时间积分或指导强度调度。

| 域 | 头 | 法向输出能量 / 总输出能量 | 投影相对 reference-MSE 收益 | 镜像平均后投影相对收益 |
|---|---|---:|---:|---:|
| teacher | full | 5.595e-07 | 0.000128% | 0.007749% |
| teacher | base | 8.040e-07 | 0.000160% | 0.022808% |
| teacher | ig | 8.982e-07 | 0.000210% | 0.033475% |
| rollout | full | 5.523e-07 | 0.000035% | 0.002275% |
| rollout | base | 7.262e-07 | 0.000045% | 0.007065% |
| rollout | ig | 8.453e-07 | 0.000055% | 0.009466% |

Teacher IG 的简单投影收益仅约百万分之二，镜像平均后投影约 **0.0335%**，不足以支持支付两倍模型查询来进入 1K。这里的幅度判断不把 MSE 映射成 FID 上界，也不证明所有法向设计都不可能。

**曲率理论与测量。** 对真实 conditional posterior mean $F_t(z)=E[X\mid Z_t=z,y]$，Gaussian score identity 给出

\[
\nabla_z\log p_t(z\mid y)=\frac{(1-t)F_t(z)-z}{t^2},\qquad
t^2\nabla_z^2\log p_t=(1-t)J_{F_t}-I.
\]

这是 Gaussian Tweedie 恒等式的直接多维微分形式，非本实验新理论；参见 [Efron, 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3325056/)。对单位方向 $v$ 定义 $\rho=(1-t)v^T J_Fv-1$。实际网络不预设 conservative 或 accurate，所以本实验只测对称 Jacobian 的方向二次型。VJP 得到 $J_F^Tv$，但标量 $v^TJ_F^Tv=v^TJ_Fv$，无需假定 Jacobian 对称。精确模型在已知法向上应有 $\rho=-1$；正 score 曲率本身也可能来自合法多峰密度，不能等同模型错误。

在上述首 8 张已保存状态的 10 个时刻测量 teacher/rollout：每个状态分别使用 detached 当前 gap 的切向分量、独立随机切向方向、随机法向组合，整体归一化。后两方向跨时刻复用，gap 方向每个状态重算。微批为 1，FP32/no autocast/no TF32；每个方向 Full、Base 各一个 VJP，IG 二次型按官方系数线性合成。

- 480 行中，所有域和方向的 Full/Base/IG **正 $\rho$ 比例均为 0%**。
- 法向所有二次型与 $-1$ 的最大差为 **9.782e-04**。
- 排除必然 $\rho=-1$ 的 $t=1$，144/144 个配对点的 Full gap-tangent $\rho$ 高于独立随机 tangent，最小差 **0.04289**。

Rollout 各时间的 8 样本均值：

| 实际 t | gap tangent：Full / Base / IG | random tangent：Full / IG |
|---:|---:|---:|
| 0.900212 | -0.7281 / -0.7664 / -0.6982 | -0.9961 / -0.9965 |
| 0.797583 | -0.7697 / -0.7792 / -0.7623 | -0.9867 / -0.9894 |
| 0.603774 | -0.7407 / -0.7572 / -0.7279 | -0.9445 / -0.9563 |
| 0.410256 | -0.6131 / -0.6335 / -0.5972 | -0.8296 / -0.8595 |
| 0.198347 | -0.3218 / -0.3969 / -0.2632 | -0.4804 / -0.4926 |

当前可描述为：gap 指向较高响应、较平坦的切向方向；上述 rollout 截面上，IG 使 gap 方向更平坦，同时使随机方向更收缩。这个结构现象尚未解释其是否属于模型误差，更未导出新的质量修正算子。Teacher 在部分时刻的 IG−Full gap 曲率差略负，因此也不存在所有状态上的统一排序。

FP32 重新查询与原 BF16 保存输出的 RMS 差异如下（均值 / 最大值）：

| 域 | Full | Base |
|---|---:|---:|
| teacher | 0.011483 / 0.014682 | 0.023128 / 0.034628 |
| rollout | 0.011555 / 0.014771 | 0.023451 / 0.036077 |

Full、Base 差异分别约为 raw gap RMS 的 7% 和 14–15%；它们不是 gap 差异的直接测量。曲率属于 FP32 网络，不能无条件移植为 BF16 数值映射的微分性质。

**成本与边界。** 法向审计实际 1600 个 B8 模型调用，12800 次样本模型评估，110.70 秒；曲率审计 160 次 B1 forward、960 次 VJP、51.60 秒，峰值显存 4.20 GiB。这些是机制审计开销，不是候选生成性能或成本基线。若以后提出合格的新方法，仍须与当前官方 100-step IG 1.78 以及额外调用匹配的可靠基线比较。

32 类／8 类、单噪声 bank 不支持 ImageNet 整体质量结论；同一样本的多个时间和方向不是独立重复。三个被测方向均为负不等于全谱负定，接近理想的 normal Rayleigh 也不证明全部法向交叉导数消失。本次没有生成 1K 候选图，没有得到 FID 阴性结论，不因局部指标通过而放行候选。

源产物目录：`~/data/eqvae/experiments/raev2_guidance_restart_20260906/normal_noise_audit_seed202609071/` 与 `curvature_audit_seed202609072/`；运行状态以两个完成的 `summary.json` 为准。运行器为 [normal audit](../experiments/audit_raev2_guidance_normal_noise.py)、[curvature audit](../experiments/audit_raev2_guidance_curvature.py)。
