# RAEv2 可观测误差的有限势函数求解器

日期：2026-09-06。**固定 2048 更新训练、全部 100×1000 validation 和完整 100-step 成本 benchmark 已完成。Validation 的加权 coupling MSE 改善 1.0968%，但剩余 weak witness 明显非零，完整 Poisson 问题未解；实测推理耗时增加 4.783%，这不是 FID 提升。** 配对 1K 筛查已完成：FID 37.562703724300 → 37.531664406887，仅下降 0.08263334%，按预注册第一阶段停止；见[负结果归档](RAEV2_OBSERVABLE_POTENTIAL_SCREEN_20260906_ZH.md)。它补充[机制文档](RAEV2_OBSERVABLE_ERROR_GUIDANCE_20260906_ZH.md)的有限实现，不将总体定理写成有限求解成功或质量提升保证。

**机制与单位。** 沿用官方 DINOv3-L K7、EMA step 100080、100-step shifted Euler 和 IG 1.78。原噪声时间为 t，真实 bridge 为 $Z_t=(1-t)X+t\epsilon$；$X$ 是原图的 normalized encoder latent。冻结 baseline 的 clean prediction 为

\[
G=\begin{cases}B+1.78(F-B),&t\in[0.1,1],\\F,&\text{otherwise}.\end{cases}
\]

Full/Base 及上述 IG 算术保留原生 head BF16 dtype，算完才转 FP32；不能用先转 FP32 的重排式替换数值 baseline。Stage 2 冻结，state 与势网络 FP32，TF32 全局关闭。

机制对象是相对真实加噪边缘的最小修正 $u^\star=\Pi(v^\star-v)$，即 weighted gradient projection，并非普通去 curl。当前有限势用 clean 单位参数化：$\Phi_\theta=t\phi_\theta$，因此

\[
c_\theta=\nabla_z\Phi_\theta(z,t,y),\qquad G_{\rm new}=G+c_\theta,\qquad
u_\theta=c_\theta/t=\nabla_z\phi_\theta.
\]

部署系数固定为 **1**。同一势网络用于全部 100 个正查询时刻，不增加 gain、势修正窗口、clamp 或手调时间系数；原 IG 窗口保留。当前最低查询 $t\simeq0.074766>.05$，未触发原速度转换的分母 clamp，故 dataward baseline drift 为 $v=(G-z)/t$。若改变 grid，必须重新核对 clamp 区域。

令 $U=X-\epsilon$，有 $U-v=(X-G)/t$。于是固定时刻的目标为

\[
\frac1{t^2}\mathbb E[\tfrac12\|c_\theta\|^2-(X-G)\cdot c_\theta].
\]

这仍是带梯度场结构的 residual conditional flow matching，未绕开高维条件均值估计。连续性方程解唯一、终点极限与完整梯度闭包等总体条件，不能替代有限网络、数据、优化与 Euler 离散误差的核对；也不预设辅助问题比 denoising 更容易。

**唯一生产结构。** [势模块](../experiments/raev2_observable_potential.py)固定 **608000** 个可训练参数，只输入 `[B,1024,16,16]` 的 z、原始 t 和 1000 类标签，不输入 backbone head 或中间特征。实际顺序为：

1. `Conv1×1(1024→128)` 后 `SiLU`，再加 128 维类别 embedding、时间 MLP 输出和 `[1,128,16,16]` 的可学习 position。
2. 时间特征为 32 维 `cos` 后 `sin`，频率 $\exp[-\log(10000)k/16]$、$k=0,\ldots,15$；直接使用原 t，不重标度。时间 MLP 为 `32→128→SiLU→128`。
3. 两层 `Conv3×3(128→128,padding=1)`，各接 `SiLU`；最后无 bias 的 `Conv1×1(128→1)` 对空间与输出 channel 求和，得到每样本一个标量势。

Position 与最终 readout 权重零初始化，其余层用 PyTorch 默认初始化；不是全部参数归零。没有 normalization、dropout 或随机层。宽度、卷积和时间表示是一次预先固定的数值近似选择，**不是理论唯一导出的架构**，不延伸为架构搜索。

修正始终由真实输入梯度形成，无零 readout shortcut。训练用 `create_graph=True` 保留势参数经过输入导数的反传；已有 z 的上游图也保留。Baseline 只作固定 target，且势没有引用其特征，因此可 detach；这不授权把一般 backbone-feature 势的链式导数截断。势推理仍需要 forward 和输入梯度；支持外层 `no_grad`，不支持 `inference_mode`/inference tensor。

**固定训练。** [runner](../experiments/train_raev2_observable_potential.py)的协议名为 `raev2_observable_potential_solver_v1`。正式训练从固定 seed 新建零 readout 势，不继承 pilot 八步拟合；非 validation 模式禁止加载已有 potential checkpoint。

| 项目 | 冻结值 |
|---|---|
| 更新与 batch | 2048 次，B32，共 65536 次有放回图像抽样 |
| 图像 | train5000 bank IDs 均匀独立抽样，读取 metadata 标签 |
| 优化器 | Adam，lr `1e-4`，betas `(0.9,0.999)`，eps `1e-8`，weight decay `0` |
| 其他优化 | 无 EMA、lr schedule、gradient clipping；非有限或全参数梯度全零即报错 |
| checkpoint | 只取第 2048 次更新 `potential_final.pt`，不用 validation 选优 |
| 初始化 seed | `202609091` |
| 图像/时间 RNG | NumPy `default_rng(202609092)` 的固定顺序 |
| 训练 Gaussian RNG | GPU generator `202609093`，每次重新抽噪声 |
| validation Gaussian RNG | `202609094`，每个时刻重置，使每图跨时间噪声相同 |
| 精度 | Stage 2 BF16 autocast；state/势/目标累加 FP32；TF32 关闭 |

每图独立抽 t 并构造真实 bridge。Loss 为逐维均值后 batch 均值的 $\tfrac12c^2-(X-G)c$。本轮不传额外 `sample_weights`，时间权重已由概率吸收，不能再乘一次 $t^{-2}$。维度均值与概率归一化仅引入总体常数，但优化步长依赖这些实际数值尺度。

Runner 严格校验官方 config/checkpoint SHA、完成的 bank、与 pilot 相同的三份源码快照及模型身份。Validation 还核对 final checkpoint 指向的训练 request SHA、bank、baseline 和执行源码。实际完成 2048 个 B32 Stage-2 forward（65536 个样本 forward）、2048 次势 forward/输入梯度及 2048 次势参数 backward；完成产物和成本见后文。

**时间概率严重偏尾。** 使用实际 FP32 shift-8 grid：$b_k=1-k/100$，$t_k=8b_k/(1+7b_k)$，$h_k=t_k-t_{k+1}$。查询 $k=0,\ldots,99$，不在 $t_{100}=0$ 查询。冻结

\[
w_k=h_k/t_k^2,\quad p_k=w_k/\sum_jw_j,\quad \sum_jw_j=21.75769184184151.
\]

它对应速度目标的有限 grid 左端离散和 $\sum_kh_k\mathbb E[\tfrac12\|u_k\|^2-(U-v_k)\cdot u_k]$，除全局常数外与抽样目标一致。它不是部署的 extrapolation schedule，也不是未经近似的连续时间积分；尤其零端点仍有适定性和离散边界。

下表读取冻结 request 的概率；期望次数是 $65536p_k$，实际次数来自完成的 `sampling_counts.npz`，两者明确区分。

| step | 实际 t | 概率 | 期望训练次数 | 实际训练次数 |
|---:|---:|---:|---:|---:|
| 0 | 1.000000000 | 0.005796% | 3.798 | 2 |
| 47 | 0.900212348 | 0.020761% | 13.606 | 5 |
| 67 | 0.797583044 | 0.053895% | 35.321 | 26 |
| 84 | 0.603773594 | 0.232081% | 152.097 | 159 |
| 90 | 0.470588237 | 0.599182% | 392.680 | 385 |
| 92 | 0.410256416 | 0.939843% | 615.936 | 594 |
| 95 | 0.296296299 | 2.423712% | 1588.404 | 1653 |
| 96 | 0.250000000 | 3.798411% | 2489.327 | 2362 |
| 97 | 0.198347092 | 6.775403% | 4440.328 | 4567 |
| 98 | 0.140350878 | 15.302356% | 10028.552 | 10086 |
| 99 | 0.074766353 | 61.472515% | 40286.628 | 40051 |

| 末端范围 | 累计概率 |
|---|---:|
| 最后 1 个查询 | 61.472515% |
| 最后 2 个查询 | 76.774871% |
| 最后 3 个查询 | 83.550274% |
| 最后 5 个查询 | 89.772397% |
| 最后 10 个查询 | 94.961393% |
| 最后 20 个查询 | 97.724029% |

前 80 个查询合计仅 **2.275971%**。$t=1$ 全训练期望仅 **3.798 次**，远不足以覆盖该时刻 1000 类。最后 $t=0.074766353$ 占 **61.472515%**，且该点官方 IG 已关闭、baseline 是 Full。有限共享势存在跨时间拟合折中，不能借总体逐时刻最优解掩盖早期欠覆盖。实际 $t=1$ 只抽到 **2** 次，最后时刻抽到 **40051** 次，前 80 个时刻合计 **1496** 次。未因偏尾添加手动温度、混合均匀概率或时间门限；validation 仍保留全部时刻。

**固定原图数据。** `potential_clean_bank_fp32_v1/` 根 `summary.complete=true` 后才能读取。Train 为旧 seed20260801 scale-response `sample_protocol.real_source_rows` 的全部 5000 行，按旧 global sample ID 排序，每类 5 张。Validation 从旧 seed20260802 每类 5 个候选中取 global ID 最小且 source row 不在 train 的一张；任一类无候选即报错。

源行交集 **0**。Seed02 全候选有 **21** 行与 train 重叠，最终 **4** 类使用后一个候选：

| 类别 | 选中旧 global ID | 新 validation bank ID |
|---:|---:|---:|
| 13 | 1013 | 996 |
| 51 | 1051 | 997 |
| 292 | 1292 | 998 |
| 668 | 1668 | 999 |

读取 `metadata.labels`，不能把 bank ID 当类别。Metadata 键为 `ids, source_sample_ids, rows, labels, source_test_mask`。旧 800/200 类 mask 仅作来源记录；当前两 bank 均覆盖全部 1000 类，均来自 ImageNet **train split**。Seed02 早已用于研究，因此只是**本次参数拟合之外的 validation，不是从未观察过的质量确认集**。

源为 Packed 原始 RGB 图像，与旧原图分支相同的 ADM 256 center crop、FP32 `/255`、无翻转、无 index map；不使用重构图 `decoded/real` 或旧 BF16 clean latent。已核对 Packed 对应原 sorted parquet 的顺序、大小、行数及全部候选标签，未重哈希整套原图 payload。官方 config、DINOv3-L 权重、stats 和相关编码源码与先前已审计 FP32 C bank 严格一致。

[数据脚本](../experiments/prepare_raev2_potential_clean_bank.py)复用现有 RAE Stage 1 和 reader。GPU 1、B8，参数/输入/encoder 输出 FP32、无 autocast/TF32，`RAE.encode` 内仅应用一次 $(z-\mu)/\sqrt{\mathrm{var}+10^{-5}}$；保存值已规范化，不得再 normalize。Train/validation 的 `latents.npy` 可 mmap，分别为 `[5000,1024,16,16]` / `[1000,1024,16,16]`、FP16；同时保留逐图 crop tensor SHA、元数据、文件哈希和精确 runner 快照。

| 数据 | FP32→FP16 RMS | 相对 RMS | 最大绝对误差 | encoder forward |
|---|---:|---:|---:|---:|
| train | 2.076673142e-04 | 2.075568586e-04 | 0.002646446 | 625 × B8 |
| validation | 2.075700025e-04 | 2.075402874e-04 | 0.001953125 | 125 × B8 |

编码完成 6000 张、750 个 B8 encoder forward，记录时间 **113.860739 秒**（request 后模型实例化开始，含编码、输出 hash 和释放；不含先行 CPU 行号选择/身份哈希），峰值分配显存 **1387917824 bytes = 1.293 GiB**。独立 CPU 复核所有保存 latent 的有限性、shape/dtype、所有 metadata SHA 和两份大数组 SHA，均通过。该开销是辅助方法的数据准备成本，没有 FID。

**已完成的 pilot 与成本。** 以 `observable_potential_pilot_v2` 为当前 runner 的准入产物，已 `complete=true, pilot_passed=true`。固定 train IDs 0–31，step `[0,47,67,84,92,97,98,99]` 重复四次，noise seed `202609193`。原生 BF16 official wrapper 一致、零修正 clean 和 Euler 下一状态逐元素一致；输入梯度和参数反传通过。以下对零初始化 readout 的梯度方向作参数中心差分，所有步长保留：

| ε | analytic derivative | finite difference | relative error |
|---:|---:|---:|---:|
| 0.001 | 1.032371619658e-05 | 1.032372013475e-05 | 3.814684867e-07 |
| 0.0003 | 1.032371619658e-05 | 1.032372036742e-05 | 4.040056797e-07 |

零初始化全参数梯度 norm 为 **1.032371641e-05**。固定 batch 八步 loss 从 0 降至 **−8.212520441e−8**，未写训练 checkpoint。这是实现可训练的数值检查，不是 heldout 改善、完整弱残差消除或正式训练结果。

| B32 微基准（3 次平均） | 秒 |
|---|---:|
| baseline forward | 0.231458010 |
| 势 inference：forward＋输入梯度 | 0.001817535 |
| 势训练导数：forward＋输入梯度＋参数 backward | 0.008898748 |

最后一项 **未包含 Adam 更新**，不是完整训练 iteration wall。势推理约为该 teacher batch baseline 耗时的 0.785%，也不是实际 100-step rollout 的总增量成本。

Pilot v2 内部核对/微基准耗时 **1.498043942 秒**；含 bank/model 哈希与加载等的主程序 wall 为 **19.183404064 秒**，CPU **19.440478163 秒**（均不含 imports 与末次 summary 写入）。峰值 **4924754432 bytes = 4.587 GiB**。共 4 个 B32 Stage-2 forward/128 个样本 forward、19 次势 forward/输入梯度、12 次参数 backward。

先行 pilot v1 也通过，内部核对耗时 **1.509750093 秒**，同为 128 个 Stage-2 样本 forward、19 次势输入梯度、12 次参数 backward；未记录完整含加载 wall，不能补算。V2 加强 frozen baseline/checkpoint 身份、禁止继承拟合、总计时与 witness clean-unit 命名，核心结构/目标不变，两份参数差分逐值相同，八步记录存在微小数值差异，最终 loss 相同。**V1 仍是额外实际审计成本**：两次共 256 个 Stage-2 样本 forward、38 次势输入梯度、24 次参数 backward；不得只报 v2 内部 1.498 秒为全部准备成本。

**正式训练已完成。** `observable_potential_train_v1` 固定 2048 次更新完成，5000 张训练图全被抽到，各图复用 2–26 次。只使用最后 checkpoint [potential_final.pt](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/observable_potential_train_v1/potential_final.pt)，SHA256 `495b3313e945f4b845307ae0d520c3c8d8fa70e60e3983002297cd3f7ccc632e`；未依 validation 选 checkpoint、系数或时间窗口。末 batch loss 为 −0.0001907775004，仅是训练记录，不作泛化结论。完整逐步记录、实际 time/image histogram 与源 SHA 见[机器归档](data/raev2_guidance_restart_20260906/observable_potential_solver_audit.json)。

**全部 validation 已完成。** 三个 shard 各覆盖 34/33/33 个时刻，合并为全部 **100 时刻 × 1000 图像**，无缺漏、重复或换行。每图跨时刻使用同一个固定独立噪声。定义逐样本维度均值 $R^2=\operatorname{mean}(X-G)^2$、$C^2=\operatorname{mean}c^2$、$RC=\operatorname{mean}(X-G)c$；clean gain 为 $2RC-C^2$，冻结势自身的 clean-unit witness 为 $RC-C^2$。

先按**同一图像**对全部时刻加权，再对 1000 个图像/类别计算描述性标准误，保留了跨时刻协方差；不能把 100000 个相关条目当成独立样本。速度单位的积分 gain/witness 使用 $w_k=h_k/t_k^2$，其中 $\phi=\Phi/t$，故 $\nabla\phi=c/t$。真实 100 个 NPZ 的 SHA、ID、label 与银行元数据，以及重算的 per-image 向量都与聚合产物逐项匹配。

| 全时刻配对统计（每维） | 均值 | 描述性 SEM |
|---|---:|---:|
| $\sum p_k(2RC-C^2)$：weighted clean gain | 0.000385990600 | 2.651409399e-06 |
| $\sum (h_k/t_k^2)(2RC-C^2)$：integrated velocity gain | 0.008398264519 | 5.768854864e-05 |
| $\sum (h_k/t_k^2)(RC-C^2)$：remaining velocity witness | -0.000218965868 | 2.462621026e-05 |

Weighted baseline coupling MSE 为 **0.0351934610284**，相对下降 **1.096767946%**；1000/1000 张图的全时刻加权 gain 为正。**83/100 个时刻的均值 gain 为正，最早 step 0–16（t=1 至 0.976744235）的 17 个均值为负**，负均值时刻的概率质量为 **0.117494023%**。这些是均值符号计数，不是 83/17 个逐时刻显著性结论。

| step | t | clean gain 均值 | gain SEM | clean-unit remaining witness |
|---:|---:|---:|---:|---:|
| 0 | 1.000000000 | -4.098389140e-05 | 1.213113840e-05 | -9.835902769e-05 |
| 16 | 0.976744235 | -5.647378475e-07 | 7.844833630e-06 | -7.757553937e-05 |
| 17 | 0.975036681 | 2.286894232e-06 | 7.609778509e-06 | -7.610883758e-05 |
| 47 | 0.900212348 | 6.660896098e-05 | 4.255143388e-06 | -4.239755949e-05 |
| 67 | 0.797583044 | 1.115058103e-04 | 3.176317876e-06 | -1.916850325e-05 |
| 84 | 0.603773594 | 1.410195439e-04 | 2.298334630e-06 | -1.321316090e-05 |
| 92 | 0.410256416 | 1.425642583e-04 | 2.222938662e-06 | -4.515592861e-05 |
| 97 | 0.198347092 | 3.158942312e-04 | 3.430238382e-06 | -2.009956862e-05 |
| 98 | 0.140350878 | 6.214230627e-04 | 4.552534343e-06 | 1.092506566e-04 |
| 99 | 0.074766353 | 3.972879731e-04 | 2.768177509e-06 | -2.924158283e-05 |

全时刻 integrated remaining witness 约为 **−8.892 个描述性 SEM**，明显偏离零：本有限求解器没有解掉完整 Poisson/连续性残差。接近零的有限 witness 也不足以证明完整求解；这里已有非零剩余见证，不能用正 coupling gain 绕过它。当前 validation 是研究已见、仅对本次参数拟合留出的每类一图；标准误不是精确的类条件 repeated-noise 置信区间。早期坏点与时间采样偏尾同时出现，但本实验不能将二者直接写成已识别因果。

**完整 100-step 成本 benchmark 已完成。** 同一固定 B8 噪声，每臂一次完整 warmup 和三次完整计时，交替顺序、每次重置状态、边界同步；每臂四次 endpoint hash 相同。它实际运行 latent 轨迹，但没有 decoder 查询、输出图像或 FID。

| 分支 | 三次完整计时（秒/B8） | 平均秒/B8 |
|---|---|---:|
| official 100-step | 4.994973088 / 4.999288173 / 5.024813505 | 5.006358255 |
| potential 100-step | 5.234016265 / 5.252569702 / 5.250850114 | 5.245812027 |

候选/官方 wall ratio 为 **1.047829931372**，即**耗时增加 4.782993%**，不是质量提升。候选独有 potential 加载为 **0.010639778 秒**（CPU checkpoint 读取 0.003785855、构建/拷贝 0.006853923）；共同 backbone/decoder 加载 10.221878148 秒，噪声生成/拷贝/审计 1.270407007 秒。全部 warmup+计时轨迹共 **41.292937768 秒**，800 个 B8 Stage-2 forward/6400 个样本 forward，400 次 B8 势输入梯度/3200 个样本输入梯度，无势参数 backward。所有 warmup 与计时均是额外研究成本。

**阶段与总成本的实际边界。** 以下 wall 是各运行真实计时窗口，不混称 CUDA kernel 时间。三个 validation worker 并行，所列加总不是整个阶段的 critical-path wall。`main` 计时均排除 imports 与最后 summary 写入；数据编码只记录较窄窗口，见表后边界。

| 阶段 | 内部 phase/轨迹 wall 秒 | 主程序/已测总窗口 wall 秒 | 主程序 CPU 秒 |
|---|---:|---:|---:|
| train5000 编码（含读取、误差、序列化和 hash） | 90.446105799 | 见合并窗口 | 未记录 |
| validation1000 编码 | 17.396349944 | 见合并窗口 | 未记录 |
| 编码共享未分配开销 | 6.018283299 | 见合并窗口 | 未记录 |
| 合并编码已测窗口 | — | 113.860739042 | 未记录 |
| 正式训练 | 546.359467852 | 564.054960563 | 706.830304508 |
| validation shard0 | 271.833997588 | 289.703252364 | 364.231301467 |
| validation shard1 | 263.074890484 | 282.240858616 | 354.837665015 |
| validation shard2 | 263.708325268 | 281.929269729 | 353.980983741 |
| validation worker 合计 | 798.617213340 | 853.873380709 | 1073.049950223 |
| CPU validation 聚合 | 0.213301748 | 未另记外层 wall | 未记录 |
| 100-step benchmark（含两臂 warmup） | 41.292937768 | 64.221547059 | 65.121010065 |
| CPU prepare-only 重放（额外发生） | — | 9.264727710 | 22.328660000 |

正式训练、validation、benchmark 峰值 GPU allocated 分别为 **4689871872 / 4585167872 / 6302049792 bytes**。完整 validation 实际 3200 个 Stage-2 batch forward/100000 个样本 forward、同数势输入梯度，无势参数 backward。两次 pilot 的额外成本继续按前文列账，不遗漏 v1。

数据共享 **6.018283299 秒不能称为纯初始化**：它含 dataset/model 初始化与释放、GC、mmap 建立、共享哈希等。把全部共享项分配给训练可得编码训练准备已测窗口 **96.464389098 秒**，这是分摊约定；不包括原来未记录的先行 CPU prefix、encode 内开始计时前 CUDA/request 设置等。CPU prefix 重放实际 **9.264727710 秒**、child CPU **22.328660 秒**，选行与原选择完全相同，但它不恢复历史耗时，也未覆盖上述全部缺口。当前不能给出已闭合的全历史总成本；不能把重放值当作原来未测值直接填平。

**配对 1K 已完成，固定候选终止。** 官方100 FID **37.562703724300**，候选100 **37.531664406887**，下降仅 **0.08263334%**，未达预注册至少 5%。实际两组轨迹 worker wall 为 **626.963410812 / 656.521812564 秒**，候选增加 **4.714534%**。完整 paired protocol、noise/hash、FID/IS 与成本见[筛查负结果](RAEV2_OBSERVABLE_POTENTIAL_SCREEN_20260906_ZH.md)。

依据[FID 前成本协议](RAEV2_OBSERVABLE_POTENTIAL_COST_PROTOCOL_20260906_ZH.md)在第一阶段停止：官方105、full-cost baseline 和独立确认 seed202609096 均未运行。没有公平成本达标或 SOTA 结论，不以此前 bridge coupling gain 替代 FID，不自动改宽度、窗口、系数或 seed 救回本候选。此裁决不否定全部 weighted-Poisson 理论，只终止本固定有限求解器；完整 Poisson 残差尚未解的机制边界仍保留。

**冻结身份。** 下列 hash 对应完成的真实产物及当前正式训练 request；其他逐图、元数据和源代码 hash 完整保留在 data/pilot request 与 summary。大数组和模型 hash 引用已核对记录，不重复 GPU 计算。

| 产物 | SHA256 |
|---|---|
| 当前训练 runner | `d0750939786ce15f4fc8b5cb6097902b614ac96852cae33525dce5134dbe0590` |
| 势模块 | `ede326720114f8e60b8c6e47a938b59392a570582dd4c8014fe3f09cdcc4fd75` |
| 原生 guidance_utils | `bcba1a65897faa7545f9e54a8360583dfae34542d52128ec2062d09ee824a577` |
| Stage-2 官方 checkpoint 文件 | `723c56d7fa77ace9613909f7e38cb2386b898608218dc9b52649bb373d513c9a` |
| 官方 config | `3062762f2f0f12e0d4b64b074fc5b45628e5022937bbf7857cc6dc6e2720d342` |
| DINOv3-L encoder 权重 | `8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035` |
| Normalization stats | `40e57d9d38a267dc258043c081094d276382c29ebccc1bd8c38f4dd11e81ba77` |
| Packed manifest | `a3ef360766c3cb0c6ed00639f1a1d23da8a93e37ad04f03217af1830f9756a38` |
| 数据 selection.json | `b3316586c486ffea047493f9ff3f70ec0a30333eb88a7266ae913ab6d445c423` |
| 数据 request.json | `2e09052c91262816e3eeb13070bb3ee095ecbcb690791d3c7e364ea1f4e25e60` |
| 数据 summary.json | `d17d6e22ea47f7d2603f7c6843145abd6c90caf961b90a146fcc864ddc61333b` |
| 数据编码 runner | `f2a0fffa664c3fdd9ab307d01220e5802896b013a5b0653cd2cd5d3e0f9f6d82` |
| Train latents.npy | `b0283d24ab47214983d4904a2c08b62cfc81efb19b6d1459d68e9cea79702f11` |
| Validation latents.npy | `41ff1e73b2ce9ce580aee921a0129d6d32bb8b085c07f79752e3106afc834ad0` |
| Train source rows 数组 | `d63ba55b583afd887f8be0794c40c4eda12df0f333902070ee1158954285ad31` |
| Validation source rows 数组 | `60ce7ac668738b6b1c92a3b1ab858e979bc4f4e6898413412734ccded18c7d31` |
| Pilot v1 summary | `59fe6a82e1c647330fc90aef809dce1b74bc3f9053900fb08185d596d85163f3` |
| Pilot v2 request | `a3a3ea6932d96b99f3a888fcf6aadcea4be57ce37bca743e042477351589f981` |
| Pilot v2 summary | `dad90fc9f1b9156a37d855978d20ecaa72b7867d6b87f48f7965303c4acbd74f` |
| 正式 train request | `4753995cd54afb320772d8473b73f144fdb54ae8b2cad41c7812ee6a2da20443` |
| 固定 final potential checkpoint | `495b3313e945f4b845307ae0d520c3c8d8fa70e60e3983002297cd3f7ccc632e` |
| 正式 train summary | `45e3c0ded8d0ffd7165dc346bc935c30485d89dfa4ee7cf7e81840aeff726521` |
| 实际训练 sampling_counts | `6de452520775f9bb44516f5ce6fd941aedd9e8ac39e9b4ae513a62d7c282a5db` |
| Validation 聚合 request | `410bba45dad37c069c245ee0cf2f793600f8d296ac35dee7610d031df840e5d2` |
| Validation 聚合 summary | `b46cd84af46cde06f927ed8d0adac9b8f8c30991ab6adadbaba7b9680283cf0b` |
| Validation by_time.csv | `23e1b3e46065f36960bbdbf52737fc7f08721f16aeb77571a394386e91ec7272` |
| Validation paired_by_image.npz | `742dc0ba785d43e95ab51e3ccce658aa287eb72f4c68ec1a1ea3e6482e5f3a01` |
| 完整轨迹 benchmark summary | `5c915890019c83fa6de2b31270e21cf004d22efd5c87c495e7100ccf38e2f5c2` |
| 完整轨迹 benchmark.csv | `305c5152152e12c96ab4e631ee787fcc982094ce955998cf012213d2fc571589` |
| CPU prefix replay summary | `d79df1273608ea9913bc42ff67bbbf5ca8942474a8322ba20dc75e1a6c20b0ea` |
| FID 前分阶段筛查 request | `ad6cec9e53b49bb72bf5abf842ab4d35085221bcc000da276d1c7324cd7971cd` |

真实数据：[selection](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/potential_clean_bank_fp32_v1/selection.json)、[request](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/potential_clean_bank_fp32_v1/request.json)、[summary](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/potential_clean_bank_fp32_v1/summary.json)、[train metadata](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/potential_clean_bank_fp32_v1/train/metadata.npz)、[validation metadata](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/potential_clean_bank_fp32_v1/validation/metadata.npz)。Pilot：[v1 summary](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/observable_potential_pilot_v1/summary.json)、[v2 request](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/observable_potential_pilot_v2/request.json)、[v2 summary](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/observable_potential_pilot_v2/summary.json)、[v2 八步记录](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/observable_potential_pilot_v2/pilot_updates.csv)、[v2 参数差分](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/observable_potential_pilot_v2/parameter_derivative.csv)。正式训练：[冻结 request](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/observable_potential_train_v1/request.json)。完整实测：[训练 summary](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/observable_potential_train_v1/summary.json)、[validation summary](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/observable_potential_validation_summary_v1/summary.json)、[100-step benchmark](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/observable_potential_benchmark_v1/summary.json)、[CPU prefix replay](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/potential_cost_prefix_replay_v1/summary.json)、[机器归档](data/raev2_guidance_restart_20260906/observable_potential_solver_audit.json)。后续配对 1K 已完成并按首阶段停止，详见[screen 归档](RAEV2_OBSERVABLE_POTENTIAL_SCREEN_20260906_ZH.md)。
