# RAEv2：解码协方差形状的固定跨类检验

日期：2026-09-06。**本次完整谱检验未确认可从 800 训练类迁移到 200 留出类的 IG 协方差形状误差方向。** 两个旧 seed 的主差区间均跨零，训练／留出误差矩阵的 Frobenius cosine 约为 0.012。不据此挑选秩或方向，也未产生 guidance、训练、采样或新 FID。

**先固定协议，再读取特征。** 两个既有 `n5000_seed20260801/20260802_scales7_v1` bank 均为 1000 类、每类 5 张，沿用各自原 800/200 类划分。原始 FP32 Inception 特征提升到 FP64；`source`、`real`、`scale_s1p000000`、`scale_s1p780000` 分别对应原图、重构、Full、IG。按 `global_id=rank+4×local_row` 还原，标签和划分来自 `sample_protocol.npz`。全部 32 个特征分片、两个 manifest 和两个 protocol 的 SHA 与已完成[类矩审计](RAEV2_DECODED_CLASS_MOMENT_AUDIT_20260906_ZH.md)一致。

对各 seed 的训练类，以 ddof=1 covariance 定义

\[
D_{\rm train}=\frac{\Sigma_{IG}}{\operatorname{tr}\Sigma_{IG}}
-\frac{\Sigma_{source}}{\operatorname{tr}\Sigma_{source}},
\qquad K=D_{\rm train}/\|D_{\rm train}\|_F.
\]

该定义剥离已知的总迹差，保留全部 2048 个谱方向；没有逆协方差、白化、正则化、裁负或选秩。固定 K 后，留出类四分支全部计算

\[
T_A=\frac{\operatorname{tr}(K\Sigma_A)}{\operatorname{tr}\Sigma_A}.
\]

主差为 `T_IG−T_source`。另四项对比在 request 中预先固定、全部报告。每类五张先聚合 covariance-trace 比率的一阶 influence，再按相同类别相减，SEM 为 200 类 influence 的标准差除以 \(\sqrt{200}\)。区间为均值 ±1.96 SEM，仅描述固定训练 K 后的类别异质性；不是条件于这些固定类别的精确 repeated-noise 置信区间。两个 bank 和旧划分早已用于研究，本次是回顾性的跨类迁移检查。

**结果。** 下表所有数值均乘 \(10^3\)，方括号为描述性 95% 区间。

| 留出对比 | seed 20260801 | seed 20260802 |
|---|---:|---:|
| IG − 原图：主差 | 0.5076 [−0.3162, 1.3313] | 0.5337 [−0.7448, 1.8122] |
| IG − 重构 | 0.2663 [−0.5835, 1.1160] | 0.1935 [−1.0821, 1.4691] |
| Full − 原图 | 0.7784 [−0.0356, 1.5924] | 0.0975 [−0.9210, 1.1161] |
| IG − Full | −0.2708 [−0.8595, 0.3178] | 0.4362 [−0.4351, 1.3074] |
| 重构 − 原图 | 0.2413 [−0.0430, 0.5256] | 0.3402 [−0.0136, 0.6940] |

训练 D 的 Frobenius norm 为 0.022550／0.022479；留出 D 为 0.042881／0.044580，不能把训练拟合量当泛化信号。训练／留出 D 的 cosine 为 **0.011837／0.011972**；将留出参考换成重构后为 **0.006283／0.004329**。IG−Full 对比的均值还跨 seed 反向，两个区间均跨零。全部训练特征值及留出误差在同一完整特征基中的 2048 个对角条目均保存，没有事后抽取谱子集。

**与旧证据合并后的边界。** 现有类矩审计中，IG 的修正类间散布相对原图仍约为 0.913／0.921，相对重构约为 0.930／0.929；这确实是跨 seed 同向的标量缺口。类内迹已接近原图，不能据此提出统一方差扩张。[四格审计](RAEV2_PREDICTED_CLEAN_INTERACTION_AUDIT_20260906_ZH.md)则没有检出交互项 I 在误差轴上的稳定方向。本次没有从这些标量缺口中获得可迁移的协方差形状对象。

这不证明总体 covariance 已正确，也不能排除类特有结构或有限样本噪声掩盖信号。它只说明本次事先固定的完整谱形状诊断没有提供足以推进新结构的方向证据。即使发现显著 shape 差，也仍不等于质量改善方向、实际轨迹原因或 FID 保证。历史缓存使用 float 后 clamp／量化的 decoder 路径，未改变当前 native BF16 基线协议。

**产物与核对。** 输出根为 `/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/decoded_covariance_shape_v1/`。其中 [request](</home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/decoded_covariance_shape_v1/request.json>) 在数值特征读取前落盘；[独立源码](</home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/decoded_covariance_shape_v1/runner_source.py>)、[summary](</home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/decoded_covariance_shape_v1/summary.json>)、完整谱 NPZ、逐类 influence NPZ 和输入／输出 hash 均保留。

另一个不调用 producer 函数的[轻量复核](</home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/decoded_covariance_shape_v1/independent_check.json>)通过 97 项 hash／统计恒等检查，最大绝对数值残差为 \(5.56\times10^{-17}\)。它从保存的全谱与逐类 influence 重算主差、全部 SEM 和区间，没有独立从原特征重建 covariance，不能表述成第二次完整原始统计复算。

主分析 **5.256 秒 wall、17.815 秒 CPU**，峰值 RSS **957908 KiB（约 0.914 GiB）**；轻量复核另为 **0.577 秒 wall、0.888 秒 CPU**。均为 0 GPU、0 模型调用。计时从各脚本导入 time 后开始，主分析包含 SHA、读取、完整谱和全部统计；不含之前的解释器启动及最后 summary 写入／退出，也不含历史 bank 生成成本。

`summary.json` SHA256：`d3cfdf2f263f9ca2961134c180045480539eee43dfc02374f1b2dc05e5d8f9ba`。
