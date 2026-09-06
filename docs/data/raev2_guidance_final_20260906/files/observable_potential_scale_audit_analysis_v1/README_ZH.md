# 配对 5K FID：局部一阶误差分析

这份分析只读官方 evaluator 已完成的三臂特征，给出固定 reference、固定 1000 类且每类五个噪声重复下的配对 FID 差和相对改善的一阶 SE。它不生成图像、不提取特征、不使用 GPU，也不改变方法或决定是否达标。

## 已有数学核验

`fid_influence.py` 提供数学函数。`toy_validation.json` 已成功完成 96 项低维检查，最大 scaled absolute error 为 `4.424856106580879e-7`，包含独立 `scipy.linalg.sqrtm` FID、均值/协方差方向导数、归一化污染权重 IF、同类配对差与 ratio 的有限差分、独立成对差方差公式、相同两臂零 SE，以及四种退化输入停止。2026-09-06 接手审阅时，两份源文件 SHA 与这次成功测试完全相符；没有重复执行成功测试。

对每臂特征，令 `r_i=x_i−μ`、`S=Σ r_i r_iᵀ/(N−1)`，`T=S^(−1/2)(S^(1/2) R S^(1/2))^(1/2)S^(−1/2)`。所用 IF 是

```
IF_i = 2(μ−μ_ref)ᵀr_i + [N/(N−1)] r_iᵀ(I−T)r_i − tr[(I−T)S].
```

这里 `N/(N−1)` 来自固定 N 的样本协方差函数；不能在读到结果后去掉它。配对绝对差使用 `IF_candidate−IF_baseline`，相对改善 `1−FID_candidate/FID_baseline` 使用相应 delta method。先按同 ID 配对，再计算每类五个值的类内样本方差，最后以 `Σ_c (m_c/N)² s_c²/m_c` 合并。**不是**把五个 1K 子集 FID 当成独立重复。

所有矩阵均用 FP64。样本、reference 和夹心矩阵必须数值正定，transport 的相对残差须 ≤1e−8；不满足则停止，禁止加 ridge、截断秩或事后改参数。实现仅消除算术造成的反对称舍入误差。

## 入口

在三臂共同 evaluation 完成后运行 `run_scale_analysis.py`。入口首先要求 `execution_summary.json` 为完成且 evaluator return code 为 0，校验官方 metrics 的 SHA，随后验证三臂样本 SHA、merge summary、ID/label、reference SHA 和 feature cache key。读取样本 NPZ 时只解压 `ids` 与 `labels`，不解压 `arr_0` 图像。输出目录必须是本 analysis 目录内不存在的子目录。

参数：

```
--arm NAME FEATURE_PT SAMPLES_NPZ MERGE_SUMMARY  # 三次，名字固定如下
--evaluation-request /.../evaluation/request.json
--metrics-json /.../evaluation/metrics.json
--execution-summary /.../execution_summary.json
--reference /.../imagenet_256_fid_stats.npz
--output-dir /.../observable_potential_scale_audit_analysis_v1/results_v1
```

三个名字为 `official100`、`potential100`、`official105`；seed 固定 `202609101`。特征为原始 `torch.float32[5000,2048]`，文件名必须是 `{NAME}_seed202609101-{sample_sha256[:16]}-inception.features.pt`。各自样本和 summary 为 `{NAME}/merged/samples.npz` 与 `{NAME}/merged/summary.json`。库不在运行时寻找其他种子、reference 或候选。

输出 `request.json`（输入、来源和口径）、`influences.npz`（全部配对 IF、每类方差、联合臂协方差）、`summary.json`（官方/CPU FID 一致性、SPD 审计、两个预定候选对照、SE、局部正态区间和成本）。CPU 重算 FID 与官方值绝对差超过 `1e−5` 会停止。该容差只检查计算一致性，不表示统计精度。

`--help` 和语法已核对。真正 2048 维运行前不宣称实际 covariance 已通过 SPD 检查。没有读取本轮任何新 5K 特征或部分 FID。

## 解释边界

- 这是局部一阶 influence / delta-method 近似。2048 维、N=5000 时，高维二阶项可能显著；toy 检查不证明正态区间具有精确的 95% 覆盖率。
- SE 只覆盖固定类别配额内的噪声变化；reference、候选训练、协议固定。没有加入 reference 估计、辅助训练及方法选择的不确定性。
- 不消除模型依赖的有限样本 FID 偏差，不把 5K 结果外推到 50K 或总体，不扣除所谓 FID floor。
- 特征缓存本身没有 ID；特征与样本顺序的联系依赖已核对的 evaluator 顺序、cache key 和输入样本 SHA，不是特征内嵌的独立身份校验。
- 不能替代独立噪声确认、足够规模评估和完整准备/训练成本比较。局部区间无论结果如何，都不能单独完成 ≥5% 目标。
