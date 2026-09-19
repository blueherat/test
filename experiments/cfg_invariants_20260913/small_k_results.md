# K=.03 的定向补验

作者默认 K=.3 在 SiT velocity 单位上出现过强反转，因此仅补 K=.03 的三个配置，其他参数不变。
这是确定量纲敏感性的补验，不是新一轮广泛调参。每个配置重新生成同一 bank 的400张，单输出224次前向；FID与分类器实算。
vanilla基线来自主屏同噪声/类别，不重复计为新采样。小样本 FID 只筛选。
ctrl_physical 使用原始 gap 历史与物理时间导数，而作者保存 modified gap；差异不能全部归因参数单位。
历史在 Heun 两个 stage 内冻结，仅接受步后提交左端 proposal。单位变换的代数核查仅适用于可逆的中间时刻；clean 的 t=1 与 epsilon 的 t=0 是奇异端点。

|arm|FID400↓|top1↑|mean extra∥|extra norm/native|off-gap|
|---|---:|---:|---:|---:|---:|---:|
|cfg_base|79.7772|0.8825|1.2500|1.0000|0.0000|
|ctrl_author|83.4461|0.8225|1.0289|1.0109|0.5438|
|ctrl_gap_project|84.7650|0.8075|0.8513|0.6811|0.0000|
|ctrl_physical|84.6260|0.8275|1.0411|1.0289|0.5510|

产物：[small_k_0p03](</home/zhoushunyu/data/eqvae/experiments/cfg_invariants_20260913/images/small_k_0p03>)；[同seed图集](</home/zhoushunyu/data/eqvae/experiments/cfg_invariants_20260913/images/small_k_0p03/comparison_grid.png>)；[CSV](</home/zhoushunyu/data/eqvae/experiments/cfg_invariants_20260913/images/small_k_0p03/results.csv>)。

<!-- FINAL_RESULT_AUDIT -->

CSV 的 sample_seconds 与 classifier_seconds 分开，FID 耗时未记录，因此 fid_seconds 留空；采样成本逐项取原始 summary.json，未使用分类器耗时替代。
