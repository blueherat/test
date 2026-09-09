# 一次写入后撤条件：固定1K质量筛查

2026-09-08。64图机制pilot已经观察到48/64的类别保留和同校准起点条件/null终点差减少。现在检验生成质量，不能将此前classifier结果称为质量收益。

冻结t=1两轮校准，H=.125、gamma=2：条件CFG前向、null反向Euler，随后100步全部null。与此前pilot同一方法，不调H/K/强度。正式筛查改为生产BF16+TF32、B4、100步shift8，使用固定seed202609413、labels0..999，每类一图，与仓库现有1K对照配对。该seed是已有筛查bank，不是新的独立确认集。

先完成8图FP32写入状态及最终像素对上一轮pilot逐位复现，再分别8图复现已有Full条件及原生IG1.78，另做候选BF16 8图smoke并核对正式1K前8图。所有1K输入noise/label/checkpoint SHA必须与对照一致。

对照复用 `raev2_pfr_working_point_20260908/full/quality` 的Full条件100，以及 `raev2_fsg_clock_transfer_20260908/quality/ordinary100` 的原生IG100。复用只是避免重生成相同基准，并不将PFR方法带入当前路线。候选106次Full，对照100次Full；本轮不是严格同计算量比较。若候选值得扩展，最终须补同预算对照；不能直接用小幅FID差称为效率或方法突破。

官方nanogen evaluator、ImageNet256固定reference；缓存Inception features，独立CPU FP64重算FID mean/covariance项。没有以1K FID作为最终成功标准，候选有效还需独立seed和更大样本、成本及机制对照；当前不启动5K、不写论文。

执行 `experiments/run_raev2_handoff_quality.py`，采样 `sample_raev2_handoff_quality.py`，独立分析 `analyze_raev2_handoff_quality.py`。输出 `/home/zhoushunyu/data/eqvae/experiments/raev2_handoff_quality_20260908/`。协议写于正式结果前，当前未得出质量结论。

## 完成后的结果（协议参数未变）

执行93827退出0，独立审计99805退出0；pilot、基准像素复现、smoke/formal前8图和配对输入检查通过。官方FID与独立FP64复算误差小于2e−4。

|方法|FID 1K↓|均值项|协方差项|Full调用/图|采样及保存秒数|
|---|---:|---:|---:|---:|---:|
|写入后全部null|45.726964|3.590190|42.136791|106|856.562|
|普通Full条件|38.874134|.269684|38.604474|100|804.146|
|原生IG|38.264239|.152374|38.111889|100|802.983|

结果不支持该固定写入方案的质量收益。先前64图的类别保留仍是机制证据，不能替代质量证据；这也不是对原论文完整采样算法的否定。只完成此前已启动的CFG参考，不继续将此配方扩至5K，不把写入条件本身当作创新。下一条路线独立研究AG/IG的有效纠错能否由状态保留。

可复核结果：`experiments/results/terminal_defect_20260908/raev2_handoff_quality.csv`。1K为已有筛查bank，不能据此断言所有写入方案都失败。
