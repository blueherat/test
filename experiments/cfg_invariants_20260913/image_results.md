# CFG 不变量：真实图像筛查

实际完成 13/13 个配置；每个 400 张，共同初始噪声和类别。
SiT-S/2，Heun64，t=0 为噪声、t=1 为图像；guidance 在左端 t<0.75 开启。
每输出固定 224 次单分支前向；CFG 默认 extra α=1.25（总权重 w=2.25），half 对照 α=.625。

这是普通生成采样轨迹的 5 个时间点，不是回灌，未假定图片恒等。
作者 CFG-CTRL 使用公开代码默认 λ=.05、K=.3，移植到 SiT velocity；Heun 两次查询冻结历史，接受后只提交左端 proposal。不是论文设置复现。
ctrl_physical 使用原始 gap 的物理时间导数和 λ_time=3.2；历史按当前参数单位运输，velocity/clean/epsilon 等价由 CPU 检查。它与作者版本还存在 raw vs modified 历史区别。
fixedset_projection 是当前固定集合 {a·gap:0≤a≤1.25} 的正交投影；单 gap 时仅为标量截断，不能宣称新方向。跨时间集合变化，不承诺整条轨迹幂等。
ctrl_direction_matched 将相对 conditional 的额外向量范数逐样本配平至 vanilla 的 α||gap||；它用于检查方向效应。
gap/fixedset/norm/evidence 的 time-mean 对照来自独立 100 张校准 bank 的各自轨迹，未用质量指标选参；只能配平时间平均增益，不能使各样本轨迹相同。
clean_evidence_proxy 在 t=12/64、24/64、36/64 解码条件 clean 预测并用 ConvNeXt 读取目标类概率，gain=2α(1−q)，更新之间保持；初值 q=.5。这是 clean 图像代理，不是 noisy 后验，也不是生成模型自身后验。每输出额外 3 次 VAE 与分类器读取。

FID400 只用于小样本筛选，有显著样本数偏差；分类器读数也不是图像质量真值。evidence arm 的终点分类器与反馈分类器相同，不能当独立语义验证。100 张校准每类别仅 1 张，时间平均配平精度有限。需优先比较配平对照，不能仅将 guidance 变弱记为结构收益。

|arm|FID400↓|top1↑|target p↑|mean extra ∥|extra norm/native|off-gap|
|---|---:|---:|---:|---:|---:|---:|
|cfg_base|79.7772|0.8825|0.6181|1.2500|1.0000|0.0000|
|cfg_half|90.5327|0.7675|0.4767|0.6250|0.5000|0.0000|
|ctrl_author|208.1370|0.0975|0.0506|-0.0686|5.3143|0.9648|
|ctrl_gap_project|138.8065|0.0000|0.0002|-3.6878|2.9502|0.0000|
|ctrl_direction_matched|133.5908|0.4225|0.2443|-0.0132|1.0000|0.9654|
|ctrl_physical|162.7446|0.1200|0.0636|-0.2236|4.6254|0.9714|
|fixedset_projection|92.3827|0.7300|0.4715|0.6802|0.5441|0.0000|
|norm_feedback|78.9014|0.8900|0.6323|1.3506|1.0804|0.0000|
|norm_time_mean|79.2537|0.8875|0.6341|1.3472|1.0777|0.0000|
|clean_evidence_proxy|78.5666|0.9250|0.6786|1.6313|1.3051|0.0000|
|evidence_time_mean|78.6571|0.9050|0.6601|1.6074|1.2859|0.0000|
|gap_time_mean|138.0146|0.0000|0.0002|-3.6887|2.9510|0.0000|
|fixedset_time_mean|91.9574|0.7475|0.4794|0.6800|0.5440|0.0000|

产物：[CSV](</home/zhoushunyu/data/eqvae/experiments/cfg_invariants_20260913/images/results.csv>)、[配平对比](</home/zhoushunyu/data/eqvae/experiments/cfg_invariants_20260913/images/comparisons.json>)、[同 seed 图集](</home/zhoushunyu/data/eqvae/experiments/cfg_invariants_20260913/images/comparison_grid.png>)。
每个 arm 下有 grid.png、trajectory_grid.png、trajectory_5_times.npz、samples.npz、diagnostics.npz、FID 与 classifier 文件。
运行：tmux session cfg_invariants_images_0913；GPU 0/1；旧队列未恢复。

<!-- FINAL_RESULT_AUDIT -->

最终复核：

CSV 将 sample_seconds、classifier_seconds、fid_seconds 分开。采样时间包含最终解码及每组40张轨迹可视化解码；证据反馈的3次额外读取已包含在该组采样时间中。ADM 工具没有记录耗时，fid_seconds 留空。
正式生成并评估 6400 张；另有 500 条校准轨迹。采样设备秒合计 712.89s、校准 56.11s；这是并行各配置时间之和，非墙钟耗时。
fixedset_projection 与 ctrl_direction_matched 是作者控制器输出上的包装，保留作者原始 modified-gap proposal 为历史；ctrl_gap_project 将投影后 gap 提交为历史。ctrl_physical 保存 raw gap，且使用物理时间导数。
velocity/clean/epsilon 单位运输的等价核查在可逆中间时刻成立；epsilon 的 t=0 和 clean 的 t=1 为奇异端点，不要求逆变换。

|arm|K|sample s|classifier s|
|---|---:|---:|---:|
|cfg_base|—|43.30|1.74|
|cfg_half|—|43.04|1.69|
|ctrl_author|0.3|43.10|0.76|
|ctrl_gap_project|0.3|42.88|0.76|
|ctrl_direction_matched|0.3|43.26|0.75|
|ctrl_physical|0.3|43.84|0.79|
|fixedset_projection|0.3|43.38|0.74|
|norm_feedback|—|42.89|0.86|
|norm_time_mean|—|43.25|0.79|
|clean_evidence_proxy|—|63.20|0.75|
|evidence_time_mean|—|43.31|0.76|
|gap_time_mean|—|42.84|0.75|
|fixedset_time_mean|—|43.56|0.75|
|ctrl_author|0.03|43.56|1.23|
|ctrl_gap_project|0.03|43.40|0.71|
|ctrl_physical|0.03|44.07|0.70|

[16组总CSV](</home/zhoushunyu/data/eqvae/experiments/cfg_invariants_20260913/images/all_image_results.csv>)；[实际预算](</home/zhoushunyu/data/eqvae/experiments/cfg_invariants_20260913/images/evaluation_costs.json>)；[小K三组](</home/zhoushunyu/data/eqvae/experiments/cfg_invariants_20260913/images/small_k_0p03/image_results.md>)。

本轮结论：vanilla CFG 的 FID400 为 79.7772；norm-feedback 为 78.9014，但相对时间平均对照仅低 0.3524；clean 证据代理为 78.5666，相对时间平均对照仅低 0.0905，不能认定图像分布质量收益。固定集合投影与 gap 投影均未超过各自时间平均对照。K=.03 补验下，作者为 83.4461，gap 投影为 84.7650、physical 为 84.6260，均未超过 vanilla；因此不能把不变量满足直接推为质量改进。
