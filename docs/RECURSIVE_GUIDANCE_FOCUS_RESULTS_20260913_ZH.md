# 两项递归 guidance 改进：短队列实时结果

已提交 9/48 个配置轮次；成功 9。

本次 8 个配置，每个 R0 初始生成及 R1–R5 五轮完整回灌；每轮分别采样、保存 1000 张图片及 latent，并计算 FID/sFID/IS、漂移和图集。

沿用旧筛选的噪声和类别 bank，属于同 bank 的参数细化，不能称为独立确认。下表每行固定同一参数，保持全部六轮，不拼接各轮最优值。

|配置|角色|α / τ|参数|R0|R1|R2|R3|R4|R5|图集|
|---|---|---|---|--:|--:|--:|--:|--:|--:|---|
|`cfg_gain_l2`|candidate|1.25 / 0.5|`{"probe_coefficient": 2.0}`|46.5123|45.6372|45.5231|45.4795|45.6596|45.4235|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_focus_20260913/focused_screen_1k/cfg_gain_l2/rounds.png>)|
|`ig_rotation_norm_preserved`|candidate|0.4 / 0.65|`{"preserve_norm": true, "probe_coefficient": 0.5}`|68.8081|68.4110|66.9270|—|—|—|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_focus_20260913/focused_screen_1k/ig_rotation_norm_preserved/rounds.png>)|
|`cfg_native_a1p24`|control|1.24 / 0|`{}`|—|—|—|—|—|—|—|
|`ig_amplitude_only`|control|0.4 / 0.65|`{"norm_only": true, "probe_coefficient": 0.5}`|—|—|—|—|—|—|—|
|`cfg_gain_l0p5`|candidate|1.25 / 0.5|`{"probe_coefficient": 0.5}`|—|—|—|—|—|—|—|
|`ig_extrapolate_l0p25`|candidate|0.4 / 0.65|`{"probe_coefficient": 0.25}`|—|—|—|—|—|—|—|
|`cfg_gain_l4`|candidate|1.25 / 0.5|`{"probe_coefficient": 4.0}`|—|—|—|—|—|—|—|
|`ig_extrapolate_l0p75`|candidate|0.4 / 0.65|`{"probe_coefficient": 0.75}`|—|—|—|—|—|—|—|

旧实验已完成的外部参考配置（引用原有结果，不计为本次运行或独立验证）：

|旧配置|α|R0|R1|R2|R3|R4|R5|原始记录|
|---|--:|--:|--:|--:|--:|--:|--:|---|
|`ig_mlp_03`|0.8|62.6471|62.7819|64.0699|63.1785|66.0306|66.4514|[原始目录](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_mlp_03>)|
|`cfg_native_03`|1.25|46.5318|45.6094|45.4965|45.4946|45.6927|45.6081|[原始目录](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/cfg_native_03>)|
|`ig_mlp_00`|0.2|74.4310|75.8979|74.0879|72.2448|73.8296|69.2679|[原始目录](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_mlp_00>)|
|`ig_mlp_01`|0.4|68.9886|68.4676|66.6231|64.8171|66.2802|63.3657|[原始目录](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_mlp_01>)|
|`i07_ig_cycle_extrapolate_01`|0.4|66.7281|65.3703|63.7548|62.1266|63.8960|61.4378|[原始目录](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i07_ig_cycle_extrapolate_01>)|
|`ig_mlp_02`|0.6|64.9661|64.6367|63.4066|62.3933|64.0818|61.9321|[原始目录](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_mlp_02>)|
|`ig_mlp_04`|1|62.1206|63.9684|66.6819|67.2625|72.9170|75.2969|[原始目录](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_mlp_04>)|
|`ig_mlp_05`|1.2|62.7729|65.8365|70.8648|74.7118|81.9824|85.8613|[原始目录](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_mlp_05>)|
|`ig_mlp_adg_00`|0.4|68.0159|68.3563|66.9547|63.4863|66.3547|63.4212|[原始目录](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_mlp_adg_00>)|
|`ig_mlp_adg_01`|0.6|63.7830|64.9644|61.9098|59.9445|62.2022|60.4315|[原始目录](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_mlp_adg_01>)|
|`i01_cfg_cycle_gain_04`|1.25|46.4903|45.6019|45.4515|45.4847|45.6955|45.4041|[原始目录](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i01_cfg_cycle_gain_04>)|
|`ig_mlp_adg_02`|0.8|62.0624|62.3056|61.3854|60.5270|62.4550|62.3183|[原始目录](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_mlp_adg_02>)|
|`ig_mlp_adg_03`|1|61.6620|62.3234|62.6133|61.8465|65.1044|65.2377|[原始目录](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_mlp_adg_03>)|
|`ig_mlp_adg_04`|1.2|62.0750|63.9603|64.6728|64.9418|68.2895|69.3148|[原始目录](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_mlp_adg_04>)|
|`cfg_native_00`|0.25|64.9212|66.2444|64.3242|—|—|—|[原始目录](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/cfg_native_00>)|

[筛选依据与对照摘要](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_focus_20260913/baseline_report.json>)。

输出：`/home/zhoushunyu/data/eqvae/experiments/recursive_guidance_focus_20260913/focused_screen_1k`。

[每配置每轮完整 CSV](</home/zhoushunyu/eqvae/docs/data/recursive_guidance_focus_20260913/all_results.csv>)。
