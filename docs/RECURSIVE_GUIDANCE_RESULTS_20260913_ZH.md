# 递归再生成 CFG / IG：初次生成与五轮完整回灌

已提交 **501/1554 个配置轮次**；成功 501，数值失败 0，前轮失败导致未运行 0，其他失败 0。

成功完成整条链 **83/259**；含失败状态已归档整条链 83/259。计划 200 个候选配置和 59 个对照，每个固定配置 6 轮、每轮 1,000 张，共 1,554,000 张计划输出。

R0 是初次完整生成；R1–R5 分别将本配置上一轮完整 latent 加噪到 t=0.25 后运行采样后缀，每轮分别保存图像、latent、FID/sFID/IS、漂移和图集。内部再生成探针不计作完整回灌轮。

所有配置在相同轮次共享噪声和类别；六轮具有共同样本祖先，不能视为彼此独立样本。FID 排名来自 1K 调参筛选；漂移小不等于质量或多样性好。

下面每行始终是同一个固定配置的完整曲线。摘要仅在已成功完成的完整链中按末轮 FID 选择一个配置，随后展示该配置所有轮次；尚无完整链时展示请求顺序中首条已启动链。不拼接各轮不同参数的最优值。

|轮次|成功|数值失败|前轮失败未运行|尚未提交|
|---|--:|--:|--:|--:|
|R0|84|0|0|175|
|R1|84|0|0|175|
|R2|84|0|0|175|
|R3|83|0|0|176|
|R4|83|0|0|176|
|R5|83|0|0|176|

## 十项候选的固定配置曲线

|对象|固定配置|路线/参考|α / τ|成功轮数|FID R0|FID R1|FID R2|FID R3|FID R4|FID R5|选择说明|图集|
|---|---|---|---|--:|--:|--:|--:|--:|--:|--:|---|---|
|1. 按再生成方向增益限制CFG（整链 7/20）|`i01_cfg_cycle_gain_04`|CFG/native|1.25 / 0.5|6/6|46.4903|45.6019|45.4515|45.4847|45.6955|45.4041|按末轮FID选完整链|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i01_cfg_cycle_gain_04/rounds.png>)|
|6. 按再生成前后的强弱差方向一致性门控（整链 7/20）|`i06_ig_cycle_agreement_04`|IG/mlp|0.6 / 0.5|6/6|71.7381|72.4040|70.0969|67.4896|68.5473|65.3594|按末轮FID选完整链|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i06_ig_cycle_agreement_04/rounds.png>)|
|2. 用再生成传输后的条件方向引导（整链 7/20）|`i02_cfg_cycle_transport_05`|CFG/native|1.25 / 0.65|6/6|47.4256|46.5191|45.9861|46.2381|46.4361|45.8452|按末轮FID选完整链|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i02_cfg_cycle_transport_05/rounds.png>)|
|7. 对回灌造成的强弱差旋转作有限外推（整链 7/20）|`i07_ig_cycle_extrapolate_01`|IG/mlp|0.4 / 0.65|6/6|66.7281|65.3703|63.7548|62.1266|63.8960|61.4378|按末轮FID选完整链|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i07_ig_cycle_extrapolate_01/rounds.png>)|
|3. 抑制与再生成漂移同向的CFG分量（整链 6/20）|`i03_cfg_cycle_antidrift_04`|CFG/native|1.25 / 0.5|6/6|46.5176|45.5766|45.4610|45.4561|45.6791|45.6072|按末轮FID选完整链|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i03_cfg_cycle_antidrift_04/rounds.png>)|
|8. 限制弱头相对强头的额外响应放大（整链 6/20）|`i08_ig_cycle_weakgain_04`|IG/mlp|0.6 / 0.5|6/6|64.9759|64.6611|63.4278|62.3620|64.0803|61.9502|按末轮FID选完整链|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i08_ig_cycle_weakgain_04/rounds.png>)|
|4. 校正条件与null的共同横向漂移（整链 6/20）|`i04_cfg_cycle_common_drift_04`|CFG/native|1.25 / 0.5|6/6|47.1601|45.9901|45.7756|45.7044|45.7840|46.1428|按末轮FID选完整链|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i04_cfg_cycle_common_drift_04/rounds.png>)|
|9. 按相反噪声下强弱差的一致性门控（整链 6/20）|`i09_ig_cycle_antithetic_10`|IG/mlp|0.8 / 0.8|6/6|73.5574|75.3026|73.7183|72.2769|73.0878|69.1132|按末轮FID选完整链|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i09_ig_cycle_antithetic_10/rounds.png>)|
|5. 按同噪声再生成非线性限制CFG（整链 6/20）|`i05_cfg_cycle_curvature_04`|CFG/native|1.25 / 0.5|6/6|46.6315|45.6243|45.5029|45.5704|45.5118|45.2680|按末轮FID选完整链|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i05_cfg_cycle_curvature_04/rounds.png>)|
|10. 按两轮回灌的方向保持和增长门控（整链 6/20）|`i10_ig_cycle_twohop_10`|IG/mlp|0.8 / 0.8|6/6|66.2541|66.3556|65.1081|62.7914|64.0165|61.8301|按末轮FID选完整链|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i10_ig_cycle_twohop_10/rounds.png>)|

## 对照的固定配置曲线

|对象|固定配置|路线/参考|α / τ|成功轮数|FID R0|FID R1|FID R2|FID R3|FID R4|FID R5|选择说明|图集|
|---|---|---|---|--:|--:|--:|--:|--:|--:|--:|---|---|
|strong|`strong_00`|IG/native|0 / 0|6/6|83.8286|85.9553|85.2657|82.4875|85.0947|81.4587|按末轮FID选完整链|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/strong_00/rounds.png>)|
|ig_mlp|`ig_mlp_02`|IG/mlp|0.6 / 0|6/6|64.9661|64.6367|63.4066|62.3933|64.0818|61.9321|按末轮FID选完整链|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_mlp_02/rounds.png>)|
|cfg_native|`cfg_native_03`|CFG/native|1.25 / 0|6/6|46.5318|45.6094|45.4965|45.4946|45.6927|45.6081|按末轮FID选完整链|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/cfg_native_03/rounds.png>)|
|ig_native|`ig_native_01`|IG/native|0.4 / 0|6/6|68.7402|68.4128|67.5274|65.0558|67.4446|64.4138|按末轮FID选完整链|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_native_01/rounds.png>)|
|ig_mlp_adg|`ig_mlp_adg_01`|IG/mlp|0.6 / 0|6/6|63.7830|64.9644|61.9098|59.9445|62.2022|60.4315|按末轮FID选完整链|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_mlp_adg_01/rounds.png>)|
|cfg_apg|—|—|—|0/6|—|—|—|—|—|—|尚未启动|—|
|cfg_ctrl|—|—|—|0/6|—|—|—|—|—|—|尚未启动|—|
|cfg_more_steps|—|—|—|0/6|—|—|—|—|—|—|尚未启动|—|
|ig_mlp_more_steps|—|—|—|0/6|—|—|—|—|—|—|尚未启动|—|
|norm_i02|—|—|—|0/6|—|—|—|—|—|—|尚未启动|—|
|norm_i03|—|—|—|0/6|—|—|—|—|—|—|尚未启动|—|
|norm_i04|—|—|—|0/6|—|—|—|—|—|—|尚未启动|—|
|norm_i07|—|—|—|0/6|—|—|—|—|—|—|尚未启动|—|
|reverse_i03|—|—|—|0/6|—|—|—|—|—|—|尚未启动|—|
|reverse_i04|—|—|—|0/6|—|—|—|—|—|—|尚未启动|—|
|reverse_i07|—|—|—|0/6|—|—|—|—|—|—|尚未启动|—|
|cfg_probe_only|—|—|—|0/6|—|—|—|—|—|—|尚未启动|—|
|ig_probe_only|—|—|—|0/6|—|—|—|—|—|—|尚未启动|—|

## 已展示配置的逐轮质量、成本与漂移

Full/prefix 和辅助调用按该轮每张输出计；GPU 秒是采样与解码累计 batch 时间。像素 MSE 单位为 uint8 强度平方；PSNR 由该轮平均 MSE 计算，∞ 表示所有图像与参考逐像素相同，— 表示未测或缺失。它们衡量保真/漂移，不是独立质量评分。CSV 还保留逐图 PSNR 均值及无穷计数。

|固定配置|轮次|sFID|IS|Full/prefix（辅助）|GPU秒|MSE前轮 / R0|PSNR前轮 / R0 dB|本轮图集|
|---|---|--:|--:|---|--:|---|---|---|
|`i01_cfg_cycle_gain_04`|R0|207.172|64.119|244/0 （20）|175.607|0.000 / 0.000|∞ / ∞|[R0](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i01_cfg_cycle_gain_04/round00/gallery.png>)|
|`i01_cfg_cycle_gain_04`|R1|208.029|64.209|176/0 （16）|130.277|2294.519 / 2294.519|14.524 / 14.524|[R1](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i01_cfg_cycle_gain_04/round01/gallery.png>)|
|`i01_cfg_cycle_gain_04`|R2|207.863|63.383|176/0 （16）|130.126|2291.880 / 3057.064|14.529 / 13.278|[R2](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i01_cfg_cycle_gain_04/round02/gallery.png>)|
|`i01_cfg_cycle_gain_04`|R3|209.578|62.607|176/0 （16）|129.827|2280.845 / 3572.164|14.550 / 12.601|[R3](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i01_cfg_cycle_gain_04/round03/gallery.png>)|
|`i01_cfg_cycle_gain_04`|R4|206.905|64.128|176/0 （16）|131.058|2309.716 / 3926.490|14.495 / 12.191|[R4](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i01_cfg_cycle_gain_04/round04/gallery.png>)|
|`i01_cfg_cycle_gain_04`|R5|208.097|66.438|176/0 （16）|130.139|2295.756 / 4137.100|14.522 / 11.964|[R5](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i01_cfg_cycle_gain_04/round05/gallery.png>)|
|`i06_ig_cycle_agreement_04`|R0|212.474|33.676|134/0 （6）|109.738|0.000 / 0.000|∞ / ∞|[R0](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i06_ig_cycle_agreement_04/round00/gallery.png>)|
|`i06_ig_cycle_agreement_04`|R1|211.204|34.492|100/0 （4）|83.826|2396.722 / 2396.722|14.335 / 14.335|[R1](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i06_ig_cycle_agreement_04/round01/gallery.png>)|
|`i06_ig_cycle_agreement_04`|R2|210.466|35.770|100/0 （4）|83.376|2353.776 / 3187.942|14.413 / 13.096|[R2](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i06_ig_cycle_agreement_04/round02/gallery.png>)|
|`i06_ig_cycle_agreement_04`|R3|210.347|35.309|100/0 （4）|83.920|2362.088 / 3731.683|14.398 / 12.412|[R3](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i06_ig_cycle_agreement_04/round03/gallery.png>)|
|`i06_ig_cycle_agreement_04`|R4|207.136|36.330|100/0 （4）|83.967|2385.041 / 4077.087|14.356 / 12.027|[R4](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i06_ig_cycle_agreement_04/round04/gallery.png>)|
|`i06_ig_cycle_agreement_04`|R5|208.035|37.352|100/0 （4）|83.216|2332.660 / 4299.195|14.452 / 11.797|[R5](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i06_ig_cycle_agreement_04/round05/gallery.png>)|
|`i02_cfg_cycle_transport_05`|R0|207.375|61.516|244/0 （20）|182.223|0.000 / 0.000|∞ / ∞|[R0](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i02_cfg_cycle_transport_05/round00/gallery.png>)|
|`i02_cfg_cycle_transport_05`|R1|207.784|61.941|176/0 （16）|131.590|2288.539 / 2288.539|14.535 / 14.535|[R1](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i02_cfg_cycle_transport_05/round01/gallery.png>)|
|`i02_cfg_cycle_transport_05`|R2|207.291|61.123|176/0 （16）|134.279|2278.943 / 3044.331|14.553 / 13.296|[R2](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i02_cfg_cycle_transport_05/round02/gallery.png>)|
|`i02_cfg_cycle_transport_05`|R3|208.993|60.145|176/0 （16）|131.710|2268.650 / 3561.154|14.573 / 12.615|[R3](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i02_cfg_cycle_transport_05/round03/gallery.png>)|
|`i02_cfg_cycle_transport_05`|R4|206.487|62.532|176/0 （16）|132.660|2291.723 / 3917.482|14.529 / 12.201|[R4](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i02_cfg_cycle_transport_05/round04/gallery.png>)|
|`i02_cfg_cycle_transport_05`|R5|207.397|63.914|176/0 （16）|131.033|2271.242 / 4130.171|14.568 / 11.971|[R5](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i02_cfg_cycle_transport_05/round05/gallery.png>)|
|`i07_ig_cycle_extrapolate_01`|R0|207.425|37.207|134/0 （6）|109.962|0.000 / 0.000|∞ / ∞|[R0](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i07_ig_cycle_extrapolate_01/round00/gallery.png>)|
|`i07_ig_cycle_extrapolate_01`|R1|207.057|38.062|100/0 （4）|83.132|2389.478 / 2389.478|14.348 / 14.348|[R1](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i07_ig_cycle_extrapolate_01/round01/gallery.png>)|
|`i07_ig_cycle_extrapolate_01`|R2|206.357|39.147|100/0 （4）|84.910|2354.849 / 3196.601|14.411 / 13.084|[R2](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i07_ig_cycle_extrapolate_01/round02/gallery.png>)|
|`i07_ig_cycle_extrapolate_01`|R3|206.088|39.709|100/0 （4）|82.137|2355.794 / 3760.033|14.409 / 12.379|[R3](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i07_ig_cycle_extrapolate_01/round03/gallery.png>)|
|`i07_ig_cycle_extrapolate_01`|R4|204.027|38.637|100/0 （4）|83.223|2372.961 / 4148.910|14.378 / 11.951|[R4](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i07_ig_cycle_extrapolate_01/round04/gallery.png>)|
|`i07_ig_cycle_extrapolate_01`|R5|204.550|40.379|100/0 （4）|83.100|2334.440 / 4443.069|14.449 / 11.654|[R5](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i07_ig_cycle_extrapolate_01/round05/gallery.png>)|
|`i03_cfg_cycle_antidrift_04`|R0|207.186|64.259|234/0 （10）|168.226|0.000 / 0.000|∞ / ∞|[R0](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i03_cfg_cycle_antidrift_04/round00/gallery.png>)|
|`i03_cfg_cycle_antidrift_04`|R1|208.100|64.173|168/0 （8）|154.445|2304.279 / 2304.279|14.505 / 14.505|[R1](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i03_cfg_cycle_antidrift_04/round01/gallery.png>)|
|`i03_cfg_cycle_antidrift_04`|R2|207.842|63.627|168/0 （8）|144.492|2302.897 / 3070.148|14.508 / 13.259|[R2](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i03_cfg_cycle_antidrift_04/round02/gallery.png>)|
|`i03_cfg_cycle_antidrift_04`|R3|209.594|63.138|168/0 （8）|141.847|2291.842 / 3588.205|14.529 / 12.582|[R3](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i03_cfg_cycle_antidrift_04/round03/gallery.png>)|
|`i03_cfg_cycle_antidrift_04`|R4|206.959|64.112|168/0 （8）|140.749|2321.792 / 3945.269|14.473 / 12.170|[R4](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i03_cfg_cycle_antidrift_04/round04/gallery.png>)|
|`i03_cfg_cycle_antidrift_04`|R5|208.163|66.713|168/0 （8）|145.193|2307.880 / 4156.689|14.499 / 11.943|[R5](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i03_cfg_cycle_antidrift_04/round05/gallery.png>)|
|`i08_ig_cycle_weakgain_04`|R0|206.039|36.761|152/0 （24）|143.035|0.000 / 0.000|∞ / ∞|[R0](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i08_ig_cycle_weakgain_04/round00/gallery.png>)|
|`i08_ig_cycle_weakgain_04`|R1|206.012|38.539|112/0 （16）|89.753|2325.045 / 2325.045|14.466 / 14.466|[R1](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i08_ig_cycle_weakgain_04/round01/gallery.png>)|
|`i08_ig_cycle_weakgain_04`|R2|205.703|38.471|112/0 （16）|89.891|2293.713 / 3144.379|14.525 / 13.155|[R2](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i08_ig_cycle_weakgain_04/round02/gallery.png>)|
|`i08_ig_cycle_weakgain_04`|R3|206.602|37.815|112/0 （16）|90.901|2282.006 / 3715.422|14.548 / 12.431|[R3](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i08_ig_cycle_weakgain_04/round03/gallery.png>)|
|`i08_ig_cycle_weakgain_04`|R4|204.350|38.841|112/0 （16）|91.987|2277.283 / 4132.614|14.557 / 11.969|[R4](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i08_ig_cycle_weakgain_04/round04/gallery.png>)|
|`i08_ig_cycle_weakgain_04`|R5|205.126|39.429|112/0 （16）|91.918|2245.537 / 4456.126|14.618 / 11.641|[R5](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i08_ig_cycle_weakgain_04/round05/gallery.png>)|
|`i04_cfg_cycle_common_drift_04`|R0|210.693|63.160|244/0 （20）|180.125|0.000 / 0.000|∞ / ∞|[R0](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i04_cfg_cycle_common_drift_04/round00/gallery.png>)|
|`i04_cfg_cycle_common_drift_04`|R1|211.709|64.056|176/0 （16）|150.687|2644.458 / 2644.458|13.907 / 13.907|[R1](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i04_cfg_cycle_common_drift_04/round01/gallery.png>)|
|`i04_cfg_cycle_common_drift_04`|R2|211.319|61.693|176/0 （16）|146.195|2678.992 / 3517.219|13.851 / 12.669|[R2](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i04_cfg_cycle_common_drift_04/round02/gallery.png>)|
|`i04_cfg_cycle_common_drift_04`|R3|213.610|63.104|176/0 （16）|149.006|2686.709 / 4102.596|13.839 / 12.000|[R3](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i04_cfg_cycle_common_drift_04/round03/gallery.png>)|
|`i04_cfg_cycle_common_drift_04`|R4|210.779|63.953|176/0 （16）|147.146|2731.840 / 4506.577|13.766 / 11.592|[R4](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i04_cfg_cycle_common_drift_04/round04/gallery.png>)|
|`i04_cfg_cycle_common_drift_04`|R5|212.611|66.863|176/0 （16）|134.596|2724.996 / 4736.177|13.777 / 11.377|[R5](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i04_cfg_cycle_common_drift_04/round05/gallery.png>)|
|`i09_ig_cycle_antithetic_10`|R0|213.899|33.822|134/0 （6）|106.353|0.000 / 0.000|∞ / ∞|[R0](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i09_ig_cycle_antithetic_10/round00/gallery.png>)|
|`i09_ig_cycle_antithetic_10`|R1|213.891|34.603|100/0 （4）|81.501|2358.954 / 2358.954|14.404 / 14.404|[R1](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i09_ig_cycle_antithetic_10/round01/gallery.png>)|
|`i09_ig_cycle_antithetic_10`|R2|212.852|34.431|100/0 （4）|81.714|2336.089 / 3129.979|14.446 / 13.175|[R2](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i09_ig_cycle_antithetic_10/round02/gallery.png>)|
|`i09_ig_cycle_antithetic_10`|R3|213.175|34.702|100/0 （4）|82.526|2357.773 / 3650.911|14.406 / 12.507|[R3](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i09_ig_cycle_antithetic_10/round03/gallery.png>)|
|`i09_ig_cycle_antithetic_10`|R4|210.073|33.661|100/0 （4）|82.246|2389.453 / 3976.354|14.348 / 12.136|[R4](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i09_ig_cycle_antithetic_10/round04/gallery.png>)|
|`i09_ig_cycle_antithetic_10`|R5|211.350|35.906|100/0 （4）|82.023|2350.720 / 4178.879|14.419 / 11.920|[R5](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i09_ig_cycle_antithetic_10/round05/gallery.png>)|
|`i05_cfg_cycle_curvature_04`|R0|207.227|63.486|254/0 （30）|184.574|0.000 / 0.000|∞ / ∞|[R0](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i05_cfg_cycle_curvature_04/round00/gallery.png>)|
|`i05_cfg_cycle_curvature_04`|R1|208.047|64.174|184/0 （24）|136.001|2290.126 / 2290.126|14.532 / 14.532|[R1](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i05_cfg_cycle_curvature_04/round01/gallery.png>)|
|`i05_cfg_cycle_curvature_04`|R2|207.929|62.424|184/0 （24）|135.655|2286.512 / 3047.787|14.539 / 13.291|[R2](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i05_cfg_cycle_curvature_04/round02/gallery.png>)|
|`i05_cfg_cycle_curvature_04`|R3|209.583|61.750|184/0 （24）|135.700|2279.690 / 3564.464|14.552 / 12.611|[R3](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i05_cfg_cycle_curvature_04/round03/gallery.png>)|
|`i05_cfg_cycle_curvature_04`|R4|206.878|64.350|184/0 （24）|141.554|2308.559 / 3916.049|14.497 / 12.202|[R4](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i05_cfg_cycle_curvature_04/round04/gallery.png>)|
|`i05_cfg_cycle_curvature_04`|R5|208.083|66.469|184/0 （24）|136.955|2293.568 / 4124.390|14.526 / 11.977|[R5](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i05_cfg_cycle_curvature_04/round05/gallery.png>)|
|`i10_ig_cycle_twohop_10`|R0|207.434|35.971|146/0 （18）|113.882|0.000 / 0.000|∞ / ∞|[R0](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i10_ig_cycle_twohop_10/round00/gallery.png>)|
|`i10_ig_cycle_twohop_10`|R1|207.176|37.273|108/0 （12）|87.872|2317.410 / 2317.410|14.481 / 14.481|[R1](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i10_ig_cycle_twohop_10/round01/gallery.png>)|
|`i10_ig_cycle_twohop_10`|R2|206.925|38.133|108/0 （12）|87.087|2286.026 / 3102.123|14.540 / 13.214|[R2](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i10_ig_cycle_twohop_10/round02/gallery.png>)|
|`i10_ig_cycle_twohop_10`|R3|207.136|39.195|108/0 （12）|87.012|2290.003 / 3641.426|14.532 / 12.518|[R3](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i10_ig_cycle_twohop_10/round03/gallery.png>)|
|`i10_ig_cycle_twohop_10`|R4|204.309|40.283|108/0 （12）|88.562|2305.796 / 4018.848|14.503 / 12.090|[R4](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i10_ig_cycle_twohop_10/round04/gallery.png>)|
|`i10_ig_cycle_twohop_10`|R5|205.309|40.392|108/0 （12）|108.616|2273.423 / 4276.573|14.564 / 11.820|[R5](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i10_ig_cycle_twohop_10/round05/gallery.png>)|
|`strong_00`|R0|220.058|29.752|128/0 （0）|94.931|0.000 / 0.000|∞ / ∞|[R0](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/strong_00/round00/gallery.png>)|
|`strong_00`|R1|220.839|28.279|96/0 （0）|77.177|2456.285 / 2456.285|14.228 / 14.228|[R1](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/strong_00/round01/gallery.png>)|
|`strong_00`|R2|220.069|30.777|96/0 （0）|78.065|2430.728 / 3264.834|14.273 / 12.992|[R2](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/strong_00/round02/gallery.png>)|
|`strong_00`|R3|220.300|31.511|96/0 （0）|76.409|2452.001 / 3812.555|14.236 / 12.319|[R3](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/strong_00/round03/gallery.png>)|
|`strong_00`|R4|218.272|28.997|96/0 （0）|77.092|2484.377 / 4149.887|14.179 / 11.950|[R4](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/strong_00/round04/gallery.png>)|
|`strong_00`|R5|219.371|32.148|96/0 （0）|76.571|2449.300 / 4364.958|14.240 / 11.731|[R5](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/strong_00/round05/gallery.png>)|
|`ig_mlp_02`|R0|206.012|36.691|128/0 （0）|104.185|0.000 / 0.000|∞ / ∞|[R0](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_mlp_02/round00/gallery.png>)|
|`ig_mlp_02`|R1|206.009|38.482|96/0 （0）|81.597|2324.541 / 2324.541|14.467 / 14.467|[R1](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_mlp_02/round01/gallery.png>)|
|`ig_mlp_02`|R2|205.722|38.432|96/0 （0）|81.051|2293.059 / 3143.658|14.527 / 13.156|[R2](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_mlp_02/round02/gallery.png>)|
|`ig_mlp_02`|R3|206.637|37.806|96/0 （0）|81.197|2281.473 / 3715.273|14.549 / 12.431|[R3](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_mlp_02/round03/gallery.png>)|
|`ig_mlp_02`|R4|204.427|38.660|96/0 （0）|82.331|2276.806 / 4133.115|14.558 / 11.968|[R4](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_mlp_02/round04/gallery.png>)|
|`ig_mlp_02`|R5|205.136|39.408|96/0 （0）|92.356|2245.786 / 4458.603|14.617 / 11.639|[R5](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_mlp_02/round05/gallery.png>)|
|`cfg_native_03`|R0|207.084|64.306|224/0 （0）|160.553|0.000 / 0.000|∞ / ∞|[R0](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/cfg_native_03/round00/gallery.png>)|
|`cfg_native_03`|R1|207.960|64.348|160/0 （0）|119.405|2292.716 / 2292.716|14.527 / 14.527|[R1](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/cfg_native_03/round01/gallery.png>)|
|`cfg_native_03`|R2|207.769|63.669|160/0 （0）|120.249|2290.418 / 3055.228|14.532 / 13.280|[R2](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/cfg_native_03/round02/gallery.png>)|
|`cfg_native_03`|R3|209.470|63.132|160/0 （0）|119.459|2279.319 / 3571.127|14.553 / 12.603|[R3](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/cfg_native_03/round03/gallery.png>)|
|`cfg_native_03`|R4|206.844|64.243|160/0 （0）|119.699|2310.341 / 3928.304|14.494 / 12.189|[R4](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/cfg_native_03/round04/gallery.png>)|
|`cfg_native_03`|R5|208.110|66.470|160/0 （0）|118.777|2295.415 / 4137.778|14.522 / 11.963|[R5](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/cfg_native_03/round05/gallery.png>)|
|`ig_native_01`|R0|208.550|34.083|128/0 （0）|103.486|0.000 / 0.000|∞ / ∞|[R0](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_native_01/round00/gallery.png>)|
|`ig_native_01`|R1|207.662|34.966|96/0 （0）|79.915|2356.534 / 2356.534|14.408 / 14.408|[R1](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_native_01/round01/gallery.png>)|
|`ig_native_01`|R2|206.885|36.165|96/0 （0）|79.288|2304.421 / 3141.472|14.505 / 13.159|[R2](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_native_01/round02/gallery.png>)|
|`ig_native_01`|R3|207.343|36.701|96/0 （0）|79.648|2300.209 / 3673.298|14.513 / 12.480|[R3](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_native_01/round03/gallery.png>)|
|`ig_native_01`|R4|204.662|38.091|96/0 （0）|79.594|2322.654 / 4048.854|14.471 / 12.057|[R4](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_native_01/round04/gallery.png>)|
|`ig_native_01`|R5|205.707|37.594|96/0 （0）|83.152|2278.626 / 4290.605|14.554 / 11.806|[R5](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_native_01/round05/gallery.png>)|
|`ig_mlp_adg_01`|R0|205.673|36.975|128/0 （0）|109.212|0.000 / 0.000|∞ / ∞|[R0](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_mlp_adg_01/round00/gallery.png>)|
|`ig_mlp_adg_01`|R1|205.631|37.669|96/0 （0）|84.311|2262.423 / 2262.423|14.585 / 14.585|[R1](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_mlp_adg_01/round01/gallery.png>)|
|`ig_mlp_adg_01`|R2|205.779|37.989|96/0 （0）|82.946|2225.276 / 3046.275|14.657 / 13.293|[R2](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_mlp_adg_01/round02/gallery.png>)|
|`ig_mlp_adg_01`|R3|205.569|39.200|96/0 （0）|82.909|2210.224 / 3586.668|14.686 / 12.584|[R3](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_mlp_adg_01/round03/gallery.png>)|
|`ig_mlp_adg_01`|R4|204.385|40.211|96/0 （0）|82.925|2236.441 / 3992.215|14.635 / 12.119|[R4](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_mlp_adg_01/round04/gallery.png>)|
|`ig_mlp_adg_01`|R5|205.226|41.089|96/0 （0）|83.570|2214.515 / 4293.492|14.678 / 11.803|[R5](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_mlp_adg_01/round05/gallery.png>)|

## 全部固定配置的六轮记录

|对象|固定配置|路线/参考|α / τ|成功轮数|FID R0|FID R1|FID R2|FID R3|FID R4|FID R5|选择说明|图集|
|---|---|---|---|--:|--:|--:|--:|--:|--:|--:|---|---|
|strong|`strong_00`|IG/native|0 / 0|6/6|83.8286|85.9553|85.2657|82.4875|85.0947|81.4587|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/strong_00/rounds.png>)|
|ig_mlp|`ig_mlp_03`|IG/mlp|0.8 / 0|6/6|62.6471|62.7819|64.0699|63.1785|66.0306|66.4514|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_mlp_03/rounds.png>)|
|cfg_native|`cfg_native_03`|CFG/native|1.25 / 0|6/6|46.5318|45.6094|45.4965|45.4946|45.6927|45.6081|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/cfg_native_03/rounds.png>)|
|ig_native|`ig_native_03`|IG/native|0.8 / 0|6/6|64.3005|66.1023|66.8362|67.6334|71.3149|72.3239|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_native_03/rounds.png>)|
|i01_cfg_cycle_gain|`i01_cfg_cycle_gain_06`|CFG/native|1.25 / 0.8|6/6|46.5464|45.6031|45.4915|45.5045|45.7252|45.5638|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i01_cfg_cycle_gain_06/rounds.png>)|
|i06_ig_cycle_agreement|`i06_ig_cycle_agreement_10`|IG/mlp|0.8 / 0.8|6/6|71.6751|72.3569|70.3324|68.2045|69.1968|66.1706|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i06_ig_cycle_agreement_10/rounds.png>)|
|i02_cfg_cycle_transport|`i02_cfg_cycle_transport_06`|CFG/native|1.25 / 0.8|6/6|47.1738|46.2896|45.8607|46.0342|46.2986|45.8536|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i02_cfg_cycle_transport_06/rounds.png>)|
|i07_ig_cycle_extrapolate|`i07_ig_cycle_extrapolate_10`|IG/mlp|0.8 / 0.8|6/6|62.0894|63.3819|66.7284|67.7839|72.9436|74.9939|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i07_ig_cycle_extrapolate_10/rounds.png>)|
|ig_native|`ig_native_00`|IG/native|0.2 / 0|6/6|74.4035|75.5300|73.6172|72.5936|73.7863|69.0169|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_native_00/rounds.png>)|
|i03_cfg_cycle_antidrift|`i03_cfg_cycle_antidrift_06`|CFG/native|1.25 / 0.8|6/6|46.5368|45.6149|45.5142|45.5123|45.6483|45.6149|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i03_cfg_cycle_antidrift_06/rounds.png>)|
|i08_ig_cycle_weakgain|`i08_ig_cycle_weakgain_10`|IG/mlp|0.8 / 0.8|6/6|62.6166|62.7467|63.8284|63.0037|65.2997|65.9313|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i08_ig_cycle_weakgain_10/rounds.png>)|
|i04_cfg_cycle_common_drift|`i04_cfg_cycle_common_drift_06`|CFG/native|1.25 / 0.8|6/6|47.1506|45.9742|45.9068|45.9020|45.9532|46.2045|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i04_cfg_cycle_common_drift_06/rounds.png>)|
|i09_ig_cycle_antithetic|`i09_ig_cycle_antithetic_10`|IG/mlp|0.8 / 0.8|6/6|73.5574|75.3026|73.7183|72.2769|73.0878|69.1132|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i09_ig_cycle_antithetic_10/rounds.png>)|
|ig_mlp|`ig_mlp_00`|IG/mlp|0.2 / 0|6/6|74.4310|75.8979|74.0879|72.2448|73.8296|69.2679|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_mlp_00/rounds.png>)|
|i05_cfg_cycle_curvature|`i05_cfg_cycle_curvature_06`|CFG/native|1.25 / 0.8|6/6|46.6171|45.6160|45.4377|45.5178|45.6934|45.3858|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i05_cfg_cycle_curvature_06/rounds.png>)|
|i10_ig_cycle_twohop|`i10_ig_cycle_twohop_10`|IG/mlp|0.8 / 0.8|6/6|66.2541|66.3556|65.1081|62.7914|64.0165|61.8301|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i10_ig_cycle_twohop_10/rounds.png>)|
|i01_cfg_cycle_gain|`i01_cfg_cycle_gain_00`|CFG/native|0.75 / 0.5|6/6|48.9797|48.5410|48.2598|47.2532|47.3716|46.2947|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i01_cfg_cycle_gain_00/rounds.png>)|
|i06_ig_cycle_agreement|`i06_ig_cycle_agreement_00`|IG/mlp|0.4 / 0.5|6/6|74.8918|76.0724|74.3189|71.7546|73.4347|69.1966|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i06_ig_cycle_agreement_00/rounds.png>)|
|ig_native|`ig_native_01`|IG/native|0.4 / 0|6/6|68.7402|68.4128|67.5274|65.0558|67.4446|64.4138|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_native_01/rounds.png>)|
|i02_cfg_cycle_transport|`i02_cfg_cycle_transport_00`|CFG/native|0.75 / 0.5|6/6|51.2479|50.5269|49.8798|48.5597|48.7347|47.5422|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i02_cfg_cycle_transport_00/rounds.png>)|
|i07_ig_cycle_extrapolate|`i07_ig_cycle_extrapolate_00`|IG/mlp|0.4 / 0.5|6/6|66.7035|65.3450|63.7380|62.3039|64.5006|62.0882|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i07_ig_cycle_extrapolate_00/rounds.png>)|
|i03_cfg_cycle_antidrift|`i03_cfg_cycle_antidrift_00`|CFG/native|0.75 / 0.5|6/6|48.9476|48.4883|48.2004|47.1903|47.2864|46.1987|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i03_cfg_cycle_antidrift_00/rounds.png>)|
|i08_ig_cycle_weakgain|`i08_ig_cycle_weakgain_00`|IG/mlp|0.4 / 0.5|6/6|68.9979|68.5009|66.6367|64.8603|66.2975|63.4146|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i08_ig_cycle_weakgain_00/rounds.png>)|
|ig_mlp|`ig_mlp_01`|IG/mlp|0.4 / 0|6/6|68.9886|68.4676|66.6231|64.8171|66.2802|63.3657|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_mlp_01/rounds.png>)|
|i04_cfg_cycle_common_drift|`i04_cfg_cycle_common_drift_00`|CFG/native|0.75 / 0.5|6/6|49.5622|48.7341|48.6222|47.6970|47.8512|46.7956|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i04_cfg_cycle_common_drift_00/rounds.png>)|
|i09_ig_cycle_antithetic|`i09_ig_cycle_antithetic_00`|IG/mlp|0.4 / 0.5|6/6|80.4229|83.2998|81.8741|80.2290|82.3754|78.2743|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i09_ig_cycle_antithetic_00/rounds.png>)|
|i05_cfg_cycle_curvature|`i05_cfg_cycle_curvature_00`|CFG/native|0.75 / 0.5|6/6|49.4339|49.0158|48.7664|47.6948|47.7575|46.6015|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i05_cfg_cycle_curvature_00/rounds.png>)|
|i10_ig_cycle_twohop|`i10_ig_cycle_twohop_00`|IG/mlp|0.4 / 0.5|6/6|72.4420|72.9768|71.7808|69.5952|70.6033|67.0395|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i10_ig_cycle_twohop_00/rounds.png>)|
|ig_native|`ig_native_02`|IG/native|0.6 / 0|6/6|65.1858|65.8369|65.3823|63.9094|65.7180|65.0229|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_native_02/rounds.png>)|
|i01_cfg_cycle_gain|`i01_cfg_cycle_gain_01`|CFG/native|0.75 / 0.65|6/6|48.9972|48.5675|48.2390|47.2508|47.3657|46.3105|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i01_cfg_cycle_gain_01/rounds.png>)|
|i06_ig_cycle_agreement|`i06_ig_cycle_agreement_01`|IG/mlp|0.4 / 0.65|6/6|75.6208|77.0038|75.4868|72.7811|74.4118|70.3514|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i06_ig_cycle_agreement_01/rounds.png>)|
|i02_cfg_cycle_transport|`i02_cfg_cycle_transport_01`|CFG/native|0.75 / 0.65|6/6|50.8125|50.1719|49.4920|48.2191|48.1571|47.2195|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i02_cfg_cycle_transport_01/rounds.png>)|
|i07_ig_cycle_extrapolate|`i07_ig_cycle_extrapolate_01`|IG/mlp|0.4 / 0.65|6/6|66.7281|65.3703|63.7548|62.1266|63.8960|61.4378|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i07_ig_cycle_extrapolate_01/rounds.png>)|
|ig_mlp|`ig_mlp_02`|IG/mlp|0.6 / 0|6/6|64.9661|64.6367|63.4066|62.3933|64.0818|61.9321|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_mlp_02/rounds.png>)|
|i03_cfg_cycle_antidrift|`i03_cfg_cycle_antidrift_01`|CFG/native|0.75 / 0.65|6/6|48.9356|48.4779|48.2273|47.1769|47.3287|46.2022|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i03_cfg_cycle_antidrift_01/rounds.png>)|
|i08_ig_cycle_weakgain|`i08_ig_cycle_weakgain_01`|IG/mlp|0.4 / 0.65|6/6|69.0170|68.5436|66.6610|64.8517|66.3915|63.3182|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i08_ig_cycle_weakgain_01/rounds.png>)|
|i04_cfg_cycle_common_drift|`i04_cfg_cycle_common_drift_01`|CFG/native|0.75 / 0.65|6/6|49.4380|48.7183|48.6849|47.5993|47.7211|46.6425|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i04_cfg_cycle_common_drift_01/rounds.png>)|
|i09_ig_cycle_antithetic|`i09_ig_cycle_antithetic_01`|IG/mlp|0.4 / 0.65|6/6|79.8334|82.3652|80.9800|79.4699|81.0944|77.2197|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i09_ig_cycle_antithetic_01/rounds.png>)|
|ig_native|`ig_native_04`|IG/native|1 / 0|6/6|65.1388|68.5957|72.7375|75.0241|80.9921|84.3759|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_native_04/rounds.png>)|
|i05_cfg_cycle_curvature|`i05_cfg_cycle_curvature_01`|CFG/native|0.75 / 0.65|6/6|49.4293|48.9483|48.7233|47.6412|47.7087|46.5381|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i05_cfg_cycle_curvature_01/rounds.png>)|
|i10_ig_cycle_twohop|`i10_ig_cycle_twohop_01`|IG/mlp|0.4 / 0.65|6/6|72.7579|73.5908|72.2162|70.0882|70.9197|67.4759|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i10_ig_cycle_twohop_01/rounds.png>)|
|i01_cfg_cycle_gain|`i01_cfg_cycle_gain_02`|CFG/native|0.75 / 0.8|6/6|48.9670|48.5063|48.2225|47.1878|47.3023|46.3099|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i01_cfg_cycle_gain_02/rounds.png>)|
|i06_ig_cycle_agreement|`i06_ig_cycle_agreement_02`|IG/mlp|0.4 / 0.8|6/6|76.8063|78.4235|76.9861|74.8610|76.4315|72.1811|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i06_ig_cycle_agreement_02/rounds.png>)|
|ig_mlp|`ig_mlp_04`|IG/mlp|1 / 0|6/6|62.1206|63.9684|66.6819|67.2625|72.9170|75.2969|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_mlp_04/rounds.png>)|
|i02_cfg_cycle_transport|`i02_cfg_cycle_transport_02`|CFG/native|0.75 / 0.8|6/6|50.2503|49.5111|48.9604|47.8805|47.8098|46.9161|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i02_cfg_cycle_transport_02/rounds.png>)|
|i07_ig_cycle_extrapolate|`i07_ig_cycle_extrapolate_02`|IG/mlp|0.4 / 0.8|6/6|66.6058|66.1205|64.3127|62.4295|65.0289|63.1204|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i07_ig_cycle_extrapolate_02/rounds.png>)|
|i03_cfg_cycle_antidrift|`i03_cfg_cycle_antidrift_02`|CFG/native|0.75 / 0.8|6/6|48.9447|48.4799|48.2288|47.2221|47.2878|46.2573|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i03_cfg_cycle_antidrift_02/rounds.png>)|
|i08_ig_cycle_weakgain|`i08_ig_cycle_weakgain_02`|IG/mlp|0.4 / 0.8|6/6|69.0601|68.5669|66.7899|64.8772|66.4272|63.3867|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i08_ig_cycle_weakgain_02/rounds.png>)|
|ig_native|`ig_native_05`|IG/native|1.2 / 0|6/6|67.2460|73.9711|80.4220|86.2706|93.5939|99.4481|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_native_05/rounds.png>)|
|i04_cfg_cycle_common_drift|`i04_cfg_cycle_common_drift_02`|CFG/native|0.75 / 0.8|6/6|49.3982|48.7437|48.7481|47.5930|47.6606|46.5761|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i04_cfg_cycle_common_drift_02/rounds.png>)|
|i09_ig_cycle_antithetic|`i09_ig_cycle_antithetic_02`|IG/mlp|0.4 / 0.8|6/6|78.2645|80.5799|78.8281|77.3573|78.4407|74.7382|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i09_ig_cycle_antithetic_02/rounds.png>)|
|i05_cfg_cycle_curvature|`i05_cfg_cycle_curvature_02`|CFG/native|0.75 / 0.8|6/6|49.2674|48.8121|48.5745|47.5064|47.5988|46.5204|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i05_cfg_cycle_curvature_02/rounds.png>)|
|i10_ig_cycle_twohop|`i10_ig_cycle_twohop_02`|IG/mlp|0.4 / 0.8|6/6|72.8243|73.5670|72.1815|70.1608|71.0240|67.4860|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i10_ig_cycle_twohop_02/rounds.png>)|
|ig_mlp|`ig_mlp_05`|IG/mlp|1.2 / 0|6/6|62.7729|65.8365|70.8648|74.7118|81.9824|85.8613|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_mlp_05/rounds.png>)|
|i01_cfg_cycle_gain|`i01_cfg_cycle_gain_03`|CFG/native|0.75 / 0.9|6/6|48.9397|48.4697|48.2130|47.1586|47.3047|46.2747|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i01_cfg_cycle_gain_03/rounds.png>)|
|i06_ig_cycle_agreement|`i06_ig_cycle_agreement_03`|IG/mlp|0.4 / 0.9|6/6|78.1684|80.6129|78.8039|77.2438|78.5173|74.4334|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i06_ig_cycle_agreement_03/rounds.png>)|
|i02_cfg_cycle_transport|`i02_cfg_cycle_transport_03`|CFG/native|0.75 / 0.9|6/6|49.9729|49.2969|48.7364|47.6875|47.5669|46.7402|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i02_cfg_cycle_transport_03/rounds.png>)|
|i07_ig_cycle_extrapolate|`i07_ig_cycle_extrapolate_03`|IG/mlp|0.4 / 0.9|6/6|66.8376|66.6176|64.5948|62.9160|65.1451|62.9723|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i07_ig_cycle_extrapolate_03/rounds.png>)|
|ig_mlp_adg|`ig_mlp_adg_00`|IG/mlp|0.4 / 0|6/6|68.0159|68.3563|66.9547|63.4863|66.3547|63.4212|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_mlp_adg_00/rounds.png>)|
|i03_cfg_cycle_antidrift|`i03_cfg_cycle_antidrift_03`|CFG/native|0.75 / 0.9|6/6|48.9590|48.4798|48.2259|47.2217|47.2905|46.2658|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i03_cfg_cycle_antidrift_03/rounds.png>)|
|i08_ig_cycle_weakgain|`i08_ig_cycle_weakgain_03`|IG/mlp|0.4 / 0.9|6/6|69.0557|68.5615|66.7061|64.8549|66.4193|63.3986|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i08_ig_cycle_weakgain_03/rounds.png>)|
|i04_cfg_cycle_common_drift|`i04_cfg_cycle_common_drift_03`|CFG/native|0.75 / 0.9|6/6|49.2272|48.6967|48.6239|47.4427|47.4130|46.4430|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i04_cfg_cycle_common_drift_03/rounds.png>)|
|i09_ig_cycle_antithetic|`i09_ig_cycle_antithetic_03`|IG/mlp|0.4 / 0.9|6/6|77.1247|79.1261|77.2247|75.8942|77.0229|72.6154|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i09_ig_cycle_antithetic_03/rounds.png>)|
|ig_mlp_adg|`ig_mlp_adg_01`|IG/mlp|0.6 / 0|6/6|63.7830|64.9644|61.9098|59.9445|62.2022|60.4315|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_mlp_adg_01/rounds.png>)|
|i05_cfg_cycle_curvature|`i05_cfg_cycle_curvature_03`|CFG/native|0.75 / 0.9|6/6|49.1234|48.6732|48.4200|47.3311|47.4635|46.3751|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i05_cfg_cycle_curvature_03/rounds.png>)|
|i10_ig_cycle_twohop|`i10_ig_cycle_twohop_03`|IG/mlp|0.4 / 0.9|6/6|72.9520|73.7213|72.3703|70.4545|71.3293|67.7086|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i10_ig_cycle_twohop_03/rounds.png>)|
|i01_cfg_cycle_gain|`i01_cfg_cycle_gain_04`|CFG/native|1.25 / 0.5|6/6|46.4903|45.6019|45.4515|45.4847|45.6955|45.4041|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i01_cfg_cycle_gain_04/rounds.png>)|
|i06_ig_cycle_agreement|`i06_ig_cycle_agreement_04`|IG/mlp|0.6 / 0.5|6/6|71.7381|72.4040|70.0969|67.4896|68.5473|65.3594|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i06_ig_cycle_agreement_04/rounds.png>)|
|ig_mlp_adg|`ig_mlp_adg_02`|IG/mlp|0.8 / 0|6/6|62.0624|62.3056|61.3854|60.5270|62.4550|62.3183|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_mlp_adg_02/rounds.png>)|
|i02_cfg_cycle_transport|`i02_cfg_cycle_transport_04`|CFG/native|1.25 / 0.5|6/6|47.6385|46.6223|46.0773|46.4546|46.4172|45.8769|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i02_cfg_cycle_transport_04/rounds.png>)|
|i07_ig_cycle_extrapolate|`i07_ig_cycle_extrapolate_04`|IG/mlp|0.6 / 0.5|6/6|62.2261|62.7826|62.1375|61.4415|63.1350|63.0289|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i07_ig_cycle_extrapolate_04/rounds.png>)|
|i03_cfg_cycle_antidrift|`i03_cfg_cycle_antidrift_04`|CFG/native|1.25 / 0.5|6/6|46.5176|45.5766|45.4610|45.4561|45.6791|45.6072|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i03_cfg_cycle_antidrift_04/rounds.png>)|
|i08_ig_cycle_weakgain|`i08_ig_cycle_weakgain_04`|IG/mlp|0.6 / 0.5|6/6|64.9759|64.6611|63.4278|62.3620|64.0803|61.9502|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i08_ig_cycle_weakgain_04/rounds.png>)|
|ig_mlp_adg|`ig_mlp_adg_03`|IG/mlp|1 / 0|6/6|61.6620|62.3234|62.6133|61.8465|65.1044|65.2377|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_mlp_adg_03/rounds.png>)|
|i04_cfg_cycle_common_drift|`i04_cfg_cycle_common_drift_04`|CFG/native|1.25 / 0.5|6/6|47.1601|45.9901|45.7756|45.7044|45.7840|46.1428|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i04_cfg_cycle_common_drift_04/rounds.png>)|
|i09_ig_cycle_antithetic|`i09_ig_cycle_antithetic_04`|IG/mlp|0.6 / 0.5|6/6|79.2424|81.9787|80.3506|78.8225|80.4733|77.5465|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i09_ig_cycle_antithetic_04/rounds.png>)|
|i05_cfg_cycle_curvature|`i05_cfg_cycle_curvature_04`|CFG/native|1.25 / 0.5|6/6|46.6315|45.6243|45.5029|45.5704|45.5118|45.2680|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i05_cfg_cycle_curvature_04/rounds.png>)|
|i10_ig_cycle_twohop|`i10_ig_cycle_twohop_04`|IG/mlp|0.6 / 0.5|6/6|68.9767|69.1193|67.5245|65.1011|66.9293|63.4338|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i10_ig_cycle_twohop_04/rounds.png>)|
|ig_mlp_adg|`ig_mlp_adg_04`|IG/mlp|1.2 / 0|6/6|62.0750|63.9603|64.6728|64.9418|68.2895|69.3148|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/ig_mlp_adg_04/rounds.png>)|
|i01_cfg_cycle_gain|`i01_cfg_cycle_gain_05`|CFG/native|1.25 / 0.65|6/6|46.5461|45.5231|45.4701|45.4755|45.6648|45.5090|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i01_cfg_cycle_gain_05/rounds.png>)|
|i06_ig_cycle_agreement|`i06_ig_cycle_agreement_05`|IG/mlp|0.6 / 0.65|6/6|72.5933|73.5679|70.8721|69.0953|69.9872|67.0002|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i06_ig_cycle_agreement_05/rounds.png>)|
|i02_cfg_cycle_transport|`i02_cfg_cycle_transport_05`|CFG/native|1.25 / 0.65|6/6|47.4256|46.5191|45.9861|46.2381|46.4361|45.8452|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i02_cfg_cycle_transport_05/rounds.png>)|
|i07_ig_cycle_extrapolate|`i07_ig_cycle_extrapolate_05`|IG/mlp|0.6 / 0.65|6/6|62.2713|62.3409|62.2738|61.6365|64.2468|63.1453|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/i07_ig_cycle_extrapolate_05/rounds.png>)|
|cfg_native|`cfg_native_00`|CFG/native|0.25 / 0|3/6|64.9212|66.2444|64.3242|—|—|—|固定配置；不跨轮重选|[逐轮图集](</home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k/cfg_native_00/rounds.png>)|
|i03_cfg_cycle_antidrift|`i03_cfg_cycle_antidrift_05`|CFG/native|1.25 / 0.65|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i08_ig_cycle_weakgain|`i08_ig_cycle_weakgain_05`|IG/mlp|0.6 / 0.65|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i04_cfg_cycle_common_drift|`i04_cfg_cycle_common_drift_05`|CFG/native|1.25 / 0.65|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i09_ig_cycle_antithetic|`i09_ig_cycle_antithetic_05`|IG/mlp|0.6 / 0.65|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|cfg_native|`cfg_native_01`|CFG/native|0.5 / 0|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i05_cfg_cycle_curvature|`i05_cfg_cycle_curvature_05`|CFG/native|1.25 / 0.65|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i10_ig_cycle_twohop|`i10_ig_cycle_twohop_05`|IG/mlp|0.6 / 0.65|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i01_cfg_cycle_gain|`i01_cfg_cycle_gain_07`|CFG/native|1.25 / 0.9|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i06_ig_cycle_agreement|`i06_ig_cycle_agreement_06`|IG/mlp|0.6 / 0.8|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|cfg_native|`cfg_native_02`|CFG/native|0.75 / 0|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i02_cfg_cycle_transport|`i02_cfg_cycle_transport_07`|CFG/native|1.25 / 0.9|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i07_ig_cycle_extrapolate|`i07_ig_cycle_extrapolate_06`|IG/mlp|0.6 / 0.8|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i03_cfg_cycle_antidrift|`i03_cfg_cycle_antidrift_07`|CFG/native|1.25 / 0.9|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i08_ig_cycle_weakgain|`i08_ig_cycle_weakgain_06`|IG/mlp|0.6 / 0.8|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|cfg_native|`cfg_native_04`|CFG/native|1.75 / 0|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i04_cfg_cycle_common_drift|`i04_cfg_cycle_common_drift_07`|CFG/native|1.25 / 0.9|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i09_ig_cycle_antithetic|`i09_ig_cycle_antithetic_06`|IG/mlp|0.6 / 0.8|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i05_cfg_cycle_curvature|`i05_cfg_cycle_curvature_07`|CFG/native|1.25 / 0.9|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i10_ig_cycle_twohop|`i10_ig_cycle_twohop_06`|IG/mlp|0.6 / 0.8|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|cfg_native|`cfg_native_05`|CFG/native|2.25 / 0|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i01_cfg_cycle_gain|`i01_cfg_cycle_gain_08`|CFG/native|1.75 / 0.5|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i06_ig_cycle_agreement|`i06_ig_cycle_agreement_07`|IG/mlp|0.6 / 0.9|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i02_cfg_cycle_transport|`i02_cfg_cycle_transport_08`|CFG/native|1.75 / 0.5|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i07_ig_cycle_extrapolate|`i07_ig_cycle_extrapolate_07`|IG/mlp|0.6 / 0.9|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|cfg_native|`cfg_native_06`|CFG/native|2.75 / 0|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i03_cfg_cycle_antidrift|`i03_cfg_cycle_antidrift_08`|CFG/native|1.75 / 0.5|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i08_ig_cycle_weakgain|`i08_ig_cycle_weakgain_07`|IG/mlp|0.6 / 0.9|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i04_cfg_cycle_common_drift|`i04_cfg_cycle_common_drift_08`|CFG/native|1.75 / 0.5|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i09_ig_cycle_antithetic|`i09_ig_cycle_antithetic_07`|IG/mlp|0.6 / 0.9|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|cfg_apg|`cfg_apg_00`|CFG/native|0.75 / -0.5|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i05_cfg_cycle_curvature|`i05_cfg_cycle_curvature_08`|CFG/native|1.75 / 0.5|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i10_ig_cycle_twohop|`i10_ig_cycle_twohop_07`|IG/mlp|0.6 / 0.9|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i01_cfg_cycle_gain|`i01_cfg_cycle_gain_09`|CFG/native|1.75 / 0.65|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i06_ig_cycle_agreement|`i06_ig_cycle_agreement_08`|IG/mlp|0.8 / 0.5|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|cfg_apg|`cfg_apg_01`|CFG/native|0.75 / -0.25|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i02_cfg_cycle_transport|`i02_cfg_cycle_transport_09`|CFG/native|1.75 / 0.65|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i07_ig_cycle_extrapolate|`i07_ig_cycle_extrapolate_08`|IG/mlp|0.8 / 0.5|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i03_cfg_cycle_antidrift|`i03_cfg_cycle_antidrift_09`|CFG/native|1.75 / 0.65|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i08_ig_cycle_weakgain|`i08_ig_cycle_weakgain_08`|IG/mlp|0.8 / 0.5|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|cfg_ctrl|`cfg_ctrl_00`|CFG/native|0.75 / 0.2|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i04_cfg_cycle_common_drift|`i04_cfg_cycle_common_drift_09`|CFG/native|1.75 / 0.65|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i09_ig_cycle_antithetic|`i09_ig_cycle_antithetic_08`|IG/mlp|0.8 / 0.5|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i05_cfg_cycle_curvature|`i05_cfg_cycle_curvature_09`|CFG/native|1.75 / 0.65|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i10_ig_cycle_twohop|`i10_ig_cycle_twohop_08`|IG/mlp|0.8 / 0.5|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|cfg_apg|`cfg_apg_02`|CFG/native|1.25 / -0.5|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i01_cfg_cycle_gain|`i01_cfg_cycle_gain_10`|CFG/native|1.75 / 0.8|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i06_ig_cycle_agreement|`i06_ig_cycle_agreement_09`|IG/mlp|0.8 / 0.65|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i02_cfg_cycle_transport|`i02_cfg_cycle_transport_10`|CFG/native|1.75 / 0.8|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i07_ig_cycle_extrapolate|`i07_ig_cycle_extrapolate_09`|IG/mlp|0.8 / 0.65|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|cfg_apg|`cfg_apg_03`|CFG/native|1.25 / -0.25|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i03_cfg_cycle_antidrift|`i03_cfg_cycle_antidrift_10`|CFG/native|1.75 / 0.8|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i08_ig_cycle_weakgain|`i08_ig_cycle_weakgain_09`|IG/mlp|0.8 / 0.65|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i04_cfg_cycle_common_drift|`i04_cfg_cycle_common_drift_10`|CFG/native|1.75 / 0.8|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i09_ig_cycle_antithetic|`i09_ig_cycle_antithetic_09`|IG/mlp|0.8 / 0.65|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|cfg_ctrl|`cfg_ctrl_01`|CFG/native|1.25 / 0.2|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i05_cfg_cycle_curvature|`i05_cfg_cycle_curvature_10`|CFG/native|1.75 / 0.8|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i10_ig_cycle_twohop|`i10_ig_cycle_twohop_09`|IG/mlp|0.8 / 0.65|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i01_cfg_cycle_gain|`i01_cfg_cycle_gain_11`|CFG/native|1.75 / 0.9|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i06_ig_cycle_agreement|`i06_ig_cycle_agreement_11`|IG/mlp|0.8 / 0.9|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|cfg_apg|`cfg_apg_04`|CFG/native|1.75 / -0.5|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i02_cfg_cycle_transport|`i02_cfg_cycle_transport_11`|CFG/native|1.75 / 0.9|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i07_ig_cycle_extrapolate|`i07_ig_cycle_extrapolate_11`|IG/mlp|0.8 / 0.9|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i03_cfg_cycle_antidrift|`i03_cfg_cycle_antidrift_11`|CFG/native|1.75 / 0.9|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i08_ig_cycle_weakgain|`i08_ig_cycle_weakgain_11`|IG/mlp|0.8 / 0.9|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|cfg_apg|`cfg_apg_05`|CFG/native|1.75 / -0.25|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i04_cfg_cycle_common_drift|`i04_cfg_cycle_common_drift_11`|CFG/native|1.75 / 0.9|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i09_ig_cycle_antithetic|`i09_ig_cycle_antithetic_11`|IG/mlp|0.8 / 0.9|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i05_cfg_cycle_curvature|`i05_cfg_cycle_curvature_11`|CFG/native|1.75 / 0.9|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i10_ig_cycle_twohop|`i10_ig_cycle_twohop_11`|IG/mlp|0.8 / 0.9|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|cfg_ctrl|`cfg_ctrl_02`|CFG/native|1.75 / 0.2|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i01_cfg_cycle_gain|`i01_cfg_cycle_gain_12`|CFG/native|2.25 / 0.5|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i06_ig_cycle_agreement|`i06_ig_cycle_agreement_12`|IG/mlp|1 / 0.5|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i02_cfg_cycle_transport|`i02_cfg_cycle_transport_12`|CFG/native|2.25 / 0.5|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i07_ig_cycle_extrapolate|`i07_ig_cycle_extrapolate_12`|IG/mlp|1 / 0.5|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|cfg_apg|`cfg_apg_06`|CFG/native|2.25 / -0.5|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i03_cfg_cycle_antidrift|`i03_cfg_cycle_antidrift_12`|CFG/native|2.25 / 0.5|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i08_ig_cycle_weakgain|`i08_ig_cycle_weakgain_12`|IG/mlp|1 / 0.5|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i04_cfg_cycle_common_drift|`i04_cfg_cycle_common_drift_12`|CFG/native|2.25 / 0.5|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i09_ig_cycle_antithetic|`i09_ig_cycle_antithetic_12`|IG/mlp|1 / 0.5|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|cfg_apg|`cfg_apg_07`|CFG/native|2.25 / -0.25|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i05_cfg_cycle_curvature|`i05_cfg_cycle_curvature_12`|CFG/native|2.25 / 0.5|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i10_ig_cycle_twohop|`i10_ig_cycle_twohop_12`|IG/mlp|1 / 0.5|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i01_cfg_cycle_gain|`i01_cfg_cycle_gain_13`|CFG/native|2.25 / 0.65|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i06_ig_cycle_agreement|`i06_ig_cycle_agreement_13`|IG/mlp|1 / 0.65|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|cfg_ctrl|`cfg_ctrl_03`|CFG/native|2.25 / 0.2|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i02_cfg_cycle_transport|`i02_cfg_cycle_transport_13`|CFG/native|2.25 / 0.65|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i07_ig_cycle_extrapolate|`i07_ig_cycle_extrapolate_13`|IG/mlp|1 / 0.65|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i03_cfg_cycle_antidrift|`i03_cfg_cycle_antidrift_13`|CFG/native|2.25 / 0.65|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i08_ig_cycle_weakgain|`i08_ig_cycle_weakgain_13`|IG/mlp|1 / 0.65|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|cfg_apg|`cfg_apg_08`|CFG/native|2.75 / -0.5|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i04_cfg_cycle_common_drift|`i04_cfg_cycle_common_drift_13`|CFG/native|2.25 / 0.65|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i09_ig_cycle_antithetic|`i09_ig_cycle_antithetic_13`|IG/mlp|1 / 0.65|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i05_cfg_cycle_curvature|`i05_cfg_cycle_curvature_13`|CFG/native|2.25 / 0.65|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i10_ig_cycle_twohop|`i10_ig_cycle_twohop_13`|IG/mlp|1 / 0.65|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|cfg_apg|`cfg_apg_09`|CFG/native|2.75 / -0.25|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i01_cfg_cycle_gain|`i01_cfg_cycle_gain_14`|CFG/native|2.25 / 0.8|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i06_ig_cycle_agreement|`i06_ig_cycle_agreement_14`|IG/mlp|1 / 0.8|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i02_cfg_cycle_transport|`i02_cfg_cycle_transport_14`|CFG/native|2.25 / 0.8|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i07_ig_cycle_extrapolate|`i07_ig_cycle_extrapolate_14`|IG/mlp|1 / 0.8|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|cfg_ctrl|`cfg_ctrl_04`|CFG/native|2.75 / 0.2|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i03_cfg_cycle_antidrift|`i03_cfg_cycle_antidrift_14`|CFG/native|2.25 / 0.8|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i08_ig_cycle_weakgain|`i08_ig_cycle_weakgain_14`|IG/mlp|1 / 0.8|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i04_cfg_cycle_common_drift|`i04_cfg_cycle_common_drift_14`|CFG/native|2.25 / 0.8|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i09_ig_cycle_antithetic|`i09_ig_cycle_antithetic_14`|IG/mlp|1 / 0.8|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|cfg_more_steps|`cfg_more_steps_00`|CFG/native|1.25 / 0|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i05_cfg_cycle_curvature|`i05_cfg_cycle_curvature_14`|CFG/native|2.25 / 0.8|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i10_ig_cycle_twohop|`i10_ig_cycle_twohop_14`|IG/mlp|1 / 0.8|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i01_cfg_cycle_gain|`i01_cfg_cycle_gain_15`|CFG/native|2.25 / 0.9|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i06_ig_cycle_agreement|`i06_ig_cycle_agreement_15`|IG/mlp|1 / 0.9|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|ig_mlp_more_steps|`ig_mlp_more_steps_00`|IG/mlp|0.8 / 0|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i02_cfg_cycle_transport|`i02_cfg_cycle_transport_15`|CFG/native|2.25 / 0.9|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i07_ig_cycle_extrapolate|`i07_ig_cycle_extrapolate_15`|IG/mlp|1 / 0.9|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i03_cfg_cycle_antidrift|`i03_cfg_cycle_antidrift_15`|CFG/native|2.25 / 0.9|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i08_ig_cycle_weakgain|`i08_ig_cycle_weakgain_15`|IG/mlp|1 / 0.9|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|cfg_more_steps|`cfg_more_steps_01`|CFG/native|1.25 / 0|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i04_cfg_cycle_common_drift|`i04_cfg_cycle_common_drift_15`|CFG/native|2.25 / 0.9|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i09_ig_cycle_antithetic|`i09_ig_cycle_antithetic_15`|IG/mlp|1 / 0.9|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i05_cfg_cycle_curvature|`i05_cfg_cycle_curvature_15`|CFG/native|2.25 / 0.9|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i10_ig_cycle_twohop|`i10_ig_cycle_twohop_15`|IG/mlp|1 / 0.9|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|ig_mlp_more_steps|`ig_mlp_more_steps_01`|IG/mlp|0.8 / 0|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i01_cfg_cycle_gain|`i01_cfg_cycle_gain_16`|CFG/native|2.75 / 0.5|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i06_ig_cycle_agreement|`i06_ig_cycle_agreement_16`|IG/mlp|1.2 / 0.5|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i02_cfg_cycle_transport|`i02_cfg_cycle_transport_16`|CFG/native|2.75 / 0.5|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i07_ig_cycle_extrapolate|`i07_ig_cycle_extrapolate_16`|IG/mlp|1.2 / 0.5|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|norm_i02|`norm_i02_00`|CFG/native|1.25 / 0.65|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i03_cfg_cycle_antidrift|`i03_cfg_cycle_antidrift_16`|CFG/native|2.75 / 0.5|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i08_ig_cycle_weakgain|`i08_ig_cycle_weakgain_16`|IG/mlp|1.2 / 0.5|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i04_cfg_cycle_common_drift|`i04_cfg_cycle_common_drift_16`|CFG/native|2.75 / 0.5|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i09_ig_cycle_antithetic|`i09_ig_cycle_antithetic_16`|IG/mlp|1.2 / 0.5|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|norm_i02|`norm_i02_01`|CFG/native|1.25 / 0.8|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i05_cfg_cycle_curvature|`i05_cfg_cycle_curvature_16`|CFG/native|2.75 / 0.5|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i10_ig_cycle_twohop|`i10_ig_cycle_twohop_16`|IG/mlp|1.2 / 0.5|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i01_cfg_cycle_gain|`i01_cfg_cycle_gain_17`|CFG/native|2.75 / 0.65|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i06_ig_cycle_agreement|`i06_ig_cycle_agreement_17`|IG/mlp|1.2 / 0.65|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|norm_i03|`norm_i03_00`|CFG/native|1.25 / 0.65|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i02_cfg_cycle_transport|`i02_cfg_cycle_transport_17`|CFG/native|2.75 / 0.65|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i07_ig_cycle_extrapolate|`i07_ig_cycle_extrapolate_17`|IG/mlp|1.2 / 0.65|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i03_cfg_cycle_antidrift|`i03_cfg_cycle_antidrift_17`|CFG/native|2.75 / 0.65|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i08_ig_cycle_weakgain|`i08_ig_cycle_weakgain_17`|IG/mlp|1.2 / 0.65|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|norm_i03|`norm_i03_01`|CFG/native|1.25 / 0.8|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i04_cfg_cycle_common_drift|`i04_cfg_cycle_common_drift_17`|CFG/native|2.75 / 0.65|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i09_ig_cycle_antithetic|`i09_ig_cycle_antithetic_17`|IG/mlp|1.2 / 0.65|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i05_cfg_cycle_curvature|`i05_cfg_cycle_curvature_17`|CFG/native|2.75 / 0.65|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i10_ig_cycle_twohop|`i10_ig_cycle_twohop_17`|IG/mlp|1.2 / 0.65|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|norm_i04|`norm_i04_00`|CFG/native|1.25 / 0.65|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i01_cfg_cycle_gain|`i01_cfg_cycle_gain_18`|CFG/native|2.75 / 0.8|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i06_ig_cycle_agreement|`i06_ig_cycle_agreement_18`|IG/mlp|1.2 / 0.8|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i02_cfg_cycle_transport|`i02_cfg_cycle_transport_18`|CFG/native|2.75 / 0.8|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i07_ig_cycle_extrapolate|`i07_ig_cycle_extrapolate_18`|IG/mlp|1.2 / 0.8|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|norm_i04|`norm_i04_01`|CFG/native|1.25 / 0.8|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i03_cfg_cycle_antidrift|`i03_cfg_cycle_antidrift_18`|CFG/native|2.75 / 0.8|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i08_ig_cycle_weakgain|`i08_ig_cycle_weakgain_18`|IG/mlp|1.2 / 0.8|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i04_cfg_cycle_common_drift|`i04_cfg_cycle_common_drift_18`|CFG/native|2.75 / 0.8|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i09_ig_cycle_antithetic|`i09_ig_cycle_antithetic_18`|IG/mlp|1.2 / 0.8|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|norm_i07|`norm_i07_00`|IG/mlp|0.8 / 0.65|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i05_cfg_cycle_curvature|`i05_cfg_cycle_curvature_18`|CFG/native|2.75 / 0.8|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i10_ig_cycle_twohop|`i10_ig_cycle_twohop_18`|IG/mlp|1.2 / 0.8|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i01_cfg_cycle_gain|`i01_cfg_cycle_gain_19`|CFG/native|2.75 / 0.9|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i06_ig_cycle_agreement|`i06_ig_cycle_agreement_19`|IG/mlp|1.2 / 0.9|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|norm_i07|`norm_i07_01`|IG/mlp|0.8 / 0.8|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i02_cfg_cycle_transport|`i02_cfg_cycle_transport_19`|CFG/native|2.75 / 0.9|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i07_ig_cycle_extrapolate|`i07_ig_cycle_extrapolate_19`|IG/mlp|1.2 / 0.9|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i03_cfg_cycle_antidrift|`i03_cfg_cycle_antidrift_19`|CFG/native|2.75 / 0.9|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i08_ig_cycle_weakgain|`i08_ig_cycle_weakgain_19`|IG/mlp|1.2 / 0.9|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|reverse_i03|`reverse_i03_00`|CFG/native|1.25 / 0.8|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i04_cfg_cycle_common_drift|`i04_cfg_cycle_common_drift_19`|CFG/native|2.75 / 0.9|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i09_ig_cycle_antithetic|`i09_ig_cycle_antithetic_19`|IG/mlp|1.2 / 0.9|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i05_cfg_cycle_curvature|`i05_cfg_cycle_curvature_19`|CFG/native|2.75 / 0.9|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|i10_ig_cycle_twohop|`i10_ig_cycle_twohop_19`|IG/mlp|1.2 / 0.9|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|reverse_i04|`reverse_i04_00`|CFG/native|1.25 / 0.8|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|reverse_i07|`reverse_i07_00`|IG/mlp|0.8 / 0.8|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|cfg_probe_only|`cfg_probe_only_00`|CFG/native|1.25 / 0.5|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|cfg_probe_only|`cfg_probe_only_01`|CFG/native|1.25 / 0.9|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|ig_probe_only|`ig_probe_only_00`|IG/mlp|0.8 / 0.5|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|
|ig_probe_only|`ig_probe_only_01`|IG/mlp|0.8 / 0.9|0/6|—|—|—|—|—|—|固定配置；不跨轮重选|—|

请求 SHA256：`fb1a729e99870d13fde0240d820738565e02b8ba4fe0fe3e779375167f5636e4`。

原始输出：`/home/zhoushunyu/data/eqvae/experiments/recursive_guidance_20260913/recursive_screen_1k`。

[每个配置每轮的完整 CSV](data/recursive_guidance_20260913/all_results.csv)。
