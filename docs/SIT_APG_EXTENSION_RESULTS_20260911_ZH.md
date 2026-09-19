# APG机制延伸的1K筛选

已提交 76/201 组；60组新增候选、141组对照，数值失败 0。

沿用53-idea队列的同一1K噪声及标签，每类10张；机制实验的32个种子独立。不会自动追加5K。

|方案|完成数|最低FID|对照|ΔFID|成本×对照|引导量 / 参数|内部解码、分类张数/输出|
|---|--:|--:|---|--:|--:|---|---|
|54 非保守响应触发主采样细分|0/12|—|—|—|—|—|—|
|55 带语义约束的未来幅度投影|12/12|44.405479|moment_without_semantic_01|-0.093595|1.003|1.25 / 10.0|21, 21|
|56 真实续生成选择APG平行保留量|12/12|44.527007|moment_without_semantic_01|+0.027933|0.667|1.25 / 0.0|9, 9|
|57 边际条件价值驱动的撤条件|12/12|43.918159|moment_without_semantic_01|-0.580915|0.652|2.0 / 0.0|10, 10|
|58 统一终点检验的前瞻时距选择|0/12|—|—|—|—|—|—|

|对照family|完成数|最低FID|参数|
|---|--:|--:|---|
|strong|0/1|—|—|
|ig_local|0/1|—|—|
|cfg_native|0/4|—|—|
|cfg_apg|0/15|—|—|
|clean_apg|4/36|44.499828|2.0 / -0.5|
|projection_apg|12/12|44.541561|1.25 / 0.0|
|uniform_refinement|0/8|—|—|
|embedded_refinement|0/12|—|—|
|erk_guid|0/12|—|—|
|curl_gain_shrink|0/4|—|—|
|semantic_direction|4/4|44.528325|1.25 / 1.0|
|moment_without_semantic|4/4|44.499074|1.25 / 1.0|
|fixed_null_release|12/12|44.855441|1.25 / 0.75|
|probability_null_release|4/4|47.753454|2.75 / 0.5|
|fixed_horizon_verified|0/12|—|—|

ΔFID为候选减已完成的同语义资产组最佳对照。另需查看逐组CSV中的计算成本和专门消融；预算未强行相等。网格最低值只是筛选结果，不代表统计显著性或独立确认。

原APG适配与新增clean-buffer APG分别保留。主采样、全部候选探针、内部VAE/ConvNeXt和最终解码均计入采样成本；FID提取单列。逐批审计实际选择和接管记录。

[机制与推导](APG_MECHANISM_AND_FIVE_EXTENSIONS_20260911_ZH.md) · [冻结协议](SIT_APG_EXTENSION_PROTOCOL_20260911_ZH.md) · [逐组结果](data/apg_mechanism_extension_20260911/all_results.csv)。

请求SHA256：`ae614bf3b730445db287cbdfa66df46d94a49eebbee65ba84dd04a2610ccdc6e`。
