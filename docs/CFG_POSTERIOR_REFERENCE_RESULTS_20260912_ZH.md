# 纯CFG：配对条件teacher与独立条件teacher训练null

最终判定：7/7组完成、全部raw及缓存FID/sFID详细审计通过，posterior候选未达到预设推进条件。配对teacher的FID45.6688优于独立类别teacher46.6401，但未胜过同预算real/FM45.5388或APG44.5038；不追加训练、调强度或启动5K。独立类别对照的sFID204.7193反而更低，不能把它描述为所有质量维度都更差。当前只有单个训练种子与重复探索bank，不能据此认领后验一致性机制解释了生成收益。

[完整研究报告](GUIDANCE_REFERENCE_LOSS_RESEARCH_20260912_ZH.md) · [全组详细审计](data/posterior_reference_screen_1k/audit.json)

已完成 7/7 组配对1K。强主干冻结，IG/CFG均不增加主干查询。

|方法|FID↓|sFID↓|IS↑|Full/prefix|采样及解码GPU秒|
|---|--:|--:|--:|--:|--:|
|cfg_original_00|45.707369|206.410081|60.6777|224/0|155.79|
|cfg_real_00|45.538838|206.387671|60.6405|224/0|160.35|
|cfg_posterior_00|45.668802|206.413923|60.7464|224/0|154.33|
|cfg_independent_00|46.640140|204.719274|58.9866|224/0|160.82|
|cfg_cfg_00|45.056926|207.874933|61.5936|224/0|164.93|
|cfg_half_00|51.007606|210.398537|52.5425|224/0|159.28|
|cfg_apg_07|44.503757|209.855289|62.4432|224/0|177.64|

仅两个新null头：拟合配对真实类别下的strong，或拟合独立类别下的strong。null自身均看不到类别。已有real/FM和CFG-data头复用前轮权重；全部质量组重新生成。1500步，null输入、初始化及噪声时间随机流相同。

|训练|读出参数|训练GPU秒|验证guided MSE前→后|
|---|--:|--:|--:|
|cfg_posterior|308000|42.98|0.740157→0.740160|
|cfg_independent|308000|42.34|0.740157→0.741959|

预测风险改善不自动等于生成质量改善。此1K沿用历史探索bank，不是独立确认；未调guidance强度。

[冻结协议与理论边界](CFG_POSTERIOR_REFERENCE_PROTOCOL_20260912_ZH.md) · [逐组数据](data/cfg_posterior_reference_20260912/all_results.csv)

请求SHA256：77f5dc2b094b92ff6cddcd20b433bef47f714c81802e57f94640097faa018cbc。
