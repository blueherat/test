# 先修改概率再聚合的最小方法实验

已提交 9/9 组配对1K，数值失败 0。

一个候选，额外概率对比强度固定为1；其余为原生基线和有区分度的对照。没有逐层选择、局部质量探针或强度网格。

|方法|FID|Full/prefix调用|采样及解码GPU秒|完成|
|---|--:|--:|--:|---|
|strong|87.083500|128/0|101.14|True|
|ig_local|63.202395|128/64|127.88|True|
|cfg_native|45.707434|224/0|172.25|True|
|cfg_apg|44.503971|224/0|174.48|True|
|routing_odds|84.780223|128/0|137.31|True|
|routing_linear|84.308418|128/0|141.07|True|
|routing_temperature|261.624191|128/0|113.17|True|
|embedding_extrapolation|86.979916|128/0|95.89|True|
|manual_kernel|87.077339|128/0|112.62|True|

概率候选每次前向还有12次额外QKV和logit乘法，不能把同为128次完整前向称为等算力。当前attention只作为待验证的解释概率代理；保持attention聚合的凸性不保证终点图像质量。

[概率对象、反例与失败条件](SIT_POSTERIOR_GUIDANCE_PROTOCOL_20260912_ZH.md) · [完整逐组结果](data/sit_posterior_guidance_20260912/all_results.csv)

完成后复盘，不自动加密参数或启动旧宽队列。

请求SHA256：523e8c22f0c4e566021e4b83fe3cd5b751b97460d56872889f6d33cf45149fe6。
