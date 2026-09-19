# 高阶粗化弱前缀：固定深度和强度实验

已完成 4/4 组配对1K。强度0.8、深度4、原IG时间区间全部固定。

|弱参考|FID|Full/prefix调用|采样及解码GPU秒|完成|
|---|--:|--:|--:|---|
|ig_prefix_original|65.402327|128/64|114.30|True|
|ig_local|63.205382|128/64|127.43|True|
|ig_prefix_native|61.819696|128/64|118.23|True|
|ig_prefix_clt|66.300268|128/64|116.09|True|

两个独立弱前缀均由同一强模型前4层及原50K头初始化，训练1500步；训练开销另报，不混入采样NFE。

- native：68.27 GPU秒，11233552 个训练参数。
- clt：67.38 GPU秒，11233552 个训练参数。

四组每图均使用128 full+64 prefix，局部IG作为正式同bank对照。原IG重算前缀仅为计算布局对照。CLT必须超过原IG、同量native训练与局部IG，才进入固定配置确认；失败则终止本轮C_4构造链。

[理论、训练约束和失败条件](SIT_PREFIX_COARSE_PROTOCOL_20260912_ZH.md) · [完整逐组结果](data/sit_prefix_coarse_20260912/all_results.csv)

请求SHA256：439f12298d7121e7f10bdab95ba8eae23fc8a9f2929831c9ffa1f17843b84743。
