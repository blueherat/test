# 高阶粗化内部预测头：固定强度实验

已完成 3/3 组配对1K。强度0.8、深度4、原IG时间区间全部固定。

|内部预测头|FID|Full/prefix调用|采样及解码GPU秒|完成|
|---|--:|--:|--:|---|
|ig_internal_original|65.402814|128/0|97.92|True|
|ig_internal_native|65.000022|128/0|99.71|True|
|ig_internal_clt|65.651362|128/0|101.54|True|

两个新头均在同一原50K头上短训1500步；训练开销另报，不混入采样NFE。

- native：10.60 GPU秒，301840 个训练参数。
- clt：10.64 GPU秒，301840 个训练参数。

历史同bank局部IG为63.202383，但需要额外64次prefix；当前三组不使用局部化。若CLT只胜过短训native、不胜过原头，不能认领IG改进。

[理论、训练约束和失败条件](SIT_INTERNAL_COARSE_PROTOCOL_20260912_ZH.md) · [完整逐组结果](data/sit_internal_coarse_20260912/all_results.csv)

请求SHA256：7048d24af0c77bf68929186660ca511aa13ec5f91c5370ce3c228e3cfaf67377。
