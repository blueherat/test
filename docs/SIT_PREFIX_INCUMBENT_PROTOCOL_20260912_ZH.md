# 条件性补充对照：固定历史融合IG

2026-09-12。在当前native弱前缀的5K结果尚未生成时记录。仅当原三组5K确认中native同时优于原IG与局部IG，才执行本补充；否则保留协议而不生成。

复查历史资产发现`ig_local_residual_11`在2026-09-10另一组5K中FID为38.998526，优于当轮原IG40.457926。当前三个对照若未包含这项已有结果，便不足以判断是否超过仓库较强的已确认IG配置。本补充只重放这个固定配置，不调整其中的强度、局部化或其他参数。

使用`/home/zhoushunyu/data/eqvae/experiments/sit_guidance_fusion_20260910/selected_5k/request.json`保存的完整配置及原算法。内部分发所需的`inherited_exact=True`只将请求交给原采样函数，不改变原函数参数行为；预检必须与旧5K首批latent逐位一致。

生成5000图，CUDA噪声种子2026091205、标签种子2026091206，逐字节验证noise及labels与当前`prefix_confirm_5k`完全相同。每图128 full+64 prefix。这个补充属于同一次确认中的额外基线，**不是第二个独立噪声复验**。

比较native与该旧方法的同bank FID、sFID、IS和实际GPU时间。即使native超过它，也只称为当前固定配置的质量优势；没有跨模型结论或独立的新理论。如果native落后，保留对原IG/局部IG的已确认收益，同时明确尚未超过较强旧方法。不把结果触发成新的参数搜索。

开始前核验历史请求、源码、checkpoint、输入与完成审计；一条旧轨迹重放通过后冻结新包装器。全部分片/聚合数据、覆盖率、调用次数与缓存特征独立FP64 FID/sFID均须核验。

[原三组确认协议](SIT_PREFIX_CONFIRMATION_PROTOCOL_20260912_ZH.md) · [历史融合确认](SIT_GUIDANCE_FUSION_CONFIRMATION_RESULTS_20260910_ZH.md) · [研究循环复盘](GUIDANCE_RESEARCH_CYCLES_20260912_ZH.md)
