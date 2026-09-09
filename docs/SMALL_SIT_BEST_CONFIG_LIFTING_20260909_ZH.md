# 在历史最强分段 IG/PFR 工作点验证 lifting

用户纠正：应在历史强 IG/PFR 配置上检验 lifting，而非固定 .35 的迁移基线。
确认历史 bank：pfr_query_controls_v1/fid1k_seed0；IG64.85129841789944，
projected PFR61.85920747125374。既有结果复用，样本按历史规则已删除，标签、
manifest、指标和preview仍在；不能声称新的像素逐元素复现。

固定 ImageNet-100 SiT-S/2 v800K EMA、depth4 v50K、B8、FP32/TF32、
Dopri5 rtol1e-3 atol1e-6、原VAE/ADM reference。精确复用旧单bank RNG：
第b批CUDA generator.manual_seed(b)，先randn再randint。这里只复现历史
配对bank，不将旧seed0/1视为独立重复。先验证全局噪声和标签哈希。

新方法仅一组 lifting_best_schedule：原生data-time [0,.25) alpha=.6，
[.25,.5) alpha=.7，[.5,1] alpha=0。lifting承担引导，不额外叠加普通IG。
沿用现有lifting的100点细网格、每段最多4细步的有限写入区间，段边界
不得越过.25或.5；目标Full和Base逆流保持Heun细步。
主Strong推进改用历史Dopri5精度；每次写入后从该段起点积分到段终点，
.5后一次Strong Dopri5到1。不把附加的分段重启称为与原单次ODE调用
完全相同：这是有限写入方法的离散实现差别，记录主/辅助调用和实测成本。
不因该次结果临时改变horizon、强度或求解器。无额外PFR、无新基线全量采样。

四卡合作分片同一个1K，完整校验模型、源码、标签、噪声、样本覆盖与评价。
接口预检直接比较旧Runtime与新接口的Strong/Weak输出；已删除的旧样本不伪造。
使用1K判断是否值得进一步验证，不与另一bank的66.4540直接比优劣。
