# 小 SiT 两阶段普通 IG 对照

用户要求补充两阶段 IG 对照，以检验 lifting 低噪关闭的收益是否仅来自引导窗口。
仅新增 ig035_cut02、ig035_cut05 两组各 1K；普通全程 IG、PFR 和四组 lifting 均复用。

强模型为 ImageNet-100 SiT-S/2 v800K EMA，弱头为已有 depth4 v50K。
主采样 100 步原生 data-time 0->1 Euler，前段 F+.35*(F-W)，后段仅 F。
cut02 在 data-time>=.8 停止 IG（前80步有引导）；cut05 在>=.5 停止
（前50步有引导）。使用整数步边界，与 lifting 相同，类别条件始终保留。
每步只运行一次 Full，有引导时同时读 depth4 头，不额外运行 prefix。
每张100 Full，共享弱头分别80/50次；不开辅助流、不做状态 lifting。

seed202609417、continuous CUDA RNG、B8、原100类随机标签序列、FP32/TF32、
原VAE和像素转换、ImageNet-100 ADM reference 均保持不变。四卡协同一组，
两组顺序运行，保存输入/源码/输出哈希、调用数、时间和完整样本。
核对全局输入哈希与旧基线相同；8张全程IG像素复现仅作接口预检，不重跑基线。

比较相同 cutoff 的普通 IG 与 lifting，而非只比较全程 IG。
FID-1K 是探索性结果，不作独立5K结论。JiT 保持暂停，XL不运行。
