# 小 SiT portfolio：固定候选新 5K 确认

用户授权先做 5K，再融合候选继续调优。本阶段固定上一轮 649 组 paired 1K 筛选出的四个候选，与三个原生对照一起生成全新、配对的 5000 图；不使用本 5K 选择参数。

|配置|固定参数|来源|
|---|---|---|
|原生 Heun IG|peak=.7，cutoff=.5|12 点 IG 曲线最佳|
|#40 局部 attention 弱前缀|peak=.8，locality=2，cutoff=.5|原 12 点最佳|
|原生 Dopri5 IG|peak=.7，cutoff=.5|12 点 IG 曲线最佳|
|原生 CFG|strength=1.25，cutoff=.75|两时间窗、24 点最佳|
|#1 CFG+APG|strength=2，beta=-.5，cutoff=.75|原 12 点最佳；强度在上界|
|#38 CFG 条件割线|strength=1.5，condition_scale=.5，cutoff=.75|原 12 点最佳；条件参数在下界|
|#3 CFG 通道方差回缩|strength=1.5，mix=.75，cutoff=.75|原 12 点最佳；mix 在上界|

全部使用原封不动的已冻结 portfolio 数值算子、模型、VAE、精度设置和 ADM 参考；模型为 ImageNet100 SiT-S/2 800K、原生已训练 depth4 v-head。新 bank 使用连续 CUDA generator seed=202610060，顺序 B8 抽样；标签 seed=202610061，每类50张后排列。随机查询仍按 batch noise SHA256 决定，保持与原版本相同的定义。

主配置均为 Heun64；Dopri5 对照保持原分段及 rtol=.001、atol=1e-6。IG 前 .25 段峰值乘6/7、.25至.5用峰值，之后关闭；peak=.7 时保持旧 .6/.7 浮点常数。CFG strength 为额外增量系数，常见 CFG scale=1+strength。所有类标签、初始噪声按样本一一配对。四卡共同完成每一个配置，依次评估。

准备时校验上一轮25个冻结源文件、模型和拟合资产、原输入及评估图；冻结本阶段脚本、协议、输入哈希、选择记录和旧首批输出。每张卡在新采样前逐元素复现全部七个配置对应的旧首批 B8 latent，并确认原生双头/prefix、零强度和 hook 恢复。任意不匹配在正式质量采样前停止。所有原始 batch、聚合像素/latent、feature 与 FID/sFID/IS、实际 Full/prefix 调用、采样/解码计时保留。

主比较为 #40 对当批原生 Heun/Dopri5 IG；CFG 候选对当批原生 CFG。报告 FID、sFID、IS 和计算成本，保留负结果。确认的是固定配置跨新输入 bank 的表现，不等同于统计显著性、同成本最优、已证明新颖或所有超参下的优势。后续融合调优使用另一套独立 1K bank，胜者需要第三套新 5K；本阶段数据不用于融合参数选择。

入口：`python -m experiments.sit_guidance_portfolio_confirmation_20260910 --prepare`，然后 `--run-prepared`。tmux 会话计划为 `sit_portfolio_5k_0910`。输出：`/home/zhoushunyu/data/eqvae/experiments/sit_guidance_portfolio_confirmation_20260910`。脚本与原始父实验的版本保持分离；完整运行后更新可读结果报告。
