# Z-Sampling 缓存反演：按用户要求停止

2026-09-13。**按用户“没有新的 idea 其实不用运行了”的要求，已停止这批实验：12/36 个 1K 设置完成，控制器及四个 worker 均已退出，未发现相关残留任务。核心改动是缓存预测与固定点求逆的组合，目前未建立实质性研究新意。已完成数据保留，其余设置不再自动运行。** 同批强基线未完成，没有超过强 baseline 或独立确认收益的证据。

[协议](Z_SAMPLING_IDENTITY_SCREEN_20260913_ZH.md)固定新 seed=2026091481、同一 balanced 1K、同一 ADM 5K ImageNet100 reference。两种 Z 都为 56 个 Euler 主步，前 42 步做事件、后 14 步普通 conditional。每个设置均为 **224 次分支调用/图**。以下 α 是一阶有效 extra 系数，实际前进 w_s=(1+α+w_b)/2；不把它误写成每次强查询的 extra。

|有效 extra α|逆腿 w_b|原 Z FM 适配 FID|缓存反演 Z FID|差值，新−原|相对下降|
|---|---:|---:|---:|---:|---:|
|0.75|0|50.147082|49.355837|-0.791245|1.57785%|
|0.75|1|51.198267|49.443484|-1.754783|3.42743%|
|1.125|0|46.009605|45.858318|-0.151287|0.32882%|
|1.125|1|45.955207|45.895419|-0.059788|0.13010%|
|1.25|0|45.657715|45.621031|-0.036683|0.08034%|
|1.25|1|45.662721|45.593989|-0.068732|0.15052%|

首批 α=.75 的两组对应 sFID 为 218.33956→210.39748、219.22276→210.52824。每个 1K 的采样加解码分别为 150.42→148.52 秒、147.97→149.16 秒；单次运行的这些时间差不作为稳定加速结论。没有重采样置信区间、独立 5K 或已完成的本批 CFG/APG/CTRL 结果。

方法只改变逆腿的初始化和缓存利用：q1=x+h[A(x)−B(x)]，q2=x+h[A(x)−B(q1)]，再从 q2 强前进。A/B 同场时，两次逆迭代都保持 x，这已在 CPU 和实际 SiT 上验证；这个恒等性质本身不等于生成质量。真实模型预检覆盖全部 36 设置及 6 个同场例，预算和原生基线一致性均通过。

初始四个参数臂的缓存批次已核对标签、像素形状和 dtype、有限 latent、224-call 计数。停止时共有 12 个 FID 落盘；α=1.25 两组仅降低约 0.037 和 0.069 FID，不能以首批弱参数的较大差值概括全部收益。原 tmux 为 **z_identity_0913**，其中这批实验已结束。中断属于用户停止决定，不记为数值失败。

- [已完成参数结果 CSV](/home/zhoushunyu/data/eqvae/experiments/z_sampling_identity_20260913/screen_1k/results.csv)
- [状态](/home/zhoushunyu/data/eqvae/experiments/z_sampling_identity_20260913/screen_1k/status.json)、[冻结请求](/home/zhoushunyu/data/eqvae/experiments/z_sampling_identity_20260913/screen_1k/request.json)
- [缓存反演、普通条件逆腿首批图册](/home/zhoushunyu/data/eqvae/experiments/z_sampling_identity_20260913/screen_1k/z_anchored_euler_a0.75_b1/grid.png)
- [实际模型核查](/home/zhoushunyu/data/eqvae/experiments/z_sampling_identity_20260913/model_check/summary.json)

之前的[真实反演先验尝试](CFG_INVERSE_PRIOR_RESULTS_20260913_ZH.md)在完整 bank 的数值可行性阶段停止，未拟合真实先验或评估 FID。当前 Z 是用户随后明确授权的方向，不把前一项的数值失败误说成质量方法被证伪，也不恢复其它暂停的混合/调制扫描。
