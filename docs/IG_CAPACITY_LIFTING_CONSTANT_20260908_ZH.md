# 用户指定：持续 lifting 强度 .5 / .65 / .85 / .9

此协议替换旧四组计划；不修改旧冻结协议或源码。
原a078队列已停止，保留228图；1.25和fade均未启动。
FK仍暂停，保留500图，不自动恢复。

方法不变：L=(Phi_W)^(-1) Phi_S，z'=z+alpha*(L(z)-z)，
随后由Strong推进同一区间。lifting本身承担引导，不额外叠加nativeIG。
Full目标和Base逆流采用Heun；主Strong采用Euler细步。
逆流只是连续流逆的数值近似，不是精确离散逆。
每段最多4个原始细步，每段一次部分写入；不求同时间固定点。

只做用户指定四组，按此顺序运行：
1. a050_constant，alpha=.5
2. a065_constant，alpha=.65
3. a085_constant，alpha=.85
4. a090_constant，alpha=.9

全部持续策略：保持固定alpha直到原生有效窗口结束(t<.1)，
无提前停止、无衰减。每图298Full+198Base-prefix，无梯度或训练。
四组顺序运行，每组四卡协同分片1K，禁止一张卡独跑一个设置。

同原生baseline：seed202609413、B4、labels0..999、100步shift8、
FP32state/BF16autocast/TF32，原EMA100080及Stage1。
复用原生IG已有FID38.264238946601836和样本，不重跑baseline。
每组自动合并及官方FID评估，完整保存源码/协议/噪声/标签/输出哈希及耗时。

同一个1K上选择最佳alpha具有选择偏差；不能当成独立验证或论文级突破。
先跑完用户指定对照，不擅自扩大系数/策略扫描。不写论文。
