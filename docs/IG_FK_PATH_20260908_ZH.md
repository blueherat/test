# 直接随机路径 Feynman–Kac 修正：固定首轮协议

用户提供分析：attachments/a11a1076-1658-4353-8514-f5868767a867/pasted-text.txt。
本轮检验其随机未来累积势方法，不训练/蒸馏 PFR，不拟合 value 网络。

理论与仓库 SEMIGROUP_CONSISTENT_GUIDANCE_THEORY_ZH.md 第7节相同，不能声称
首次提出。旧实现用 OU 坐标归一化 HJB 标量网络，真实 RAEv2 配对质量失败；
这里直接对当前采样状态进行随机路径积分并求输入梯度，绕过该网络估计。

## 明确实现

RAE bridge z=(1-t)x+t eps。heat 坐标 y=z/(1-t)，tau=(t/(1-t))²。
强弱 clean 输出 F/B，w=1.78。heat score 为 (D-y)/tau。
辅助随机路径始终使用普通 IG score [B+w(F-B)-y]/tau。
势 V=.5*w*(w-1)*||(F-B)/tau||²，使用平方和，不按维度平均。

每个 0.1<=t<1 的主采样节点：取两个 antithetic 随机分支；
在 [tau, .75tau] 上做两步等 heat-time Euler-Maruyama，左端点势积分。
第二步末尾的随机数不影响左端点积分，被解析积分掉，因此只需生成第一次创新。
C 的末端边界设为1；这是截断，不是全未来解。
计算 logmeanexp(Lplus,Lminus)，对初始 z 求完整路径梯度，包括随机未来
状态对 z 的依赖以及沿途反复计算的强弱头。输入重参数化噪声独立于 z。
主 ODE 加入 -t/(1-t)*grad_z log C；原 IG 全程保留。
不进行 gain、范数限制、温度、维度缩放、时间窗口扫描。

有限粒子 log moment 及其梯度有 Monte Carlo 偏差；高维指数权重可能退化。
记录 ESS、两分支 logweight 差、修正 RMS，但不以机制指标代替生成质量。
神经头未被证明对应同一前向过程的合法 score；本轮只检验 plug-in 方法。

## 质量与成本

复用 native IG 的已有 1K：seed202609413，B4，100 Euler，shift8，
FP32 state / BF16 autocast / TF32，labels0..999；FID38.264238946601836。
只采样新方法，四卡按全局 batch 下标轮流分片，初始噪声与 baseline 哈希一致。
辅助噪声使用独立 CUDA generator，seed202609801+global_batch_start，
不改变主初始噪声 RNG。100个主 Full forward；98个修正节点，每节点
2分支×2次 Full forward，加392次checkpoint backward重计算；
共884次 Full forward执行以及196次输入梯度反传/图像，0 prefix。
先测首批（保留正式输出），再完成四卡1K；实际耗时与峰值显存落盘。

新方法若1K无实质提升，不扩5K，不扫描参数。若有清晰提升再决定独立5K。
理论正确不等于质量提升；此结果也不代表完整精确 FK 目标必然无效。
不写论文。
