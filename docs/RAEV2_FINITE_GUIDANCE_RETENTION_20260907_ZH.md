# 原生有限 guidance 写入与 Full 后续读取

2026-09-07。状态：128样本完成，独立CPU快照及有限递推核验通过；下文保留执行前协议。用户强调 guidance 信息进入 latent 的核心，并指出 IG 外推很大，不能以未验证的一阶展开论证写入。前一轮单次 Jacobian 方案已经暂停，见[方向回顾及修正](RAEV2_LATENT_GUIDANCE_REVIEW_20260907_ZH.md)。

## 这次只回答一个机制问题

在一条真实 RAEv2 原生 IG 轨迹上，只保留某一步的完整 IG 写入，随后撤去额外 IG、让 Full 推进。这一步的信息是否能通过 Full 的有限响应进入后续预测及终点？是否存在主要被擦除、改变方向或持续增大的阶段？这是建立新写入约束前需要的证据，不把“写入后有影响”视作新方法，更不替代当前5K约3% FID目标。

这里的“信息”仅指可观测的冻结 guidance 消息方向及配对响应，不预设 Shannon 信息量、类别语义正确性或数据流形。Full/Base 同类训练，两者差异不自动等于真实条件信息。

## 精确的有限响应关系

当前为 t、下一步为 s，q=(t−s)/t，F 为 Full、G 为完整原生 IG。使用原始 FP32 Euler 运算分别形成

    z_s^0 = Euler(z_t,F)， z_s^1 = Euler(z_t,G)， δ_s = z_s^1−z_s^0。

实数下 δ_s=q(G−F)；机器上保留两次原生 Euler 的真实差异，并记录重排公式的舍入误差。写入实际发生在后继状态 s，不能原样搬到当前 t。

从 s 开始，两条轨迹都只用 Full。记 a_j=t_(j+1)/t_j、q_j=1−a_j，则精确有限关系为

    Δz_(j+1) = a_j Δz_j + q_j [F(z_j^1,t_j)−F(z_j^0,t_j)]。

括号项直接查询真实网络；不换成 J_F Δz，不对 β−1 展开。对应机器运算也记录递推舍入残差。没有后续模型响应时，原始写入被动传递为 (t_j/s)δ_s；总差减去它便得到后续模型响应累计造成的差别。噪声时间趋零时，被动项消失，但这不能推出终点差异朝向真实数据或改善 FID。

## 冻结的实际模型诊断

- 模型：strict RAEv2 DINOv3-L K7 EMA100080，官方100步 shift8 Euler、IG=1.78、原窗口[.1,1]，原生 BF16 heads/mix、FP32 states、TF32 on、B8。checkpoint/config实际 SHA 与历史配对5K身份核对。
- 输入：已有 `actual_ratio_bank64k/train` 的原生实际生成状态。固定 query index `[4,16,32,48,64,80,92,96]`，每个 query 两个原始 B8，合计128样本。仅按预定 namespace 对 batch id 的 SHA 排序选取，先写 cohort 再读状态值。该 bank 已用于探索，不称独立确认。
- 核验：四个源 request/metadata 哈希、每个选中 B8 的 state/noise 原始字节 SHA；不假称重新哈希全部16GB文件。另为每个 worker 的首个 B8 从缓存初始 noise 重建原生前缀，要求实际状态逐位一致。原生 IG wrapper parity、相同输入 Full 重复读取及 equal-head 零消息作实现控制。
- 三臂：当前步 Full、后续 Full；当前步 IG、后续 Full；当前步及后续均原生 IG。它们共享当前状态、类标签、时间网格，无新随机数、无输入梯度。第三臂用于观察单次写入与全部后续指导的差别。
- 读取：记录所有剩余原始 query 和终点，不选有利 horizon。冻结当前 `M=G_native−F.float()`，报告真实 Full 差异对 M 的投影系数、cosine、RMS，以及状态差异、被动传递与累计响应。另报告两分支 BF16 输入不同的坐标比例，帮助判断数值可见性。
- 终点：报告单次写入终点差对 M 的投影；另投影到“完整剩余IG−Full”的实际终点差上。后者只是向量投影，不是指导效果、互信息或 FID 的占比。
- 保存：固定输入、首个读取和三个终点快照、每个 query 的逐样本统计、模型调用和耗时；大数组放数据盘。

只有两个 B8 cluster/时刻，本轮作描述性机制诊断，不根据128行样本制造精确的群体显著性结论。本轮不解非线性写入优化，不扫描系数、不生成图像、不评价 FID；候选约束是否有依据由实际结果决定，之后再冻结一个生成方法。

实现：[raev2_finite_guidance_retention.py](../experiments/raev2_finite_guidance_retention.py)。数据目录：`/home/zhoushunyu/data/eqvae/experiments/raev2_finite_guidance_retention_20260907`。父进程使用有界 detached coordinator 管理四个 worker，退出后停止，不创建长期守护任务。

## 结果与实际改变的判断

全部128个样本的单次写入终点差对冻结消息有正投影；不支持“RAEv2普遍擦除一次guidance写入”。8个时刻组的终点消息投影均值分别为 `.01719/.02287/.01814/.01513/.01279/.01985/.06780/.18038`。除以原一步 q 后的有限投影增益约为 `12.62/13.09/7.02/3.59/1.59/.92/.81/.87`；这些是本次有限干预的实测比例，不是Jacobian或beta趋零的导数。

终点差与原消息的平均 cosine 则从早期 `.041/.073/.103` 增至最后两个时刻组 `.560/.811`。与完整后续IG终点差的 cosine 也存在相似时间差异。这支持“信息已经进入latent，但后续响应会改变方向”这一有限观察；不证明正交部分有害，不把低cosine直接叫作语义丢失。不同query使用不同cohort，不能把均值折线称作同一批样本的时间演变。

[描述性图及可导出PDF](../experiments/results/raev2_final_three_rounds_20260907/prelude_native_retention/finite_retention.pdf)；[全部逐query统计与数据索引](../experiments/results/raev2_final_three_rounds_20260907/prelude_native_retention/archive_manifest.json)。

保存5760行逐样本读取记录；总17760次样本模型前向，无输入反向、解码或FID。四worker采样/记录耗时32.63/32.38/25.54/25.88秒；完整控制器另含模型加载和合并。四个原生前缀重建均逐位一致。独立 NumPy 从快照重算全部首读/终点统计、核对全部后续投影递推，共8320项比较，最大快照指标误差1.78e-15，实际有限递推舍入RMS最大4.95e-8。核验没有重新执行另一个完整模型实现。

这一观察促成了[最后三轮中的第1轮](RAEV2_FINAL_THREE_ROUNDS_20260907_ZH.md)：固定原生latent写入及读取能量预算，尝试改变编码方向；该固定候选随后未通过自己的机制筛查，不能以本节的正投影现象代替它的成功。
