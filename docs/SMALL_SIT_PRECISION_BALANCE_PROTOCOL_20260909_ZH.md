# 小SiT：后验精度平衡的直接生成筛查

2026-09-09。用户要求继续从FSG研究IG的等式与信息载体，并已授权具体候选直接在小SiT试。
本轮测试一个由局部Gaussian后验近似得到的平衡式，不恢复已停止的lifting宽扫。

理想后验qS=N(mS,CS)、qW=N(mW,CW)，额外强度gamma、beta=1+gamma。
若Q=beta*inv(CS)-gamma*inv(CW)正定，则power posterior的均值满足

    beta*inv(CS)*(m-mS) = gamma*inv(CW)*(m-mW),
    m = mS + gamma*CS*inv(beta*CW-gamma*CS)*(mS-mW).

这是标准Gaussian乘除法的应用，允许mS!=mW，不是新定理或质量保证。
两头实际网络不一定对应合法后验；本轮使用局部低维响应近似，不能把近似当精确posterior。
Gaussian后验闭合、Tweedie协方差读数和低维投影均是实质假设。

固定同一ImageNet-100 SiT-S/2 v800K EMA、depth4 v50K、FP32/TF32、B8。
noise/labels与上一组三配置完全相同，各1000张。主Dopri5 rtol=.001、atol=1e-6，
分段0/.125/.25/.375/.5/1；gamma=.6/.6/.7/.7/0。复用上一组同分段普通IG
FID65.139320396445及其完整像素、latents、特征，不重采对照。

仅两个新配置：

- precision_rank1：沿当前clean头差的方向读取两头Jacobian的Rayleigh响应；仅改变该方向的系数。
- precision_rank2：第二方向固定为两头clean响应之差对第一方向的正交余量；在此二维空间求平衡。

采用RMS内积，方向的RMS归一到1；中央差分位移RMS为当前state RMS的1%。
clean预测为m=z+(1-t)*v；其Jacobian与理想后验协方差相差同一正标量，平衡公式中消去。
第一方向每次需两次额外Full配对查询，第二方向再需两次；全量实际NFE/耗时报告。
这不是相同计算预算的质量比较。

两个投影矩阵先取对称部分。只有CS、CW及beta*CW-gamma*CS的最小特征值>1e-6，
且各自条件数<1000时采用该级近似；否则二维退回一维，一维退回原普通IG。
原时间t=0直接普通IG，因为不能在那里从Jacobian反推出后验协方差。
方向RMS<1e-6时不采用该方向。这些是固定数值适用性规则，不能读FID后修改。
记录各级采用率、有效引导系数、正交修正、局部不对称程度与全部batch终点。
预检额外记录1%和0.5%差分半径的输出差异；它是近似敏感度，不是精确导数认证。

四卡协作每个新配置，先复现旧同分段IG首批latent与两头原入口，检查zero-gamma退化。
然后直接各采完整1K并做ADM FID/sFID/IS；数值失败保留已完成batch，不静默调参。
保留源码/模型/输入/输出哈希与独立FP64特征空间FID核验。
两个新配置完成后结束，不自动扩大秩、步长、强度或窗口。
所有结果仍是已用于历史探索的1K，同bank比较不构成独立确认或新颖性证据。
