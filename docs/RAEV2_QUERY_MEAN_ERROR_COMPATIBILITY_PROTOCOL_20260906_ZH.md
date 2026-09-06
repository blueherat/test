# 固定 query 均值弱分支：同类误差放大机制检验

日期：2026-09-06。状态：在计算新弱分支与 clean X 的交叉量前冻结设计。现有[结构诊断](RAEV2_DECODER_QUERY_MEAN_RESULTS_20260906_ZH.md)已经读过全部 Δ/d 的方向和大小；本轮是后续机制检验，不声称是在未见状态上预注册。它不定义 sampler、gain、窗口或图像质量结论。

## 问题、恒等式与解释范围

在 teacher bridge `z=(1−t)X+tε` 下，令 `m=E[X|z,t,y]`、`e_F=F−m`、`e_W=W−m`、`Δ=F−W`。只要 Δ 是该状态与类别的确定函数，就有

`C = E[(X−F)·Δ]/D = −E[e_F·Δ]/D`，`Q = E||Δ||²/D`。

因此对任意固定实数 a，准确的 clean-risk 恒等式是

`R(F+aΔ)−R(F) = −2aC+a²Q`，`R(W)−R(F)=2C+Q`。

这里 `D=1024×16×16`，所有风险和内积均按坐标平均。若弱误差确为同一强误差的共线放大 `e_W=κe_F, κ>1`，且误差非零，则 C>0。若总体 C≤0、Q>0，没有正 a 能降低这个 teacher clean risk。正 C 是该解释的必要方向证据，不能证明误差共线，更不是 FID 保证。

与直接比较 `cos(X−F,X−W)` 不同，这个交叉量中不可约 posterior noise 的总体期望为零，不会把共享目标噪声本身当作误差兼容。有限数据的交叉量仍有随机误差；本轮只有 8 个真实图像/类别、各自复用同一 ε 的 10 个时刻，不能把 80 行当独立样本，也不能从有限负值证明总体 C≤0。图像内坐标相关，不将名义 latent 维数当有效样本量。

`C/Q` 只作为该二次风险的描述性 oracle 最优系数；不拿它作为部署 gain，不拟合其时间曲线，不筛选时刻。全部时间保留并按时间汇总。Base 原 gap 的对应 C/Q 同样计算，作为已有 IG 的机制对照；现有证据已表明 paired MSE 与 IG 的 FID 可以方向相反，因此不能将这个局部风险当最终准入门槛。

t=1 也保留为 clean 预测诊断；由 clean 到 epsilon 的公式在该端点会退化，不能据该行另声称 epsilon guidance 有效或两头对应 exact scores。

## 唯一固定输入与前向

使用 `decoder_query_mean_v2` 中完全相同的 10 个 teacher snapshot、每个 8 个 ID：step `000,047,067,077,084,089,092,095,097,099`，共 80 行。**不使用 rollout 与原 X 的人为配对作为 posterior witness。** 不增加 seed、图像、时间点或另选弱层。

按 id、label 和原始 source_row 精确联结 `heldout_clean_c_current_fp32/clean_rank00.npz`。该 bank 的名字指编码器计算精度，其 latent 存储是 FP16；本轮的 teacher 目标是这份量化 clean bank，而非未量化真实分布。step000 的 teacher 状态提供原 ε，不重新抽样；验证其与每个 teacher 状态及 X 的原 FP32 bridge 构造一致，若仅有运算顺序的舍入差应明确量化，而不默默替换输入。

重新计算同一个 Full/Base/Weak helper，`capture=False`，只计算这 10 个 B8 teacher 输入；FP32 常驻权重、BF16 CUDA autocast、TF32off，EMA100080。两个弱 decoder 的 post-RoPE JQ 结构完全不变。每个输入的 FP32 `F−W` 和 `F−B` 必须与 v2 保存的对应 teacher 向量逐位一致；同时核对原有单头 norm。若不通过，保留失败记录并调查数值实现，不调干预公式或删除数据。

主风险恒等式的算术是先把保存的 F/B/W 升格为 FP64，再相减并归约；它与为 v2 parity 保留的 FP32 差分别记录。逐行量化 FP32 差相对该 FP64 差的舍入误差，不将二者混用后把恒等式残差解释为模型机制。

本轮不为此重复额外原生 reference forward；完整原生 parity 已在 v2 对所有状态通过。保存本轮 F/B/W 与联结的 X、全部行身份，供独立 CPU 重建风险与交叉量。不要只保留 oracle 系数或均值而删除符号与原始尺度。

## 记录与停止边界

保存逐图 `R_F,R_W,R_B,C_W,Q_W,C_B,Q_B`、两项 `R_weak−R_F−2C−Q` 恒等式残差及全部时间的 8 图平均/范围。对 Q=0 明确给零标记并将 C/Q 标为缺失，不删行。若报告跨时间汇总，明确它只对应这 10 个固定时间的等权描述量，不代表训练时间权重或轨迹积分。

固定协议、源码、模型、clean bank、v2 差向量与全部状态 SHA 后再运行。预期实际调用为 10 个共享 encoder（280 个 encoder block）、20 个 Full decoder block、20 个 weak decoder block、20 次 Full readout（含 weak）、10 次 Base readout；记录实际事件与预期的核对。模型加载、共享原分支、额外弱两层、CPU 读取/统计、传输、I/O 分开记录；新的复算全部列入研究成本。

这次检查只回答 JQ 所产生的差是否支持“同类条件均值误差被弱支放大”这一具体解释。即使 C>0，也不因此套用 C/Q 或启动图像实验；即使 C<0，也不否定全部结构 guidance，只是否定从当前证据直接认领该误差修复机制。下一步设计仍须交代生成偏置与质量的联系及公平成本，保持原来的至少 5% 目标。
