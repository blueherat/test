# Query 对比的局部几何：两个不同的错误假设

2026-09-06。以下是阅读 PAG / SEG / CPC 后的独立代数推导，经独立审阅核对。没有实现新 attention、选择权重、调用模型或开展质量实验；它不改变已冻结的 JQ 与 paired-error 检验。

固定某一层的 K，且没有额外 mask/bias，记 `ℓ_i=Q_iKᵀ/√d`、`m=mean_i ℓ_i`、`P_i=softmax(ℓ_i)`、`r=softmax(m)`。r 是原 attention 行的反向 KL 重心，通常并非算术 key marginal。

**中心化 query：Q→(I−J)Q。** 新 attention 为 `C_i=softmax(ℓ_i−m)=Normalize_keys(P_i/r)`，其反向 KL 重心严格为 uniform。跨 query 的交叉 log-odds

`log(P_ij/P_ik)−log(P_hj/P_hk)`

保持不变；每一行的 odds、行间 KL 或互信息不因此保持或单调改善。因此它删除共同 key 偏好，并没有放大上述交叉对比。因为 r 不是算术 marginal，不能自动把 P_i/r 称为 PMI 或正确 Bayes prior removal。

一个精确但有条件的误差机制是：若观测 logits 为 `ℓ_ij=ℓ*_ij+b_j+a_i`，即多出 query 共享的 key 偏置 b，中心化对 b 完全不变，并恢复目标的中心化 logits。只有目标本身的共同 key 偏好应为 uniform，或研究目标就是那个中心化分量时，才能进一步说恢复了目标 attention。若原共同偏好包含物体、背景或全局上下文信息，这个操作会一并删除信号。当前没有 RAE 满足该偏差假说的证据。

**保留重心的外推：Q(w)=JQ+w(I−J)Q。** 这给出另一族：

`P_i(w)=Normalize_keys(r^(1−w) P_i^w)`。

它的反向 KL 重心仍为 r，而交叉 log-odds 乘以 w。其变分表述为

`argmax_p {w E_p log(P_i/r)−KL(p||r)}`。

该表达式给出指数倾斜的对象，但不确定 w。若真实错误是 query 特有信号被统一缩小，即 `ℓ=m*+a e*`、`mean e*=0`，则只有 `w=1/a` 才恢复它；目前没有 `a=1/2` 或任何具体 a 的证据，不能仅凭对比几何指定 w=2。

固定同层 V 的值响应还有准确切线

`∂_w[P_i(w)V]|_(w=1)=Cov_(P_i)(V, log(P_i/r))`。

它选择与相对检索证据相关的 value 方向，是否纠正 Full 的错误仍要检查目标误差及后续网络响应。两层同时修改时，第二层 K/V 也会改变；不能把局部指数族恒等式升级为整个 Full 输出的指数族公式。

已读 PAG / SEG / CPC 中未发现明确的 `(I−J)Q` guidance 配方：SEG 的无限平滑是 JQ，CPC 比较数据/后验协方差。这里只留下两个可区分的假设——多余的公共 key 偏置，或被削弱的 query 特有信号——不宣称全局新颖性、无条件误差下降或 FID 改善。当前运行的 JQ weak 分支检验的是另一个完整网络响应，不能拿它直接充当 centering 的实验结果。

相关记录：[PAG / SEG](RAEV2_GUIDANCE_READING_ATTENTION_WEAK_20260906_ZH.md)、[CPC](RAEV2_GUIDANCE_READING_LINEAR_CPC_20260906_ZH.md)、[JQ 结构结果](RAEV2_DECODER_QUERY_MEAN_RESULTS_20260906_ZH.md)、[配对误差协议](RAEV2_QUERY_MEAN_ERROR_COMPATIBILITY_PROTOCOL_20260906_ZH.md)。
