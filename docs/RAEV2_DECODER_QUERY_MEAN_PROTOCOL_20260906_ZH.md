# RAEv2：固定 DDT 解码器 query 均值弱分支的机制诊断

日期：2026-09-06。状态：实现与 CPU 检查准备中，尚未运行本诊断 GPU。本文件冻结一个有界结构检查；它不定义新的 guidance 强度，也不准入图像质量实验。来源为 [PAG/SEG 正文、附录和官方实现的阅读](RAEV2_GUIDANCE_READING_ATTENTION_WEAK_20260906_ZH.md)，不是已结束的空间能量球的改参版本。

## 结构与保证对象

固定修改 RAEv2 **全部两个 DDT decoder blocks**，不搜索层号。对每个样本和 attention head，将经过原始 q RMSNorm 和 RoPE 后的 query 矩阵 Q 替换为

\[
JQ,\qquad J=\frac1n\mathbf1\mathbf1^T,\quad n=256.
\]

之后不再归一化或匹配范数。K、V 的计算、scaled dot product attention、输出投影、残差、条件调制和 MLP 保持各分支的原有计算。完整 28 层 encoder 和原 Full 读出共享权重；第二个弱 decoder block 必须从第一个弱 block 的实际输出重新计算 Q/K/V，不能借用 Full 第二层的缓存。

JQ 是 Q 到所有行相同的矩阵集合的唯一 Frobenius 度量投影。在当前 decoder 没有 attention mask 或额外位置 bias 的条件下，令每行原 logits 为 ℓᵢ、pᵢ=softmax(ℓᵢ)，替换后的每行 attention 为

\[
r=\mathrm{softmax}\left(\frac1n\sum_i\ell_i\right)
=\arg\min_{r\in\Delta}\frac1n\sum_i\mathrm{KL}(r\Vert p_i).
\]

这是反向 KL 重心，即归一化的几何平均；正向 KL 重心才是 pᵢ 的算术平均。它删除位置特有的 query 检索，保留 key 的非均匀选择及 value 内容。残差、条件调制与逐 token MLP 仍可保留空间结构，因此不能将弱输出称作空间常量图。

**保证只针对该局部注意力操作。** 它不保证完整网络的能量曲率下降、弱误差与 Full 误差兼容，或生成分布/FID 改善。SEG 的有限 σ、原文 gain 与手工层组合均不迁移。本轮没有选择“替换 Base”还是“在现有 IG 上叠加”的采样公式；未经机制证据和成本分析，不因这个弱分支可运行就启动 1K。

## 固定输入与执行范围

使用现有 `normal_noise_audit_seed202609071/states/` 的 **10 个 snapshot**：step `000,047,067,077,084,089,092,095,097,099`。每个 snapshot 的 teacher 和 rollout 两个域、各 8 个 ID 全部保留，共 20 个 B8 输入、160 条图像—时间—域记录。t=1 的两个域有重复输入；这些记录不独立，也不通过图像或时间筛选去改善汇总。

缓存仅提供 z、t、ID、label。其历史 rollout 使用 FP32 合成 guidance、TF32on，不能称作当前生产轨迹；现有缓存 F/B 不作为位级真值。本轮重新在所有固定状态上计算当前 strict DINOv3-L K7、EMA step100080 的 Full/Base，模型 FP32 常驻、BF16 autocast、TF32off、B8。

每个输入只做：一次显式共享 encoder 的 Full/Base/弱 decoder 分支；另一次原始 `model.forward` 用于 Full/Base dtype、shape 和逐位 parity。显式 Full/Base 必须通过全部输入的原始 forward 对照。若不通过，保留失败结果并修复实现，不按响应改动 JQ 公式或挑输入。

两层弱 decoder 的局部 Q 使用 post-RoPE 均值，没有 sigma、强度、活动时间或阈值候选。该检查不生成新噪声，不读取 clean X，不拟合残差或尺度，不调用 stage1 图像解码器，不积分采样轨迹，不计算 FID。

## 保存的证据

全部记录 Full−Weak、Full−Base 向量及逐图范数、范数比、两方向余弦、相对原 gap 的平行/正交能量。平行投影系数仅是描述统计，不会拿去作为部署增益。零向量按明确缺失/零值规则处理，不删除对应记录。

保存每个 block、分支、图像和 head 的 query 行一致性、实际 SDPA 输出行一致性，以及由记录的有效 Q/K 重算的显式 FP64 attention 统计。非均匀 key 分布以 KL 到 uniform keys 等量描述。该显式 softmax 是概率参照，不能声称直接读取到 fused SDPA 的内部概率；query 求均值、转换到 BF16 的舍入会造成理论重心与实际输入之间的小差异，必须报告，不能靠重新归一化掩盖。

完整两层结构检查还需区分：首个弱 block 的 K/V 应与 Full 首个 block 相同；第二个弱 block 的 K/V 由其自身输入计算，不要求与 Full 第二层相同。没有全局 monkeypatch，原模型代码不改写。

## 成本与后续判断

先冻结协议、实现、模型配置/权重及所有输入文件 SHA，再执行单 GPU 检查。记录模型加载、共享 encoder、Full decoder/readout、Base readout、弱 decoder/readout、原始 forward parity、张量传输、CPU attention 统计和输出 I/O。必要的 native parity 属于研究验证成本，不伪装成部署时必须重复的模型调用；弱分支的额外两层重放也不因共享 encoder 被称作免费。此小样本分段计时不能代替未来实际采样吞吐测量。

该诊断要回答：是否实现了预期的信息删除；弱分支在本模型中是否产生非零且不同于原 gap 的结构响应；成本主要落在哪里。若仅有位级或近零扰动，不用范数放大制造信号；若出现额外方向，也不能据此认定它就是质量纠偏。下一步仍须明确误差机制、采样公式和公平成本条件后才能开展质量实验。当前至少 5% 的最终目标保持不变。
