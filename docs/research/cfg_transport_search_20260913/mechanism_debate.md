# 用真实误差检验 conditional 运输是否值得额外计算

本轮提出并完成了一个真实机制诊断：**把有限条件逆流产生的额外方向，用真实图像的 FM 监督残差校准，再考虑与 APG 组合。结果尚未通过独立监督门槛，暂不推进 5K 图像实验。** 不先假定高 CFG 的未来更好，也不通过减小 mixed-loop 位移给模型打分。另一个考虑过的方案——按条件模型密度接受或拒绝未来——存在直接反例，同样不投入图像实验。

这是一个待证伪的、小规模监督校准方案，不是已经成立的质量改进，也不是免训练新 guidance。其研究价值必须来自一个具体发现：有限运输提供了便宜 secant/APG 无法解释、且在独立真实图像上能修正主模型误差的方向。

## 已有证据划定的边界

[旧 lifting 宽强度结果](../../LIFTING_WIDE_SCALE_RESULTS_20260909_ZH.md)在 SiT-XL、RAEv2 的已观察最佳 FID 上没有优势，计算约为普通 IG 的 3.5 倍。[FSG Jacobian 审计](../../SIT_FSG_JACOBIAN_AUDIT_RESULTS_20260911_ZH.md)证明精细逆流能准确实现指定未来，但目标换成算法自己的 Euler 终点时，方法排名会变；latent 转移误差不能替代质量。

[主报告](../../CFG_INVERSION_RESEARCH_20260913_ZH.md)已排除了两种表面新方法：最后用同区间条件流推进，则条件逆腿精确抵消；最后仍用强场推进，则三腿 reflection 首阶只是更大的有效 CFG。有限步差异仍可能有价值，但必须超过充分调参、计算可比的 APG、CTRL 和旧 secant 对照。

[旧 APG #55–58](../../APG_MECHANISM_AND_FIVE_EXTENSIONS_20260911_ZH.md)已经包含未来 moment 投影、语义保护、真实多步前瞻选择和不同前瞻长度的共同终点验收。因此“换成 conditional 后缀后再做未来投影”不足以构成新方向。

近期原始论文也缩小了空间。APG 已将条件差分解并抑制相对条件预测的平行分量；TAG 在 noisy state 的径向/切向坐标中放大切向更新；FDG 分开低频与高频指导；ERK-Guid 利用嵌入求解器响应提供额外方向。它们分别构成当前投影、径向约束、频带重权和数值响应的近例，不能换一个几何名称重新计算贡献。[^apg][^tag][^fdg][^erk]

## 唯一保留的方向：监督校准运输的额外分量

时间仍采用 \(t=0\) 噪声、\(t=1\) 数据。设冻结模型

\[
A(z,t)=v_c(z,t),\quad U(z,t)=v_u(z,t),\quad g=A-U,
\quad B=A+\gamma g.
\]

在固定区间 \([t,t+H]\) 构造

\[
d=\Phi_A^{t\leftarrow t+H}\Phi_B^{t+H\leftarrow t}(z)-z.
\]

有限精度实际采用带同场对照的差

\[
\widehat d=\widehat I_A(\widehat F_B(z))
-\widehat I_A(\widehat F_A(z)).
\]

它只消除了公共往返偏差，不等于把数值逆变成精确逆。改变子步数后还须检查方向与监督效应是否稳定。

先建立同一点上的已知方向矩阵

\[
S=[g,\;p_{\rm APG},\;s_{\rm Euler},\;s_{\rm embed}],
\]

其中 \(p_{\rm APG}\) 是无历史 APG 投影方向。\(s_{\rm Euler}\) 是强/条件 Euler 未来的条件场响应差；\(s_{\rm embed}\) 是仓库历史胜出方法所用的 condition embedding scale=.5 割线。这两个 secant 不是同一个对象；后者额外需要一次模型查询。对四个方向逐样本正交化，提取

\[
q_0=(I-P_S)\widehat d/H.
\]

这一步是识别额外信息的消融，不是宣称这些被去除的方向有害。若 \(q_0\) 太小或小于求解器差异，就停止；不要把数值尾部归一化成一个很大的新向量。

### 为什么真实监督能区别合法运输与模型错误

从真实训练图像取 VAE latent \(X\)，独立取 \(\epsilon\sim N(0,I)\)，构造

\[
Z_t=tX+(1-t)\epsilon,\qquad Y=X-\epsilon.
\]

当 \(q=q(Z_t,t,c)\) 不读取隐藏的 \(X,\epsilon\) 时，条件期望恒等式给出

\[
\mathbb E\langle q,Y-A\rangle
=\mathbb E\langle q,v_c^*-A\rangle,
\quad v_c^*=\mathbb E[Y\mid Z_t,t,c].
\]

因此对冻结的运输特征，平方风险差为

\[
L(\lambda)-L(0)=\lambda^2\mathbb E\|q\|^2
-2\lambda\mathbb E\langle q,Y-A\rangle.
\]

单个时间桶的无正则人口最优值是

\[
\boxed{\lambda^*=
\frac{\mathbb E\langle q,Y-A\rangle}{\mathbb E\|q\|^2}.}
\]

若 conditional 已是准确 oracle，分子严格为零；即便 mixed loop 明显不恒等，也不应写入任何额外修正。这正是区别“网络错了”和“不同场本来不同”的监督锚。此结论来自条件 FM 回归的基本性质，不是新的条件概率定理。后训练校准降低 score matching 损失已有明确先例。[^calibration]

实际只拟合三个时间桶，不训练主干。可将 \(q_0\) 用 calibration 集的全局 RMS 归一化，便于保存系数；这一常数缩放与系数乘积抵消，不允许按每个微小残差单独放大。所有归一化、ridge、截断规则须在 holdout 前固定。

最小执行场是

\[
\boxed{v_{\rm new}(z,t)=v_{\rm tuned\;APG}(z,t)+\lambda_{b(t)}q_0(z,t).}
\]

在投影精确且 APG 更新确在已投影子空间内时，\(\langle q,Y-A\rangle=\langle q,Y-v_{\rm APG}\rangle\)：这个校准系数不是通过缩小已有 APG 分量得到的。teacher 状态采用无历史 APG，不能直接代表带真实动量的 rollout；部署时仍要检查实际更新的内积。

这里额外项直接修正速度，随后按共同主网格推进；它不以“完整回灌后再 conditional 前进”实现，因而没有 \(CR=H\) 的整段抵消。APG 仍用冻结主模型自己的分支与历史；本轮不把校准项重新递归写进运输教师，避免移动训练目标。

### 最小实验及可直接否定的条件

1. **拟合与 holdout。** 从原始训练缓存按类别选两组不重叠图像；每类 2 张拟合、2 张 holdout，先在 \(t=.25,.5,.7\)、\(H=.125\)、\(\gamma=1\) 做诊断。各图像的多个时刻/噪声必须留在同一分组，不能按状态随机切分。这里的 holdout 不是 FID 使用的 ImageNet validation。
2. **解释量先行。** 保存 \(\|q_0\|/\|d/H\|)、与已知方向的内积、便宜 secant 的定义及查询数。一步 Euler 的同场去偏 mixed-loop 必在 (\operatorname{span}\(g,s_{\rm Euler}\)\) 内；若实现报告明显额外方向，先查数值或投影。
3. **独立监督判据。** 仅在 calibration 拟合 \(\lambda\)，holdout 报告配对 FM 风险差与按图像聚类的置信区间。噪声目标很嘈杂，应直接计算差 \(\lambda^2\|q\|^2-2\lambda\langle q,Y-A\rangle\)，而非相减两个很大的平均损失。三个时间桶分别报告，也报告所有桶的联合结果，避免只挑一个时间。
4. **数值判据。** 小子集将两个 Heun 子步加到四个；若剩余方向主要由求解器误差组成，或者监督作用反号，停止“精确运输修正模型误差”的解释。后续可另外研究离散适配器，但不能混用名称。
5. **图像阶段。** 只有独立监督信号出现，才比较 APG、最强历史 secant+APG、APG+校准额外方向、APG+反号方向、以及相同监督预算的便宜方向校准。至少比较 \(\lambda=0\) 和冻结拟合值；不再以 20 档系数扫描代替监督预测。
6. **计算。** 密集精细逆流不可接受。首轮只在预先固定的少数主步更新，额外速度只作用在这些步；或者先证明低成本预测器能复现有用方向，再考虑持有多步。将同一个向量直接持有八步会改变算法，不是免费实现精确运输。按实际单分支 NFE 和 GPU 秒，对照增加主求解器步数或使用既有 secant。

两子步 Heun 的强前瞻、条件前瞻及两条条件反演，复用起点评估后需 19 次单分支预测；Euler 状态 secant 需 2 次 conditional 查询，embedding 割线再需 1 次，共 22 次/teacher state。两子步 Euler 可以作为之后的廉价近似，但不能在监督信号未出现前默认它保留相同机制。

### 反方意见

FM 误差下降并不自动降低 FID。误差只在训练桥分布上被识别，而 guided rollout 访问的状态不同；条件均值更准也可能改变原先有效的 guidance 偏差。这个方向的成功标准仍是超过充分调参、计算可比的 APG/CTRL/已有 secant，而非只在 teacher 状态赢一个损失。

把完整 APG 场直接拟合到 FM target，会把合法的偏好引导一起撤掉。本方案只用原始 conditional 的残差校准额外方向，再保留已有 APG；这也不是严格保证，只使“我们想纠正谁的什么误差”可审查。

若便宜 secant 与相同 supervision 已达到相同风险/质量，有限 inverse 没有增量价值。若正确符号系数和反号系数在质量上没有稳定区别，监督锚不能解释收益。若只在一个种子或同一 FID bank 调参后有效，应停止宣传新机制。

## 未保留的第二方向：用 conditional 密度判定坏目标

标准 Gaussian FM 的准确场在 \(0<t<1\) 满足

\[
s_c(z,t)=\nabla\log p_t(z\mid c)=\frac{t v_c(z,t)-z}{1-t}.
\]

因此同一未来时间层上的两个候选 \(y_0,y_1\)，理论上可通过

\[
\log p_t(y_1\mid c)-\log p_t(y_0\mid c)
=\int_0^1\langle s_c(y_0+u(y_1-y_0),t),y_1-y_0\rangle du
\]

评估条件模型密度变化。但实际网络未必保守，路径积分未必是一个真实 log density 差；更根本的是，**即便 score 完全准确，逐样本密度上升仍不等于分布质量提升。** TAG 的一阶密度解释和几何更新是直接近例，不能继承比其假设更强的质量结论。[^tag]

反例：真实目标为 \(N\(0,I_6\))，把每个样本收缩为原来的一半。期望 log density 上升 2.25，但输出协方差变为 (0.25I\)，与目标的 (W_2^2=1.5\)。密度门控会赞成已经破坏自然变化的动作。条件密度下降也未必说明坏目标：较低密度的真实稀有模式可能正是需要保留的多样性。

所以这类条件 score 探针只值得作为分组诊断：它能识别“高 CFG 相对 conditional 模型做了什么”，不能单独决定接受/拒绝。识别坏目标最终还需独立于构造信号的条件正确性、真实类别内分布与多样性读出。近期对强引导的实证评估也明确指出，指导分类器的成功率会漏掉类内质量破坏；这里采用其评估提醒，不接受“某个预测参数化普遍保证不离流形”的无条件推广。[^evaluation]

## 已完成的 CPU 检查

使用六维相容联合 Gaussian：类别变量 \(M\) 是 Gaussian 均值，(X\mid M\) 为对角 Gaussian，目标条件 (M=0\)，两原始模型使用规范 Gaussian FM；再给它们加入同一个已知线性误差。精细矩阵流取 DOP853，独立加严容差检查。

calibration seed=173，每组 100000 张；独立 evaluation seed=811、812，各 100000 张。CPU 额外方向逐样本投掉 gap、无历史 APG 与便宜 Euler secant。该 Gaussian 模型没有真实网络的 condition embedding，因而这项 CPU 正例没有排除历史 embedding secant；真实 SiT calibrator 另外投掉它。没有拟合图像质量，这组 CPU 检查没有 GPU。

|设置|额外方向占完整运输 RMS|拟合系数 / 使用真实均值的系数|独立真实速度误差：原始→校准|
|---|---:|---:|---:|
|准确原始场|1.798%|0.004737 / **0**|0→0.00002241、0.00002247|
|已知共享线性误差|4.310%|0.369668 / 0.368278|0.435336→0.299114；0.437198→0.300412|

后一设置中反号修正把误差升至 0.846614、0.850312。前一设置却显示有限样本的陷阱：两个 holdout 的观测 FM 风险均有极小下降，但真实误差实际增加。因此置信区间、预设阈值和独立图像分组必要，不能看见任何负损失差就宣称有用。

一步 Euler 的额外残差最大约 \(1.91\times10^{-16}\)，符合其被便宜 secant 完全解释的代数。相对流矩阵加严容差后的变化不超过 \(4.04\times10^{-12}\)。这些检查只证明构造与监督识别逻辑，不证明 SiT 具有该误差结构。

复现：

```bash
OPENBLAS_NUM_THREADS=1 python experiments/cfg_transport_search_20260913/mechanism_cpu_audit.py
```

实现：[mechanism_cpu_audit.py](../../../experiments/cfg_transport_search_20260913/mechanism_cpu_audit.py)。原始结果：[checks.json](/home/zhoushunyu/data/eqvae/experiments/cfg_transport_search_20260913/mechanism_cpu/checks.json)。

## 真实 SiT 校准结果：尚未通过独立监督门槛

在完成上述 CPU 设计后，按明确安排使用空闲 GPU0 跑完冻结的真实诊断。没有改公共 runtime，没有生成新图像，也没有读取 FID validation latent。拟合和 holdout 各 200 张，均来自原始 ImageNet 训练缓存，每类各 2 张；400 个 source ID 全部唯一。latent 使用缓存 mean/std、固定 posterior noise 与训练相同的 0.18215 缩放。反方独立复核了输入重建、分组和特征未读取隐藏真值。

模型为 SiT-S/2、step 800000 EMA；所有模型字段 FP32、关闭 TF32，投影和损失统计 FP64。三个时刻为 .25/.5/.7，H=.125、γ=1；主特征每状态 22 次单分支预测，含 condition embedding=.5 查询。8 张固定图像另做 4/8 子步检查。总计 29136 次单图单分支等效预测，观测耗时约 40.6 秒。

|时间|冻结系数 λ|holdout 风险差 ΔMSE|按图像 bootstrap 95% 区间|λq / 无历史 APG 方向 RMS|该比值的 holdout 最大值|
|---|---:|---:|---:|---:|---:|
|.25|.165026|−3.3902e−5|[−1.1239e−4, +4.5363e−5]|5.976%|9.856%|
|.50|.216282|−1.4104e−5|[−6.2396e−5, +3.5929e−5]|6.844%|9.124%|
|.70|.000554|+5.6346e−8|[−5.0625e−9, +1.1618e−7]|0.01608%|0.01993%|

比值分母是未乘额外系数 α 的无历史 APG 方向；比较具体 α 的实际 guidance 应再除以 α。所有系数只在拟合集估计，未看 holdout 调整、截断或重拟合。联合 holdout 风险差为 −1.5983e−5，95% 区间 [−4.7564e−5, +1.4018e−5]，同样跨零。每个重采样单元保留同一原图的三个时刻，避免将相关状态当独立图像。

剔除四个已知方向后，q 仍占完整运输 RMS 的 44.29%、39.46%、34.23%。因此“有限运输有非平行分量”已经可观察，但它并没有因此获得质量机制证明。t=.25/.5 的平均风险降低只约为原风险的 0.0051%/0.0018%；当前样本不能确认它们是稳定信号。

数值检查中，2→4 子步的 q 相对变化中位数为 6.65%，4→8 子步为 1.94%；最小方向余弦分别为 .98316、.99546。使用相同冻结 λ 的 8 图子集，在 t=.25 的平均风险差从 2 步的 +1.2649e−4 变为 4 步的 +1.1638e−4、8 步的 +1.0771e−4，始终变差；t=.5 三个分辨率均为小幅改善。该子集只有 2 张拟合图、6 张 holdout，不能替代全体独立风险结论，也没有通过分辨率选择方法的正当性。

q 对无历史 APG 方向正交，使 conditional 残差与无历史 APG 残差得到的配对风险差最多相差 2.2e−19；这只核对该静态方向，不证明对 rollout 中真实 momentum/cap 历史同样正交。是否能改善 guided rollout 仍是未回答问题。

**当前裁决：不将该候选直接扩到 5K 图像质量验证。** 这不是证明运输永远无效，而是明确的有界负结果：已找到独立方向，尚未证明它修正真实主模型误差，额外 22 次查询目前缺少支付依据。没有为得到显著性继续扩大数据量或搜索更多系数。

私有实现：[transport_calibration.py](../../../experiments/cfg_transport_search_20260913/transport_calibration.py)。可核查结果：[输入与协议](</home/zhoushunyu/data/eqvae/experiments/cfg_transport_search_20260913/transport_calibration/request.json>)、[冻结系数及完整分位数](</home/zhoushunyu/data/eqvae/experiments/cfg_transport_search_20260913/transport_calibration/calibration_summary.json>)、[细分后的监督作用](</home/zhoushunyu/data/eqvae/experiments/cfg_transport_search_20260913/transport_calibration/solver_risk_audit.json>)、[观测与成本](</home/zhoushunyu/data/eqvae/experiments/cfg_transport_search_20260913/transport_calibration/observation_manifest.json>)。全部 q 字段与逐图充分统计也保存在该输出目录。

复现使用新的输出目录，避免覆盖已冻结数据：

~~~bash
python -m experiments.cfg_transport_search_20260913.transport_calibration prepare --output /tmp/transport_calibration_replay
CUDA_VISIBLE_DEVICES=0 /home/zhoushunyu/miniconda3/envs/myenv/bin/python -m experiments.cfg_transport_search_20260913.transport_calibration observe --output /tmp/transport_calibration_replay --device cuda:0 --batch 8 --save-fields
python -m experiments.cfg_transport_search_20260913.transport_calibration fit --output /tmp/transport_calibration_replay
~~~

## 原始来源

[^apg]: Sadat, Hilliges, Weber. [Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models](https://proceedings.iclr.cc/paper_files/paper/2025/file/d1450d6c10c6b6cf1b80964357f5fa08-Paper-Conference.pdf). ICLR 2025。条件预测的平行/正交分解、重标度与动量。
[^tag]: Cho et al. [TAG: Tangential Amplifying Guidance for Hallucination-Resistant Sampling](https://arxiv.org/html/2510.04533v2). ICML 2026；v2，2026-05-26，§4与Appendix E。noisy state 径向/切向投影，一阶 log-density gain；该局部结论不等于目标分布正确性。
[^fdg]: Sadat et al. [Guidance in the Frequency Domain Enables High-Fidelity Sampling at Low CFG Scales](https://arxiv.org/html/2506.19713). 2025 预印本及后续版本。低/高频分离强度；不把普通多频段缩放计作本轮新贡献。
[^erk]: Kong et al. [Error as Signal: Stiffness-Aware Diffusion Sampling via Embedded Runge-Kutta Guidance](https://arxiv.org/html/2603.03692v2). ICLR 2026；v2，2026-04-19，§4。嵌入求解器差异、刚性代理及额外方向。
[^calibration]: [On Calibrating Diffusion Probabilistic Models](https://papers.neurips.cc/paper_files/paper/2023/file/9a645c38d4ec6f94633a35aeb2079596-Paper-Conference.pdf). NeurIPS 2023。后训练校准与 score matching 风险降低，是监督校准的近例；本提案的差异只在候选方向及相对既有 guidance 的增量识别。
[^evaluation]: Lee and Lee. [Not All Prediction Targets Keep Training-Free Diffusion Guidance on the Manifold](https://arxiv.org/html/2607.00647v1). 2026-07-01 预印本，§4–5、Appendix C–D。借鉴独立指导/评价分类器、类内质量与强度曲线的区分；不据跨架构比较宣称通用因果定理。
