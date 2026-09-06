# RAEv2 guidance 阅读：PAG、SEG 与有结构的弱分支

日期：2026-09-06。**两篇论文最有价值的贡献是决定“删掉什么信息”，而不是证明任意弱分支都能纠错。** PAG 删除跨 token 的关联检索、保留 value；SEG 删除 query 的空间变化、保留 key/value。后者的无限平滑极限还有一个可独立证明的解释：每个 head 的注意力行被替换为反向 KL 重心。它给出了无需选择 blur 半径的结构，但没有给出 RAEv2 的正确外推系数或 FID 保证。本轮没有模型运行、GPU、训练或质量评估。

## 版本、全文与实现

- PAG：Ahn et al., **Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance，ECCV 2024**。[会议正文](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09184.pdf) 17 页，[会议补充材料](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09184-supp.pdf) 38 页。补充 PDF 大部分是栅格页面，附录文字以[作者 arXiv v2 全文](https://arxiv.org/html/2403.17377v2)核读 A–F，另查看会议补充材料的 E.1、Figures 41–42 页面确认。v2 于 2025-07-07 更新，作者注明对应 ECCV camera-ready，并有 G 节 changelog；没有把该更新时间当作会议年份。
- SEG：Susung Hong, **Smoothed Energy Guidance: Guiding Diffusion Models with Reduced Energy Curvature of Attention，NeurIPS 2024**。使用[会议正式 PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/7b3f7b6670fdab2933411b5b922cdcc3-Paper-Conference.pdf) 30 页，读正文及附录 A–D；[会议补充 ZIP](https://proceedings.neurips.cc/paper_files/paper/2024/file/7b3f7b6670fdab2933411b5b922cdcc3-Supplemental-Conference.zip)包含一个 attention processor 源文件。arXiv 对应 `2408.00760v2`，2024-10-01；本笔记的公式编号以会议稿为准。
- 官方源码固定到 [PAG `ba1c0c9`](https://github.com/cvlab-kaist/Perturbed-Attention-Guidance/tree/ba1c0c9d246bc88e235c3c2d7bb199a1b17465bd) 与 [SEG `cf8256d`](https://github.com/SusungHong/SEG-SDXL/tree/cf8256d640d5373541cfea3b3b6caf93272cf986)。只做静态读取，没有导入作者 pipeline。

完整材料及 SHA 在 [reading_attention_weak_v1](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/reading_attention_weak_v1/manifest.json)。PAG 主文 / 补充 PDF SHA 分别为 `f4e4c7e26112f77ca2c2b981a790fde080d12029e1af4669399308bc40814e1a` / `47f34f61d5d891eb27bae08c7428ad0790e5308fa50bbdd8490e7ed7c19adcf6`；SEG 会议 PDF 为 `e547d93c503d5ecb47337a17fe30cbc32404a28b0f30a80cfd09be4842de1939`。

## PAG：选择性移除关联检索

写一个 attention head 为

\[
A=\operatorname{softmax}(QK^\top/\sqrt d),\qquad O=AV.
\]

PAG 的扰动是 **softmax 后的概率矩阵** `A→I`，所以扰动输出为 `V`；不是把 softmax 前 logits 换成 I，也不是跳过整个 residual attention block。图像位置仍得到自己位置的 learned value，但不能通过此层检索其他位置的内容。作者用“结构 / 外观”概括 QK 关系与 V 内容的不同作用，并用 Hopfield 的关联记忆解释这一选择。[PAG §4.2、E.1](https://arxiv.org/html/2403.17377v2#Pt0.A5.SS1)

这条直觉的实质是让正负分支保持可比较：仅删除一种关系信息，弱预测更容易出现与原预测相关的结构缺失；两者相减可能突出缺失器官、错误连接等，而不是减去完全不同场景。Figure 42 的对照有价值：直接平均 V 或跳过 attention 会改变图像内容；只删 attention 关系保留了更多原场景。Figure 41 中单独训练的条件/无条件模型相减效果差，也支持“同一模型内的差分兼容性”值得关注。这些是具体机制及案例证据，不能升级为所有 V 都只编码外观、所有扰动都保持 in-domain 的定理。

主文 Eq.10 为

\[
\epsilon_{\rm PAG}=\epsilon_F+s(\epsilon_F-\epsilon_W).
\]

作者用假想 desirable/undesirable 标签给差分一个 density-ratio discriminator 解释。若两个场确为相应密度的 score，该代数合法；事后修改 attention 没有被证明就是某个加噪弱分布的精确 score，也没有终点密度或质量保证。Hopfield 内层检索的性质不能直接跨越残差网络、读出、采样器而成为这个保证。[PAG §4.1](https://arxiv.org/html/2403.17377v2#S4.SS1)

作者确实得到较强实证：ADM ImageNet 256²、50K、DDPM250，unconditional FID `26.21→16.23`，conditional `10.94→6.32`；SD1.5 COCO30K 的 CFG `15.00`，PAG `10.08`，二者结合 `8.73`。但后者对照的 CFG 系数不同，组合增加第三条预测，不是对强基线的同成本 41.8% 收益。ADM 量化使用 `s=1`，部分展示使用 `s=3`；SD 单独 PAG 与组合又采用其他值。D.3 按 FID 排序后搜索层及层组合，没有一个通用定理选出这些层。[主表与实验附录](https://arxiv.org/html/2403.17377v2#Pt0.A1)

源码还须分清系数约定：所读 ADM `gaussian_diffusion.py` 实际使用 `W+guide_scale*(F−W)`，故相对于主文 `s`，`guide_scale=s+1`。它有可选 guidance schedule；不能把可选接口误记为正文方法必须调时窗，也不能原样接进本项目。[固定源码](https://github.com/cvlab-kaist/Perturbed-Attention-Guidance/blob/ba1c0c9d246bc88e235c3c2d7bb199a1b17465bd/guided_diffusion/gaussian_diffusion.py#L625)

## SEG：先保留正确的算子机制，再审视能量解释

SEG 操作的是 **softmax 前 logits**：令 B 为沿 query 空间位置的线性平滑算子，

\[
B(QK^\top)=(BQ)K^\top.
\]

这条 Proposition 3.1 的矩阵恒等式准确。它避免显式构造、平滑 N×N logits，却仍要计算原预测和弱预测。它不等于 `B softmax(QKᵀ)`；也不能把 PAG 附录里对 softmax 后 attention 图的 Gaussian blur 当成 SEG 的同一算子。弱分支保持对 key/value 的访问，只减少不同 query 位置各自提出不同检索请求的能力。[SEG §3.3](https://proceedings.neurips.cc/paper_files/paper/2024/file/7b3f7b6670fdab2933411b5b922cdcc3-Paper-Conference.pdf)

作者进一步提出 `E(a)=−logsumexp(a)` 的 attention 能量，试图证明平滑减少曲率，并把原预测与“更钝”的预测相减，类比 CFG。这个类比给出了有用设计方向，但其普适曲率保证不能直接继承：

- 在保均值的双随机平滑下，Jensen 给出 `lse(Ba)≤lse(a)`，所以 **负** lse 能量上升。主文 Lemma 3.2 的 lse 方向文字有误；附录 A.2 的展开支持上述正确方向。
- 完整 Hessian 是 `H(a)=ppᵀ−diag(p)`，始终有 `H1=0`，完整行列式为零。A.3 忽略非对角项后比较行列式，丢掉了这项结构；Eq.22 根据“模糊后分母变小”写出的不等号也反向。对复合标量能量 `E(Ba)`，正确链式 Hessian 是 `BᵀH(Ba)B`，不是逐元素乘 `b_ij`。因此这里没有成立的一般严格曲率下降定理，更没有 latent score Jacobian 或 FID 的下降定理。

这些问题不否定 SEG 代码产生的图像，也不否定 query 平滑的线性代数。它们要求改用能够准确描述干预的信息结构，见下一节。官方有限核实现还使用 reflect padding；这种边界不保证精确保全局均值。`σ→∞` 的直接 query 均值分支则没有这个问题。[官方实现](https://github.com/SusungHong/SEG-SDXL/blob/cf8256d640d5373541cfea3b3b6caf93272cf986/pipeline_seg.py#L70)

SEG 的量化结果同样应保留：SDXL 空提示 FID `129.496→88.215`，文本条件但无 CFG 的基线 `53.423→26.169`，对应无限 query 平滑；相对同组 PAG 空提示 `105.271` 也有明显改进。其 LPIPS 是到未引导图像的距离，说明改动程度，不能本身证明“副作用更少”。Table 2 主要不是强 CFG 基线，SAG 对照还使用不同 scheduler；论文未提供本任务要求的公平总成本 RAEv2 结果。[SEG §5、Tables 1–2](https://proceedings.neurips.cc/paper_files/paper/2024/file/7b3f7b6670fdab2933411b5b922cdcc3-Paper-Conference.pdf)

“固定 scale”是实验选择：§5.5 比较 `1/3/5`，又扫描 σ，并选 `10/∞` 做主比较。更有一个必须保留的约定差异：Eq.9 写 `γF−(γ−1)W`，代码却写 `F+seg_scale*(F−W)`；因此两者数值相等时，实际外推幅度相差 1。默认代码 `seg_scale=3` 对应公式 `γ=4`，不能默默把它们当作相同的“3”。本文不推断作者所有表格究竟采用哪个约定，也不把固定 3 当成理论常数。[代码 guidance](https://github.com/SusungHong/SEG-SDXL/blob/cf8256d640d5373541cfea3b3b6caf93272cf986/pipeline_seg.py#L1474)

## 一个严格的正向解释：query 投影与反向 KL 重心

本节是本次研究讨论提出、独立核对的推导，不是 SEG 原文已经证明的结论。固定一个样本、一个 head，令

\[
J=\frac1N\mathbf1\mathbf1^\top,\quad
\ell_i=Q_iK^\top/\sqrt d,\quad p_i=\operatorname{softmax}(\ell_i).
\]

`JQ` 是 Q 到所有行相同的线性子空间的唯一 Frobenius 最小距离投影；它保留平均 query，删除所有位置间 query 差异。其注意力每一行都等于

\[
r=\operatorname{softmax}\!\left(\frac1N\sum_i\ell_i\right)
=\frac{\exp(\frac1N\sum_i\log p_i)}{\sum_j\exp(\frac1N\sum_i\log p_{ij})}.
\]

并且

\[
\boxed{r=\arg\min_{v\in\Delta}\frac1N\sum_i\mathrm{KL}(v\Vert p_i).}
\]

证明只需展开：目标为 `Σv log v−Σv mean(log p_i)`；更强地，任意 v 的目标值与 r 的目标值之差正好为 `KL(v||r)`。有限 logits 下 r 正且唯一。反方向 `mean_i KL(p_i||v)` 的最小点是算术平均 `mean_i p_i`，一般不同。

因此这个弱算子保留各位置共同支持的 key，移除位置特有的路由分歧。**相同行不等于 key 上的均匀概率**：归一化几何平均仍可偏好特定 key。它也不是输出特征 `AV` 的投影、图像分布的 KL 投影或无条件 score；下游非线性不会自动保留上述最优性。这是无需 σ 的明确干预定义，而不是质量结论。

## RAEv2 迁移与具体证伪入口

已核查 [DDT.py](../external/RAEv2/src/stage2/models/DDT.py) 与 [NormAttention](../external/RAEv2/src/stage2/models/model_utils.py)：当前为 28 个 encoder blocks、两个 DDT decoder blocks，Base 在第 8 层读出。encoder 序列还含 4 个 time tokens 与 8 个 class tokens；不能把全部 268 token 随意当作空间 query 做均值。两个 decoder blocks 只有 256 个空间 token，条件从完整 encoder 输出经逐位置调制注入。

| 本任务要求 | 对两篇方法的判断 |
|---|---|
| 同类共享 Full/Base | 同模型扰动与之兼容，但原生 Base 不是 attention 扰动模型；新 W 应保留 Full 的读出，避免重新做跨深度 head-swap。 |
| 不调大量系数/窗口 | PAG 的 I、SEG 极限的 J 是明确结构；层选择和外推 scale 不由论文推导。不能搬用论文扫描。J 可删除 σ 这个自由度。 |
| 理论导出机制 | 信息删除与 KL 重心成立；错误兼容性、外推符号及系数尚缺，不能用曲率语言代替。 |
| 公平成本 | PAG/SEG 常规路径需第二预测，叠加 CFG 则三预测。RAE 原生 F/B 共享 encoder，不能套用“成本与 CFG 相同”。PAG 附录 3090 单 batch 速度为未引导 19.16、CFG 12.67、PAG 12.68 iter/s；它是两路 CFG 的比较，不是免费扰动。 |

**具体候选：在固定的整个两层 DDT decoder 模块内构造 JQ 弱分支。** 每个 attention 先按原代码完成 `q_norm/k_norm` 与 RoPE，再对该 head 的全部 256 个 query 求均值；K/V 在该次 attention 内不直接修改。保留完整 28 层 encoder、条件调制、MLP、残差与相同 Full final layer。NormAttention 在这里没有额外位置 bias，两个 decoder 调用也没有 attention mask，因此上述 logit 平均与反向 KL 重心正好对应。均值之后不能再归一化 query 或匹配幅度，否则已改变该机制。

“保留 K/V”是局部算子说明：第一层弱输出改变第二层输入后，第二层 K/V 必须在弱分支重新计算，不能复用 Full 的第二层激活。可共享原 encoder 输出及分叉前状态，新增成本仍包括两个 decoder blocks 和 final readout。选择整个 decoder 模块是固定架构边界下的一种候选，**不是论文证明的最佳位置**，不继续逐层搜索。

此时 `d=F−W_J` 衡量的是移除位置路由后的完整输出响应。若 `d` 主要只是 `F−B` 的倍数，新增机制在输出层面可能仍退化为 IG scale 改动；若 d 有独立方向，也不自动有益。一个可证伪的“放大已有误差”假说是

\[
F=m+e,\qquad W_J=m+\kappa e+\xi,\quad\kappa>1,
\]

其中 ξ 是未对齐的扰动误差。只有 κ 稳定、ξ 足够小，才可能从错误模型推出类似 `1/(κ−1)` 的抵消系数；两篇论文没有证明这些条件，也没有给出当前 RAE 的 κ。

证据入口应固定输入、时间、类标签，同时比较 F/B/W 的方向、能量和与真实 paired residual 的关系；真实 bridge 与 actual rollout 分开记录。已有 bridge 输入若带 clean X，可检查

\[
R_F=E\langle X-F,d\rangle,\quad
R_G=E\langle X-G,d\rangle
=R_F-E\langle G-F,d\rangle.
\]

`R_F≤0` 会直接反对“正向小幅外推纠正 Full 的均方误差”；`R_F>0` 而 `R_G≤0` 则提示该方向未必能加在现有 IG 上。不能把没有真实终点配对的 rollout 状态套入这条 bridge 风险解释。理论上单一方向的 MSE 最优系数是 `R_G/E||d||²`，但它可能撤销有用的 IG，**本笔记不据此拟合部署系数，更不把 MSE 最优当成 FID 准入**。需要先看到稳定的错误结构，再解决系数与实际生成质量，不能因弱图像“更差”便开始外推。

当前可接受的是这个固定信息删除算子的机制诊断；尚不接受已经得到无需调参的 RAEv2 guidance。没有从历史 DC/AC 能量偏差推导它，也没有把它当作继续调整能量球的理由。

## 归档与计算边界

`algebra_check.py` / `algebra_results.json` 用固定小矩阵验证 query/logit 等价、反向 KL 重心、投影恒等式，以及曲率与 padding 的边界；不是生成实验。重心恒等式残差 `5.55e−17`、query 投影勾股残差 `8.88e−16`。同一算例中注意力各行相同但 key 概率不是均匀值。

这次 CPU 计算 wall `0.069992 s`、CPU `3.739006 s`，包含库导入、截至计算结束，不含最后序列化/退出。网络请求逐项记录 wall，不能把并发请求耗时简单相加当总墙钟；阅读、网页浏览、PDF 选页渲染及部分文本转换没有完整精确成本。GPU / 模型 / 训练 / 生成 / FID 调用均为零。GitHub API 曾返回 rate-limit 403，改用公开 `git ls-remote` 固定提交；失败响应保留，没有安装依赖或运行下载的代码。
