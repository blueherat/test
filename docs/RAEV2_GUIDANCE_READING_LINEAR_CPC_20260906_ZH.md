# RAEv2 阅读：线性 CFG 的均值与对比主成分机制

日期：2026-09-06。本文是论文阅读和小型 CPU 代数核验，没有运行模型、GPU、训练或质量实验；不修改已有冻结协议。

**可保留的设计直觉：guidance 应辨认相对于弱参照缺失的结构，而不是把高方差、大范数或全部主成分当作有用方向。** 这篇论文在线性条件/无条件模型中给出精确的均值与协方差对比分解，并用真实模型实验验证其中部分解释。它没有推出唯一 guidance 强度，也没有证明同类 Full/Base 的差就是条件/无条件的 CPC。现阶段适合作为机制判断依据，不能直接准入新的 RAEv2 sampler 或 1K。

## 1. 一手材料与实际贡献

Xiang Li、Rongrong Wang、Qing Qu，*Towards Understanding the Mechanisms of Classifier-Free Guidance*，NeurIPS 2025。[正式全文](https://papers.nips.cc/paper_files/paper/2025/file/5ac55a8d65fb5ecd9ccaa852e21325db-Paper-Conference.pdf) 共 55 页，包含正文、理论和实验附录以及 checklist；本笔记公式与页码以该正式 PDF 为准。[arXiv 身份页](https://arxiv.org/abs/2505.19210) 显示 v1 为 2025-05-25，v3 为 2025-11-11。[作者官方代码](https://github.com/Morefre/Towards-Understanding-the-Mechanisms-of-Classifier-Free-Guidance) 冻结到 `bcf8dedfdc7ef0178d2a17d5d9543a7fc9f46af0`。README 的正文链接误指另一 arXiv 编号，论文身份按正式会论文及其 Data Availability 指向的仓库核实。

阅读了正式正文 §1–5、理论附录 A/B/H，以及 C、D、E、F、G、I 中的实验设定与边界，并核对官方线性 CPC 实现、非线性 notebook 和 Jacobian 工具。正式 PDF SHA256：`54e819012a314bb64efd0e290eda3fee7fad579729913260a6c98ba86e708c3f`。

作者关注一种实际错误：无 guidance 的条件生成仍过多共享跨类别的粗结构。解释是，线性近似只保留低阶统计，而真实类别的领先 PCs 常包含亮度、前景/背景等共有变化；高方差本身并不等于类别特有信息。将条件与无条件的响应作对比，可以突出相对属于目标类的变化，并抑制参照中更显著的变化。这里“语义特有”还依赖图像和实验佐证，不是从一个正特征值自动推出的结论。[正式 PDF §2–3、附录 D/E](https://papers.nips.cc/paper_files/paper/2025/file/5ac55a8d65fb5ecd9ccaa852e21325db-Paper-Conference.pdf)

## 2. 精确机制：不需要先指定活动窗口

采用论文的 VE 坐标，观测为 \(x=X+\sigma\epsilon\)。数据均值为 \(\mu\)、协方差为 \(\Sigma\) 时，平方损失下最优仿射去噪器为

\[
D(x,\sigma)=\mu+A_\sigma(x-\mu),\qquad
A_\sigma=\Sigma(\Sigma+\sigma^2 I)^{-1}.
\]

这一仿射最优式只需数据二阶矩；将它解释为完整 Bayes 去噪器及高斯采样分布，才需要高斯数据模型。其谱权重 \(\lambda/(\lambda+\sigma^2)\) 随噪声变化是机制自然给出的，区别于人工拼出的时间增益曲线。

对条件与无条件模型，令 \(A_c,A_u\) 如上，则有精确恒等式

\[
D_c-D_u=
\underbrace{(A_c-A_u)(x-\mu_c)}_{\text{随样本变化的 CPC 项}}
+\underbrace{(I-A_u)(\mu_c-\mu_u)}_{\text{与样本无关的均值项}}.
\tag{1}
\]

论文 score guidance 是 \(\gamma(D_c-D_u)/\sigma^2\)，不是直接在状态上加这个去噪器差。正文式 (12)–(15) 和附录 B.2 给出该分解。

\(A_c-A_u\) 是对称矩阵，可以按正、负谱分成 \(M_++M_-\)。在逆向积分的局部更新中，正谱沿相应方向增强 \(x-\mu_c\) 的分量，负谱抑制它们。对于精确 Bayes 去噪器，\(\operatorname{Cov}[X\mid x]=\sigma^2 J_D(x)\)，所以这里比较的是**后验**协方差，而不只是原始数据 covariance。附录 A 的对比重构误差优化也直接导出这个“相对方差”方向选择。

均值项在高噪声时近似 \(\mu_c-\mu_u\)，降低噪声后经 \(I-A_u\) 过滤；它不会因为换一张初始噪声就改变方向。这使“公共均值移动”与“围绕均值的结构变化”成为两个可区分的机制。式 (1) 不要求两协方差有共同特征基；但一般情况下，\(A_c-A_u\) 的特征向量随 \(\sigma\) 变化，不能把一套固定 PCs 用到底。[正式 PDF pp.3–6、15–18](https://papers.nips.cc/paper_files/paper/2025/file/5ac55a8d65fb5ecd9ccaa852e21325db-Paper-Conference.pdf)

## 3. 全轨迹定理保证什么

Theorem 1 额外假设 \(\Sigma_c,\Sigma_u\) 可同时对角化。设共同方向上方差为 \(\lambda_{c,i},\lambda_{u,i}\)，从 \(\sigma_T\) 积分到 \(\sigma<\sigma_T\)，CPC 相对于无 guidance 轨迹的模态缩放为

\[
h_i^{\gamma/2},\qquad
h_i=\frac{\lambda_{c,i}+\sigma^2}{\lambda_{c,i}+\sigma_T^2}
\frac{\lambda_{u,i}+\sigma_T^2}{\lambda_{u,i}+\sigma^2}.
\]

因此正 CPC 被增强，负 CPC 被抑制；均值 forcing 另外产生对所有初始噪声相同的平移。这个定理给出方向、作用方式和完整轨迹表达式，**不提供最优 \(\gamma\)、FID 非增、类语义正确性或真实神经网络的逐模态保证**。正文将均值积分核简记成 \(B_\sigma\)，但附录式 (63) 明确还含 \(\gamma\) 与起点 \(\sigma_T\)，不能将其误读为与 gain 无关的固定时间核。

两处迁移时需要更严格：

- 在线性系统中，单独增加样本无关的 forcing 只改共同平移，不改中心协方差、样本间欧氏距离或微分熵。论文关于均值项降低多样性的视觉和非线性解释，不能升级为“共同平移必然降低概率分布多样性”的定理。
- 正文式 (16) 的 \(D(x)=J_D(x)x\) 需要一阶齐次等额外结构。一般“没有显式 bias”或 \(D(0)=0\) 均不充分；\(D(x)=W(x)x\) 也不说明 \(W(x)=J_D(x)\)。精确后验 covariance/Jacobian 恒等式与这个局部齐次假设是两件事。

附录 H 的混合高斯推广仍有有用的精确分解：参照变成 posterior responsibility 加权的各分量响应，均值项也依输入变化。但其中加权的 within-component 后验 covariance 不等于完整混合后验 covariance，后者还含分量后验均值间的 covariance；不能省掉这一项后声称得到完整 \(J_{D_u}\)。[正式 PDF Theorem 1、§4.2、附录 B/H](https://papers.nips.cc/paper_files/paper/2025/file/5ac55a8d65fb5ecd9ccaa852e21325db-Paper-Conference.pdf)

## 4. 实验支持与参数自由度

| 证据 | 支持的结论 | 不能据此声称 |
|---|---|---|
| 线性模型与 EDM 的同噪声可视化、CPC 方向直方图 | 对比特征增强/抑制可解释部分观察 | 方向投影增大就等于 FID 改善 |
| 正文 §4.1、附录 F.2 的逐类 50K FDDINOv2 | 高至中噪声下线性解释有实际对应，且不同分量因类别而异 | 全 ImageNet、固定无窗口、同成本 SOTA |
| 附录 G 的 EDM-2 latent 512 实验 | 解释并非完全局限于像素 EDM-1 | 已经验证 RAEv2 高维同类双头 |
| 附录 F.4 的非线性正 CPC 式 (17) | 选择相对特征比放大全部 conditional PCs 更合理 | 它与真实 CFG 完全等价或优于 CFG |

需要保留具体实验身份：附录 C 的 ImageNet 线性近似主要用**每个所选类别 50K 模型生成样本**估计均值/covariance，而不是仅用约 1K 张真实类样本。附录 F.2 明确按类别经验选择 guidance 区间；图 5/8/36 用 \(\gamma=15\) 做单时刻作用；EDM-2 图 37 固定 golden retriever、区间 \([16.02,80)\)，扫描 \(\gamma\in\{0,5,10,15,20\}\)。所以其机制很有参考价值，但实验设置不满足我们“无手工窗口、无大量扫参”的准入要求。

附录 D 表 1 的 `Avg. FID` 是**类与类之间的 FID**：真实数据 226.6，naive 20-step Euler 214.6，100-step Heun 216.3，CFG \(\gamma=4\)、20-step Euler 258.9。它显示类分离度变化，且 CFG 超过真实类间距离；不是普通 generated-vs-real FID 的 SOTA 表。附录 F.3 的均值偏置噪声初始化也扫描多个强度，不能拿来替代本任务的 guidance 设计。[正式 PDF pp.19–20、25、34–48](https://papers.nips.cc/paper_files/paper/2025/file/5ac55a8d65fb5ecd9ccaa852e21325db-Paper-Conference.pdf)

## 5. 官方实现如何落地，成本在哪里

[官方 `cpc_utils.py`](https://github.com/Morefre/Towards-Understanding-the-Mechanisms-of-Classifier-Free-Guidance/blob/bcf8dedfdc7ef0178d2a17d5d9543a7fc9f46af0/edm/Utils/cpc_utils.py) 确实计算 \(A_c-A_u\) 并通过 `eigh` 分离正负谱；它另收 `guidance_strength`、`sigma_high`、`sigma_low`。推理时间结构中 \(1/\sigma\) 因子是去噪器差到 VE ODE 更新的单位转换，不能误称作者又手调了一个时间 schedule；真正自由选择的是强度和活动区间。

[非线性官方 notebook](https://github.com/Morefre/Towards-Understanding-the-Mechanisms-of-Classifier-Free-Guidance/blob/bcf8dedfdc7ef0178d2a17d5d9543a7fc9f46af0/edm/Investigate_CFG_in_Nonlinear_Regime.ipynb) 明确将式 (17) 作为启发式：先计算 \(J_{D_c}-J_{D_u}\) 的正谱，再作用到 **\(D_c(x)\)**，而非从线性恒等式直接延用 \(x-\mu_c\)。作者说明前者视觉效果更好；这一步不是由 Theorem 1 唯一导出。当前示例选单样本、step 12，展示完整 Jacobian；论文全部预计算脚本在 notebook 中注明按请求提供。

[`jacobian_utils.py`](https://github.com/Morefre/Towards-Understanding-the-Mechanisms-of-Classifier-Free-Guidance/blob/bcf8dedfdc7ef0178d2a17d5d9543a7fc9f46af0/edm/Utils/jacobian_utils.py) 用 `torch.autograd.functional.jacobian` 逐图生成完整矩阵。RAEv2 的 \(D=1024\times16\times16=262144\)，仅一个稠密 FP32 Jacobian 就需 **256 GiB**，尚未计另一个分支、计算图和特征分解，不能称同成本可迁移。改成矩阵自由 JVP/VJP 也会产生真实额外调用及数值近似条件，本次没有实施。

另一个实现前提不能隐藏：notebook 直接对网络 Jacobian 差调用 `eigh`，未见对称性检查或对称化。最优 Bayes Jacobian 的确对称，但任意学习网络不受此硬约束。`eigh` 默认只读下三角，不会自动验证对称性或替用户计算 \((J+J^T)/2\)。因此不能照抄该代码后就称取得“真实后验 covariance 的正谱”。[PyTorch 2.11 官方 API](https://docs.pytorch.org/docs/2.11/generated/torch.linalg.eigh.html)

## 6. 对 RAEv2 和当前 JQ 弱分支的可迁移问题

本地 [配置](../experiments/configs/raev2_strict_lpl_dinov3l_k7.yaml) 使用 clean-X prediction。[Full/Base](../external/RAEv2/src/stage2/models/DDT.py) 共享类别条件，Base 在 encoder depth 8 分出。若两头都收敛到同一最优条件 Bayes 估计，它们的均值项与 covariance 项之差都应归零。实际非零 \(F-B\) 表示两头近似误差不同，不能把 Base 的参照均值写成 \(\mu_u\)，也不能直接把它的 Jacobian 当作无条件 posterior covariance。

坐标转换也必须明确。对 RAE 线性桥 \(z_t=(1-t)X+t\epsilon\)，当 \(0<t<1\) 时可令 \(y=z_t/(1-t)\)、\(\sigma=t/(1-t)\)。精确 clean-X 后验满足

\[
\operatorname{Cov}[X\mid z_t,c]
=\frac{t^2}{1-t}\,J_{z_t}D_t(z_t,c).
\]

不能原样搬用 \(\sigma^2J_{z_t}D_t\)，也不能在 \(t=1\) 使用这个含除零的表达式。RAE 的 RMSNorm、SiLU、softmax、条件调制及带 bias 的线性层，亦不满足式 (16) 所需的一阶齐次假设。

**下一条可证伪的机制问题**是：固定 JQ 删除位置特有检索后，\(F-W\) 是否主要保留“样本相关、相对缺失的结构响应”，而不只是一个公共偏置或原 \(F-B\) 的幅值重写。这与 CPC 的“对比而非全量增强”直觉相关，但 JQ 的反向 KL 重心保证本身不能证明这个输出方向有益。[当前 JQ 结构协议](RAEV2_DECODER_QUERY_MEAN_PROTOCOL_20260906_ZH.md) 能检验信息删除、响应和原 gap 的关系；不能由这些指标宣布获得 class-specific CPC。

若后续要声称均值/CPC 机制，必须有同类别、同噪声水平下多个独立样本来区分公共均值与中心变化；当前 8 个 ID 各属不同类别、重复十个时间的 160 行不足以估计这样的类条件分解。若再声称它在纠正 clean-X 误差，需要独立的目标误差证据。例如对固定方向 \(d=F-W\)，平方误差恒等式

\[
\mathbb E\|F+\alpha d-X\|^2-\mathbb E\|F-X\|^2
=2\alpha\mathbb E\langle F-X,d\rangle+\alpha^2\mathbb E\|d\|^2
\]

可用于证伪某个明确的“去噪误差纠正”解释；它不是 FID 保证，也不能把对该损失拟合的系数直接升级为部署 gain。相反，若目标是分布层面的 class-specificity，仍须另外定义并验证该目标。本文不据论文或结构诊断选 \(\alpha\)、窗口、谱秩或训练目标。

**结论：接受对比结构与自然谱时间权重的机制启发；拒绝将这篇论文直接视为 RAEv2 的无参数 guidance 配方。** 下一步应先建立弱分支删除了什么、其差是否对应 Full 的实际不足，再由明确目标推导系数，并最终做公平成本下至少 5% 的质量验证。

## 7. 本次 CPU 核验与归档

归档目录：[reading_linear_cpc_v1](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/reading_linear_cpc_v1)。包含正式全文及提取文本、核读的官方源码和 notebook、版本树、相关本地模型/config、主源网页、可重跑代数脚本与 SHA 清单。

[小型代数核验](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/reading_linear_cpc_v1/algebra_check.json) 验证非对易协方差下式 (1) 的误差为 `5.55e-17`，共同特征基的附录 B 解与二维 ODE 数值解误差为 `3.36e-12`，公共平移前后中心 covariance 误差为 `1.73e-18`。这只是公式核对，不使用模型或真实数据；固定二维系数只用于验证恒等式，没有 guidance 扫参。CPU wall `0.3402865 s`，进程 CPU `0.3370333 s`，边界含 NumPy/SciPy imports、截止 JSON 写出前。

下载及 `pdftotext` wall `5.0258744 s`；下载进程 CPU `0.2646999 s` 不含 `pdftotext` 子进程 CPU。官方代码下载另有独立计时。网页浏览、阅读、页面渲染、文字写作和整个会话总成本没有统一计时，不伪报为上述 CPU 数字。一次 PyMuPDF 渲染尝试因环境无 `fitz` 失败，随后用现有 `pdftoppm` 核读正式页 7/46；未安装依赖、未运行官方 notebook。模型前向、VJP、GPU 调用均为零。
