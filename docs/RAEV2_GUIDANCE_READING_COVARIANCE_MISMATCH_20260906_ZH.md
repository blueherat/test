# Covariance Mismatch：分量可辨识性与冻结模型的边界

日期：2026-09-06。已取得并阅读全文及附录说明，目视核对表 2。[Everaert、Süsstrunk、Achanta 的 2024 年 EPFL 预印本，30 页](https://infoscience.epfl.ch/server/api/core/bitstreams/72923daf-0e38-4b3c-8ad8-96799be59469/content)有很好的机制直觉：不同方差分量在不同噪声级才可辨识。但它没有直接给出满足本研究条件的推理 guidance。最强的协方差对齐结果依赖改变训练任务；不重训版本只校正起点分布。没有启动 GPU、训练、采样或新实验。

[正文 §3–4，pp.3–6](https://infoscience.epfl.ch/server/api/core/bitstreams/72923daf-0e38-4b3c-8ad8-96799be59469/content)通过分频输入/输出 SNR 分析生成次序；提出 whiten data 或 color noise，并在各自新任务上训练。DFT 只近似按空间平稳的逐通道协方差对齐，并非完整条件协方差。§4.3 的免重训方法则保留原 denoiser，仅用 `x_T=α_T ε_colored+σ_T ε_white` 初始化，两个噪声独立，协方差为 `α_T²Σ+σ_T²I`。它缓解 late start 的起点错配；论文明确承认频段分工、早停和编辑的限制仍在。这里没有每步新 guidance 向量场。

可以将其直觉补成一个闭式机制。**以下 Gaussian 公式是本次独立推导，不能称为论文已给出的定理。** 固定原有白噪声路径

`X_0∼N(μ,Σ)`，`X_t=α_t X_0+σ_t ε`，`ε∼N(0,I)`。

设 `Σ=U diag(λ_i) Uᵀ`，在其本征坐标中，后验均值和后验方差为

`m_i(y,t)=μ_i+ α_t λ_i/(α_t²λ_i+σ_t²) · (y_i−α_t μ_i)`，

`Var(X_0,i | X_t)=λ_i σ_t²/(α_t²λ_i+σ_t²)`。

由此，输入携带的相对信息量为 `r_i=α_t²λ_i/σ_t²`，先验均值对预测的影响为

`∂m_i/∂μ_i = σ_t²/(α_t²λ_i+σ_t²)=1/(1+r_i)`。

当 `r_i≫1`，预测接近 `y_i/α_t`，主要继承输入，改变先验/条件均值的效应很小；当 `r_i≪1`，预测回归 `μ_i`，不能从当前输入辨识该分量。`r_i≈1` 的过渡点完全由谱和噪声路径给出，不需要手写时间窗口。高、低 λ 的过渡点分离，解释了为什么“同一个 scalar t”不是各方向相同的信息状态。这是比笼统强调低频/高频更可迁移的机制：关键是相对于噪声的方差，而非预先命名某个频段。

若改成 `ε_colored∼N(0,Σ)`，Gaussian 情形的后验变为

`m_col(y,t)=μ+α_t/(α_t²+σ_t²)·(y−α_t μ)`，

`Cov(X_0 | X_t)=σ_t²/(α_t²+σ_t²)·Σ`。

所有方向相对后验不确定性相同，协方差对齐确实消除了上面的方向性分工。白化数据后以白噪声训练，反变换后有相同理想后验；两种 MSE 的有限容量优化权重却不同。这给出它的设计为何成立，而不是只说“归一化可能有用”。但它描述的是**新噪声路径对应的新后验**。旧模型学到的是原路径，不能给 frozen RAEv2 的输出乘一个白化矩阵，就宣布它变成这个后验。

更一般的独立线性代数观察：对数据和噪声同时做可逆坐标变换 A，协方差对 `(Σ_data,Σ_noise)` 变成 `(AΣ_data Aᵀ,AΣ_noise Aᵀ)`；其广义本征值不变。因为新矩阵比值与 `Σ_noise⁻¹Σ_data` 相似，方向 SNR 谱不会被纯坐标重写消除。**保持原 forward path 的输入/输出白化包装，无法免费获得论文改变 data/noise 相对几何后的全部效果。** 对角协方差和高斯只是上面闭式后验所需；这个不变性本身只需协方差正定与 A 可逆。

免重训初始化为什么有用也能清楚界定：若数据真为零均值 Gaussian，`α_T ε_colored+σ_T ε_white` 正是原路径的真实 `p_T`，而纯白初始化通常不是。若数据非 Gaussian，它只匹配二阶矩；若均值非零，还缺 `α_T μ`。即使完全修复 p_T，原路径后验的各方向信息差异仍然存在。它改的是边界条件，不是重新学会在低噪声阶段生成已经保留在输入中的高方差成分。

对“能否导出不改变已训练路径的 guidance”，有一个严格但受限的正向答案：若能先建立**目标与当前模型的条件分布误差模型**，可在原路径上直接求差。例如两者真为 Gaussian，目标/基准参数为 `(μ_*,Σ_*)` 与 `(μ_b,Σ_b)`，令 `C_j(t)=α_t²Σ_j+σ_t²I`，则

`Δs_t(x)=−C_*(t)⁻¹(x−α_t μ_*)+C_b(t)⁻¹(x−α_t μ_b)`。

在精确 Gaussian 模型下，将这个已知差补到基准 score 得到目标 score，随后使用该原路径的正确 probability-flow/反向过程；时间依赖来自 Bayes 公式，没有外推系数扫描。这个例子展示“机制→推理干预”的逻辑入口，而不是一个已准入的 RAEv2 方法：真实非 Gaussian score 不由协方差唯一决定，Full/Base 都是同类共享主干，也不天然等于这两个 Gaussian 分布。必须解释目标误差是什么，以及为什么修复它改善条件生成；论文的**全局无条件**谱不够确定这个 Δs，更不保证 FID。尚未据此拟合矩阵、选频段或发起实验。

已训模型与少步结果需要按原表读。以下数据来自主文表 2（p.7；截图已归档）：普通、late start、early stop 均为 **10 次 denoiser 调用**；单噪声级为 **3 次调用**。**表内 `w_cfg=1`，并不是通常的强 CFG 质量设置。** [附录 B.3.11，p.28](https://infoscience.epfl.ch/server/api/core/bitstreams/72923daf-0e38-4b3c-8ad8-96799be59469/content)说明每格仅生成 **200 张 DrawBench 图**，对 **10k COCO val2014 真图**计算 torchmetrics FID。

| 方法 | 完整范围 | Late start：T=700 | Early stop：t_last=600 | 单噪声级：t=600，3 次调用 |
|---|---:|---:|---:|---:|
| 原白噪声任务 | 194 | 264 | 230 | 279 |
| 原任务＋仅初始化校正 | 193 | 233 | 227 | 323 |
| Colored-noise 任务重训 | 207 | 226 | 226 | 227 |

表中“退化更少”以每个方法各自完整范围 FID 为分母，并不等于优于最佳完整范围基准。Colored-noise 在完整范围反而从原模型的 194 变成 207。单噪声级列用了另外训练的模型，不能当成前三列同一 checkpoint 的纯采样消融。200 张样本及 DrawBench/COCO 的内容分布差异，也限制了把这些数字解释成泛化质量排序；论文展示的是鲁棒性/可用性，尚非本研究所需的公平成本 SOTA。

训练与选择自由度已在原文披露。[附录 B.2，p.26](https://infoscience.epfl.ch/server/api/core/bitstreams/72923daf-0e38-4b3c-8ad8-96799be59469/content)：彩噪模型从 SD2.1-base 初始化，以 100k 图文对训练 18k 步、batch 32、约 9 A100-80GB 小时；单噪声级版本另训 22k 步、约 11 小时。[B.3.9，p.28](https://infoscience.epfl.ch/server/api/core/bitstreams/72923daf-0e38-4b3c-8ad8-96799be59469/content)固定 `t=600, Δt=200` 重复 denoise；Fig.10（p.11）明确说看过 n 的效果后，3 次是在质量与调用数间选择的折中。还存在 late-start T、early-stop t_last、训练 noise range 和展示用 CFG 的选择；它不是由协方差机制唯一导出的无调参少步算法。Fig.11–13 声明所示更多样本没有挑图，这与已经选择采样设定是不同层面的事情。

需要分开论文与后来发布的模型资料。[当前作者 SD2.1 模型卡](https://huggingface.co/EPFL-IVRL/sd2.1-base-covariance-mismatch)确认改变噪声或数据任务后再次训练，也提供单时刻推理代码（含 `fixed_delta_t=200`）；但卡上写的是 20k 步，并增加了 DCT、whitened-data 等版本。这里以 **2024-11-27 PDF** 的 18k/22k 和表 2 作为论文证据，不将后来模型卡的数据混入原实验。当前 [官方 GitHub](https://github.com/IVRL/covariance-mismatch/tree/2b9e660cbe5c27264c888e11114f23e9596817c5)只有项目网页及建设中 README；没有据此声称完整训练源码已公开。没有下载受限模型权重或接受账号授权条款。

本论文可留下的研究标准是：以方向上的后验可辨识性理解“何时能改动什么”，先区分初始化、已训练路径和条件生成误差。用旧模型纠正分布误差需要独立的目标机制；直接换噪声/白化或凭谱设置 guidance gain 都缺这一步。正文把低噪声阶段“不能自纠正”说得较强；上面的 Gaussian 分析给出的是连续衰减，非普遍零响应定理，真实跨分量依赖还可能传递信息。我们保留其机制直觉，同时不把它当作新 guidance 的理论保证。

归档：`/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/reading_covariance_mismatch_v1`。原下载入口 405，官方 DSpace `/server/api/core/bitstreams/.../content` 一次成功取得 146,140,771 bytes；PDF MD5 与 API 元数据一致。全文、表 2 截图、作者公开模型卡、GitHub commit/tree、失败请求边界均保存。阅读范围和数值见 [reading_evidence.json](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/reading_covariance_mismatch_v1/reading_evidence.json)，完整 SHA 见 [manifest.json](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/reading_covariance_mismatch_v1/manifest.json)。下载阶段有实际 wall/CPU 记录；pdftotext、页面渲染及人工阅读总耗时未完整计时，如实未测，不拼接成总成本。
