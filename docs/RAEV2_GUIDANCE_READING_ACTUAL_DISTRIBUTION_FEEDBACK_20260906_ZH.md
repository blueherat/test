# RAEv2：实际生成分布反馈与判别器梯度的可估计性

日期：2026-09-06。只读原文、证明和作者代码；未训练、采样、调用 GPU 或新增性能实验。

**两篇分别补上一个实质缺口：分类准确不等于校正梯度准确；旧负例上的准确也不等于当前生成分布上的准确。** Sobolev 正则给出第一点的统计机制，Discriminator Flow 给出第二点的具体生成反馈实现。它们尚未给出适用于 RAEv2 的自然有限步幅度或公平成本优势。本次唯一值得继续保留的方向，是已有“实际 rollout 密度比输运”设想的梯度正则与反馈刷新版本，并非一个获准实施的新方法。

## 1. 与已有轨迹的区别及阅读范围

已先核对 [阅读索引](RAEV2_GUIDANCE_READING_SYNTHESIS_20260906_ZH.md) 与 [已有判别器笔记](RAEV2_GUIDANCE_READING_DISCRIMINATOR_20260906_ZH.md)。后者已经读过 ICML 2023 DG 和 ECML 2025 改进 DG，已经区分真实前向边缘、实际 rollout 边缘、终点假样本再加噪边缘，并推导了实际分布校正的 KL 一阶贡献。本次不把这些结论再计作新发现，也不重启已有 X−G 势函数回归。

| 本次新增原文 | 原始来源与版本 | 实际阅读／核验范围 |
|---|---|---|
| Xie, Blanchet, Xu, *Sobolev Regularized Score Difference Estimation in Diffusion Models* | [arXiv 2608.18237v1](https://arxiv.org/abs/2608.18237)，[65 页 PDF](https://arxiv.org/pdf/2608.18237v1)，[全文 HTML](https://arxiv.org/html/2608.18237v1)。首页脚注明确 accepted at ICML 2026；这里归档的是 arXiv v1，不声称与会议最终稿逐字一致 | 正文 §1–6、Assumption 2.1、Theorems 3.1/3.5/4.1/5.2；逐项核验 Lemmas 3.3/3.4 的 Appendix A.1/A.2 证明，读 A.4 与 C.3 的误差分解／末端界及全部实验附录 D.1–D.3。未逐行复证其余统计复杂度和 minimax 附录 |
| Franceschi et al., *Unifying GANs and Score-Based Diffusion as Generative Particle Models*，NeurIPS 2023 | [会议页面](https://proceedings.neurips.cc/paper_files/paper/2023/hash/bbc461518c59a2a8d64e70e2c38c4a0e-Abstract-Conference.html)，[正式正文及附录](https://proceedings.neurips.cc/paper_files/paper/2023/file/bbc461518c59a2a8d64e70e2c38c4a0e-Paper-Conference.pdf) | 聚焦 Discriminator Flow：正文 §3.2、§4.2–4.3，Appendix A.4/B.2/C.5/D.1–D.3，Algorithms 2/3、Tables 3/4、官方补充代码及作者仓库。未将全文统一框架的全部外引结论重新证明 |

作者代码分别固定于 [Sobolev commit 33f5769](https://github.com/chenghands-on/Sobolev-Regularized-Score-Difference-Estimator/tree/33f57698261a26fcb3bc5a332ea2a4d7b9ec9f89) 和 [GPM commit 3be2653](https://github.com/White-Link/gpm/tree/3be265398d40f0bb1511d94f1eb19eb7c40b7aa0)。只读核心代码，没有导入或执行作者 Python。PDF、补充包、核心源文件与 SHA 均归档于文末目录。DF 的两个核心文件与会议补充包逐文件 SHA 相同。

## 2. Sobolev：从“估计 logit”转向“同时控制其导数”

以下先沿用该论文的记号：source 为 p，target 为 q，与我们习惯的真实 p／生成 q 恰好相反。平衡二分类的混合分布为 \(\mu=(p+q)/2\)，真实标签为 target=1；Bayes logit 为 \(f^\star=\log(q/p)\)。作者优化

\[
J_\lambda(f)=\mathbb E_\mu \operatorname{BCE}(Y,f(X))
 +\lambda\,\mathbb E_\mu\|\nabla f(X)\|^2 .
\]

这里没有 clean 残差 X−G 标签。CE 的小幅高频误差可以在求导后放大；Dirichlet 项直接对使用中的空间梯度施加代价。Lemma 3.3 的关键不只是“加正则更稳定”，而是有限 logit 界 \(|f|,|g|\le M\) 下的强凸不等式：

\[
J_\lambda(f)-J_\lambda(g)-DJ_\lambda(g)[f-g]
\ge {c_M\over2}\|f-g\|_{L^2(\mu)}^2
 +\lambda\|\nabla(f-g)\|_{L^2(\mu)}^2,\qquad
c_M={1\over4\cosh^2(M/2)}.
\]

这是对实际所需梯度误差的明确约束。证明只需 logistic 曲率与平方梯度项的展开；本次已核验。它也明确暴露了代价：有限 \(\lambda\) 的总体最优 \(f_\lambda\) 不是精确 log 比值。在 Assumption 2.1 的光滑性、密度上下界及加权 Neumann 边界条件下，Lemma 3.4 给出

\[
\|f_\lambda-f^\star\|_{L^2(\mu)}^2\le C\lambda^2,\qquad
\|\nabla(f_\lambda-f^\star)\|_{L^2(\mu)}^2\le C\lambda.
\]

该证明由约束最优的一阶变分不等式、logistic 曲率及对 \(f^\star\) 的加权 Green 恒等式组成，不需知道训练样本的真实 score。作为独立解释，在约束不活跃的内点，Euler–Lagrange 方程是
\(\rho(\operatorname{sigmoid}f_\lambda-\eta)-2\lambda\,\operatorname{div}(\rho\nabla f_\lambda)=0\)，其中 \(\rho=d\mu/dx\)、\(\eta=q/(p+q)\)。正则通过一个椭圆平滑项改变估计目标；不能既使用有限正则又宣称精确 Bayes logit。

Theorem 3.1 报告的是**平方 H¹ 误差**率
\(\|\hat f-f^\star\|_{H^1(\mu)}^2\lesssim n^{-(s-1)/(d+2s-2)}\log n\)，而非未平方范数同速率。条件包括有界连通域、正且有上下界的 C² 密度、\(2\le s\le4\)、受控输出和梯度的 ReLU³ 网络类、以及取得全局经验极小值。其网络大小 \(N\asymp n^{d/(d+2s-2)}\)、\(\lambda\asymp N^{-(s-1)/d}\) 给出理论量级，但未唯一决定有限样本常数，也依赖未知光滑度。与 Theorem 4.1 的 minimax 下界仍有指数差，不能称 minimax 最优或已经消除高维困难。

§5/Theorem 5.2 把此估计推广到同一 VP 核下的 source／target 前向加噪分布；需远离零噪声、紧支撑初始分布、空间截断／重标以及相应导数控制，得到带次多项式因子的积分平方 H¹ 界。这一部分增加时间维度，不是实际逆向 rollout 分布的闭环保证。有限噪声带来的光滑性有帮助，但不意味着当前 RAE 超高维、有限样本 critic 自动满足定理。

### 实际效果、刷新成本及代码与理论的对应关系

合成三组分布、三种样本量的九组比较中，Sobolev 相对同类无正则 classifier 的梯度 MSE 全部改善，表中降幅 18.88%–52.86%。这是直接针对所需 score 差的正面证据；不是只有分类准确率。Office–Caltech-10 的 PCA100 域适配则有胜有负，不能把“总体较好”改成逐设置全面领先。[原文 §6、Tables 1–4](https://arxiv.org/html/2608.18237v1)

作者 [WGF 演示](https://github.com/chenghands-on/Sobolev-Regularized-Score-Difference-Estimator/blob/33f57698261a26fcb3bc5a332ea2a4d7b9ec9f89/gradest2/demo_caltechoffice.py#L175) 显式打开逐步重训；[更新函数](https://github.com/chenghands-on/Sobolev-Regularized-Score-Difference-Estimator/blob/33f57698261a26fcb3bc5a332ea2a4d7b9ec9f89/gradest2/wgf_da.py#L118) 每次用当前移动粒子／更新伪标签的联合分布暖启动 classifier，最多 1000 epochs，early stopping。然后计算输入梯度、以 0.01 移动，再刷新分布。Table 4 的 0.0004 s 对 2.0617 s 是**梯度查询计时**；源码把每步重训另计，不能读成完整反馈总成本。另有不刷新分支，本结论特指实际演示中的 refresh=true。Sobolev 系数、训练退火与粒子步长有手选值，不能称零参。

ECG 迁移是 PTB-XL 到仅使用 10% ICBEB2018 的目标域。主文 Table 5 的下游 AUC/F2/G2 从 TGDP 的 0.905/0.662/0.436 提升至 0.915/0.693/0.453。附录 Table 6 的 Fréchet 指标使用目标任务 xresnet1d50 特征，**并非 ImageNet Inception FID**：

| ECG 方法 | 特征 Fréchet 距离 | 可训练／报告参数 | 论文训练时间 |
|---|---:|---:|---:|
| Vanilla Diffusion | 11.171 | 50.2M | 1 h |
| Finetune Generator | 8.415 | 50.2M | 40 min |
| TGDP | 8.100 | 2.8M | 30 min |
| TGDP-SOB | 8.097 | 2.8M | 30 min |

保留少量数据迁移及下游任务改善的价值；Sobolev 相对最接近 TGDP 的 Fréchet 降幅仅约 0.037%，没有现代图像模型、公平墙钟下 5% 的证据。没有完整披露训练硬件、推理墙钟和逐步 NFE。[原文 Appendix D.3](https://arxiv.org/html/2608.18237v1)

此外，固定版本中 ECG 路径与论文 CE 理论有两个必须分开的实现事实：

1. [clean ratio 函数](https://github.com/chenghands-on/Sobolev-Regularized-Score-Difference-Estimator/blob/33f57698261a26fcb3bc5a332ea2a4d7b9ec9f89/SSSD-ECG-main/src/sssd/density_ratio_guidance.py#L139) 返回 clean target/source 的 log 比值；[训练更新](https://github.com/chenghands-on/Sobolev-Regularized-Score-Difference-Estimator/blob/33f57698261a26fcb3bc5a332ea2a4d7b9ec9f89/SSSD-ECG-main/src/sssd/density_ratio_guidance.py#L674) 把这个值作为 source 加噪输入的 MSE 标签，变量虽叫 loss_ce，却不是两种 noisy 分布的 CE。理想无正则极限得到 \(\mathbb E_p[\log r_0(X)\mid X_t]\)，而真正 noisy log 比值为 \(\log\mathbb E_p[r_0(X)\mid X_t]\)。二者一般不相等；Sobolev 项不会消除这种条件期望／log 不交换。它也不是旧 X−G 残差标签，不能混为同一个回归问题。
2. [输入梯度](https://github.com/chenghands-on/Sobolev-Regularized-Score-Difference-Estimator/blob/33f57698261a26fcb3bc5a332ea2a4d7b9ec9f89/SSSD-ECG-main/src/sssd/density_ratio_guidance.py#L618) 在 [采样器](https://github.com/chenghands-on/Sobolev-Regularized-Score-Difference-Estimator/blob/33f57698261a26fcb3bc5a332ea2a4d7b9ec9f89/SSSD-ECG-main/src/sssd/utils/util_generation.py#L236) 中直接以正号加到 epsilon 预测。若其字面含义真为 \(\nabla\log(q_t/p_t)\)，VP epsilon 参数化应是
   \(\epsilon_{\rm target}=\epsilon_{\rm source}-\sqrt{1-\bar\alpha_t}\nabla\log(q_t/p_t)\)。
   此代码的符号与噪声尺度不能直接作为理论 score 校正公式。代码还支持手选的 Sobolev warmup／decay；默认另一个 consistency 项关闭。此处只指出公开源文件的对应关系，未运行代码或获得论文表格 checkpoint，不据此断言表格实现或结果错误。

## 3. Discriminator Flow：真正使用当前生成路径的负例

Discriminator Flow 的时间是从噪声到图像的生成进度。其标量时间条件 critic \(f_\phi(x,s)\) 同时定义粒子速度 \(-\eta\nabla_x c(f_\phi(x,s))\)。Algorithm 3 与 [官方核心](https://github.com/White-Link/gpm/blob/3be265398d40f0bb1511d94f1eb19eb7c40b7aa0/gpm/models/discr_flow/base.py#L235) 的顺序是：

1. 每批用当前 \(\phi\) 从新 Gaussian 噪声执行 Euler rollout。
2. 每个样本独立抽取离散时刻，使用当时状态作为负例；正例是在同时间输入的**干净真实数据**。
3. 固定这些状态做一次 critic 参数更新；下批由更新后的场重新生成轨迹。

负例由当前实际离散生成器产生，没有把它换成某个 endpoint 的前向加噪。整个 rollout 被 no_grad 截断，局部输入梯度仍用于生成场，所以训练不用保存整条参数反向图。每批的完整生成前向成本仍存在。这里的正例是各时刻同一个静态 \(p_{\rm data}\)，并非 Gaussian bridge \(p_t\)；该方法从头训练生成场，也不是可直接接在固定 IG 上的插件。

图像实验使用 IPM loss 与零中心梯度惩罚，反馈是 \(-\eta\nabla f\)，**没有校准的 log 密度比身份**。Gaussian toy 的非饱和 logistic GAN，即使 critic 最优 \(f=\log(p_{\rm data}/q_s)\)，速度仍是 \(\eta(1-\operatorname{sigmoid}f)\nabla f\)，也不是单位 score 差。选定目标可决定反馈函数 c 的形式；\(\eta\)、GP、离散网格仍为参数。统一 particle 框架提供机制解释，不证明有限 critic、有限更新的 KL 单调性。[正式原文 §4.2、Appendix B.2](https://proceedings.neurips.cc/paper_files/paper/2023/file/bbc461518c59a2a8d64e70e2c38c4a0e-Paper-Conference.pdf)

Table 3 中 DF 的 MNIST FID 为 4，对照 EDM 为 3；CelebA64 为 41，对照 EDM 为 10。正面结果是：单一时间条件标量 critic 确实能通过当前分布反馈产生图像，无需独立 generator 或已知 data-score；不是 FID SOTA。C.5/Figure 7 在 MNIST 相同向量场 NFE 下约可比 Heun EDM，但弱于低 NFE 的 Euler EDM；每次 DF 查询还需 critic 输入反传，同 NFE 不等于同墙钟。D.3 的 MNIST 单 V100 训练约 24 h，对照 EDM 6 h、GAN 1 h。训练每批刷新轨迹的费用不可略掉。MNIST 的 \(\eta\) 为 2 或 1、训练 rollout 为 64 或 128、GP 为 0.04；CelebA 的对应值为 2、25、0.05，另用验证 FID 选 checkpoint。

更完整的源码索引、算法步数及评估边界见 [DF 独立核验](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/reading_actual_distribution_feedback_v1/particle/README_ZH.md)。

## 4. 只保留一个机制问题：能否可靠跟踪当前分布的校正梯度

恢复本仓库记号：\(p_s\) 是随 dataward 时间 \(s=1-t\) 变化的真实 bridge，\(q_s\) 是当前受控采样器实际 rollout 边缘。定义 \(r_s=\log(p_s/q_s)\)。已有笔记已证明，同一个当前快照上加入 \(a\nabla r_s\) 对 KL 导数的贡献为

\[
-a\|\nabla r_s\|_{L^2(q_s)}^2.
\]

这不要求原 Full/Base/IG 是自身边缘 score。若实际方向为 \(\nabla\hat r_s=\nabla r_s+e_s\)，贡献改成

\[
-a\left(\|\nabla r_s\|_{L^2(q_s)}^2
 +\langle e_s,\nabla r_s\rangle_{L^2(q_s)}\right).
\]

例如 \(\|e_s\|_{L^2(q_s)}<\|\nabla r_s\|_{L^2(q_s)}\) 足以保证该局部贡献为负。原速度与真实 bridge 速度的失配项仍在，不能从这一项推出总 KL 单调。这是本仓库已有机制，两个新来源补充的是：

- **估计结构：** 使用真实 bridge／当前 rollout 的两样本分类，训练时直接约束输入梯度；Sobolev 论文解释了为何需要梯度正则及其偏差—方差代价，而不是只报告 held-out CE。
- **反馈结构：** 状态分布改变时刷新负例，再更新 critic；DF 与 Sobolev 的 WGF 代码说明这种迭代是实际可实现的，并清楚显示其训练费用。

这合起来仍只是一项待验证的机制：**以实际当前分布为负例、对输入梯度作统计控制的密度比反馈。** 它与终点再加噪 DG、真实 bridge 上的 X−G 回归、Learn-to-guide 的单步边缘匹配目标均有区别；也不是换名重跑旧 pilot。

要成为 RAEv2 候选，至少还须解决：同类条件下可辨识的真实／实际分布数据；critic 梯度误差及在刷新间隔内的变化；有限精度和有限 Euler 步下的稳定传递；从目标或约束导出控制幅度及训练正则的固定规则；包含采样负例、更新、输入反传与摊销的成本。论文的正则渐近量级不唯一决定这些有限常数，不能据此手扫 a、\(\lambda\) 或时间窗口。现有来源没有同时解决这些条件，本次不指定幅度、窗口或新采样实验。

一个有证伪入口的必要问题是：在选定的可解析密度比／实际分布上，正则和刷新是否确实降低所用梯度的误差、保持正向内积，而非仅改善 CE；这个问题若失败，后续质量实验缺少所宣称的机制支撑。其成功也只建立估计和反馈的局部证据，仍需独立的生成质量及公平成本比较。这里没有把理想 KL 性质升级成 FID 定理。

## 5. 来源与归档

归档目录：[reading_actual_distribution_feedback_v1](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/reading_actual_distribution_feedback_v1)。

- [总 manifest](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/reading_actual_distribution_feedback_v1/archive_manifest.json)：原文、网页、作者核心代码、笔记及 SHA。
- [Sobolev 来源](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/reading_actual_distribution_feedback_v1/sobolev/code_sources.json)：固定 commit 的原始 URL、文件路径及哈希；论文下载来源见总 sources.json。
- [DF 独立阅读与代码定位](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/reading_actual_distribution_feedback_v1/particle/README_ZH.md)、[其来源清单](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/reading_actual_distribution_feedback_v1/particle/SOURCES.json)。

本轮没有执行作者模型或性能脚本；下载、文本抽取、代码读取和哈希均为 CPU 工作。完整阅读总墙钟及 CPU 时间未统一测量，不虚报；原始已测下载时间保留在来源记录。索引与研究状态由 root 整合，本笔记不自行改变实验准入。
