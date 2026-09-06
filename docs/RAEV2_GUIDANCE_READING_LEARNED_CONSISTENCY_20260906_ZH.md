# 从分布一致性学习 guidance：Learn to Guide 与 Adversarial Schedules

日期：2026-09-06。**两篇确实提供了“分布目标→训练信号→廉价推理 guidance”的正向路线，不能因为权重随时间学习就将其归为手工 schedule。** 前作在 ImageNet-64 相对较强 LIG 基线有超过 5% 的 FID 改善；后作放松过强的逐图条件匹配，但主要改善文本对齐和偏好分数，并未胜过主表强基线 FID。当前证据支持吸收机制，尚不足以照搬成 RAEv2 新训练。

## 1. 来源与阅读边界

- Galashov、Pokle、Doucet、Gretton、Delbracio、De Bortoli，**Learn to Guide Your Diffusion Model**，[ICLR2026 正式 PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/d199e76b714c2611051e1b9e4bd882f1-Paper-Conference.pdf)，37 页，SHA `55510f7d652657e591c1b5bf0a7d8e1991215833b4c2d9b1980ed5cb2657b5ab`。读取正文 §§2–6、附录 B–J 的推导、算法、参数、成本、定量结果与权重分析；未逐图评价定性生成图。另存 [arXiv v1](https://arxiv.org/abs/2510.00815v1)，不把两个版本混算。OpenReview 入口返回 403，正式稿由 ICLR proceedings 取得。
- Pokle、Galashov、Doucet、Delbracio、De Bortoli，**Adversarial Learning of Classifier-Free Guidance Schedules**，[arXiv2608.14038v1](https://arxiv.org/abs/2608.14038v1)，2026-08-14 预印本，SHA `4651ece3c345195eec1303c4909c2a4f0dcf31456b27867f85264d7ca860febf`。独立协作阅读正文、附录 B/C 的机制、消融与训练设置，本文再核正文公式和核心设置。详见[独立短审](</home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/reading_learned_consistency_guidance_v1/adversarial/REVIEW_ZH.md>)。

两篇均未找到可确认的官方算法仓库；已核论文、作者页面及准确标题/编号检索，不能断言代码不存在。本次实现核查限于正文与附录伪代码，不称源码复现。旧 [distributional gate 文档 §6](RAEV2_DISTRIBUTIONAL_GATE_GUIDANCE_20260906_ZH.md)已经引用前作的 energy-kernel 思路，本轮是补正式全文精读，不能称首次发现分布损失学习 gate。

## 2. 应当保持的是哪个分布

统一用 RAE 的噪声时间：\(Z_t=(1-t)X+t\epsilon\)，\(s<t\) 表示去噪一步。设 \(K^{\omega}_{s\leftarrow t,c}\) 是实际有限步 guided 核，输入只有当前状态、时刻和条件，**没有原始真实 X**。

**边缘一致性（MC）**要求

\[
p_{t|c}K^{\omega}_{s\leftarrow t,c}=p_{s|c}.
\]

它允许不同真实样本之间重新分配概率质量；目标是下一时刻的正确分布。两文正文先把 c 也积分掉，写 pooled marginal。前作附录 C.2 另讨论固定 c 的 conditional consistency；后作判别器实际输入 c，真/假各配自己的条件时，理想 logit 对应 \((z,c)\) 联合密度比，等价于上述条件边缘比。不能只匹配 pooled marginal 就宣称各类别正确。

前作实际采用的 **self-consistency（SC）**强得多：要求对每个 \((X,c)\)，

\[
q_{t|X}K^{\omega}_{s\leftarrow t,c}=q_{s|X}.
\]

它意味着原图先加较多噪声、再通过不知原图的 guided 核处理，仍恢复这张原图的较低噪声条件分布。若精确成立，积分后确实推出 MC；反向不成立。前作明确承认其过强且难以精确达到，将它作为低方差的近似训练信号。

**独立机制核验：SC 一般不仅是模型容量不足，而有信息论障碍。** 固定 c，取非退化 \(X\sim N(0,\tau^2)\)，则

\[
I(X;Z_t\mid c)=\tfrac12\log\!\left(1+\frac{(1-t)^2\tau^2}{t^2}\right).
\]

噪声减少至 s 后，这个量严格增大。但 \(X\to Z_t\to Z'_s\) 为条件 Markov 链，数据处理不等式不允许增长。前作 Eq.(13) 的核确实不接原 X；允许权重依赖当前状态也不会消除该障碍。真正的反向核能恢复边缘，却不恢复与原始 X 的同一 forward joint。该结论不否定 SC 作为近似损失的实证价值，也不要求 guidance 拥有无条件 FID 定理。

## 3. 有限步核为何不同于均值回归

前作 Eq.(4) 的准确反向核对 \(p(X\mid Z_t,c)\) 积分；Eq.(13) 用 guided clean 预测的 delta 替代该 posterior。即便均值预测完全准确，**有限步更新仍可能缺少分布宽度**，所以 MC 不是换名的 denoising MSE。

在上述标量 Gaussian 中，令 \(V_t=(1-t)^2\tau^2+t^2\)、\(m_t=(1-t)\tau^2/V_t\)，取确定性 DDIM / RAE Euler 的均值替代更新

\[
Y=\frac{s}{t}Z_t+\left(1-\frac{s}{t}\right)\mathbb E[X\mid Z_t]
=k_{\rm mean}Z_t.
\]

全方差公式精确给出

\[
V_s=k_{\rm mean}^2V_t+\left(1-\frac{s}{t}\right)^2
\operatorname{Var}(X\mid Z_t).
\]

均值替代漏掉最后一项；调整有限步核的分布目标因此有具体内容。这不意味着 probability-flow ODE 必须逐状态添加 posterior noise：该 Gaussian 的精确 PFODE 本来就是确定性伸缩，斜率 \(\sqrt{V_s/V_t}\)，与 Euler 的 \(C_{st}/V_t\) 不同，其中 \(C_{st}=(1-s)(1-t)\tau^2+st\)。在 \(\tau^2=1,t=.6,s=.4\) 的固定代数例中，真实前后边缘方差同为 .52，均值更新仅 .4430769，缺 .0769231。恒等映射就能正确匹配这两个边缘，仍不满足逐原图 SC。这是说明目标与离散误差区别的解析例，**不是新性能机制，也不是建议 RAE 使用该映射、方差系数或另做半径校准**。该 Gaussian 矩修复属于旧有限步 moment projection 的特例，不能换名重启。多维非 Gaussian 的缺失 covariance、实际模型误差及现有 Full/Base 方向是否相容，仍需各自证据。[CPU 代数与结果](</home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/reading_learned_consistency_guidance_v1/learn_to_guide/gaussian_consistency_check.json>)没有随机抽样或模型调用。

## 4. 前作：从 SC 到可训练 energy loss

固定真实 \((X,c)\)，独立产生 m 个低噪声目标和 m 个高噪声输入；后者经一次 guided DDIM 跳到同一时刻。训练比较这些条件分布，而非从纯噪声回传完整生成轨迹。去掉真实—真实常量后的损失为

\[
\mathbb E\|Y^\omega-Y_{\rm real}\|^\beta
-\frac{\lambda}{2}\mathbb E\|Y^\omega-Y^{\omega\prime}\|^\beta.
\]

\(0<\beta<2,\lambda=1\) 才是这里的有效 energy-kernel MMD；此时吸引与生成—生成项的比例由分布距离决定。实际图像设置确实用 \(\lambda=1,\beta=1.75,m=4\)，没有单独弱化排斥项。生成粒子之间的经验项去掉对角、分母 \(m(m-1)\)，所有训练粒子参与损失；它们不是推理后多生成再挑图。

权重 \(\omega_\phi(c,s,t)\) 由小 MLP 学习，非负 ReLU 是额外结构约束。时间条件允许一个网络适应不同步长，不需要训练后再手写逐时系数。但有限函数类的最优解依赖训练时间分布、核指数、随机 DDIM 核及优化选择，目标并未唯一固定这些设置。

**与旧 \(X-G\) 的准确区别。** 前作附录 C.1 的 guided score matching 就是旧 posterior-risk 同类目标；若 F 为精确 posterior mean，且 gap 非零，最优额外权重为零。真实有限模型有剩余误差时不必为零，不能把“训练得好”当成严格等号。SC 的完整 energy loss 比该回归多条件输出分布和生成—生成项；其简化 \(\beta=2,\lambda=0\) 也不是一般的 clean MSE。在无 floor、确定性 RF、目标噪声与 proposal 噪声独立时，设 \(a=1-s/t\)，可直接展开为

\[
\mathbb E\|Y-Y_{\rm real}\|^2
=a^2\mathbb E\|G-X\|^2
+2as\mathbb E[(G-X)\cdot\epsilon]+2s^2D.
\]

交叉项通常非零；仅到 \(s=0\) 才退回 clean MSE。因而不能将所有 SC 变体都说成旧 residual 回归；也不能把逐图 SC 改称完整 rollout 分布监督。

## 5. 后作：用判别器匹配 MC，推理只保留 guidance 网络

后作时间方向相反：噪声 0、数据 1。训练先抽真实图文对，加噪得到真实 \(p_s\) 输入，沿冻结 conditional-minus-negative 分支做**一个** guided Euler 跳跃；另抽图文对直接加噪到 t 作真实端。这里的假分布是 \(p_sK^\omega\)，不是当前完整 rollout 实际访问的 \(q_sK^\omega\)。

理想无正则 Bayes 判别器 logit 为 \(d^*=\log[p_t/(p_sK^\omega)]\)，生成器最小化假样本上的 \(-d_\phi\)，以估计 \(KL(p_sK^\omega\Vert p_t)\) 的下降方向。判别器训练时接状态、两时刻与条件，不接原图；因此比较的是边缘分布，不要求原图配对重建。真/假样本共享噪声或图文对，会改变梯度估计的相关性，**不会仅因共享就自动变成逐 X 的 SC**；关键是目标和判别器的条件层次。

生成器是小 MLP，输入文本、时刻和由现有两支预测构造的 log-norm、余弦等统计，输出 softplus 非负权重。推理时删除判别器；新增动作仍限于当前 CFG gap，不输出任意独立场。这是把训练时昂贵的分布差异监督摊入廉价控制器的具体设计。

但部署函数并非纯 KL 目标唯一解：实际另加 \(a t^2\omega^2\) 正则和 CLIP 奖励，XS/S 的 a 分别为 .1/1e−5、奖励系数 .25；\(t^2\) 是作者指定的训练偏好，不是 MC 恒等式推导的时间权重。R1、时间分布及网络统计也是设计选择。原文 Eq.(15) 的负 R1 与 Eq.(12) 负 BCE、Algorithm1 梯度下降的符号不一致；无官方代码时只能记录复现疑点，不能认定作者执行时采用了错误符号。

**MC 的闭环含义是有条件的，但不是空的。** 若推理同一网格、同一条件核精确满足每步 MC，且初始边缘正确，就可逐步推出终点边缘正确。近似时，回到本文 §2 的时间记号，在 TV 度量下由 Markov 核非扩张得到 \(e_s\le e_t+\varepsilon_{s,t}\)；W₂ 则还需核的稳定性系数。训练采用宽跳跃而推理用小步、critic 有限容量、只优化有限时间分布时，不能自动得到这些全网格误差界。这一目标也不同于旧[完整 rollout 终点 energy gate](RAEV2_DISTRIBUTIONAL_GATE_GUIDANCE_20260906_ZH.md)：后者直接监督实际整条轨迹的特征分布，两文主要监督真实 bridge 输入上的局部核。

## 6. 强基线实证与成本

| 来源与设置 | 原文结果及准确含义 |
|---|---|
| Learn，ImageNet-64，50K，同 100 步配置 | constant 2.40、LIG 2.11、clamp-linear 2.24、SC 1.99；相对最佳所列 LIG 降 **5.6872%**。底座为自训约 260M U-Net，不是 DiT/RAE。 |
| Learn，CelebA-64，50K | LIG 2.37→SC 2.10，降 **11.3924%**；保留这项正面生成结果。 |
| Learn，MMDiT T2I，10K、128 Euler | SC FID18.01/CLIP.295；加奖励28.37/.306。主表 LIG24.95 是偏对齐选出的设置，附录另有 LIG **18.89/.2979**；SC 相对这个更低 FID 仅降 **4.6585%**，且对齐较低，不能宣称全面胜出。 |
| Learn，SD1.5，10K | constant FID23.479/CLIP.313→SC+reward19.36/.311；存在明显 FID 收益，但没有该组最优 LIG/完整同总成本比较。 |
| Adversarial，MMDiT-XS，128 Euler | constant/LIG/GAN+MC 的 FID29.73/25.57/**31.31**；GAN 的 CLIP/Aesthetic/HPSv2 是主表最高，PickScore 按显示精度持平。 |
| Adversarial，MMDiT-S，128 Euler | constant/LIG/GAN+MC 的 FID31.11/24.95/**31.70**；同样主要赢对齐/偏好分数，未赢 FID。 |

前作实际成本不能只用 MLP 参数量描述：ImageNet/CelebA 每个训练 run 为 **100K iterations、B256、m4**；每个 noisy proposal 都要查 frozen 条件/无条件 denoiser。它不需穿过完整 rollout 的大模型状态 VJP，确有计算结构优势，但查询并不免费。训练扫描 δ 与 Smin 各四值，用 2048 图 FID 选 checkpoint；SC 训练用随机 DDIM \(\epsilon=1\)，推理用确定性 \(\epsilon=0\)。故已有最优 1.99 的证据伴随具体训练分布/核和选择成本，不能称完全无调参。β、m 的附录消融相对稳定，是可保留的正面信息。

Learn 的 T2I 为50K iterations、B256、m4；Adversarial 为 D/G 各60K iterations、B128，并以3000图 CLIP 选 checkpoint。后作还需 critic 的前反向、R1及奖励链；两文都没有完整硬件墙钟、所有选择成本和可据以计算的摊销阈值。推理模型调用结构接近普通 CFG，小 MLP 可摊销；LIG 区间外若跳过负支，还可能更便宜，实际代码未确认。**同 128/100 步与相同 backbone 不能代替当前 T/W 公平成本协议。** 不因训练非零而否定方向，也不把预训练大模型远贵于 guidance 训练当作免计训练的理由。

## 7. 对 RAEv2 的具体取舍

可保留的机制是：**先识别实际有限步 kernel 与目标条件边缘的差异，再在允许的 guidance 方向中学习修复函数。** 它允许系数由数据和分布目标决定，满足用户允许理论结构/学习机制、禁止大量手调 schedule 的区别。前作的 posterior-方差替代解释还说明，即便均值回归已很好，有限步分布误差也可能存在。

尚缺的条件是：当前同类 Full/Base 的可达方向是否能修复这种误差；训练核能否忠实采用实际 RAE 网格/floor/算术；类条件统计是否有足够数据与判别力；teacher 局部改进如何在闭环保留；训练与选择如何按固定预算完成。不能仅用通用 KL/MMD 目标给旧2049参数 affine gate补一个新名字；也不能要求先有普适 FID 定理，才承认该方向值得机制研究。本轮未确定新结构、训练配置、权重、窗口或实验臂。

本笔记与来源归档于 [reading_learned_consistency_guidance_v1](</home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/reading_learned_consistency_guidance_v1/manifest.json>)。只有下载、文本读取与一个固定 Gaussian 代数核验；零 GPU、训练、采样和新 FID。已测下载/归档/代数成本分别保存；未统一计时的阅读成本留空，未把父进程 CPU 当作转文本子进程 CPU。
