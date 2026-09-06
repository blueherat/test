# 强弱 × 条件双差：推理纠偏先例的有限检索

日期：2026-09-06。**找到同型四路差分用于蒸馏损失与校准的明确先例，未找到它直接作为四路推理 guidance 的明确先例。** 这不是新颖性证明，也不是采样准入。最接近的两篇分别是 DASH 的分支可辨识性分析，以及 FBG 的显式密度误差模型。FBG 的加性机制已被仓库旧审计引用，本轮补充正式正文、理论附录和作者实现精读，不能称作新发现。本笔记暂不改变当前 34 篇总表。

## 四路恒等式究竟说明什么

设同一状态、同一噪声时刻有两模型的条件和无条件 **score**，记

\[
D_s=(s_{F,c}-s_{B,c})-(s_{F,u}-s_{B,u}).
\]

如果每个模型的条件密度和无条件密度来自同一个自洽联合分布，且四个 score 精确，则 Bayes 公式给出

\[
D_s=\nabla_x\log\frac{p_F(c\mid x)}{p_B(c\mid x)}.
\]

仅有“四个精确密度 score”还不够：每个无条件密度须确实是相应条件族按类别先验边缘化的结果。类别先验可以不同，因其对 x 的梯度为零。真实网络输出也不自动构成精确、保守且相互相容的 score。

RAEv2 本地输出是 clean prediction；沿 \(Z_t=(1-t)X+t\epsilon\)，\(0<t<1\)，相应换算是

\[
s_{a,c}=\frac{(1-t)F_{a,c}-z}{t^2},\qquad
D_s=\frac{1-t}{t^2}D_{\rm clean}.
\]

因此 clean 双差不能不带单位因子直接称为 posterior score。现有同类共享 Full/Base 只给出 \(F_c,B_c\)，并未自动给出两路合法无条件预测；还须核对 checkpoint 的空条件训练与标签接口。

这个算子准确删除的是**模型强弱差异中的无条件边缘对比**。若 \(s_{a,c}=s^*_c+e_{a,c}\)、\(s_{a,u}=s^*_u+e_{a,u}\)，则 \(D_s=(e_{F,c}-e_{B,c})-(e_{F,u}-e_{B,u})\)。与条件无关的共同误差被消去，剩下“模型强弱与条件的交互误差”；它没有自动等于 \(-e_{F,c}\)。符号组合由双差确定，但有效纠偏方向、幅度及是否丢掉有益的边缘质量修正尚未确定。

例如在固定时刻把 \(\eta D_s\) 加到 \(s_{F,c}\)，若积分存在，只得到局部密度

\[
\widetilde p_t(x\mid c)\propto p_{F,t}(x\mid c)
\left[\frac{p_{F,t}(c\mid x)}{p_{B,t}(c\mid x)}\right]^\eta.
\]

这既未证明更接近真实分布，也未证明各 t 的乘积密度来自同一 clean 分布的加噪路径。“posterior 对比”比笼统的 density sharpening 更准确，但仍是算子身份，尚不是质量机制。这里不选择 \(\eta\)，不据此提出参数扫描或新采样实验。

## DASH：同型双差有实证用途，但用于训练

Al Shafi 等，[*DASH: Dual-Branch Score Distillation for Guidance-Calibrated Compact Diffusion Models*](https://arxiv.org/abs/2606.00798v2)，2026-08-29 修订预印本，30 页。阅读正文机制、Table1、附录 A 的设置与成本、TableA6、附录 C 的可辨识性证明及 RemarkC.5，并核查作者训练、损失、采样和校准代码。

其 gap-matching 基线明确使用

\[
\mathcal L_{\rm gap}=E\left[\omega_t\|\Delta_S-\Delta_T\|^2\right],
\qquad \Delta_j=\epsilon_j^c-\epsilon_j^u.
\]

令强/弱对应教师/学生，残差与目标四路双差只差整体符号，平方相同。**TableA6 的用途是训练损失；没有把该有符号残差加入采样器。** 教师/学生也不能仅凭容量顺序就认作满足 autoguidance 误差关系的两模型。

论文的正向机制是先辨认监督目标看不到的误差方向，再选择可以识别两支的监督。只匹配合成 CFG 输出，允许两支错误按 guidance 系数相互抵消；只匹配 gap，允许两支一起漂移 \(\delta_c=\delta_u\)。DASH 因而分别监督条件与无条件输出，另加真实噪声 anchor，并转移教师学到的时间权重。附录 C 的平坦方向严格针对固定输入的**输出空间**，不是共享参数网络必有同维参数平坦方向的证明。

Table1 的 CIFAR-10 / CIFAR-100 FID：gap-match 为 11.42 / 18.46，DASH 为 8.87 / 10.47；gap MSE 从 .063 / .086 降至 .028 / .040。所有方法使用 50 步 DDIM。教师 FID 为 5.47 / 6.80，不能把压缩学生相对其他学生的收益写成超越强教师。该结果支持“校准需区分支路误差”，未证明双差推理纠偏。

RemarkC.5 的最优标量 \(a^*=E[\Delta_S^\top\Delta_T]/E\|\Delta_S\|^2\) 需要成对二阶矩，不能由平均范数比和平均 cosine 恢复，也不能旋转方向误差。论文实际还有 TAG \((w_{\min},w_{\max},\kappa)=(1,4,5)\)、anchor .1（从五个值选择）及学习后的时间权重；这些不符合当前无需大量调参的直接推理方案要求。

每训练条目需要两次教师、两次学生前向及学生反传；推理仅两路学生 CFG，50 步即 100 次学生预测。TableA3 单 T4、batch64 的教师→DASH 延迟为 14217.3→3925.8 ms，来自 35.8M→6.1M 模型压缩，不能用作四路推理的成本证据。[作者代码](https://github.com/C-loud-Nine/DASH_Dual-Branch-Score-Distillation/tree/682e2fdad149109b053fafa34c797820252c6345)中 `evaluate.py:227–245` 确实计算四路校准残差；`diffusion/ddim.py:78–89` 只执行学生两路 CFG。所查训练入口未找到 TableA6 gap-match 实现，部分默认训练配置与论文表格不同，实证配置身份尚不完整。详细定位见归档的 `search_support/dash_short_review_ZH.md`。

## FBG：补读已有机制，区分理想路径与实际反馈

Koulischer 等，[*Feedback Guidance of Diffusion Models*，NeurIPS 2025 正式全文](https://papers.neurips.cc/paper_files/paper/2025/file/73c90c0c16fe835bc58a815d4a8f28be-Paper-Conference.pdf)，重点读 §3–4、附录 A/B/C 的推导、F 的配置与 H 的资源，并对照作者 sampler。它是**条件/无条件两路**方法，不是强弱与条件的四路双差。

其值得保留的条件性结论是：如果 clean 条件模型满足固定加性污染

\[
q_{c,0}=\pi p^*_{c,0}+(1-\pi)p^*_{u,0},\quad 0<\pi\le1,
\]

且无条件模型准确、各分布使用相同线性加噪算子，则相同等式保持于全部噪声时刻。记 \(r_t=q_{c,t}/p^*_{u,t}\)，目标 score 为

\[
s^*_{c,t}=s^*_{u,t}+\frac{r_t}{r_t-(1-\pi)}(s_{q,c,t}-s^*_{u,t}).
\]

在目标正密度区域，分母为正；有限比值和精确 score 等条件下，理想公式恢复真实 conditional path，而不只是任意时刻的乘积密度。噪声极限 \(r=1\) 时系数为 \(1/\pi\)，大比值时趋向 1。附录 B 的加法与共同 noising 可交换，是这个错误模型能导出整条路径的原因；它不自动扩展到 FBG+CFG/LIG 混合方案。

**这不是本轮新增的 RAEv2 机制。** [已有密度去污染审计](RAEV2_DENSITY_DECONTAMINATION_AUDIT_20260906_ZH.md)已借鉴 FBG v2，并从官方 \(w=1.78\) 直接假设 Full 含 43.8202% Base 污染。两个 transport 探针在 797 个共同可测条目中给出 82 个共同 covariance 必要违例；该固定 Full/Base 假说及估计实现未通过准入。数据不能区分假说错误与比值估计错误，也没有 FID 结果。它不否证原论文的条件/无条件特例，却足以阻止把本次文献阅读当作重跑旧比例、密度积分或调 temperature/clamp 的理由。

正式原文与代码深化了理想公式到实际实现之间的距离。Eq9 累积的是反向路径条件似然；Eq10 用条件/无条件 Gaussian transition 对实际 successor 的平方误差差估计比值。只有兼容的共同前向联合分布及精确反向核等条件，路径 posterior 才能化成当前状态 posterior；任意学到的核、guided 路径和确定性 ODE 不自动满足此点。作者 §3.4 明确讨论用自己的预测评估自己造成的自我偏好，实际引入温度、偏置，并重参数化为手选的两个时间位置 \(t_0,t_1\)，还设最大 guidance。

附录 F 明确：先手调选 \(\pi\) 范围，再联合搜索 \(t_0,t_0-t_1\)，64 步时网格分辨率 1/64，\(\pi\in\{.999,.9999,.99999\}\)；T2I 的时间分辨率 .05，再对 \(\pi\) 做对数扫描。表 3 报 \(\pi=.999\)，邻近正文称重点扫 .9999，不能把两者默认为同一配置。所谓反馈并未消除手选时间形状。

[作者实现](https://github.com/FelixKoulischer/Feedback-Guidance-of-Diffusion-Models/blob/cec69babd78c44b72e644cc422e6e5f076bbf9e1/generate_images_FBG.py#L103)进一步确认：每次 denoise 固定调用条件、无条件两个网络；累积量虽名为 `log_posterior`，以 0 初始化并允许最大值 3，按公式应理解为归一化 likelihood ratio 的对数，不能当作数值范围合法的类别概率。代码按 Eq8 算 gain，使用 log-ratio clamp；最大 gain 的 CLI 默认值为 2，函数默认和 README 示例为 10，并非理论确定常数。\(t_0,t_1\) 转温度/偏置还使用固定参考 gain 3 和经验误差规模 10。

| 设置 | CFG FID | LIG FID | FBGpure FID |
|---|---:|---:|---:|
| EDM2-XS，ImageNet512，50K，随机采样 | 5.00 | 3.59 | 3.76 |
| 同模型，Heun PFODE | 2.97 | 2.31 | 2.50 |
| SD2，COCO 3K | 19.64 | 18.81 | 18.63 |

这是正向结果：FBG 对 CFG 有显著收益，T2I 也略优于 LIG；但 EDM2 两种 sampler 的 FID 均未胜过 LIG。FID 与 FDDinoV2 分别选最优参数。不能只以 CFG 作参照，就宣称当前所需的同成本最佳基线提升。

官方代码 64 个随机步对应 128 次网络预测；README 要求 Heun 外步减半，32 步对应 63 次双支 denoise、126 次网络预测，最后一步跳过校正。其 LIG 即使区间外也计算无条件支，因此并未利用所有可能的省算。比值反馈本身无额外模型前向，但多配置搜索和比值状态仍须计入实际评估成本。附录 H 给出每配置随机 50K 在 V100 上约 7.5 小时、Heun 50K 在 4090 上约 8 小时、SD2 3K 在 V100 上约 2 小时；硬件不同，不能把这些时间当成方法间直接加速比，也未报告总搜索成本。

## 对当前研究的用途与边界

两篇共同带来的具体启发是：**先分清对比误差与共同误差，再给出待修复误差在整条 noise path 上的生成机制。** DASH 说明仅保留条件 gap 可漏掉共同漂移；FBG 说明一个具体的 clean 分布误差律可以导出可采样的理想路径。前者没有推出四路推理，后者的具体固定 Full/Base 迁移已有未通过准入记录。

现阶段只接受双差的有条件代数身份，不据它接受新的 guidance 实现。所缺的不是给式子取名，而是：真实 Full/Base 的条件交互误差为什么应沿此方向纠正；合法的无条件支是否存在；整条路径是否相容；以及在原有共享 F/B 之外，第二个条件状态查询的成本是否值得。两组条件各自或可共享 F/B，但不能默认跨条件复用完整 encoder，也不能把“四个输出”直接当成“四个独立网络成本”。这些文献不提供无需校准的方向增益或 RAEv2 的 FID 保证。

有限检索还筛掉 AG 附录的三路组合、替换 unconditional prior 的两路方法、prompt 双差和本地 depth×query 双差；它们都不等于这里的 strong/weak×conditional/unconditional 四路推理。未将这些摘要筛选计入精读。本轮没有模型执行、GPU、密度积分、训练、采样或新实验。

来源 PDF/正文、作者代码 commit、旧审计快照、检索范围及 SHA 见 [归档 manifest](</home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/reading_guidance_double_difference_v1/manifest.json>)。FBG 下载加转文本实测 wall 7.291798 秒、下载进程 CPU 0.486609 秒（不含转文本子进程 CPU）；其他阅读、检索及源码获取未统一计时，不重构总成本。
