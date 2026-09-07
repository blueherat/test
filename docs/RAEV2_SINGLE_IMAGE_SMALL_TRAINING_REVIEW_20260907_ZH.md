# RAEv2 单图质量：极小后训练的独立复核

日期：2026-09-07。**这轮没有找到足以推荐进入训练的新切口。** 对 head-only、归一化、表示对齐和 EMA 四类可能性，已有论文给出了有用机制，但尚未找到“RAEv2 的实际缺陷—极小参数可触及—生成质量改善”的完整证据链。不把训练参数少、局部目标下降或工程上能实现当作立项理由。

本轮只读文献、源码及已有结果；没有 GPU、训练、新采样或冻结代码修改。用户已拒绝的多速率/缓存加速方向保持关闭；正在执行的四个旧候选 1K→5K 补测保持原协议。这里的目标始终是生成图像质量，不是速度、新任务或多图筛选。

## 实际模型与历史结果限制了什么

当前官方 DDT 为 28 层 encoder、2 层 Full decoder、depth8 Base，hidden 为 1440/2048，输出 1024 通道。它直接预测 clean latent，已有原生双头训练及 IG。参数/模块依据 [`DDT.py`](../external/RAEv2/src/stage2/models/DDT.py)、[`model_utils.py`](../external/RAEv2/src/stage2/models/model_utils.py) 与 [官方训练配置](../external/RAEv2/configs/stage2/training/imagenet-dinov3l-k7.yaml)。

必须采用 [9/7 最新收束](RAEV2_GUIDANCE_FINAL_CLOSEOUT_20260907_ZH.md)的结论：九个完成的 5K 候选中最好改善约 0.564%，未达该阶段 3% 目标。[精确反例](RAEV2_GUIDANCE_THEORY_LESSONS_20260907_ZH.md)又说明，最终分布 KL 下降也可能伴随 FID 上升。因此小模型后训练不能用配对风险的保证代替用户要求的公平成本质量收益。

| 已完成的相关研究 | 实际证据与限制 |
|---|---|
| [严格 RAEv2 LPL 续训](RAEV2_LPL_STRICT_CONTINUATION_ZH.md) | 恢复完整训练态、有效 batch1024、修复并核对 GMuon 内部 LR。IG=1 时，配对 5K 的 Flow100 为 11.6942、LPL100 为 12.4807；完整主场本身仍会受损，不能仅归咎两头差值。这个旧协议与 9/7 的 FID 数字不能跨表比较。 |
| 同一续训中的归一化控制 | raw 与 prediction-detach 大幅消除 LPL 退化，但旧 5K Flow10=10.827822、raw=10.841450、detach=10.851805，仍没有净收益；归一化分母不是尚未检查的空白。 |
| [函数级 contrast-preserving adapter](RAEV2_SELF_GUIDANCE_RESEARCH_LEDGER_ZH.md) | 全部 875,216,500 个源模型参数冻结，只训 268,480 参数的共同残差。三个 5K seed 的平均差约 −0.01，远小于该历史协议约 0.12 的 seed 波动。小 adapter 加感知监督已做过，并非缺少实现。 |
| [旧 SiT 冻结 clean head](IMAGENET100_SIT_FROZEN_V_CLEAN_HEAD_RESULTS_ZH.md) | 6,160 参数也训练了 50K 步；只改输出参数化时，clean-derived velocity 在数据端显著恶化。它否定该具体弱头及外推构造，不能推成所有 RAEv2 原生 head 微调都失败；同时说明少参数不等于少训练。 |
| [LPL 路线结题](LPL_LINE_CLOSURE_AND_SOLID_RESEARCH_AGENDA_ZH.md) | 旧 RAE 的正结果保留；成熟 RAEv2 已出现单图 feature 统计与跨样本 covariance 的分叉。不得沿用早期“局部 decoder metric 重分配解释收益”的已被后续否定叙事。 |

## 四篇一手论文及适用范围

只选与问题直接相关的四篇，阅读原文对应段落；没有以论文总数替代判断。

| 论文 | 本次实际阅读 | 对极小后训练的约束 |
|---|---|---|
| [LPL，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/204fee94c982a19230c39045aa54f977-Paper-Conference.pdf) | §3.2 decoder feature 目标与实验训练阶段 | 感知目标是明确先例；原文主实验有长预训练及 120K/200K 后续训练，不能直接借用其收益预测本项目短训效果。本仓库已经进行了更直接的模型内检验。 |
| [REPA，ICLR 2025](https://arxiv.org/html/2410.06940v4) | §3.2–3.3，训练设置、表2 | 通过可训练投影对齐 clean 视觉表示，同时改变 diffusion 主干；主表是数十万训练步。学习一个辅助投影头并不是独立的生成质量修补。 |
| [REPA-E，ICCV 2025，v3](https://arxiv.org/html/2504.10483v3) | §3–4，表8、附录A/B | tokenizer 与 diffusion 共同学习，diffusion loss 对 tokenizer stop-gradient，另有重构正则。表8的“可替换 tokenizer”仍重新训练 diffusion 400K 步，不是直接放进冻结 prior 就能得到收益。 |
| [EDM2，CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Karras_Analyzing_and_Improving_the_Training_Dynamics_of_Diffusion_Models_CVPR_2024_paper.pdf) | 幅值保持设计、§3 后验 EMA 合成 | 幅值控制改变训练动态；post-hoc EMA 依赖训练中预存的多种平均快照，并用质量评估选择平均长度。不能从一个终点 checkpoint 任意恢复这些历史。 |

## Head-only：函数可改，瓶颈未被识别

冻结前缀后，输出头可看成 \(\widehat x=Wu+b\)。用配对 clean latent 重估 W 是标准回归；它在指定特征类与指定训练分布上最小化风险。它并不说明部署轨迹上的 \(u\) 同分布，也不说明改后原 IG 场的总体误差会下降。过去 LPL 与共同 adapter 已经表明，这个缺口具有实际影响。

要把它升格为一个机制，需要先有原生输出头的可避免误差证据：同一 frozen feature 上存在不依靠换参数化、换 guidance 的稳定修正，并且错误集中在该头确实能够表达的方向。当前档案没有这项 RAEv2 原生头的独立证据。深度与读出交叉实验显示的是未经训练的 head 互换失配，不证明原生最后一层欠拟合。

即便有这样的回归缺口，它首先也是普通 head 微调的理由；若只以 MSE/LPL 作为目标，它没有区别于旧失败的新质量机制。本文不提出新的 head 容量、rank 或损失系数扫描。

## 归一化：存在小参数集合，但没有“统计过期”证据

本地 RMSNorm 为

\[
 \operatorname{RMSNorm}_w(h)
 =w\odot h/\sqrt{\operatorname{mean}(h^2)+\varepsilon}.
\]

它按当前 token 计算，没有 BatchNorm 的 running mean/variance。给它重新累计“训练集统计”没有对应接口。Stage1 的外部 latent normalization 是冻结坐标定义，修改时还要同时处理 prior、初始噪声与 decoder；不能当作无害 BN 校准。

静态数目上，28 层 encoder 的两个主体 RMSNorm 与 Q/K RMSNorm 共
\(28(2\cdot1440+2\cdot72)=84,672\) 参数；两层 Full decoder 共
\(2(2\cdot2048+2\cdot128)=8,704\)；两个最终读出 norm 共 3,488。合计 **96,864**，约为源模型参数的 **0.0111%**。这是源码结构计数，不是实际训练性能测量。

只训这组参数确实可保留原推理结构。但梯度仍需经过被其影响的后续深层计算，不能把 0.0111% 当作训练算量占比。Q/K norm 参数还会改变 attention logits 的相对尺度、方向及熵，属于模型函数修改。当前没有证据表明 RAEv2 的生成错误由这些 norm 的失配导致。

EDM2 说明幅值与更新平衡值得研究，但它采用系统性训练设计，不保证在已有 DDT 上改几处归一化即可改善。若一个尺度改变被下一线性层完全补偿，则只是函数不变的重参数化；若不补偿，则质量方向需要另证。本文不建议据此开启 norm-only 微调。

## 表示对齐：辅助指标可以改善而生成函数完全不动

REPA 原文中的投影 \(P_\phi(h_\theta)\) 是训练辅助分支。如果 \(\theta\) 与所有生成路径冻结，只训 \(\phi\)，推理又丢弃该辅助分支，则

\[
 \partial G/\partial\phi=0,
 \qquad (G_\phi)_\#\mathcal N=(G_0)_\#\mathcal N.
\]

此时对齐损失下降不能改变任何图像。若让 norm/LoRA 改变主干以打通影响路径，后续冻结层的适配与训练成本重新成为问题；换一个更轻参数集合并不能自行解决。

更一般地，取目标 \(X\sim\mathcal N(0,I_d)\)，生成器 \(G_a(z)=az\)，a>0。对每个非零 z，\(\cos(az,z)=1\)，但相对目标的 Gaussian FID 为 \(d(a-1)^2\)。这个解析例仅说明 cosine 对齐不控制输出尺度与分布，不能用完美对齐推导 FID。实际 REPA 联合 diffusion loss 的实证价值不受此例否定；这里否定的是把对齐分数单独当质量保证。

本地配置未开启额外 REPA，并不等于缺少 clean 表示训练目标：RAEv2 本身就在 DINOv3 聚合表示上训练 clean prediction。它是否仍有一个可由极小参数修复的语义不足，需要模型内证据。原论文在 SD-VAE/SiT 上的表示差距不能替代这项诊断。REPA-E 又改变 tokenizer 与 prior 的共同坐标关系，其预算和冻结边界均不匹配本任务。

## EMA：可估计的对象与保存下来的信息不同

官方配置的 EMA decay 为 0.9995；旧严格续训已经核对 online/EMA，150 步新更新进入 EMA 的总权重只有 \(1-.9995^{150}\approx7.23\%\)。这解释了短训 EMA 改变小，但不是“重新挑 EMA 就会好”的证据。旧 online LPL 的明确退化也排除了仅由 EMA 淹没收益的单因解释。

当前官方模型目录只有一个 `checkpoint.pt`；旧续训快照存在，但来自不同分支目标，不是官方训练期间的多个 EMA basis。尚未发现 EDM2 所需的官方历史快照集合。

设已存平均为 \(\bar\theta=\sum_k a_k\theta_k\)，还知道终点 \(\theta_T\)。对任意更改历史 \(\delta\theta_k\) 满足 \(\sum_k a_k\delta\theta_k=0\)、\(\delta\theta_T=0\)，现有文件都相同；但另一平均 \(\sum_k b_k\theta_k\) 通常不同。因此仅靠现有终点及单一 EMA，目标平均一般不可识别。混合 online/EMA 可以定义新模型，却不能声称重建了某个未保存的最优 EMA；通过 FID 扫多个混合比例也不符合当前少调参的要求。

## 当前裁决

没有足够证据把上述任何一种作为新方法推荐。局部 head 风险、norm 幅值、表示 cosine、参数平均都可以是诊断对象，但目前没有一个被识别为这台成熟 RAEv2 的实际质量瓶颈。最有用的结论是收紧准入条件：**先识别具体函数缺陷及可触及它的参数，再讨论训练；不可反过来从“这个参数集合很小”构造质量故事。**

这不构成对所有小训练方法的否定。它明确排除本轮几种缺少证据的直接迁移，避免继续重复旧的 paired proxy→FID 推断。若其他方向找到直接针对真实生成分布、同时有可验证质量机制的切口，可独立评审；本文不自动触发新的诊断或训练。

原文捕获与 SHA256 阅读范围：`/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/single_image_small_training_review_v1/reading_manifest.json`。四篇原文已保存；大模型 checkpoint 未加载，参数计数来自源码，没有新增质量结果。
