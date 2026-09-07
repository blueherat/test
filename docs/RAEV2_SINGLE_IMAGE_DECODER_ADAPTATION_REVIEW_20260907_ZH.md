# 真实生成 latent 下的小型 decoder 适配：有界反方评审

日期：2026-09-07。任务边界：提升单图生成质量；固定生成 prior 与采样流程，只评审 decoder 末端小 adapter 的可能性。没有训练、GPU 实验、采样器实现或 Git 操作。此前 [surrogate 采样加速方向](RAEV2_ADJACENT_SURROGATE_REVIEW_20260907_ZH.md) 的 user-rejected 状态保持。

**判断：训练域缺口真实存在，但目前不足以把“真实 q + decoder-only GAN/OT/FD”推荐为一个有新机制的新方法。** 它可能成为有效后训练基线；缓存真实 q 也确实避开了单步监督与完整生成之间的输入分布错位。然而，decoder 对生成误差后训练已有直接图像先例，冻结 prior 后在真实生成 latent 上训练 decoder 也有明确跨模态先例。把 GAN 或 FD 的可训练参数限制在 decoder 末端，并不产生新的分布匹配原理。本轮没有查到与全部限制严格相同的图像论文，也没有建立这种组合的独占新颖性。

目前最有价值的研究问题应进一步收窄为：**官方生成图像的剩余分布误差，有多少能由这个小末端参数空间修复？** 这是决定是否值得训练的机制缺口；“p 与 q 不同”本身还不够。没有证据支持其必然带来 5% FID 改善。

## 1. 六篇原文分别覆盖到哪里

| 原论文与核查位置 | 训练输入和参数 | 对本提议的限制 |
|---|---|---|
| [RAE，§4.3](https://arxiv.org/html/2510.11690v1#S4.SS3) | decoder 输入为真实 encoder latent 加各向同性高斯噪声；噪声尺度随机 | 已明确提出重构域与部署域错配。但其平滑分布不是固定实际 sampler 的终点 q |
| [RobusTok / Image Tokenizer Needs Post-Training，§4.3、A.5–A.6](https://arxiv.org/html/2509.12474v1) | 固定 encoder/quantizer，对 decoder 后训练并复用 discriminator。AR 保留部分真实 tokens；diffusion 用真实图像出发的 SDEdit 维持配对 | **最直接图像先例。** 完整生成 q 与真实图像缺少配对，这个问题论文已明确指出。其 preservation ratio 是实质训练变量；不能把它说成纯噪声增强，也不能把其输入等同于无原图保留的部署 q |
| [Token2Wav，§2.3，式8–9](https://arxiv.org/html/2606.18072v1#S2.SS3) | 从噪声经过实际一步 MeanFlow 得到生成 latent；冻结 MeanFlow，只更新 waveform decoder 与使用到的 discriminator，采用 STFT、对抗和特征匹配损失 | **跨模态已直接覆盖 frozen prior + actual generated latent + decoder-only。** 语义 token/说话人条件比 ImageNet 类别提供更强配对信息，不能把语音效果当成图像证据 |
| [AdvFD，§4.1](https://arxiv.org/html/2608.11205v1#S4.SS1) | 在实际生成图像与真实图像间做静态和可学习表征中的 FD 后训练 | 没有直接验证 RAEv2 decoder-only，但把复合生成器的前半部分冻结，是该目标的参数限制，不是新的目标原理；论文也明确讨论指标投机 |
| [GenFirst，§5.1及作者 Method](https://tom-zgt.github.io/GenFirst/#method)，[原文 PDF](https://arxiv.org/pdf/2608.29335) | 联合训练 VAE 与 prior；后续 diffusion 阶段冻结 VAE、训练新的 prior | 不属于固定 prior 的终点 decoder 适配。不能因其 reconstruction refinement 名字就视为目标协议的直接先例 |
| [ASUKA，§IV-B、IV-C](https://arxiv.org/html/2601.15368v1#S4.SS2) | 利用冻结生成器的一步 clean-latent 估计构造增强数据，保持原图配对，并适配带图像/mask 条件的 decoder | 近邻是生成误差感知的 decoder 适配；作者为降低成本明确采用一步估计，故不是完整部署 q，也不是这里的无配对生成分布学习 |

RobusTok §4.3 还写到生成图像重编码，因此仅凭论文文字不能保证其每条后训练路径直接输入原始生成 latent。以上只主张原文明确支持的 preservation-ratio/SDEdit 配对机制，没有额外推断代码实现。其 A.6 对连续 tokenizer 的实验使用 20 万真实训练图像、10/20 epochs；这不能当作“几百步、不到百万参数”已经被验证的预算。

以上六篇为本轮阅读范围；有限检索未命中严格匹配项不是新颖性证明。未把搜索结果中的二手综述作为机制证据。

## 2. 真实 q 解决了什么，又没有解决什么

记真实图像分布为 \(p_X\)，\(p_Z=E_\#p_X\)，固定采样器为 \(S\)，实际终点分布为 \(q_Z=S_\#p_\epsilon\)。类别条件省略，但实际训练与评估必须保持相同类别混合。原 decoder 为 \(D_0\)，候选为 \(D_\phi\)。目标分布直接是

\[
P_\phi=(D_\phi)_\#q_Z,\qquad
\min_\phi\mathcal D(P_\phi,p_X).
\]

只要 prior、采样器和随机数协议固定，缓存独立完整终点确实提供这个部署分布的样本。对固定可微样本目标且满足交换微分与期望的条件，有

\[
\nabla_\phi\mathbb E_{z\sim q_Z}\ell(D_\phi(z))
=\mathbb E_{z\sim q_Z}J_\phi D_\phi(z)^\top\nabla_x\ell(D_\phi(z)).
\]

不需要经过 S 反传。这个等式说明梯度对应的分布是对的；它不是 GAN 博弈的收敛定理，有限批次 FD/OT 的梯度也不因此成为总体 FD/OT 梯度的无偏估计。

相比之下，真实 latent 加噪声训练使用 \(\widetilde p_Z\)，SDEdit 训练使用 \(q_Z^{\rm edit}\)。两者都可能不同于 \(q_Z\)。即便某种 pairwise loss 在前两种分布下降，也不能直接得出 \(\mathcal D(P_\phi,p_X)\) 下降。这是值得保留的机制理由。

但把 \(G_\phi(\epsilon)=D_\phi(S(\epsilon))\) 视为生成器，上式就是普通生成器后训练，只冻结了 S。GAN 在无限容量和理想优化条件下的目标分布结论，不会证明一个有限末端 adapter 能表示所需修复；OT 选择了一种分布耦合，也没有发现每个生成 latent 的“真实对应图像”。

尤其不能把同类别的任意真实图像 x 当成生成 z 的配对真值：若给定类别后两者独立，平方误差最优解为 \(D(z,c)=\mathbb E[X\mid c]\)，会丢失类别内部多样性。使用无配对分布目标确实避开这个错误，但这也是已有 GAN/OT 的基本用途。

## 3. 为什么末端容量是关键，而非把 mismatch 再解释一遍

**语义保持与分布修复可能冲突。** 设 s 是某个可测图像语义统计；若要求 adapter 对每个生成 latent 保持

\[
s(D_\phi(z))=s(D_0(z))\quad q_Z\text{-几乎处处},
\]

则 \(s_\#P_\phi=s_\#P_0\)，从而由映射下总变差不增加，

\[
\operatorname{TV}(P_\phi,p_X)
\geq\operatorname{TV}(s_\#P_0,s_\#p_X).
\]

因此，如果主要问题是物体属性、布局或模式权重，一个严格保留相应语义的小修复器无法消除这一部分误差。允许改写语义当然可能改善分布，但此时需要相应表达能力与真实数据监督；不能仍把成功或失败都归结为“decoder 理解了 latent”。这只是条件性下界，并未断言 RAEv2 的误差主要落在该部分，也不推出 FID 下界。

**唯一值得保留的窄问题是误差可达性。** 对固定诊断表征 \(\psi\)，设

\[
d=\mathbb E_{p_X}\psi(X)-\mathbb E_{q_Z}\psi(D_0(Z)),\quad
A=\mathbb E_{q_Z}\left[J_\psi(D_0(Z))J_\phi D_0(Z)\right].
\]

在足够小且可验证的参数邻域，均值误差近似为 \(d-A\delta\phi\)。不加约束的线性最小二乘剩余量为

\[
\min_{\delta\phi}\|d-A\delta\phi\|^2
=\|(I-AA^\dagger)d\|^2.
\]

这可以检查目标偏差是否在末端参数的一阶响应空间内。它是熟知的线性化可达性分析，**不是新方法或 FID 理论**；均值只是诊断统计，实际协方差、全局结构、非线性响应与独立样本效果仍可能失败。若只在训练表征可达、在未参与拟合的表征不可达，应该先怀疑指标适配。反过来，小邻域失败不排除更大非线性网络，只会否定当前“小末端修复”的廉价论据。

## 4. 与仓库旧实验的区别和不能忽略的负证据

- [RAEv2 可逆 latent LPL](RAEV2_INVERTIBLE_LATENT_LPL_PILOT_ZH.md)：593,024 参数、500 步；训练来自真实数据加噪后的单步预测，明确未使用生成终点。LPL 风险下降，没有可靠转化为终点 FID。当前 q 分布监督与它不同；但“只有小 adapter，所以不会出问题”的论据已被削弱。
- [decoder inverse 训练代码](../experiments/train_decoder_inverse_adapter.py)：在真实图像编码的 adapted latent 上加人为噪声，学习 inverse 的 latent/reconstruction 目标；不是 actual-q 的无配对分布训练。不能把这个源码的存在等同于新协议已被实验否定。
- [旧 RAE 噪声几何](RAE_DECODER_NOISE_GEOMETRY_RESULTS_ZH.md)：真实生成 latent 与球形噪声造成的 decoder hidden 偏移并不相同；支持“不应以球形扰动代替 q”的谨慎态度。该结果属于旧 RAE，不证明 RAEv2 的末端小参数足以修复。
- [RAEv2 IG pushforward](RAEV2_IG_DECODER_PUSHFORWARD_MECHANISM_ZH.md)：两组 5K 显示方向旋转、各向异性重加权；还否定了“IG 只是抵消 decoder 重构偏差”。不能用 clean reconstruction bias 直接设计 deployment 修正量。
- [AdvFD 严格配对复验](ADVFD_OFFICIAL_PMF_B_10K_RESULTS_ZH.md)：ADM-FID 的额外改善没有跨 reference/held-out 表征成立，三个 held-out FD 都变差。故把训练目标换成 adaptive critic 并未自动解决质量证明。
- [9/7 理论教训](RAEV2_GUIDANCE_THEORY_LESSONS_20260907_ZH.md)与[最终收束](RAEV2_GUIDANCE_FINAL_CLOSEOUT_20260907_ZH.md)：精确 NLL 改善仍可恶化 FID；条件方差方向的大样本结果没有建立实用增益。新方向不能再次用某个正确的代理风险代替最终图像质量。

这些证据支持更直接监督终点，但没有证明终点 decoder 后训练必然有效。

## 5. 小预算到底是否可触及

当前配置的 ViTXL decoder hidden 为 1152，最终 patch readout 输出为 \(16^2\times3=768\)。见[配置](../external/RAEv2/configs/decoder/ViTXL/config.json)与[readout 定义](../external/RAEv2/src/stage1/decoders/decoder.py)。完整末层增量含 bias 共 \(1152\times768+768=885,504\) 参数；例如 rank-16 增量加 bias 为 \(16(1152+768)+768=31,488\) 参数。这些只是容量预算例子，不是已选择的算法或允许搜索的 rank 网格。

仅训练末端时，可以缓存 frozen decoder 的末层输入，避免每个训练 step 重算整个 decoder；缓存并不改变推理的实际费用。如果每图 256 tokens、BF16、hidden=1152，则每图约 0.5625 MiB，8192 图约 4.5 GiB。若同时缓存同规模真实 latent 的 hidden，合计约 9 GiB，不含图像和评价特征。

一个计算上可触及但**本次不启动**的上限示例是：最多 8192 个独立 q 训练样本、512 updates、batch 32，训练暴露 16,384 次；训练数据、原样本索引与最终评价噪声完全分开。若 q 缓存不存在，8192 图的原生 100-step 采样需要 819,200 次主采样查询，IG 分支计算还须照实际协议计入；不能只报 adapter 的训练秒数。critic/可学习表征的参数、前反向与缓存成本也必须单列，不能把“generator 不足 1M”写成“总训练不足 1M”。

readout 的输入经过完整 Transformer，含有全局上下文，因此不能先验断言它只能改颜色；但共享末端映射能否修改需要的全局结构，仍要证据。图像中的 256 patches 也不能当作 256 个独立完整生成样本来虚增分布训练样本量。

若未来有人把此项作为工程基线启动，至少应固定一次训练预算，比较原 decoder、同容量的已有噪声/配对后训练基线与 actual-q 分布训练；实际 q 的独立评价、未参与训练的表征、随机完整样本视觉检查都要保留。不能用挑选 1K FID 最优 checkpoint 的方式决定继续训练多久。当前 5K 补样是冻结的评价工作，不可转为该 adapter 的训练集。

## 6. 决定

**不把这条组合推荐为当前“高质量新想法”的首选，也不启动小 GAN/OT 训练来替代缺失的机制论据。** 它与旧 LPL 和球形噪声训练有明确实验差别，可作为将来证明新方法增量时必须面对的基线；目前能留下的是“末端参数是否能修复真实部署分布误差”的窄机制问题。

若这个可达性问题没有可重复的跨表征信号，或者改动主要改善色彩/纹理统计却不改善完整单图，应该明确停止末端适配路线。若信号存在，也只能说明值得做一次固定预算验证，不能升级为新理论、SOTA 或 5% 改善保证。
