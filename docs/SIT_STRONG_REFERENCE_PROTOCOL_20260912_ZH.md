# 冻结 strong 后，weak 究竟学习什么

本轮响应用户的两个假设：weak 学习冻结 strong 的分布可能形成兼容的误差；weak 的损失函数需要专门研究。前两轮已完成且不改动：直接 guided-risk 训练未超过同预算续训；IG/CFG 各自生成数据的参考训练偏好自身来源，但未通过推进阈值。这里不再延长原训练或重调强度。

## 解释及其可失败之处

固定噪声时间和条件，真实平均速度为 u，strong 为 S=u+e。把 weak 的有限函数族理想化为 L2 中的闭线性子空间，P 是对它的正交投影。真实目标训练的总体最优解为 W_N=P u；teacher 预测训练为 W_D=P S。两者同用 G=S+a(S-W)，则

\[
 P(G_N-u)=(1+a)P e,\qquad P(G_D-u)=P e,
\]
\[
 \|G_N-u\|_{L^2}^2-\|G_D-u\|_{L^2}^2
 =a(a+2)\|Pe\|_{L^2}^2\ge0.
\]

含义是：在这些条件下，teacher 训练让 weak 继承 strong 能被这个读出表达的偏差，避免这部分偏差被额外放大；并未消除它。无需假设全部强弱误差共线。这是标准投影代数，不是新定理。真实 AdaLN 与 linear 联合训练跨时间共享参数，不是任意闭线性函数族；有限训练也不保证达到总体最优，真实数据状态上的风险更不保证 FID。P 只用于解释，不在采样中做分量分解。

另一边，令 q_S 为冻结 strong 无 guidance 采样的终点分布。用它的样本训练 weak 的原始 FM loss，学习的是重新加噪后的 q_S 对应速度 u_{q_S}；直接 teacher loss 学习 S。有限学习的 S 一般不等于 u_{q_S}，即使 S 是光滑保守场。简单例子：S(z,t)=kz，从标准高斯出发，其终点方差为 exp(2k)，但该终点经线性加噪后的 FM 速度系数是 [t exp(2k)-(1-t)]/[t² exp(2k)+(1-t)²]，一般不是 k。若 strong 是完全自洽的精确 FM，两种总体目标才一致。

若 unrestricted weak 精确拟合 S，IG 项归零。因此 teacher loss 变好不是越拟合越好；容量约束和失配内容才可能重要。固定半强度原 IG 用于排查一种简单替代解释，但一个半强度点不能排除全部时间强度曲线。

仓库 09-09 的 [predictable-gap](SMALL_SIT_PREDICTABLE_GAP_PROTOCOL_20260909_ZH.md) 已用浅层条件特征回归 S-W；raw 等价于拟合 teacher 的一种受限读出，当时 raw FID 没有改善，带额外范数调整才有小幅信号。本轮完整原尺寸 readout 的 teacher loss 是不同容量/训练方式下的对照复测，不能包装为首次提出蒸馏 weak。[AG](https://arxiv.org/html/2406.02507v1) 已讨论兼容退化；[SIMS](https://arxiv.org/html/2408.16333v1) 与 [SSG](https://arxiv.org/html/2607.29122v1) 已覆盖合成数据训练负参考。这里要检验的是预测目标和生成分布目标的区别，不宣称这些操作本身新颖。

## 冻结实验

原 SiT-S/2 800K EMA 冻结。只增加一份 strong 无 guidance 的 2000 个 clean latents，Heun64、每类20个；与前轮 IG/CFG 生成训练数据使用相同新噪声 seed2026121014 和类别。保留连续 latent，不解码再编码，不筛选样本。真实对照复用前轮固定每类20个真实 latent，训练/验证图像身份互斥。所有训练噪声独立于 FID bank。

IG 为 2×2 对照：训练 clean 来源 real 或 strong-generated；目标为 X-ε 或 stopgrad(S(z,t,c))。real/FM 复用前轮同预算 ig_real 最终权重，另训 ig_teacher、ig_strong、ig_teacherstrong。CFG 另训 cfg_strong：null head 学习 strong 条件样本的类混合、使用原 FM loss；real 和 CFG 自身数据训练头作为已有对照。CFG null 不具备真实类别输入，不把 IG 的逐条件 projection 推导直接搬到它上面。也不做原 null head 自蒸馏：其初始化已经等于冻结 null teacher，该目标会退化为零更新。

四个新读出全部 1500 steps、batch32、AdamW lr1e-4、betas(.9,.999)、weight_decay0、clip1、EMA.995，seed2026120913，最终 EMA，不选 checkpoint。IG 原 depth4_v 读出，CFG 原尺寸 final head，仅 null 使用新头。IG 时间(.01,.5)，CFG(.01,.75)，IG label dropout .1。与前轮逐线同一索引、噪声、时间和标签随机流。两个相同数据来源的 IG loss 使用完全相同 z、特征和 strong。strong 梯度为空且训练前后权重哈希相同。验证用独立真实 validation pool，报告而不选权重。

正式配对1K共14组：IG original / real / teacher / strong / teacherstrong / IG-data / 固定半强度 / ADG，CFG original / real / strong / CFG-data / APG，以及 strong 无引导。除半强度对照外均原定 IG .8、CFG extra1.25、时间窗不变、Heun64；不扫参数。全部重新采样，使用历史探索 bank202610100，不称独立确认。IG 和无引导为128 full / 0 prefix；CFG 为224 full / 0 prefix。不添加独立模型查询、弱前缀或推理时 teacher。单独报告实际采样及解码时间、离线合成数据及训练成本。

推进规则：IG 三候选若同时比 original、real、IG-data、固定半强度、ADG 至少低 .5 FID，选其中最低者冻结进独立5K；CFG strong 若同时比 original、real、CFG-data、APG 至少低 .5 才进5K。此为探索推进规则，不是显著性阈值。否则停止本构造，不延长训练、不扩大强度网格。每条线通过后仅确认固定候选和强对照，未通过不启动迁移。

正式1K之前核对损失目标及梯度、strong 不变、readout 输出通道、零引导回到原轨迹、四个旧采样基线精确复现、所有方法完成短批轨迹且不增加 prefix。训练数据逐样本覆盖/类别/原始哈希核对。结果全量原始 batch 哈希与覆盖、缓存特征 FP64 FID/sFID 独立公式复算。当前旧大队列与取消的独立前缀均保持停止。
