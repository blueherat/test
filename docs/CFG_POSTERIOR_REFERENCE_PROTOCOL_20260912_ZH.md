# 用 conditional teacher 训练 null 参考：后验平均的损失

2026-09-12。在 frozen-strong 预测/生成分布实验之后执行。这是用户持续研究中的纯 CFG 分支，独立于 IG 成功与否；不更改前三轮协议或停止规则。

## 从概率关系到训练目标

固定噪声时间，X,C 来自真实配对训练数据，Z=tX+(1-t)ε，Y=X-ε，conditional strong 为 S(Z,t,C)。候选 null 读出 U 只接受原 null 主干特征 H_null(Z,t)，不接受真实类别。

\[
 L_{\rm paired}(U)=\mathbb E_{X,C,\epsilon,t}
 \|U(H_\emptyset(Z,t))-\operatorname{sg}S(Z,t,C)\|^2.
\]

无限容量下，其解是 E[S(Z,t,C)|Z,t]。给定 Z 后，训练数据中的 C 自然按 p(C|Z,t) 分布，不是按先验 p(C) 分布；无需显式求类别后验，也无需遍历100类。若 S=E[Y|Z,t,C] 精确，则该目标的解由 tower property 等于 E[Y|Z,t]，即原 null FM 目标。更一般地，对任何只依赖 Z,t 的 U，

\[
 \mathbb E\|U-Y\|^2=
 \mathbb E\|U-S\|^2+\mathbb E\|Y-S\|^2
\]

在精确 teacher 下成立，目标是已有 Rao–Blackwell / 条件期望结构。teacher 有偏时不再无偏：它继承 E[e_C|Z,t]，可能避免 CFG 额外放大类别共享偏差，但仍不纠正它。真实冻结特征读出有限；上述关系不保证有限预算或生成质量改善。它也不是 null 预测自身的自蒸馏，后者在当前初始化下没有更新。

关键替代对照是 teacher 使用独立类别 C'~p(C)，同时保持 Z、null 特征、初始化及训练预算完全相同。其总体目标变为 E_{C'~p(C)}S(Z,t,C')。若 S 是精确 score 对应的速度，前者对应算术混合 p(Z)=Σ_c p(c)p(Z|c)，后者的 score 对应几何平均 q_geo(Z)∝∏_c p(Z|c)^{p(c)}。二者一般不相同。例如两个等方差高斯类别：算术混合保留两个类中心，几何平均集中在中心之间。独立标签不是后验平均的无偏估计。

[ICG](https://arxiv.org/html/2407.02687v2) 已在推理时随机替换条件来近似 null；本对照不声称完整复现或否定 ICG，其 embedding 选择、随机采样和有限网络行为都不同。[Unconditional Priors Matter](https://arxiv.org/html/2503.20240v2) 已指出更好的 null 可能改善 CFG，而不是 null 越弱越好。[DASH](https://arxiv.org/html/2606.00798v1) 已做条件与无条件分支分别蒸馏；其 null target 是教师的 null 输出，本轮 target 是配对条件教师输出、student 看不到标签。这些最近邻限制新颖性表述；条件期望恒等式不是新理论。

## 固定质量检验

只训练 cfg_posterior 与 cfg_independent 两个原尺寸 null final readout。复用上一轮的真实每类20个固定 clean latents，合计2000；不新增合成数据。模型仍 SiT-S/2 800K EMA、strong 完全冻结。1500 steps、B32、AdamW lr1e-4、betas(.9,.999)、wd0、clip1、EMA.995、最终 EMA；时间(.01,.75)，其余随机流 seed2026120913 与已有 cfg_real 完全相同，不用验证选 checkpoint。

teacher 的独立类别由另一 CUDA Generator(seed2026121216)产生；两组每步都绘制相同的独立标签，paired 组不使用它作 target，保证它不改变原始 data/noise/time/label 随机流。null 输入从不接收真实或打乱类别；只有离线 conditional teacher 的条件不同。定期保存 Z、null特征、conditioning 的配对哈希，并记录教师类别。strong 权重及梯度检查与前三轮相同。

正式各1K共7组：original CFG、同预算 real/FM 读出、posterior teacher、independent teacher、前轮 CFG 自身生成数据读出、原 CFG extra强度固定减半(.625)、APG。全部重新生成，历史探索 bank202610100、Heun64、窗口.75；除明确半强度对照外 extra1.25，APG沿用既有配置。所有 CFG 仍224 full / 0 prefix，训练后推理不额外调用 teacher，不加主干或 adapter，不扫强度。

唯一候选是 posterior；independent 是替代目标对照，不允许事后转为主候选。候选须比 original、real/FM、CFG自身数据、半强度、APG 均至少低 .5 FID，才冻结进入新5K；否则停止本构造，不改数据规模、步数或权重。若 paired 胜过 independent 但未胜过强基线，只说明两种 loss 行为不同，不能认领质量进展或机制证明。

准备前冻结源码、协议、数据、模型与复用读出身份。生产前验证解析混合公式、全方法短轨迹、零引导、原生采样复现、strong不变、无新增prefix。正式质量完成后核对每组全部raw batch覆盖/哈希，并以缓存特征的独立FP64公式复算FID/sFID。成本分离训练、采样/解码与评估；重复探索 bank 不作独立确认。旧大队列、独立 weak prefix、已取消迁移仍保持停止。
