# 先修改解释概率再聚合的guidance实验

核心假设是：生成模型的一部分质量损失来自多种解释被平均，直接外推平均值又可能把结果推到解释集合以外。若模型内部已有与这种歧义有关的概率表示，就可以先改变解释的相对概率，再让模型完成聚合与解码。这一轮只检验一个具体实现：条件Transformer在每层对token对应关系的概率对比。

这是一个纯CFG方向的候选，编号64。它不使用外部语义模型、不训练局部价值函数、不分解最终guidance向量，也不进行层或head选择。IG方向的概率读出仍存在未解决的信息缺口，不把本实验称为IG方法。

## 从平均值外推到概率对比

设一个预测由共同候选值v_j与概率p_j构成，m_p=Σp_jv_j，负参照的概率为q_j。普通平均值外推对应m_p+α(m_p−m_q)。这些组合权重之和虽然为1，却可能为负，因此可以离开共同候选值的凸包。

若希望强化正参照相对于负参照偏好的解释，可以定义

\[
r_j=\frac{p_j^{1+\alpha}q_j^{-\alpha}}{\sum_kp_k^{1+\alpha}q_k^{-\alpha}},\qquad m_r=\sum_jr_jv_j.
\]

对于严格正的p、q，它是最小化KL(r∥p)−αE_r log(p/q)的唯一解。这里的概率公式、KL变分解释及凸包性质都是既有数学，仓库9月10日的[概率组合分析](IG_HYPOTHESIS_POSTERIOR_RETHINK_20260910_ZH.md)也已经推导过，不能作为新理论再记一次。

关键区别有可检查的反例。两个候选值为−1、1，正概率为(.25,.75)，负概率为(.5,.5)。在α=2时，均值外推得到1.5，概率对比后再平均约为.92857。这个例子说明两种操作不同，以及概率操作如何避免这类越界；它没有证明连续图像模型的错误一定来自相同原因。

在α=0处，概率方法的切向量为p_j[log(p_j/q_j)−E_p log(p/q)]，也一般不等于p_j−q_j。因此这不是把原guidance换一个标量强度，也不要求两个误差向量共线。

## 可直接运行的计算对象

对于某层当前hidden state H，保留条件分支的value、残差门、MLP以及最终输出层。仅对attention的对应关系概率进行处理。用同一个H分别施加条件c与null的AdaLN调制，得到两组query/key logits L_c(H)、L_u(H)，定义

\[
A_g(H)=\operatorname{softmax}\big(L_c(H)+\alpha[L_c(H)-L_u(H)]\big).
\]

随后使用A_gV_c。每层继续更新一个hidden state，不在最终速度上附加CFG。所有12层统一应用同一操作，α固定为1，沿用原CFG的t<.75时间区间。这个单位强度表示一次额外对数概率比对比，不声称它由真实独立观测次数或误差比例估计得到。

需要特别区分：L_u(H)是相同当前hidden state上的局部null调制，不是从输入开始完整运行null分支的结果。因此A_c、A_u是模型计算中明确归一化的权重，但目前没有把它们认证为图像模式的精确后验。value也不是完整图像候选。这个代理是否有用是实验的中心，不以“attention天然就是Bayes后验”代替验证。

有限value凸包性质只发生在该层聚合中。后续残差、MLP和解码可以改变幅度，因而不存在由此直接推出的最终图像有界、无过饱和或FID保证。

## 为什么值得做这一个小实验

DoLa在语言模型的共同词表上对比深浅层logits，提供了一个有实际生成用途的先例；它依赖真正的词表概率，不能直接搬到只有均值读出的IG。[^dola] 本候选沿着这一信息缺口寻找已存在的归一化计算对象，而不是从两个最终均值虚构一个完整后验。

最近的注意力引导工作必须作为近邻。HeadHunter/SoftPAG已经研究attention扰动和包括log-linear在内的概率插值，不能认领“在attention概率空间运算”本身。[^heads] UNCAGE使用attention对比来安排离散token的解掩码顺序，操作对象和本候选不同，但同样限制宽泛的“对比注意力引导”新颖性主张。[^uncage] SAG通过attention选择区域后构造模糊负参照，再外推网络输出，也已有明确方法。[^sag]

本次可能的差异只在完整设置：相同hidden state、条件与null对应概率、共享条件value、直接传播引导后的单个hidden state，以及对“先聚合还是先对比”的生成实验。检索未证明该设置没有先例；方法收益和相对近邻的实质差异都要另行建立。

## 九组固定实验

沿用旧control配对1K噪声202610100及标签202610101、100类均衡、B8、四卡、64步Heun。模型、VAE、ADM reference与旧队列一致。没有新增训练。

|组别|改动|回答的问题|
|---|---|---|
|原strong|原入口|无额外guidance的起点|
|原局部IG|原α=.8、locality=2|入口复现与已有IG参照|
|原CFG|原额外强度1.25|本路线必须面对的强基线|
|原APG|原额外强度2、beta=−.5|已有改进CFG的更强基线|
|概率候选|softmax(2L_c−L_u)，α=1|先修改概率是否有用|
|线性attention对照|2A_c−A_u，使用相同V_c|收益是否来自保持非负概率的非线性操作|
|温度对照|softmax(2L_c)|条件null对比是否必要|
|embedding对照|e_c+(e_c−e_u)，完整原网络|是否只是通用条件增强|
|手工attention核对照|原A_c，使用候选相同矩阵计算入口|排除SDPA与显式softmax路径的数值差异|

九组中只有一个新候选，不进行强度、层数、head或截止时间网格。如果不超过CFG/APG，就不加密α寻找最低点；若某个对照解释了结果，应据此修改或放弃对象假设。出现非有限轨迹时保留失败，不以部分样本计算质量指标。

若候选优于CFG但未超过APG，只能作为中间线索；若超过APG，需要新噪声确认与计算可比的比较。只有候选优于线性attention、温度及embedding等对应替代解释，才支持概率操作本身有增量价值。即使如此，也不直接证明attention等于真实图像模式后验。

## 计算与执行

候选每图128次主网络前向；前96次各有12次额外QKV和logit乘法。记录这些额外操作及全部采样、解码时间，不能用相同主前向计数宣称等成本。没有Hutchinson/JVP、前瞻质量评价或在线拟合。

CPU只核对有限概率公式、两候选反例、切向量和不变量。正式采样前，四卡检查全部九组完整短批轨迹、零强度退化、四个旧基线逐元素复现、原生输出在候选之后不变。保留手工核的实际数值误差。原模型中输出语义以FieldSemantics.prediction_target=velocity核对，并检查实际张量形状。

此阶段等待五候选59组完成并审计，随后预检和采样。结束后进入研究复盘；旧宽队列仍暂停。训练或质量结论不由执行器自动产生。根目录为 `/home/zhoushunyu/data/eqvae/experiments/sit_posterior_guidance_20260912`。

## 来源

[^dola]: Yung-Sung Chuang et al. [DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models](https://arxiv.org/abs/2309.03883), ICLR 2024，arXiv v2，2024-03-11。引用其共同词表上的深浅层logit对比，不移用语言任务效果。
[^heads]: Donghoon Ahn et al. [Where and How to Perturb: On the Design of Perturbation Guidance in Diffusion and Flow Models](https://github.com/cvlab-kaist/HeadHunter), NeurIPS 2025，作者实现与方法说明，含SoftPAG的概率插值方式。
[^uncage]: Wonjun Kang et al. [UNCAGE: Contrastive Attention Guidance for Masked Generative Transformers in Text-to-Image Generation](https://arxiv.org/html/2508.05399v1), 2025-08-07，方法节。
[^sag]: Susung Hong et al. [Improving Sample Quality of Diffusion Models Using Self-Attention Guidance](https://openaccess.thecvf.com/content/ICCV2023/papers/Hong_Improving_Sample_Quality_of_Diffusion_Models_Using_Self-Attention_Guidance_ICCV_2023_paper.pdf), ICCV 2023。
