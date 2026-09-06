# 官方 IG 上的固定类别信息补充

2026-09-07。新3%目标内的固定两臂设计，尚无本候选FID。旧 `semantic_seed202609061` 仅完成ordinary/combined各56张，因旧用户范围修订停止，未算FID。这里保留旧档案，采用新的原生算术和完整配对协议，不拼接旧样本、不把部分预览视作正向证据。

此前固定方法大多只处理同类Full/Base差或全局latent统计。新的问题是：类别与空条件的差是否能补充IG尚未利用的方向。相关基础是 [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)；[Autoguidance §5.2 / B.2](https://arxiv.org/html/2406.02507v2)已有CFG与模型质量引导的线性组合先例。后者的可视化组合不证明本地RAEv2超越强IG，这里不宣称组合本身新颖。

源码中 `ConditionEmbedder` 有1001个类别embedding，`get_null_cond` 使用标签1000；本地配置及官方训练入口使用conditional dropout .1。实际空条件查询通过原checkpoint的这一接口进行，不以随机类别或平均embedding冒充unconditional。

记同一状态上的conditional Full/Base为F、B，null Full为U，D=F−B，C=F−U。保留官方native BF16

    G_native = B + 1.78 (F−B).

在z=(1−t)X+tε、0<t<1下，若Full条件/空条件是自洽联合分布的精确score，(1−t)C/t²等于类别log posterior的输入梯度。有限网络只近似这个身份；D也不被预设为保守的质量梯度。

固定两个同成本臂：

    semantic_add:        G_new = G_native + .15 C
    semantic_orthogonal: G_new = G_native + .15 (C − Proj_D C).

第二式是逐图严格凹局部问题

    max_u  <C,u> − ||u||²/(2×.15),  subject to <D,u>=0

的唯一解。D=0时约束为空，恢复.15C。它保留新增修正在原IG方向上的零分量，并保持类别方向的非负一阶内积：<C,u>=.15||C_perp||²。它不保证全程类别posterior上升、保留全部图像质量，也不证明FID下降。若orthogonal不优于additive，不以几何形式声称结构增益。

**唯一新增全局系数.15直接沿用旧未完成协议的温和CFG1.15预设。** 这不是理论唯一决定的常数，也不是论文给RAEv2的推荐值；本次只用它一次，不扫描幅度，不用FID估计它。理论决定类别差与去重叠结构，不把经验幅度包装成定理。两臂都沿用当前官方IG的[.1,1]活动区间，不保留旧手工t>.5窗口，不增加独立窗口。原生F/B的B8 conditional forward保持不变，另调用一次B8 null forward；已有G不重新用FP32重组，新增差/投影/加法在FP32。

每图100个conditional forward、99个null forward、一次最终decoder，预计推理成本约1.99倍，以实测为准。所有NFE均计入，`extra_sample_unconditional_calls`是`sample_model_calls`的子集，不能重复相加。无新增训练、无额外模型checkpoint、无外部分类器反传。

检查顺序：两个针对性解析/原生anchor测试；CPU上固定8图×10时刻的非冗余与有限性诊断；当前two_mode 5K结束后若仍未过3%，应用事先准备的精确补丁，运行official/两候选8图smoke，复核original official8逐像素不变，再完整paired seed202609071 1K。所有结果保留，不选择时间/图像。若有可靠信号，冻结候选进入独立seed确认及必要成本对照；若失败，停止这两个固定设置，不靠参数扫描救分。

源码与补丁：`experiments/raev2_semantic_complement.py`、`experiments/locks/raev2_semantic_complement_20260907`。当前two_mode采样源码在其完成核验前保持不变。
