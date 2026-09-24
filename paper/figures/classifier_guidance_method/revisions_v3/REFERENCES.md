# 视觉参考

以下参考均查看了论文实际方法图；采用其排版方式，图中的模块和训练目标仍以本项目实现为准。

| 论文及具体图 | 观察到的表达方式 | v3 中的应用 |
|---|---|---|
| [AdvFD，Figure 3](https://arxiv.org/html/2608.11205v1#S4.F3) | 双栏解释两类更新，冻结／训练标识，底部分布与边界示意 | A 保留双栏和说明性分布；三版保留雪花、火焰 |
| [Improved Distribution Matching Distillation for Fast Image Synthesis（DMD2），Figure 3](https://arxiv.org/html/2405.14867v1#S4.F3) | 简化的连接块状网络轮廓、水平数据流、不同颜色的反馈路径 | 用稀疏连接块表示 Strong／Weak／D，橙色虚线表示反传 |
| [Adversarial Diffusion Distillation（ADD），Figure 2](https://arxiv.org/html/2311.17042v1#S2.F2) | 区分样本、网络和损失模块，并明确截断梯度的位置 | B 区分前向流程与目标框，C 明示判别器更新中的 detach |
| [Diffusion Adversarial Post-Training for One-Step Video Generation（APT），Figure 1](https://arxiv.org/html/2501.08316v2#S3.F1) | 统一无衬线文字、浅色模块、规整连线，并把局部网络细节单独组织 | 统一字体家族与粗细层级，减淡填色，减少小网络线条和视觉装饰 |

DMD2、ADD 属于包含对抗目标的扩散蒸馏，APT 是对抗后训练参考。本项目图没有据此增加蒸馏教师、单步生成假设或扩散骨干判别器；当前判别器仍读取最终 RGB 的冻结 Inception 特征。

选择 Liberation Sans 是本次统一排版的实现决定，并不声称上述论文使用了该具体字体。

方法核对来源：[`sit_joint.py`](../../../../classifier_guidance/sit_joint.py)、[`schedules.py`](../../../../classifier_guidance/schedules.py)、[`binary_critic.py`](../../../../experiments/adversarial_weak_training_20260915/binary_critic.py)。
