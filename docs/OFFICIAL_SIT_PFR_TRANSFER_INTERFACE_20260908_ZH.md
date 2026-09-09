# 官方 ImageNet-1K SiT-XL：补齐 PFR 迁移对照的接口检查

现有 PFR 正结果主要来自 ImageNet-100 SiT-S/2 和冻结主干后训练的
depth4 readout；RAEv2 则是 ImageNet-1K、联合训练的 Full/Base。
两者还有模型规模、latent、时间网格、输出参数化等同时变化。
因此不能把迁移失败单独归因于 RAE 表征。

仓库已有官方 SiT-XL/2+IG 的 800 epoch 权重，1000 类、depth8 联合
弱头、4×32×32 SD-VAE latent。历史实验验证过它的 IG 扰动传播，
当前检索没有找到该权重上的 PFR 质量结果。它能减少类别数与弱头
训练方式的混杂，但不能单独识别所有剩余因素。

本轮先做接口检查，代码 audit_official_sit_pfr_interface.py：
seed202609427、8 个噪声、标签0..7、B4，官方100步线性递减网格，
FP32模型/无TF32、FP64状态。显式Full路径与官方euler_sampler终点
逐位比较；在四个时刻比较完整前向弱头与depth8提前退出弱头。
同时运行普通IG和time-only PFR路径，scale1.35、h1/32、noise-time>.5，
这个系数沿用小SiT迁移对照，不称作官方最优配置。

只保存latents和调用/源码/权重身份，无decoder、训练或FID。
预期800次B4 Full（包含官方对照）和124次B4 prefix（包含检查），
不把调用类型当作等成本。接口通过后仍需要实际测量成本和质量，
且任何收益都首先是现有方法的迁移证据，不是新的核心idea。
当前会话40374运行中；论文写作保持暂停。
