# 一页勘察：用 ODE-copy 配对训练末端残差修复器

2026-09-13；只读仓库与 SIGN/IGN 原文，未下载、未训练、未运行 GPU。

**可实现；在本次检索范围内，未发现已经训练并验证过的 SiT `P(x)≈x, P(C(x))≈x` 同构实验。** 今天的 [FM fixed-point 笔记](fm_fixedpoint_construction.md)已经提出近图邻域 P，但只是方案。不能把本次最小适配称为全新的“幂等生成”范式。

| 已有代码 / 记录 | 可复用之处与不相同之处 |
|---|---|
| [TinyVelocityUNet](../../../experiments/mnist_spectral_rollout_toy.py:224) | 两尺度 U-Net、残差块、末层零初始化；原为单通道、带时间输入，可私有复制为 4 通道、无时间的 residual U-Net。 |
| [CouplingNet](../../../experiments/latent_equiv_adapter.py:107) | 可单独借用零输出三层 CNN，`channels=4, hidden_channels=64` 共 8,836 参数，作为廉价容量下界。**不要套可逆 coupling 容器**：严格可逆且幂等的 P 只能是恒等映射。 |
| [decoder inverse adapter](../../../experiments/train_decoder_inverse_adapter.py:44)；[已完成结果](../../RESEARCH_STATUS.md:2094) | 冻结 RAE decoder，在 8K 真实图 latent 加人为噪声后训练。重建 L1 `.17418→.16324`，配对 5K FID **`19.1944→19.7137` 变差**。这是必须保留的负证据；它没有用同图 ODE-copy 退化配对。 |
| [small-image residual adapter](../../../experiments/small_image_residual_adapter.py:73)；[RAEv2 common adapter](../../../experiments/raev2_common_adapter.py:126) | 前者加在每步速度，后者观察 noisy state/time/full/base，每步修正预测；都不是只看最终 latent 的 P。[FashionMNIST gate](../../MECHANISM_TO_QUALITY_STUDY_ZH.md)和 [RAEv2 LPL pilot](../../RAEV2_INVERTIBLE_LATENT_LPL_PILOT_ZH.md)均未建立稳定生成收益。 |

**论文对齐。** [IGN §2、§4](https://arxiv.org/html/2311.01462v1)训练承担生成任务的 autoencoder，使真实图重建、输出幂等，并加 tightness；不是给冻结 diffusion 接小后处理器。[SIGN §3、Appendix C](https://arxiv.org/html/2509.21470v1)从 diffusion 蒸馏无显式时间的生成学生，采用与 teacher 相同架构并初始化其权重；还使用 teacher flow、生成配对等目标。两文都没有直接实证“保留完整 SiT 采样，再接轻量 P 修复”的本题配置；SIGN 的蒸馏数据和生成目标也不能省略后照搬其效果。

**最小实现建议。** 私有文件定义 `P_phi(z)=z+R_phi(z)`；优先借 TinyVelocityUNet 的两尺度结构，宽度 32、4→4 通道、末层零初始化，移除时间分支；先不加类别、注意力或 critic。冻结 SiT/VAE，离线保存真实 train 图的 `(source_id, c, x, C_j(x))`。`C_j` 必须是事先固定、同类别的逆-copy 操作，例如共享参考 `I_R` 后用 strong/weak 生成；同图的逆结果可复用，不能拿同一模型精确自往返的零残差当修复信号。模型、方向、步数和精化结果随配对保存。

\[
L=\operatorname{MSE}(P_\phi(C_j(x)),x)
  +\lambda\operatorname{MSE}(P_\phi(x),x),\qquad \lambda=1\ \text{作为首个固定配置}.
\]

首轮不加幂等损失：先验证能保留真实图并修复未见过的配对，再检查重复 P 是否继续漂移。所有同源图及多种 C_j 输出按 source ID 整组分区；已有 16 图 validation pilot 保留为诊断，不转成训练数据。训练只读缓存，无需每个 update 穿过 SiT/VAE；**主要成本是离线构造 C，不是小网络训练**。

部署为新噪声生成 `y=G(z,c)` 后只调用一次 `P(y)`，再统一 decode。成立的窄目标是“学习消除指定 copy 操作的误差”；能否修复实际 from-prior 生成仍是额外迁移问题。最小判据同时报告：未见真实图被改动多少、配对修复误差、独立生成端点修复前后质量与多样性；不以配对 MSE 或幂等残差下降宣布生成收益。
