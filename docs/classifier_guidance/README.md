# 分类器方法研究入口

导航：[代码与检查](../../classifier_guidance/README.md) · [理论与审计目录](#理论与审计目录) · [训练图表与固定快照](figures/README.md)。以下实验状态按各条目日期记录。

2026-09-23：[JiT / SiT 的中间 checkpoint FID-5K 评估](GAN_CHECKPOINT_FID_20260923_ZH.md)已完成，每模型抽 3K、6K、最新存档点，raw/EMA 各 5K，共12组。[FID PNG](figures/gan_checkpoint_fid_20260923.png)。两模型在本轮抽取的三个点中均持续改善，最好为最新点：JiT 10.5K raw/EMA 为12.0285/12.0144，相同1-block全程初始常数基线16.3054；SiT 9.3K raw/EMA 为27.4263/27.4136，初始全程a=.75基线33.2634。已于北京时间14:18从暂停的完整训练状态恢复，权重、两个优化器及数据随机状态恢复检查通过。继续到累计30K次更新；JiT为GPU0+2，SiT为GPU1，batch32/microbatch8、每300步存档。

2026-09-23：[SiT 联合训练弱 provider 与全程有符号系数](SIT_JOINT_PROVIDER_SCALE_20260922_ZH.md)：1-block 50K EMA 初始化，全程a=.75；有效batch32。单卡step1455保存后接GPU1+3限时双卡，最迟北京时间9月23日09:00前释放GPU3并回GPU1续训；08:57开始安全保存。实际配对检查通过，含共享全局probe的非线性尺度约束。实时PNG：[独立系数曲线](figures/sit_joint_schedule.png) · [系数及尺度](figures/sit_joint_guidance.png) · [GAN健康](figures/sit_joint_gan_health.png)。5K质量结果见上方checkpoint评估。

JiT 实时 PNG（约每30秒更新）：[系数曲线：raw、EMA及训练历史](figures/jit_gan_schedule.png) · [GAN 训练诊断](figures/jit_gan_health.png)。图片直接保存在仓库中，可用编辑器的图片预览打开。

2026-09-22：[JiT 1-block 的全程 GAN 系数学习与性能检查](JIT_BLOCK1_GAN_SCHEDULE_20260922_ZH.md)：固定强弱模型，从全程 w=1.50（额外 a=.50）初始化，50区间保留原Heun/Euler求解器。GPU0+GPU2，全局batch32、每卡microbatch8累积2份；真实模型配对完整更新由12.93秒降至6.52秒（1.98×）。已启动累计30K次系数更新的训练，约6.7秒/更新、每卡reserved4.39GiB。每300步保存完整checkpoint，每30秒刷新raw/EMA系数曲线与GAN诊断；5K质量结果见上方checkpoint评估。

2026-09-22：[SiT 原生 Transformer 弱头对照](SIT_TRANSFORMER_CAPACITY_20260921_ZH.md)已完成四组 50K 与 107 组 5K。普通 FM 训练后，前半程引导的 fresh shallow/原生输出头/1-block/2-block 最好 FID 为 36.8451/40.4591/30.9767/29.0626；2-block 最好点在 a=1.95 扫描边界。弱头自身误差和独立生成也随原生 0/1/2-block 改善；该对照不含 GAN 后训练，上方新增实验开始训练1-block弱头与系数。

2026-09-21：[JiT真实数据SSG容量对照](JIT_SSG_REAL_CAPACITY_20260920_ZH.md)已完成三头50K及38组5K。无CFG基线31.6918，线性/1-block/2-block最好FID分别29.7378/16.3054/13.9338；弱头自身MSE与独立生成也随这三档改善。带CFG的固定系数收益较小。

2026-09-20：已核对[终点 GAN 学习系数的相关论文](ENDPOINT_GAN_COEFFICIENT_RELATED_WORK_20260920_ZH.md)。[SiT两档MLP的50K训练、全部5K系数扫描及弱头能力检查](MLP_CAPACITY_DIFFUSION_20260920_ZH.md)均已完成：增强MLP改善弱头自身MSE与独立生成，但未超过旧浅MLP的引导FID。随后按用户要求推进[JiT第6层SSG结构、仅真实数据的三档容量对照](JIT_SSG_REAL_CAPACITY_20260920_ZH.md)。

当前方向是冻结强模型，以最终生成图像的真假分类反馈训练弱头。
2026-09-19 已完成工作区整理与训练优化，详见[性能、显存和正确性报告](PERFORMANCE_20260919_ZH.md)。截至当日尚未启动新的方法质量搜索，后续进展见上方日期条目。
随后完成[理论与完整离散梯度审计](THEORY_AUDIT_20260919_ZH.md)，以及[进一步优化、批量吞吐和数值重复性对照](REFINEMENT_20260919_ZH.md)。
关于判别器应看图像还是 latent、训练与采样系数为何可能不同，见[判别空间与系数自洽分析](DISCRIMINATOR_SCALE_THEORY_20260919_ZH.md)，包含实现核对、固定点的条件与反例，以及 CPU 解析检查。
关于是否固定外推系数，以及时间/状态 gate、一般向量修正的实际自由度，见[可学习修正场分析](GUIDANCE_FREEDOM_20260919_ZH.md)。
SiT 弱 MLP 增加一层、固定系数 1.05 的实现及短程测速，见[加层实验](DEEPER_SIT_MLP_20260919_ZH.md)。
冻结最初 depth4 IG、用 GAN 学习全程正负系数，见[有符号时间系数实验](NATIVE_IG_SIGNED_SCHEDULE_20260919_ZH.md)。30K 训练及 10 个 checkpoint 的 5K 评测已完成；相对本次前半 0.6/后半 0 初始化，最终 FID-5K 为 41.3450 → 38.8507。
该结果未超过下表已有的 36.444090；已核对双方 5K 噪声、标签与 ADM 评测一致，主要方法差异是原生弱头冻结仅学系数，与此前训练 MLP 弱头。

## 理论与审计目录

| 主题 | 报告 | 脚本与紧凑证据 |
|---|---|---|
| 判别空间、系数自洽与可学习修正场 | [判别器与尺度](DISCRIMINATOR_SCALE_THEORY_20260919_ZH.md)、[引导自由度](GUIDANCE_FREEDOM_20260919_ZH.md) | [解析检查](../../experiments/classifier_guidance_theory_20260919/README.md) |
| 终点 GAN 学习采样系数的相关工作 | [论文核对](ENDPOINT_GAN_COEFFICIENT_RELATED_WORK_20260920_ZH.md) | 报告内来源与实现对照 |
| 参考动力学、分布不变量与状态系数 | [研究综合](RESEARCH_BREAKTHROUGH_SYNTHESIS_20260922_ZH.md)、[深入理论](SELF_GUIDANCE_DEEP_THEORY_20260922_ZH.md) | [gauge 与理论证据](../research/self_guidance_breakthrough_20260922/) |
| RAM、局部密度响应与概率流 | [RAM 核对](RAM_INTERNAL_GUIDANCE_REVIEW_20260923_ZH.md)、[局部响应与流](LOCAL_RESPONSE_AND_FLUX_THEORY_20260923_ZH.md) | [RAM / Riesz / flux 证据](../research/self_guidance_ram_20260923/) |
| RL、伴随与反传成本 | [RL 与反传](SELF_GUIDANCE_RL_AND_BACKWARD_20260923_ZH.md) | [伴随审计与推导](../research/self_guidance_rl_20260923/) |
| 有符号时间曲线与控制解释 | [曲线形状](SIGNED_SCHEDULE_SHAPE_THEORY_20260923_ZH.md)、[二阶控制与滑模](SECOND_ORDER_AND_SLIDING_CONTROL_20260923_ZH.md) | [曲线快照与控制证据](../research/self_guidance_schedule_shape_20260923/) |
| 最新末端回落、源码哈希与离散反传 | [末端与反传核验](TAIL_ROLLOFF_BACKWARD_AUDIT_20260923_ZH.md) | [反传审计](../research/self_guidance_schedule_shape_20260923/latest_tail_backward_cpu_audit.json) |

理论脚本及输出对应关系见[复现目录](../../experiments/theory_self_guidance_20260922/README.md)。
这些解析和小模型检查的适用边界以各报告为准；生成质量结论使用对应的 FID 评估报告。

## 早期 MLP 对抗头结果

以下是原 MLP 容量下的历史对照。本轮普通 FM 训练的 Transformer 头已有更低 FID，见上方新实验；不能将跨架构差异直接归因于 GAN 目标。

SiT-S/2、ImageNet-100、64 步 Heun、同一固定 5K 噪声与标签：

| 方法 | 外推额外系数 a | FID-5K |
|---|---:|---:|
| RGB 二分类 GAN 续训 v2，step 1456，raw head | 1.05 | 36.444090 |
| RGB 二分类 GAN v1，step 656，raw head | 1.05 | 36.506244 |
| 未做对抗训练的 guided_weak 50K 头 | 1.05 | 36.688848 |

系数约定：`S + a f(t) (S-W)`，a 是额外系数；各模型的时间窗和 f(t) 必须显式记录。
这次为 8 个已训练头、各 8 个相邻间距 .05 的系数，每点 5K，共 64 点，包含 7 个已有结果。
5K 用于选参，不是独立验证；不据约 .245 FID 的差异声称显著胜出。

## 阅读顺序

1. [二分类方法](../ADVERSARIAL_WEAK_BINARY_GAN_20260915_ZH.md)与[真实 RGB 数据版本](../ADVERSARIAL_WEAK_BINARY_RGB_20260915_ZH.md)。
2. [代码审查](../ADVERSARIAL_WEAK_CODE_REVIEW_20260915_ZH.md)和[端点反传](../ADVERSARIAL_GUIDANCE_ENDPOINT_20260915_ZH.md)。
3. [完整 5K 结果](../ADVERSARIAL_HEADS_5K_NEIGHBORHOOD_20260916_ZH.md)及[较早头部结果](../GUIDANCE_COMPLETE_RESULTS_20260915_ZH.md)。
4. [代码与运行入口](../../classifier_guidance/README.md)。

其他四来源 GAN、energy、矩匹配、SG/log-SG、IG 读出实验保留为历史对照；不混入当前二分类主线。
历史报告里的“运行中”只代表写作当时状态，以本入口和具体完成记录为准。
