# 分类器方法研究入口

当前方向是冻结强模型，以最终生成图像的真假分类反馈训练弱头。
2026-09-19 开始先整理工作区，再优化显存和训练速度；尚未启动新的方法质量搜索。

## 已有最好结果

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
