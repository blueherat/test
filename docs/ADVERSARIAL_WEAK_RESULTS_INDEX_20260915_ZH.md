# 端点弱头实验：完整评估结果

更新时间：2026-09-15T16:34:50.650685+00:00。本轮训练、完整扫描、两组5K与同图CPU复核全部完成，已按用户要求停止后续任务。

比较基准：[旧最佳guided-weak 50K EMA](/home/zhoushunyu/data/eqvae/experiments/guidance_dynamic_50k_20260915/sit_small/points/guided_weak__c0042/n5000/metrics.json)，系数1.05，5K FID **36.68884780**。

| 版本 | 最好5K对应系数 | 对应1K FID | 5K FID | 与旧最佳之差 | 记录 |
| --- | ---: | ---: | ---: | ---: | --- |
| 分布矩目标（此前版本） | 1.175 | 63.608062 | 37.300448 | +0.611601 | [endpoint_moments_v1](/home/zhoushunyu/data/eqvae/experiments/adversarial_weak_training_20260915/endpoint_moments_v1/search_step000600_head_full/comparison_audit.json) |
| 图像与类别联合分布矩（此前版本） | 1.125 | 63.814686 | 36.811706 | +0.122858 | [endpoint_joint_moments_v1](/home/zhoushunyu/data/eqvae/experiments/adversarial_weak_training_20260915/endpoint_joint_moments_v1/search_step001000_ema_full/comparison_audit.json) |
| 条件energy目标（此前版本） | 1.075 | 63.705513 | 36.744179 | +0.055331 | [endpoint_energy_guided_v1](/home/zhoushunyu/data/eqvae/experiments/adversarial_weak_training_20260915/endpoint_energy_guided_v1/search_step000416_ema_full/comparison_audit.json) |
| 二分类GAN，真实侧为VAE重建图 | 1.2 | 63.941463 | 37.099522 | +0.410675 | [endpoint_binary_gan_v1](/home/zhoushunyu/data/eqvae/experiments/adversarial_weak_training_20260915/endpoint_binary_gan_v1/search_step000416_ema_full/comparison_audit.json) |
| 二分类GAN，真实侧为原始RGB | 1.15 | 63.477063 | 36.715812 | +0.026964 | [endpoint_binary_gan_rgb_v1](/home/zhoushunyu/data/eqvae/experiments/adversarial_weak_training_20260915/endpoint_binary_gan_rgb_v1/search_step000656_head_full/comparison_audit.json) |
| 二分类GAN，真实RGB，累计1200次对抗更新 | 1.15 | 63.413804 | 36.588825 | -0.100023 | [endpoint_binary_gan_rgb_v2](/home/zhoushunyu/data/eqvae/experiments/adversarial_weak_training_20260915/endpoint_binary_gan_rgb_v2/search_step001456_head_full/comparison_audit.json) |

每个版本先完成0.4→0.2→0.1→0.025的连通系数扫描，再将1K前两名各自扩展到5K。表格只显示两组5K中较好的一个，另一组在对应记录中。5K包含筛选用1K，不是独立留出集。早期两个四来源GAN仅完成固定系数初筛，没有完整5K，未放入表格。

最终两组CPU复核FID分别为**36.58882453（1.15）**、**36.65847940（1.175）**；[复核记录](/home/zhoushunyu/data/eqvae/experiments/adversarial_weak_training_20260915/endpoint_binary_gan_rgb_v2/confirmation_step001456/result.json)中的GPU/CPU差均小于0.00001。

当前有相同固定5K下的小幅FID改善，但sFID从旧最佳71.0530变为新最佳候选的72.3383，没有全面改善，也没有完成独立随机种子验证。D仍能区分真假，不将其描述成分布匹配已收敛。

所有GPU训练与采样已结束。本轮累计1200次弱头对抗更新，承接已有完整数据50K弱头初始化；不是1200次从零训练。后续新训练与JiT迁移未启动，旧队列保持停止。
