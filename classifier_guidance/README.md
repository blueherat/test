# 分类器引导弱头：当前主线

只训练弱头和真假二分类器，强生成模型、图像解码器和 Inception 特征网络冻结。
分类器区分真实训练图像与**强弱外推后的最终图像**；弱头通过完整采样轨迹反传来欺骗分类器。
冻结强模型的参数不意味着可以截断它对输入的梯度。

## 代码入口

整理时的实现保留在原命名空间，避免破坏历史源码哈希和绝对路径：

| 职责 | 文件 |
|---|---|
| RGB 二分类器训练 | [`train_binary.py`](../experiments/adversarial_weak_training_20260915/train_binary.py) |
| 分类器和 logistic loss | [`binary_critic.py`](../experiments/adversarial_weak_training_20260915/binary_critic.py) |
| 真实 RGB 训练数据及类别映射 | [`rgb_data.py`](../experiments/adversarial_weak_training_20260915/rgb_data.py) |
| 图像解码 | [`rendering.py`](../experiments/adversarial_weak_training_20260915/rendering.py) |
| 完整离散采样器梯度 | [`discrete_adjoint.py`](../experiments/adversarial_guidance_endpoint_20260915/discrete_adjoint.py) |
| 强模型及弱头适配 | [`models.py`](../experiments/guidance_dynamic_50k_20260915/models.py) |
| 固定噪声 5K 评测 | [`evaluate.py`](../experiments/adversarial_weak_training_20260915/evaluate.py) |

新优化实现放在本目录，保留历史实现作为精度和速度对照。
后续接入 SiT、JiT、RAEv2 时共享训练与反传机制，单独实现模型输入、时间约定和解码适配。

## 结果和数据

- [当前方法及结果](../docs/classifier_guidance/README.md)。
- [完整 64 点 5K 扫参](../docs/ADVERSARIAL_HEADS_5K_NEIGHBORHOOD_20260916_ZH.md)。
- 本地数据导航：`$HOME/data/eqvae/projects/classifier_guidance/`。
- 历史训练原件：`$HOME/data/eqvae/experiments/adversarial_weak_training_20260915/`。
- 环境：`$HOME/miniconda3/envs/myenv/bin/python`。

当前扫参已完成，旧训练的停止标记保持有效。整理目录不会恢复任何暂停实验。
