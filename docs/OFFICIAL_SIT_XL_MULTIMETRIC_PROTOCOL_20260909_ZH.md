# XL 探索曲线的覆盖率与类别读数

在读取新增辅助指标前固定：对本次十八个 Euler 强度点和原生 SDE250 的两种 VAE 解码完整报告同一套指标，不按结果挑选图像或曲线点。使用已经生成的图像；不把再评价同一批数据称独立确认。SDE 两种解码成本不同于 Euler，图中需分开标识。

Inception 特征空间中计算 k=3 的 improved precision/recall，以及三次多项式核 `(x·y/2048+1)^3` 的标准去对角 KID 统计量。参考为 `/data/shared/adm_refs/VIRTUAL_imagenet256_labeled.npz` 的全部 10000 张 RGB 图像，重新提取原生 nanogen/torch-fidelity Inception 特征，记录像素与特征哈希。其样本参考不是 FID 所用全部统计量的替代，不由文件名推断参考图像的类别构成。

复用生成图像的原始 FID 特征，每个 bank 首 32 张重新前向交叉检查。precision/recall 使用 1000 个生成点与同一批 10000 个参考点，邻域样本数不对称且稀疏，绝对数值不能直接与 50K 论文指标比较。KID 使用全部点的核和，不反复抽子集制造新的标准误；生成 bank 固定每类一图、并非 IID 类别抽样，因此这里不声称去对角统计量是目标分布距离的严格无偏估计。保留原始数值，负数不裁零。

类别读数使用既有 `torchvision ConvNeXt-Tiny IMAGENET1K_V1` 权重及官方预处理，记录权重与实际状态哈希。全部 1000 张均按目标 ImageNet 类别计算 top-1、top-5、目标概率，保存 logits；它衡量固定分类器下的条件识别，不能当真实质量、人类偏好或总体错误比例。强引导可能提高识别率却降低覆盖率，所有指标一起报告。

定义依据：[Improved Precision and Recall（NeurIPS 2019）](https://arxiv.org/abs/1904.06991)、[KID / Demystifying MMD GANs（ICLR 2018）](https://arxiv.org/abs/1801.01401)。本地 precision/recall 复用已有实现，距离及核归约用 float64。Inception 沿用原 nanogen 评价子进程的已核对默认设置：FP32、matmul TF32 关闭、cuDNN TF32 开启；类别分类器的两项 TF32 都关闭。生成模型本身的 TF32 在此前采样时已经关闭；不将其与独立评价子进程的默认值混为一谈。所有 GPU 评价在采样队列之间单独执行，保持已记录的生成时间可比。

这是一次预先明确范围的探索诊断，不能据此补写已证实的质量机制，不能代替新的 5K/50K 生成或决定 ICLR 目标已经达成。
