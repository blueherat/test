# 官方 ImageNet-1K SiT-XL PFR 固定 1K

接口检查40374正常结束：Full官方Euler终点逐位一致、depth8 prefix逐位一致，
800 B4 Full、124 prefix，计算29.3806秒；无质量结论。

现在在质量运行前固定 ordinary100、PFR100、ordinary115 三组。
官方800 epoch EMA（SHA a7f4eb9f417295a14e5063b70020ab3f55b60a4905ccda7b244343deaf9840cd），
FP32模型和VAE、关闭TF32、官方FP64 Euler状态；seed202609428，B4，1000类各一张。
采用官方SD-VAE-ft-mse / .18215及(255*(decoded+1)/2).clamp→uint8。
每组先8张smoke，再1K；保留前缀像素匹配供后续核验。

三组IG均为 B+1.35(F−B)，全程启用。PFR仅noise-time>.5时加
1.35[B(z,t)−B(z,max(.5,t−1/32))]。同小SiT固定迁移系数，
不是官方最优IG或完整原论文SDE协议。无CFG、强度/窗口搜索或训练。

调用/图：ordinary100为100 Full；PFR为100 Full+50 depth8 prefix；
ordinary115为115 Full。主干28块，块计算分别2800、3200、3220；
115步对照略高，但块数不等于严格FLOPs或实际耗时，必须实测。
预计三组采样合计约 .75 GPU小时，加载/哈希/检查/评估另计。

统一使用nanogen ImageNet256参考与官方FID提取器，保存特征，独立FP64复算。
不能与ImageNet100 SiT的FID绝对值比较。验证checkpoint/VAE/源码、噪声、
标签、实际调用和smoke/quality前8图。结果首先是PFR迁移证据；
即使有效也不算核心创新，不自动5K或写论文。
