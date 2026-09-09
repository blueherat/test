# 官方 SiT PFR：新噪声固定 5K 确认

三组1K采样、配对和独立FID核验完成后，固定下一步只比较PFR100
与ordinary115，不更改scale1.35、h1/32、noise-time>.5或采样/解码算术。
新seed202609429，B4，5000张、每类5张；与探索1K seed202609428不同。
运行前固定此协议，不依据新结果调整强度、窗口、选择样本或提前停止。

sample_official_sit_pfr_5k.py 与原sampler唯一内容差别为seed；
驱动先8张smoke，再5000张，自动官方ImageNet256 FID并保存特征。
前8图像素、噪声/标签/权重/源码、每图Full/prefix调用及实际时间均需核验。
确认数据与1K输入银行的非重叠需核对，不能只根据seed不同作保证。

PFR每图100 Full+50 prefix；ordinary115每图115 Full。
按1K实测预计两组采样约2.82 GPU小时，加载/哈希/smoke/评估另计。
5K必须从特征独立FP64重算FID，不用质量代理替代最终分数。
不与ImageNet100的绝对FID比较，也不声称击败官方最优SDE/IG配置。

这是一次经1K审查后作出的确认决定，不是此前协议自动扩展。
如果固定5K无收益，保留结果、不做参数搜索；如果有收益，先确定
既有方法迁移证据的边界，再寻找核心创新。论文写作保持暂停。

噪声独立性检查已完成：audit_official_sit_pfr_noise.py 按相同 CUDA
B4 调用重建1000/5000个噪声，分别逐图去重并交叉比较，重叠0。
旧1K全队列哈希和新两组smoke前8噪声均匹配。
预期5K noise SHA e37923b88a937c6bc1d5e55b78e9a31992e50f1583b20ae8b8073222cf868e03，
labels SHA 7880f6745ab864b6483071752367154af08721122db79ee5065bed00f95bcdb6。
最终采样仍需与上述完整哈希匹配；脚本/全部逐图哈希保留于
experiments/results/terminal_defect_20260908/official_sit_pfr_noise.json。
会话96571正常退出，计算.3003秒，CUDA启动另计。
5K独立审计脚本 analyze_official_sit_pfr_5k.py 已准备，使用2048维
协方差平方根公式，尚未运行，也未预先宣告质量通过。

在5K尚未完成时固定一个不增加模型调用的补充：主审计通过后，
从配对特征估计固定类分配下的一阶FID差值波动。说明与数值自检见
[局部波动检查](OFFICIAL_SIT_PFR_5K_UNCERTAINTY_20260908_ZH.md)。
此近似不作为校准置信区间，不改变主实验或据此调整样本。
