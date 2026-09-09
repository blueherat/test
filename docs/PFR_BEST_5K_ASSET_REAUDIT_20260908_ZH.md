# 最强SiT PFR资产的重新核验

本轮回到OU strong方向加raw范数结果，不新增采样或训练，不写论文。
从原采样manifest重建seed5/6的CUDA噪声：每组1250批、B4、latent4×32×32，
seed=(run_seed<<32)|batch_index，namespaced_v2。两组全部噪声总hash与历史记录
一致，每组5000个逐样本hash全部唯一，跨组相同噪声数0。平衡标签及其hash
也完整重现。因此这两组不受早期additive批次种子重叠问题影响。

从保留的5000×2048 ADM pool3特征，独立FP64对称协方差平方根重算FID：

| seed | 原FID | 独立FID |
|---|---:|---:|
| 5 | 36.19015567 | 36.19015462 |
| 6 | 35.75879012 | 35.75879212 |

重要范围：两个历史samples_n5000.npz均已不在manifest指定位置。因此本轮
验证的是噪声协议和保留特征上的指标，不是从原像素重新提取特征；不能把
这项审计说成完整生成重现。未重新核验baseline的全部像素链或重新测吞吐。
两组强模型hash相同，方法均strong OU degree1 direction + raw norm / Heun22。

RAE历史对应结果使用同一高噪声证书区间t>.75，但原raw revision为time-only，
沿原IG活动区间而非全部采用SiT的原投影路径；弱头训练、表示和积分器也不同。
原OU迁移失败是真实协议下的负结果，不能解读为仅改变某单一因素的因果实验。
先前已补原PFR投影对照，它同样失败；尚不能据此归因OU失效的唯一来源。

研究决策：这两组SiT正结果仍是可用资产；没有因本轮审计产生新方法或新增
质量证据。不要重复从零拟合参考读出，也不要重启已失败的RAE同场/未来IG拆分。
下一项方法对照应明确区分原PFR收益与OU方向选择的额外收益，不能用前者
在SiT-x有效就断言后者也与clean输出无关。

代码 experiments/audit_pfr_best_5k_assets.py；机器审计结果
experiments/results/terminal_defect_20260908/pfr_best_5k_assets_audit.json。
