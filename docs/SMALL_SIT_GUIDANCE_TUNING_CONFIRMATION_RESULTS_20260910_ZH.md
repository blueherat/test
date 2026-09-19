# 小SiT：调过强度后的IG、残差方向与ADG

状态：`complete`，已完成 3/3 组，每组 5000 张。
完整表保留每个预先指定的配置；FID越低越好。

本轮最低FID为 `adg_a070`：**39.499792**。

IG当前已测最低为 `ig_a070`：40.152568。
本轮参数由旧1K选择后冻结，新噪声与平衡类别均在确认前生成；模型、训练数据与真实图参考未独立更换。

## 各强度FID

| 峰值额外强度 | IG | 残差方向 | ADG |
|---:|---:|---:|---:|
| 0.7 | 40.152568 | 39.834159 | 39.499792 |

## 完整指标与实测成本

| 配置 | FID | sFID | IS | 主干NFE/图 | 采样＋解码GPU秒 |
|---|---:|---:|---:|---:|---:|
| ig_a070 | 40.152568 | 69.522766 | 37.4534 | 114.333 | 461.882 |
| residual_norm_a070 | 39.834159 | 69.588246 | 37.2036 | 114.314 | 486.977 |
| adg_a070 | 39.499792 | 69.115820 | 37.8852 | 113.536 | 496.925 |

成本为各batch实测GPU采样＋解码秒之和，包含诊断、新投影或角度算术以及自适应步数变化；不含加载、预检、ADM评价和历史训练/拟合。
每次RHS仅一次主干双头前向，额外主干调用为0。残差方法使用此前冻结的6160系数线性读出；不将其称为非线性预测组合。
ADG为已有论文公式在当前Strong/Weak上的移植；它是本轮非线性比较，不能视作新方法。

## 完整性核验

核对冻结源码、权重、参考、输入、每批输出、汇总样本与Inception特征哈希；检查所有图索引无重复无遗漏。
四rank预检验证Strong/Weak入口一致、零强度恢复以及旧IG/残差工作点逐位复现。
FID/sFID用已保存的Inception特征独立以FP64重算；它不等于独立提取特征。
最大绝对复算差：FID 4.16e-06；sFID 1.82e-06。

## 证据

- [冻结请求](/home/zhoushunyu/data/eqvae/experiments/small_sit_guidance_tuning_confirmation_20260910/request.json)
- [原始指标](/home/zhoushunyu/data/eqvae/experiments/small_sit_guidance_tuning_confirmation_20260910/results.json)
- [审计JSON](/home/zhoushunyu/eqvae/docs/data/small_sit_guidance_tuning_confirmation_20260910/audit.json)
- [完整CSV](/home/zhoushunyu/eqvae/docs/data/small_sit_guidance_tuning_confirmation_20260910/metrics.csv)
- [采样代码](/home/zhoushunyu/eqvae/experiments/small_sit_guidance_tuning_confirmation_20260910.py)
- [分析代码](/home/zhoushunyu/eqvae/experiments/analyze_small_sit_guidance_tuning_20260910.py)

三个族各自最优配置的相同前8个输入，未按图片质量挑选：

![固定样例](/home/zhoushunyu/eqvae/docs/data/small_sit_guidance_tuning_confirmation_20260910/paired_first8.png)
