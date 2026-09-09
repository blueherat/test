# OU额外方向收益的SiT-v/x配对对照

冻结已有v800/x800及其各自后训练depth4 velocity读出；不训练。沿用已完成的
SIT_PFR_OUTPUT_PARAMETERIZATION_PROTOCOL设置：gamma=.35常量、Euler100、
seed202609417连续RNG、B8、FP32/TF32、ImageNet100均匀随机类别共1K（原文每类10张为误写）。
该设置不等于历史最强Heun22、分段gamma=.6/.7配置，不称作其完整复现。

每模型两个新组：time_only修订weak(t,z)-weak(t+h,z)；ou组在t<.25把该修订
投影到strong OU degree1 defect并恢复raw范数，.25≤t<.5保持time_only。
h=min(1/32,.5-t)，t≥.5不修订。OU查询固定标准化坐标，复用既有谱函数。
两组均执行同一未来OU Full查询，time_only丢弃其值，从而匹配实际调用：
100普通Full+25额外Full+50prefix/样本。总修订系数1.35不改。

先native8、ordinary8逐像素一致，projected PFR8与已有同模型smoke逐像素
一致；time_only8/ou8检查未来prefix与共享full/head逐元素一致。全部通过
才执行两组1K。正式配对hash/模型/实际调用检查与独立ADM特征FID审计。
v/x在GPU0/1分别运行，预计四组共 .13 GPU小时采样，加载及评估另计。
不根据结果改gamma、积分器、区间或h；单次1K探索不替代独立5K确认。

本轮检验OU方向额外收益是否能在共同协议下跨输出参数化出现；两模型训练
不同，不能称为单一变量的严格因果实验。若v在该协议本身无额外收益，不能
仅凭x的表现把差异归因于clean输出。没有论文写作。

## 完成结果与裁决

四组固定1K全部完成。所有前置输出/未来prefix一致性通过，正式每样本实际
125 Full+50prefix，四组噪声/标签hash完全相同。独立FP64特征Gram重算FID
最大差3.05e-5，模型/head/源文件检查通过。总采样443.60秒（.12322 GPU小时，
加载/评估另计）。time_only额外Full查询被丢弃，这是机制配对，不是对最快
time_only实现的效率优势。

| 模型 | time_only FID | OU FID | time_only sFID | OU sFID |
|---|---:|---:|---:|---:|
| SiT-v | 68.447133 | 65.779885 | 211.346180 | 213.171513 |
| SiT-x | 68.022327 | 65.440884 | 209.602223 | 211.509873 |

FID降低2.667248/2.581443；IS由34.0663/32.9885升至35.8298/37.2548，但
sFID都变差，不能宣称所有质量指标改善。sFID本轮仅记录官方评估结果，
没有独立重构spatial协方差指标；独立重算仅针对pool3 FID。

在这组固定共同协议中，OU选择的额外FID收益同时出现在v/x；因此clean
输出本身并非该收益的普遍障碍。两个模型并非仅输出参数化不同的严格因果
干预，不能据此证明RAE失效与输出完全无关，更不能推出联合训练弱头就是
原因。此处是1K而非新5K，且gamma/积分器与历史最强5K不同。

后续方法需要解释OU方向为何在相同VAE表示的两种源模型均有用，到了RAE
不再有额外收益；不再把clean输出作为单独的充分解释。论文保持暂停。
结果 experiments/results/terminal_defect_20260908/sit_ou_output_control.csv。

后续完整标签审计：1K各类4–20张，100类均覆盖；四组标签逐项一致。代码
沿用torch.randint而非严格均衡标签。详见独立5K协议勘误，不修改原结果。
