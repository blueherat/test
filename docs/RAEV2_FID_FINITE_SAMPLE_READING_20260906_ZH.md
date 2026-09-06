# RAEv2：FID 有限样本偏差、1K 筛查与历史特征复用审计

日期：2026-09-06。范围是原文阅读、只读来源检查及 CPU 描述性计算；没有训练、重新采样、运行 GPU、修改 goal 或更改旧候选裁决。

**结论：最终相对 FID 改善至少 5%，不推出配对 1K 点估计也必须改善至少 5%。** 不仅有统计学理由，本地已保存特征也给出直接反例：同一历史协议下，IG 相对 Full 在两套 5K 的改善为 **8.4659% / 5.0949%**；将每套特征分成五个完整类别均衡 1K 子集，十个子集的改善**全部小于 5%**。这支持重新审视未来筛查规则的统计含义，但没有证明已终止 potential 候选在更大样本上会成功。

历史原图的 1K FID 均值为 **39.2278 / 38.9119**，重建为 **38.9035 / 38.8137**；相应 5K 值只有约 6.8–6.9。这些是描述性参照，**不是生成 FID 下界，也不能作为一个可直接减去的“公共偏差”**。全部逐 fold 数值、来源 SHA、CPU 代码与数值验证保存在[独立结果目录](data/raev2_fid_finite_sample_20260906/)。

## 1. 原论文到底证明和验证了什么

深读来源：Chong 与 Forsyth，*Effectively Unbiased FID and Inception Score and Where to Find Them*，CVPR 2020，pp. 6070–6079。阅读了正文 §2.3、§3.1–3.2、§4.1–4.5、图 2–7，以及补充材料 Algorithm 1；不是只读摘要。[CVF 正文](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chong_Effectively_Unbiased_FID_and_Inception_Score_and_Where_to_Find_CVPR_2020_paper.pdf)，[CVF 补充材料](https://openaccess.thecvf.com/content_CVPR_2020/supplemental/Chong_Effectively_Unbiased_FID_CVPR_2020_supplemental.pdf)。

论文固定真实参考统计，把生成侧均值、协方差看作 Monte Carlo 积分估计。FID 对这些估计量是非线性函数，即使矩估计无偏，FID 一般仍有偏。其 §3.1 给出 Taylor 展开形式：

\[
\mathbb E[\widehat F_n(g)]
=F_\infty(g;\text{fixed reference})+K_g/n+O(n^{-2}).
\]

系数依赖生成器的特征分布，因此固定相同样本数不能保证两个模型的偏差抵消。这里的 \(F_\infty\) 仍然针对**固定的有限真实参考统计**，不自动等于对无限真实数据的距离。[正文 §2.3、§3.1–3.2](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chong_Effectively_Unbiased_FID_and_Inception_Score_and_Where_to_Find_CVPR_2020_paper.pdf)。

以下是对该展开的解释，而非论文额外定理：令 \(\theta=(\mathbb E\phi,\mathbb E\phi\phi^T)\)，\(G(\theta)\) 为固定参考的 FID。在足够矩存在、相应点附近矩阵平方根足够光滑、IID 抽样等条件下，二阶 delta method 的项包含

\[
K_g=\tfrac12\operatorname{tr}\!\left(H_G(\theta_g)\,\Omega_g\right),
\qquad
\Omega_g=\operatorname{Cov}_g\!left[(\phi,\operatorname{vec}\phi\phi^T)\right],
\]

另需按实际矩估计方式处理其自身偏差。\(H_G\) 与 \(\Omega_g\) 都可随 guidance 改变；不能只用同噪声配对消除。配对主要减少**差值方差**，不会使两个非线性估计的期望偏差相同。

关键实验证据是图 3：两份相同架构、独立训练的 DCGAN，其 \(FID_n\) 排序随 \(n\) 改变；图 2 中不同模型有不同斜率。作者用 50K 图池拟合并预测 100K 分数，跨 50 次重复检验，推荐在样本数轴均匀取点；正文和补充算法都将最小拟合规模设为 **5K**，并未验证“仅用 1K/5K 即可可靠推到 50K/无穷”。补充算法对激活 shuffle 后取前缀，与正文 §4.3 的“有放回抽样”文字存在区别，复现时应明确采用哪种。[正文图 2–6、§4.3，补充 Algorithm 1](https://openaccess.thecvf.com/content_CVPR_2020/html/Chong_Effectively_Unbiased_FID_and_Inception_Score_and_Where_to_Find_CVPR_2020_paper.html)。

不能直接搬用的条件有三点：

- 本任务每类固定一张或五张，是分层抽样，不是对总体标签分布的 IID 抽样。其波动与偏差系数需按此设计理解。
- 1K 样本的 2048 维协方差秩至多为 999。FID 仍可定义和计算，但不能未经验证就把大样本、光滑区域的 \(1/n\) 直线当作可靠外推模型。
- 论文的 QMC 方差降低实验主要面向 GAN 等模型；不能据此直接保证 RAEv2 高维初始噪声上的相同收益。本次完全保留已有噪声和图像，不引入 QMC。

## 2. 1K 的 5% 门槛与用户最终目标并非同一个条件

[当前 goal 原文](RAEV2_GUIDANCE_GOAL_20260906_ZH.md)要求最终在公平计算成本下相对可靠基线改善至少 5%，并要求先做配对 1K 筛查、再独立确认。它**没有显式要求每个候选必须先达到 1K 相对 5%**。

[旧 potential 筛查](RAEV2_OBSERVABLE_POTENTIAL_SCREEN_20260906_ZH.md)另行固定了第一阶段 \(1-FID_\Phi/FID_0\ge0.05\) 这个必要条件，并因此在 **37.562703724300 → 37.531664406887（0.08263334%）** 时停止。按当时冻结的运营规则，记录“未通过该筛查并停止”是准确的；把它解释成“已证明最终 5% 不可能”则超出了证据。

设基线与方法在可用展开区间有 \(F_{b,n}\simeq F_{b,\infty}+K_b/n\)、\(F_{m,n}\simeq F_{m,\infty}+K_m/n\)，则

\[
R_n\simeq
\frac{F_{b,\infty}-F_{m,\infty}+(K_b-K_m)/n}
{F_{b,\infty}+K_b/n}.
\]

即使恰好 \(K_b=K_m\)，小样本公共偏差也会抬高分母、压低相对改善；若系数不同，分子和排序也会变化。所以 \(R_{1000}\ge5\%\) 不是 \(R_{50000}\ge5\%\) 的数学必要条件，反方向也不保证成立。

仅用于展示逻辑的假设例子：若两个 \(F_\infty\) 为 2 与 1.8，共同 \(K=35000\)，则 1K 为 37 与 36.8，相对改善 0.54%；50K 为 2.7 与 2.5，改善 7.41%。**这些数不是对本模型的拟合，也不是对 potential 的预测。**

因此，对当前 0.0826% 的正确判断同时包括：

1. 它没有通过旧预注册条件，也没有稳定收益、独立确认或公平总成本成功证据。
2. 不能推定 50K 改善仍是 0.0826%；也不能用原图 FID、别的方法斜率或比例换算，把它“校正”为至少 5%。
3. 1K 的结果是具体规模下的点估计；极小正差不能称显著。修改未来筛查规则的统计解释，不等于重新训练、重开这个候选或抹去旧阴性记录。

## 3. 来源审计：哪些数据能够直接复用

两套历史 5K 目录为：

```text
/home/zhoushunyu/data/eqvae/experiments/raev2_ig_scale_response/
  n5000_seed20260801_scales7_v1/
  n5000_seed20260802_scales7_v1/
```

每套有 completed `manifest.json`、完整 `sample_protocol.npz`、运行日志、原图/重建基线指标，以及四 shard 的 Inception `.npy`。**`inception/source_*` 是原图；`inception/real_*` 是 D(E(x)) 重建，不能望文生义把后者当原图。** `scale_s1p000000_*` 为 Full；`scale_s1p780000_*` 为官方 IG 1.78。

本次直接检查得到：

| 检查对象 | 结果 |
|---|---|
| 两套 global IDs | 均恰好 0–4999；label = ID mod 1000 |
| 类别与原图源行 | 每类恰好 5 张；每套 5000 个源行无重复 |
| 两个 seed 的源图交集 | 21 张；不是完全不重叠的真实图池 |
| 八组有序特征 | 均为 float32 `[5000,2048]`、全部有限、各自无完全重复特征行 |
| shard 顺序 | rank r 对应 IDs `r, r+4, ...`；不能直接按 rank 拼接后把连续行当 global IDs |
| 原图/重建配对 | 使用同一 `real_source_rows` 和 global ID；Full/IG 为同标签、同初始噪声配对 |
| 5K 旧指标复核 | 八项全部重现；本次 FP64 均值与旧 FP32 均值造成的最大差仅 **4.63e−7** |

原图路径是 ImageNet train 的固定源行，ADM center crop 到 256、无水平翻转。冻结模型及数据配置可从 manifest/log 追溯。它们足够支持**对已保存特征做同源、同参考、同类别配额的 CPU 子集描述**。

但不能把历史管线和当前官方筛查称为完全相同：

| 项目 | 历史 5K | 当前配对 1K |
|---|---|---|
| Inception 实现 | torch-fidelity `inception-v3-compat`，2048 | nanogen 内部仍调用同一 torch-fidelity 类，2048 |
| 提取器输入处理 | uint8 NCHW；内部 TF1 风格 bilinear 到 299；`(x−128)/128` | 同样的类及处理，没有另加外部 resize/标准化 |
| decoder 图像量化 | decoder 后 `.float()`，clamp，**FP32 ×255**，截断到 uint8 | 保存筛查协议明确为 **BF16 ×255** 后 uint8 |
| 历史精度证据 | 源脚本显式开启 CUDA matmul / cuDNN TF32 | 当前生成协议关闭 TF32；evaluator 是独立进程，不能由采样设定推定它的所有后端开关 |
| 真实参考 | ADM `VIRTUAL_imagenet256_labeled.npz` 的 mu/sigma | `imagenet_256_fid_stats` 的 mu/sigma |
| FID 数值实现 | torch-fidelity covariance-product eigenvalues | nanogen FP64 moments + scipy `sqrtm` |

对两个 seed、四组图像、每组固定 IDs 0/4 共 16 张作了 CPU 提取抽查，使用源图重新读取或已存 decoded 图像，最大特征差范围 **0.00212–0.00440**，RMSE **0.000330–0.000609**。原图也有此级差异，因此不能把差异全部解释成 decoded 图像的 float16 存储；历史 TF32 CUDA 与当前 CPU 的运行差异同样可能参与。没有历史权重 SHA 和完整运行环境快照，不能声称 bitwise 一致。当前权重身份、逐样本抽查、当前读到的源文件 SHA 均已保存。[抽查结果](data/raev2_fid_finite_sample_20260906/cpu_feature_spotcheck.json)，[源文件指纹](data/raev2_fid_finite_sample_20260906/reading_source_hashes.json)。

**复用边界：** 本次保留原始特征，不混入 CPU 重提取特征，也不把当前精度差当作待调参数。历史 source/recon/Full/IG 之间的特征及参考口径是一致的；换成当前参考的结果只命名为“历史特征＋当前参考”，不伪装成重新完成了当前官方采样/提取协议。

## 4. 两份参考统计不同，但受控换参考只改变约 0.1

旧参考的 mu 是 FP64，新参考 mu 是 FP32；sigma 均为 FP64。两份数组不相同，最大绝对差分别为 **0.0308858280 / 0.0152054295**，它们之间的 FID 为 **0.10179148**。

当前官方 1K 缓存特征保持不变，只做 CPU 复核：

| 计算 | FID |
|---|---:|
| 归档官方 nanogen | 37.562703724300 |
| CPU 原 nanogen sqrtm 公式＋当前参考 | 37.562703622977 |
| CPU 稳定 Gram 公式＋当前参考 | 37.562727413688 |
| CPU 稳定 Gram 公式＋历史 ADM 参考 | 37.650181227471 |

原公式复现误差约 **1.0e−7**；Gram 与非对称 `sqrtm` 在秩亏协方差上的差约 **2.38e−5**，远小于这里讨论的百分点与数十点规模效应。两种公式的微小数值差不被计作模型收益。受控换参考只改变约 **0.08745**，不能解释约 37 与约 7 的主体差距。[验证结果](data/raev2_fid_finite_sample_20260906/validation.json)。

参考来源仍有实际限制：nanogen catalogue 将新文件归于 JiT / iMeanFlow，并从 HF 缓存读取；本地缓存 snapshot 为 `0227134b29f25704c3856ec002ce4a2183cc7419`，文件仅含 mu/sigma。旧 VIRTUAL 文件除统计外还附 10000 张图像，但**不能由这个 arr_0 数量推断 mu/sigma 就由这 10000 张计算**。两者均未在当前文件中记录完整真实样本 ID、参考构建精度和全部预处理环境；原图源行与参考原图的重叠也无法审计。完整数组 SHA 见 [provenance](data/raev2_fid_finite_sample_20260906/provenance.json)。

## 5. 实际 CPU 结果：类别均衡的五个完整 1K fold

取 fold j 的 global IDs 为 `[1000j,1000(j+1))`，j=0,…,4。每 fold 正好每类一张；同一 fold 同时索引原图、重建、Full、IG，不另抽样、不选有利 fold。以下是**历史 ADM 参考**，`±` 为五个不相交 fold 的样本标准差，不是置信区间。

| seed | 对象 | 原 5K FID | 五个 1K 的均值 ± SD | 1K 最小–最大 |
|---:|---|---:|---:|---:|
| 20260801 | 原图 | 6.9217 | 39.2278 ± 0.6359 | 38.5908–40.1697 |
| 20260801 | 重建 | 6.8821 | 38.9035 ± 0.7156 | 38.1858–40.0055 |
| 20260801 | Full | 7.7493 | 39.5434 ± 0.2391 | 39.2836–39.9349 |
| 20260801 | IG | 7.0932 | 38.6722 ± 0.2693 | 38.4140–39.0935 |
| 20260802 | 原图 | 6.8028 | 38.9119 ± 0.2485 | 38.6269–39.1849 |
| 20260802 | 重建 | 6.8415 | 38.8137 ± 0.2785 | 38.3531–39.0285 |
| 20260802 | Full | 7.5879 | 39.1562 ± 0.6061 | 38.4673–40.1310 |
| 20260802 | IG | 7.2013 | 38.7513 ± 0.4192 | 38.2426–39.2708 |

换成**当前参考但仍使用这些历史特征**，两个 seed 的原图 1K 均值为 **39.1576 / 38.8692**，重建为 **38.8383 / 38.7662**；对应 5K 分别为 **6.8246 / 6.7344** 与 **6.7902 / 6.7689**。因此大幅规模效应在两种固定参考下都存在。[全部 96 项逐规模、逐 fold、逐参考值](data/raev2_fid_finite_sample_20260906/subsample_fid.csv)，[描述性汇总](data/raev2_fid_finite_sample_20260906/summary.json)。

同一历史 ADM 参考下，IG 相对 Full 的改善如下，逐 fold 全部列出：

| seed | 5K | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---:|---:|---:|---:|---:|---:|---:|
| 20260801 | **8.4659%** | 2.1068% | 2.0826% | 2.0942% | 2.6552% | 2.0770% |
| 20260802 | **5.0949%** | −0.0557% | −0.7733% | 0.5841% | 1.6558% | 3.6744% |

更换为当前参考后，两组 5K 改善为 8.5064% / 5.6253%，十个 1K fold 仍全部低于 5%。这是本地“1K 必须先过 5%”会错过在另一规模达到 5% 的固定对照的直接证据。**这是评估方法学反例，不是当前任务的成功候选：现在的可靠基线本身已经是 IG 1.78，不能用旧 IG 胜过 Full 冒充新方法胜过官方 IG。**

## 6. 能否声称“显著的 model-dependent 有限样本偏差”

需要把三个判断分开：

**模型相关偏差在理论及论文实验上成立。** 不能假设 Full、IG、新候选、原图、重建具有同一个可加偏差。

**本地大幅样本规模效应已直接确认。** 在固定特征、固定参考内，1K 均值与 5K 的差为约 31.55–32.31。这不再只是拿两个不同 pipeline 的 37.56 和 7 进行猜测。各对象的差也不完全相同：例如 seed 20260801，Full 为 31.7941，IG 为 31.5790，原图为 32.3060，重建为 32.0214。

**但本地模型总体偏差系数的统计显著性仍未识别。** 这些差包含固定 5K 图池的抽样波动、分层抽样效应和非线性 FID 的尺度变化；5K 本身仍有偏，未知总体 FID。五个 fold 与 5K 全集共享数据，两个真实图池也有 21 张重合。不能把这些差直接标成 \(K_g(1/1000-1/5000)\) 的无偏估计，不能由十个 fold 生成一个貌似独立样本的显著性结论，也未据此估计 FID∞。

原图/重建数值只说明同类平衡的小样本估计会有很大的非零 FID。FID 是均值与协方差的非线性函数；生成图的特征分布可以比某个真实小子集或重建分布更匹配参考矩，所以这些数值不构成性能下界。当前官方 1K 的约 37.56 低于这些原图均值，本身并不矛盾。

## 7. 可执行 CPU 协议及下一步信息价值

已执行的计算入口：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
/home/zhoushunyu/miniconda3/envs/myenv/bin/python -u \
docs/data/raev2_fid_finite_sample_20260906/compute_cpu.py
```

该脚本先验证特征、类别、源行和 reference，保存源文件/数组 SHA，然后计算完整 5K 与全部五个 1K fold。使用 FP64 均值与 `ddof=1` 协方差；不做 whitening、正则化协方差、样本重复填充、重新归一化或删除样本。

为避免 2048 维秩亏 covariance-product 的数值复数问题，1K 采用等价的低秩形式。若 \(A=(X-\bar X)/\sqrt{n-1}\)，参考协方差为 \(S_r\)，则

\[
\operatorname{tr}\sqrt{S_r^{1/2}A^TA S_r^{1/2}}
=\operatorname{tr}\sqrt{A S_r A^T}.
\]

两者非零特征值相同；仅截断机器精度级负特征值，出现实质负值会报错。对完整 5K 使用对称 Bures 矩阵，并以原归档八项指标及当前官方原 `sqrtm` 公式作交叉验证。主计算 CPU wall 为 **28.71 秒**；图像提取抽查另由 [spotcheck_cpu.py](data/raev2_fid_finite_sample_20260906/spotcheck_cpu.py) 执行，计时见其 JSON。所有额外工作都没有 GPU 调用。

对未来固定方法，**跨 n、完全同口径的配对评价有信息价值**：先锁定方法、可靠成本对照、标签/噪声池、decoder 量化、Inception 权重/精度及真实 reference；在一个足够大的固定池上，预先指定每类等额的 1K/5K/10K/50K 子集，保留全部规模的绝对差、相对差及同图配对关系。用多份独立完整噪声池确认跨规模行为；共享大池的嵌套点只能作为一条尺度曲线，不能当作独立重复。

若考虑 FID∞，应另外固定拟合点、拟合方法、最小 n、残差检查与独立大样本验证；论文的 5K 起点及 50K→100K 验证不支持从现有 1K 直接外推。正式达标仍应使用约定规模与公平总成本对照，不能把外推值混入原先直接测量的 FID 门槛。

本地两套历史 5K 足以完成上述**描述性子集审计**，但不包含 potential 的 5K/50K 特征，不能回答那个候选在更大 n 上的效果。这里不决定为它新增采样、重开训练或更改 goal；它的 1K 停止与未完成后续成本/独立确认的记录保持原样。
