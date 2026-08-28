# Bad/Good 轨迹指标筛选：2026-08-27 结果与下一步

## 结论先说

旧的“跨尺度路径证据直接识别坏图”假设没有通过独立确认，不能再靠改方向、时间窗或
阈值挽救。把搜索范围扩展到 6,884 个标量、200 条互不重名的轨迹，并严格采用“旧组
只负责选量和方向，新组只负责复核”的流程后，目前只保留两个预终点候选：

1. **主候选：中途预测干净图的局部模糊严重度均值。** 新组 AUC 为 `0.9333`，三个
   类别分别为 `0.8920 / 0.9561 / 1.0000`；在三个 tier 共 578 个 winner 一起做 Holm
   校正后，`p=0.00578`。
2. **机制副候选：第 4 个 latent 通道的结构能量最大正跃升不足。** 新组 AUC 为
   `0.8631`，三个类别分别为 `0.9000 / 0.7895 / 0.8259`；联合 Holm 校正后
   `p=0.02296`。

这两个结果值得进入第三批全新数据，但仍然不是已经确认的检测器。原因是：它们是在
大规模清单筛选后才被命名；新组只有 9 个 `clear_bad`；主候选在旧组的 AUC 只有
`0.5989`，而且两组 bad 的构成明显变化；副候选又只出现在一个 latent 通道。当前不
允许据此启动 guidance、拒绝或回滚。

随后完成的双候选严格审计没有改变候选，而是补上了两个重要缺口：按 `global_seed`
整块搬动三类标签（包括 mild）的 100,000 次置换，以及由 discovery clean-good 固定的
三档操作阈值。整块置换后，模糊量的原始/Holm p 为 `0.00001/0.00002`，c3 跃升量为
`0.00004/0.00004`。在 expansion 的 10% conformal-style 风险档，二者分别抓到
`8/9` 与 `5/9` 个 clear-bad，同时告警 `23/304` 与 `18/304` 个 clean-good；5% 档则
分别为 `4/9` 对 `8/304`、`3/9` 对 `7/304`。这些仍是**事后操作审计**，不是新的独立
确认，也不是 Ville 的 anytime 保证。

终点表征的补充审计回答了“这些人工 bad 是否也偏离模型自己的正常样本云团”。在
replication 的 360 张图上，以 discovery clean-good 为固定参照，Inception/FID pool3
表征的类内均值距离、收缩 Mahalanobis 距离和 5-NN 距离分别得到
`0.9514 / 0.9480 / 0.9050` 的 class-matched AUC；三项 Holm p 都为 `0.00006`。但
同样三个量在 discovery 交叉拟合上的 AUC 只有 `0.4991 / 0.3870 / 0.4098`。因此它支持“replication 里以
模糊/融合为主的 bad 是强视觉统计离群点”，不支持“任何离均值远的图都是坏图”。
DINOv2 的对应 replication AUC 为 `0.7455 / 0.7624 / 0.6595`，也提示当前异常更容易被
偏纹理、边缘和局部结构的 Inception 表征看见，而不总会破坏高层语义。所有这些量都
偷看了最终图，只能作回顾性诊断，不能替代预终点触发器。

**2026-08-28 方法边界修正：**AUC、人工标签、Inception、DINO 和 FID 都是外部评判，
不是候选指标本身。AUC 只回答一个已经固定的内部量能否把盲化 bad 排在 good 前面；
FID 只回答两批最终输出的分布是否更接近参考数据。它们均不得进入在线分数、阈值、
类别排名、回滚条件或干预方向。当前方法候选仍只能来自生成完成前可见的内部量：latent、
`pred_xstart`、score/variance head、实际 innovation 和预先定义的跨尺度反事实分支。

第三池的 1,800 条轨迹现已完成，但正式 Stage A 只得到 6 个 clear-bad、其中 4 个
blur/soft-fusion，低于预先冻结的 30/15 事件门。因此正式流程以
`EVENT_GATE_FAILED_NO_SCORE_ACCESS` 停止，B/C 都**没有在第三池正式通过或失败**。三名
独立模型 reviewer 的 clear-bad 二元 Fleiss \(\kappa\) 只有 `0.147`，且单名 adjudicator
把 29 个多数 bad 中的 23 个降级；这说明主要瓶颈同时包括低事件率和标签系统不可靠。

门失败后另做了一次严格标为 exploratory 的标签敏感性诊断：候选、方向、时间窗和旧
阈值均不改，只比较五种已经存在的盲评口径。内部模糊量 \(B\) 的 AUC 范围为
`0.668--0.848`，旧 10% 档的 TPR/FPR 范围为 `0.250--0.593 / 0.045--0.073`；其方向在
五种口径下都一致。内部 c3 结构跃升量 \(C\) 的 AUC 为 `0.428--0.728`，TPR 为
`0--0.483`，明显随标签口径摆动。它支持把 B 作为下一池主候选、C 作为待证伪对照，
但不提供确认性 p 值、置信区间、pass/fail 或干预授权。

## 一、数据和比较口径

冻结模型是官方 DiT-XL/2 ImageNet-256，采样器为 250 步 ancestral DDPM、CFG 4、
MSE VAE。比较只包含相对于这个冻结模型通常水平仍明显更差的输出：主要区域严重
模糊、物体或肢体融合、明显错位、重复或拓扑错误。普通模型质感、轻微瑕疵、合理
裁切和仅仅“不够完美”的图不算主正例。

| 队列 | seed | 总数 | clean-good | clear-bad | mild/disputed | 用途 |
|---|---:|---:|---:|---:|---:|---|
| discovery | 50–129 | 240 | 216 | 8 | 16 | 每条轨迹只选一个标量和一个方向 |
| independent replication | 130–249 | 360 | 304 | 9 | 47 | 原样复核 discovery 的选择 |

所有视觉标签先于候选分数连接而锁定。`mild/disputed` 不参加二元主比较。筛选输出只含
聚合 AUC、计数和校正后的显著性，不发布逐样本标签、分数、seed 或排名。

锁定的文字理由还能解释一个重要的 composition shift：discovery 的 8 个 bad 大致是
`1` 个全局模糊、`5` 个重复/额外肢体拓扑错误、`2` 个脱离或重复物体；replication 的
9 个 bad 则是 `5` 个犬类广泛模糊/融合、`1` 个体操运动员头肩手融合模糊、`3` 个雪橇板
或固定器的严重整体模糊。这个 subtype 构成在连接分数前就已经冻结，因此可以合法地
解释为什么直接模糊量在 discovery 较弱、在 replication 很强。它支持“模糊/融合
subtype detector”，却削弱“所有明显结构错误的统一 detector”这一说法。

## 二、主候选：模型的中途草稿是否一直偏糊

### 1. 字母的含义

- \(i\)：第 \(i\) 个生成样本；
- \(k\)：采样步，候选只查看
  \(\mathcal K=\{69,79,89,99,109,119,129,139,149\}\)；
- \(\widehat x_{0,i,k}\in\mathbb R^{4\times32\times32}\)：模型在第 \(k\) 步预测的最终
  干净 latent，即 `pred_xstart`；
- \(D_{\rm VAE}\)：冻结的 SD MSE VAE 解码器；
- \(I_{i,k}\in[0,1]^{3\times256\times256}\)：把该步的预测干净 latent 临时解码后的
  RGB 草稿；
- \(Y_{i,k}\)：草稿的灰度图；
- \(j\in\{1,\ldots,16\}\)：固定 4×4 网格中的 tile 编号；
- \(\mathcal A_{i,k}\)：灰度方差最大的 8 个 tile，称为 active tiles；
- \(g_x,g_y\)：对高斯平滑后灰度图计算的水平、垂直 Sobel 梯度；
- \(\ell\)：同一平滑灰度图的离散 Laplacian；
- \(\varepsilon=10^{-12}\)：只用于防止除零和取对数；
- \(Q_{0.25}\)：集合的 25% 分位数。

解码公式是

\[
I_{i,k}=\operatorname{clip}\!\left(
\frac{D_{\rm VAE}(\widehat x_{0,i,k}/0.18215)+1}{2},0,1
\right).
\]

对 active tile \(j\) 定义

\[
q_{i,k,j}=
\frac{\operatorname{mean}_{(h,w)\in j}\ell_{i,k,h,w}^{2}}
{\operatorname{mean}_{(h,w)\in j}
\left(g_{x,i,k,h,w}^{2}+g_{y,i,k,h,w}^{2}\right)+\varepsilon}.
\]

直觉上，一阶梯度描述“有多少边缘”，二阶 Laplacian 描述“边缘是否足够陡”。在边缘
总量相近时，模糊边缘的二阶变化更弱，所以 \(q\) 更小。取 active tiles 是为了少受大块
纯背景影响；取 25% 分位数是为了关注较差但不是单个最极端的局部区域。

单个检查点的模糊严重度为

\[
B_{i,k}=-\log\!\left(
Q_{0.25}\{q_{i,k,j}:j\in\mathcal A_{i,k}\}+\varepsilon
\right),
\]

最终候选量为九个检查点的平均：

\[
B_i=\frac1{|\mathcal K|}\sum_{k\in\mathcal K}B_{i,k}.
\]

\(B_i\) 越大，表示模型从较早到中段给出的“最终草稿”整体越缺少清楚的局部边缘。
它在第 149 步模型求值结束后、该步随机 innovation 抽取前已经完全可得。因此它是
preterminal、predictable 的观察量，而不是偷看终点的图像分数。提取阶段不保存这些
中途图，也不读取标签。

### 2. 目前结果和边界

| 量 | discovery AUC | replication AUC | 207 | 602 | 795 | 原始置换 p | 联合 Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|
| \(B_i\)，大为 bad | 0.5989 | 0.9333 | 0.8920 | 0.9561 | 1.0000 | 0.00001 | 0.00578 |

这个量最符合 replication 中保留 bad 的视觉定义，但目前应明确叫作“模糊/融合型失败
探测器候选”。它不自动等价于肢体错位、重复物体或语义拓扑错误的通用后验；discovery
以这些拓扑错误为主时，它也确实只有接近随机的区分度。另外，在线计算它需要在九个
检查点做 VAE 解码，工程开销高于纯 latent 量。

## 三、机制副候选：结构有没有在中段真正长出来

令 \(c=3\) 表示从 0 开始编号的第 4 个 latent 通道。对该通道定义 alpha 补偿后的
空间梯度能量

\[
G_{i,k,3}=\bar\alpha_k\left[
\operatorname{mean}_{h,w}
(\widehat x_{0,i,k,3,h+1,w}-\widehat x_{0,i,k,3,h,w})^2
+
\operatorname{mean}_{h,w}
(\widehat x_{0,i,k,3,h,w+1}-\widehat x_{0,i,k,3,h,w})^2
\right].
\]

这里 \(\bar\alpha_k\) 是该步的累计信号系数；乘它是为了减轻不同噪声尺度下
`pred_xstart` 振幅不可直接比较的问题。再定义中段最大正跃升

\[
J_i=\max\!\left(0,
\max_{k=100,\ldots,148}(G_{i,k+1,3}-G_{i,k,3})
\right).
\]

筛选得到的方向是 \(J_i\) **小**更像 bad。通俗地说，正常轨迹常在中段出现一次清楚的
“结构定型”；模糊或融合轨迹可能一直缓慢拖着，没有明显把空间结构长出来。为了让风险
分数方向统一，可以写成 \(C_i=-J_i\)，于是 \(C_i\) 越大越危险。

| 量 | discovery AUC | replication AUC | 207 | 602 | 795 | 原始置换 p | 联合 Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|
| \(C_i=-J_i\)，大为 bad | 0.6567 | 0.8631 | 0.9000 | 0.7895 | 0.8259 | 0.00004 | 0.02296 |

这个量不需要 VAE 解码，工程上便宜，也比直接模糊分数更接近“生成过程为何失败”的
机制。但是四通道平均版本没有复现，其余三个单通道也弱，说明 `c3` 特异性必须在新池
里被证伪，不能事后改通道。

## 四、哪些路线已经失败

1. 冻结的 Candidate-v5 跨尺度 A/B 联合分数在精确 600 条轨迹上失败：class-matched
   pair-weighted AUC `0.5398`，单侧置换 `p=0.2931`；合并数据共有 17 个 clear-bad、
   520 个 clean-good，预先校准的 10% 告警对 bad 为 `0/17`，同时误报 `23/520`。
2. 规范化 posterior 拆分量也没有复制。代表性 replication AUC 为：每维
   \(\log D\) 最大正跳 `0.2579`、tile concentration 波动 `0.4084`、effective-count
   波动 `0.4740`、标准化全局对齐 `0.5973`；所有联合 Holm 值均为 `1.0`。
3. 原始高维路径似然比仍满足相对于实际两条 Markov chain 的记账恒等式，但发生
   likelihood-ratio collapse。其校准正确不等于有质量检测功效，不能再被当成主指标。

因此当前经验结果支持的是“预测干净结构在中途保持模糊或缺少结构定型”，而不是
“坏轨迹普遍更像高噪声分支”。

## 五、统计上为什么还不能宣布成功

- 6,884 个标量先在 discovery 中搜索，每条轨迹和每个 tier 只保留一个赢家，再去
  replication。新组确实没有参与 feature/direction 选择，但最终候选名称仍来自大清单，
  所以这一轮是强候选生成，不是最终确认。
- replication 只有 9 个 clear-bad；其中类别 602 只有 1 个。单类 AUC 的小数看起来很
  精确，但实际事件数很少。
- `<=149` tier 完全嵌套于 `<=199` tier。同一 hypothesis 出现两行不能算两次复制；把
  重复行也放进 Holm 只会更保守。
- 跨 class 的同名 global seed 相关性已经通过“整组三类标签向量按 seed 一起置换”补审；
  两个候选仍然显著。但候选本来就是从这些旧队列中筛出来的，所以这个 p 值只能排除
  一个具体的相关性解释，不能把旧数据重新变成独立确认集。
- discovery clean-good 校准的操作点已经补齐，能描述在 expansion 上实际抓到多少 bad、
  误报多少 good；但 discovery 同时参与过候选选择，因此不宣称严格的条件 FPR 保证。
  真正预先冻结的阈值另用从未看标签的 seeds 30–49 构造，并只能解释为类内可交换条件下
  的**单个未来样本、单候选、类内边际触发概率界**，不是 600 张样本的硬总预算，不是
  两候选联合触发界，不是 clean-good 条件 FPR，更不是 Ville alpha。

## 六、严格操作审计和已经冻结的第三池

1. 双候选操作审计已完成。它固定候选、方向和窗口，不组合分数；100,000 次置换的单位
   是一个 seed 对应的完整三类标签向量，并对恰好两个候选做 Holm。
2. expansion 操作点如下。这里的 `alpha` 是 conformal-style 排名档位，不是 Ville alpha：

   | 候选 | alpha | bad 告警 | good 告警 | TPR | FPR | 全体干预率 |
   |---|---:|---:|---:|---:|---:|---:|
   | 模糊量 | 0.05 | 4/9 | 8/304 | 0.444 | 0.026 | 0.050 |
   | 模糊量 | 0.10 | 8/9 | 23/304 | 0.889 | 0.076 | 0.111 |
   | c3 低跃升 | 0.05 | 3/9 | 7/304 | 0.333 | 0.023 | 0.039 |
   | c3 低跃升 | 0.10 | 5/9 | 18/304 | 0.556 | 0.059 | 0.092 |

3. 第三池已经在任何新图生成前冻结并完成采样：seeds 250–849，三个类别各 600，共
   1,800 条轨迹；
   三名只看 endpoint 的独立 reviewer 必须先锁定严重度和 `blur/fusion`、
   `duplication/detachment/topology`、`misalignment` 等 phenotype，之后才允许连接分数。
   若 blur/fusion clear-bad 少于 15 或总 clear-bad 少于 30，只发布事件计数，不打开第三池
   的 score-label join。
4. 第三池只有两个不组合的 co-primary。模糊量的主 endpoint 是 blur/fusion bad 对 good，
   门槛为 AUC≥0.75、Holm p<0.05，且冻结的 10% 阈值至少抓到 3 个 bad 并满足 TPR>FPR；
   c3 的主 endpoint 是所有 clear-bad 对 good，门槛为 AUC≥0.70、Holm p<0.05。只有模糊量
   通过才授权 blur-specific 干预；c3 通过只支持机制解释。
5. 只有独立新池通过上述门槛，才进入 matched-seed 干预：在
   第 149 步保存 prefix，告警时独立重采后缀；与完全不动的 baseline 比较视觉胜率和
   FID/precision-recall。若主候选只对 blur subtype 有效，就把论文定位收紧为 subtype
   detector，不能宣称通用伪影检测。
6. 第三池采样源码、模型资产和四个 phase-1 identity 已由 sampling source lock 固定。
   最终 evaluator V5 在任何第三池标签或 feature identity 出现前冻结为严格两阶段：
   Stage A 只打开 consensus 的 manifest、completion 和聚合事件计数；只有 blur/fusion
   clear-bad≥15 且总 clear-bad≥30，独立进程 Stage B 才可打开逐行 consensus 和两个
   label-free 候选产品。实际 Stage A 以 6/4 失败，access audit 确认没有打开逐行标签或
   score；Stage B 从未运行。后门 exploratory 敏感性审计物理独立、只读冻结 B/C，且
   明确不能救回正式门或授权干预。
   若未来 Stage B 被调用，仍只发布聚合 AUC、整块置换、Holm、操作点和预注册结论，
   不输出任何逐样本分数、标签、排名或图像。
   V5 还要求 consensus 目录恰好只有四个冻结文件，并拒绝任何 CSV 多余单元格；V4
   因缺少这两项 fail-closed 检查而保留为 superseded，不再使用。

固定时序还有一个有用但只属描述性的线索：在 expansion 中，模糊量在第 69 步几乎没有
分开（AUC `0.603`），到第 79/89 步已经升到 `0.760/0.890`。c3 能量则不是简单的“bad
一直较低”：bad 在第 100 步反而较高，约在第 126 步与 good 相交，随后增长落后；最强的
单步差异集中在第 141–149 步。这更像“开始时已有粗糙结构，但后段没有继续长实”的
动力学交叉，而不是一个静态边缘分数。第三池仍按冻结的九点均值和 q2 最大正跃升检验，
不会据此事后改窗口。

独立锁审计还澄清了一个措辞：协议 locker 为验证完整性会逐字节计算
`sample_features.csv` 的 SHA，calibrator 也会遍历完整 CSV 行；“未打开”应准确理解为
**未解析非候选分数、未读取任何标签/图片/screen、未用这些内容作科学选择**，而不是文件
字节从未被读过。后续 sampling source lock 会把这个区别和四个冻结 identity 一起钉死。

## 七、终点表征距离：这批 bad 是否真的“不像模型的正常输出”

FID 是两个样本集合的均值和协方差距离，不是单张图的天然分数。这里只有 9 个
replication clear-bad，而 Inception pool3 有 2,048 维，直接为 9 张 bad 估计一个 FID
协方差会严重秩亏且偏差很大。因此本审计保留 FID 所用的 Inception 表征，但把问题改写成
更适合单图的三个冻结距离：

1. 单图到同类别 clean-good 表征均值的余弦距离；
2. 用三类 clean-good 的类内残差共同拟合 Ledoit-Wolf 收缩协方差后计算 Mahalanobis
   距离；
3. 单图到同类别 5 个最近 clean-good 邻居的平均余弦距离。

同样三种距离也在 DINOv2-with-registers-large 的 CLS 表征上计算。discovery 使用按
`global_seed mod 5` 的五折交叉拟合；replication 的参照只由全部 discovery clean-good
拟合。距离名称、方向和分析代码均在打开标签前锁定；置换按一个 seed 的三类别标签向量
整块搬动并对恰好六个检验做 Holm。

这里的“正常均值”不是全部生成样本的无标签均值，而是人工标签锁定后的 discovery
clean-good 类内参考；所以它是 held-out 的监督式 typicality 审计，不应包装成完全无监督
质量分。AUC `0.9514` 的精确含义是：在同一类别内随机抽一张 replication clear-bad 和
一张 clean-good，约 95.1% 的配对中，bad 的 Inception 均值距离更大。

| 终点表征距离（大为 bad） | discovery AUC | replication AUC | 207 | 602 | 795 | 原始 p | Holm p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Inception 到类内均值 | 0.4991 | 0.9514 | 0.9520 | 0.9211 | 0.9630 | 0.00001 | 0.00006 |
| Inception 收缩 Mahalanobis | 0.3870 | 0.9480 | 0.9820 | 0.7719 | 0.9593 | 0.00001 | 0.00006 |
| Inception 类内 5-NN | 0.4098 | 0.9050 | 0.9680 | 0.5614 | 0.9333 | 0.00001 | 0.00006 |
| DINOv2 到类内均值 | 0.2925 | 0.7455 | 0.6320 | 0.6754 | 0.9852 | 0.00578 | 0.01156 |
| DINOv2 收缩 Mahalanobis | 0.3730 | 0.7624 | 0.9120 | 0.7368 | 0.4963 | 0.00308 | 0.00924 |
| DINOv2 类内 5-NN | 0.3730 | 0.6595 | 0.9260 | 0.7018 | 0.1481 | 0.05267 | 0.05267 |

replication 类别 207 中有足够 bad 可以发布组统计：Inception 均值距离的 clear-bad
中位数为 `0.1545`，clean-good 中位数为 `0.0810`；Mahalanobis 中位数为 `77.64` 对
`53.21`；5-NN 中位数为 `0.1833` 对 `0.1039`。这不是几个极端值单独拉高均值。
另外两类 bad 数分别只有 1 和 3，按预先冻结的最小组规模 5 不发布组内分位数，只报告
AUC 中的配对次序。

作为事后稳健性检查，逐一删掉 9 张 clear-bad 中的任意一张，三个 Inception AUC 的
范围仍为 `0.946–0.964 / 0.943–0.974 / 0.894–0.956`。固定按完整三类别 seed block
重采的 3,000 次描述性 bootstrap 95% 区间，前两项为
`0.908–0.984 / 0.888–0.991`，第三项为 `0.790–0.980`。这排除了“只由一张极端坏图
拉动”的简单解释，但 bootstrap 是看到结果后补做的敏感性分析，不替代第三池确认。

这个结果的正确直觉是：replication 的明显模糊/融合坏图既被人看成坏，也落在模型通常
视觉统计的稀疏区域；但“稀疏”不是“坏”的同义词，罕见构图、少见姿态和真正有价值的
细节也可能远离均值。discovery 上的反向或近随机结果正是这个边界的实证提醒。终点
Inception 三个距离对 mild 对 clean-good 的描述性 AUC 反而在两批都较稳定：discovery
为 `0.618–0.647`，replication 为 `0.620–0.647`。这更像一个温和的“视觉不寻常度”轴，
而不是严重度的单调刻度；某些离群 mild 会得高分，某些已成为常见错误模态的 clear-bad
会留在正常云团内。终点
Inception 距离与预终点模糊量 \(B\) 的类内中心化秩相关也只有 `0.071–0.113`，与 c3
风险 \(C\) 约 `0.201–0.216`；两者不是同一个分数的改名。第三池应把终点表征距离用作
对盲化视觉标签的旁证和 phenotype 分层工具，而不是用它重新定义 good/bad 或在线回滚。

第三池的表征旁证也已经在任何新池标签、图像访问或 embedding 提取前冻结，不再把六个
量都当作可挑选的检验。恰好两个 secondary hypothesis 是：E1 用 Inception 类内中心
距离检验 blur/fusion clear-bad（AUC 门槛 `0.75`）；E2 用 DINOv2 收缩 Mahalanobis
检验全部 clear-bad（AUC 门槛 `0.70`）。二者共同做两项 Holm 且都要求 adjusted
`p<0.05`；其余四个距离、类别和 subtype 切片只能描述。参考模型只使用旧 discovery
的 216 张 clean-good（各类 `71/70/75`），第三池禁止重拟合。该 secondary family 与
B/C 主 family 分离：无论结果多好，都不能救回失败的 B/C，也不能授权干预；事件数门槛
失败时连这两个终点距离也不得与标签连接。

## 八、可审计产物

- 6,884/200 visual-v2 筛选源码：
  `experiments/explore_dit_bad_good_inventory_replication_visual_v2.py`
- 冻结协议：
  `experiments/locks/dit_bad_good_inventory_replication_visual_v2_protocol/protocol.json`
- 聚合结果：
  `/data/users/zhoushunyu/eqvae/cross_scale_evidence/bad_good_metric_confirmation_expansion_v1/inventory_replication_exploratory_visual_v2`
- 结果 manifest identity：
  `a5d5b13317fdc6e5881398da1c91e92a140c6c1f77c9cd15f7f8a57119e1527c`
- 双候选严格操作审计：
  `/data/users/zhoushunyu/eqvae/cross_scale_evidence/bad_good_metric_confirmation_expansion_v1/dual_candidate_operational_audit_v1`
- 审计 manifest identity：
  `43f65c7c32fb21b3fc6ddc151d5cf761e1a9645733ef0369ed1b72111eb23021`
- 第三池协议锁：
  `experiments/locks/dit_bad_good_third_pool_protocol_lock_v1/third_pool_protocol.json`
- 第三池阈值锁：
  `experiments/locks/dit_bad_good_third_pool_threshold_lock_v1/thresholds_locked.json`
- 第三池 sampling protocol / manifest identity：
  `330661e87de7846e1f590660f03ecef6270fa45e2f39c4fc54d992e3260950d8` /
  `eae86d48c1c1b9c732fbeea4838b2418b9b7261b61db0355fd7306469f5b6df3`
- 第三池 evaluator V5 source / scientific contract / manifest identity：
  `006a9337295d1a3f27ad8626fdff21d227038cf11f291a769dde1af8c41aff5c` /
  `6638f75eef792fa313fa14ebb0b6c65a696dab881c193f2bf8fa83615e1475e2` /
  `d7467275fab416a5eddadf528fd24b98ffb6bfeed499711c3e8ba6b6f72cd6e8`
- label-free endpoint embedding manifest identity：
  `3116d850421cae89245ff50e84d915e87c74fb1218f12de5c1dda10f77f04912`
- 终点表征距离协议 identity：
  `1489f4d5f9844977fe3ec5950e9cca1e08b2dc4b734d6d1ef6786a585df5299f`
- 终点表征距离聚合审计：
  `/data/users/zhoushunyu/eqvae/cross_scale_evidence/bad_good_metric_confirmation_expansion_v1/endpoint_representation_distance_audit_v1`
- 终点表征距离审计 manifest identity：
  `4510c4489ba082a8961793851537484efef299acda9d395e2f3fd36e48ae9335`
- 第三池终点表征 secondary protocol / manifest identity：
  `575e7a8081144f17d16d99387561f950d7551017faf7d6b0a2b5d93a921a1bdd` /
  `905a634ba68e0f3277e7eae3f4218951593e2686da3deef203e55bac86cc8500`
- 第三池冻结 clean reference summary / logical-array identity：
  `e0538863a1edc02e6f21d6ad7c6ddd32f8cc57046adcd4b88016d7ae5e1889fd` /
  `51c3e80e712912a39c65cca977756b4bf125feba83579c4820ba0f0c143e5070`
- 第三池正式 Stage-A manifest / receipt identity：
  `4681e7f416e5e5f72aacdd1ba80fbe9f617f45d4b993e51d592255cdb4d43d61` /
  `8b3915be23d03ad9e4b4ea379bf2c8e86d49c83e83e401892cc2499b192f2161`
- post-gate B/C 标签敏感性聚合审计：
  `/data/users/zhoushunyu/eqvae/cross_scale_evidence/bad_good_metric_confirmation_third_pool_v1/bc_label_sensitivity_exploratory_v4`
- 该审计 source protocol / result / manifest identity：
  `4b9e03f4e07044ec48fa149245bdd7b0d8b5ce9a4b55c49ba328e482494fd290` /
  `2c7bc875cedbe92328d5c826ba4b66cd547059d44e19242f7250cc93386ad609` /
  `adc68fc093fee6084b0809e116e4523ee0931a47769f250c2e56c036131c8d94`
- 事件富集新池 v3 scientific protocol / manifest identity（尚未真实采样）：
  `04e933793992e2a7ce62aa4ac66836412f3c4f221cce731f2e072da97e892dd7` /
  `0778e0ad2732256a1377d61ba7f04c6ad4f1fdca3a7fd9dec00d0b89e0247e36`
- endpoint-only sampling protocol / source manifest identity（真实输出仍为 0）：
  `6e0a222ff313262f641b1a2b6d70cdfbc73fd71da844396a5d4db3974ceb58d3` /
  `36cc1dfa90aff69737338f8970703c726a6794cc8a7677bbe4d1494ac8e40da4`
