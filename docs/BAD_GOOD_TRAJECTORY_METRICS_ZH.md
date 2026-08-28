# Bad/Good 轨迹指标：当前假设、严格定义与验证协议

> **2026-08-27 更新：本文件开头以下的大部分“当前候选”已经被后续实验取代。**
> 冻结 Candidate-v5 跨尺度分数已在 600 条轨迹上失败。最新的 6,884 个标量、200 条
> 轨迹的 discovery→independent-replication 筛选，保留了“中途预测图局部模糊严重度”
> 和“第 4 latent 通道结构能量正跃升不足”两个新候选。公式、AUC、多重检验、限制和
> 第三池计划见
> [BAD_GOOD_METRIC_SCREEN_RESULTS_2026-08-27_ZH.md](BAD_GOOD_METRIC_SCREEN_RESULTS_2026-08-27_ZH.md)。
> 下文保留为研究轨迹和旧协议，不应再被当成最新结论。

## 结论先说

当前还没有一个已经验证成功的 bad-case 检测器，但旧的 64 条 DiT 轨迹里出现了
一个比原跨尺度似然比更具体的线索：4 张一致判为明显模糊、块状的坏图，在成图
之前都经历了两段异常。

1. 较早的 `k=50..99` 阶段，网络预测的干净 latent 在扣除幅度后仍然更粗糙；
2. 随后的 `k=100..149` 阶段，预测干净 latent 的空间幅度又偏大。

把这两项相加后，在同一旧池中得到 AUC 1.0。但是，这 4 个正例来自同一个类别，
都是同一种全局模糊失败；参考中心、时间窗和组合也是看过标签后选出来的。因此
这个数目前只能叫**待证伪的两阶段假设**，不能叫 posterior、质量概率、通用伪影
分数或干预开关。

更重要的机制性观察是：同期的采样状态粗糙度、重建噪声粗糙度和 learned
reverse variance 粗糙度没有复制该信号。这暂时支持“去噪网络对最终干净结构的
承诺发生异常”，而不是“某一步随机噪声整体过大”。接下来必须用新类别、新 seed、
完整轨迹和先锁定的盲标来判断这是真信号还是 4 个样本的巧合。

## 一、先固定什么叫 bad 和 good

本文的 bad 不是“图片不完美”，而是相对于同一冻结模型、同一采样设置的通常水平，
仍然明显更差的输出。主正例只包括：

- 明显的全局或主要区域模糊、块状退化；
- 主要物体或身体结构融合、重复；
- 明显的肢体、身体部位或物体错位；
- 明显的连接、附着或拓扑错误。

轻微瑕疵、风格差异、自然裁切、遮挡、合理的横向尾巴，以及模型普遍都有的普通
缺陷，都不能自动标成 bad。当前旧池的负标签严格说是 `not-clear-bad`，并不等同于
经过单独确认的完美 good；新数据会另外保留 clean-good、mild、clear-bad 和 uncertain。

标签必须先锁定，之后才能与轨迹指标连接。旧 64 条已经解盲，只能用于提出假设和
开发代码，不能再提供确认性 AUC 或显著性结论。

## 二、采样过程和每个字母

令第 (i) 张图在第 (k) 个采样步的 latent 状态为

\[
x_{i,k}\in\mathbb R^{C\times H\times W}.
\]

各字母含义如下：

- (i)：样本编号；
- (k\in\{0,\ldots,249\})：数组里的采样步编号；(k=0) 对应内部扩散时间
  (t=249)，(k=249) 对应 (t=0)；
- (t)：扩散实现使用的内部时间编号，随生成进行而递减；
- (C=4)：DiT VAE latent 通道数；
- (H=W=32)：latent 的高和宽；
- (c,h,w)：分别表示通道、行、列索引；
- \(\bar\alpha_k\)：该步对应的累计信号系数 `alpha_bar`；
- \(\widehat x_{0,i,k}\)：模型根据当前 (x_{i,k}) 预测的最终干净 latent，代码中
  是 `pred_xstart`；
- \(\widehat\epsilon_{i,k}\)：由 (x_{i,k}) 和 \(\widehat x_{0,i,k}\) 重建的噪声预测；
- \(\mu_{i,k}\)：实际 reverse kernel 在该步给出的均值；
- \(\sigma_{i,k}\)：实际 reverse kernel 的逐像素标准差；
- \(\zeta_{i,k}\)：该步真正抽到的标准高斯 innovation；
- (Y_i\)：最终盲化视觉标签，clear-bad 记为 1，确认的 ordinary/clean 记为 0，
  uncertain 不参加主比较。

在 `EPSILON + LEARNED_RANGE + clip_denoised=False` 的 DiT 实现中，

\[
\widehat x_{0,i,k}
=\frac{x_{i,k}-\sqrt{1-\bar\alpha_k}\,\widehat\epsilon_{i,k}}
       {\sqrt{\bar\alpha_k}}.
\]

实际一步写成

\[
x_{i,k+1}=\mu_{i,k}+\sigma_{i,k}\odot\zeta_{i,k},
\qquad \zeta_{i,k}\sim\mathcal N(0,I).
\]

其中 \(\odot\) 表示逐元素乘法，(I) 是单位协方差。`pred_xstart`、
\(\mu_{i,k}\) 和 \(\sigma_{i,k}\) 都在当步 \(\zeta_{i,k}\) 抽取前得到，因此只依赖
这些对象的当步指标是 predictable。使用当步实际 innovation 或相邻两步差分的指标
是 online-causal，只能干预后续步骤或触发回滚。使用未来状态、全程最大值或终点图的
量是 retrospective，只能用于挖掘和解释。

## 三、当前最重要的候选量

把 \(X_{i,k,c,h,w}=\widehat x_{0,i,k,c,h,w}\) 简写为模型的预测干净 latent。
对每个通道先定义空间均值和方差：

\[
\overline X_{i,k,c}=\frac1{HW}\sum_{h,w}X_{i,k,c,h,w},
\]

\[
V_{i,k,c}=\frac1{HW}\sum_{h,w}
\left(X_{i,k,c,h,w}-\overline X_{i,k,c}\right)^2.
\]

这里 (V) 表示该通道的空间幅度。再定义水平、垂直相邻位置的一阶差分能量：

\[
G_{i,k,c}
=\operatorname{mean}_{h,w}(X_{h+1,w}-X_{h,w})^2
+\operatorname{mean}_{h,w}(X_{h,w+1}-X_{h,w})^2.
\]

为简洁起见，上式右边的 (X) 都带相同的 (i,k,c)。(G) 大既可能因为结构更碎，
也可能只是整个 latent 幅度更大，所以不能直接把 (G) 当粗糙度。定义幅度归一化
Dirichlet 粗糙度：

\[
R_{i,k,c}=\frac{G_{i,k,c}}{V_{i,k,c}+\varepsilon},
\qquad
r_{i,k}=\frac1C\sum_c\log(R_{i,k,c}+\varepsilon).
\]

其中：

- (R)：每通道的相对空间粗糙度；
- (r)：四通道平均后的对数粗糙度；
- \(\varepsilon\)：仅防止除零和取对数的很小常数，不是随机噪声。

因为分子、分母都会随通道幅度的平方缩放，(R) 对非零的通道仿射幅度缩放基本
不变。另定义预测干净 latent 的对数幅度

\[
a_{i,k}=\frac1C\sum_c\log(V_{i,k,c}+\varepsilon).
\]

为了只找“相对同模型通常轨迹”的异常，在每个 (k) 上用参考轨迹计算

\[
m_k=\operatorname{median}_{j\in\mathcal R} u_{j,k},
\]

\[
d_k=1.4826\operatorname{median}_{j\in\mathcal R}|u_{j,k}-m_k|+10^{-6},
\]

\[
z_{i,k}^{(u)}=\frac{u_{i,k}-m_k}{d_k}.
\]

这里：

- (u) 可以取粗糙度 (r) 或幅度 (a)；
- \(\mathcal R\) 是预先冻结的参考轨迹集合；
- (m_k) 是该步的参考中位数；
- (d_k) 是经 1.4826 校正的 median absolute deviation，简称 MAD；
- (z^{(u)}) 是相对于模型正常轨迹的稳健标准化偏差。

旧池中事后得到的两阶段分数为

\[
S_i=\frac1{\sqrt2}\left[
\frac1{50}\sum_{k=50}^{99}z_{i,k}^{(r)}
+\frac1{50}\sum_{k=100}^{149}z_{i,k}^{(a)}
\right].
\]

这里 (S_i) 是候选风险分数；第一项是早期预测结构异常，第二项是中段预测幅度
异常。计算完 (k=149) 的 `pred_xstart` 后、抽取内部 (t=100\to99) 的 innovation
之前，(S_i) 已经可用。因此若它以后被独立验证，它在时间上确实可能支持后缀
重采样；现在还不支持。

旧池里的 \(\mathcal R\) 由 59 个 `not-clear-bad` 标签构成，这本身使用了标签，属于
泄漏式的同池参考。确认实验绝不能重算它。直接复现 class-207 假设时必须冻结旧参考
数组；跨类别研究则只能在 discovery 数据上学习类条件或 superclass 条件参考，再原样
带入新的 confirmation 数据。

## 四、不是同一个东西的“边缘、后验和概率”

### 1. 边缘至少有三种含义

- 生成边际密度 (p_k(x_k\mid c))：当前 class (c) 下 latent 的概率密度。仅有 score
  网络通常不能便宜、精确地得到它；概率流 ODE 加 divergence 可以近似，但计算昂贵，
  而且高密度不等于视觉质量好。
- 图像边缘强度：Sobel、Laplacian、FFT 高频能量或 Canny 边缘密度。它们能很好识别
  全局模糊，却不天然识别融合、错位和语义拓扑错误。
- 空间异常边界：某个轨迹差分是否集中在少量 latent tile。它描述局部化，不是概率。

所以代码会同时保留这些量，但不会把图像 edge strength 写成 marginal probability。

### 2. 后验至少有四种含义

- reverse kernel (p_\theta(x_{k+1}\mid x_k,c))：实际采样的一步条件分布；它的
  \(\mu,sigma\) 是可计算的；
- 训练前向过程的解析 posterior (q(x_{t-1}\mid x_t,x_0))：需要真实 (x_0)，生成时
  没有真实终点；
- 预测干净图 \(\widehat x_0\)：是模型的点预测，不是 bad-case posterior；
- 监督风险 \(P(Y=1\mid\mathcal F_k)\)：这是我们最终可能拟合的 bad-case 概率，只有
  在足够多盲标数据和严格外层验证后才有资格这样称呼。

把这四者混在一起会制造虚假的理论保证。当前研究先比较可审计的原始量；若样本量
足够，最后才用稀疏、交叉拟合的模型把它们组合成风险 posterior。

## 五、已经实现的指标族和直觉

1. **预测干净图结构**：(r_{i,k})、(a_{i,k})、原始 (G)、逐通道版本、时间窗均值、
   最大跳变、总变差和 peak-to-terminal。它们问“网络此刻承诺的最终结构是否反常”。
2. **预测时间不稳定性**：

   \[
   J_{i,k}=\frac{\operatorname{mean}(\widehat x_{0,i,k+1}-\widehat x_{0,i,k})^2}
   {\operatorname{Var}(\widehat x_{0,i,k})+
    \operatorname{Var}(\widehat x_{0,i,k+1})+\varepsilon}.
   \]

   (J) 是幅度归一化的相邻预测变化；它在观察到下一次预测后才可用。
3. **状态和噪声控制**：对 (x_k) 和重建的 \(\widehat\epsilon_k\) 计算同样粗糙度，
   排除“所有张量都只是更粗糙”的解释。
4. **reverse-kernel uncertainty**：`mean(log sigma)`、空间标准差及其形态，问模型给出的
   当步 learned variance 是否异常。内部 (t=0) 的随机张量虽然为了保持 RNG 流仍会
   抽取，但会被 nonzero mask 丢弃；因此该点只能叫 `variance-head diagnostic`，不能叫
   实际 stochastic transition uncertainty。操作性 kernel 指标只使用前 249 步。
5. **实际 transition surprise**：

   \[
   \zeta_{i,k}=\frac{x_{i,k+1}-\mu_{i,k}}{\sigma_{i,k}},
   \qquad
   n_{i,k}=\frac12\operatorname{mean}(\zeta_{i,k}^2)
             +\operatorname{mean}(\log\sigma_{i,k}).
   \]

   (n) 是省略高斯常数后的逐维负 log transition density。还检查 innovation 能量是否
   突然集中在某个 4×4 tile。
6. **跨尺度差异**：当前/高噪声预测的 RMS、夹角、空间集中、与实际 innovation 的方向
   对齐、原始 KL 和精确路径 LR。原冻结 LR 已经失败；这些只作为广义候选和对照。
7. **终点图控制**：Laplacian、Sobel、FFT 高频、局部 sharpness、灰度熵、饱和度及
   ImageNet 分类器。它们用于核对标签和解释 subtype，不是预终点检测器。

新完整轨迹记录器已经从官方 `forward_with_cfg` 内部的同一次 raw model forward 捕获
条件/无条件分支，保存两支各自的 4 个 epsilon 通道和 4 个 learned-range 通道，不增加
神经网络求值。CFG disagreement 可能度量“类别条件要求”和模型无条件自然演化之间的
张力，但必须分别控制前三个被 CFG 引导的 latent epsilon 通道、第 4 个未按同样方式
引导的通道，以及 learned variance 通道。严格 seed-16 对照已经同时通过 3 个 CUDA RNG
状态哈希、初态/两半终态/decoded tensor 共 4 个 tensor 哈希和 8 张 PNG 逐像素一致。

## 六、旧 64 条轨迹到底说明了什么

| 量 | 旧池探索性 AUC | 当前解释 |
|---|---:|---|
| 终点 Laplacian variance，低值为坏 | 1.000 | 4 个标签确实都是明显全局模糊；只做标签一致性检查 |
| q1 预测干净图归一化粗糙度 | 0.919 | 较早出现结构异常的线索 |
| q1 预测干净图时间不稳定性 | 0.911 | 相邻结构承诺也更不稳定 |
| q2 预测干净图幅度 | 0.970 | 随后出现幅度 overshoot |
| q2 alpha 补偿梯度能量 | 0.987 | 与幅度/结构线索一致，但同池多重搜索严重 |
| 两阶段 (S_i) | 1.000 | 4 个 bad 排在全部 59 个负例之前；纯假设生成 |
| 跨尺度 epsilon 差的最大负跳 | 0.966 | 353 个搜索量中的事后赢家之一；需要看完整 t60..0 路径，只能回顾分析 |
| q1 状态粗糙度 | 0.381 | 不支持“当前 latent 普遍更粗糙” |
| q1 重建 epsilon 粗糙度 | 0.373 | 不支持“噪声预测普遍更粗糙” |
| q1 learned variance 粗糙度 | 0.445 | 不支持“posterior variance 形态同步异常” |
| 冻结 34 路高噪声 LR running max | 0.441 | 前瞻检验失败，已经退役 |

AUC 的方向统一取“数值越像 bad 越高”；小于 0.5 的原始量仅在探索表中报告反向 AUC。
因为正例只有 4 个，而且总共检查了 353 个标量，任何接近 1 的同池 AUC 都可能被选择
偏差严重夸大。这里没有合法的确认性 p 值，也没有可以上线的阈值。

## 七、接下来的验证怎样防止再自我欺骗

1. 先由互相独立、看不到任何轨迹量的审阅者给新图打 0/1/2/3/U；主 bad 需要至少
   2/3 审阅者判为 2 或 3，clean-good 需要至少 2/3 判为 0。
2. 新 discovery 初始生成 1024 条多类别完整轨迹；只根据锁定标签中的 bad 事件数决定
   是否以 256 条递增，目标至少 60 个 clear bad，最多 2048 条。
3. 先分别检验预先定义的单指标族、异常时间和 subtype；不足 40 个 clear bad 时禁止拟合
   组合。
4. 组合只用 class/superclass 分组的 nested cross-validation，优先 elastic-net 或低阶 GAM；
   非零参数不超过 clear-bad 数的十分之一。
5. 最终只冻结一个主分数和最多两个单指标备份，随后生成完全独立的 confirmation 数据。
6. confirmation 至少需要 50 个、目标 80 个 clear bad；不允许重算时间窗、方向、参考
   归一化、阈值或组合权重。
7. 成功门槛包括 AUC、FPR、TPR、(TPR-FPR)、跨类泛化和相对总步数至少 10% 的中位
   提前量。只在终点有效的量最多是 bad-case mining 工具，不能用于在线修复。

即使某个检测分数通过，也还要单独验证“触发后怎样做”是否提高配对视觉偏好、降低同一
失败 subtype，同时不破坏正常样本和多样性。检测成功不自动推出 guidance 或 rollback
成功。

## 八、可复现入口

- 冻结协议：`experiments/configs/bad_good_metric_discovery_v1.json`
- 旧 64 条探索分析：`experiments/analyze_dit_bad_good_metric_discovery.py`
- 新 custom-batch 全轨迹记录器：`experiments/trace_dit_imagenet256_custom_batch.py`
- 当前不可覆盖的旧池分析输出：
  `/data/users/zhoushunyu/eqvae/cross_scale_evidence/bad_good_metric_discovery/dit_class0207_legacy64_v7`

`legacy64_v1` 到 `legacy64_v6` 已被 v7 明确取代：前两版把按内部 (t=0..249)
保存的 `alpha_bar` lookup table 直接对到了按 (t=249..0) 保存的采样轨迹，并把若干
全路径 reduction 错标成了 predictable/online。该错误使旧版的重建 epsilon 和 alpha
补偿梯度数字无效，但不影响不使用 `alpha_bar` 的两阶段粗糙度加幅度分数。v3 纠正主
时间轴，v4 排除被 mask 的 t=0 假 innovation 并加强锁文件/schema 校验，v5 又精确修正
249 行操作性轨迹在 q4 的最晚可用时刻。最终 v7 通过显式
`alpha_schedule[internal_timestep]` 对齐，并给每个标量保存最晚所需采样步、内部时间、
观测时机及是否仍可在终点前行动；同时把分析源码和协议原文快照进不可覆盖的产物包，
避免未提交工作树中的源码随后变化而无法恢复；并交叉核对 consensus、private mapping
和 public blind pack 确实绑定到同一个 manifest identity。

该输出逐项核对 8 个输入 trace 的 SHA-256、盲化映射、标签计数、trace 解码图与最终 PNG
的逐像素一致性，并保存全部 feature table、时间曲线、图和 manifest/completion 哈希。
