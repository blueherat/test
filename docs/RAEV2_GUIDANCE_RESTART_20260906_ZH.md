# RAEv2 guidance：开放机制探索，2026-09-06

**最新目标已重新设置，见 [当前目标](RAEV2_GUIDANCE_GOAL_20260906_ZH.md)：理论自洽优雅，导出去噪轨迹内 guidance 设计，在公平计算成本下 FID 至少降低 5%，并独立确认。以下保留研究经过；与当前目标冲突的旧条目不再适用。**

## 当前目标与硬约束（用户第二次修正后）

**目标：提出有可证明、可检验的理论机制支撑的 guidance，在冻结的 RAEv2 官方协议上首先争取配对 FID-1K 相对降低约 5%，随后做独立样本确认。**

1. 不绑定 NeurIPS 2025 fixed-point 或仓库 PFR；两者只作可放下的参考，不能成为研究主线的先决条件。
2. 机制先于设计与实验：必须说明所保证的数学对象、成立条件，以及它与模型误差或生成分布质量之间的联系；单纯正交、残差下降、算子收敛或 oracle 恒等式不能冒充质量保证。
3. 禁止用大量手动调参争取目标；尤其不手动设计、搜索外推系数随时间的变化或拼接 guidance 窗口。只有由明确理论或结构自然导出的规则才可成为方法组成部分。
4. 原官方配置（`IG=1.78, t in [.1,1]`）作为冻结基线，不重新优化其系数/窗口。旧分段 IG 结果仅为历史实验事实，不作为本轮新设计的依据。
5. 5% 是必须用真实实验验证的研究目标，不是尚未建立的普适 FID 保证；达不到时如实保留阴性结论。
6. **用户进一步明确：方法必须是在去噪轨迹内起作用的 guidance。** 多生成完整图像再筛选、排序或拒绝采样不属于本任务；不能用通用密度重加权的 KL 恒等式替代对 RAEv2 guidance 的误差机制解释。

## 最新机制检查

[法向与曲率检查](RAEV2_NORMAL_CURVATURE_AUDIT_20260906_ZH.md)、[深度与读出交叉检查](RAEV2_DEPTH_READOUT_AUDIT_20260906_ZH.md)和[线性密度去污染检查](RAEV2_DENSITY_DECONTAMINATION_AUDIT_20260906_ZH.md)均已完成。它们没有支持相应候选进入 1K 质量实验；具体假设、测量范围与计算成本见各归档。这些是机制检查，不能写成已测 FID 结果。当前仍未达到公平成本下至少 5% 的目标。

## 阶段记录：独立校准与径向分布机制

- 原逐坐标 convex proximal 候选已完成 A/B 各 1000 张、全部 100 步的真实模型审计。只有 23 步留出 coupling 风险改善；74 步的 Bonferroni 正态近似区间明确显示恶化。未进入 FID，不用未加权局部 RMS 总和的 0.196% 改善掩盖逐步失败。
- B 的历史 encoder 缓存存在 TF32 数值差异，已按官方 FP32/no-TF32 重新编码，固定样本与独立重编码逐哈希一致；A/B 的真实训练行互斥。
- 曾推导 channel 共享凸锥以降低统计方差，尚未在 C 或 FID 上试验。随后得到更强的 **global zero-anchor** 机制：每步只有一个非负标量，由原始矩闭式确定，无均值偏移、无人工幅度或时间选择。对任意有限二阶矩分布，它的总体版本保证从真实输入边缘出发的单步实际 W₂ 不增，不要求高斯或零均值。这仍不保证一般递归采样的终点 FID。
- 在 C 的任何 Stage-2 结果出现前，依据上述更强定理，将 global 设为唯一主候选；channel 方案只保留理论，diagonal 在 C 仅用于复现失败。C 的 1000 类各一图由固定种子均匀选取，排除 A/B 全部训练行，没有按损失或指标挑图。
- C 全 100 步审计通过：39 个活动步的 H 同时置信下界全部为正，61 个非活动步精确恒等。非零斜率最大约 `7.556e-6`，没有人为放大。
- 冻结 `seed=202609066` 的配对 FID-1K：官方 `38.3977875`，global proximal `38.4083633`，相对降低 `−0.02754%`，目标未达到。噪声/标签/配置/权重/decoder/评价协议全部一致。保持阴性结论，不继续搜索此候选的幅度或窗口。
- [能量球机制](RAEV2_ENERGY_BALL_GUIDANCE_20260906_ZH.md)在上述 FID 出现前已独立固定：每步对实际完整 1000 粒子 cohort 做单向二阶矩球投影，目标只取 A 的 clean 矩及 bridge 公式。真实 GPU 的 16 图 loop 重排核验中 endpoint 与像素逐位一致；完整配对 1K 的 FID 为 `38.4086042`，相对降低 `−0.02817%`，同样未达到目标。它对经验分布有直接实际输入保证，但有限数据矩、粒子边缘与终点 FID 的边界确实不能忽略。
- 上述两个径向结构停止扩展，不手调强度或窗口救结果。随后尝试的 accept-D 属于生成后拒绝采样，偏离用户所要求的 guidance；用户指出后已停止四个 GPU 任务（832 个 proposal，321 个接受结果），未完成配对 cohort，也未计算 FID。其独立 KL 证书不构成本任务中机制设计有效的证据，该方向退出。

产物根目录：`/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/`。
原逐坐标审计：`proximal_calibration_seed202609062/merged/summary.json`；新独立确认：`proximal_global_c_seed202609065/`；冻结配对采样：`proximal_seed202609066/`。

## 已执行的范围纠正

含手选 `cfg=1.15` 和 `t>.5` 的 semantic/quality 两分支已停止，各完成 56/1000；保留逐批产物，未计算 FID。routing 两分支也已停止，末次进度 ordinary 704/1000、preserve_self 512/1000；原 runner 只最终保存完整样本，因此保留的是配置、日志、进度与预览，没有完整样本包。四张 GPU 均已释放，停止原因为 `user_scope_revision`。

随机校正模块仅完成数学审计；手选 `eta=.25` 与 `(.5,.95)` 窗口不符合当前方法准入，未启动真实 RAEv2 采样。

以下为第二次修正前的探索计划，**已被上述约束替代，不是待执行队列**。

## 初始计划留档（已替代）

用户在本轮明确修正：不必沿 NeurIPS 2025 fixed-point 或 PFR；二者可能是死路，只可借鉴，不应成为研究约束。保留的目标是从机制出发设计 guidance，先在 RAEv2 配对 FID-1K 上争取约 5% 的相对降低，再决定扩大验证。1K 结果不称为公开 benchmark SOTA。

当前使用冻结的官方 DINOv3-L K7、EMA step 100080、官方 decoder、100-step shifted Euler。此前最强可复现轻量基线是只在 `t > .5` 使用 `IG=1.78`。原始 `t>=.1` 官方 IG 作为额外基准；新方法的主要比较对象仍是较强的分段 IG。

旧 PFR 的两个 1K 正结果在 5K 反转，因而同 seed 筛选与独立 seed 确认必须分开。原始结果、失败和计算量全部保留；指标不进入在线 guidance。

## 从已失败路线中释放出来

已经阅读 `RAEV2_GUIDANCE_EXPLORATION_ARCHIVE_20260905_ZH.md`、`RAEV2_PFR_TRANSFER_RESULTS_ZH.md` 及 decoder/IG 机制记录。它们否定了多种局部投影、跨时一致、未来流和固定点修补的直接质量推论。新一轮首先测试不同的信息来源或动力学，而非继续微调同一 PFR 项。

### A. 类别证据与深度证据的双轴对照

官方 checkpoint 使用 `.1` conditional dropout，因此在相同状态有四个受训练的预测：`F_c, B_c, F_u, B_u`。

记 `d=F_c-B_c`，`c=F_c-F_u`。加法控制为

`clean = F_c + (beta-1)d + (omega-1)c`。

精确 score 的理想条件下，`c` 对应类别后验的梯度，而 `d` 是模型深度对比；有限网络的 `d` 不被预设为合法密度比。这只是两个现有 guidance 的基线组合，不作为新方法贡献。

进一步可检验 `c` 是否在 `d` 以外保留有用信息：`c_perp=c-<c,d>d/||d||²`。这是受约束局部线性目标的欧氏正交方向，只保证瞬时内积为零，不保证终点质量或全程不干扰。与 additive 和四角 bilinear 对照一起测试；若不超过 additive，不能以几何形式宣称创新。

所有分支采用相同 conditional/null 拼接布局、FP32 guidance、相同初始噪声和标签，避免 BF16 或 batch-layout 改变冒充效果。主筛查先用温和额外 CFG（`omega=1.15`），IG 仍为 `beta=1.78, t>.5`。先看效应方向，再依据结果选择下一次有判别力的实验。

### B. 用空间路由构造负参考

复用仓库已经完成数值审计但没有正式 1K 结果的 routing reference：最后 DDT decoder attention 保持每行 self mass，把 off-diagonal 质量均匀分配。给定 self mass 时它是最大熵行分布；该性质不等于图像质量保证，也不声称注意力扰动 guidance 的新颖性。

机制问题是：深度头缺失的信息与空间路由缺失的信息是否产生不同质量效果？固定逐样本 correction 范数为普通 IG 的范数，固定 `beta=1.78` 与 `t>.5`，仅更换方向。首个配对 bank 为 `seed=202609061`、每批 8、每类 1 张，共 1000 张。比较 ordinary 与 `preserve_self`，必要时用 PAG/identity 作后续对照。

### C. 随机校正作为独立候选

归档中的大量失败发生在确定性 ODE 上。可从 full conditional score 构造配对的 score drift + diffusion，检验是否能改善 guidance 引起的质量/覆盖折衷。先推导有限步稳定实现并用 Gaussian oracle 验证保持分布的性质；不能把纯加噪的偶然改善误认为 Langevin 校正机制，也不能把精确 score 的定理套到有限模型。

## 评价与产物

- 主指标：官方 `nanogen-evals/fd_evaluator`、`imagenet_256_fid_stats` 的同规模 FID；同时保留 IS。
- 记录所有 noise/label hashes、源代码与 checkpoint SHA、计算量、随机 schema、样本量。
- 若候选达到约 5%，固定配置，用独立新 bank 做同噪声配对确认；然后视结果扩大至 5K。
- 数据目录：`/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/`。

## 外部依据

- [RAEv2 原论文](https://arxiv.org/html/2605.18324v1)：官方机制、已训练 CFG 与内部头、评价协议。
- [Foresight Guidance 原论文](https://proceedings.neurips.cc/paper_files/paper/2025/hash/b56d827a2b8433517e722e0272c7f464-Abstract-Conference.html)：作为已阅读的设计参考；本轮不以其固定点为目标。
- [AutoGuidance 原论文](https://users.aalto.fi/~laines9/publications/karras2024autoguidance_paper.pdf)：质量与条件引导的区别；组合基线不主张新颖性。

本文件在本轮新质量结果产生前建立，后续结果应另列并保留阴性条件。
