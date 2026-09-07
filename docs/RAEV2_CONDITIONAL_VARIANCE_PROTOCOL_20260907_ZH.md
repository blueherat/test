# 条件反向方差：最后阶段的一个固定候选

2026-09-07，第 2/8 轮提出。提出时前缀密度比的单位强度 5K 仍运行中，原两条 legacy5K 仍排队。本候选只在这些既定质量检验未达到目标后使用剩余额度；提出时没有新数据提取、拟合或采样结果；第3轮实测进展见文末。它不更改原方差方案的参数，原负结果保留。

## 原文改变了什么判断

重新获取并阅读 Bao 等 [ICML2022 正式原文](https://proceedings.mlr.press/v162/bao22d/bao22d.pdf)的 §3–6、Lemma A.1 证明、Appendix E.1–E.2。与最早 Analytic-DPM 的时间依赖球形方差相比，本文明确建模状态条件 covariance；固定均值时还要纳入均值偏差。它用共享主干降低推理开销，但 NPR 的似然收益没有普遍转化为比 SN 更好的 FID。原文已有这个反证，不能将本地扩大条件模型写成质量保证。此次没有审查作者代码或复现表格；不更新全仓“新读论文篇数”的计数。

本地旧公式 `v(t)=q² E||X−G||²/D` 是只依赖 t 的球形族最优值，尚未测试同一统计目标中输入状态的可预测异质性。新设计只补这一项，并利用 native 已计算的特征，避免再次为每个采样节点做前缀输入反传。

## 本地推导：固定 guided mean，学习一个条件方差

令 t>s，a=1−t，q=(t−s)/t。真实线性桥在 deterministic coupling 下的一步目标为

    Y = (s/t) Z_t + q X，
    m_G = (s/t) Z_t + q G(Z_t,t,c)。

部署时保留原 native guided mean，只把反向核写为 `N(m_G, q² m(Z_t,t,c) I)`。记 `R=||X−G||²/D`。条件期望的 Gaussian 负对数似然与 m 有关的部分为

    ℓ(m|z,t,c) = (D/2)[log m + E(R|z,t,c)/m] + 常数。

只要条件风险正且有限，唯一最优值为 `m*=E(R|z,t,c)`。不要求 G 是 Bayes 均值；这仍是给定均值、给定球形条件族的最优性。

对固定 t，理想条件模型相对理想全局模型的 NLL 差额为

    (D/2)[log E r(Z_t) − E log r(Z_t)] ≥ 0，
    r(z)=E(R|z,t,c)。

由 Jensen 得到的严格收益只在条件风险非恒定时存在。**不能用单次带噪标签 R 的散布代替 r 的异质性**；给每个样本直接设 m=R 会把无法由部署输入预测的噪声也拟合进去。因此必须在独立真实样本和噪声上检验。

## 唯一函数族、目标和成本

沿用旧全时间校准 `m0(t)>0` 作基准，设

    mθ(z,t,c)=m0(t) exp(hθ(z,t,c))，
    hθ=θᵀφ̃(z,t,c)。

φ 仍是固定 base_model_depth=8 后的 patch-token RMS 归一化均值和二阶矩，2880 维，加 bias 共 2881 参数。**本次特征直接来自 native BF16 full/base 主干的一次实际前向，池化时转 FP32**；原 density-ratio 训练的纯 FP32 特征不能当作逐位相同的缓存复用。训练和部署都截取相同原生 block 输出，部署不加额外模型查询，也不对输入反传。

θ=0 恰好还原原 global-variance 方案。把 Gaussian NLL 按坐标平均、减去 θ=0 常数，得到一个凸目标：

    J(θ)=mean_i .5[h_i+(R_i/m0(t_i)) expm1(−h_i)] + λ||θ||²/2，
    λ=2881/64000。

其 Hessian 是 `.5 mean[(R/m0)exp(−h) φ̃φ̃ᵀ] + λI`，严格正定。采用相同单位预测尺度的先验和 dim/N 规则、一次从零开始的 L-BFGS-B，原固定 maxiter500/gtol1e−8/ftol1e−13/maxls30；这些是明确建模及工程选择，不称理论唯一值，不搜索正则、层、温度或时间窗。

用已有 64K train/8K heldout 的真实图与已保存 initial epsilon；每个真实行只取一次对应噪声，避免把重复正例当新独立样本。按 global B8 均衡打乱全部原 100 个查询节点（含 t=1），训练/验证时间 seed 固定为 202609109/202609110。先重构真实 forward 状态，再一次 native G 前向同时取得特征与 R；总共 72K 逐样本 teacher 查询，不重走 3,590,360 次 native rollout，也不重新编码真实图。该额外准备成本单列。

特征仅用 train 均值与一个全局 RMS 做标准化，保存所有 source/model/data SHA。纯噪声端有直接训练样本，无需额外手工关闭时间窗。所有 100 个 m0(t) 都来自同一旧估计式，hθ 是同一个共享模型，不是几十段外推系数。

## 预定裁决

1. 数据/特征来源与 native 算术必须对齐；新 θ=0 完整 8 图与旧 `calibrated` 在相同噪声下逐像素一致，official8 也须保持原像素。
2. 唯一拟合须收敛；8K validation 相对原全局方差的 NLL 差，按 1000 类平均后加 2SE 必须低于零。有限值、预测方差分布和全部时间的风险都保留；不挑有利时间窗。
3. 通过后只用最终 θ、原 100 步和全部时刻，完整做固定 1K 与独立 5K，无论 1K FID 正负。沿用 seed202609071/202609072，与原官方及原全局方差比较；保存所有额外成本。
4. 任一入口失败就停止该实现，不改容量、采样倍率、噪声窗或训练时长。第八轮期限不变。

条件 NLL 优于全局 NLL 不保证实际闭环 FID：真实 forward 输入与生成状态有分布差异，球形模型仍忽略方向 covariance，加入 guided bias 对似然正确也可能有损感知质量，原文 NPR/SN 已体现这一差别。本候选保留这些边界；其价值是一次成本较低、假设明确且可被 heldout 证伪的剩余检验。

## 第 3/8 轮实施记录

两条原方案的固定5K已经完成并独立审核：概率校准密度比6.938002（+.1693%）、全局方差6.944997（+.0687%），均未达标，因此触发本预定候选。新增的风险/梯度、解析有偏均值最优MSE、全时间覆盖与零头噪声算术四项检查通过；这仍不替代完整原生8图像素对齐。

提取器使用实际原生BF16-autocast前向的block7输出，转FP32池化；保存所观察到的原始token dtype，不把autocast环境等同于每个中间张量都为BF16。真实latent从原FP16缓存恢复FP32，`X−G`按FP32计算后转FP64平方并按坐标求均值。每个rank/split首批另做一次不带hook的同输入前向，full/base输出须逐位相同；原epsilon首末B8逐位重生成检查。72K主查询之外的64个逐样本parity查询单列。

复用输入的42个文件（模型/配置、真实latent与身份元数据、initial noise及元数据和请求）已完整散列核验，训练/验证真实行无交集；没有再次读取或散列未使用的庞大native states。旧5K审核第一次被新增的非计算`source_law`说明阻断，导致本提取队列在GPU启动前暂停。独立审核修复后，保留首次执行和源码快照，复用刚完成的输入散列记录；没有已完成特征或拟合可供挑选，也没有重采图像。恢复只改控制器，提取器、风险、拟合超参数及其余源码身份必须与首次计划相同。

代码入口：`prepare_raev2_conditional_variance.py` → `extract_raev2_conditional_variance.py` → `fit_raev2_conditional_variance.py`；唯一后续队列为 `continue_raev2_conditional_variance.py`。采样补丁在 `experiments/locks/raev2_conditional_variance_20260907/`；在旧5K及特征父进程真正退出、唯一拟合通过后才可应用。原生official8、原calibrated8和新零头8要求全部像素相同，随后对最终头做固定1K和5K，无论1K正负。

### 唯一拟合与原生检查已完成

固定L-BFGS-B经过292次迭代，最大梯度4.400824e−6，满足原先1e−5准入标准；训练NLL差−.005330815，留出NLL差−.005166167，按1000类SE .000144258、两SE上界−.004877652，通过原门槛。权重范数 .089278756，头SHA `4b714004cd57addae8c833970574c13fd6f3c3addfb9b119a913ae0767108faa`。留出预测方差/原方差的[min,p01,p50,p99,max]为[.509719,.684108,.997103,1.337260,1.618850]；没有裁剪或再次缩放。CPU拟合25.146秒；72000个teacher查询加64个parity查询，提取worker耗时合计517.080492秒（包括数据读取）。原native中间token实测为FP32，来自原生BF16-autocast计算流程，与显式全FP32前缀路径不同。

原official8与旧anchor的sample文件SHA相同；原calibrated8与旧anchor相同，新零头8也与calibrated8相同，三个比较均检查全部8图像素和ID。正式1K已经启动，固定5K接续，不改权重、种子、全部100个时刻或函数。参见[特征身份与费用](../experiments/results/raev2_guidance_20260907/conditional_variance_features_complete.json)、[唯一拟合](../experiments/results/raev2_guidance_20260907/conditional_variance_fit.json)、[完整8图检查](../experiments/results/raev2_guidance_20260907/conditional_variance_smoke8.json)。

为区别状态信息与旧时间均值的校准偏差，另用每个原生time的640条train残差取同一个样本均值公式，仅在CPU上评价8K heldout。新的时间均值相对原时间均值NLL差+.0000200553；条件头相对该新时间均值仍改善−.005186223，按类SE .000144039、两SE上界−.004898144。这个诊断支持收益不只是换了一组时间校准均值，不能当作总体信息论证明。它没有生成新的time-only图像、改变入口门槛或筛选theta，见[风险分解诊断](../experiments/results/raev2_guidance_20260907/conditional_variance_risk_diagnostic.json)。

### 固定1K结果，5K继续

1K独立审核完成：FID **38.54711465462702**，原official为38.486773927092486（恶化 .156783%），历史interval为38.3350240324；原global variance为38.5411998671，新头也略差于这个机制控制。每图100次主调用和100次已有特征读取，零额外主干/前缀前向和输入反传，实测推理成本为原official **1.0369487622倍**。图像SHA `fd6221d9fa81846c06bbd2e0b3cbad96b0323f00d26abef5630cf6bb7909e38c`。全部合并像素/初始noise/标签、EMA/config/decoder/stats、同一θ及source快照核验通过；FID另以rank-N Gram谱重算通过。

[1K完整精度审计](../experiments/results/raev2_guidance_20260907/conditional_variance_screen1k_audit.json)。这说明真实forward条件NLL增益尚未转化为这次闭环FID增益，不能仅由1K给5K下结论。相同θ的seed202609072完整5K已启动，所有100个时刻和采样公式保持不变。采样补丁已经在两个旧父进程结束后按SHA应用；锁目录的`applied:false`是应用前不可变计划，实际激活与像素验证见continuation及smoke8记录。
