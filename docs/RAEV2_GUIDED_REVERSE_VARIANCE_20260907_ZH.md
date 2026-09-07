# 固定 guided mean 的反向方差：第二个可实施检验

记录在首轮随机 guidance 的新 FID 出现之前。没有调整外推系数，官方 IG=1.78 及区间 [.1,1] 保持一致。本候选只针对有限步 Gaussian 反向核中遗漏的方差，采用与 guided mean 配套的统计最优值。

来源：[Analytic-DPM，ICLR 2022，§3 Theorem 1、Eq.8](https://arxiv.org/abs/2201.06503)；其扩展 [Estimating the Optimal Covariance with Imperfect Mean，ICML 2022](https://arxiv.org/abs/2206.07309)。本地已读取前者的 Gaussian 矩匹配、全方差及 Monte Carlo 部分。以下直接从给定均值的高斯负对数似然推导，不要求 G 是精确 score。

## 固定均值下的解析解

时间定义、r/q/σ₀ 见[第一轮协议](RAEV2_GUIDANCE_GOAL_20260907_ZH.md)。给定真实 clean X 和 noisy Z_t，前向通道定义的反向目标为

    Z_s = r Z_t + q X + σ₀ ξ。

部署时固定均值 m_G=r Z_t+q G(Z_t,t,c)，只选择全局各向同性方差 v。期望负 log-likelihood 与 v 有关的项是

    L(v) = (D/2) log(v) + [D σ₀² + q² E||X−G||²]/(2v)。

唯一最优解（非零风险）为

    v* = σ₀² + q² MSE_G(t)，  MSE_G=E||X−G||²/D。

这个结论允许 G 有偏、包括 IG 引入的偏差；不把“模型均值误差”等同于 Bayes posterior variance。它最小化真实 forward-pair 下、**这个固定均值和球形方差族**的反向交叉熵，不声称实际闭环 FID 或 KL 必然下降。使用 guided MSE 而非 unguided F 的 MSE，正是本候选与 guidance 的联系。

先检验 η=0 的最小版本：r=s/t，q=(t−s)/t，σ₀=0，因此

    Z_s = official_Euler(Z_t,G) + q sqrt(MSE_G(t)) ξ。

仅一个新噪声张量，无额外模型调用、无新增人工强度或时间窗；这是 variance-calibrated DDIM / aDDIM 思路的 RAEv2 IG 适配，不冒称首创。

## 为什么这不是逐段调参

100 个数是同一冻结估计式在 100 个原采样查询时刻的 Monte Carlo 统计量，每个用全部 1000 个既有真实 forward 样本的 MSE 均值。没有优化 FID，没有人工改值，没有选择时段，也没有可调外推系数。每个值都可以直接追溯到原始残差数组并重算。

源：`/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/observable_potential_validation_v1/shard*/step*.npz`。这些是 ImageNet **train** 中保留的校准图像，在以前的辅助场研究中叫 validation；本次使用其原生 G 的 `residual_mse`，不使用辅助场 correction、FID、Inception 或生成图。源模型/decoder/config 与当前官方基线相同。原校准 batch32/TF32-off，与部署 B8/TF32-on 不是逐位相同；其统计迁移误差需披露。独立采样 seeds 不参与方差估计。

作为背景诊断，已对既有 5000 train clean latents 汇总 channel covariance（所有空间 token 池化）：trace≈1025，有效秩≈396/1024，最大/最小特征值≈15.30/0.00122。它说明整体单位方差不等于各向同性。该 covariance 不进入当前方法；保留球形估计的局限，不从中另选频带或强度。

## 预定验证

- 先 CPU 验证：精确 Gaussian denoiser、任意非零数据方差下，完整反向转移包含 q² posterior variance 后，逐步边缘方差精确；有偏均值时验证固定 mean 下的交叉熵最优点。
- 只测试上述 η=0 一个方法，不扫描噪声倍率；沿用 seed202609071、B8、1000 图及官方 evaluator，与首轮 official/piecewise 配对。
- 若有方向信号则冻结并在 seed202609072 的 5K 确认。统计校准成本与在线成本分别列出。不能因最优交叉熵定理宣称 FID 成功。

## 第 3/8 轮：按用户要求补做固定 5K

原 1K FID38.5411998671 略差于 official38.4867739271；未据此排除 5K。相同旧校准、全部 100 步与原噪声公式在 seed202609072 的完整 5K 得到 **6.944996552691919**，对照 6.9497684777115865，改善 **0.06866307899%**；推理成本为原版 **1.0004617144 倍**。独立对称矩阵 FID、全部合并像素、各类初始噪声、模型/decoder/config 和旧参数逐项核验通过，见[两条旧 5K 审计](../experiments/results/raev2_guidance_20260907/final8_legacy5k_audit.json)。

它确实出现了“1K 略差，5K 略好”，但远未达到 3%。两个规模同时改变样本数和 seed，不能把差异只归因于 N。本方案至此保留为固定控制，不再调噪声强度或时段；[条件方差候选](RAEV2_CONDITIONAL_VARIANCE_PROTOCOL_20260907_ZH.md)检验的是状态可预测的风险差异，原全局结果不被替换。
