# 独立 5K 配对 FID 不确定性工具

实现：`experiments/cfg_transport_search_20260913/paired_fid_uncertainty.py`。当前只完成 CPU 小矩阵自检，没有使用 GPU、没有对当前 1K 特征做正式 bootstrap，也没有生成 5K 结论。

## 估计对象与旧实现的差别

固定现有 ADM reference 的 `mu/sigma`，比较同一套 5K noise/label 下候选与 APG、CTRL 的 **pool_3 FID 差值**：

\[
\widehat\Delta_j=\operatorname{FID}(X^{\rm cand};\hat\mu_R,\hat\Sigma_R)
-\operatorname{FID}(X^j;\hat\mu_R,\hat\Sigma_R).
\]

负值偏向候选。每类固定 50 个输入，100 类共 5K；每个 bootstrap replicate 在各类内部各有放回抽取 50 个行号，所有方法共用这组行号。计算的是全体 5K 的 pooled FID，不是 100 个 per-class FID 的均值。协方差分母始终为 `N−1`，与 ADM 一致。

旧 `experiments/analyze_imagenet100_sit_terminal_distribution_audit.py::_bootstrap_distribution_metrics` 每边独立无放回抽半数，比较两组经验特征分布并汇报均值/标准差。它没有固定真实 reference、没有保留共同 noise 配对、没有固定类计数，也没有构造差值区间，不能直接复用为本任务的确认统计。

预先固定主读数为 **中心化 bootstrap 的 Monte Carlo 不确定性区间**。若各次差值为 \(\Delta_b^*\)，令 \(e_b^*=\Delta_b^*-\overline{\Delta^*}\)，则名义 95% 区间为

\[
[\widehat\Delta-q_{.975}(e^*),\;\widehat\Delta-q_{.025}(e^*)].
\]

这个区间使用 bootstrap 的波动形状与尺度，主动把 bootstrap 均值漂移单独报告；它不等于对有限 N 偏差的校正，也没有经实证校准的 95% 覆盖率。文件同时报告标准 percentile 区间 `[q.025(Δ*), q.975(Δ*)]`、basic 区间 `[2Δhat−q.975(Δ*), 2Δhat−q.025(Δ*)]`，用于检查结论是否依赖均值漂移。不能事后选择最有利的区间。

两个 baseline 同时比较时，另报 Bonferroni 调整：每个比较名义 97.5%，两端分位数为 .0125/.9875。默认 500 次每侧只有约 6.25 个 tail draws，尾分位数本身仍有 Monte Carlo 误差；这不是需要精确 p 值的工具。

## 必须限制解释

1. **不是 FID∞ 或无偏质量差。** 有限样本 FID 的偏差依赖生成器，同样 N 不会保证差值偏差抵消；配对主要降低随机输入导致的差值方差。此问题的原始来源是 [Chong & Forsyth, CVPR 2020 / arXiv:1911.07023](https://arxiv.org/abs/1911.07023)。本工具没有实现该论文的无限样本外推。
2. **只对当前经验 reference 条件化。** 只有参考均值/协方差，不能重采样真实图像，也不能估计 reference 本身的不确定性。不同模型可能对同一参考抽样误差反应不同，所以配对也不能消除这个遗漏。
3. **不含搜索与训练的全部不确定性。** 1K winner 不能作为独立确认；需要冻结候选及 baseline 后使用新 seed/新 source bank。区间不覆盖训练 seed、类别分布、特征提取器选择，也不补偿未声明的多候选、重复试 seed 或失败结果筛选。
4. **高维有限样本 bootstrap 是近似。** 5K 对 2048 维仍不宽裕；bootstrap 的重复样本改变协方差谱，尤其当比较方法接近或低秩时，不能据此承诺理论覆盖率。中心化、percentile、basic 的明显分歧需要原样报告。
5. **不是完整图像质量。** 此处只声明固定 ADM pool_3 FID 读数的稳定性，不覆盖语义正确性、感知伪影或人类偏好。

## 身份检查的可证明边界

现有 `compute_adm_fid.py::_save_activations` 只保存 `pool_3/spatial`，没有 label/noise ID。因此不可能从缓存本身检测 **绑定前、类内、只重排特征** 的静默错误。这一点不能用事后附上共同 ID 来伪装解决。

当前可执行方案依赖经过审阅的 runner/extractor 顺序：runner 按 `start` 递增汇集图像，核对 batch 的 noise digest 和标签；ADM 顺序读该 `samples.npz`。工具要求显式 `--attest-ordered-extraction`，核对 request/config/input/image/FID 路径与 hash，绑定特征文件 hash 和共同 source ID sidecar。之后的缓存修改、有 ID 的类内重排、不同输入银行或 label 顺序差异都会被拒绝。若未来缓存原生包含 `row_ids/labels`，也会直接核验。

`plan` 只接受同一 frozen stage 中的候选及 baseline，要求 100×50、新 seed，且与声明的旧 selection stage 没有重复 noise+label source ID；要求选定臂的确认 FID 尚未出现。声明所有参与选参的 stage，例如 baseline_1k、blend_controls_1k、modulation_1k。哈希能核对源输入是否重用，统计独立性仍依赖新 seed 生成流程与实验纪律。

## FP64 计算与成本

一次计算 \(S=\hat\Sigma_R^{1/2}\)，每臂预变换 \(Y=XS\)。每轮 bootstrap 用整数重复计数 \(w_i\) 计算

\[
\mu_X={\sum_iw_iX_i\over N},\quad
\operatorname{tr}\Sigma_X={\sum_iw_i\|X_i\|^2-N\|\mu_X\|^2\over N-1},
\]
\[
M=S\Sigma_XS={Y^T\operatorname{diag}(w)Y-N\mu_Y\mu_Y^T\over N-1}.
\]

FID 为 `||muX−muR||² + trSigmaX + trSigmaR − 2*sum(sqrt(eigvalsh(M)))`。计算全部 FP64；只裁剪容差内的微小负特征值，不加改变度量的 ridge。正式开始前，每臂点值必须与已有 ADM `fid.json` 在预设 `1e−3` 内相符，否则中止并调查。原 ADM 对 float32 features 的均值累加与这里的 FP64 均值略有差异；目前这个全维容差检查尚未实际运行。

500 次、3 臂意味着 1500 次 2048×2048 对称特征值分解，以及每次约 `2*N*D²` 的协方差乘法，另加参考平方根和点值。三臂的 X/Y FP64 常驻主数组约 492 MB；求解器还需要工作空间。4090 的 FP64 吞吐有限，不能根据图像生成速度推算这一步。`benchmark` 先做 2 次完整配对 draw，输出实际每次耗时与 500 次线性外推；预计应按分钟级至更长的独立计算安排，而非承诺秒级。此处尚未占用 GPU 做测速。

## 使用接口

以下臂名与路径为待替换占位，先运行 runner 的正式 5K `prepare`，在确认 FID 出现前创建计划；脚本源码由计划 hash 冻结后不要修改。

```bash
/home/zhoushunyu/miniconda3/envs/myenv/bin/python -m experiments.cfg_transport_search_20260913.paired_fid_uncertainty plan \
  --stage /home/zhoushunyu/data/eqvae/experiments/cfg_transport_search_20260913/CONFIRMATION_5K \
  --output /home/zhoushunyu/data/eqvae/experiments/cfg_transport_search_20260913/CONFIRMATION_5K_UNCERTAINTY \
  --candidate CANDIDATE_ARM --baseline APG_ARM --baseline CTRL_ARM \
  --selection-stage /home/zhoushunyu/data/eqvae/experiments/cfg_transport_search_20260913/baseline_1k \
  --selection-stage /home/zhoushunyu/data/eqvae/experiments/cfg_transport_search_20260913/blend_controls_1k \
  --selection-stage /home/zhoushunyu/data/eqvae/experiments/cfg_transport_search_20260913/modulation_1k \
  --reps 500 --seed 202609135001

# 等 5K 三臂的 samples、endpoints、summary、FID 和 features 全部完成：
/home/zhoushunyu/miniconda3/envs/myenv/bin/python -m experiments.cfg_transport_search_20260913.paired_fid_uncertainty bind \
  --output /home/zhoushunyu/data/eqvae/experiments/cfg_transport_search_20260913/CONFIRMATION_5K_UNCERTAINTY \
  --attest-ordered-extraction

# 仅在分配到空闲 GPU 后执行；CUDA_VISIBLE_DEVICES 映射由操作者安排：
/home/zhoushunyu/miniconda3/envs/myenv/bin/python -m experiments.cfg_transport_search_20260913.paired_fid_uncertainty benchmark \
  --output /home/zhoushunyu/data/eqvae/experiments/cfg_transport_search_20260913/CONFIRMATION_5K_UNCERTAINTY --device cuda:0

/home/zhoushunyu/miniconda3/envs/myenv/bin/python -m experiments.cfg_transport_search_20260913.paired_fid_uncertainty run \
  --output /home/zhoushunyu/data/eqvae/experiments/cfg_transport_search_20260913/CONFIRMATION_5K_UNCERTAINTY --device cuda:0
```

最终 `uncertainty.json` 分开记录 bootstrap 时间、样本生成时间和每图模型调用预算，不读取 runner 表格中被 FID 文件路径覆盖的 `samples` 字段。`bootstrap_values.npz` 留下每次各臂 FID，方便核对差值；共享重采样计数序列由冻结 seed 和 `counts_sha256` 复核。

## 本次已完成的验证

CPU self-test：48 样本、8 维、4 类，每类 12 样本，32 次 bootstrap。

- 同方法所有差值严格为 0，区间 `[0,0]`。
- 每轮类计数保持不变；类内 paired ID 重排被拒绝；特征重排改变绑定 digest。
- 与从 `train_gen/evaluator.py` 原样 AST 读取的 ADM class 比较：SPD 点值误差 `1.27e−14`，有放回 weighted 重采样最大误差 `3.38e−14`，奇异协方差误差 `8.58e−9`。
- CUDA 未初始化。尚未声称验证 2048 维真实缓存的数值或速度。

可复核 JSON：`/home/zhoushunyu/data/eqvae/experiments/cfg_transport_search_20260913/paired_fid_uncertainty_selftest.json`。
