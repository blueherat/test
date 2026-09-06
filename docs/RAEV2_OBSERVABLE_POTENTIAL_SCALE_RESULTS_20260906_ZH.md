# 固定势函数：配对 5K 规模审计结果

日期：2026-09-06。**原候选在 5K 上仍没有实际收益证据：相对官方 100 步仅改善 0.00359028%，相对推理成本接近且略高的官方 105 步恶化 0.20444746%。这条有限势函数候选的质量验证线结束，不追加种子、50K 或扩大训练。** 至少 5% 的公平成本提升目标尚未实现，研究目标保持不变。这个停止决定不宣称方法在所有规模上等价或必然无效。

本页记录结果，不修改[冻结规模审计协议](RAEV2_OBSERVABLE_POTENTIAL_SCALE_AUDIT_20260906_ZH.md)。唯一候选仍为 608000 参数的最终势函数，checkpoint SHA256 `495b3313e945f4b845307ae0d520c3c8d8fa70e60e3983002297cd3f7ccc632e`；训练 2048 updates、采样全部 100 时刻加 `∇zΦ`、系数 1 均未改变。理论和有限实现的保证对象仍以[机制](RAEV2_OBSERVABLE_ERROR_GUIDANCE_20260906_ZH.md)及[求解器](RAEV2_OBSERVABLE_POTENTIAL_SOLVER_20260906_ZH.md)为准：理想连续性修正的分布性质不能直接转成有限网络、有限步数的 FID 保证。

## 为什么仍需纠正原 1K 门槛

[有限样本论文与历史特征复核](RAEV2_FID_FINITE_SAMPLE_READING_20260906_ZH.md)发现，历史 IG 在两套完整 5K 上分别改善 8.4659% / 5.0949%，十个类别均衡的 1K 子集却全部不足 5%。因此，“最终改善至少 5%”不推出“1K 必须先改善至少 5%”。此前的操作性必要条件过强；本次阴性结果不会使那个推理重新成立。

旧 1K `37.562703724300 → 37.531664406887`（改善 0.08263334%）及当时停止记录保留。本次是知道旧阴性结果后重新冻结的一次规模审计，没有重训、挑种子、扣除 FID floor 或从 1K 外推 50K。也没有把 `5K≥5%` 新设成所有更大规模成功的数学必要条件。

## 三臂结果与配对证据

| 固定分支 | FID ↓ | Inception Score ↑ | 候选相对此对照的 FID 改善 |
|---|---:|---:|---:|
| official100 | 6.974897847697434 | 160.2887924194336 | +0.0035902811% |
| potential100 | 6.974647429255924 | 160.1660919189453 | — |
| official105 | 6.960417033373517 | 160.81580963134766 | −0.2044474608% |

三组全部采样并合并后，使用同一条 evaluator 命令统一计算。每组 seed `202609101`、5000 张、固定 1000 类各五张，ID `0..4999`，label=`ID%1000`。四个 B8 分片为 1256/1248/1248/1248 张；全部 625 个配对 batch 的初始噪声 SHA 一致，完整噪声 cohort digest 为 `7e51efbf0bb4b906413238c16b337fa63e34162c483f3e14a9ef7a0dd71eb303`。这里核对的是保存的噪声身份与实际分片记录，没有用 CPU 重新生成 CUDA 噪声。

三组共同 request 除 mode、步数及其 time grid 外相同；使用同一官方 stage2 EMA、native BF16 `B+1.78*(F−B)`、IG1.78 原区间、shift8 Euler、`t_eps=.05`、FP32 state、BF16 decoder 和 BF16 clamp/×255 后 uint8。`nanogen-evals` commit 为 `19dfb4c2705333eb8b97e454fb354d47d1fe135b`，reference 为 `imagenet_256_fid_stats`，B64、evaluator seed2020。三组样本不被重排、选择或丢弃。

归档时独立重读并核对了 18 对冻结/当前源码、候选权重和两个 evaluator 资产、12 个 worker、三组所有 batch manifest、样本 SHA、完整 ID/label、计数及时间求和。样本 NPZ 只解压 ID 和 label；图像字节只用于 SHA 校验。所有检查通过；该次 CPU 复核 main wall `3.586663 s`、CPU `6.432934 s`，未调用 GPU。

## 一阶配对不确定性

| 候选对照 | FID 差：候选−对照 | 配对一阶 SE | 相对改善 SE（百分点） | 相对改善的局部 95% 正态区间 |
|---|---:|---:|---:|---:|
| official100 | −0.0002504184 | 0.0344809522 | 0.49433125 | [−0.96529898%, 0.97247954%] |
| official105 | +0.0142303959 | 0.0327806517 | 0.47256441 | [−1.13067371%, 0.72177879%] |

使用固定 reference 的 FID influence function，先按相同 ID 作候选/对照差，再按 1000 个固定类别计算每类五个噪声重复的类内方差；ratio 使用 delta method。没有把五个 1K 子集 FID 当成独立重复。全部 FP64 协方差及夹心矩阵通过数值 SPD 检查；未加 ridge、截断或修复。CPU FID 与官方最大绝对差 `3.789857316860434e−12`，transport 最大相对残差 `1.7441664437063694e−12`。

这些区间是**局部一阶近似**：2048 维、N=5000 时不保证精确的 95% 覆盖率；它们仅描述固定 reference、模型和类别配额内的噪声变化，不包含训练、reference 估计或方法选择不确定性，也不消除模型依赖的有限样本 FID 偏差。区间跨零不能证明等价，更不能替代独立确认、适当规模评估及完整公平成本比较。数学源码原有 96 项低维检查和源 SHA 已核对，没有重复成功测试。

## 推理、准备与研究审计成本

以下采样窗口均为**四个 worker 的时间之和**，单位秒；不是四卡并行 makespan，也不是纯 CUDA kernel time。窗口互有包含，不能把各行相加。

| 计时窗口 | official100 | potential100 | official105 |
|---|---:|---:|---:|
| 完整轨迹 | 3149.843292 | 3295.046910 | 3306.492899 |
| 解码及 uint8 | 17.337507 | 17.415603 | 17.354997 |
| sampling 含输出 | 3188.435792 | 3332.429220 | 3343.607007 |
| 公共 backbone/decoder 加载 | 44.431253 | 52.046953 | 44.080626 |
| 势 checkpoint 加载 | 0.052725 | 0.062760 | 0.046165 |
| 噪声生成、拷贝与审计 | 26.573317 | 25.889316 | 25.642996 |
| worker 主程序 wall，缺 imports/末尾 summary | 3306.741633 | 3459.022358 | 3459.359293 |
| worker 主程序 CPU，同一边界 | 3358.952906 | 3495.742535 | 3511.681223 |
| 外层 worker wall 和，含导入/退出及观察延迟（poll 间隔设为 1 秒） | 3335.594589 | 3486.733754 | 3487.776980 |
| 外层四卡采样阶段 elapsed | 842.917208 | 882.952028 | 881.962727 |
| 外层采样子进程 CPU | 3388.259271 | 3522.662990 | 3540.788768 |

候选完整轨迹比 official100 多 **4.609868%**；official105 比候选多 **0.347370%**，含输出 sampling 窗口多 **0.335425%**。105 步源于读取本轮 FID 前冻结的 B8 基准 ceiling，不是按本轮质量挑出的步数。实测说明它是预算略高的推理对照；外层并行 elapsed 的轻微次序差异不能代替 worker 总成本。所有分支都加载了未必使用的势 checkpoint，官方分支实际势调用为零；其加载时间属于脚本共有冗余，不是官方算法所需的部署开销。

| 实际调用 | official100 | potential100 | official105 |
|---|---:|---:|---:|
| stage2 batch forwards / sample forwards | 62500 / 500000 | 62500 / 500000 | 65625 / 525000 |
| 势 forward+输入梯度 batch calls / sample calls | 0 / 0 | 62500 / 500000 | 0 / 0 |
| 采样期间势参数反传 | 0 | 0 | 0 |
| decoder batch forwards / decoded samples | 625 / 5000 | 625 / 5000 | 625 / 5000 |

各组最大 PyTorch GPU allocated 均为 `10496353792 bytes`；这包含完整 5K 噪声 cohort 的内存峰值，不是单图部署内存。三次**完成的** CPU merge 含导入/退出 wall 为 `19.549953 / 17.441803 / 16.919293 s`，CPU 为 `19.811021 / 17.738860 / 17.222542 s`。共同 evaluator 处理全部 15000 张图，wall `27.614767 s`、CPU `68.520224 s`。本轮配对 IF 分析含导入/退出 wall `14.966121 s`、CPU `46.830308 s`，最大 RSS `1142196 KiB`，GPU/模型/新特征提取调用均为零。中断过的一次 CPU merge 没有完整计时，见下一段，不能把这部分研究成本记为零。

方法必要准备仍只有已记录窗口：训练 `564.0549605630804 s` 加训练 5K 编码及分配给它的共享开销 `96.46438909810968 s`，合计 **`660.5193496611901 s + U`，U≥0 尚未闭合**。训练涉及 65536 个 stage2 sample-forwards；没有在这轮 5K 审计中再次训练或编码。验证专用编码、原始训练/验证/benchmark、prefix 重放及既有诊断成本仍保留在[成本协议](RAEV2_OBSERVABLE_POTENTIAL_COST_PROTOCOL_20260906_ZH.md)、[旧筛查](RAEV2_OBSERVABLE_POTENTIAL_SCREEN_20260906_ZH.md)和[补空间审计](RAEV2_POTENTIAL_OMITTED_WITNESS_20260906_ZH.md)。原公式在 N=5000 下仅给 `K_total≥126`，126 不是完整总成本匹配；没有运行该对照。**本轮不能称为总成本公平达标。**

## 中断与续接

原运行从 `06:21:41 UTC` 开始，官方 100 步采样及 merge 正常完成；候选四个 worker 最晚在 `06:50:46.831 UTC` 全部以 return code 0 退出。`failure.json` 随后记录 `06:50:53.269 UTC` 收到 signal15。候选第一次 CPU merge 的日志为空，且没有完整 process 成本记录；没有据此推断它的计算成本为零。

续接在 `06:57:11.164 UTC` 重新核验同一个根 request、冻结源码、权重和资产。`resume_runner.py` 对前两臂只复用已完成的采样；官方100复用完整 merge，候选重新做 CPU merge，然后首次启动冻结的 official105 四个 worker。源码的复用分支、八个中断前 worker 的退出时间/日志和四个续接后 worker 的记录相互一致；没有重采样前两臂或改种子。原 failure 与第一次日志均保留。

最终共同 evaluator 于 `07:12:59.771 UTC` 以 return code 0 结束，`execution_summary.json` 为 complete=true。曾观察到的 resume PID 已正常消失后，完成文件与成功子进程记录共同证明任务完成；没有因观察窗口结束自动重启作业。`outer_resume_wall_seconds_after_identity_checks=948.608595` 只覆盖续接段，不能当作整个实验耗时。

## 归档入口与决定

原始实验目录为 `/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/observable_potential_scale_audit_v1`；CPU 分析为相邻 `observable_potential_scale_audit_analysis_v1`，其中保存源码、精确命令、输入 SHA、全部 IF、日志与外层成本。

| 证据 | SHA256 |
|---|---|
| [冻结 request](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/observable_potential_scale_audit_v1/request.json) | `48fb15248083680e95f647c3bb1dea8da85031ca6c0570a88b4f01aa74667b08` |
| [完整执行](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/observable_potential_scale_audit_v1/execution_summary.json) | `e43c235a77115d29b4c131e1659d5551a6eec8cbc178b86437d649809d9a8fbe` |
| [官方 metrics](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/observable_potential_scale_audit_v1/evaluation/metrics.json) | `37a32708c9d26b4d277a6b902ae5af8ee0a289c1e97c0e84a57df6f78e0ad4f3` |
| [续接来源](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/observable_potential_scale_audit_v1/execution_resumed.json) | `91783696051522cafecb1a5bef0c4d5de1f9f11e73757224443ef4774de652de` |
| [配对 IF 结果](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/observable_potential_scale_audit_analysis_v1/results_v1/summary.json) | `d0962f0f4d6a08675c6af2392298e7a96bd37af2c01085ec7f8379b11c276c14` |

预留 seed `202609102` 没有使用，不做该候选的独立确认、50K、全成本额外步数对照、宽度/训练量扩大或手调时间权重救援。当前证据支持结束这条有限实现的质量实验线，同时保留合理的 1K 筛查规模修正。后续 guidance 设计仍须从新的、可证伪的误差机制自然导出；不能把本次代理误差下降或阴性 FID 换一种名字继续调参。
