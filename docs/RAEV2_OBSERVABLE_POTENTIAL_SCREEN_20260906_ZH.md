# Observable potential：配对 1K 筛查未达标

> **后续评估更正（2026-09-06）**：下文保留首次 1K 的原规则、结果和当时停止决定。有限样本论文与历史特征复核说明，1K≥5% 不是最终目标的必要条件；现冻结原权重补做配对 5K。新执行状态以[规模审计协议](RAEV2_OBSERVABLE_POTENTIAL_SCALE_AUDIT_20260906_ZH.md)为准。旧结果未改变，尚无公平成本达标结论。

日期：2026-09-06。**固定候选 FID 从官方 100-step 的 37.562703724300 降至 37.531664406887，相对下降仅 0.08263334%，未满足预注册的至少 5%。按第一阶段停止规则，终止这个固定有限求解器。** 不继续官方 105-step、完整准备摊销成本对照或独立 seed 确认；没有公平成本达标或 SOTA 结论。

真实来源为 `observable_potential_screen_v1/` 的两组 completed merged samples、同一次官方 evaluator 结果及根筛查 summary。[机器归档](data/raev2_guidance_restart_20260906/observable_potential_screen.json)保存全部请求、指标、125 对 batch 清单、成本、源 SHA 和独立 CPU 核对；本次归档没有重新采样或重算 FID。

**预先冻结的候选和必要条件。** 候选是 [固定求解器](RAEV2_OBSERVABLE_POTENTIAL_SOLVER_20260906_ZH.md)的 608000 参数势，唯一训练为 B32、2048 updates，使用固定最终 `potential_final.pt`，SHA256 `495b3313e945f4b845307ae0d520c3c8d8fa70e60e3983002297cd3f7ccc632e`。每步在原生官方 clean prediction 上加入 $c=\nabla_z\Phi$，系数 **1**，全部 100 个正查询时刻启用，无新增窗口或手动系数。

[FID 前成本协议](RAEV2_OBSERVABLE_POTENTIAL_COST_PROTOCOL_20260906_ZH.md)与根 request 冻结 **候选100 vs 官方100 → 官方105 → 闭合准备成本后的唯一 full-cost 对照 → 独立噪声确认**。第一阶段即要求 $1-FID_\Phi/FID_0\ge0.05$；本次需候选 FID **≤35.684568538085**，实际未达。更后面的基线不能解除这个必要条件，故按原规则停止，并非看到质量后换对照或成本口径。

**完整配对协议。** Seed **202609095**，每分支 N=1000、1000 类各一张，global IDs 0–999、label=`global_id%1000`。完整 B8 按 batch 序号模 4 分片，在同型号 RTX 4090 上执行；四 shard 为 256/248/248/248 张。各 ID 恰好输出一张、全部保留，合并后按 global ID 排序，没有候选选择或拒绝采样。

两分支同一官方 DINOv3-L K7 Stage-2 EMA step **100080**，CFG **1.0**、IG **1.78**、区间 `[.1,1]`；shift **8**、Euler **100** 步、原 `t_eps=.05`。原生 BF16 heads 内计算 `B+1.78*(F−B)`，区间外用 Full，之后转 FP32；state、势、势输入梯度与 Euler 为 FP32，TF32 关闭。Base 来自同一次 Stage-2 forward，不是独立 NFE。唯一干预为是否加入上述势修正。

共同 decoder 与 normalization stats 身份一致；像素路径为 **原生 BF16 decoder → clamp(0,1) → BF16 ×255 → uint8 NHWC**，没有换成 FP32 乘法或另加四舍五入。两份 evaluator 输入均是 `arr_0: uint8[1000,256,256,3]`，附 `ids,labels: int64[1000]`。CPU 独立确认完整 IDs/类别、形状、dtype 和两份样本文件 SHA 与 merge/evaluator 记录一致。

**噪声一致性。** 一个 CUDA Generator(seed) 一次生成 `[1000,1024,16,16]` 全局噪声，再按 global ID 切片；无 rank/batch seed 偏移。两份完整 common request 除 `mode` 外完全相同。独立 CPU 比较 **125/125 对 batch 的 noise hash 与 ID 完全相同**，8 个 worker 保存的首噪声 tensor/RNG state 字节一致；这复核存储证据，没有重新生成完整 CUDA cohort。

| 身份 | SHA256 |
|---|---|
| 全局噪声 cohort | `7f8a1840d9ae8071db5a50ff74df1e063b747c6b8dbff37beebaf657e252e57a` |
| 全局 labels | `702746827e553786bb026ac120cb58745fef3d3f554c33891809001cc37639f0` |
| 首样本 noise tensor | `9e09ed6f0748825f300c7f7b21bef1779f2ba792219d06383fb561eba8cadca1` |
| 全局生成后的 RNG state | `badab2ec19ee9091deb2d81c92766fb1e72be1dd2bcf42b7ff2a9371f7ddeee0` |

**真实 FID 与 IS。** 同一 `nanogen-evals`，commit `19dfb4c2705333eb8b97e454fb354d47d1fe135b`，reference `imagenet_256_fid_stats`，evaluator seed **2020**、batch **64**，两组样本在同一次命令中评价。没有换 reference、混用旧 FID 或按 evaluator 选择结果。

| 分支 | FID ↓ | Inception Score ↑ |
|---|---:|---:|
| 官方 IG 1.78，100-step | 37.562703724300 | 60.024336242676 |
| 固定势修正，100-step | 37.531664406887 | 59.727458953857 |

绝对 FID 差为 **0.031039317413**，相对下降 **0.08263334%**；IS 相对变化 **-0.494595%**。这里是一个 paired 1K 的点估计，不能把极小的正 FID 差称为稳定收益或显著提升。它足以裁决本次预注册筛查未达到 5%，不能宣称这个方法在总体上严格等效于 baseline。

**实际采样成本。** 每分支 125 个 B8 batch，每图 100 次 Stage-2 查询，共 **12500 batch forward / 100000 sample forward**。候选额外 **12500 batch 势输入梯度 / 100000 sample 输入梯度**，无参数 backward；两分支各 125 个 decoder batch forward / 1000 sample forward。以下是四 worker 的实际 wall/CPU **求和**，不是并行 critical-path elapsed 或 CUDA kernel 时间。

| 计时范围 | 官方100 秒 | 势修正100 秒 |
|---|---:|---:|
| 轨迹（模型＋势输入梯度＋Euler） | 626.963410812 | 656.521812564 |
| 解码与 uint8 | 3.506546344 | 3.525303016 |
| 采样含输出 | 634.174259597 | 663.987626764 |
| 势 checkpoint 读取/构建 | 0.043644231 | 0.061064622 |
| 共同 backbone/decoder 加载 | 42.754161982 | 47.315624872 |
| 全局噪声生成/拷贝/审计 | 5.105630066 | 5.184430528 |
| worker 主程序 wall 合计 | 729.230981172 | 764.085072016 |
| worker 主程序 CPU 合计 | 742.729362563 | 775.697452325 |

轨迹 wall 比为 **1.047145337738**，增加 **4.714534%**；采样含输出增加 **4.701132%**。这与前置 B8 benchmark 的约 4.783% 开销相符，但两者是不同计时窗口，均不是 FID 提升。

为保持进程驻留模型一致，官方分支也加载了未使用的 final potential；表内如实保留这项实际成本。轨迹计时不包含它，不能把这次官方驻留安排说成官方部署必需。两分支峰值 allocated 均 **6302049792 bytes（5.869 GiB）**。Worker 主程序窗口排除 imports 与末次 summary 写入；不能据此声称所有历史准备费用已经闭合。

CPU merge wall/CPU 分别为官方 **13.018606726/15.503826232 秒**、候选 **13.065560777/15.508747620 秒**。Evaluator 实际处理 2000 张图，进程 wall **15.551619297 秒**、child CPU **41.058515000 秒**，包括 imports 和退出。

数据编码、一次固定训练、两次数值 pilot、完整 validation、benchmark、CPU prefix 重放在[求解器成本归档](RAEV2_OBSERVABLE_POTENTIAL_SOLVER_20260906_ZH.md)另列，**没有乘以四 shard，也没有藏进轨迹 ratio**。训练准备与研究审计的归属沿用成本协议。此前 bridge coupling MSE 下降 1.0968% 是不同观察量；现在不能用它替代 FID，或将二者百分比混写。

**停止与否定范围。** 本固定有限求解器终止；官方 **105-step 未运行**、**full-cost 对照未运行**、独立确认 seed **202609096 未运行**。已知准备计时窗口的 211-step 下界仍不是完整成本值，也没有用于质量比较。因此没有达到完整公平成本目标，不继续自动扫宽度、窗口、系数、optimizer 或随机 seed。

这次否定的是具体 608000 参数势、固定 2048 更新、固定 coefficient 1/全时刻部署，在当前 paired 1K 必要条件下的候选。它不否定所有 weighted-Poisson/continuity 机制，也没有证明更大的训练预算或别的有限空间必然成功。前次 validation 中剩余 witness 明显非零，完整 Poisson 问题本就未解；不能把理想总体定理用于保证本次有限解的 FID。

**主要源文件 SHA。** 完整 request/summary、模型/decoder/stats、8 worker manifest 与噪声审计的记录见[机器归档](data/raev2_guidance_restart_20260906/observable_potential_screen.json)。

| 产物 | SHA256 |
|---|---|
| FID 前 screen request | `ad6cec9e53b49bb72bf5abf842ab4d35085221bcc000da276d1c7324cd7971cd` |
| 停止裁决 summary | `88ee8db22835edd265f079ee50a38a2972d223c328872d20ad5ac31cbe674389` |
| 官方 merged samples.npz | `4824d1b51990ba531e9f9af0d36b2a95df04e60300eeac5c0a9a3a17086cb573` |
| 候选 merged samples.npz | `3c2c7cbe306952e9d5f3742b19f2a5c6b4c8e597c0e4b55d2fea1b3e0a951c6e` |
| 官方 merged summary | `dc752c8b741dd492e059704f8f4501989914a92f827f79ca49fae032f7e9a91e` |
| 候选 merged summary | `a584fa16ebd401a4c3070b9c920ef25ef762b01f52bfa229096386c19182e454` |
| evaluation request | `bb3fd1efd244a52491a096ce23812020445cdee016c44a85faf85af65c2ea3a7` |
| metrics.json | `e8cb250295ecf8526b8edda00d0ecb4e4acff407243316708281930a37a51c39` |
| evaluation process summary | `0eeb4b7dbc85eccf1ee3dcec81e680c957177485904dcee50ee8847895526a94` |
| 采样器 | `29fff5850f464dfbbfd2ca2f034fb34979fc49495936f78bb73cc39ad5e8bbaa` |
| 官方 Stage-2 checkpoint | `723c56d7fa77ace9613909f7e38cb2386b898608218dc9b52649bb373d513c9a` |
| 官方 decoder | `779871dc7e81d4c31cc5dc80824760c79b40e4a8b90e0d743d1e79c10d515d87` |
| normalization stats | `40e57d9d38a267dc258043c081094d276382c29ebccc1bd8c38f4dd11e81ba77` |

实际产物：[根裁决](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/observable_potential_screen_v1/summary.json)、[官方 merged summary](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/observable_potential_screen_v1/official100/merged/summary.json)、[候选 merged summary](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/observable_potential_screen_v1/potential100/merged/summary.json)、[metrics](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/observable_potential_screen_v1/evaluation/metrics.json)、[evaluation request](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/observable_potential_screen_v1/evaluation/request.json)、[评价成本](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/observable_potential_screen_v1/evaluation/summary.json)。
