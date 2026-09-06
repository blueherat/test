# RAEv2：固定 native BF16 heads 的 guidance 混合数值审计（2026-09-06）

已有缓存足够回答一个窄问题：**给定同一状态上原生 BF16 Full/Base 的准确保存值，三次 BF16 算术相对于 FP32/FP64 混合产生多大误差？** 全部既有缓存的结果为：混合误差范数约为 `F−B` 范数的 **1.06%–1.07%**，约为混合输出范数的 **0.184%–0.185%**；约 90% 的坐标减法精确，主要误差项来自最终相加舍入。未发现足以支持质量改进的强输出方向相关。这是基准数值诊断，**不是新 guidance、采样结果或 FID 证据**。

数据和可复核产物位于 [numerical_guidance_mix_v1](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/numerical_guidance_mix_v1)。先写入 [request.json](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/numerical_guidance_mix_v1/request.json)，SHA256 为 `73d79ce5edf7184a07f84e05edc814a634add94df89cc59a5e682575a249af54`，然后才读 tensor 数值计算。固定系数 1.78、原有全部十个时点、两个 domain、八个 ID；没有系数、噪声或时间窗口扫描。

最小可用数据来自 [normal_noise_audit_seed202609071/states](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/normal_noise_audit_seed202609071/states) 的十个 `step_*.pt`。每个文件包含 teacher/rollout 的 state、Full、Base，形状均为 `[8,1024,16,16]`，存储 dtype 为 FP32。原 runner 在 CUDA BF16 autocast 中执行模型，直接对两份 head 调用 `.float()` 保存。对应 [depth_readout/native_parity.csv](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/depth_readout_audit_seed202609071/native_parity.csv) 对全部二十批记录了 native forward 与重建 readout、其 FP32 提升值与保存值的逐位一致。历史 DDT 源码 SHA 也匹配：最后一次 Linear 位于 BF16 autocast 内，此后仅 unpatchify。此次再确认所有 head 有限且转回 BF16 再升 FP32 无任何变化。**dtype 来源依靠源代码、历史 parity 与数值格点三项证据，不单凭数值落在 BF16 格点推断。**

这些 state 的适用范围必须保留：历史 rollout 使用 FP32 `F+0.78*(F−B)`，backbone 的 TF32 开启；它们并非当前 TF32 关闭、native BF16 `B+1.78*(F−B)` 的 rollout。这里只保存十个时点，并非完整一百步轨迹。快照只保留源审计第一批 IDs/classes `0..7`，source rows 为 `[985579,801642,769290,1066283,356051,843767,756795,67027]`。两个 domain、不同时间依赖于相同八个样本，且 `t=1` 的两域 state/F/B 逐位相同；**160 行不等于 160 个独立样本，更不等于独立实验**。未计算 SEM 或跨类泛化区间。

历史 `raev2_ig_scale_response` 的 latents 是不同 scale 下的最终端点，不能拼作同状态 heads；既有 predicted-clean audit 保存 Inception 特征和标量范数，不能据此恢复逐坐标舍入。此次没有追加这些数据，也未触碰当前 5K bank。

设 `d=F−B`，这里 `F,B` 均为缓存 BF16 值的精确提升，**FP64 reference 不是 FP64 模型 forward**。活动区间 `[0.1,1]` 内重建：

\[
\widehat d=\operatorname{RN}_{16}(F-B),\quad
\widehat m=\operatorname{RN}_{16}(1.78\widehat d),\quad
\widehat G=\operatorname{RN}_{16}(B+\widehat m),\qquad
G_{64}=B+1.78(F-B).
\]

这里每个 `RN16` 是 eager BF16 逐操作舍入，乘法的 Python scalar 按 PyTorch 的 FP32 opmath 处理，不能预先把 1.78 舍入成 BF16 1.78125。CPU PyTorch 的每个结果都与单独的 FP32 位运算 round-to-nearest-even 参照比较，总计 **125,829,120 个值完全一致**。这证明 CPU 重建的内部数值一致性；缓存**没有 CUDA 混合后的 G**，本任务也**没有验证 CUDA kernel 的结果**。

定义 `e=G_hat−G64`。逐图的精确误差分解为

\[
e=\underbrace{1.78(\widehat d-d)}_{e_{sub}}
+\underbrace{(\widehat m-1.78\widehat d)}_{e_{mul}}
+\underbrace{[\widehat G-(B+\widehat m)]}_{e_{add}}.
\]

全部逐元素分解误差最大 `4.44e−16`。以下均为九个既定活动时点 × 八个样本的**逐图统计量算术平均**，teacher 与 rollout 分列；RMS 以 `D=262144` 归一化。

| 固定数值量 | teacher | 历史 rollout |
|---|---:|---:|
| `||e|| / ||F−B||` | 1.05771% | 1.07398% |
| `||e|| / ||G64||` | 0.185137% | 0.184106% |
| `RMS(e)` | 0.00174600 | 0.00182697 |
| FP32 混合相对 FP64 的 RMS 误差 | 2.62341e−8 | 2.74848e−8 |
| `RMS(e_sub)` | 0.000312599 | 0.000313212 |
| `RMS(e_mul)` | 0.000499351 | 0.000509672 |
| `RMS(e_add)` | 0.00164636 | 0.00173017 |
| 将 `G64` 仅最终舍入到 BF16 的 RMS 误差 | 0.00156629 | 0.00164720 |
| 逐操作 BF16 与仅最终舍入的坐标不同比例 | 20.2345% | 19.9541% |
| Sterbenz 充分条件覆盖 | 82.3811% | 82.8079% |
| 实际 BF16 减法完全精确比例 | 89.9532% | 90.2096% |
| Sterbenz 条件内违例 / 输入 subnormal 数量 | 0 / 0 | 0 / 0 |
| `cos(e,G64)` | −0.00210277 | −0.00279930 |
| `cos(e,B)` | −0.00617616 | −0.00646883 |
| `cos(e,F−B)` | +0.0109147 | +0.0100696 |

RMS 各项有交叉项，不能把表中三项的平方直接解释成独立方差贡献。[per_image.csv](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/numerical_guidance_mix_v1/per_image.csv) 保留全部 160 行、交叉项、无中心 cosine、按坐标中心化 Pearson 和投影斜率。Pearson 与 cosine 很接近。`e` 在 gap 上的平均投影斜率仅 `0.00010984 / 0.00010410`；八个 ID 重复出现，不能据此声称一个已验证的普遍偏差方向。

各时点的 `||e|| / ||F−B||` 如下，完整 twenty-batch 统计在 [per_snapshot.json](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/numerical_guidance_mix_v1/per_snapshot.json)。最后一个时点早已在官方区间外，所有分支均取 `F`，误差为零；没有为结论挑选窗口。

| t | teacher | 历史 rollout |
|---|---:|---:|
| 1.000000 | 1.06912% | 1.06912% |
| 0.900212 | 0.836606% | 0.873996% |
| 0.797583 | 0.884546% | 0.918752% |
| 0.704981 | 0.924678% | 0.957286% |
| 0.603774 | 0.970131% | 0.999736% |
| 0.497175 | 1.03223% | 1.05954% |
| 0.410256 | 1.10552% | 1.12989% |
| 0.296296 | 1.26790% | 1.27799% |
| 0.198347 | 1.42864% | 1.37950% |
| 0.074766 | 0（G=F） | 0（G=F） |

Sterbenz 的充分条件为同号非零浮点数且 `1/2 ≤ |F|/|B| ≤ 2`；这一范围内相减的结果可精确表示。本任务另计两者同时为零，并直接检查 BF16 减法与 FP64 差值相等，未将充分条件外的坐标都判为有误差。**结论针对两份既定 BF16 值的相减算术。若 backbone 先产生的 head 误差在 `F−B` 中被放大，即使减法完全精确，那个问题仍然可能存在。** 缓存缺少同状态高精度 forward 的 `F*,B*`，因此此处无法回答 head 前向误差问题。

这组结果不支持用“混合阶段相减灾难性丢精度”作为新 guidance 的主要机制。升精度混合当然可以减少这部分表示误差，但 **误差更小并不保证 FID 更好**；历史 TF32 状态、当前轨迹、decoder/采样误差和固定 head 误差之间也没有建立质量因果联系。此次没有修改 sampler，没有把精度变化包装成新 guidance，也未启动重放或训练。

复核和成本：[audit_cpu.py](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/numerical_guidance_mix_v1/audit_cpu.py) 固定在 request 的 SHA 列表中，十九个输入均重新核对；[summary.json](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/numerical_guidance_mix_v1/summary.json) 包含全部汇总、版本、产物 SHA 与耗时。另一个只读 [verify_cpu.py](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/numerical_guidance_mix_v1/verify_cpu.py) 检查身份、23 项 SHA、全部行和误差能量恒等式，最大能量误差 `4.24e−21`；这不是独立研究者复核。冻结/计算/复核 wall time 分别为 **0.728 / 4.533 / 0.751 秒**，CPU time 分别 **0.843 / 22.654 / 0.751 秒**。全部设置 `CUDA_VISIBLE_DEVICES=''`，模型加载与 forward 次数为零，GPU 时间为零。
