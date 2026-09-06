# 配对桥流固定机制实验结果

日期：2026-09-06。本次固定结构完成了 CPU 检查、GPU 数值 pilot、一次2048步联合训练、10000条 teacher 验证，以及96条实际完整轨迹。**有有限边缘统计改善的证据，尚无 FID 结果。决定保持公式和权重不变，准备一次配对1K质量筛查；不是宣布5%目标达成。**

依据是[有限步配对桥理论](RAEV2_ACTUAL_ROLLOUT_FINITE_TRANSPORT_20260906_ZH.md)，具体数据、两网、时间覆盖、midpoint和成本口径均在[冻结协议](RAEV2_PAIRED_BRIDGE_PILOT_PROTOCOL_20260906_ZH.md)。该结构不依赖固定点、手工外推时间表或生成后筛图。

## 结果必须分开解读

候选与 mean-only 对照各3652224参数，同初始化、相同teacher批次，各训练2048 updates×B32；真实主模型始终以B8调用。唯一最终checkpoint SHA为 `5013cbf075ddffa5c2ae021fc916be0615ca41d9817d3af91e6b3e46c86e1983`。没有按验证选权重、补训或重新采样训练。

在全部1000张当前fit外图、全部100时刻的10000条teacher记录中，归一化速度目标的平均能量为0.3670627152。相对零预测的风险改善定义为 `2E[R/β·f]−E||f||²`：

| 固定查询 | 风险改善 | 相对目标能量 |
|---|---:|---:|
| candidate τ0 | −0.000302758465 | −0.0824814% |
| candidate τ0.5 | −0.000302359543 | −0.0823727% |
| candidate τ1 | −0.000307555334 | −0.0837882% |
| mean-only τ0 | −0.000301480040 | −0.0821331% |

**有限CNN的速度回归没有胜过零预测。** 这不能被数值pilot上固定批次的损失下降掩盖，也不支持宣称已学到精确条件速度。这里的监督只在真正teacher插值上；没有将部署midpoint的状态错配到原配对标签。

另一方面，执行固定两次校正后，预先指定的φ（全1024通道空间均值＋空间二阶矩）的经验均值差平方发生如下变化。数值均为全部100时刻的平均，每时刻100张：

| 有限map | 全2048维 | 均值1024维 | 二阶矩1024维 |
|---|---:|---:|---:|
| official | 8.278915813e−6 | 1.802272164e−6 | 1.475555946e−5 |
| candidate | 7.674876449e−6 | 1.450301162e−6 | 1.389945174e−5 |
| mean-only | 8.305535992e−6 | 1.414301818e−6 | 1.519677017e−5 |

候选全φ差约降低7.30%，mean-only总体略增。候选与对照都在99/100时刻比official小，但末步均更差；候选末步恶化较少。**候选相对mean-only的净优势有98.51%来自末步**，不能把总体数字解释为每个时刻都额外恢复了covariance。全部时刻继续保留，下一次采样不据此改成末步窗口或调整强度。

这些是固定异质类别/图像cohort的描述量，不是FID、完整分布距离或KL。非对角交叉量也已报告，但不称总体均值差平方的无偏估计。每图及其noise复用10个时刻，不能把10000行当独立样本构造CI。完整逐时、逐类描述见归档。

三臂各32条实际完整100步轨迹全部完成，末端FP32、有限、原始张量SHA与存盘一致，重新计算moment最大误差不超过1.78e−15。相对official，候选和mean-only的配对末端latent变化RMS分别为0.1209218与0.1171864。未发现这组32类轨迹的数值失稳；它们不能代表全类别生成质量。没有用这些实际状态与某张teacher X计算denoising risk，没有解码或FID。

## 为什么继续到质量筛查

这次同时留下了一个负结果和一个正结果：速度回归风险未改善，预设的有限边缘统计有改善。精确理论并不把这两个量等同，原协议也没有将teacher配对MSE作为分布校正门槛。候选相对等结构mean-only的末步二阶统计差异，与需要辅助时间响应的机制相容，但远未构成完整证明。

因此，下一步用同一固定checkpoint、完整100步公式、独立新采样noise检查实际图像分布。mean-only保留为机制对照，官方100及按测得推理成本确定的官方步数作为质量参照。一次1K仍只作筛查；准备/训练成本和最终成本达标条件单独披露，不将7.30%的局部统计变化当作5%FID收益。

## 核验与投入

[只读CPU汇总](../experiments/summarize_raev2_paired_bridge_pilot.py)正式执行一次，74项源/输出身份、完整request/checkpoint链、类别/时刻覆盖、调用计数和96条末端均通过。CPU汇总wall13.897838 s、CPU14.156530 s，无模型调用。17项模块CPU检查通过；GPU数值pilot的原生IG公式与零校正parity通过，其权重被丢弃。

| 阶段 | 主模型sample-forward | candidate / control sample-forward | runner墙钟 |
|---|---:|---:|---:|
| 数值pilot | 32 | 320 / 320 | 24.158207 s |
| 固定联合训练 | 65536 | 65536 / 65536 | 513.030591 s |
| teacher验证 | 10400（含400条补齐调用） | 50000 / 30000 | 115.985816 s |
| 96条实际轨迹 | 9600 | 6400 / 6400 | 88.135410 s |

四阶段runner墙钟共741.310023 s；共同训练循环490.104689 s。主模型总计10696 calls/85568 sample-forwards；candidate4858/122256，control4058/102256；各2056次参数backward包含数值pilot。不把共享主模型和数据成本平分，mean-only与验证的研究开销全保留。主模型GPU始终物理卡3（UUID `GPU-7d3e4e7d-abfa-e06e-c264-796052797949`）。

原训练driver/session跨turn后消失；检查确认进程不存在、最终训练summary和权重完整后，只续跑此前未启动的validation和rollout，两者退出码均为0，外层墙钟分别116.958644 s与89.059500 s。**没有重启训练。** 原训练外层退出码和精确单调墙钟未捕获；UTC边界只允许给出[513.080760,580.071819] s范围，不伪造返回码或拼成精确总计时。

既有6000图bank的编码记录为750个B8前向、113.860739 s，更早数据来源选择/研究耗时未完整统一计量。两份核心model_utils和环境身份另补充归档，明确捕获于初次pilot启动之后、正式训练之前，后续阶段启动前验证未变。这些边界限制总成本达标结论，不影响本次已完成机制结果的身份。

## 归档

- [完整实验目录](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/paired_bridge_v1)。
- [CPU核验与描述汇总](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/paired_bridge_v1/analysis_v1/summary.json)，SHA `5dc55e8543d012e47ab868e45271cbce9fe753c9e94f1cb7efb135d9a1c9566f`。
- [teacher逐行记录](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/paired_bridge_v1/validate/teacher_regression.csv)、[全部逐时有限统计](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/paired_bridge_v1/validate/finite_map_moments.csv)。
- [续跑进程及成本记录](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/paired_bridge_v1/pipeline_resume_status.json)。
