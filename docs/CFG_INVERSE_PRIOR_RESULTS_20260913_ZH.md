# 真实图反演先验：实际结果与停止记录

2026-09-13。**本轮在完整数据集的数值反演阶段结束，未进入先验拟合或生成质量评估。不能写成“逆先验没有 NLL/FID 收益”，也不能据此断言 CFG 映射不可逆。** 按有界协议，普通 Picard 失败后只增加一次 Anderson 修复；扩大到真实数据后再次失败，停止其余 worker，转向用户已授权的 Z-Sampling。

[原始协议](CFG_INVERSE_PRIOR_SCREEN_20260913_ZH.md)；[明确停止原因及发出的 SIGINT](</home/zhoushunyu/data/eqvae/experiments/cfg_inverse_prior_20260913_anderson/stop_request.json>)。

## 数值检查实际说明了什么

固定 S800 EMA、CFG extra=1.25、cutoff=.75、64 步 Heun，FP32、关闭 TF32。逆求解针对实际离散生成映射，同一物理区间的两次查询共用正向左端点的 guidance 开关。预检的正向输出及保存快照均与冻结 baseline **逐位一致**。

| 阶段 | 结果 | 限定 |
|---|---|---|
| 普通 Picard，逐步 RMS≤1e−5、最多16次更新 | 已知噪声8例通过；真实图 bank row60 在 k=20 失败，残差由5.1281e−5增至1.2765e−4 | 表明该 Picard 迭代未收敛，不证明正向映射没有逆 |
| Anderson，记忆5、逐源FP64混合、同一上限与容差 | 同8类已知噪声和真实图均通过 | 只覆盖预先固定的8张真实图 |
| Anderson，一次收紧至1e−7、仍最多16次 | 同8+8例全部通过 | 支持这些样本的数值精度；不保证3000图覆盖或生成收益 |
| Anderson 完整 bank，生产容差仍为1e−5 | 保存1360个通过验收的源；新真实图 row142 失败后终止 | 未完成预定3000源，不能筛除困难图后继续拟合 |

收紧检查保留了全部反演噪声、正向端点和65个状态。以下是**最坏单图 latent RMS**，不是像素或感知指标：

| 8例检查 | 容差1e−5 | 容差1e−7 |
|---|---:|---:|
| 已知噪声的前像恢复误差 | 4.1318e−4 | 4.5233e−6 |
| 已知生成端点的完整往返误差 | 1.7443e−3 | 1.4046e−5 |
| 真实图的完整往返误差 | 2.6214e−3 | 5.1999e−5 |

真实图收紧前后反演噪声的整体 RMS 差为1.5207e−4，约为其单位尺度的0.0152%；完整往返整体 RMS 从1.3041e−3降至2.8669e−5。它是容差敏感性证据，并非真实图精确前像误差的严格上界。[原Picard报告](</home/zhoushunyu/data/eqvae/experiments/cfg_inverse_prior_20260913/preflight.json>)、[Anderson报告](</home/zhoushunyu/data/eqvae/experiments/cfg_inverse_prior_20260913_anderson/preflight.json>)、[收紧结果及数组索引](</home/zhoushunyu/data/eqvae/experiments/cfg_inverse_prior_20260913_anderson/refine_tol1e_7/summary.json>)。

## 扩大数据后的停止触发

失败源为 **bank row142 / source ID83885 / cache row6331 / class4 / holdout**，位于 shard0 的 batch128、slot14。在正向区间 k=36，即 t=0.5625，Anderson 残差从2.3722e−3一度降至4.6448e−4，但16次后仍为 **5.811921e−4**，约为阈值的58.1倍；同批其余15源最终残差约为1.7e−8至5.1e−8。

该批整体拒收。不能把当前 Anderson 配置未收敛等同于无逆，也不能将首批8例通过外推成整个真实分布可稳定反演。没有追加迭代、换求解器或改变容差来继续挽救本轮。[失败元数据](</home/zhoushunyu/data/eqvae/experiments/cfg_inverse_prior_20260913_anderson/shards/0/failure_000128/failure.json>)；[完整17次残差及失败状态](</home/zhoushunyu/data/eqvae/experiments/cfg_inverse_prior_20260913_anderson/shards/0/failure_000128/failure.npz>)。

## 完成量与实际计费

独立只读审计重新验证了全部85个完成批次的文件哈希、源ID、逐步残差和调用计数。**有效保存1360/3000源，含918 fit、442 holdout，只覆盖类别0–59的部分源。** 这是按分片顺序运行并提前停止的非完整集合，不估计总体失败率，也不作为选择性删样后的拟合 bank。

成本单位为一次单分支模型查询×图像数；前向 CFG64 为224/图。中断进程的计数器在进入模型查询时递增，故未完成部分包含最后被打断的调用，不能理解为这些调用全部产生了验收结果。

| 分片 | 完整批次数 / 源数 | 完成批次查询量 | 失败或中断批次查询量 | 进程记录合计 |
|---|---:|---:|---:|---:|
| 0 | 2 / 32 | 22,016 | 4,448 | 26,464 |
| 1 | 28 / 448 | 304,544 | 8,640 | 313,184 |
| 2 | 28 / 448 | 311,488 | 3,520 | 315,008 |
| 3 | 27 / 432 | 304,928 | 6,256 | 311,184 |
| **生产合计** | **85 / 1360** | **942,976** | **22,864** | **965,840** |

分片0失败批次的独立计数为(已完成局部步骤206+失败步骤72)×16=4,448。分片1–3分别在 batch1808、1824、1776收到 KeyboardInterrupt；这三项是主动停止，不是另外三项数值失败。[分片1](</home/zhoushunyu/data/eqvae/experiments/cfg_inverse_prior_20260913_anderson/shards/1/failure_001808/failure.json>)、[分片2](</home/zhoushunyu/data/eqvae/experiments/cfg_inverse_prior_20260913_anderson/shards/2/failure_001824/failure.json>)、[分片3](</home/zhoushunyu/data/eqvae/experiments/cfg_inverse_prior_20260913_anderson/shards/3/failure_001776/failure.json>)。

另计普通 Picard 预检13,600、Anderson预检16,928、收紧检查20,144；**本轮全部已记录查询成本为1,016,512**，包括失败和中断成本。

最终 `bank.npz` 未生成；两个实际运行目录均无真实 bank 的 `fit/` 或 `generation/` 产物。因此**没有先验选择结果，没有真实 heldout NLL 结果，没有本候选的 FID/sFID 结果**。本次终止的是当前预算和求解器下的数值可行性流程，尚未检验逆先验能否改善生成质量。原始脚本和失败记录均保留，后续工作已转向 Z-Sampling。
