# 最新末段 schedule 与训练实现审计

本次只读日志、源码及 CPU 检查点。未占用 GPU，未改训练。新证据保存在 [latest_tail_snapshot.json](latest_tail_snapshot.json)，上一轮 `schedule_snapshot.json` 保持不变。

**结论：JiT 与旧 SiT 的末格都经历了先升后降，尾峰逐渐前移；末格下降伴随非零梯度和实际 Adam 更新，不能解释成末格被冻结、只更新 EMA 或 optimizer epsilon 压死梯度。** 但这只能确立训练现象，不能单凭日志判定它来自连续动力学、离散求解器还是当前判别器的偏好。

## 1. 固定捕获范围与最新末 10 格

捕获时间为 2026-09-23 05:16 UTC 左右（北京时间 13:16）。JiT 与 joint 正在运行，下述数值只属于本次固定快照。`latest.json` 是最近检查点，可能落后于 `train.jsonl`；这里采用后者最后完整、有限数值的记录。索引从 0 开始，时间为**区间中点标签**，不是两级 Heun 实际调用的时间。

| 运行 | step / 参数更新数 | 尾峰 index / 中点 / a | 最后格 a | 最后格 EMA |
|---|---:|---|---:|---:|
| JiT `training_30k_gpu0_gpu2_g32_m8` | 10354 / 10226 | 46 / .93 / 1.844655 | .015998 | .029955 |
| 旧 SiT native `training_30k_monitored` | 30128 / 30000 | 61 / .9609375 / 4.045915 | .431281 | 未在该日志记录 |
| SiT joint `single_after_deadline_20260923` | 9198 / 9070 | 62 / .9765625 / .832859 | .791977 | .789878 |

joint 的全局最大值其实是 index 23 的 1.030784；表中列的是最后 10 格内的峰。它仍然全正，不能与旧 SiT native 的负谷混称。

| 末 10 格内序号 | JiT：index 40–49 | 旧 SiT：index 54–63 | SiT joint：index 54–63 |
|---:|---:|---:|---:|
| 0 | -1.299575 | -1.134494 | .466416 |
| 1 | -1.211409 | -.986939 | .455973 |
| 2 | -.836072 | -.651378 | .457244 |
| 3 | .202782 | -.091510 | .477895 |
| 4 | 1.374506 | .716698 | .526716 |
| 5 | 1.567098 | 1.761246 | .609128 |
| 6 | 1.844655 | 2.976373 | .719856 |
| 7 | 1.696536 | 4.045915 | .817310 |
| 8 | 1.258303 | 3.399452 | .832859 |
| 9 | .015998 | .431281 | .791977 |

JiT 已不是上一轮 3927 次更新时“峰在 .97、末格约 .938”的形状。当前最低点 index 38 / t=.77 / a=-1.434048，也已越过 a=-1。

## 2. 历史是“尾峰前移”，而不只是静态末格较小

| 参数更新数 | JiT 尾峰 index / a | JiT last | 旧 SiT 尾峰 index / a | 旧 SiT last |
|---:|---|---:|---|---:|
| 2000 | 49 / .9481 | .9481 | 63 / .5859 | .5859 |
| 3927 | 48 / 1.1226 | .9377 | 63 / 1.1009 | 1.1009 |
| 5000 | 47 / 1.2299 | .7981 | 63 / 1.3299 | 1.3299 |
| 8000 | 46 / 1.5900 | .3201 | 63 / 1.8520 | 1.8520 |
| 10000 | 46 / 1.8439 | .0557 | 63 / 2.0584 | 2.0584 |
| 15000 | — | — | 63 / 2.3778 | 2.3778 |
| 20000 | — | — | 62 / 2.7625 | 2.1001 |
| 25000 | — | — | 61 / 3.4049 | 1.5019 |
| 30000 | — | — | 61 / 4.0459 | .4313 |

JiT 最后格历史最高为 2680 次更新时 1.024314；之后首次低于 .5 在 6921 次、低于 .1 在 9734 次。旧 SiT 最后格历史最高为 15440 次时 2.407777，之后降至当前 .431281。以上是同一条训练链的描述，没有把独立 smoke/preflight 混入。

SiT joint 目前只表现为**同一条时间曲线上最后格低于倒数第二格**；它的最后格在最近 500 次训练更新中反而净上升 .018108。这与 JiT/旧 native 的“末格随训练持续下降”应严格区分。

## 3. 梯度和实际参数更新

`head_gradient` 是跨 rank 平均后、裁剪前的参数梯度；`head_update` 是实际 optimizer step 的参数差。下表采用最后 500 条诊断，JiT 覆盖 step 2368–10352，每 16 步采样；旧 SiT 覆盖 step 17650–30125，每 25 步采样。不同覆盖窗口不能当同等样本量的独立实验比较。

| 运行 / bin | 平均 gradient | gradient 正号率 | 平均 update | update 正号率 | gradient 零率 |
|---|---:|---:|---:|---:|---:|
| JiT 峰 46 | -5.987e-4 | 43.2% | +1.689e-4 | 76.0% | 0 |
| JiT last 49 | +3.257e-4 | 54.0% | -1.190e-4 | 31.0% | 0 |
| 旧 SiT 峰 61 | -7.136e-4 | 44.2% | +1.428e-4 | 72.4% | 0 |
| 旧 SiT last 63 | +6.614e-4 | 59.4% | -1.591e-4 | 25.8% | 0 |

再用连续训练日志中的相邻系数差，检查**最后 500 次实际更新**，避免诊断抽样偏差：

| 运行 / bin | 平均实际 update | 下降比例 |
|---|---:|---:|
| JiT 峰 46 | +3.926e-5 | 43.0% |
| JiT last 49 | -1.773e-4 | 74.6% |
| 旧 SiT 峰 61 | +1.007e-4 | 33.0% |
| 旧 SiT last 63 | -2.766e-4 | 89.0% |

最后 500 training steps 内抽到的诊断中，JiT last 平均梯度 +.001967，旧 SiT last +.000395；两者都没有零梯度记录。不同步的 Adam 一阶矩与当前随机梯度可异号，因此不要要求每步 update 必须与该步 raw gradient 相反。

最后 100 条诊断中，两模型 `head_zero_gradient_fraction`、`head_near_zero_gradient_fraction` 和 `g_loss_saturated_fraction` 均为 0。JiT/旧 SiT 的平均 loss 对 logit 的敏感度分别约 .554/.720。它排除了这些已记录诊断意义下的整体 loss 饱和，不能排除部分像素被 clamp、表征丢失信息或 VJP 实现存在数值误差。

## 4. 优化器、AMP、loss scaling 与多卡

| 项目 | JiT pure schedule | 旧 SiT native | 当前 SiT joint |
|---|---|---|---|
| schedule Adam | lr=.001，β=(.9,.99) | 同左 | lr=.0002，β=(.9,.99) |
| eps / weight decay | 1e-8 / 0 | 同左 | 同左 |
| D Adam | lr=.0001，β=(0,.99) | 同左 | 同左 |
| 梯度裁剪 | schedule 全向量 norm≤1；D≤10 | 同左 | weak≤1、schedule≤1 分组；D≤10 |
| batch | global32，2 ranks，各16，micro8，每 rank 累积2次 | global24，3 ranks，各8，无外层 microbatch 累积 | global32，1 rank，micro8，累积4次 |
| 精度 | backbone BF16 autocast；冻结 Linear/Conv 预存 BF16；系数与 Adam moments FP32 | backbone FP32/TF32，非 BF16 | backbone FP32/TF32 |
| 动态 loss scaling | 没有 GradScaler | 没有 | 没有 |
| 训练 EMA | .99，只作旁路平滑，不回灌实际 rollout | 同左 | 同左 |

训练源码依据：[JiT 主程序](../../../classifier_guidance/train_jit_schedule.py)、[旧 SiT 主程序](../../../classifier_guidance/train_schedule.py)、[GAN step](../../../classifier_guidance/training.py)、[microbatch step](../../../classifier_guidance/training_accumulation.py)、[joint step](../../../classifier_guidance/sit_joint.py)。

这里是手工 collectives，未使用 DDP wrapper 的隐式梯度同步。microbatch loss 乘 `microbatch/local_count`，累计后再跨 rank 平均，随后裁剪、只执行一次 Adam step。JiT 将 detached feature 在各 rank 聚合，D 每次只对完整 global feature batch 执行一次训练 forward，避免 spectral normalization 的 power iteration 随 microbatch 数量增加。

CPU 读取固定检查点验证：JiT step10200 的 Adam 参数年龄为 10072，旧 SiT step30128 的为 30000，joint step9000 的为 8872；都等于 step−128 warmup。D 的 Adam 年龄等于 global step。支持持续恢复优化器，未见年龄重置。末 10 格的 `eps/sqrt(v_hat)`：JiT 约 1.9e-6–3.8e-6，旧 SiT 约 4.2e-7–3.0e-6；当前尾梯度不是被 Adam epsilon 主导。

全向量裁剪会把早段大梯度与末段小梯度耦合，Adam 又改变不同 bin 的有效步长，因此 scale 的幅值不能直接解释成物理响应强弱。最近 1000 条训练记录中 schedule clip 比例：JiT 7.4%，旧 SiT 1.0%，joint .8%；没有证据支持“每步都被严重裁剪”这一解释。

## 5. 真正存在的简化与仍未验证的风险

- **固定离散问题。** 当前针对固定网格与 Heun/Euler 规则反传；一个区间的系数供两个 Heun stage 共用。没有为同一时间点的不同 stage 独立设置参数。它优化的是该离散 sampler，不是步长趋零的连续控制问题。
- **省内存不等于截断梯度。** [Sampler](../../../classifier_guidance/sampler.py) 保存各步状态、反向逐 field 重算 VJP，并传播输入 cotangent；没有只取末几步的截断。`once_differentiable` 禁止二阶反传，但不删减一阶链式梯度。本轮未进行真实 CUDA 最新尾 bin 的独立差分核验。
- **JiT 精度与末端规则。** BF16 网络计算、clean prediction 的差、分母 floor=.05、最后 Euler 都真实存在。系数特意扩展为 FP32 张量参与 clean mixture，避免标量乘 BF16 时系数作用被一并降精度。网络输出/差方向的舍入误差仍不能由此消除。
- **共同的末端 clamp。** JiT 的 RGB 是 `((x+1)/2).clamp(0,1)`；SiT 的 VAE 输出也进行范围 clamp。超界像素对该连续 surrogate 的局部导数为零。训练保留连续像素，导出图像的 uint8 截断没有参与反传。这些会改变终点反馈，尤其需要与末段效应一起考虑。[转换实现](../../../experiments/adversarial_weak_training_20260915/rendering.py)
- **受限终点 critic。** D 是冻结 Inception2048 特征上的 class-conditional 小网络，R1 也作用于特征空间。完整反传的是这个 feature GAN surrogate，不是完整图像分布距离，更不是 FID 的梯度。
- **joint 对照不等价。** joint 同时改变 weak head、有 gap-energy anchor、lr_a 只有 pure schedule 的 1/5，且全段从 .75 初始化。native 从前半 .6、后半 0 初始化；JiT 从全段 .5 初始化。不同 curve 不能仅归因于网络类型。

## 6. Resume、事件与来源有效性

JiT 在 step129 从 global8 改为 global32，step195 从1卡改2卡，step256进入当前长跑；记录均恢复权重与两个优化器，batch/world-size 变化明确不保证 bitwise trajectory。末格持续下降发生在数千次更新之后，不与这些早期切换同步。

旧 SiT 在 step384、640、745续训，最后一段保持3卡/global24；读取源码与恢复日志支持继承优化器。joint 在 step1455、1457、1458、7451经历1/2卡切换，恢复 optimizer、data/probe RNG；当前单卡段从7452开始。

所选训练链分别有 10354、30128、9198 条有效训练记录，step 无重复、无断档，系数长度固定，未发现无效 JSON/非有限系数。各链 worker.log 中未匹配到 overflow/nonfinite/NaN/Traceback/Exception/optimizer reset 等事件；代码遇到非有限梯度会报错，不存在 GradScaler 静默 skip 分支。结论限于所查日志，不能证明一切底层数值问题不存在。

当前 JiT 长跑和当前 joint 的相关源码与 request hash 一致。旧 SiT 的 `sampler.py` 后来增加了逐区间 Heun/Euler 选择和只 capture 使用状态的工程泛化；已比较运行 source_snapshot，旧模型使用的全 Heun 递推未改变。更早 pilot 的源码版本差异保存在 JSON 中，未把当前源码无条件套给所有历史阶段。

优先需要解释的观测是：**不同模型的尾峰都向前移动，而最后格在可用梯度和持续 Adam 更新下被压低。** 下一步应对这一特定现象进行真实模型的局部响应与求解器核验；当前证据不足以把它命名为普适振荡机制或证明某种 boundary 理论。
