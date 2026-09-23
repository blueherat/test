# 当前完整反传的计算与显存审计

2026-09-23。只读审计当前 SiT joint 与 JiT signed schedule 实现；没有操作 GPU、修改训练代码或干预运行。运行快照取自北京时间 2026-09-23 00:11 已有 `progress.json`，不是在本文发布时重新测速。

## 1. 当前如何控制成本

当前已使用**完整离散梯度、逐场重算和 CUDA graphs**。采样前向不保留各时间点神经网络的内部激活，只保存轨迹状态与 Heun predictor；反向逆序重新执行每个场的 forward，随即计算状态与可训练参数的 VJP。它保留完整离散链式法则，不是梯度截断，也不是连续 ODE 近似 adjoint。

源码定位：

- [sampler.py:105](../../../classifier_guidance/sampler.py#L105)：前向保存 states/predictors；[sampler.py:125](../../../classifier_guidance/sampler.py#L125) 保存反向所需张量。
- [sampler.py:71](../../../classifier_guidance/sampler.py#L71)：反向 replay 包含一次场 forward 和 `autograd.grad(output, (x, *parameters), cotangent)`。
- [sampler.py:147](../../../classifier_guidance/sampler.py#L147)：每个 Heun 区间先计算第二阶段，再计算第一阶段 VJP，完整包含 predictor 的依赖。
- [sampler.py:53](../../../classifier_guidance/sampler.py#L53)：捕获独立 CUDA graphs 并共享适用的图内存池；没有减少数学上的求值次数。

`requires_grad_(False)` 只消除了冻结模型的**参数梯度**，没有消除穿过它们的**输入梯度**：

- SiT 的 [JointField](../../../classifier_guidance/sit_joint.py#L26) 保留 strong 的输入导数，以及 weak 所依赖的 strong 早期特征对状态的导数。
- JiT 的 [Field](../../../classifier_guidance/jit_schedule.py#L56) 中 strong 与 weak 均冻结，但二者输入导数参与全部 50 个区间。
- SiT 的 VAE decoder、冻结 Inception 和 G 更新阶段的冻结 D 仍提供端点输入导数。
- SiT [GapProbe](../../../classifier_guidance/sit_joint.py#L67) 是独立真实插值上的辅助损失：这里可以在 strong/context 上使用 `no_grad`，因为 probe 输入不依赖当前可训练参数。这个例外不能套用到完整生成轨迹。

## 2. 每次更新的准确求值计数

一个“场 forward”同时产生 strong 与其内部挂接的 weak 输出，不是两个完整 backbone 各自独立运行一次。

| 项目 | SiT joint | JiT schedule |
|---|---:|---:|
| 求解器 | 64 Heun | 49 Heun + 1 Euler |
| 每条轨迹采样场 forward | 128 | 99 |
| 反向重算场 forward | 128 | 99 |
| 场 VJP | 128 | 99 |
| 单条轨迹合计 | 256 FWD + 128 VJP | 198 FWD + 99 VJP |
| 可训练生成侧参数 | weak head + 64 scales | 50 scales |
| 全局 batch / microbatch | 32 / 8 | 32 / 8 |
| 每卡处理样本 | 32，单卡 | 16，双卡 |
| 每卡更新的 batched 场总计 | 1024 FWD + 512 VJP | 396 FWD + 198 VJP |

最后一行的每次调用处理 8 个样本；不能把 batched 调用次数直接当单张图的 NFE。以上不含 CUDA capture 预热、端点反馈网络、D/R1、SiT norm probe 和周期性 profile。VJP 的耗时也不等于一次 forward，故这些数字不是直接的墙钟加速比。

求解器定义见 [sit_joint.py:33](../../../classifier_guidance/sit_joint.py#L33)、[jit_schedule.py:82](../../../classifier_guidance/jit_schedule.py#L82)。

**D、G 共用同一次采样端点**，不是各跑一次完整生成：

- SiT：[sit_joint.py:125](../../../classifier_guidance/sit_joint.py#L125) 生成并保存端点；[sit_joint.py:160](../../../classifier_guidance/sit_joint.py#L160) 在 D 更新后重算 decode/Inception 并反传同一轨迹。
- JiT：[training_accumulation.py:25](../../../classifier_guidance/training_accumulation.py#L25) 与 [training_accumulation.py:78](../../../classifier_guidance/training_accumulation.py#L78) 采用同样安排。

真实图片特征计算一次；假图片特征先无梯度计算供 D 使用，再有梯度计算供 G 使用。D 的 R1 在特征空间计算，没有通过真实图片的 Inception 反传。

SiT 正式入口 [train_sit_joint.py:56](../../../classifier_guidance/train_sit_joint.py#L56) 启用冻结 VAE decoder 与 Inception 的 block checkpointing，其反向还会重算相应局部块。JiT 速度档使用冻结权重 BF16 预转换；backbone/feedback checkpoint 是额外低显存选项，当前正式速度档未启用，见 [jit_schedule.py:71](../../../classifier_guidance/jit_schedule.py#L71) 与 [train_jit_schedule.py:41](../../../classifier_guidance/train_jit_schedule.py#L41)。

## 3. 显存复杂度与状态保存量

设状态维度为 \(d\)、区间数为 \(N\)、每卡有效 batch 为 \(B_{\mathrm{local}}\)、microbatch 为 \(b\)。当前主要显存项为

\[
O(NB_{\mathrm{local}}d)
+O(\text{单次场 tape}(b))
+O(\text{单批反馈 tape}(b))
+\text{模型、优化器、图内存池及临时张量}.
\]

因此已经避免 \(O(N\times\text{完整 backbone 激活})\)，但不是严格常数内存。由于先收集整个有效 batch 的端点用于一次 D 更新，所有微批轨迹的状态同时保留；随后逐微批反传并释放。

只计算 FP32 states/predictors，不计算模型与网络激活：

- SiT：\(128\times32\times(4\cdot32\cdot32)\times4\) bytes，约 **64 MiB**。
- JiT 每卡：\(100\times16\times(3\cdot256\cdot256)\times4\) bytes，约 **1.172 GiB**。

这里按保存的状态序列核算实际张量量级。最后 Euler predictor 与返回端点共享引用，不能把端点再次当成额外完整复制；参数和标签引用也不按每个时间点重复计为完整独立拷贝。其他短暂中间量不包含在这两个数字里。

JiT 的像素状态轨迹本身已有明显显存成本；SiT 的潜变量状态保存量则相对较小。

### 只读运行快照

| 训练 | UTC 时间 / 北京时间 | step / 生成侧更新 | 秒/更新 | peak allocated | peak reserved |
|---|---|---:|---:|---:|---:|
| SiT，单卡 | 2026-09-22 16:11:32.123382 UTC / 2026-09-23 00:11:32 | 1067 / 939 | 8.906040 | 6.592550 GiB | 7.867188 GiB |
| JiT，双卡 | 2026-09-22 16:11:34.166799 UTC / 2026-09-23 00:11:34 | 3483 / 3355 | 6.831522 | 每卡最大 2.452291 GiB | 每卡最大 4.386719 GiB |

快照文件：

- [SiT progress.json](/home/zhoushunyu/data/eqvae/projects/classifier_guidance/sit_joint_gan_20260922/training_30k/progress.json)
- [JiT progress.json](/home/zhoushunyu/data/eqvae/projects/classifier_guidance/jit_block1_gan_schedule_20260922/training_30k_gpu0_gpu2_g32_m8/progress.json)

这些文件继续随训练更新；上表固定记录此次读取值。step 包含 128 步 D warmup。`seconds` 是最近一次训练循环耗时，不是长期吞吐统计，不含循环外的周期性存盘/profile。JiT 取两卡测量值的最大值。allocated、reserved 与进程总显存占用并不等同。

既有配对 benchmark 见 [JiT 性能报告](../../classifier_guidance/JIT_BLOCK1_GAN_SCHEDULE_20260922_ZH.md#性能比较与短训练)：同 batch4，完整更新从普通 checkpoint-autograd 的 8.7098 秒降至 1.8145 秒，约 4.80 倍。工程优化已经显著降低开销，但没有减少完整反向本身的运算次数。

## 4. 改为 RL 后能省掉什么

若将生成控制定义为随机策略动作，并使用 score-function / REINFORCE / PPO：

1. 128 / 99 次环境 forward 仍需要。
2. 可省掉全部 sampler 状态 VJP、相应反向重算 forward，以及端点 decode/Inception VJP。
3. 已缓存的 fake features 可在 D 更新后再通过小 D 计算标量 reward，无须为该 reward 反传 VAE/Inception。
4. 策略自身仍需参数反传；若 frozen strong 特征只是 observation，可以将它们 detach。
5. 不必一直保存 GPU 轨迹/激活，但 replay buffer 的 observation/action 仍有存储成本。保存高维全部状态只是把成本移到了另一位置。

不能仅由此宣称总训练加速：策略梯度样本需求、重复 rollout、动作探索与 critic 训练成本可能抵消每个 update 的计算节省。

SiT 特别需要明确策略/环境边界：若 weak 依赖可训练 \(\phi\)，不能将其作为“环境”后仅对尺度做 REINFORCE，并声称更新到了完整 joint 目标。应将需要学习的 weak 输出纳入 policy/action，或另行提供有效的 weak 梯度估计。JiT 的 frozen-weak scale-only 对应更直接的 RL 接口。

## 5. one-step actor 并不自动消除全部 backbone Jacobian

对 Heun 单步，令

\[
z=x+h v_\theta(x),\qquad
F_\theta(x)=x+\frac h2[v_\theta(x)+v_\theta(z)].
\]

即使把 replay state \(x\) detach，仍有

\[
\partial_\theta F
=\frac h2\left[
\partial_\theta v(x)
+\partial_\theta v(z)
+h J_xv(z)\partial_\theta v(x)
\right].
\]

这里两处 \(\partial_\theta v\) 表示固定其输入时的直接参数导数，最后一项是 predictor 对参数的依赖。

由此应区分：

- **通过已知 Heun transition 做精确单步 actor gradient**：仍需第二阶段 1 次完整 frozen field 输入 VJP；第一阶段只需参数 VJP，可使用缓存 frozen 输出/特征。当前全轨迹实现每个 Heun 区间是 2 次输入 VJP。
- **Euler 局部 actor gradient**：只需 \(\partial_\theta v(x)\)。在 \(x\) detach 的前提下，strong 输出/context 可以缓存，无须 strong 输入 Jacobian。
- **连续时间 Hamiltonian/value gradient**：若需要 \(\nabla_xV\)，而 \(V\) 是 frozen strong 特征加小 head，依然要通过该 strong 前缀计算输入导数。
- **学习 \(Q(x,a)\)，只对动作求 \(\nabla_aQ\)**：可 detach state features，不需要环境 Jacobian；代价是正确学习 Q 的动作导数。
- **把 Heun 的两个阶段写成精确增广 MDP**，使用 learned stage-Q/costate：也可把后续 predictor 依赖吸收入 Q。这与上面的 Heun 导数一致，但属于以 learned continuation derivative 替代显式环境 VJP，不能声称无误差地跳过了那项链式法则。

使用 Euler 或连续时间局部公式替代实际 Heun transition 的精确梯度，需要报告相应离散偏差；它与使用增广阶段状态保留真实求解器，是不同的方法选择。
