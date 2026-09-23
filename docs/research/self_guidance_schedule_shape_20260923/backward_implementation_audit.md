# 末端 scale 回落：实际反传实现审计

日期：2026-09-23。审计方式：只读生产实现及既有日志；重新运行 CPU 测试；未启动 GPU、未修改训练。

**结论：当前实现采用完整离散求解器的反向链式法则，没有发现会截断尾部信用、遗漏 Heun predictor 导数或冻结强模型输入 Jacobian 的“简化反传”。** 省显存的方法是保存每步状态、重算单次网络 VJP，并用 CUDA graph 重放；这与删掉导数不同。已有实际 GPU 对照支持这一判断，但尚不能由全向量误差较小，推出最新 checkpoint 最后几个小梯度的符号必然正确。

因此，目前没有证据把最新末端回落归因于截断 BPTT；也不能据此将回落直接确认为某种最优控制规律。需要区分：反传实现、实际离散求解器的边界响应、混合精度误差、GAN/Adam 训练动力学。

## 1. 审查的代码就是训练记录对应版本

逐项 SHA256 记录见 [backward_source_hash_audit.json](backward_source_hash_audit.json)。

- JiT 正式 `training_30k_gpu0_gpu2_g32_m8/request.json` 中记录的 `sampler.py`、`jit_schedule.py`、`features.py`、`training_accumulation.py`、`training.py`、`schedules.py`、`train_jit_schedule.py` 均与本次读取文件一致。
- 当前 SiT joint 的 `single_after_deadline_20260923/request.json` 与此前 `dual_until_deadline_20260923/request.json` 中，以上共同核心文件及 `sit_joint.py`、`train_sit_joint.py` 均与本次文件一致。
- 最早 native SiT 的 sampler SHA 与当前不同。已找到当时 `source_snapshot/classifier_guidance/sampler.py`，其 SHA 与当时 request 完全一致。与当前的 [差异](old_native_sampler_diff.txt) 是增加逐步 Heun/Euler 选择、仅捕获实际使用的 active 模式；原全 Heun 分支的导数公式未改变。旧 `features.py`、`training.py`、`schedules.py`、`train_schedule.py` 与当前一致。

这验证本次审计核心源码的归属，不等于对环境、全部外部依赖、GPU 算子行为作完整复现认证。

## 2. Heun 两个 stage 的依赖没有漏

记某步长度为 h，参数为 θ；两个 stage 的向量场可有不同时间，但使用同一个该步 scale：

\[
f_1=f(x,t;\theta),\qquad
y=x+h f_1,\qquad
f_2=f(y,t+h;\theta),\qquad
x^+=x+\tfrac h2(f_1+f_2).
\]

令 \(J_1=D_xf_1,J_2=D_yf_2,P_1=D_\theta f_1,P_2=D_\theta f_2\)，其中 P 是固定本 stage 输入时的偏导。给定末端 cotangent \(g=\partial L/\partial x^+\)：

\[
s=J_2^\top(\tfrac h2g),\qquad
q=\tfrac h2g+h s,
\]

\[
\frac{\partial L}{\partial x}=g+s+J_1^\top q,\qquad
\frac{\partial L}{\partial\theta}
=P_2^\top(\tfrac h2g)+P_1^\top q.
\]

`sampler.py:150` 起的实现逐项对应此式：先重算 second VJP，然后 first VJP 的 cotangent 使用 `gradient*(h/2)+h*second[0]`，再把两者的参数导数相加。**其中 `h*second[0]` 正是 predictor 对参数与输入的影响；若删除才是有偏简化。当前没有删除。**

循环覆盖 `reversed(range(n))` 的全部步骤；Euler 分支也保留 \(g+hJ^\top g\)。这里是对实际离散 Heun/Euler map 求导，不是用另一个连续 ODE 伴随近似原离散程序。

对固定当前输入的 scalar scale \(a\)，若 \(f=S+aB\)，Heun 单步控制响应为

\[
\partial_a x^+
=\tfrac h2\big[B(x,t)+B(y,t+h)+hJ_2B(x,t)\big].
\]

所以不能用单个 \(hB\) 代替真实 Heun 的控制响应，更不能用这种替代式判断末端 scale 应怎样变化。

## 3. 看到 detach 不等于全轨迹断梯度

`FieldReplay.run` 把保存的 state 变成本次局部 VJP 的叶节点，然后显式取对该输入和全部可训练参数的导数。`_Rollout.backward` 将局部输入导数送到上一时间步。这就是自定义反向模式链式法则；局部重算叶节点无需保留先前的巨大网络 tape。

关键边界逐项检查如下：

| 位置 | 实际行为 | 是否丢弃当前一阶训练梯度 |
|---|---|---|
| `FieldReplay.run` 的 state detach | 建立局部 VJP 叶节点，再手动连接所有时间步 | 否 |
| strong/weak `requires_grad_(False)` | 冻结参数；state 仍参与 autograd | 否 |
| D 更新使用 fake features detach | 交替 GAN 的 D 步不更新 G；G 步另算并保留图 | 否，符合当前交替优化目标 |
| `training_accumulation.py` 的无梯度 fake features | 供 D 使用；保存的 generated endpoint 仍带 sampler 图，G 步重算 feature | 否 |
| `features.py` 的分块重算 | 对每个样本的 decoder/feature 精确局部 VJP，再返回拼接输入梯度 | 确定性 eval、样本独立条件下否 |
| `once_differentiable` | 不支持此自定义 op 的高阶导数 | 不影响当前一阶训练 |
| joint `GapProbe` 的无梯度 strong/context | 独立 real-interpolant 诊断/约束分支；不在 GAN rollout 上截断状态 | 不能据此推断 sampler 截断 |

特别容易混淆的是 `jit_ssg.py` 的旧 weak-head 训练接口 `Runtime.prefix`：它确实有 detached prefix，但当前 JiT GAN schedule field 调用的是 `runtime.net.forward_with_intermediate`。所固定的 `literature/model_jit_ssg.py:145` 中，此函数从输入经过全部 strong blocks，并在第 6 层分支运行 weak adapter；两条路径都没有 detach/no_grad。

schedule 的 `amounts` 缓冲区保存的是固定**索引**，不是待学习系数。系数是 `GuidanceSchedule.coefficients` 参数，`gather` 仍对其求导。全部时间步 active，零系数与负系数没有被 gated off。通用 Sampler 拒绝可训练 `amounts`/grid，不表示 schedule 没有梯度。

## 4. 重放、checkpoint 与数值精度的边界

每次 CUDA graph replay 前，state、time、labels、index、cotangent 都复制到捕获缓冲区；每次输出在下一个 replay 前 clone，避免共享 graph 内存覆盖已保留的结果。实现检查输入形状、dtype、device、同一 CUDA stream，以及参数对象和存储地址；forward 保存的参数还使 autograd 检测 forward/backward 之间的版本变化。

checkpoint/replay 的正确性依赖于重算同一个确定性场。当前调用使用冻结 eval 网络；现有 JiT attention/dropout 配置不引入随机 mask。`preserve_rng_state=False` 在这个前提下不会漏导数；若以后引入随机 dropout，必须重新审计，而不能沿用此结论。

“完整离散导数”指链式依赖完整，以及各次局部 autograd 对当前程序求导。它不保证混合精度下与另一种运算结合顺序逐 bit 相同：拆分两个 VJP、BF16 舍入、不同 batch 形状与梯度归约，均可能产生小差异。BF16 程序的自动微分也不等于对浮点量化函数取严格经典导数。应同时报告实际 AD 对照与适当尺度的有限差分，避免把很小 epsilon 的量化噪声当作导数错误。

## 5. 最后一步确有不同响应，但不是反传简化

| 路径 | 离散求解器 | clean-to-velocity 分母 floor |
|---|---|---|
| 最早 native SiT | 64 个 Heun 区间，最后也 Heun | 此 field 无此 floor |
| 当前 SiT joint | 64 个 Heun 区间 | 此 field 无此 floor |
| JiT frozen weak schedule | 前 49 个 Heun，最后 1 个 Euler | \(\max(1-t,0.05)\) |

JiT 最后一步 \(t=0.98,h\simeq0.02\)，令 \(C_a=(1+a)S-aW\) 为 clean prediction：

\[
x_T=x+\frac h{0.05}(C_a-x),\qquad
\partial_a x_T=\frac h{0.05}(S-W)\simeq0.4(S-W),
\]

\[
D_xx_T=I+\frac h{0.05}(D_xC_a-I)
\simeq0.6I+0.4D_xC_a.
\]

这是实际 forward 的响应，custom backward 正在求它的导数；没有人为将最后系数梯度置零。这里没有 Heun 的第二 stage 及 predictor 传播项，因此最后系数与倒数第二系数完全可能有不同的最优值。floor 还改变最后几个区间的时间尺度。JiT decoder 的 clamp 也会使饱和像素的终端梯度为零，这是实际目标的 clipping，而不是 sampler 时间截断。

但 **JiT 的 Euler/floor 不能作为最早 SiT 与当前 JiT 共同出现回落的充分解释**，因为最早 SiT 不具有这两项。跨模型形状相似只能提出机制假说；不能据此排除不同模型各自的离散边界效应。

## 6. 已有实际 GPU 结果支持到哪里

下表直接读取保存的 JSON；这些是历史已有结果，本次没有新跑 GPU。相对误差均为整个梯度向量的 L2 相对误差，除非另行标注。

| 已有审计 | 对照范围 | 结果 | 不能推断的内容 |
|---|---|---|---|
| native SiT `audit.json` | 实际 SiT，普通 checkpoint autograd 对自定义 sampler；两种 schedule/输入 | endpoint 差 0；梯度相对误差 \(1.73\times10^{-5}\)、\(8.10\times10^{-6}\)；32 个 tail 梯度均非零 | 未检查最新最后一个 bin 的完整 GAN 导数 |
| 同上有限差分 | 当时 index 48 为零的系数，epsilon 0.003 | AD 0.00275538，FD 0.00276168，相对误差 0.2288% | 不是 index 63，也不是最新 checkpoint |
| JiT `audit_fresh_leaf/result.json` | 实际 JiT，普通 autograd 对 eager/graph，自变量包括 input 与全部 50 个系数 | schedule 误差 0.05044%；换 signed/zero schedule 后 0.04321%；50/50 非零；endpoint 差 0 | 小尾部坐标的相对误差与符号没有逐项报告 |
| JiT `benchmarks/precast_b4/comparison.json` | 完整 GAN，一般 checkpoint 路径对选定 graph+precast 路径 | 三次 G 梯度误差 0.4314%、0.4885%、0.2999%；D 更新一致 | 不保证 latest tail 逐坐标精度 |
| JiT `audit_gpu0_gpu2/result.json` | checkpoint step 195，同批数据单 GPU 对双 GPU | G 梯度差 0.2086%；更新后系数最大差 \(5.96\times10^{-7}\)；D 一致 | 这是分布式一致性，不是独立 ordinary-autograd 证明 |
| SiT joint `audit_v1/audit.json` | 实际 SiT，64 步，随机终端线性 cotangent；input、weak、schedule 对 ordinary checkpoint | 保存的 endpoint 与全部梯度误差为 0；含负/零系数；零系数导数非零 | 不是最新完整 GAN 尾部 loss |
| SiT joint `distributed_audit_20260923/result.json` | step 1455，实际 GAN 单/双 GPU | weak 误差 0.04562%，scale 误差 0.07436%；更新参数最大差 \(1.19\times10^{-7}\)；当时最后 4 个 scale 梯度符号一致 | 仍不覆盖最新训练时点 |

对应数据根目录为 `/home/zhoushunyu/data/eqvae/projects/classifier_guidance/`，子目录分别是 `sit_native_signed_schedule_20260919`、`jit_block1_gan_schedule_20260922`、`sit_joint_gan_20260922`。

**全向量相对误差小于 1% 并不证明最后一个梯度可靠。** 如果 \(g_{49}\) 远小于前几个分量，最后一个分量符号改变仍可与极小全向量误差同时成立。这里必须避免从旧报告的 aggregate pass 推出逐 bin 的强结论。

本次当前源码 CPU 测试：

```text
/home/zhoushunyu/miniconda3/envs/myenv/bin/python -m pytest \
  tests/test_classifier_guidance.py tests/test_classifier_schedules.py \
  tests/test_sit_joint.py -q
15 passed, 4 skipped in 9.70s
```

跳过的是未启用的 CUDA 测试。CPU 覆盖 Euler/Heun、混合最后 Euler、非均匀 grid、输入与系数导数、负/零系数、gradcheck、多轨迹复用、参数版本检查、feature 链式法则、完整交替 GAN 和 joint 参数导数。这些验证实现结构，不代替最新实际大模型 GPU 检查。

## 7. 最有辨识力的剩余验证：尾部后缀精确审计

固定最新 checkpoint、输入噪声、类别、strong/weak、feature 与某个确定的 critic 状态。缓存倒数 K 步之前的精确状态 \(x_{N-K}\)，然后只用普通 autograd 重建最后 K 步，并与 production custom sampler 比较 \(a_{N-K:N}\) 的梯度。

**对于逐步独立 schedule 系数，缓存并 detach 此前 prefix 不会近似这些尾部系数的梯度**：前缀从未使用尾部参数，故 \(\partial x_{N-K}/\partial a_{N-K:N}=0\)。这一点使精确验证可以只付最后几步的显存。若改为共享参数的连续 schedule 网络，或审计 joint weak 参数，此论证不成立，因为这些参数也出现在 prefix。

建议必须记录：

1. 最后每个 bin 的 custom/reference 梯度、绝对差、带数值 floor 的相对差、符号；同时给出整向量 norm，不能只给后者。
2. 普通 autograd 与 custom 使用完全相同 precision、求解器、floor、decoder clipping 和终端 cotangent；另作 FP32 局部计算敏感性检查时，明确它改变了数值程序。
3. 多个 epsilon 的局部有限差分及方向一致性；不要以一个极小 epsilon 下的 BF16 差分直接否定 AD。
4. 将“实际最终标量 GAN loss 的梯度”与“固定终端 cotangent 的 sampler VJP”分开比较。前者审计全反馈路径，后者定位 sampler 本身。
5. 若比较一个完整 GAN update，应同样冻结或复现该次 D 更新，且比较 Adam 的历史动量；当前梯度符号不一定等于当次参数增量符号。

若此审计通过，便可更有把握把研究重心从“反传被简化了”转向真实边界响应、末段 strong/weak 差异方向、终端 feature/clipping、各模态的信用分配，以及 GAN/Adam 的非平稳优化。改变步数/尾段求解器的 shape 稳健性是另一类实验：它检验学到的是连续控制规律还是离散求解器适配，不能由导数正确性检查替代。
