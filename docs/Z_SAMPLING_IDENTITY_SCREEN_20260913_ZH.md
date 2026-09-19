# 从 Z-Sampling 出发：同场不变的缓存反演

2026-09-13。用户明确授权：先做完手上工作，若无效果就优化 Z-Sampling。真实图 inverse-prior bank 在一次 Anderson 修正后仍遇到未通过容差的样本，本轮已停止该有界尝试，没有删除难样本拟合，没有先验 NLL 或 FID 收益结论。以下是新的活动实验。

## 具体修改

统一时间 t=0 噪声、t=1 图像。令 A=v_u+w_s(v_c−v_u)、B=v_u+w_b(v_c−v_u)，w_b∈{0,1}。Euler 正向 S(x)=x+hA(x,t)。原 Z 在先进后的 y 上查询弱速度，仍用旧模型时间 t：

\[
y=x+hA(x,t),\quad q_Z=y-hB(y,t),\quad Z(x)=q_Z+hA(q_Z,t).
\]

这保留作者已核实现的模型查询时间与操作顺序，但将 SDXL scheduler 改写为 SiT/FM Euler，**是 FM 适配，不是原 SDXL checkpoint 的严格复现**。多余未使用的逆腿 conditional 查询被省去，明确按实际分支次数核算。[原文和源码边界](research/cfg_inversion_20260913/z_sampling_handoff.md)

新版本求实际弱 Euler 映射的逆 q+hB(q,t)=y，以本次原状态 x 初始化。第一次强 CFG 已得到 v_c(x,t)、v_u(x,t)，因此 B(x,t) 可免费复用：

\[
q_1=x+h[A(x,t)-B(x,t)],\quad
q_2=x+h[A(x,t)-B(q_1,t)],\quad
Z_{\rm anchor}(x)=q_2+hA(q_2,t).
\]

两种 Z 的一般强 CFG 前后各两次分支查询，中间弱查询一次，均 5 次/事件。新版本使用一次额外的、由缓存实现的逆迭代，没有增加网络查询。它仍是近似离散逆，不宣称精确求解。

**明确的不变性：A=B 时 q1=q2=x，整个事件等于原来一次 S(x)。** 因为两场没有差别，往返本应不改变原状态；原 Z 的显式负步通常不满足这个有限步契约。这个性质只证明不会凭空制造同场循环误差，不证明图像质量。普通无源图生成的 conditional/null 差场仍允许非零。

小步长展开中，新旧两种 Z 的一阶有效场同为 2A−B；新版本消除了原 Z 的一个同场 Euler 回退误差项。二次逆迭代局部有 O(h³) 的前像误差，但完整生成方法并未因此成为二阶 ODE 求解器。详情见[数学审查](research/cfg_inversion_20260913/z_cached_inverse_audit.md)。这一优化复用了经典固定点求逆思想，尚未确立文献新颖性或生成收益。

## 参数与完全相同的模型调用预算

采用 56 个 Euler 主步，前 42 步（t<.75）每步一个 Z 事件，最后 14 步普通 conditional：42×5+14=**224** 分支调用/图。以下控制也恰为 224：

- CFG Euler128：96×2+32=224。
- CFG/APG/CTRL Heun64：48×4+16×2=224。

Z 有效 extra α∈{.75,1.125,1.25,1.5,2}，分别配逆腿 w_b=0/1，实际前进权重 w_s=(1+α+w_b)/2。全部活动区间都做事件，因此不暗中额外改变早期 guidance schedule。α=1、w_b=0 会使 w_s=1，此时原版可进一步减少分支查询，缓存不再是相同的免费信息，所以未把它放进“每事件同为 5 次”的网格。α=.75、w_b=0 时 w_s=.875，诚实允许低于普通 conditional，不擅自裁剪。

原 Z 与新 Z 各 10 个设置，另有 CFG Euler/Heun 各 5 档、APG 与 CTRL 各 3 档，共 **36 个 1K 设置**。不引入第二个更昂贵的自循环补偿候选；先把这个直接改进检验完整。

## 执行与胜负

新 balanced 1K 噪声 seed=2026091481，固定标签顺序；各设置共享同一输入。FP32、无 TF32；计时包含采样与 VAE 解码，两者在现有 runner 中合计，FID 评估另计。224 NFE 相同不等于时延必然相同。沿用 ADM 5K ImageNet100 reference，但这些真实验证图不用于拟合任何控制参数。

先做 CPU 不变量与计数检查、真实模型前向和有限性检查，再冻结源/模型/输入启动 tmux。各参数独立输出图像、端点、状态快照和 FID；轨迹快照只是采样时间，不冒充五次图像回灌。

比较同时包括同参数原 Z、本轮固定 56 步/42 事件/5 档 α/2 种逆腿网格内最好的原 Z、Euler/Heun CFG、APG、CTRL。这还没有遍历 Z 的所有循环分配或参数，不能声称 Z 全局最优已排除。新 Z 必须有超出同参数数值改变的实际质量收益，不能只以更小循环残差获胜。明确有竞争力的 1K 结果才补充必要的原 Z 分配对照、冻结进入另一组独立 5K；否则继续依据结果优化 Z，不将筛查当成已达成目标。

源码：[采样器](../experiments/z_sampling_identity_20260913/sampler.py)、[CPU 检查](../experiments/z_sampling_identity_20260913/cpu_check.py)、[参数表](../experiments/z_sampling_identity_20260913/configs.py)、[评估入口](../experiments/z_sampling_identity_20260913/run.py)。

## 启动记录

CPU 检查和实际 SiT 预检均通过。真实预检覆盖全部 36 个设置、每设置 4 张输入；均为 224 次分支调用，另检查 6 个同场端点与 CFG/APG/CTRL 原生逐位一致性。总成本 37,704 次分支图像查询，约 50.2 秒，不包括正式质量筛查。见[预检原始结果](/home/zhoushunyu/data/eqvae/experiments/z_sampling_identity_20260913/model_check/summary.json)。

正式请求已经冻结，新种子 2026091481，batch16，1000 张/设置，36 个设置。tmux 为 `z_identity_0913`，四个 worker 使用 GPU0–3。初始四个参数臂的批次已保存并核对标签、像素形状、有限端点与 224-call 成本。采样、FID 和汇总依次自动执行；[运行目录](/home/zhoushunyu/data/eqvae/experiments/z_sampling_identity_20260913/screen_1k)中保留全部 request、batch、endpoint、图册与评估结果。

本入口使用独立汇总函数，避免旧通用 runner 将 FID 输入路径覆盖 `samples` 数量；当前 `seconds` 为采样加解码的合计。此处只是运行状态，不是质量胜出或整体目标完成的声明。
