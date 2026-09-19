# Blend 与 CTRL modulation 独立只读审查

2026-09-13。结论：blend 的一个浮点端点问题已经修复；当前 blend 与 control_modulation 未发现阻止固定筛查的实现问题。本审查未修改采样代码、未运行 GPU。真实 8 图预检由 root 独立执行。

## 1. 已修复：sign 前不能任意重排浮点计算

旧 CTRL 使用 `(gap-previous)+lambda*previous`；blend 初版将其化简成 `gap+(lambda-1)*previous`。实数相同，但 FP32 阈值附近可能改变 sign。独立 CPU 反例：

```python
p = torch.tensor(1., dtype=torch.float32)
g = torch.nextafter(torch.tensor(-4.), torch.tensor(0.))
old = (g-p)+5*p
reassociated = g+4*p
```

|量|实际 FP32 值|
|---|---:|
|gap|−3.999999761581421|
|旧 sliding|0|
|重排 sliding|2.384185791015625e−7|
|旧 modified，K=.2|−3.999999761581421|
|重排 modified，K=.2|−4.199999809265137|

因此最终输出重新写回 `u+(1+alpha)*modified` 仍不够，sliding 本身也须保持旧运算顺序。root 已修复 [independent_blend.py](../../../experiments/cfg_transport_search_20260913/independent_blend.py)，并报告重新执行的真实 8 图 η=0/1 整轨迹均逐元素等于 APG/CTRL，224 full/0 prefix。初版已准备的 `blend_1k` 不用于正式结果；正式阶段另名 `blend_controls_1k`，避免运行中修改冻结源。

## 2. Blend 其他检查

- η=0 使用原 APG 的计算顺序；η=1 同时保留原 CTRL 的 sliding 与最终输出计算顺序。两套历史分别是 APG 的 raw momentum 和 CTRL 的 modified gap；两个 Heun stage 都读取旧历史，仅接受后提交第一 stage 的两份提案。没有相互喂入控制器输出。
- 记两个相对 conditional 的增量为 A、C，完整混合是 `A+eta*(C-A)`；parallel 对照是 `A+eta*P_A(C-A)`，保留带符号投影，不以范数替代正负方向。它不保证轨迹始终有相同增量范数。
- 七臂配置中只有 η=.5 有相应 parallel 臂。若 η=.25/.75 胜出，不能直接用 parallel_.5 认领其额外方向的收益；当前仍是组合筛查。新增 APG2.25、CTRL2.5/3 是强度对照。
- 每步共享 conditional/null 预测，不新增随机输入、后验、解码或真值接口；64步、48活跃步应计 `2*64+2*48=224` 个单分支 Full。两控制器额外向量运算仍计入实际耗时。
- [runner.py](../../../experiments/cfg_transport_search_20260913/runner.py)冻结配置、源与资产 hash；每 batch 核对 request/noise hash 和标签。本次读到的 baseline_1k 与初版 blend_1k 同为1000图、seed2026091397，noise hash `19d1e65c1981829a43c88a28a3824ce22f2a6b8f4f42a64fff8a4eb567880d09`、labels hash `9369110b6e3bedf2f2c685fd67ed9e2ede0315576c18ff09f98271164865bba9` 相同。正式新阶段仍应以其冻结 request 为准。

非采样阻塞的汇总问题：`runner.report` 用 FID JSON 更新整行后，`samples` 会从数量被覆盖为文件路径；该 FID JSON 没有 seconds，因此采样计时不受此覆盖。最终表应从 `summary.samples` 或 `fid.sample_count` 读数量，并从冻结配置补列 eta/rho 等参数。无需为此修改已冻结运行源。

## 3. CTRL modulation 检查

[control_modulation.py](../../../experiments/cfg_transport_search_20260913/control_modulation.py)直接保持旧 sliding 顺序。块起点计算两个虚拟 modified gap，得到 mean 与 AC；两步分别加 `w*(mean+rho*AC)`、`w*(mean-rho*AC)`。每步的两个 RHS 仍以真实状态和时间调用模型，仅加性控制冻结。

每个完整两步块的附加积分相同，ρ=±1 的局部控制能量相同；块尾始终提交虚拟第二更新，不使用 rho 改写后的实际输出。下一块的输入会随生成路径变化，所以跨臂整个 mean 序列并不严格一致。实现与 [机制边界](control_modulation_debate.md)都没有把它冒充为原 CTRL 或固定整条控制序列的因果对照。

零 K 回到普通 CFG、零 alpha 回到 conditional；活跃步必须为偶数以禁止半块。计数仍为224/0，snapshot 为初始及四个四分位时间。已读 [FP32/FP64 实现检查](control_modulation_implementation_check.json)，其恒 gap、零 K/alpha、64/96步和局部 mean/energy 检查与代码一致。三臂只比较固定 ρ=0/+1/−1，没有额外输入或监督泄漏。

这些结论只支持实现可进入既定对照实验，不构成质量、互补性或调制机制成立的证据。
