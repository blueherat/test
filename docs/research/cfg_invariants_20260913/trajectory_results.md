# SiT 轨迹、参数单位与五次固定投影检查

已完成：8 个固定初始噪声／类别，SiT-S/2 EMA，常数 CFG `w=2.25`，noise→data 的 Heun 64 步。TF32 关闭，网络 FP32；这是一组数值与表示检查，不是图像质量实验。运行耗时 31.5 秒，3338 次批量单分支调用，GPU 3。

在 `t=0,.25,.5,.75,1` 各自保存状态，并**分别重新执行**余下全部采样步骤，没有把缓存终点作为结果返回。比较同一离散网格、128 步细网格的剩余后缀，以及未来条件强度改为 `w=1` 三种情况；同时保存普通 conditional clean 预测。

| t | 同网格重算终点最大绝对差 | 细网格后缀与原终点 RMS | 普通 clean 预测与终点 RMS | 未来改为 w=1 后的终点 RMS 差 |
|---|---:|---:|---:|---:|
| 0 | 0 | 0.0062085 | 0.81406 | 0.37484 |
| .25 | 0 | 0.0010287 | 0.46950 | 0.15487 |
| .5 | 0 | 0.00020994 | 0.24163 | 0.05878 |
| .75 | 0 | 0.00010301 | 0.08195 | 0.01529 |
| 1 | 0 | 0 | 0 | 0 |

同网格结果是固定离散流组合的代数／实现契约；细网格差异反映该契约与连续流之间的数值误差。普通 clean 预测的变化不是模型不相容的证据；改变未来指导后，也没有必须得到原终点的要求。

对三个内部时间点的真实 gap 序列，另测试 `gap_epsilon=-t*gap_v`、`gap_clean=(1-t)*gap_v`。正确换算控制幅值并运输历史，换回 velocity 后最大误差 `2.22e-16`；直接复用数值 K 与旧单位历史，RMS 差最大 `0.92290`。此处使用冻结的三状态输入序列，只隔离表示法则，没有据此主张一个新的高质量采样器。

将控制 proposal 投影到固定线段 `{a*gap:0≤a≤1}`，再用**同一集合独立投影五次**，第二到第五次相对第一次最大差 `2.22e-16`。每次变化都保存于 CSV。这个幂等性来自固定凸集合投影；它不能证明所选集合在语义或图像质量上正确。

复现：

```bash
CUDA_VISIBLE_DEVICES=3 /home/zhoushunyu/miniconda3/envs/myenv/bin/python -m experiments.cfg_invariants_20260913.trajectory_contracts --out /path/to/new/contracts
```

- [代码](../../../experiments/cfg_invariants_20260913/trajectory_contracts.py)。
- [五个时间点分别重算的终点图片](/home/zhoushunyu/data/eqvae/experiments/cfg_invariants_20260913/contracts/recomputed_endpoints.png)。
- [五个时间点的普通 clean 预测](/home/zhoushunyu/data/eqvae/experiments/cfg_invariants_20260913/contracts/clean_predictions.png)。
- [改变未来指导的终点图片](/home/zhoushunyu/data/eqvae/experiments/cfg_invariants_20260913/contracts/changed_future.png)。
- [完整摘要](/home/zhoushunyu/data/eqvae/experiments/cfg_invariants_20260913/contracts/summary.json)、[轨迹读数](/home/zhoushunyu/data/eqvae/experiments/cfg_invariants_20260913/contracts/trajectory.csv)、[参数单位读数](/home/zhoushunyu/data/eqvae/experiments/cfg_invariants_20260913/contracts/representation.csv)、[每次投影读数](/home/zhoushunyu/data/eqvae/experiments/cfg_invariants_20260913/contracts/five_projections.csv)。
