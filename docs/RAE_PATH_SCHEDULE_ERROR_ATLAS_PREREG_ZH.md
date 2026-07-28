# RAE floor path：held-out time/subspace error atlas 事前预测

## 问题

5k 生成门控已经失败。最后做一个不训练的诊断，区分两种解释：

1. floor 没有真的改善它声称改善的 rank-16 endpoint conditioning；
2. floor 改善了局部 endpoint conditioning，但这不是生成差距的主因。

## 设置

- 模型：step-5000 EMA 的 static、original annealed、`floor=0.2,p=2`。
- held-out cache indices `[100000,100128)`；三模型共享 latent、label、noise、time。
- time：`0.97/0.85/0.70/0.50/0.30/0.10`。
- 在各模型自己的 exact path state 上评估 teacher-forced velocity prediction。
- 拆分为 rank-16 middle-guided basis 与其余 semantic complement。
- 局部 endpoint error 使用该 path 的解析 observation factor 做反演。

## 固定预测

1. static 的 semantic velocity/endpoint error 在多数时间点低于 annealed 和 floor candidate，
   与 5k 生成优势一致。
2. floor candidate 在高噪声时间的 basis endpoint error 显著低于 original annealed，因为
   `k_floor=0.2+0.8*k_old` 消除了 `k->0`。
3. floor candidate 的 semantic error 不会因此明显优于 annealed；floor 只改了低秩 detail
   coefficient。
4. 若预测 2 成立但生成门控仍失败，则结论是：逆条件数是可修复的局部数值问题，但不是
   当前 generation gap 的主导瓶颈。更可能的主因是 time-dependent target/path 本身增加了
   学习复杂度，或改变了关键误差在时间上的分配。

## 停止规则

本 atlas 之后停止该 schedule 路线，不因任何结果追加训练。它只用于解释已有负结果。
