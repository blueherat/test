# RAE SPC 标准目标机制检查：看结果前的预测

## 为什么需要这项检查

五 seed 训练前两个已完成分支显示：切换后 SPC 与 static 的数据、time、noise、target 和
学习率逐值一致，但 SPC 的 standard static MSE 略高。因此“SPC 只是让标准目标收敛更快”
已不成立。下面的机制预测在运行五 seed gradient probe 之前固定。

## 路径实际做了什么

对 rank-16 detail，`floor=0.2, p=2` 的 clean state 系数与 clean velocity 系数幅值分别为：

```text
c(t) = 0.2 + 0.8(1-t)^2
|a'(t)| = 0.2 + 2.4(1-t)^2
```

当前 RAE 的 shifted logit-normal time 中位数约为 `0.874`。固定 200 万 Monte Carlo 样本：

- `98.8%` 的样本满足 detail velocity 系数小于 static 的 `1`；
- detail velocity 系数平方期望约为 static 的 `12.1%`；
- 只有约 `1.2%` 样本处在系数大于 static 的近数据端区域；
- guided rank-16 subspace 捕获约 `13.74%` 的 final spatial-residual energy。

因此 Phase 1 的主要作用是暂时卸载一个低维但高能的细节任务，而不是普遍降低所有任务难度。

## 固定 probe

所有 checkpoint 都在同一批 held-out latent、label、noise 和 `t={0.3,0.1}` 上，使用相同
**standard static state/target**。比较 step 2000 与 step 5000，不使用各自训练路径作 probe。

## 事前预测

### P1：2k 容量重分配

相对 paired static，SPC 在 step 2000 应同时表现为：

- rank-16 `basis_loss` 更高；
- complement `semantic_loss` 更低。

五个 seed 至少四个满足两个方向，才支持“卸载 detail 释放共享容量”。这里的
`semantic complement` 只是操作性名称，包含 token mean 和 rank-16 正交补，不等同于纯语义。

### P2：5k 细节追赶

- `basis_loss` 的 SPC-static 相对差距从 2k 到 5k 至少缩小 30%；
- `semantic_loss` 不得从 2k 的优势系统性翻为劣势。

满足 P1 与 P2 才支持“先学 complement、后补 detail”的两阶段解释。

### P3：梯度关系

在 step 5000、`t=0.1`、last shared block：

- 两组 semantic descent ratio 都应接近或高于 `1`；
- SPC 不应重新出现旧 fixed-floor 的负 cosine / 强 basis-gradient pressure。

这只检查切回 static 后冲突已解除，不要求 SPC 比 static 的局部梯度更优。

### P4：与生成的联系

仅在五 seed 生成指标可用后检查：若 seed-level FID 改善越大，对应的 2k semantic-loss
改善也越大，且 5k basis gap 越小，机制链条得到支持。五个点的相关性只作描述，不作显著性
结论。

## 否定条件

- basis 变差但 semantic complement 不改善：否定“释放容量学主体”。
- 5k basis 不追赶：说明 SPC 只是欠拟合 detail。
- gradient 正常但生成不改善：梯度恢复不是质量提升的充分原因。
- 生成改善但 P1/P2 失败：保留经验 curriculum 结果，但不声称已解释机制。
