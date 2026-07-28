# RAE floor path：held-out error atlas 结果

## 一句话结论

> floor 几乎完全修复了 original annealed 在高噪声处的解析逆放大，但没有改善主要的
> semantic-complement 学习，也没有追上 static。由此可把断层定位为：局部条件数问题真实
> 存在，却不是当前生成差距的主导原因。

## 实验正确性

- 使用 step-5000 EMA 的 static、original annealed、`floor=0.2,p=2`。
- 三路共享 cache indices `[100000,100128)`、标签、噪声和六个时间点。
- 5k 训练只消费 logical cache `[0,80000)`，所以该切片未参与训练。
- 在各路径自己的精确 teacher-forced state/target 上评估，不含 ODE 采样累积误差。
- 误差按 rank-16 middle-guided basis 及其正交补拆分；终点误差使用解析 observation
  factor 反演。
- 文中的 `semantic complement` 是操作性名称：它包含 token mean 和 rank-16 正交补，不等价
  于经过监督验证的纯语义子空间。

## 核心数字

| t | 指标 | static | annealed | floor 0.20,p=2 |
|---:|---|---:|---:|---:|
| 0.97 | semantic velocity error | **0.6533** | 0.6554 | 0.6568 |
| 0.97 | rank-16 velocity error | 0.7793 | **0.1124** | 0.3449 |
| 0.97 | semantic endpoint error | **0.8525** | 0.8565 | 0.8562 |
| 0.97 | rank-16 endpoint error | **0.8473** | 20.6963 | 0.8936 |
| 0.85 | semantic endpoint error | **0.6420** | 0.6702 | 0.6693 |
| 0.85 | rank-16 endpoint error | **0.6304** | 1.1938 | 0.7368 |
| 0.50 | semantic endpoint error | **0.3792** | 0.3909 | 0.3874 |
| 0.50 | rank-16 endpoint error | **0.2656** | 0.5059 | 0.4356 |

在 `t=0.97`，floor 把 annealed 的 rank-16 endpoint error 从 `20.6963` 降到
`0.8936`，降低 `95.68%`。128/128 个配对样本都改善；配对 bootstrap 的中位差 95% CI
为 `[-20.35,-19.34]`。因此原 annealed 的高噪声逆条件数病态不是测量噪声。

但同一位置上，floor 的 rank-16 velocity relative error 是 annealed 的 `3.07x`。也就是：
floor 并没有把该速度分量预测得更准，它主要靠把 observation factor 从 `0.00265` 提高到
`0.20212`，避免误差在终点反演时被放大。更重要的是，static 在六个时间点的 rank-16
endpoint error 全部低于 floor。

semantic-complement 方面，floor 与 annealed 基本重合：六个时间点的相对差异都约在 1%
以内。static 在其中 5/6 个时间点更低，与 5k 生成结果一致。floor 改动的只是低秩 detail
coefficient，本来也没有改变 semantic path；实验证明这种局部修补没有迁移成主要学习任务
的改善。

## 事前预测判定

1. static 在多数时间点 semantic error 更低：**成立，5/6**。
2. floor 在高噪声处显著降低 basis endpoint error：**强成立**。
3. floor 不会明显改善 annealed 的 semantic error：**成立**。
4. 局部条件数修好但 5k generation gate 仍失败：**成立**。

## 机制判断

现有证据支持下面这条因果链的前半段，但否定后半段：

```text
k(t) -> 0
  -> rank-16 终点估计产生严重逆放大          （支持）
  -> 这是 annealed 生成差于 static 的主因     （不支持）
```

更符合数据的解释是：time-dependent data endpoint 同时改变了输入状态和速度目标，使网络
必须学习一个随时间变化的非线性任务；它还可能在高噪声阶段隐藏了对后续语义成形有用的
细节。正 floor 只修复了其中一个解析病态，没有恢复 static 的简单直线路径与一致目标。
这部分仍是机制假设，当前 atlas 不能进一步区分“target complexity”和“information
withholding”。

## 决策

- 按预注册停止规则，不追加该 schedule 的训练、多 seed 或 50k 采样。
- 不把 2k 候选改善写成正结论；它是未通过时间持久性检验的早期瞬态。
- 可保留的研究结论是一个负但清晰的边界：**修复局部 endpoint conditioning 不足以修复
  generative learning**。
- 若以后重启这条问题，只应使用更便宜的 toy/小模型，把 target complexity 与 information
  withholding 做正交干预，而不是继续扫 floor/power。

完整数据与图位于：

```text
~/data/eqvae/experiments/rae_path_schedule_error_atlas/step5000_seed3407_holdout128/
```
