# RAE path：动态梯度翻转与 generated-latent closure 结果

## 总结

本轮得到的不是“高噪声 rank-16 梯度一直抢占 semantic 容量”这个原假设。该预注册假设
失败。真正经独立数据确认的是一个更窄的动态现象：

> time-dependent detail path 的 basis/semantic 梯度关系在低噪声共享 DiT block 中随训练
> 非平稳翻号；static 则保持稳定同向。floor 减弱但没有消除这种翻转。与此同时，floor 的
> self-generated endpoints 在 decoder closure 风险上仍与 annealed 聚类，而非回到 static。

因此当前最合理的链条是：

```text
time-dependent detail path
  -> 随训练变化的共享任务几何
  -> 低噪声 basis/semantic 更新由协同变为冲突
  -> teacher loss 仍可下降，但 self-generated latent 没有回到 decoder 熟悉区域
```

最后一箭仍是机制一致性，不是单因素因果证明。

## 1. 首轮宽假设失败

首轮使用新 held-out indices `[100128,100160)`，比较 static、annealed、floor 的 online
2k/5k checkpoint。原预注册 gate 失败：

- floor 在 `t=0.97/0.85` 的 basis gradient pressure 没有达到 annealed 的 `2x`；命中
  `0/4`。
- static 没有在六个时间点、两个参数组上普遍比 floor 少至少 `0.05` 的 semantic 干扰。
- 只有“2k 到 5k 的 static-floor gap 增大”和 cross-split 符号稳定成立。

所以不能说“floor 在高噪声阶段持续抢占共享容量”。高噪声处 static 的 basis pressure
反而更大，因为 static 从一开始就完整暴露该分量。

## 2. 独立确认：低噪声共享梯度翻转

post-hoc 发现随后用完全独立的 `[100288,100416)`、128 张、noise seed `20260725`
确认；加入 online step `500/1000/2000/5000`，确认协议 C1--C5 全部通过。

semantic descent ratio 定义为：

```text
<g_sem, g_sem + g_basis> / ||g_sem||^2
```

大于 1 表示 basis loss 帮助 semantic 的一阶下降，小于 1 表示削弱。

### Last block，t=0.1

| path | step 500 | step 1000 | step 2000 | step 5000 |
|---|---:|---:|---:|---:|
| static | 1.020 | 1.020 | 1.024 | 1.028 |
| floor 0.20,p=2 | 1.167 | 1.016 | 1.174 | **0.901** |
| annealed | 1.231 | 0.952 | 1.194 | **0.788** |

static 始终轻微同向。floor/annealed 不是平滑恶化，而是在不同 checkpoint 间发生非单调
翻号；这说明问题是训练路径依赖的非平稳几何，不是一个固定 basis 权重过大。

cross-split 结果复现主翻转：

| path | 2k aggregate / cross-split | 5k aggregate / cross-split |
|---|---:|---:|
| static | 1.024 / 1.029 | 1.028 / 1.032 |
| floor | 1.174 / 1.229 | 0.901 / 0.879 |
| annealed | 1.194 / 1.245 | 0.788 / 0.760 |

16 个配对 batch 的 bootstrap：

- 2k `static-floor` 均值差 `-0.0449`，95% CI `[-0.0517,-0.0387]`；当时 floor 更协同。
- 5k `static-floor` 均值差 `+0.0449`，95% CI `[0.0359,0.0545]`。
- 5k `static-annealed` 均值差 `+0.0837`，95% CI `[0.0745,0.0934]`。

5k、`t=0.1` 的 static-floor gap 在 last block 是 output head 的 `4.31x`；
static-annealed 为 `4.27x`。因此主要变化发生在共享表征，不只是输出投影对正交分量的
机械几何。

## 3. 为什么还不能把梯度翻转叫作 FID 原因

2k 到 5k，floor 的 teacher semantic loss 在多个时间点下降比例并不比 static 小。例如：

- `t=0.1`：static 改善 `8.30%`，floor 改善 `9.33%`；
- `t=0.3`：static 改善 `13.37%`，floor 改善 `14.53%`。

但同期 static FID 改善 `16.95%`，floor 只有 `3.42%`。因此局部梯度冲突是真实现象，
却没有阻止 teacher objective 继续下降；它不能单独解释 rollout generation gap。

这再次否定“找一个更好的局部训练指标就能预测生成”的简单叙事。

## 4. Generated-latent closure 外推通过

用相同 64 个新 noise/label 对三条 5k EMA 路径做 50-step rollout，再使用 frozen official
RAE encoder/ViT-XL decoder 测量：

| source | cycle residual | decoder local sensitivity | FID 1k screen |
|---|---:|---:|---:|
| clean test | 0.3988 | 1.3244 | - |
| **static** | **1.2148** | **1.8615** | **229.67** |
| annealed | 1.3228 | 1.9094 | 267.85 |
| floor 0.20,p=2 | 1.3000 | 1.9114 | 267.03 |

四项预注册预测全部通过：

- floor/annealed cycle ratio `0.983`；local sensitivity ratio `1.001`。
- floor 在 static→annealed 区间的位置分别为 `0.789/1.043`；FID 位置为 `0.978`。
- floor cycle 虽比 annealed 小 `0.0226`，配对 95% CI `[-0.0261,-0.0195]`，但仍比
  static 高 `0.0799`，CI `[0.0712,0.0935]`。
- floor 与 annealed 的 sensitivity 差异 CI 跨零；二者都显著高于 static。

旧四路径中发现的 closure 风险成功外推到新 floor candidate。floor 没有把生成 latent
带回 static 的 decoder 区域；它只是略微改善 cycle，而没有改变 sensitivity 或生成簇。

## 5. 当前机制边界

现在可以支持：

1. annealed 的解析 endpoint 逆病态真实存在，floor 确实修好了它。
2. 固定 floor 没有稳定改善共享任务几何；低噪声梯度关系随训练非平稳翻转。
3. teacher semantic loss、局部梯度下降和 self-generated closure 是三个不同层面，不能互相
   替代。
4. 与生成质量最一致的仍是 self-generated latent 的 closure/sensitivity，而不是
   teacher loss 或解析条件数。

现在不能支持：

1. 梯度翻转单独造成 FID 分叉；它可能是学习状态变化的结果。
2. decoder 是唯一瓶颈；cycle 同时经过 encoder、clamp 和 decoder。
3. 该现象跨训练 seed、模型规模或其他 tokenizer 普遍成立。
4. 直接加入 LPL、cycle loss 或 gradient surgery 就会超过 static。

## 6. 研究决策

- 不再扫描固定 `floor/power/rank`；这条方法线关闭。
- 不把 gradient descent ratio 当 checkpoint/FID proxy；它随训练非单调。
- 若做下一项因果实验，最小候选是：从同一个 floor-2k checkpoint 分叉，一路继续 floor，
  一路切回 static path，只训练到 5k。它只回答“移除晚期 time-dependent task 是否恢复
  static 的后程学习”，不是最终方法。
- 该分叉必须同时恢复生成指标、低噪声梯度方向和 closure；只改善 teacher loss不算通过。
- 在投入该训练前，需要接受它大概率只能解释失败、未必超过从头 static，因此其论文价值
  主要是机制而非 SOTA。

## 产物

```text
~/data/eqvae/experiments/rae_path_gradient_interference/confirm_n128_seed20260725/
~/data/eqvae/experiments/rae_path_schedule_closure/step5000_n64_seed20260726/
```
