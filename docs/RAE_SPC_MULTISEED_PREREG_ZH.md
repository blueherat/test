# RAE 子空间路径课程五 seed 验证：事前协议

## 研究问题

单 seed crossover 显示，前期使用 rank-16 detail floor path、后期切回标准 static path，
可能同时改善梯度关系、decoder closure 和 5k 生成质量。本轮只回答：该优势能否跨独立训练
随机性复现，而不继续搜索 floor、rank、power 或切换时刻。

## 固定实验

- 模型：当前小型 `DiT-DH-S`，RAE-DINOv2 latent，fp32，关闭 TF32。
- 数据：同一份 160k ImageNet-1k frozen latent cache。
- seeds：`1201, 2309, 3413, 4517, 5623`。
- 每个 seed 使用独立模型初始化、cache permutation、time 和 noise。
- 同一 seed 内两组严格共享初始化、数据顺序、label、time、noise、优化器和 scheduler。
- 每组训练 5000 updates，global batch 16。
- 在 step 2000 两组同时把 EMA 重置为 online model，消除早期 EMA 历史污染。

两组定义：

| condition | step 0--1999 | step 2000--4999 |
|---|---|---|
| `static` | standard linear path | standard linear path |
| `spc` | floor=0.20, power=2, rank=16 | standard linear path |

切换在同一个训练进程内完成，不使用 step-2000 checkpoint 恢复。

## 审计条件

解释指标前必须满足：

1. 同 seed 两组首 batch 的 latent、label、time 和 noise hash 完全相同。
2. 两组 manifest 的 seed、cache order、batch、optimizer、scheduler 和 EMA reset 一致。
3. step-2000 checkpoint 中 EMA 与 online model 逐 tensor 相同。
4. 所有训练无 NaN/Inf，且恰有前 2000 个 floor updates、后 3000 个 static updates。

## 主要指标与判据

主要权重为 step-5000 online model；phase-reset EMA 作为并列稳健性检查。固定 50-step Euler、
sampling seed `20260718`、每模型 5000 张类别均衡样本。

SPC 进入长程验证必须同时满足：

1. 至少 4/5 seeds 的 FID 与 KID 都优于 paired static。
2. 五 seed 平均 paired FID 相对改善至少 5%。
3. seed-level paired FID 差值 bootstrap 95% CI 上界小于 0。
4. phase-reset EMA 的平均 FID/KID 方向不反转。
5. paired closure 的 cycle residual 或 decoder sensitivity 不得在 4/5 seeds 系统变差。

IS 只作辅助，不参与是否继续的决定。原 seed-3407 crossover 是先验发现集，不并入五 seed
置信区间。

## 停止规则

- 审计失败：只修实验控制，不解释质量结果。
- 主要判据失败：不扩大模型，不搜索更多路径超参数；将结果限定为单 seed 偶然性或弱早期效应。
- 主要判据通过：固定当前超参数，继续 10k/20k 持久性实验，再决定是否进行 50k 正式评估。
