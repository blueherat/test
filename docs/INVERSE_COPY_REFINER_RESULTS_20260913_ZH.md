# 原图锚定的反演复制修复器：本轮结果

2026-09-13。**本候选没有改善生成 FID，不扩大到 5K。** 单向配对训练按事先约定的验证目标选回恒等映射；双向配对只学到很小的修正，四个 CFG/APG 后处理设置的 FID 均略差于原采样器。这个结果限定于本次构造，不是否定所有反演或不变性方法。

## 实际方法及其假设

冻结 S800、W400 同类别 conditional FM，时间 0 为噪声、1 为图像。对真实训练图 x，以 S 反演至 τ∈{.75,.5}，再由 W 返回图像端：

\[
C_\tau(x)=\Phi_W^{1\leftarrow\tau}\Phi_S^{\tau\leftarrow1}(x).
\]

以原图 x 为外部目标，训练 P(Cτ(x))≈x，同时训练 P(x)≈x。第二版增加等权的反向配对 Cτ⁻¹(x)→x；实际使用 Heun 近似对应 ODE，未把离散往返说成精确互逆。每张图的起始 latent 只固定一次，配对没有新高斯噪声。

500 张训练缓存原图，400 fit、100 source-disjoint holdout；每类分别 4/1 张。P 为 592,100 参数的类别条件残差 UNet，末层初始化为零，训练 1500 步，以 holdout 上 repair MSE + 2×identity MSE 选模，step 0 也参与。P 不接收原始目标、反演深度或配对方向。

这已经增加了“模型扰动邻域应回到真实图”的修复假设。原始零编辑指令本身要求保留输入，包括它已有的缺陷，并不直接推出修复坏图。因此本实验与用户动机有关，但不能将它当作原始多模态复制现象的等价实现。

部署是对既有 CFG/APG 的输出应用一次 Pβ(z)=z+β(P(z)−z)。**反演用于离线构造训练对，部署增加一个学习得到的后处理网络；它不是免训练的新 CFG 速度场。**

## 训练和生成结果

- 单向版本选择 step 0，恰好保持恒等。训练确实执行了；非零检查点没有赢过预定目标。
- 双向版本选择 step 1000。相对恒等，holdout repair MSE 降低 0.246197%，计入 identity 代价后总目标仅降低 0.142379%。原始正向 C 两档的平均 repair 反而变差 0.031680%，微小收益主要来自新增 C⁻¹。

以下是相同 1K 生成 bank、相同 5K ImageNet100 reference 的配对比较，越低越好。所有 2000 张原始 CFG/APG latent 重新解码后与已有像素逐位一致。

| 采样器 | 原始 FID | Pβ，β=.5 | Pβ，β=1 |
|---|---:|---:|---:|
| CFG extra=1.25，64 步 | 44.910314 | 44.917227 | 44.921496 |
| APG extra=2，64 步 | 43.464164 | 43.465360 | 43.471239 |

差值分别为 +.006912、+.011181、+.001197、+.007075，均没有 FID 改善。没有为这些微小变化给出统计显著性结论；sFID 微降也不能改写为已获稳健质量收益。这里复用筛选 bank，没有独立确认样本。

生成原采样器均为 224 次单分支等价调用/图。后处理不再查询 SiT，但新增一次 P；实测每 1K 的 P 前向约 .11–.25 秒，另行解码约 14.3–14.5 秒，原采样约 104–106 秒。第一臂包含较多启动开销，不能由这四次短计时推断稳定加速比。离线两组配对共 160,000 次分支图像评估，另有 2,560 次精度检查；两次训练各约 68 秒。这些成本不能记为零。

## 每轮输出与解释

两版 P 都实际执行了五轮递归，每轮用上一轮 latent 作为输入并单独保存 R0–R5。真实 heldout 与 CFG 生成图各 8 张，β=.5/1。单向恒等版本各轮完全不动。双向版本 β=1 的真实图 MSE 从 R1 的 9.98e−7 增至 R5 的 2.47e−5；CFG 图由 9.70e−7 增至 2.40e−5。前五轮对原误差约按轮数平方累积，未体现固定点收敛。

这里递归的是 **Pβ**，不是每轮再做 Cτ，也不是 RGB 解码后重新编码。它检验修复器是否持续改图；前一项[公共反演复制实验](FM_COMMON_INVERSE_COPY_RESULTS_20260913_ZH.md)才是每轮完整重新反演、重新生成，三个参考共完成 960 个输出。两种五轮实验不混为一谈。

- [训练与 FID、逐轮状态独立复核](research/identity_operator_20260913/inverse_copy_refiner_results_audit.md)
- [双向五轮输出及统计](/home/zhoushunyu/data/eqvae/experiments/inverse_copy_refiner_20260913/symmetric_evaluation/rounds/summary.json)
- [四个生成设置](/home/zhoushunyu/data/eqvae/experiments/inverse_copy_refiner_20260913/symmetric_evaluation/infer)
- [冻结训练与评估源码](../experiments/inverse_copy_refiner_20260913/README.md)
- [构造依据和单一逆映射捷径](research/identity_operator_20260913/inverse_copy_refiner_rationale.md)、[对称化推导及反例](research/identity_operator_20260913/symmetric_copy_refiner_audit.md)

本轮任务进程已完成。当前还没有超过充分调参 baseline 的新方法；下一步应回到“原图作为保留条件时，CFG 哪些响应必须不变”的明确问题，而不是因这个小修复器接近恒等就继续扩大训练或搜索。
