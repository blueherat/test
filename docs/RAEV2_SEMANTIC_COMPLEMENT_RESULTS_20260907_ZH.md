# 固定类别补充：1K结果与独立5K确认

2026-09-07。3%目标尚未达成，但当前获得了本轮最强的固定1K信号。实现遵循[冻结两臂协议](RAEV2_SEMANTIC_COMPLEMENT_20260907_ZH.md)，未更改.15幅度或官方[.1,1]活动区间。

| 方法 | paired 1K FID | 相对official改善 | 推理与解码GPU秒 |
|---|---:|---:|---:|
| original official100 | 38.486774 | 0 | 632.3 |
| historical interval | 38.335024 | +0.3943% | 636.3 |
| semantic_add .15 | 37.746546 | +1.9233% | 1259.2 |
| semantic_orthogonal .15 | 37.704792 | +2.0318% | 1263.0 |

两个候选大部分收益相近；orthogonal对additive仅约0.1106%的点估计优势，不能由此声称去重叠机制已被独立证实。主方向是冻结类别差对原生IG的补充，几何约束提供明确局部结构。全部1000图计入，样本/类别/噪声hash与现有控制一致。每图199次主模型前向（100 conditional+99 null），实测成本比1.9973。

选择保持不变的semantic_orthogonal进入独立seed202609072的5K。原official/piecewise同seed 5K复用，输入hash必须匹配 `b59864ce96fcfb63735061f00ecb903b07ad894ffb504a73918d7b704db86cc8`。没有重新选seed、图像、窗口或幅度。5K开始时未知候选分数；需要FID≤6.7412754234才达到原官方的3%门槛。

## 在候选5K分数前固定的成本对照

用1K测得的T_candidate=1262.9642567和T_official=632.3276813，确定

    H = ceil((100 T_candidate/T_official + 1)/2) = 101.

在同一个原生IG场上使用显式Heun，最后一步Euler，对应2H−1=201次模型调用，与候选199次接近。所有positive query time均高于原生t_eps=.05；直接增加Euler到约200步会激活这个floor，而101步Heun没有这一额外变化。最终仍保留原始Euler100作主要控制；不从更差的控制制造成功。

如果候选在独立5K超过原控制3%，顺序应用已经审阅的精确Heun补丁，先检查official8逐像素不变及Heun8数值/201调用计数，再跑同seed全部5K。实测时间与最终FID一起披露；结果完成后仍需最后核对，队列不会自动宣告goal完成。若质量本身未过门槛，不追加成本采样。

数据：[完整1K ledger](../experiments/results/raev2_guidance_20260907/screen_ledger.json)、[预先固定的成本计划](../experiments/results/raev2_guidance_20260907/semantic_cost_plan.json)。后续新增的独立Gram形式FID审计只校验指标计算，不构成新的质量重复。
