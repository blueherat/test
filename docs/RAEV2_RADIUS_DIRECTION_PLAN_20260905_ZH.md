# RAEv2 clean-prediction 半径与方向：2026-09-05 机制筛查计划

状态（2026-09-05 独立确认完成）：**未达到约 5% 改善目标，也没有可重复的质量胜出。
半径保持与切向更新在首轮恶化；radial 首轮微小正信号在独立 bank 反向，未通过确认。**
本轮检验普通 Internal Guidance 的 clean-prediction 径向变化是否包含可删除的伤害。
它不是球面数据模型，也不是后验不确定性校准或 FID 改善定理。以下保留采样前声明的
条件，实际结果另列第 6–7 节。

## 1. 从 FSG 继承什么

[Towards a Golden Classifier-Free Guidance Path via Foresight Fixed Point Iterations](https://arxiv.org/html/2510.21512v1)
把状态校准与后续生成拆开，并将一致性区间、算子、强度和迭代预算变成可比较的设计轴。
值得继承的是先说明希望修正什么，再选择更新算子。

其 Theorem 1 控制的是预测差距，依赖正确固定点、收缩、光滑性，以及区间终点差距对当前
预测差距的下界；它没有证明差距下降会改善 FID。Table 9 的随机扰动平均收缩比也不能
排除特定 guidance 方向上的扩张。更长区间和更多迭代因此不能自动成为质量机制。

本地边界已经明确：

- [FSG/AG 复验](FORESIGHT_FIXED_POINT_CFG_AG_STUDY_ZH.md)中，一致性校准能降低局部
  gap，却从微小松弛强度开始恶化 AG 的 FID。
- [RAEv2 PFR 迁移](RAEV2_PFR_TRANSFER_RESULTS_ZH.md)中，raw PFR 的两个 FID-1K 正信号
  未通过正式 5K：ordinary IG 为 `7.034546`，PFR 为 `7.224213`，同时 IS 上升。

本轮不再追求把 strong/base gap 消到零，而是直接分离一次有效 guidance 中的不同作用。

## 2. 表征与坐标边界

官方配置选用 `dinov3mls-vit-l16[layers=11.13.15.17.19.21.23]`。
当前仓库 `external/RAEv2/src/encoders/vision_encoder.py` 的 registry 将 `dinov3mls`
映射到 `DINOv3MultiLayerSimpleAddEncoder`；该类位于第 300–344 行，其 forward 将
7 个归一化层取均值，再加最后一层 patch tokens 的空间均值广播。
第 690–738 行使用层求和的是另一个 `EUPEMultiLayerSimpleAddEncoder`，不是本轮配置。
这里已用静态 AST 逐类检查，避免从相邻代码片段误认实现。

此外，`external/RAEv2/src/stage1/rae.py` 对 encoder latent 使用训练统计量进行
mean/variance normalization。因此：

1. K7 token 不具有单层 LayerNorm 的固定半径；
2. 半径不能直接等同于语义置信度或后验不确定性；
3. 本轮明确使用 **stage-2 标准化 latent 坐标**，而非声称恢复某个原生球面几何。

保持当前 full prediction 半径只是局部更新约束。它不要求真实数据位于球面，也不保证
后续状态、其他时间的预测半径或终端分布保持不变。

## 3. 无新增强度参数的四个初始条件

对一个 token 的 1024 维通道向量，记 full clean prediction 为 \(F\)，base 为 \(B\)，
\(d=F-B\)，\(\gamma=\beta-1=0.78\)。所有分解和组合使用 fp32。

普通 IG 为

\[
G=F+\gamma d.
\]

当 \(\|F\|>0\) 时，定义

\[
d_\parallel=\frac{\langle F,d\rangle}{\|F\|^2}F,
\qquad d_\perp=d-d_\parallel.
\]

初始筛查固定为：

| 条件 | clean prediction | 所检验的干预 |
|---|---|---|
| `ordinary` | \(G\) | 同一新采样器和算术路径的基线 |
| `retracted`，token | \(\|F\|G/\|G\|\) | 保留 ordinary 的当前预测方向，恢复 full 的当前 token 半径 |
| `radial`，token | \(F+\gamma d_\parallel\) | 只保留 guidance 相对 full 的径向分量 |
| `tangent`，token | \(F+\gamma d_\perp\) | 去掉一阶径向分量，检验有限步范数保持是否必要 |

非退化时，retracted 是半径为 \(\|F\|\) 的球面上距离 \(G\) 最近的点；这是局部
欧氏几何事实，不是数据流形或生成质量保证。`radial` 的缩放系数若为负可能反向，不能
无条件称为保持 full 方向；应记录实际余弦与范数。

退化约定与实现一致：\(F=0\) 时退回 ordinary；\(F\ne0,G=0\) 时 retracted 取 \(F\)；
\(\beta=1\) 时各模式均精确返回 \(F\)。所谓精确范数保持指非退化实数代数，实际有
浮点舍入误差。

[Adaptive Projected Guidance](https://arxiv.org/html/2410.02416v1) 已提出相对 conditional
clean prediction 的平行/正交分解。因此本轮明确是 APG 相关的 RAEv2 IG 因果对照，
不声称投影概念新颖。纯切向更新 \(F+\gamma d_\perp\) 在启动采样、尚无首轮结果时
加入第四路同步诊断；它仍增加 \(\gamma^2\|d_\perp\|^2\) 的平方范数，和
retracted 的有限更新不同。

## 4. 冻结协议与判读

- 官方冻结 DINOv3-L K7 stage-2 EMA、base depth 8、clean-\(x\) prediction；沿用官方
  decoder 和 normalization statistics。
- shifted Euler 100 steps，CFG `1.0`，IG `beta=1.78`，仅在 `t in [0.1,1]` 启用；
  其余时间返回 full。保留官方 velocity conversion 与 denominator floor。
- 首 bank：`seed=202609051`，1000 样本，每类一张；单 GPU、batch size `8`。
  网络 bf16，guidance 组合 fp32；初始噪声和标签在条件间逐项配对并记录哈希。
- 初始四个条件为 `ordinary`、token `retracted`、token `radial`、token `tangent`，不增加 NFE；
  每条件均为 100000 次 sample-level full-model evaluations。
- 使用同一官方 FID evaluator/reference；同时报告 IS、半径比、径向能量比例、方向余弦
  与退化计数。历史不同 batch/算术路径的 ordinary 数值不替代本轮配对基线。
- 若出现 FID 正信号，固定候选与全部超参数，在独立 `seed=202609052` 的 1K bank
  配对确认；如修改条件，另起记录，不能称为本预声明候选的原样确认。

本轮目标是相对本轮 ordinary 将 FID 降低约 `5%`；未达到如实报告实际差值。1K 是筛查，
即使两个独立 bank 都为正，也不能忽略此前 PFR 的 1K/5K 排序反转。后续正式分布质量
判断仍需更大样本确认，本轮不作 SOTA 声明。

retracted 若优于 ordinary，只支持删除本轮局部径向增益有益；不能推出真实数据是球面
或 full 半径已校准。radial 若保留多数收益，则角度更新不是唯一有效成分。两者都失败
也是有信息的结果：说明直接拆除半径或方向不足以改善该官方基线，不继续包装理论。

## 5. 产物

- 实现：`experiments/raev2_radius_guidance.py`。
- 配对采样：`experiments/sample_raev2_radius_guidance.py`。
- 几何与退化验证：`tests/test_raev2_radius_guidance.py`。
- 每路保存 `request.json`、`summary.json`、`geometry.csv`、`samples.npz`；大数组留在
  仓库外。结果数值与执行验证由主实验记录补充。

## 6. 2026-09-05 首轮实际结果

来源：
`/home/zhoushunyu/data/eqvae/experiments/raev2_radius_direction_20260905/screen_seed202609051/comparison.json`。
本节逐项读取该 JSON；`pairing_validated=true`，四条件的 `paired` 均为 `true`，
均为 `seed=202609051` 的 1000 样本。共同 noise SHA-256 为
`1b5671d1759797211611b71c96d6ca82292a0579a0059adf05f0d54f579c3222`，label SHA-256 为
`702746827e553786bb026ac120cb58745fef3d3f554c33891809001cc37639f0`。

表中 `ΔFID = 条件 FID - ordinary FID`；相对改善为
`100 * (ordinary FID - 条件 FID) / ordinary FID`，正值代表改善。

| 条件 | FID-1K ↓ | ΔFID ↓ | 相对改善 | IS ↑ | 活跃时间平均局部半径比 |
|---|---:|---:|---:|---:|---:|
| ordinary | 38.735080 | 0.000000 | 0.000000% | 57.284344 | 1.061944 |
| radial | 38.636796 | -0.098283 | +0.253732% | 56.521525 | 1.039418 |
| retracted | 39.689400 | +0.954320 | -2.463710% | 54.337736 | 1.000000 |
| tangent | 39.449084 | +0.714005 | -1.843302% | 54.776878 | 1.017141 |

所有条件均未达到约 `5%` 的 FID 改善目标。retracted 和 tangent 同时恶化 FID 与 IS，
因此首轮不支持“删除径向变化能改善这套 RAEv2 官方 IG”的原假说。retracted 的实际
平均半径比为 `1.000000007310214`，说明所声明的局部范数约束已实现；这一数值精度没有
转化为质量收益。tangent 的半径比大于 1 与有限步二阶增益一致，但消除该增益的
retracted 在首轮反而更差。

radial 的 FID 只比 ordinary 低 `0.09828314488663636`，同时 IS 从
`57.28434410095215` 降至 `56.521524810791014`。这是较弱且指标存在分歧的筛查信号，
不能仅据单个 bank 宣称改进。按原定规则，固定 `beta=1.78`、token 分组、单 GPU B8
与全部算术/采样设置，在 `seed=202609052` 对 ordinary/radial 进行独立确认，结果见第 7 节。

本轮没有无引导 full 分支，因此不能从 radial 接近 ordinary 推断它相对 full 有收益，
也不能量化“保留了 ordinary 的多少质量收益”。各条件在第一次干预后具有不同的后续
状态；表中的局部半径比分别相对各自轨迹上当前 full prediction 计算，不表示它们沿途
共享同一个 full prediction，更不表示后验置信度一致。该闭环实验只识别所定义的更新
规则对最终指标的影响，不能把质量变化完全归因于一条固定轨迹上的孤立径向分量。

## 7. 2026-09-05 独立 bank 确认：radial 未复现

实际来源为同一实验根目录下 `confirm_seed202609052/comparison.json`，以及
`confirm_seed202609052/{ordinary,radial}/official_metrics.csv`。本次只确认 ordinary 和
radial 两路，各 1000 样本；没有重跑 retracted/tangent。比较器报告
`pairing_validated=true`，两路 `paired=true`，使用同一 `imagenet_256_fid_stats` 和
`nanogen-evals` evaluator commit `19dfb4c2705333eb8b97e454fb354d47d1fe135b`。

新 bank 的共同 noise SHA-256 为
`2660224afa97ee6e73e6e949db5c2041bacfe84aa9ef3541765f070dca040454`，与首 bank 不同；
label SHA-256 保持同一均衡类别序列。不同哈希验证了噪声 bank 内容不同，不单独作为
所有样本两两不重叠的证明。

| 条件 | FID-1K ↓ | ΔFID ↓ | 相对改善 | IS ↑ |
|---|---:|---:|---:|---:|
| ordinary | 38.773441 | 0.000000 | 0.000000% | 59.929219 |
| radial | 38.858200 | +0.084759 | -0.218601% | 57.462460 |

精确 FID 为 ordinary `38.773440920138796`、radial `38.858200089658794`。
radial 相对 ordinary 恶化 `0.08475916951999807`，与首 bank 改善
`0.09828314488663636` 的符号相反。两个 bank 的配对 FID 差平均为
`-0.006761987683319148`，几乎抵消；不能用这个微小均值替代独立确认的失败。
IS 在两个 bank 上都下降。

因此本轮结论是：**radial 的首轮微小 FID 正信号未复现，当前没有通过验证的 guidance
改进，约 5% 目标尚未实现。** 该结果不证明所有径向/切向设计无效，但不支持将本轮
参数与更新规则提升为质量改进方法，也不支持任何 SOTA 或相对无引导 full 的收益声明。
