# PFR 迁移与条件留存：新主线的首个因果检查

用户明确建议转向PFR迁移机制及NeurIPS2025 FSG的条件latent化。分区gate首轮与原生
权重对照均未超过真实模型最强单头，不再增加其调参或训练。

## 问题

FSG首先定义能在无条件生成中保留目标语义的latent，再设计校准算子。
PFR是否也将目标类别更早地写入latent？若在SiT上成立、RAEv2上不成立，这可以成为
迁移差异的一个可反证解释；若两者均成立却只有SiT质量改善，语义注入便不足以解释收益。
不把IG的strong/base一致性等同于CFG的conditional/unconditional一致性。

## RAEv2首个冻结pilot

- 原官方100080 EMA、原shifted100 Euler及decoder，precision BF16。
- 128个样本，CPU seed202609401；从1000类随机排列取128类，每类一个相同配对噪声。
- 两个前缀：ordinary IG1.78；历史raw PFR h=1/32、rho=.05。无新强度搜索。
- 在网格首次到达t<=.8时冻结中间latent。每个latent分叉为两个后缀：
  full conditional与full unconditional，两者均关闭IG/PFR，使用相同剩余时间网格。
- unconditional使用官方label conditioning空类别1000，须与conditioning实现核对。
- 输出四路pixels、latent/输入hash、实际cut时间、NFE和完整来源。
- 固定已有ConvNeXt-Tiny ImageNet权重及官方预处理，报告目标类概率、top1/top5、
  条件与无条件后缀的配对变化。分类概率只是语义留存读数，不作为FID或质量替代。
- 先8样本实现smoke，再原样128；不据smoke改变cutoff、类别、算法或模型。

该pilot检测一个阶段的语义留存，不声称已匹配SiT/RAEv2的所有SNR、表示、模型成熟度
与weak-head训练差异，也不据单模型128样本归因整个迁移失败。

## 原文阅读边界

[FSG原文v1](https://arxiv.org/html/2510.21512v1)：本轮重读Introduction、§3.1–3.2、
算法1、附录A.5、B及C的假设与定理。关键区别是校准状态与执行去噪分开；附录B用
条件生成后的无条件逆推构造匹配latent。定理约束预测gap，不直接保证生成质量。
完整附录证明与官方实现仍需继续逐项核对，不将旧仓库摘要当作本轮原文阅读。

## PFR已知证据

SiT去重后的配对5K ordinary/PFR为40.7983/37.6459；RAEv2正式5K为
7.034546/7.224213。两边表示、预测参数化、弱头训练方式、基线成熟度均变化，
现有跨模型对比本身不能唯一识别原因。详见原始PFR机制审计及迁移报告。

## SiT 对应检查（在其结果产生前冻结）

使用历史v800 EMA与冻结backbone后训练50K的depth4_v EMA，原FP32/TF32执行与
DOPRI5 atol1e-6、rtol1e-3。cut时间取1减RAEv2实际网格cut，即0.2024169564；
这匹配线性桥系数，不声称匹配每个表示方向的真实SNR。
ordinary IG使用历史两段gamma .6/.7；PFR原样使用h=1/32的正向射线查询与beta修订。
cut后两路均切为full conditional/unconditional，不再IG/PFR；空类别为100。
128样本由100类尽量均衡生成，使用manifest明确记录的ImageNet原始标签映射做同一
ConvNeXt分类。类别集合与RAEv2首个pilot不同，因此首轮仅比较每个模型内部的干预差值，
不能用跨模型top1绝对值直接归因。8样本smoke后原样128，所有结果保留。

## 首轮结果及固定跨类复核

| 模型/前缀 | 条件后缀top1 | 无条件后缀top1 | 条件目标概率 | 无条件目标概率 |
|---|---:|---:|---:|---:|
| RAEv2 IG | .812500 | .812500 | .610387 | .605557 |
| RAEv2 PFR | .820313 | .820313 | .617796 | .614869 |
| SiT IG | .515625 | .015625 | .319572 | .022441 |
| SiT PFR | .531250 | .070313 | .333739 | .034290 |

SiT source checkpoint已核对cfg_dropout=.1；其无条件标签不是未训练的新类别。
这些是首轮探索读数，不是完整PFR的FID复验。尤其不能直接把两模型top1差归因为表示。

下一次固定检验：seed202609402，200samples（同一100类各两个），两模型同序原始
ImageNet标签；各模型自身在所有前缀/切换点使用同一noise。RAEv2请求cut=.95,.8,.5，
取原生网格实际值；SiT取其1减值以匹配桥系数。三点均执行，禁止按质量挑cut。
主比较为每模型的配对类别撤除损失，以及PFR相对IG对这个损失的改变；分类器固定。
这一步排除类别集合差异并刻画时间曲线，仍不等价于质量保证。

## 测量对象及现有工作边界

这里测的是冻结无条件后缀与固定语义读出的兼容性，不是Shannon互信息。若无条件
ODE flow在latent空间可逆，确定性可逆映射保留互信息；外部分类器识别失败并不能证明
latent完全没有类别信息。本文的“条件留存”只能按上述可执行干预定义解释。

配对分析还必须比较PFR对条件后缀与无条件后缀的共同影响。首轮目标概率的
[(PFR−IG)_uncond−(PFR−IG)_cond]在RAEv2为+.001903（类别聚类SE .001945），
SiT为−.002318（SE .015636）。因此首轮并未证明PFR特异地降低对后续条件的依赖。
两模型撤除损失差异很大，不等于该差异已经解释PFR的收益。

[Adaptive Guidance](https://arxiv.org/abs/2312.12487)已经讨论后期CFG评估冗余；
[Stage-wise Dynamics](https://proceedings.iclr.cc/paper_files/paper/2026/hash/f9e2800a251fa9107a008104f47c45d1-Abstract-Conference.html)
已经研究CFG不同阶段的语义/多样性动力学。这两篇本轮已读摘要，不能把时间相关的
条件作用或“后期关guidance”本身作为新颖性。后续方法需超出这些已知结论。

## 相同100类的复核进展

三段切换点的全部四路均已完成。12个输入bank逐数组验证：各模型所有cut/mode的
噪声完全一致，两模型所有bank的原始ImageNet标签同序一致。
中间时刻ordinary IG的条件/无条件top1分别为RAEv2 85%/84.5%、SiT49.5%/5%。
类别集合差异不能独自解释首轮的巨大反差。但200样本目标概率的erasure interaction
在中间时刻为RAEv2 +.000725（SE .000620）、SiT −.020688（SE .015062），
仍未证实PFR有特异的条件留存收益。完整配对结果在
`condition_retention_confirmation_early.csv`和`condition_retention_confirmation_middle.csv`。

晚期interaction为RAEv2 −.000054（SE .000613）、SiT −.005488（SE .010844）。
三点没有一致的PFR特异语义留存收益，停止将此作为PFR质量收益的直接解释。
不能由“未检出”证明机制绝对不存在；当前固定200样本、固定分类器的证据不支持它。
完整24路读数见`condition_retention_confirmation_raw.csv`，晚期配对分析见
`condition_retention_confirmation_late.csv`，均位于`experiments/results/terminal_defect_20260908/`。

| 切换点 | RAEv2 IG 条件/无条件top1 | SiT IG 条件/无条件top1 |
|---|---:|---:|
| early | .830 / .775 | .465 / .010 |
| middle | .850 / .845 | .495 / .050 |
| late | .850 / .845 | .560 / .440 |

这确认了两套生成系统的后续条件依赖不同，但并未隔离表示、模型成熟度、训练机制，
更不能把类别信心上升当作RAEv2 FID改善。无新的训练或参数搜索。

同时已完成[FSG公开调度器时间检查](FSG_RELEASED_CLOCK_AUDIT_20260908_ZH.md)，
具体软件组合中执行步长与名义前瞻查询时间不同。其与PFR的关系值得后续隔离检验，
但不能从源代码审计直接推导图像质量原因。
