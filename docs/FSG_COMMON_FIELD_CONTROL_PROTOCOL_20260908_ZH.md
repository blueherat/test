# 同场往返控制：区分guidance校准与数值修正

冻结时RAEv2直接迁移仍在运行，无任何该迁移FID。此前所有SiT时钟组与原PFR
对照已完成。本项针对剩余机制缺口，不调整原方法参数。

保持全部普通local guided步骤不变，仅将校准的inverse reference R替换为
与forward相同的guided field G。分别做short和asynchronous，两个family共四个1K。
相同seed202609411、continuous RNG、B8、FP32/TF32、event 0:5:2,5:5:2,15:5:1，
相同模型/heads/ADM reference。每组先8图技术检查，核对输入及总前向数与原相应组。
IG计数从weak-reference转入guided查询类别，总计算量不变；CFG每次guided仍两模型调用。
该控制没有移除整个采样器中的guidance，只移除了校准双腿之间的场身份差异。

必要性来自一个简单反例：G=R=a(t)z时，两腿若用精确ODE flow则往返为恒等映射。
但这里的离散算子为[(1+h a(t))(1−h a(t+H))]z，即便没有双场对比也会改变状态。
因此观测到校准收益不自动证明双场条件一致性是原因。

若common-field达到相近收益，则降低对条件/弱参考专属解释的信任；若明确落后，
只支持两腿使用不同场的作用，仍不证明latent信息论或原FSG完整理论。
本项1K在已用探索bank上，是在自身结果前冻结的事后机制对照，不是独立质量确认。
等待RAEv2首轮进程释放GPU后执行，不干扰其参数或重启其采样。

## 完成结果

| family | 原short | 同场short | 原async | 同场async | 原closed50 |
|---|---:|---:|---:|---:|---:|
| CFG | 55.5301 | 64.1549 | 53.4036 | 58.4403 | 61.4635 |
| IG | 66.0553 | 70.1707 | 64.6155 | 64.7713 | 69.1479 |

均为同一探索bank的1K FID，越低越好。四组输入hash、总调用匹配，像素尺寸、
有限features与完成标记通过。独立feature-Gram FID重算最大差3.4e-5。
原始汇总见`experiments/results/terminal_defect_20260908/fsg_common_field_controls.csv`，
复核脚本为`experiments/analyze_sit_fsg_common_field.py`。
新增四组采样共243.109秒（约0.0675 GPU小时，不含加载和FID）。

IG同场async与原async接近，现有证据不支持双场身份差异对该收益的必要性。
这不是统计等效性结论，也未去除普通采样步骤中的IG。CFG同场async明显落后于
原async，但仍优于closed50；两类guidance不能使用同一个专属机制解释。
同场short均退化、同场async改善，进一步要求区分数值往返偏差与条件校准。
这些结果削弱了将当前方法包装为latent条件信息创新的依据。
