# 时钟校准跨模型尺度复核

RAE逐项分解之后，使用同样32样本/B4/seed202609414、h=.025、H=.025或.125，
在SiT原生IG Euler40轨迹的0/5/15事件进行对应分解。SiT标签0–31是ImageNet100
内部类号，与RAE的ImageNet1K标签0–31不是相同语义类别；维度/精度/轨迹也不同。
这是机制量级比较，不是配对跨模型因果实验。未解码和计算质量指标。

复用了原SiT模型/后训练depth4 head加载与共享forward实现，gamma=.6/.7/0原调度。
416次B4前向、2.352秒，192条逐样本记录，所有精确向量分解核对通过。
脚本`experiments/audit_sit_clock_decomposition.py`，逐样本CSV位于
`experiments/results/terminal_defect_20260908/sit_clock_decomposition_per_sample.csv`。

H=.125下，每次校准total RMS除以当前位置latent RMS，逐样本平均：

| 生成时间 | SiT | RAEv2 |
|---|---:|---:|
| 0 | .687% | 1.319% |
| 约.125 | .714% | 1.357% |
| 约.375 | 1.452% | 1.742% |

若除以原生一步，则SiT是.269/.248/.390，RAE是9.370/1.966/.540。
这种巨大比值差异主要受不同时间网格的原生步长影响，不能直接称为场响应爆炸。
RAE初始short的clock/state抵消更强，但这也不是图像退化的因果证据。

下一步采用原同bank1K的early_only/late_only事件干预来定位失败，其他参数与
完整async一致；协议见RAEV2_CLOCK_EVENT_INTERVENTION_20260908_ZH.md。
既有OU/重定时校正失败历史仍有效，不能将这些诊断重复包装成未经验证的新方法。
