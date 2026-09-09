# 时钟对照后的机制反证

冻结时：独立5K六组仍在执行，尚无任何5K FID。本轮不修改已有采样源文件。

1K中asynchronous同时超过short和closed50，但这尚未排除更简单的解释：
额外早期guidance，或者PFR式不移动状态的未来reference查询。
固定增加两个对照，均仅使用原来五次校准事件，每次两次field evaluation：

- gap：z←z+h[G(z,t)−W(z,t)]。
- time_only：z←z+h[G(z,t)−W(z,t+H)]。

h=1/40，H=5/40；所有event、CFG/IG定义和原生40步保持原样。
每次校准后仍执行原来的guided Euler步。gap是明确的当前状态对比校准，
并不声称它与任意有限步长下调大gamma的所有实现逐像素相同。
time_only移除了asynchronous的前向状态预测，保留相同未来查询时间。

两个family均执行两个固定1K对照，seed202609411、continuous RNG、batch8、FP32/TF32，
配对原1K筛查的所有噪声与类别；固定同一ADM reference。先8图技术检查及调用审计，
不按输出调强度或窗口。它们是事后提出但在自身结果前冻结的机制对照，不能说成
最初1K预注册的一部分；也不能把同一探索bank的比较说成独立质量确认。

若time_only已解释收益，就放弃“前向状态校准必需”的叙事；若gap已解释收益，
就放弃“未来reference提供额外关键作用”的叙事。差异还须在独立数据上复核。

## 可以严格说明的局部关系

设 A(z)=z+hG(z,t)−hW(z+hG(z,t),t+H)，
Q(z)=z+h[G(z,t)−W(z,t+H)]，D(z)=z+h[G(z,t)−W(z,t)]。
若W在相关线段上关于状态为L-Lipschitz，则直接由差分得到

\[\|A(z)-Q(z)\|\le h^2 L\|G(z,t)\|.\]

另一方面，Q−D=−h[W(z,t+H)−W(z,t)]是精确恒等式；若W关于时间连续可微，

\[Q-D=-h\int_t^{t+H}\partial_s W(z,s)\,ds.\]

因此h缩小而H保持固定时，前向状态预测贡献为O(h²)，时间查询贡献可以为O(h)。
这解释了为何必须用time_only对照；它没有证明实际网络的常数足够小，
也没有保证积累整条轨迹后FID差异小。这只是标准光滑性推导，不作为新定理贡献。
在当前h=.025、H=.125有限步实验中，只能由实测判断二者的重要性。

## 已完成IG的1K对照

gap 66.505439，time_only 64.405892；原asynchronous 64.615465，short66.055254，
closed50 69.147854。8图和1K均与原对应bank的noise、label及每类调用计数完全一致。
time_only点估计略优于asynchronous，当前不支持前向状态校准必需。
此时CFG对照和IG独立5K仍运行中，不提前宣称该机制已被独立复核。

## 新识别的IG系数混杂

这里IG的W符号实际上代表R_eff(t)=S(t)−gamma(t)[S(t)−W_depth(t)]。
gamma在.25从.6变.7，在.5从.7变0。事件t=.125的远时查询到.25，
事件t=.375的远时查询到.5，恰好会切换参考场的系数。
因此不能将IG async−short全部归因于网络的时间响应。
上面的时间积分式不直接适用于跨越这些跳变的R_eff；分段光滑时还需加上跳跃项。

在迁移RAEv2或给出IG机制结论前，必须增加calibration中固定gamma为事件当前时刻
取值的对照，只改变网络查询时间，保持同一full/head配对与同调用预算。
CFG的固定scale没有该分段系数混杂，但仍需完成time_only/gap控制。

固定系数检查：原seed202609411/B8/FP32。先short8验证与原short逐像素一致，
再asynchronous8检查调用/输入一致，随后仅固定asynchronous1K。将每次reference
中的gamma固定为当前事件时间值，所有model查询时间与状态公式仍保持原样。
构造三个系数的field closure不运行模型；实际每个reference仍只有一次共享full/head
前向，调用计数由实际closure计数差额汇总。不改变已有5K源文件或输出。

## 四组机制对照全部完成

| family | gap | time_only | 原asynchronous |
|---|---:|---:|---:|
| CFG | 55.446518 | 53.466699 | 53.403568 |
| IG | 66.505439 | 64.405892 | 64.615465 |

四组的8图和1K输入/各类调用计数均精确匹配原bank。数据见
`experiments/results/terminal_defect_20260908/fsg_clock_mechanism_controls.csv`。
两个family都未显示前向状态移动带来明确的额外优势；不能将这种缺乏明显差异
升级为已证明统计等价。保留未来查询的time_only相对gap均有下降。

## 固定IG参考系数检查完成

short8与原short逐像素一致；asynchronous8/1K的noise、labels、实际各类模型调用
均与原实现完全一致。固定事件当前gamma的asynchronous1K FID64.482336，原64.615465。
因此当前证据不支持“未来系数跳变独自解释收益”；这仍不是独立5K机制确认。
记录在`experiments/results/terminal_defect_20260908/fsg_clock_frozen_reference.csv`。
本轮所有质量与控制进程已结束。下一步缺口为同bank原PFR/强求解器比较、
时间查询控制的独立确认和RAEv2迁移，尚无可宣称的新论文方法。
