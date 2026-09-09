# FSG 状态与查询时钟：固定质量对照

在任何本实验 FID 产生前固定。源于公开实现的
[调度器审计](FSG_RELEASED_CLOCK_AUDIT_20260908_ZH.md)，不声称复现 SDXL DDIM。

对一次校准定义 z1=z+h G(z,t)，z2=z1−h W(z1,t+H)。三个算子：
long 为 h=H=5/40；short 为 h=H=1/40；asynchronous 为 h=1/40、H=5/40。
时间均为 SiT 的 noise→data 线性流时间。统一在步 0、5、15 校准 2、2、1 次，
随后执行原 guided Euler 步；共 40 个原生步、10 次额外 field evaluation。
另作 closed40 与 closed50，后者与三种校准方式匹配 field evaluation 数。

两种 family 分别完整执行五个条件：CFG1.5（沿用历史前3通道协议），以及
v800 EMA + depth4_v 50K EMA 的 IG，gamma 为 .25:.6,.5:.7,1:0。
IG reference 保持已有实现 R_eff=S−gamma(t)(S−W_depth)，并非原始 weak head。
其独立 weak 查询在现实现中运行全 backbone；所有组记录实际调用计数与时间，
不把内部头查询误称为零成本。CFG 每个 field 两次完整模型调用。

每组固定 1,000 张，seed202609411、continuous CUDA RNG、batch8、FP32/TF32，
相同噪声和标签序列。先8张技术检查，不据其图像调整公式。质量主读数为原 ADM
Inception FID，固定 ImageNet100 validation 5K reference；保留所有条件。
调用计数/输入 hash 必须在匹配条件中一致，long 复用原始算子并验证像素一致。
不训练、不搜索强度、窗口、事件次数或种子。1K 只筛查，不能当独立5K成功。
若候选在同 family 同预算下超过 closed50，再冻结独立5K确认及必要机制控制。

代数上 z2−z = h(G−W_current)−h(W_future−W_current)，把当前对比与未来有限残差
分开。该恒等式不保证 FID，不证明新颖性，也不等价于 DDIM 的实际 alpha 时钟。
当前检验先回答：过长状态往返是否掩盖了查询未来 reference 的潜在收益。

## 独立5K确认（在其采样前固定）

1K已经看到IG short/asynchronous为66.055254/64.615465，closed50为69.147854；
CFG short为55.530113，closed50为61.463529。冻结下一次确认时CFG asynchronous
尚在生成，不能据其结果筛掉该组。

seed202609412，新5K continuous bank，两个family均执行closed50、short、asynchronous
全部三组，共30K图像。所有参数、checkpoint、精度、batch、reference、代码公式不变。
主要比较asynchronous−short以隔离查询时间，以及各自对closed50的质量与实际代价。
不重复明显失败的IG long，也不把已有CFG long的再现当新方法。无需训练。
预计纯采样合计约0.5–0.6 GPU小时，实际计时另报；多卡只缩短墙钟时间。
即使复核为正，也只是一个机制线索：需要额外早期guidance控制、强求解器基线、
RAEv2迁移证据及新颖性审查，才能判断有无论文贡献。不得将FSG重实现更名为新方法。

## 完整1K结果与审计

| arm | CFG FID | IG FID |
|---|---:|---:|
| closed40 | 62.051079 | 69.223805 |
| closed50 | 61.463529 | 69.147854 |
| long | 57.316520 | 122.303940 |
| short | 55.530113 | 66.055254 |
| asynchronous | 53.403568 | 64.615465 |

异步相对short在两个family上均有额外下降；不能由此直接声称其原因或迁移性。
全部十组输入noise/label hash相同，冻结的三份采样源文件hash相同。
closed50/三种往返的每样本完整模型前向数为CFG100、IG50；closed40为80、40。
全部十组采样计时合计591.799秒（约0.164 GPU小时，包含解码、采样结果落盘，
不包含加载和ADM FID）。完整CSV为
`experiments/results/terminal_defect_20260908/fsg_clock_quality_screen.csv`。

## 阅读和新颖性边界

本轮已逐项读完FSG附录C的假设、Lemma 1及主定理证明（式32–59）。
它把校准收敛与跨区间漂移分开，但不提供上述异步Euler算子的质量定理。
另外，仓库已有[ICG/TSG完整阅读](RAEV2_GUIDANCE_READING_INDEPENDENT_CONDITION_20260906_ZH.md)；
[TSG](https://arxiv.org/abs/2407.02687)已经用时间输入扰动形成guidance，
[Time-Shift Sampler](https://arxiv.org/abs/2305.15583)已经讨论状态与更合适查询时间的匹配。
本轮重新检索了这些原始来源，Time-Shift仅重读摘要；不能把“查询另一时刻”本身当新颖性。
尚待区分：额外局部guidance、求解器误差、真正的未来reference残差与参数化效应。

## 独立5K全部完成

| arm | CFG FID | IG FID |
|---|---:|---:|
| closed50 | 34.627031 | 41.533513 |
| short | 28.960328 | 39.783258 |
| asynchronous | 28.205314 | 38.395560 |

async相对short降低2.6071%/3.4881%，相对closed50降低18.5454%/7.5552%。
这些都是固定bank点估计。不能跨bank与历史PFR37.6459直接排名，也尚未超过
本仓库最强PFR或更强求解器的同bank对照。

全部六组通过5000像素形状/类型、独立seed、跨组noise/label完全配对、冻结源文件一致、
同预算模型调用计数检查。用float64特征均值/协方差与对称PSD特征分解独立重算FID，
与原ADM值最大偏差小于3e-6；该核对不重复Inception特征提取。
完整CSV见`experiments/results/terminal_defect_20260908/fsg_clock_quality_confirmation5k.csv`。
六组采样计时合计0.51418 GPU小时，不含模型加载和FID。多卡计时非严格硬件测速，
公平性主要由相同实际模型前向数保证。

[机制对照](FSG_CLOCK_MECHANISM_CONTROLS_20260908_ZH.md)已经显示1K time_only与
asynchronous相近，并识别IG参考gamma跨时间切换的混杂。当前支持质量线索，
不支持“前向状态校准必需”、新的固定点理论或RAEv2迁移成功。
