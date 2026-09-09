# 原PFR与求解器的同bank对照

本轮冻结时六组时钟5K已经完成；没有以下新对照的结果。
此前的显著普通50步相对收益不能代替最强已有PFR比较，也不能排除低步Euler误差。

固定新增三个5K：CFG ordinary DOPRI5、IG ordinary DOPRI5、IG canonical PFR DOPRI5。
保持原DOPRI5 atol1e-6/rtol1e-3，PFR h=1/32、gamma .25:.6,.5:.7,1:0，
正向射线投影原函数，未来时间截断.5，原affine_counterfactual_ratio_velocity函数。
full模型/head、seed202609412、continuous CUDA RNG、B8、FP32/TF32、decoder及ADM
reference均匹配已完成的5K。不将完整模型和depth4-only调用相加后冒称相同计算量。
记录两类调用与实际采样计时；这里先比较已有方法质量，非预先保证相同预算。

每组先8图技术检查；ordinary Euler50包装器须与原closed50逐像素相同。
PFR函数直接调用已提交的split/projection/affine helpers；不替换其浮点运算顺序。
全部三组5K完成，不因中间结果修改h、容差、gamma或seed。
若旧PFR在同bank已达到或超过时钟校准，承认本轮没有超出既有方法。

## 预算接近的PFR Euler对照

在上述三组5K均尚无FID时，补充固定PFR Euler42：SiT-S/2共有12个block，弱头
prefix4；42个完整场+前半段21个prefix4共42×12+21×4=588个block。
时钟校准50个完整场是600个block。此为Transformer block数近似匹配，
不是严格FLOPs或墙钟等价（额外embedding/head成本不同）。还需实际计数/计时核对。
gamma/h/所有模型和配对5K bank不变；42由预算决定，不由质量搜索决定。
先8图finite和调用检查，再固定5K；不根据DOPRI5结果取消或修改此组。

## 全部四组5K完成

| 方法 | FID | 完整模型前向/样本 | depth4前向/样本 | 采样秒 |
|---|---:|---:|---:|---:|
| CFG ordinary DOPRI5 | 33.372389 | 111.3088 | 0 | 413.076 |
| IG ordinary DOPRI5 | 41.278966 | 67.9712 | 0 | 304.613 |
| IG原PFR DOPRI5 | 37.786383 | 75.5840 | 34.1232 | 390.271 |
| IG原PFR Euler42 | 39.591289 | 42 | 21 | 230.364 |

同bank异步校准CFG/IG分别28.205314/38.395560。原PFR DOPRI5的质量仍高于
当前IG异步校准，因此没有超越原PFR最高质量。预算接近的PFR Euler42则落后，
异步校准相对其FID降低约3.02%，提示较低预算下存在优势；这不是全预算Pareto曲线。
IG ordinary DOPRI5只略优于Euler50，因此异步校准收益不能完全由更换求解器解释。
CFG亦保留对ordinary DOPRI5的优势，但没有RAEv2证据。

四组均通过同bank noise/label、checkpoint/head、precision与TF32配置一致检查，
原Euler50包装器的8图逐像素一致；PFR42实际调用为42完整+21prefix，与预算推导一致。
独立float64/对称PSD分解重算四组FID，与原值最大偏差小于3e-6。
原始像素、特征、冻结源码、命令与分开计数均保留；compact CSV在
`experiments/results/terminal_defect_20260908/fsg_pfr_matched_bank.csv`。
计时包含解码和落盘，不含加载与FID，跨GPU未做重复硬件benchmark。

未完成：时间查询机制的独立5K对照、RAEv2迁移、与TSG/原FSG的充分创新性区分。
这轮没有新的训练，也没有足以宣称ICLR论文完成的结果。
