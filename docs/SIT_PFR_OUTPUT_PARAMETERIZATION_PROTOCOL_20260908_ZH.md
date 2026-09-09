# PFR 同表示输出参数化可行性对照

RAE canonical查询两组正在运行，结果尚未知时冻结。复用SiT-v800、SiT-x800以及
各自冻结骨干后训练50K的depth4 velocity弱头，不训练。二者均SD-VAE/ImageNet100/
SiT-S/2。x800是原x-velocity-loss-floor0p05训练，不能称作仅推理输出记号重写。

四组1K：v/ordinary、v/PFR、x/ordinary、x/PFR。统一Euler100、FP32/TF32、B8、
CUDA continuous seed202609417，gamma=.35全程，PFR仅t<.5、h=min(1/32,.5−t)、
rho=1、原射线投影。gamma=.35来自历史x800原生IG设置，在新结果前固定，两主模型
共享该值；不宣称两者各自最优。同一噪声与标签，原ADM ImageNet100参考。

先每模型8张native采样器与wrapper ordinary逐像素复现；PFR8图检查实际future
prefix与共享full/head输出相等，再运行配对1K。PFR额外prefix成本单独记录；
本项不是等计算优越性或5K质量确认。预计四组采样约.1 GPU小时，加载/评价另计。

若x模型也有PFR收益，反对“clean输出本身阻止PFR”的强解释；若只有x失败，
仍可能是训练结果、弱头或gamma适配差异，不能直接归因于输出参数化。
该项补充机制证据，不作为新idea或新的参数搜索。

## 完成结果

| 主模型 | 普通IG 1K FID | PFR 1K FID | 普通IS | PFR IS |
|---|---:|---:|---:|---:|
| v800 | 71.169791 | 68.357811 | 32.107 | 34.072 |
| x800 | 71.667794 | 68.449145 | 33.071 | 33.056 |

四组同noise/label hash；各模型配对strong/head metadata一致，native8与wrapper
ordinary精确像素复现、实际future prefix/full一致性通过。完整1000样本每张
普通100次full，PFR100次full+50次prefix。独立feature-Gram重算最大差3.2e-5。
采样合计342.100秒（.0950 GPU小时），不含加载、smoke及评价。
CSV：`experiments/results/terminal_defect_20260908/sit_pfr_output_control.csv`；
审计代码`experiments/analyze_sit_pfr_output_control.py`。

这里两个模型都获得PFR FID改善，反对clean输出本身构成PFR障碍的强解释。
x模型IS没有随FID提升，进一步说明不能用IS单独解释PFR作用。
这个1K对照不是独立5K确认，不是等计算优越性，不证明参数化无影响；模型训练
结果、弱头、time loss权重等仍可交互。RAE直接query对照此时仍在采样。
