# 未来IG增量：同场校准的互补对照

在同场1K结果尚未产生时固定此实验。原校准A=q-hR'，q=z+hG，R'=2S'-G'，
精确分解 A-z=(q-hG'-z)+2h(G'-S')。已有同场组保留前项；本组只保留后项：

    z_new = z + 2h(G(q,t-H)-S(q,t-H)).

RAE代码采用噪声时间，G=-drift,S=-v_full，所以写成
z_new=z+.05*(qfull-qdrift)，h=.025。q仍按原始前向场计算，并非当前点gap对照。
这是标准代数分解下的因果干预，不是新的采样创新；不同完整轨迹的FID不相加。

其余完全遵循RAEV2_COMMON_FIELD_CALIBRATION_PROTOCOL_20260908_ZH.md：固定
seed202609413/B4/BF16/TF32、Euler100官方shift8、H=.125、三个事件2/2/1次、
110 Full调用/样本，labels0..999每类一张，原生IG普通采样保持不变。
先original/all与contrast/none各8张逐像素复现旧async与native100，再contrast/all
8张smoke，通过才执行1000张。GPU1，预计约 .25 GPU小时采样，不训练。
与已有none/all以及并行common构成四种保留分量的对照；不自动扩展5K或调强度。
官方FID后做独立特征Gram审计。一次探索bank不能代替独立质量确认。

## 完成结果

固定1K future_ig FID41.881665、IS52.92123；采样873.381秒（.24261 GPU小时），
每样本110次Full。前置原生与原校准逐像素重现、完整输入/模型/来源/hash检查
通过；独立FP64 Gram FID41.881693，误差2.80e-5。驱动与独立审计均exit0。

| 保留分量 | FID1K |
|---|---:|
| none：原生采样 | 38.264239 |
| common：同场往返 | 50.002931 |
| future_ig：仅未来IG增量 | 41.881665 |
| all：原两场校准 | 48.972310 |

两项单独保留均退化，删除同场往返未将另一项变成优于基线的方法。合用比
common好但比future_ig差；这是完整轨迹干预结果，不能将FID差相加或宣称
识别了可加的机制贡献比例。该组仍保留未来位移查询，不能混同于当前点gap。
同场与future_ig合计 .48745 GPU小时采样（加载/评估另计）。

本轮决策：不扩展这两组到5K，也不对它们做强度/事件搜索。原固定点时钟
实现的简单分量拆分未形成RAE方法；研究目标仍未达成，论文继续暂停。
合并CSV experiments/results/terminal_defect_20260908/raev2_calibration_components.csv。
