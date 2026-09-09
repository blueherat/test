# JiT 完成后的低噪声关闭 guidance 对照

用户授权：在 JiT 完成之后补充低噪区间不加 guidance 的对照。
当前按上一轮讨论的 RAEv2 capacity lifting 执行；JiT 保持优先。

状态：待执行，尚未启动采样。必须等 JiT 的训练和 IG/PFR/OU 方法采样、
结果评估完成后再启动。preparation_status 的 readouts_ready 仅表示读出头
训练完成，不是此后续实验的启动信号。不得抢占 JiT GPU。

首轮固定四组：alpha=.65/.78，分别在 t<=.2 或 t<=.5 关闭 lifting。
RAEv2 时间方向为 t=1 噪声、t=0 数据；关闭后只用原有 Strong 条件场
逐步 Euler 推进，不额外加入 native IG。保留类别条件；关闭的是强弱差异引导。
高噪声区间保持原 lifting 运算与强度，不用 fade、不重新调 alpha。

沿用原 100 步 shift8 网格；按实际节点 t 判断开关，并记录实际切换点。
将跨越 cutoff 的 lifting block 截断，使低噪尾段不再触发新的 lifting。
最后一个活动 block 可推进到首个关闭节点，与通常按步起点应用 guidance
的离散定义一致。其余活动 block 仍最多四个细步。
尾部每步仅一次 Full 调用，不计算 Base 逆流或 Full 目标轨迹。

每组四卡共同分片 1K，四组顺序执行；复用已有 .65/.78 constant 样本、
FID 和原生 IG baseline，不重新生成这些基线。相同 seed202609413、B4、
labels0..999、checkpoint、解码器、精度和评估器；保存配对噪声校验。
使用新的 sampler/helper 和独立结果目录，保留已有冻结源码及哈希。

比较每个 alpha 下 constant、cutoff .2、cutoff .5 的 FID、IS、
Full/Base 调用量、实际采样耗时及配对图片。主要问题是尾部引导是否
改善质量，以及移除它能否保持质量并降低开销。1K 接近不能直接证明
统计等效或跨 seed 稳健性；有明确信号后再决定独立大样本验证。
这是对照协议，不是论文，也不预设低噪关闭会改善结果。
