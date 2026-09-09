# SiT 平滑时钟补偿：固定质量控制

在生成前固定。只运行现有 SiT v800 + depth4_v，先 native/ordinary 8 张像素
一致及两种方法 prefix/full 8 张一致，再各 1000 张正式采样。使用原生 Euler100、
gamma=.35、FP32/TF32、batch8、seed202609417 continuous；标签沿现有 sampler
均匀随机抽取，两组噪声及标签必须完全一致，不声称各类均衡。

主场 G=F+.35(F-W)。t<.5 时 g=sin²(2*pi*t)，h=g/32；其余为原生 G。

- smooth: G+1.35*(W(t,z)-W(t+h,z))。
- compensated: smooth-(1.35/32)*g'(t)*W(t,z)。

两组每张 100 Full + 50 prefix，无额外 OU 查询，无训练。预算采样约
0.06 GPU 小时，加载与 ADM FID 评估另计。1.35 是现有 PFR 的实际修订系数，
不是上一份解析例子中任选的 .35；没有强度搜索。g 与 g' 在窗口两端为零，
且 h 不越过 .5。这个平滑控制改变了既有硬窗口，必须成对比较。

机制依据见 PFR_CLOCK_COMPENSATOR_20260908_ZH.md：补偿项在 W=G 时取消
一阶纯时钟作用；W!=G 时没有终点不变或质量保证。仓库已有 terminal-adjoint
Lie bracket 分解，本次不将它称为新理论，也不重跑其 toy 身份。

两组最终还需与既有同输入原生 1K 比较，不能仅胜过另一失败候选就称改善。
保留原始图片、ADM features、调用计数、输入与代码哈希。完整 FID 后独立复核。
不按此 bank 搜索窗口/系数，不自动扩展 5K。论文写作保持暂停。
