# 分类器训练的性能优化与验证

2026-09-19。整理提交为 `60d73c6`；本报告对应随后独立提交的优化实现。
代码入口是 [`classifier_guidance/`](../../classifier_guidance/README.md)。历史代码、权重与 5K 结果保持不变。

## 完整训练迭代

RTX 4090，PyTorch 2.11.0+cu130。每项在独立进程测量，先排除一次预热，再取三次更新的中位时间。
同一模型的新旧实现使用相同真实 RGB、噪声、标签、头、判别器和 Adam 初始状态。
计时包括完整采样、图像解码、Inception、真假判别器更新、弱头反传和 Adam；
不包含一次性模型/数据加载、CUDA 图捕获、数据集 I/O 和 checkpoint 保存。
短训另外覆盖真实数据读取、多卡通信、EMA 与保存/恢复。

| 模型与单卡 batch | 原版秒/更新 | 优化秒/更新 | 提速 | 分配峰值 GiB，原→新 | 保留峰值 GiB，原→新 | 保留显存下降 |
|---|---:|---:|---:|---:|---:|---:|
| SiT，6 | 3.0293 | 1.6579 | **1.83×** | 10.053→4.984 | 12.117→6.619 | **45.4%** |
| JiT，4，速度档 | 6.6084 | 1.7509 | **3.77×** | 1.842→0.889 | 2.076→1.859 | 10.4% |
| JiT，4，省显存档 | 6.6084 | 2.1659 | **3.05×** | 1.842→0.888 | 2.076→1.246 | **40.0%** |
| RAEv2，1 | 11.8317 | 3.1729 | **3.73×** | 7.359→2.829 | 7.496→3.678 | **50.9%** |

保留峰值采用 `torch.cuda.max_memory_reserved()`，包含图内存池；只看 allocated 会夸大 CUDA 图省显存的效果。
CUDA 上下文等非 PyTorch 分配不包含在此值中，`nvidia-smi` 数字会更大。
计时是当前共享主机上的实测，不是所有 batch/硬件/训练时长的固定倍数。

JiT 工程夹具是已有 3K MLP；RAEv2 是现有 depth8 50K MLP。二者此处使用确定性初始化的 D 做性能对照，
不是已经训练出有生成质量收益的跨模型分类器。SiT 使用 RGB GAN v2 的 step1456 头与 D/Adam 状态。

## 改了什么

1. **CUDA 图重放向量场及其 VJP**。把重复的模型/反传算子捕获一次，动态复制状态、时间、标签和外推强度，避免每个时间步由 Python 重新发起大量小算子。上游 SiT/JiT 每次从 CPU 创建时间频率表，JiT 每次从 CPU 创建零注意力偏置，这些已移出重复传输路径。算子与既有计算精度保留。
2. **Heun 分开反传两次向量场**。同一时间步不同时保留两份大模型 tape。对 `p=x+h f(x)`、`x'=x+h/2(f(x)+f(p))` 使用完整链式法则；没有少跑时间步或截断强模型的输入 Jacobian。
3. **冻结反馈网络按块重算**。SiT 的 VAE 与 Inception 是显存大头。默认保留整批输入形状和 eval 模式，只重算需要的激活，不启用训练态 BatchNorm/dropout。
4. **提前保存冻结 BF16 线性/卷积权重**。JiT/RAEv2 原来已在 BF16 autocast 下执行这些算子；将这些冻结权重预先转换，省去重复 cast 及两份权重存储。归一化、embedding 和其他参数仍保持原 dtype。SiT FP32 网络不做这项转换。
5. **可选主干块重算**。JiT 的省显存档用多一些 GPU 计算换较小的图内存池。
6. **合并梯度通信**。每个头/D 一次 all-reduce，仍先全局平均、再裁剪、再 Adam；不把 head 包进会干扰内部重算的 DDP forward hooks。

可学习部分仍是弱头和真假二分类器，D 仍只区分真实 RGB 与实际强弱外推后的最终图像。
R1、非饱和生成器 loss、先更新 D 再更新 W、EMA 及完整离散轨迹梯度都保留。

## 正确性与数值误差

- 三种模型完整轨迹的最终输出与原版 **逐元素相同**；SiT 为 64 步 Heun，JiT 为 100 步 Euler，RAEv2 为原 shifted-grid 100 步 Euler。
- 完整训练对照里，D 分类 loss、R1 和 G loss 的记录值相同，D 参数更新逐元素相同。
- 弱头梯度并非宣称 bitwise 相同。完整训练对照的相对误差：SiT **0.0405%**，JiT 速度档 **0.2293%**、省显存档 **0.2482%**，RAEv2 **0.7557%**；余弦相似度均高于 **0.9999**。采样器单独的梯度误差更小；图重放/重算改变浮点梯度累加与底层执行顺序。
- CPU 的 Euler/Heun 对照包含非均匀时间步、关闭引导后的强模型后缀，并在 float64 下核对完整输入与参数梯度。
- 专用 GPU 测试覆盖：同一引擎多条尚未 backward 的轨迹交错、变标签、更新 head 后重放、替换参数存储时拒绝旧图。连同分块反馈链式法则，共 **4 项测试通过**。
- SiT 双卡连续训练 4 次，每次保存时各 rank 的 head、D、EMA 哈希一致；旧 checkpoint 和新版 checkpoint 都完成恢复并继续更新。
- JiT 在真实数据上完成 2 次 D/W 更新；RAEv2 在真实数据、batch2 下完成 2 次 D/W 更新，后者峰值保留显存约 **4.32 GiB**。

数值与短训检查说明实现可用，不代替较长对抗训练的稳定性或 5K 质量验证。本次没有启动新的系数搜索、长训练或 5K 采样。

## 未采用的默认方案

逐张解码/提取特征把 SiT 保留峰值进一步降至约 2.91 GiB，但相对旧整批卷积路径出现约 **5.03%** 弱头梯度差异；速度约 1.74 秒/更新，也不优于整批 checkpoint。保留为显式 `--feedback chunk` 选项，默认不使用。

只使用 CUDA 图而不提前存 BF16 权重时，JiT/RAEv2 的保留显存反而略涨；因此最终推荐组合使用上表中的配置，不把仅 allocated 降低误报为实际内存池缩小。

## 复现与证据

- [启动参数与配置](../../classifier_guidance/README.md)。
- [逐次计时、原始配置和数值检查 JSON](performance_20260919.json)，[汇总 CSV](performance_20260919.csv)。
- 外部完整原件：`$HOME/data/eqvae/projects/classifier_guidance/performance_20260919/`。各短训目录包含源码快照、启动命令、数据来源、参数/优化器/RNG checkpoint、退出状态。
- 汇总脚本：`python -m classifier_guidance.summarize`；它读取证据、验证梯度容差与短训完成标记，不启动训练。
- 性能测量命令例子：

```bash
python -m classifier_guidance.benchmark_train --gpu 1 --model sit_small \
  --batch 6 --mode baseline --output /path/to/new_baseline
python -m classifier_guidance.benchmark_train --gpu 1 --model sit_small \
  --batch 6 --mode optimized --chunk 0 --checkpoint-feedback \
  --output /path/to/new_optimized
```

JiT/RAEv2 优化测量增加 `--precast`；JiT 省显存档再加 `--checkpoint-backbone`。
正式 checkpoint 格式兼容旧评测器，5K 噪声、标签、求解器和 FID 参考统计未改。

实现依据包括 [PyTorch 2.11 CUDA graph 语义](https://docs.pytorch.org/docs/2.11/notes/cuda.html#cuda-graphs)和
[非重入激活 checkpoint](https://docs.pytorch.org/docs/2.11/checkpoint.html)；具体收益以上述本机实测为准。
