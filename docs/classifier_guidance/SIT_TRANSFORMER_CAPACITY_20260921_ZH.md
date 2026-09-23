# SiT 原生 Transformer 弱头：重新训练与容量对照

2026-09-22 更新：四组 50K 训练及全部 5K 评测已于北京时间 2026-09-21 23:02 完成。共 107 组不同的 5K 采样，合计 535,000 张：102 个非零外推点、1 个共享无引导基线、4 个弱头独立生成点。无 1K 预选，无独立验证轮次。

本轮 SiT 原生 Transformer 头改善了弱头自身预测、独立生成和最佳外推质量。前半程引导下，fresh 浅 MLP 最好 FID-5K 36.8451，1-block 为 30.9767，2-block 为 29.0626；最后一项在扫描上界 a=1.95，尚未包围其最优系数。

此前 SiT 的容量实验只加深/加宽逐 token MLP，没有新增 token 之间的注意力。本轮说明不能把那次结果解释为“SiT 不受益于更强弱头”；仍未单独拆分注意力、容量和预训练初始化的因果贡献。

## 完成结果

全部为 50K EMA、相同 5K 噪声/类别、同一 ADM 参考统计，FID 越低越好。无引导强模型 FID-5K 为 **59.447250**。

| 本轮重训头 | 验证 velocity MSE | 弱头独立生成 FID-5K | 前半程最佳 FID-5K | 额外 a | 全程最佳 FID-5K | 额外 a |
|---|---:|---:|---:|---:|---:|---:|
| shallow MLP | 0.868950 | 197.655056 | 36.845130 | 1.05 | 39.475235 | 0.45 |
| linear 原生输出头 | 0.920499 | 244.944462 | 40.459124 | 0.80 | 44.728959 | 0.25 |
| 1-block | 0.837071 | 164.256783 | 30.976716 | 1.75 | 33.263440 | 0.75 |
| 2-block | 0.820907 | 141.205022 | **29.062644** | **1.95（边界）** | **29.852349** | 1.10 |

预测误差来自同 5,000 张验证图像 × 10 个分层时间，共 50K 状态；强模型 MSE 为 0.782598。独立生成指仅使用前四层和对应弱头作为向量场，不进行强弱外推。

在配对重训 shallow 基础上，1-block 和 2-block 最佳前半程 FID 分别降低 5.8684 和 7.7825。全程引导也有相同的架构改善趋势，但四组各自最好的前半程引导均优于其全程引导。

系数需要按头重新搜索。例如统一 a=1.0、前半程引导时，shallow/linear/1-block/2-block 的 FID 分别为 36.8758/41.4454/34.3459/35.2952，2-block 此时反而弱于 1-block；按本轮预算分别选参后 2-block 更好。不能用一个旧头的最佳系数替所有架构判断能力。

2-block 前半程在 a=1.80/1.85/1.90/1.95 的 FID 为 29.3914/29.2369/29.1483/29.0626，仍在下降，故只称“已扫描范围最好”，没有继续扩大范围。1-block 全程 a=.75/.80 的 FID 为 33.2634/33.2673，属于近乎平坦的最佳区域，不据这点差异断言 .75 更稳健。

所有组使用普通真实数据 FM loss，尚未对这些 Transformer 弱头进行 GAN 后训练。不同架构为等更新次数、等全局 batch，对应计算成本不同；本轮仍是单训练种子和同一 5K 集合选参，无额外独立验证。

完整结果：[JSON](/home/zhoushunyu/data/eqvae/projects/classifier_guidance/sit_transformer_capacity_20260921/results.json)、[逐点 CSV](/home/zhoushunyu/data/eqvae/projects/classifier_guidance/sit_transformer_capacity_20260921/results.csv)、[完成核对](/home/zhoushunyu/data/eqvae/projects/classifier_guidance/sit_transformer_capacity_20260921/completion_audit.json)。

2026-09-22 核对通过：107 组均正常退出，每个 NPZ 数组头为 5000×256×256×3、uint8；625 批噪声哈希和类别列表逐点一致，100 类各 50 张；所有点使用相同参考统计和对应 50K checkpoint；模型/头冻结检查、CUDA 图端点一致性、8 点细扫计划均通过。本次核对重算了四个训练 checkpoint 的哈希；样本完整字节哈希已在各点评分前校验，本次未重复读取全部图像字节。

## 架构与初始化

### EMA 与后期收敛复核（2026-09-22）

确认所有本轮 5K 系数点、W-only 及预测误差审计加载的是 `checkpoint_050000.pt` 中的 `ema`，衰减率 .9999。旧 moderate/large 的评测同样使用 EMA，不存在一边 raw、一边 EMA 的协议差异。强模型自身也使用原 800K EMA。

追加只读评估六种头的 40K/45K/50K，共 18 个 checkpoint、36 份 raw/EMA 权重，使用原训练验证协议的同一组 2,048 状态与 BF16 autocast。18 个 EMA 验证值均在 1e-7 内复现，权重和强模型保持不变。该协议与前面 50K 分层状态、部署 FP32 误差表不同，数值不能直接横向混合。

| 头 | EMA 40K | EMA 45K | EMA 50K | raw 50K |
|---|---:|---:|---:|---:|
| fresh shallow | 0.877008 | 0.875854 | 0.875202 | 0.874998 |
| linear | 0.922117 | 0.919307 | 0.918187 | 0.917465 |
| moderate MLP | 0.875055 | 0.873336 | 0.872364 | 0.871835 |
| large MLP | 0.867675 | 0.865328 | 0.863855 | 0.862865 |
| 1-block | 0.849615 | 0.848562 | 0.847784 | 0.847063 |
| 2-block | 0.835515 | 0.834705 | 0.834060 | 0.833460 |

1-block/2-block 最后 10K 的 EMA 验证误差分别下降 0.2155%/0.1741%，原始权重误差也缓慢下降。结论是进入平台附近的缓慢改善阶段，不能声称完全收敛或所有结构均达到各自最优。旧 large MLP 后期仍下降约 0.4403%，因此也不能把它的现有结果解释为容量的理论上限。

EMA=.9999 的参数权重半衰期约 6,931 步，几何平均年龄约 9,999 步；EMA 参数包含初始化的权重在 50K 时约 .006736。这些是参数平均的性质，不是 loss 的线性分解。早期 EMA loss 下降含有追赶原始权重的影响；最终 raw 与 EMA 的 MSE 差只有 .0002–.0010，没有严重滞后的迹象。尚未采 raw 权重的 5K，因此不能据 MSE 稍低断言 raw 的 FID 更好，也不能从 loss 平台断言 FID 完全收敛。

原始记录：[同输入 raw/EMA 复核](/home/zhoushunyu/data/eqvae/projects/classifier_guidance/sit_transformer_capacity_20260921/convergence_audit_20260922/result.json)；代码：[audit_capacity_convergence.py](../../classifier_guidance/audit_capacity_convergence.py)。

### 为什么 Transformer 与之前增大的 MLP 不同

旧 MLP 接收的 depth4 特征已经有来自前四个 Transformer block 的上下文，因此并非完全看不到其他位置。但 MLP 头内部对每个 token 独立映射，增加逐 token 非线性层不会新增 token 之间的交互。本轮 SiTBlock 新增自注意力、宽度 4 倍的 FFN，以及 adaLN 的平移/缩放/门控，且从强模型末端复制预训练参数。原 MLP 权重随机初始化，只共享固定输入归一化统计。

同时参数预算也变大：moderate/large 为 .452M/1.790M，1-block/2-block 为 2.968M/5.628M。本轮是架构、初始化及参数容量共同改变后的效果，不是单独注意力的因果消融；等 50K 更新也不等于等训练计算量。可区分这些因素的后续控制包括参数量接近的 MLP、随机初始化的原生 block，以及保留原生 FFN/adaLN 但移除 attention 的结构，尚未执行。

测得旧大 MLP 已改善弱头自身的 MSE 和独立生成 FID，但未改善引导 FID。理论上也不存在“W 的 MSE 越低，外推必然越好”：若在固定输入定义 eS=S-v*、eW=W-v*，则外推误差为 `(1+a)*eS-a*eW`。它的平方误差还取决于两种误差的内积，生成质量又取决于沿实际采样轨迹的误差积累，均不能仅由 W 的一个平均 MSE 决定。极端情况下 W=v* 而 S 仍有误差，正向外推会把强模型误差放大为 `(1+a)*eS`。

已有 50K 状态误差数组还允许直接计算固定真实加噪输入上的向量场 MSE：`E||S+a(S-W)-v*||² = ES+2aA+a²B`，其中 `A=(ES+B-EW)/2`、`B=E||S-W||²`。本轮各头在这些固定真实插值点上的 MSE 最优常数 a 略为负，而采样端点 FID 的最优 a 为正；这再次表明真实插值点预测 MSE 与引导后的端点 FID 是不同目标。这个诊断没有在采样轨迹上重算向量场，不能拿该负系数代替已测的 FID 搜索结果。详见 [残差诊断](/home/zhoushunyu/data/eqvae/projects/classifier_guidance/sit_transformer_capacity_20260921/convergence_audit_20260922/residual_geometry.json)。

### 实际结构

SiT 本身是 Transformer，源码模块为 `SiTBlock`，包含自注意力、MLP、时间/类别条件 adaLN 与残差连接。此处不引入 U-Net。

强模型固定为 ImageNet-100 的 SiT-S/2、800K EMA；12 层、宽 384，读取第 4 层输出。所有强模型权重冻结，只更新读出头。

| 组别 | 第 4 层之后的结构 | 初始化 | 头参数量 |
|---|---|---|---:|
| shallow | 原浅 MLP | 重新随机初始化，共享原固定归一化统计 | 304,528 |
| linear | 原生 adaLN FinalLayer，无新增 block | 复制强模型 final layer | 308,000 |
| block1 | 1 个原生 SiTBlock + FinalLayer | 复制强模型第 12 层及 final layer | 2,967,968 |
| block2 | 2 个原生 SiTBlock + FinalLayer | 复制强模型第 11、12 层及 final layer | 5,627,936 |

`linear` 是含条件调制的原生输出头，并非只有一个无条件 Linear。三个原生头使用相同的复制初始化原则；shallow 是用于配对旧 MLP 方法的控制，不能把 shallow 与 block 的差异全归因于参数量。

源 checkpoint 的 final projection 保留额外 sigma 通道，官方 SiT 前向在 unpatchify 后丢弃这些通道。新头保留完整输出层并按相同布局只取速度通道，已核对 FP32/BF16 输出完全一致；上表原生头参数量包含 6,160 个不影响速度输出的 projection 参数。复制模块不共享强模型参数存储，源模型的监测 hooks 不会复制到新头。

## 训练协议

- 四组各 50K 更新；预检的前 100 步计入总数，从完整 checkpoint 精确续训至 50K。
- 全部 126,689 张真实训练图像的 SD-VAE posterior moments，每批重新采 posterior、噪声与时间；不使用强模型合成样本。
- depth4 特征冻结，训练目标为普通 velocity flow-matching MSE：`E ||W(t*x+(1-t)*eps,t,c) - (x-eps)||²`，`t~Uniform[0,1]`。
- 单卡一组，global batch 256，AdamW lr 1e-4、betas (.9,.999)、weight decay 0，EMA .9999，BF16 autocast / FP32 权重。
- 四组相同数据流、随机种子、批量和更新预算；每 5K 保存 checkpoint，并用同一组 2,048 个验证状态记录 EMA loss。
- 不训练 GAN，不引入外推后的 loss。本轮首先检验弱头架构对普通 diffusion/FM 训练的影响。

旧浅 MLP 使用过四卡训练，本轮重新单卡配对训练 shallow，避免把那次不同 RNG 分配的历史训练直接当成唯一控制。仍然只有一个训练种子、固定预算；不能据此认定所有架构都达到各自最优收敛。

## 评测与系数扫描

统一使用旧 SiT 的 5K 噪声和标签（100 类，每类 50 张）、64 步 Heun、FP32/TF32 生成和相同 ADM FID 参考统计。每个系数均采 5K，无 1K 预选，无额外独立 5K 验证。

同时比较两种时间窗，区分新增 block 和引导时段的影响：

- `legacy`：`S + a*f(t)*(S-W)`，左端时间 `<.25` 时 f=6/7，`.25≤t<.5` 时 f=1，后半程 f=0。
- `full`：全程 `S + a*(S-W)`。

这里 **a 为额外系数**，全程情况下论文总系数 `w=1+a`，不能把 JiT 的 `w=1.5` 当成这里的 `a=1.5`。

每个头、每个时间窗分别从 a=1.0/1.2/1.4/1.6/1.8 开始粗扫。若较低端或无引导基线最佳，按 0.2 向下扩展至包围最优区间或 a=0。随后取粗扫最佳附近**恰好 8 个相邻系数，间距 0.05**；重合点复用，细扫上限 1.95，不运行 a=2.0。所有组共享同一 a=0 基线。发生数值发散的点明确记录 invalid，不伪装成完整 5K 结果。

另外测量每个弱头独立生成的 5K FID，以及同 5,000 张验证图像 × 10 个分层时间的 50K 状态预测误差，报告全程/前半/后半 MSE。选参使用的 FID-5K 不是独立测试集上的最终估计。

## 正确性与短程性能

已经验证：原生类与复制权重一致、参数独立、强模型冻结、复制 hooks 清理、速度通道布局准确、所有参数组在 100 步内更新，以及四组前四批图像/噪声/时间一致。额外 token 扰动只会通过 Transformer 头影响其他 token，线性头无此额外交互。

新采样器通过 CUDA 图加速前向：legacy 对照历史采样器，full 和 weak-only 对照直接求值，端点要求逐位一致。block1 的连续 100→102 与断点 100→101→102 用于检查 head、EMA、Adam、数据/运行 RNG 的精确恢复。

100 步短程预检测量如下，尚不是完整训练或生成质量结果：

| 组别 | 秒/步 | 峰值 allocated GiB | 峰值 reserved GiB |
|---|---:|---:|---:|
| shallow | 0.0410 | 1.568 | 1.816 |
| linear | 0.0295 | 1.571 | 1.811 |
| block1 | 0.0441 | 2.746 | 3.121 |
| block2 | 0.0582 | 4.008 | 4.270 |

## 运行入口

结果根目录：`/home/zhoushunyu/data/eqvae/projects/classifier_guidance/sit_transformer_capacity_20260921`。

代码：[头结构](../../classifier_guidance/sit_transformer_heads.py)、[训练](../../classifier_guidance/train_capacity.py)、[评测](../../classifier_guidance/evaluate_sit_transformer.py)、[后台队列](../../classifier_guidance/sit_transformer_pipeline.py)。

tmux 会话 `sit_transformer_0921` 已正常完成退出，本轮释放了 GPU。四组正式训练（从 step100 续至 50K，含加载/保存/验证）分别耗时 shallow 25.49、linear 24.97、1-block 35.30、2-block 48.74 分钟；正式训练峰值 allocated 显存分别 1.57/1.57/2.77/4.05 GiB。队列 2026-09-21 17:25 启动，23:02 全部完成。

`status.json`、各组 `progress.json` 和 `results.json` 为运行状态来源；此队列已无待执行项目。
