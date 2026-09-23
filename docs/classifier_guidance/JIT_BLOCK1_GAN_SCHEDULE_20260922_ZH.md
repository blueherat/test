# JiT 1-block：GAN 学习全程引导系数与性能检查

2026-09-22。初始在 GPU2 验证训练操作和性能，随后按用户要求改为 **GPU0 与 GPU2 协同、全局 batch32、每卡 microbatch8、每卡累积2份**。短训练已正常完成 step256；用户要求继续开始跑后，从该保存点启动累计 **30K次系数更新** 的长训练，每300步保存。双卡切换和实时曲线见文末，尚无本轮GAN的FID结果。

## 模型、系数与训练目标

强模型为 JiT-B/16 checkpoint 的 `model_ema1`，弱模型为已完成真实数据 50K 的 layer6、1-block adapter 的 EMA。两者均冻结。可训练参数是 50 个时间区间各自的额外系数 `a_i`，以及条件真假二分类器。

所有系数从 **a_i=.50，即论文总系数 w_i=1.50** 初始化。之前 JiT 1-block 的 w=1.50/1.55/1.60 对应 FID-5K 16.3054/16.3157/16.3585，属于较平坦的区域。这是普通 FM 训练弱头的结果，不是本轮 GAN 的结果。

保持原 50 个均匀采样区间、49 次 Heun 与最后一次 Euler，共 99 次向量场求值；不加 CFG。每个 Heun 区间的两次求值共享同一个可学习系数。所有区间从第一步起启用，无前后半程跳变、无 exp/softplus 或正数约束，也没有在系数为零时切断梯度。

用作者的 clean-prediction 混合次序实现：`x_pred = W + (1+a_i)*(S-W)`，再转为 `(x_pred-z)/(1-t).clamp_min(.05)`。保留原 BF16 预测、BF16 差值及 FP32 广播系数的算子精度。数学上等价于 `S+a_i*(S-W)`，但直接改写可能改变有限精度结果，故这里保留原顺序。可学习 a=0 时也走相同的可微表达式；不复用作者针对固定 Python 常数 w=1 的特殊分支。

判别器区分真实训练 RGB 与这条引导轨迹生成的最终 RGB，二者经冻结、可微的 Inception-2048 输入条件二分类器；R1 在特征空间计算。每次先更新 D，再用更新后的 D 的非饱和 logistic loss 更新系数。强弱模型虽冻结参数，仍保留对状态的完整输入导数，经全部 50 个区间反传。

数据为完整 ImageNet 真实训练集，类别均匀抽样、类内抽取真实图片，每次更新有新的噪声与标签；不固定一小批训练数据。只有配对性能 benchmark 固定同一批输入及起始 D/Adam 状态，便于比较不同实现。

## 实现与正确性

- [jit_schedule.py](../../classifier_guidance/jit_schedule.py)：50 个 signed 系数、SSG 1-block 加载、混合求解器、普通 autograd 参照。
- [sampler.py](../../classifier_guidance/sampler.py)：扩展为每个区间独立选择 Heun/Euler，既有纯 Euler/Heun 行为保留；只捕获实际使用的 active 模式。
- [probe_jit_schedule.py](../../classifier_guidance/probe_jit_schedule.py)：真实模型端点/梯度检查及完整 D→系数更新测速。
- [train_jit_schedule.py](../../classifier_guidance/train_jit_schedule.py)：新鲜真实数据训练、D 与系数诊断、EMA、完整保存/恢复。
- [jit_schedule_probe_queue.py](../../classifier_guidance/jit_schedule_probe_queue.py)：GPU2 串行配对测速、数值检查和短训练。

CPU float64 检查包含非均匀网格、混合 Heun/Euler、零/负系数及完整输入/参数梯度的有限差分。相关测试 11 项通过，4 项 CUDA 专用测试未在未分配 GPU 的 CPU 测试进程运行；真实 GPU 验证由下述独立 lease 实验完成。

实际 JiT 1-block 验证通过：

| 检查 | 结果 |
|---|---:|
| 常数初始化端点 vs 作者 SSG 采样器 | 逐位一致 |
| CUDA 图、冻结 BF16 权重预转换后端点 | 逐位一致 |
| 初始系数梯度 vs 普通整轨迹 autograd，相对误差 | 0.05044% |
| 初始输入梯度，相对误差 | 0.14618% |
| 改变噪声、标签及 signed 系数后，系数梯度相对误差 | 0.04321% |
| 有非零梯度的初始系数 | 50/50 |
| 将第26个系数设为0后的梯度 | -0.000157836 |

梯度差来自离散反传的浮点累加次序，不能宣称逐位相同；初始系数梯度余弦相似度约 .999999928。没有省略时间步、截断强弱模型输入 Jacobian 或使用近似连续 ODE adjoint。

首轮审计在先做普通 autograd 后保留了默认 stream 的梯度节点，导致 CUDA 图捕获失败；没有进入 GAN 训练。已在对照阶段释放图并使用同值新叶子参数隔离捕获，重跑审计成功，失败日志保留在 `audit/`，通过记录在 `audit_fresh_leaf/`。

## 性能比较与短训练

GPU2 初轮实测已完成；选择 CUDA 图 + 冻结 BF16 权重预转换作为速度档。最初选单卡 batch8，随后按用户要求增加到全局 batch32、microbatch8，详见下方同全局批量对照。

| 配置（均 batch4） | 秒/完整更新 | allocated GiB | reserved GiB |
|---|---:|---:|---:|
| 普通 checkpoint-autograd | 8.7098 | 2.673 | 2.822 |
| CUDA 图与常量缓存 | 1.9902 | 1.379 | 2.518 |
| 再加冻结 BF16 权重预转换，选用 | **1.8145** | **1.125** | **2.125** |
| 再加 Inception 激活重算 | 1.8626 | 0.882 | 1.914 |
| 再加主干及弱 block 激活重算 | 2.3486 | 0.888 | **1.264** |

速度档相比参照提速 **4.80×**，reserved 峰值下降 **24.7%**；最低显存档提速 **3.71×**、reserved 下降 **55.2%**。reserved 包含图内存池，allocated 不等于实际进程占用；`nvidia-smi` 还包含 CUDA 上下文等分配。

完整 GAN 对照中，D CE、R1、G loss 与 D 参数更新一致；速度档系数梯度与参照的最大相对差异为 **0.4885%**，最低显存档为 **0.8235%**。各优化配置在这次 Adam 更新后的系数最大差异均为 5.96e-8。此处有冻结 Inception 的图像反馈反传，不能将其梯度误差与前面只用端点平方损失的采样器检查混为一谈。只在已测输入/精度范围内核对数值，不保证任意长训练的权重轨迹逐位相同。

| 速度档 batch | 秒/完整更新 | 图片/秒 | reserved GiB |
|---|---:|---:|---:|
| 4 | 1.8145 | 2.205 | 2.125 |
| **8，选用** | **3.1990** | **2.501** | **3.732** |
| 16 | 7.1490 | 2.238 | 6.852 |

本次 batch8 比 batch4 吞吐高约 **13.4%**，batch16 未继续改善，因此采用 batch8。此选择来自当前机器的一组短测，不宣称跨硬件的最优 batch。

原始结果：[benchmark_summary.json](/home/zhoushunyu/data/eqvae/projects/classifier_guidance/jit_block1_gan_schedule_20260922/benchmark_summary.json)，各目录包含逐次计时、实际梯度数组和对照检查。

### 固定全局 batch32：microbatch8 与16

| microbatch | 累积份数 | 秒/完整更新 | allocated GiB | reserved GiB |
|---|---:|---:|---:|---:|
| **8，正式采用** | **4** | **12.7355** | **3.620** | **5.607** |
| 16 | 2 | 14.2753 | 4.375 | 8.037 |

每项均同一组32个真实样本、噪声、标签和初始参数，排除预热后取3次中位数。8×4 比16×2耗时少 **10.8%**，reserved 少 **30.2%**。这组实验同时改变了大模型/图像特征网络的实际批形状，测得两者并非浮点数值等价：D CE 分别1.400852/1.401122，最后一次系数梯度范数 .035689/.018174。没有把16作为数值等价的替代配置；其跨批形状差异尚未逐模块定位，正式固定为用户要求的8。

累积实现见 [training_accumulation.py](../../classifier_guidance/training_accumulation.py)：生成器与 Inception 每次计算8张；收集32份真/假特征后，较小的 D 一次处理整个特征批次并更新一次，保证 spectral normalization 的幂迭代也只进行一次。然后用更新后的固定 D，分4份求系数梯度，每份 loss 乘8/32，全部累积后才裁剪并更新一次 Adam/EMA。不是连续做4次小 batch 的D/G交替更新。

每份采样保留完整离散轨迹，只重算一次冻结图像特征以避免同时保留4份 Inception tape；没有重采噪声或截断梯度。CPU float64 已比较完整batch32与8×4的损失、D/系数梯度、参数更新和谱归一化缓冲，覆盖正常更新与D预热；连同求解器测试共13项通过。有限精度的跨microbatch对照与这个数学累积等价检查应区分。

初始 batch8 在step129保存并暂停（128步D预热、1步系数更新）。batch12仅完成性能测量，未进行训练。当前从step129保留系数、D、两个Adam和数据RNG，改用全局32/micro8继续到step256，共计128次系数更新；改变batch后不声称随机训练轨迹与原batch8逐位连续。恢复权重逐位一致的记录在 `pilot_global32_micro8/resume.json`。

在step135时，已完成7次系数更新，额外系数范围约 .4933–.4985，前/后半程均有非零梯度；实际更新约13.03秒、reserved约5.62GiB。这里只用于确认训练在工作，不代表FID提升。

同 batch4 比较：普通 checkpoint-autograd、CUDA 图、加冻结 Linear/Conv BF16 权重预转换、加 Inception 激活重算、再加主干/弱 block 激活重算。每组排除一次预热，对三次完整 D→系数更新取中位数；检查 D/G loss、D 更新和系数梯度一致性。计时包含完整 99 NFE 采样、RGB 特征、D 更新及 R1、全部系数梯度和 Adam，不包含首次加载、捕获、数据读取及保存。

选定优化配置后另测 batch8/16 吞吐。改变 batch 是独立吞吐实验，不把它声称为等全局 batch 的算法提速。最终短训练使用新选定的全局 batch，D 预热128步，然后系数更新128步；Adam 系数 lr=1e-3、D lr=1e-4，R1=1，系数 EMA=.99，每64步保存、每16步记录梯度与真假概率诊断。短训练不能代替生成质量结论。

完整权重保存包括系数/raw/EMA、D、两个 Adam、数据 RNG、运行 RNG、冻结来源哈希及源码记录，可从保存点继续。每次保存确认冻结模型和弱头未更新。

结果根目录：`/home/zhoushunyu/data/eqvae/projects/classifier_guidance/jit_block1_gan_schedule_20260922`。单卡阶段 tmux 为 `jit_gan_scale_0922_g32`，现已保存暂停；当前活跃目录见 `status.json`。旧 batch8 因用户变更保存暂停，原队列的缺少complete错误属于这个主动切换，不是训练数值失败。

## GPU0、GPU2 协同与实时监控

单卡在 step195（67次系数更新）完整保存暂停，续训使用 `pilot_global32_micro8/checkpoint_000195.pt`。新训练目录为 `training_gpu0_gpu2_global32_micro8`，tmux 为 `jit_gan_scale_0922_g32_2gpu`。全局 batch32 和 microbatch8 保持不变，每卡处理16张、各累积2次，所有系数覆盖完整50个采样区间。

rank0 按原状态抽取完整32份真实图片、噪声和类别，再分给两卡，避免切换卡数后复制样本或改变全局抽样流。两卡各自产生8张一份的端点；仅汇集较小的冻结2048维特征，让每个D副本仍以全局32真+32假的同一形状执行一次训练前向与谱归一化。D梯度同步后裁剪并更新；系数则在每卡累积本地平均梯度，再跨卡平均、裁剪和更新一次 Adam/EMA。并行累加次序有浮点差异，不保证长期训练轨迹逐位一致。

CPU float64 双进程测试覆盖D预热、连续两次Adam更新，以及每卡有/无额外累积的情况；比较损失、梯度、参数及谱归一化缓冲。相关测试15项通过、4项CUDA专用测试跳过。实际JiT的单卡与双卡配对检查结果写入 `audit_gpu0_gpu2/result.json`，只有检查通过后控制器才启动续训。

真实模型对照已通过。从同一个step195模型及Adam状态、同一批32份真实图片/噪声/标签出发，microbatch均为8，排除1次预热后重复3次：单卡完整更新中位数 **12.9258秒**，双卡 **6.5194秒**，提速 **1.983倍**。D的参数与谱归一化缓冲完全一致；系数梯度相对误差 **0.20860%**、余弦相似度 **0.99999785**，一次Adam更新后的系数最大差 **5.96e-7**；两卡最终参数逐位一致。该配对检查复用了单卡基准的CUDA内存池，因此不能用其中的reserved峰值判断独立双卡进程的显存节省，应看正式续训记录。

正式续训已恢复：两卡的第一批真实图片和噪声哈希与单卡续训的预期完全一致。step197–202实际更新约 **6.70–6.74秒**，每卡峰值 allocated **2.45GiB**、reserved **4.39GiB**（两卡取最大值）。step196已成功保存，保存时确认两卡系数、EMA与D完全一致。

保留所有带步数的checkpoint，不删除旧实验段。用户随后将保存间隔改为 **每300步**；训练结束或主动暂停仍保存完整状态，不再为普通续训额外保存第一步。存档包括 raw/EMA 系数、D、两份Adam、全局数据RNG、各卡运行RNG、模型来源和源码校验信息；每次保存检查两卡的 raw/EMA/D（含缓冲）逐位一致、冻结预测器未改动。`checkpoints.json` 索引所有保存点。此次修改时已到step245，当前短跑在256结束，剩余范围没有额外中间保存点，因此不为更改保存间隔重启训练；后续启动按300步执行。

CPU监控脚本 [monitor_jit_schedule.py](../../classifier_guidance/monitor_jit_schedule.py) 每30秒刷新以下文件，连续展示单卡、双卡训练历史；训练结束后输出最终图并退出：

- `index.html`：自动刷新的本地查看页面。
- `schedule.png` / `schedule.pdf`：当前 raw、EMA、初始常数、系数历史、全程梯度与训练损失。
- `gan_health.png` / `gan_health.pdf`：判别器概率、D更新、RGB端点反馈梯度、系数更新信号。
- `coefficients.csv`：50个区间的当前 raw/EMA 系数。
- `gan_health.json`：训练状态、最新诊断、耗时、保存点数量。活跃梯度不代表FID改善。

按用户查看方式，实时PNG也同步写入仓库的 [figures/jit_gan_schedule.png](figures/jit_gan_schedule.png) 和 [figures/jit_gan_health.png](figures/jit_gan_health.png)，约每30秒原子替换一次；这些是实际图片文件，不是外部数据目录的符号链接。打开PNG即可查看，不需要HTML页面。

GPU0有26 MiB的 `gnome-initial-setup` 桌面CUDA上下文，启动器通过显式PID及完整系统程序路径核对后允许共存；没有结束该进程，GPU1、GPU3未使用。

## 30K 续训

step256短训练正常退出，系数已更新128次，raw范围0.38434–0.52208；两卡参数一致，冻结模型未变化。从完整 `checkpoint_000256.pt` 继续另外29,872次更新，目标为累计30,000次系数更新（包含128步D预热的全局step为30,128）。没有重置D、raw/EMA系数、Adam或随机抽样状态。

新目录 `training_30k_gpu0_gpu2_g32_m8`，tmux `jit_gan_scale_30k_0922`。每300个全局step保存，结束或主动暂停另存最终完整状态；训练和曲线监控分别在 `train`、`curves` 窗口独立运行。当前速度约6.7秒/更新，剩余训练粗估约56小时，实际受运行吞吐影响。监控页面沿用根目录 `index.html`，每30秒刷新；历史热图最多显示1,000个均匀选取的更新时点以控制绘图成本，完整逐步系数仍保存在各段 `train.jsonl`。
