# SiT 两档 MLP 容量：纯 diffusion 50K 与 5K 系数扫描

用户要求两个更强的 MLP 头，各用 diffusion loss 训练 50K，再扫描外推系数。
本轮继续使用 ImageNet-100 SiT-S/2 800K EMA，第 4 层冻结特征。

| 头 | 隐藏宽度 | 初始融合层之外的残差隐藏层 | 参数量 |
|---|---:|---:|---:|
| 旧 Context MLP，历史参考 | 384 | 0 | 304,528 |
| moderate，稍强 | 384 | 1 | 452,368 |
| large，更强 | 768 | 2 | 1,790,224 |

融合为 `h = SiLU(token(x)+condition(c)+position(p))`，新增各层为 `h = h + SiLU(Linear(h))`，最终线性映射到每 patch 16 个速度值。
宽版仍接收宽度 384 的冻结输入；加宽的是头内部隐藏层。新增残差层零初始化，输出层零初始化。
两个头从新的随机参数开始，只共用此前全量训练数据估计的输入归一化统计；不继承已训练 MLP 或 GAN 头参数。

## 训练目标与数据

\[
z_t=t x+(1-t)\epsilon,\qquad
\mathcal L_W=\mathbb E\|W(z_t,t,c)-(x-\epsilon)\|^2.
\]

这是 SiT 的普通速度 flow-matching MSE。目标中没有外推系数、强模型预测、GAN 或 guided-weak loss。
使用完整 **126,689 张**真实训练图像的 SD-VAE posterior moments；按 epoch 重新打乱，逐次采样 posterior、时间和噪声。验证数据不参加训练。

- 每头 50,000 个 optimizer steps，全局 batch256；学习率 1e-4，AdamW betas=(.9,.999)，weight decay=0。
- BF16 autocast / FP32 参数，EMA=.9999，沿用原 diffusion 头设置。
- moderate 使用 GPU1，large 使用 GPU2，各自单卡；GPU0 存在其他进程，未占用。
- 先完成 100 步训练检查，再保留模型、EMA、Adam 和 RNG 续训至总计 50K；不是额外再训 50K。
- 每 5K 保存检查点并评估固定验证输入的速度 MSE；最终预先选定 50K EMA 做系数扫描。

前期稳态速度约 31–35 ms/step；峰值 allocated 约 moderate 1.67 GiB、large 2.08 GiB。每头训练约半小时，具体以日志为准，扫描耗时另计。

## 外推系数搜索

采样使用原来的强弱差值与时间窗：

\[
V=S+a f(t)(S-W),\quad
f(t)=\begin{cases}6/7&t<.25\\1&.25\le t<.5\\0&t\ge.5.\end{cases}
\]

`a` 是额外外推系数，不是 `w=1+a`。本轮扫描标量 `a`，不学习时间曲线。

1. 粗扫 a=1.0、1.2、1.4、1.6、1.8；按此前要求不跑 2.0。
2. 如果最优一直位于左边界，按 0.2 向左延伸，直到出现内部最优或到达 0。
3. 在粗扫最优附近选 **恰好 8 个**相邻系数，间隔 **0.05**；通常左 3、右 4 个，边界处整体平移。
4. 所有点均为 **5,000 张**，不使用 1K 排名；与粗扫重复的细扫点直接复用，不重复采样。
5. 共用此前 5K 噪声/标签，每类 50 张；64 步 Heun，batch8，同一 Heun 步的两次查询共享系数。沿用 ADM CUDA FP32 / TF32 关闭评估。
6. 最低 FID 是该扫描范围内的最好结果；若最优落在右边界会标记，不声称全局最优。不追加独立 5K 验证。

与本轮 loss 更对应的历史浅 MLP 参考是 `real__c0040`：**FID-5K=36.909452，a=1.0**，普通 diffusion loss 训练50K。
原始记录：[metrics.json](/home/zhoushunyu/data/eqvae/experiments/guidance_dynamic_50k_20260915/sit_small/points/real__c0040/n5000/metrics.json)。旧训练使用四卡，本轮两档各单卡；旧系数由 1K 预筛后扩展 5K，本轮全程用 5K 搜索，因此它是历史参考。
此前 36.688848 是 guided-weak loss，36.444090 是 GAN 续训头，不应直接称为普通 diffusion 容量对照。

## 检查与后台运行

已通过：普通 FM loss 对既有实现逐位一致；两档前四个 batch 的真实图索引、噪声和时间完全一致且随步更新；100 步后所有头参数组均有更新、主干哈希不变；两档加速采样端点都与原始 64 步 Heun 实现逐位一致。
另用真实 SiT 比较连续 100→102 与分段 100→101→102：最终头、EMA、Adam 与各项 RNG 状态逐位一致，见 `resume_audit/result.json`。既有头结构与系数测试 4 项通过，源码编译与 diff 格式检查通过。

原 tmux：`sit_mlp_capacity_0920`，已正常完成退出。两档均完成50K；全部24个新5K点（共120,000张）在2026-09-20 21:05完成。

实验目录：`/home/zhoushunyu/data/eqvae/projects/classifier_guidance/sit_mlp_capacity_diffusion_20260920/`。

- `plan.json`、`status.json`、`pipeline.log`：冻结搜索规则与全局状态。
- `{moderate,large}/training_50k/`：训练日志、检查点、验证损失、冻结源码。
- `{moderate,large}/fine_plan.json`：实际选定的 8 个系数。
- `{moderate,large}/results.json`：每点指标和已测最优。
- `sampler_audits/`、`paired_data_audit.json`：采样一致性和训练输入检查。
- `results.json`：两档完成后的最终汇总。当前是否完成，以该文件与 `status.json` 为准。

首次预检遇到 GPU0 已占用及 MSE 两种等价归约顺序的 2.4e-7 浮点差异，未完成优化更新；改用 GPU1/2 并对齐原实现归约顺序。旧预检只保留在同级 `_preflight_retired/`，不并入训练或效果比较。

代码：[头结构](../../classifier_guidance/capacity_heads.py)、[训练](../../classifier_guidance/train_capacity.py)、[采样评估](../../classifier_guidance/evaluate_capacity.py)、[后台队列](../../classifier_guidance/capacity_pipeline.py)。

## 完成结果与公平性复核（2026-09-20 22:11）

两档最优均在额外系数 a=1.05：moderate FID=37.055296，large FID=37.812169。
旧浅头普通FM历史最优为36.909452（a=1.0），差值分别+0.145844、+0.902717。
这描述已训练模型的效果，不能直接推断增加容量导致下降。

| a | moderate FID-5K | large FID-5K |
|---:|---:|---:|
| 0.8 | 38.157894 | 38.898134 |
| 0.85 | 37.785499 | 38.516653 |
| 0.9 | 37.526527 | 38.305204 |
| 0.95 | 37.310606 | 38.078557 |
| 1 | 37.148469 | 37.938994 |
| 1.05 | 37.055296 | 37.812169 |
| 1.1 | 37.106745 | 37.929967 |
| 1.15 | 37.274271 | 38.154571 |
| 1.2 | 37.557362 | 38.340081 |
| 1.4 | 38.723392 | 39.268505 |
| 1.6 | 40.517048 | 41.087669 |
| 1.8 | 43.032929 | 43.078913 |

复核了新旧5000噪声/标签：旧浅头625个批次的噪声哈希和标签逐批一致。所有新点均5000图、625批、来源检查点正确，主干和头未改变，见 `completion_audit.json`。
训练步数、global batch、数据、loss、优化器、EMA相同；旧浅头四卡的随机数据流与新头单卡不同，旧系数经过1K筛选，而新头全部5K搜索。相同global batch的不同卡数不自动使比较无效，但单训练种子、不同随机流与选择预算不足以支持严格的容量因果结论。固定50K也是固定更新预算，不证明所有结构均已收敛到各自最佳解。

### 弱头自身：补充配对验证与独立生成

对相同5000验证图像，每图10个分层时间/噪声状态（合计50,000状态），共用相同目标及depth4特征，比较部署FP32/TF32下的速度MSE。随后每个弱头单独运行全部64步Heun，完全以W为速度场，各生成相同噪声标签的5000图。独立生成指标与强模型无外推的FID是不同量。

| 头 | 全时段验证MSE | 前半程MSE | 后半程MSE | W独立生成FID-5K |
|---|---:|---:|---:|---:|
| 原生depth4输出头 | 0.919481 | 0.742773 | 1.096188 | 238.746852 |
| 旧浅MLP | 0.868912 | 0.721746 | 1.016079 | 197.784247 |
| 稍强MLP | 0.865535 | 0.719900 | 1.011171 | 193.033650 |
| 更强MLP | 0.855646 | 0.716438 | 0.994854 | 179.882077 |

同输入强模型验证MSE为0.782598。结果表明这些已训练头中，增加MLP容量确实改善自身拟合与独立生成；但并未改善作为负参考时的引导FID。不能把“引导FID变差”解释为“弱头表达能力没有增强”。原生头独立生成也较弱；其参数结构和历史训练协议不同，不是与三个MLP严格配对训练的容量消融。

原始记录：`head_strength_audit/prediction/result.json`、`per_image_errors.npz`及各`weak_*/metrics.json`。全部4个独立生成端点均与未经CUDA graph加速的W-only求解器逐位一致。未做额外的独立5K系数确认。

与SSG的区别：它在像素JiT中添加具有跨patch注意力的Transformer block，并从后部block和最终头复制初始化；这里是在latent SiT中扩大逐patch MLP，随机初始化。SSG表6展示具体结构的容量效果，不能理解为任意架构加宽都会提高IG。[论文](https://arxiv.org/html/2607.29122v1#S4.SS3)。后续JiT对照只使用完整真实训练数据，见新协议。
