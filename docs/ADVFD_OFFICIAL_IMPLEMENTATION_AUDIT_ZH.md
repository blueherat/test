# AdvFD 官方实现审计

审计日期：2026-08-23

官方仓库：`/data/users/zhoushunyu/research_repos/AdvFD`

官方提交：`4e4cfed944e4fc38a75fae3ea7701ae9e5587060`

本审计只在 paper-only 实现、配置、checkpoint、结果和歧义表完成冻结后开始。因而下列差异不会反向改写独立实验的历史记录。

## 1. 当前总判断

独立 5k 实验已经证明：在一个缩放后的 pMF-B 设置中，adaptive branch 到 4k--5k 才开始稳定优于 static branch。这个正信号是真实的，但它不能作为 AdvFD 的数值复现，因为官方实现与 paper-only 预先决策之间存在若干会改变训练动力学的差异。

目前最重要的差异是：

1. 官方 adaptive EMA 的一、二阶矩及协方差运算全程使用 FP64；独立 5k 版本训练时使用 FP32。这解释了独立版本在 4771 step 附近出现的协方差消失和 critic 梯度突增。
2. 论文 Algorithm 1 写成先 G-step、再用更新后的生成器重生成并做 D-step；官方代码实际先更新 critic，再用更新后的 critic 对同一批旧生成图计算 G loss，最后更新 generator。
3. 官方 G-step 使用 normalized adaptive FD，但 D-step 最大化 raw adaptive FD；独立版本两边都做了 normalization。
4. 官方 regularized whitening 会同时给 fake covariance 加 `epsilon I`，并把 real target 直接设为 `(0, I)`。这不等于论文 Eq. (33) 对原始 real/fake features 共同施加 `(Sigma_real + epsilon I)^(-1/2)` 的有限 epsilon 形式。
5. 官方训练路径不 clamp 生成图；独立版本将 `[−1,1]` 转 `[0,1]` 后 clamp。独立运行的越界像素比例约为 0.4%--1.0%，因此不是严格零影响。
6. 官方 Inception 连续输入归一化是 `2x-1`；独立 wrapper 延续 torch-fidelity 的 `(255x-128)/128`。权重逐 tensor 完全相同，但特征并不完全相同。
7. 官方多卡对已经由 global features 定义的 FD loss 再做参数梯度平均，产生一个额外的 `1/world_size` 缩放。这会使训练步数和 GPU 数量发生隐藏耦合；两卡数值实验已精确确认该因子。

因此后续必须明确区分：

- `paper-faithful`：遵循论文公式和 Algorithm 1；
- `official-code-faithful`：遵循公开代码实际执行顺序和数值细节；
- `scaled pilot`：为了 4x4090 可运行而缩小 batch、representation 或训练长度。

## 2. A01--A26 对照

| 编号 | 官方实现答案 | 与 paper-only 决策的关系 |
|---|---|---|
| A01 | G-step 的 static/adaptive FD 均使用 `L/(sg(L)+0.01)`；D-step 直接最大化 raw adaptive FD | 独立版 D-step 也 normalized，不同 |
| A02 | `--fd_adv_detach_real` 时 D-step 的 real encoder forward 在 `no_grad` 下执行；whitener总是 detach；G-step 冻结 critic 参数但保留 fake-image gradient | 主体一致 |
| A03 | D-step 复用 generator 更新前的同一批 `sampled.detach()`，不在 G-step 后重生成 | 与论文 Algorithm 1 和独立版相反 |
| A04 | adaptive EMA 每个 generator iteration 最多 commit 一次；若本步更新 critic，commit 的是更新后 critic 在 G-loss 路径产生的 features | “每步一次”一致，但具体时序不同 |
| A05 | adaptive real stats 从离线 real reference 初始化；adaptive fake stats 在 start step 从当前 static FD queue/EMA 初始化 | 与统一 50k warm-start 的总体意图接近 |
| A06 | real whitening 使用 adaptive real EMA；该 EMA 从离线 reference 开始并在 start step 后随 critic features 更新 | 一致 |
| A07 | adaptive EMA 使用 raw second moment `E[xx^T]`，covariance 为 `M2-mu mu^T`，即 population convention | 一致 |
| A08 | 先 differentiable all-gather global features，再计算 moments | 一致 |
| A09 | real/fake 训练 features 都是连续 float；不经过 uint8 量化 | 一致 |
| A10 | static/adaptive representation 训练前不 clamp 生成图；只有 PatchGAN/部分 bootstrap 路径 clamp | 独立版 hard clamp，不同 |
| A11 | official real-image loader 只 center crop + tensor，不做 horizontal flip | 独立版训练 real batch 随机 flip，不同 |
| A12 | adaptive Inception 使用 torch-fidelity 权重的 pool3 2048D feature | 独立 full-2048 路径一致；512D pilot 是有意缩放 |
| A13 | representation critic 始终 `eval()`，但参数可训练 | 一致 |
| A14 | 有 D 更新的 iteration 实际执行顺序是 D optimizer step -> G loss/backward -> G optimizer step | 与论文和独立版相反 |
| A15 | 条件为 `(current_step-start_step) % update_freq == 0`，所以首次 critic update 正好发生在 step 1000 | 与“完成第 2、4... 个 G-step 后 D”不同 |
| A16 | step 1000 时 critic 用完整 LR 更新；generator adaptive weight 从 0 开始，在后续 4000 step 线性升到设定值 | 基本一致，仅有索引边界差异 |
| A17 | pMF-B 为 1 NFE、CFG 8.5、interval `[0.1,0.7]`、noise scale 1 | 一致 |
| A18 | 训练时并行跟踪 EDM EMA；正式脚本默认评估 online generator | 一致 |
| A19 | moments、covariance、whitening eigendecomposition 使用 FP64；最终 FD scalar cast 回 FP32 | 独立训练统计曾使用 FP32，不同 |
| A20 | SigLIP2、MAE、Inception 各自 normalized FD，unit weight 后求和 | 一致；当前 5k pilot 仅 Inception 是有意缩放 |
| A21 | 同一 iteration 的 adaptive EMA 只推进一次，不因 D frequency 多推进一次 | 一致 |
| A22 | start step 前不更新 adaptive EMA；到 step 1000 时从当前 static stats 初始化 | 独立版从 step 0 更新 adaptive EMA，不同 |
| A23 | `main_fd.py` 的训练 forward 没有使用 `args.dtype` 对应的 autocast；按当前代码实际为 FP32。评估 `generate_images` 才使用 BF16 autocast | 独立版训练使用 BF16，不同 |
| A24 | 官方 evaluator 的 labels 按 `0..999` 循环；noise 使用当前 CUDA RNG，依赖 batch/rank 切分 | 独立正式评估采用 pMF per-sample CPU RNG/XOR 协议，不同 |
| A25 | 官方 in-memory FID 对 clamp 后的连续 `[0,1]` 图计算，不做 uint8 round trip | 独立正式评估做 pMF uint8 round trip，不同 |
| A26 | `pool_type=cls` 对无 prefix 的 SigLIP2 实际调用模型 MAP attention pool；MAE 取第一个 prefix token | 已完成权重、输出和输入梯度核对；MAE 等价，SigLIP2 因固定/动态位置编码构造存在极小数值差异，见 5.3 |

## 3. 论文与官方代码的直接冲突

### 3.1 G/D 更新顺序

论文 Algorithm 1：

```text
Xg = G(theta_t, Z)
theta_{t+1} <- G-step(Xg, omega_t)
Xg_det <- stopgrad(G(theta_{t+1}, Z))
omega_{t+1} <- D-step(Xg_det)
```

官方代码实际执行：

```text
Xg = G(theta_t, Z)
omega_{t+1} <- D-step(stopgrad(Xg))
theta_{t+1} <- G-step(Xg, omega_{t+1})
```

这不是无关紧要的代码重排。它改变了 critic 看见的 generator 版本，也改变了 generator 使用的 critic 版本。后续需要将这两种顺序作为明确对照，而不能混称为同一个 AdvFD update。

### 3.2 有限 epsilon whitening

论文 Eq. (33) 定义：

```text
psi_bar(x) = (psi(x)-mu_real) (Sigma_real + epsilon I)^(-1/2).
```

若直接应用该 affine map，则 whitened real covariance 是：

```text
Sigma_real_bar = W^T Sigma_real W,
```

而不是严格的 `I`。

官方代码计算的是：

```text
Sigma_fake_bar = W^T (Sigma_fake + epsilon I) W,
real target = I.
```

它等价于先给 real/fake Gaussian 同时加入 isotropic covariance，再做 whitening。这个版本在 real=fake 时严格为零，数值性质更好，但不是论文 Eq. (33) 的字面实现。独立版采用了论文的共同 affine transform，因此两者必须分开报告。

## 4. 多卡梯度缩放

官方 `diff_all_gather` 在 forward 中拼接全局 features，但 backward 只返回本 rank 对应的 feature chunk。设 global batch 为 `N=WB`，FD loss 已由 global moments 定义，则本 rank 得到的参数梯度为 `g_r`，正确全局参数梯度应为：

```text
g_global = sum_r g_r.
```

官方随后对参数梯度执行：

```text
all_reduce(..., AVG)
```

得到：

```text
g_official = (1/W) sum_r g_r = g_global / W.
```

因此公开脚本在 8 GPU 下存在额外 `1/8` 的梯度缩放。两卡数值实验直接调用官方 `diff_all_gather` 和 `all_reduce_grads`，结果为：

```text
rank 0 local contribution = 0.6200902356203759
rank 1 local contribution = 1.2665231416162746
SUM                      = 1.8866133772366505
centralized full-batch   = 1.8866133772366505
official AVG             = 0.9433066886183252
official / centralized   = 0.5
```

`SUM` 与集中式梯度的绝对误差为零，官方 `AVG` 与集中式梯度除以两卡后的绝对误差也为零。因此该隐藏缩放已由真实两进程 NCCL 路径确认，而不只是解析推断。机器可读结果见 `docs/data/advfd_official_distributed_gradient_scaling_2gpu.json`。由此：

- paper-faithful 多卡实现应对这些局部 gradient contributions 求和；
- official-code-faithful 实现应保留平均；
- 使用不同 GPU 数量复现时必须登记这一 world-size dependence。

但不能把它直接解释成 AdamW 的“有效学习率恰好再除以 `W`”。若所有梯度从零
optimizer state 开始被同一个正常数缩放，Adam 的一、二阶矩会分别缩放为该常数和其
平方，除 `eps` 外更新方向与幅度近似不变；critic 在 pre-clip norm 大于阈值时，后续
归一化裁剪还会进一步消掉全局常数。因而跨 world-size 的实际差异来自 Adam `eps`、
是否跨过 clip 阈值、不同参数的缺失梯度，以及更重要的 global batch/moment 估计噪声。
该缩放是实现事实和潜在耦合项，但不能仅凭它断言三卡轨迹等价于把八卡学习率放大
`8/3` 倍。

## 5. 已完成的跨实现数值检查

### 5.0 官方仓库中未进入论文正文的 critic 约束

官方提交历史显示，在论文 arXiv 发布前，仓库已经实现但未在正文表格中报告以下
FD-Adv 稳定化对照：

| 约束 | 官方实现 | 相关历史 |
|---|---|---|
| 逐样本 feature L2 cap | 将 adaptive feature 投影到固定 L2 球 | `c948040` 及 `table_3_JiT_adv_fd_sim_inception_l2_cap.sh` |
| shared residual-RMS trust region | 对 real/fake 相对冻结 pretrained feature 的 residual 使用同一可微缩放，保证 50:50 mixture residual RMS 有界 | `c948040`、`docs/plans/2026-07-28-fd-adv-shared-residual-rms.md` |
| shared feature second-moment penalty | 约束 adaptive real/fake 混合二阶矩相对 pretrained reference 的偏移 | `893017c` 及 `table_3_JiT_adv_fd_sim_inception_norm_offset.sh` |
| split real/fake second-moment penalty | 分别约束 real 与 fake feature scale | `058ea0a` 及 `table_3_JiT_adv_fd_sim_inception_norm_offset_split.sh` |

其中 residual-RMS 方案定义

```text
delta_r = psi_train(real) - psi_ref(real)
delta_f = psi_train(fake) - psi_ref(fake)
R^2 = 0.5 * (E||delta_r||^2 + E||delta_f||^2)
a = (1 + R^2 / tau^2 + eps)^(-1/2)
psi_hat = psi_ref + a * (psi_train - psi_ref)
```

并在 real/fake 全局 gather 后计算同一个 `a`。这已经直接覆盖“用 pretrained feature
作为 trust region 限制 adaptive critic”的朴素方法空间。故后续若发现当前 critic 的巨大
尺度/低秩振荡有害，以上四类必须作为既有官方实现基线；不能把 norm cap、residual RMS
或 second-moment matching 单独包装成新贡献。它们没有进入论文正式实验仍留下一个经验
问题：这些约束是否失败、收益不足，还是仅因篇幅未报告。该问题只能由同协议对照回答，
不能从代码存在性反推结果。

### 5.1 pMF-B 一步生成

同一 checkpoint、noise、class、CFG 和 interval 下：

```text
official wrapper vs upstream pMF wrapper
max absolute output difference = 1.2159e-5
RMS output difference          = 5.6188e-7
```

因此 generator 结构、checkpoint conversion 和一步采样公式基本等价。

### 5.2 Inception 特征

两边 Inception 的 566 个 state-dict tensors 逐元素完全一致，但对同一 `[0,1]` float image：

```text
max feature difference = 1.5875e-2
RMS feature difference = 1.9989e-3
```

主因是连续输入归一化不同：

```text
official:         2*x - 1
independent: (255*x - 128) / 128
```

将输入比例校正后 RMS 差异降到约 `2.33e-4`，残余来自 resize 数值路径。因此正式 code-faithful run 必须使用与 reference stats 相同的 official preprocessing，不能只保证网络权重相同。

### 5.3 MAE 与 SigLIP2 特征及输入梯度

下载官方指定的 timm 权重后，使用同一随机 `[1,3,256,256]` 连续输入，同时比较 pooled feature 和同一随机线性 feature functional 对输入图像的梯度。

MAE-L 的 294 个 state-dict tensors 完全相同；官方首 token 与独立 `forward_head(..., pre_logits=True)` 的 feature 逐元素完全相同。输入梯度只剩浮点运算舍入：

```text
feature max / RMS difference       = 0 / 0
input-gradient max difference      = 2.9802e-8
input-gradient RMS difference      = 2.2979e-10
official input-gradient RMS        = 8.3405e-3
```

SigLIP2 的 342 个公共 state-dict tensors 权值相同，但 `pos_embed` 形状不同：官方以 `dynamic_img_size=True, dynamic_img_pad=True` 构造 256px checkpoint，再在 forward 前 resize 到 224；独立 paper-only 版本直接以固定 `img_size=224` 构造。两者均调用 MAP attention pooling，但位置编码插值时机不同，得到极小而非零的差异：

```text
feature max difference             = 2.0325e-5
feature RMS difference             = 5.7739e-6
official feature RMS               = 6.7717e-1
input-gradient max difference      = 8.7204e-3
input-gradient RMS difference      = 2.2295e-4
official input-gradient RMS        = 4.9807e-1
```

因此 MAE 路径可以视为数值等价；独立 SigLIP2 路径与官方非常接近，但不能宣称逐元素等价。正式 code-faithful run 直接使用官方 `TimmReprModel`，不会继承这项 clean-room 构造差异。机器可读结果见 `docs/data/advfd_official_timm_feature_equivalence.json`。

### 5.4 官方代码端到端 smoke

官方 `main_fd.py` 通过 packed ImageNet 数据适配器完成了两类真实双卡 smoke：

```text
Inception static + Inception adaptive, local batch 1:
  peak memory = 5.42 GB/device
  critic update, generator backward, EMA and checkpoint all completed

full SIM static + Inception adaptive, local batch 1:
  peak memory = 8.52 GB/device
  steady step time about 2.96 s

full SIM static + Inception adaptive, local batch 32:
  peak memory = 22.16 GB/device
  steady step time about 4.15 s
```

这些 smoke 只证明官方代码路径和数据适配完整可运行，不用于判断方法有效性。实际中程复现采用更保守的 local batch，并跨过 adaptive start/warmup。

### 5.5 中程缩放复现的 critic 尺度诊断（运行中）

当前 official-code-faithful 中程运行使用完整 SIM static、Inception adaptive、官方
`official_avg` 梯度语义、3 卡 local batch 24（global batch 72）、50k queue、10k
generator steps。它与论文的 global batch 1024 不同，因此这里只登记机制迹象，不提前
把它解释成论文数值复现或方法失败。

直接读取 checkpoint 内保存的 adaptive real/fake FP64 EMA moments，可见 adaptive
Inception 在很小的参数位移下发生了极大的输出尺度和谱形状变化：

```text
checkpoint   real RMS / ref   fake RMS / ref   real effective rank   fake effective rank
step 1999          607.95           572.91               5.41                  5.19
step 2399          518.50           518.33               4.87                  4.78
step 2799         1081.84          1109.86               4.08                  3.99
step 3199          397.30           403.36               3.89                  3.87
step 3599          169.94           172.58               4.43                  4.42
step 3999          895.95           915.06               4.43                  4.32
step 4399          419.72           425.55               3.25                  3.25
step 4799          950.77           962.85               4.14                  4.11
step 5000          657.22           663.37               5.42                  5.39
```

这里的 RMS 定义为 `sqrt(E||feature||^2 / d)`，reference 是官方 ImageNet Inception
moments。effective rank 使用 covariance participation ratio：

```text
r_eff(Sigma) = tr(Sigma)^2 / ||Sigma||_F^2.
```

从 step 1999 到 step 3199，adaptive Inception 参数总范数只从约 `251.014` 变到
`251.019`；此前 step 999 到 step 1999 的参数差范数约为 `0.966`，相对参数范数仅
`0.385%`。因此这不是 checkpoint 损坏或参数整体乘上数百倍，而是深层 Inception 对
小参数位移产生了高度放大的、低有效秩的 feature geometry。

训练日志的 sparse raw FD 也与此一致：step 3000 的当前 raw critic/generator FD 约为
`178631/178593`，而对应 real-whitened FD 约为 `126/124`。白化确实把进入 min-max
loss 的数值压到了有限范围，却没有把白化前表示保持在冻结 encoder 附近。

这项诊断目前支持的最窄结论是：

```text
real whitening 消除了理想 population、epsilon=0 下的直接全局缩放收益，
但当前有限 epsilon + EMA + neural critic 实现仍允许巨大的共同尺度漂移和谱坍缩。
```

它**尚不能**单独证明 selective fake-only amplification，因为保存 moments 中 real/fake
总 RMS 很接近。下一步必须在相同 checkpoint 上用全新 held-out real/fake images 计算
逐样本 feature norm、分位数、reference residual，以及 train-EMA/held-out FD，才能区分：

1. 仅是 real/fake 共同的近似 affine gauge drift；
2. EMA 滞后导致的瞬时尺度震荡；
3. 集中在少量 fake/artifact 区域或少数判别方向上的 selective amplification。

论文 Figure 5 的 whitening 曲线约为个位数到几十倍，而本运行早期已达数百倍。但论文
曲线来自不同 backbone/正式 global batch，并用 10k 新图直接测 checkpoint feature；
这里首先使用保存的 EMA moments。因此二者是强烈的检查信号，不是已经公平成立的
数值反证。训练继续到足够长度，最终仍以 baseline/5k/10k 的同协议生成评估为准。

为验证这不是 checkpoint EMA 队列自身损坏，在 step 3999 上还用训练未使用的
ImageNet validation 图片和外部图片完成了 8 张端到端 smoke。adaptive/reference
feature RMS 比值分别约为：

```text
fresh ImageNet validation real: 1169.40
external fake/image control:     986.05
adaptive fake/real RMS ratio:      0.958
```

`n=8` 远小于 2048 维，不能据此比较 held-out FD，也不能精确比较 real/fake 比值；它只
证明数百到上千倍的 feature scale 在全新 validation forward 中真实存在，不是保存 EMA
moment 的读取伪象。正式流程会在与 FID 完全相同的 5k fake 和 5k validation real 上计算
原始/adaptive 两套 moments、effective rank 及稳定的半正定 Bures FD。

还必须保留训练长度的边界。论文正式 pMF 配置是 125k steps、前 6250 steps generator
warmup；当前 10k pilot 忠实保留了同样的 6250 warmup steps，adaptive weight 则在
step 5000 才达到完整的 `0.05`。所以 10k 仅包含 5000 个 full-adaptive steps，且按
global batch 72 只产生约 72 万次样本暴露，不能代替论文 125k/global-batch-1024 的正式
结果。若 5k/10k 趋势支持继续，下一轮应从 base checkpoint 重新启动更长且内部一致的
25k 或 50k schedule；不能把已经按 10k cosine schedule 降到零学习率的 optimizer 直接
续接并制造学习率跳变。

## 6. 独立 5k 结果应如何保留

独立结果仍然回答了一个有效问题：在同一 clean-room 协议下，adaptive branch 在足够训练后是否开始优于 static branch。答案是肯定的：

```text
step 1000: AdvFD - static = +0.0053 FID
step 2000: AdvFD - static = +0.0561 FID
step 3000: AdvFD - static = -0.0047 FID
step 4000: AdvFD - static = -0.1074 FID
step 5000: AdvFD - static = -0.2102 FID
```

它同时说明，1000--2000 step 的短训足以验证程序能跑，却不足以判断方法有效性。后续中程训练必须越过 adaptive start/warmup，并覆盖一个显著的 full-weight 区间。

但该实验不能用来声称复现论文表格，因为它同时包含：Inception-only static、512D adaptive projection、global batch 32、BF16、clamp、paper-order G/D、normalized D loss和不同的 Inception preprocessing。

## 7. 下一阶段验证顺序

1. MAE/SigLIP2 权重、pooling、resize、normalization、reference moments 和输入梯度核对已完成。
2. global-feature FD 的 `SUM`/`AVG` 梯度因子已由两进程 NCCL 实验确认。
3. official-code-faithful 适配路径及完整 SIM 双卡 smoke 已完成。
4. 运行跨过 adaptive warmup 的中程缩放复现，并保留 5k/10k checkpoint。
5. 对 baseline、5k 和 10k 做同协议 paired 评估，避免只看训练内 FD 曲线。
6. baseline 可靠后，再对 selective amplification、pooled whitening 等改进做因果消融。
