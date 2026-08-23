# AdvFD 论文优先复现：歧义与预先决策

冻结日期：2026-08-23

本文件在解封官方 AdvFD 实现前写成。每一项都必须在首次 paper-only 结果完成后再与
官方实现比较，不根据官方实现反向修改“独立版本”的历史结果。

| 编号 | 论文歧义 | paper-only 预先决策 | 理由/待做对照 |
|---|---|---|---|
| A01 | Eq. (2)/(9) 写 raw \(D_{adv}\)，附录统一介绍 normalized FD loss，但未明确 adaptive branch 是否归一化 | static 每个表示和 adaptive FD 都分别用 \(L/(\mathrm{sg}(L)+0.01)\)，再乘 \(\lambda_{adv}\) | 延续 FD-Loss 对不同表示做 unit-scale calibration 的原则；后续补 raw-adaptive-normalization 消融 |
| A02 | “real-feature gradients detached”可能指 real tensor、real moments 或 whitener | D-step 中完整 real feature tensor detach；real moments 和 whitener也 detach。G-step 中所有 encoder 参数冻结，生成图像梯度保留 | 与算法中 real reference 的语义一致；若只 detach moments，real branch 仍给 critic 梯度，是另一种游戏 |
| A03 | D-step 的生成图是否复用 G-step 图像 | 使用同一 noise/class，在 G-step 参数更新后重新生成并 detach | Algorithm 1 line 11 明确写 \(G_{\theta_{t+1}}(Z)\) |
| A04 | adaptive EMA 在一次 G-step+D-step 中更新几次 | 每个 generator step 更新一次 adaptive real/fake EMA；D-step 复用该状态，不再次推进 EMA 时间 | 避免 D-step 频率暗中改变统计时间常数 |
| A05 | adaptive EMA 是否 warm-start 50k | 在 critic 的 pretrained 初始表示下，用 50k real 和 50k base-generated features 初始化 | 论文明确所有 feature statistics 从 50k base samples 初始化，但未单列 adaptive 分支 |
| A06 | real whitening 使用当前 batch 还是 EMA real moments | 使用 adaptive real EMA \(\beta=0.99\) | 论文说 FD-Adv feature statistics 使用独立 EMA |
| A07 | covariance 的分母是 \(N\) 还是 \(N-1\) | 使用 population moment \(M-\mu\mu^\top\)，即分母 \(N\) | 论文以分布矩和 batch raw second moment定义，不是无偏样本统计 |
| A08 | 多卡上先算 local moments 再平均，还是先 gather feature | 先 gather 全局 features，再算 mean/covariance | 论文明确 generated features gathered across GPUs before moments |
| A09 | differentiable encoder 输入是否量化到 uint8 | real 图来自 uint8 数据后转 [0,1]；生成图保持连续值，不量化；两者按 encoder 目标分辨率可微 resize | uint8 转换会切断生成器梯度；连续路径保持 torch-fidelity 的权重与归一化 |
| A10 | 生成图超出 [-1,1] 后如何处理 | 转 [0,1] 后 hard clamp | encoder 输入域要求；同时记录 clamp 比例，检查是否造成大面积零梯度 |
| A11 | ImageNet real reference 是否做 horizontal flip | 固定真实 reference stats 只 center crop；训练时用于 adaptive critic 的 real batch 做随机 horizontal flip | 论文 B.3 对 reference 只写 center crop/tensor conversion，训练 augmentation 另列 flip |
| A12 | adaptive Inception 取哪个输出 | 先使用与 FD-Inception 一致的 global-average-pool 2048 feature | 论文称 pretrained Inception representation，但没有单独给 adaptive output dimension；这是最需要官方比对的选择之一 |
| A13 | adaptive Inception 的 BN/dropout 状态 | 始终 eval-mode layer behavior，但参数可训练 | 保持 pretrained representation 的确定性统计；不让 BN running state成为未声明的第三套动态量 |
| A14 | G-step 与 D-step 交替的精确顺序 | 每步先 G，满足频率且过 start step 后再 D | 与 Algorithm 1 顺序一致 |
| A15 | `every 2 generator steps` 的奇偶定义 | 在完成第 2、4、6... 个 G-step 后执行 D-step | 从 1 开始计数最直观；写入 checkpoint 保持续训一致 |
| A16 | adv start/warmup 是否同时控制 D-step 和 G 权重 | step <1000 不做 D-step且 \(\lambda_{adv}=0\)；1000--5000 对 G 的 \(\lambda\) 线性 warmup，D-step 从 1000 开始 full critic LR | 论文只明确 adversarial term warmup，critic LR 不做额外 warmup |
| A17 | pMF-B 训练/采样条件 | 1 NFE、CFG 8.5、interval [0.1,0.7]、noise scale 1.0、类别均匀采样 | AdvFD Appendix E.2 明确给出 |
| A18 | generator model EMA 用于训练损失还是只跟踪 | loss 和正式 released-checkpoint comparison 用 online 参数；EDM EMA 只并行跟踪、作为额外诊断 | 论文明确最终 released evaluation 用 online generator；当前缩放 pilot 尚未实现被动 EMA 跟踪，因而不影响 online 动力学/FID，但 formal reproduction 必须补齐 |
| A19 | full matrix square root 的数值算法 | 对称化后 `torch.linalg.eigh/eigvalsh`，负特征值截到 0；训练加论文规定的 calibration epsilon，不额外给 FD covariance 加未声明 jitter | 论文指定 symmetric eigendecomposition；额外 jitter 需单独消融 |
| A20 | static SIM 三项如何组合 | SigLIP2、Inception、MAE 各自 normalized FD，权重均为 1 后相加 | 论文明确三个 term unit weight；不能先相加 raw FD 再统一归一化 |
| A21 | 同一轮 G-step/D-step 是否把 FD-Adv EMA 时间推进两次 | G 和 D 都从同一个上一步已提交状态预览当前统计；每个 generator step 最多提交一次。有 D-step 时提交更新后生成器对应的批统计，否则提交 G-step 统计 | 论文只说 separate EMA，不定义交替更新语义；该选择避免 critic update frequency 暗中改变 EMA 的有效时间常数 |
| A22 | FD-Adv 在 start step 前是否继续更新其 feature EMA | 从 step 0 起更新 adaptive real/fake EMA，但在 start step 前不更新 critic、且 generator 权重为 0 | 让 start 时的 adaptive 统计匹配当前 generator，避免突然使用陈旧 base moments；论文只明确 adversarial term 的 start/warm-up，不明确统计更新是否延后 |
| A23 | AdvFD 表格未重列数值精度，但其 static recipe 继承 FD-Loss | 生成器/encoder 前向使用 FD-Loss 明确指定的 BF16；矩、协方差、eigendecomposition 和 optimizer state 使用 FP32；正式 checkpoint 额外补 pMF 官方 FP32 评估 | FD-Loss Appendix B.2 明确写 BF16；训练精度不再视为完全未知，但 BF16 FID 仍不能冒充 pMF 官方 FP32 数值 |
| A24 | 论文只写 50K evaluation，未列 class/noise 枚举顺序 | 遵循公开 pMF evaluator：类别按 class-major 均衡；第 (i) 张图使用独立 CPU RNG，seed 为 `i XOR initial_seed` | 已逐值验证 clean-room noise helper 与 pMF `BatchGenerator` 完全一致；使结果不依赖 eval batch 切分 |
| A25 | AdvFD 写生成图转到 `[0,1]`，未明确是否量化；pMF evaluator 会写 uint8 PNG | 训练 feature 始终使用连续 `[0,1]` 图以保留梯度；正式 FID 对生成图执行 `round(255x)/255`，另保留连续评估作为诊断 | 量化只能用于评估，放进训练会切断梯度；正式值应匹配生成器原发布协议 |
| A26 | FD-Loss 表 B.1 把 `vit_so400m_patch16_siglip_256.v2_webli` 写成 CLS pooling，但该精确 timm 架构没有 CLS/prefix token，默认使用 MAP attention pooling | paper-only formal SIM 使用 `forward_head(forward_features(x), pre_logits=True)`；MAE 因而取 normalized CLS，SigLIP2 取其架构定义的 MAP pooled feature；绝不把第一个 patch token 当 CLS | 这是论文规格内部不一致；在解封实现前保留为关键歧义，并分别记录 model identifier、`num_prefix_tokens=0` 与 pooling 选择 |

## 分级实验中的有意缩放

下面这些只用于 smoke/pilot，不能报告为论文数值复现：

- Inception feature 固定随机正交投影到 64/128 维；
- warm-start 少于 50k；
- 单卡或较小 global batch；
- 少于 125k generator steps；
- Inception-only static loss，而非完整 SIM。

每个产物必须包含 `paper_reproduction_metric=false`。只有恢复完整特征维数、论文统计量、
global batch、warm-start、训练长度和评估协议后，才允许标为 full numerical
reproduction。
