# AdvFD Paper-Only Clean-Room：阶段 A 结果

日期：2026-08-23

## 1. 本阶段做了什么

本阶段没有训练图像生成器，也没有查看官方 AdvFD 实现。我们先独立实现并验证：

- population mean、second moment 与 covariance；
- Gaussian Fréchet/Bures distance 的均值项和协方差项；
- FD-Loss 风格 EMA 矩，历史状态 detach、当前 batch 保留梯度；
- raw、real-whitened、pooled-whitened 三种 feature calibration；
- 对连续图像保留梯度、同时与 `torch-fidelity` 数值一致的 Inception-2048。

随后固定真假样本，训练一个小 MLP feature critic 最大化 FD。每种条件使用完全相同的
样本、初始化、优化步数和梯度裁剪。

实验包含两个数据 regime：

- `matched`：real/fake 是从同一个八模态环分布独立采样；总体差异为零，用于检测经验
  critic overfitting；
- `shift`：fake 有固定角度和半径偏移；存在真实总体差异，用于检测校准是否抹掉有效
  判别力。

每个 regime 比较：

- `none`：raw adaptive FD；
- `real`：AdvFD 式真实特征 whitening；
- `pooled`：Fisher 式真假混合 whitening 机制对照。

正式配置为 3 seeds、每个 critic 1000 steps、256 个固定训练样本和 4096 个独立
held-out 样本。

## 2. 数学与单元测试

共 16 项独立测试通过：

- clean-room FD/EMA/calibration：13 项；
- differentiable Inception：3 项。

其中数值验证包括：

1. raw FD 在公共特征缩放 \(c\) 下严格乘 \(c^2\)；
2. exact real/pooled whitening 对公共可逆仿射变换保持 FD；
3. real-whitened scalar mean supremum 等于 Pearson \(\chi^2(q\|p)\)；
4. pooled scalar supremum 等于 Fisher GAN 的对称 \(\chi^2\) 距离；
5. 支持分离时 real-only 目标可无界，而 pooled Rayleigh quotient 保持有限；
6. differentiable Inception 与 `torch-fidelity` uint8 路径在容差内一致，并能向输入像素
   反向传播。

## 3. 三 seed 正式结果

下表为 seed 均值。`raw fake RMS` 是校准前 feature RMS。

| regime | calibration | train FD | held-out FD | raw fake RMS | calibrated fake RMS | effective rank |
|---|---:|---:|---:|---:|---:|---:|
| matched | raw | \(1.4005\times10^{11}\) | \(2.7371\times10^9\) | \(1.9940\times10^6\) | \(1.9940\times10^6\) | 1.00 |
| matched | real | 0.3225 | 0.01935 | 61.00 | 0.800 | 8.48 |
| matched | pooled | 0.1346 | 0.01020 | 63.05 | 0.773 | 8.40 |
| shift | raw | \(8.6641\times10^{11}\) | \(1.1959\times10^{11}\) | \(2.1805\times10^6\) | \(2.1805\times10^6\) | 1.00 |
| shift | real | 1.3373 | 1.2043 | 162.33 | 0.714 | 4.34 |
| shift | pooled | 1.0068 | 0.9513 | 124.81 | 0.622 | 4.23 |

结果文件：

- `/data/users/zhoushunyu/eqvae/experiments/advfd_cleanroom_critic_toy_v2/metrics.csv`
- `/data/users/zhoushunyu/eqvae/experiments/advfd_cleanroom_critic_toy_v2/summary.csv`
- `/data/users/zhoushunyu/eqvae/experiments/advfd_cleanroom_critic_toy_v2/critic_generalization.png`

旧版输出把同分布 regime 命名为字符串 `null`，Pandas 默认会将其解析为缺失值。旧数据
没有删除；v2 将其改名为 `matched` 后完整重跑，避免后续报告误读。

## 4. 当前可以确认的现象

### 4.1 raw adaptive FD 的缩放退化被直接复现

raw critic 的 feature RMS 从初始化约 \(10^{-1}\) 增长到约 \(2\times10^6\)，FD
增长到 \(10^{10}\)--\(10^{12}\)，并且有效秩塌缩到约 1。即使 `matched` 的总体
差异为零，held-out FD 也会因极端尺度而变得巨大。

所以 raw adaptive FD 的数值增加完全不能解释成发现了真实分布差异。

### 4.2 whitening 解决尺度坐标，但没有解决 adaptive overfitting

real 与 pooled calibration 都把优化目标保持在 \(10^{-2}\)--\(10^0\) 量级，并保留
多维特征。它们确实阻断了 raw objective 的直接尺度通道。

但在 `matched` 条件下：

\[
\text{real: }0.3225\ \text{train}\quad\text{vs}\quad0.01935\ \text{heldout},
\]

\[
\text{pooled: }0.1346\ \text{train}\quad\text{vs}\quad0.01020\ \text{heldout}.
\]

两者都能适配固定训练样本的经验差异，而这种差异绝大部分不泛化。由此得到明确区分：

\[
\boxed{\text{affine calibration}\neq\text{adaptive critic generalization}.}
\]

### 4.3 pooled calibration 没有抹掉真实分布差异

在 `shift` 条件下，real 与 pooled 的 held-out FD 都保持约 1 的量级，且 train 与
held-out 接近得多。这说明 pooled constraint 至少在该 toy 上不是简单地把 critic
压平。

### 4.4 尚不能宣称 pooled 优于 real

matched 条件下 pooled 的平均 train/held-out gap 小于 real；shift 条件下 pooled 的
均值和方差也稍低。但只有三 seed、小 MLP 和单一二维分布，证据不足以支持真实图像
方法结论。目前 pooled 的最强依据仍是 support-safe 的解析理论，而非生成质量实证。

## 5. 对 AdvFD 复现的直接影响

真实模型 pilot 除论文指标外必须额外记录：

- D-step 训练 adaptive FD 与独立 held-out adaptive FD；
- mean/covariance 两个分量；
- 校准前/后的 real/fake feature RMS；
- covariance spectrum/effective rank；
- critic LR、更新次数与 train-heldout gap；
- raw、real-whitened 必要对照。

这些量只用于诊断，不改变论文 AdvFD 的优化目标。下一阶段先用论文正式的小骨架
`pMF-B/16` 做 Inception-only pilot，确认显存、吞吐、warm-start 和 G/D 梯度链；通过后
再扩到论文的 SIM 静态目标和正式规模。
