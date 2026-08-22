# SiT 弱头差值与 x800 冻结读出实验

本记录只整理实验配置和结果。所有 FID 均为 ImageNet-100、1,000 个配对样本、EMA 推理、无 CFG；同一扫描中的初始噪声与类别标签完全相同。

## 实验一：两个弱头的差值

冻结 `v800`、`depth4-v` 和 `depth8-v`，采样速度场为：

\[
v_{\gamma}=v_{800}+\gamma\left(v_{\mathrm{depth8}}-v_{\mathrm{depth4}}\right).
\]

共评估 13 个系数。正式细扫结果为：

| 设置 | gamma | FID | sFID | IS |
|---|---:|---:|---:|---:|
| 基线 | 0.00 | 84.9687 | 214.6226 | 26.5766 |
| 最优 | 0.65 | **69.9448** | 213.6398 | 35.1398 |

FID 改善为 `15.0240`。这里的正式结果来自 `weak_head_difference_fid1k.csv`；综合粗扫表中还保留了较早的 9 点扫描，不应以其中的 `gamma=1` 作为最终最优点。

## 实验二：冻结 x800，训练 depth4 弱头

冻结 `x800` EMA 主模型及其 backbone，在 depth 4 接入与官方 SiT 相同形式的 `FinalLayer`。分别用原生 velocity、clean 和 epsilon 目标训练三个独立弱头，每个训练 50,000 step；global batch 为 256，训练 seed 为 0。采样时先把三个弱头统一转换到 velocity space，再使用：

\[
v_{\gamma}=v_{x800}+\gamma\left(v_{x800}-v_{\mathrm{depth4,target}}\right).
\]

每个目标均评估相同的 13 个系数：

`-0.5, 0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.6, 1, 1.5, 2`。

| 弱头目标 | 最优 gamma | 基线 FID | 最优 FID | FID 改善 | 最优 sFID | 最优 IS |
|---|---:|---:|---:|---:|---:|---:|
| velocity | 0.35 | 87.3822 | **70.2712** | 17.1110 | 213.3670 | 32.4667 |
| epsilon | 0.20 | 87.3787 | **71.9081** | 15.4706 | 215.6869 | 33.2581 |
| clean | 0.25 | 87.3792 | **73.7732** | 13.6060 | 214.5166 | 30.7066 |

三个独立扫描的所有条件均具有相同的采样哈希：

- noise SHA256 前缀：`ab8419c7fdfd`
- label SHA256 前缀：`7c3ae6894e7e`

## 50K 验证误差

| 目标 | raw native MSE | raw velocity MSE | EMA velocity MSE | 冻结源模型 velocity MSE | raw gap RMS |
|---|---:|---:|---:|---:|---:|
| velocity | 0.919727 | 0.919727 | 0.920182 | 0.814685 | 0.328843 |
| clean | 0.246832 | 1.589093 | 1.602413 | 0.814685 | 0.490841 |
| epsilon | 0.359391 | 1.579744 | 1.599773 | 0.814685 | 0.617215 |

## 数据文件

便携数据、训练日志、协议和曲线图位于：

`docs/data/imagenet100_sit_weak_difference_x800_readouts_v1/`

该目录不包含 checkpoint、生成样本、ImageNet 或 latent cache。
