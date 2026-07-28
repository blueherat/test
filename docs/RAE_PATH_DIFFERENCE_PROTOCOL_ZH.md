# RAE 生成路径差分方向：机制筛查协议

## 研究问题

四条 stage-2 路径在同一 noise、ImageNet label 和 sample index 下产生不同 endpoint，
且已有独立 5k FID 排序：

```text
static 123.53 < random 128.99 < annealed 143.54 < reverse 159.05
```

本实验不训练任何模型，只检验逐样本路径差分

```text
delta_ab(i) = z_b(i) - z_a(i)
```

是否是一个具有样本特异性、剂量响应和可逆修正作用的图像级方向。它不是 clean
manifold 法向，也不因两个 endpoint 质量不同而自动成为“质量方向”。

## 证据边界

当前 256 个 endpoint 已在旧实验中用于路径级 FID、cycle 和 decoder 诊断，因此本轮
只能叫机制筛查。固定对照和门槛是在打开本实验 held-out sample-level 结果前写定；
若通过，只授权使用新 sampling seed 的确认实验，不直接授权训练或论文结论。

## 数据划分

- 路径对：`static->reverse`、`static->random`、`random->annealed`、
  `annealed->reverse`。
- calibration：endpoint indices `[0,128)`，只拟合每个路径对的全局平均差分。
- held-out test：endpoint indices `[128,256)`，一次性报告全部结果。
- clean feature reference：ImageNet validation latent cache 的 128 个 decoded-clean
  samples，与 endpoint 数据无配对关系，只用于小样本分布 proxy。
- frozen 官方 RAE-DINOv2 ViT-XL decoder 和 torchvision Inception-v3；fp32，关闭 TF32。

## 固定干预

own 插值使用不调参网格：

```text
z(alpha) = z_good + alpha * delta, alpha in {0,.25,.5,.75,1}
```

在 `alpha=.25` 的 good anchor 和 bad anchor 上加入等逐样本 RMS 对照：

- `own`：当前样本差分；
- `shuffled`：另一测试样本差分；
- `random`：Gaussian 方向；
- `global`：只在 calibration 拟合的平均差分；
- `opposite`：反方向。

此外把完整差分精确拆成 `token_mean + spatial_residual`，以自然能量而不是人为
RMS matching 测量两部分各自能解释多少 decoded feature 变化。

## 指标

逐样本主指标是 Inception pre-logit feature progress：

```text
<f(z_candidate)-f(z_good), f(z_bad)-f(z_good)>
-------------------------------------------------
             ||f(z_bad)-f(z_good)||^2
```

它只测量 decoded feature 是否沿 paired endpoint chord 前进，不等价于质量。
同时报告 target-label NLL/top-1、image/feature delta、feature cosine 和 clipping。

分布 proxy 使用相同固定 Inception features：64 维固定随机投影 Frechet 与 128 个
固定方向的 standardized SWD。它们是小样本筛查量，不是 ADM-FID。首先必须证明
它们在四个原始 endpoint 上与已知 5k FID 排序一致，否则停止质量解释。

## Fresh-seed 授权门槛

必须全部满足：

1. projected Frechet 对四个 endpoint 的 5k FID Spearman `>=0.80`；
2. 至少 3/4 路径对的 own alpha-dose Spearman `>=0.80`，且 bad-good proxy gap 为正；
3. 从 bad endpoint 沿 own 方向修正 25% 时，至少 3/4 路径恢复完整 gap 的 `>=20%`，
   且比 shuffled/random 中较好的对照多恢复 `>=10%` full gap；
4. 至少 3/4 路径对在 good anchor 的 own feature progress 比 shuffled/random 中较高者
   高 `>=0.10`。

任一主门槛失败，就不增加 alpha、不改 primary proxy、不改路径子集来挽救结论。
token mean/spatial residual 只作机制定位，不参与通过判定。

## 主结果后增加的鉴别对照

看到普通线性插值的中间点系统性坏化后，额外固定了两个 post-hoc 对照：

- RMS-preserving lerp：保持每个样本由两端 RMS 线性插值得到的目标半径；
- spherical interpolation：在每个样本的 flattened latent 球面上插值，同时线性插值
  两端半径。

它们只用于判断普通 lerp 的向量抵消和半径收缩能解释多少中点坏化，不参与上面的
fresh-seed 授权 gate，也不允许回头修改主实验门槛。
