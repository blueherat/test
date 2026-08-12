# ImageNet-100 SiT FID 实验总表

更新时间：2026-08-12

## 1. 文档目的

本文档统一登记当前机器上已经完成的 ImageNet-100 SiT 正式 FID 实验，避免把不同
prediction target、loss、训练步数或双头采样路径混在一起比较。

本次审计递归检查了：

```text
/home/zhoushunyu/data/eqvae/imagenet_sit_flow/**/fid5k_adm_results.json
/home/zhoushunyu/data/eqvae/imagenet_sit_flow/**/sampling_manifest.json
```

共找到 38 份 FID artifact。它们对应 37 个不同实验条件；额外一份是同一个 800K
velocity checkpoint 的低显存复评。

## 2. 共同评估协议

全部表格只包含满足以下条件的正式结果：

- 数据集：ImageNet-100；
- 图像分辨率：256x256；
- tokenizer/decoder：Stable Diffusion VAE；
- 生成模型：SiT-S/2；
- 权重：EMA；
- 样本数：5000；
- 训练与采样 seed：0；
- `cfg_scale=1.0`，不启用 classifier-free guidance；
- sampler：官方 SiT Dopri5；
- reference：固定的 ImageNet-100 validation 5000 张图；
- evaluator：ADM TensorFlow evaluator；
- FID 和 sFID 越低越好，IS 越高越好。

这些数字是同一协议内的 FID-5K 筛查结果，不是论文常用的 ImageNet-1K FID-50K。
它们适合内部横向比较，不能与官方 SiT 数字直接比较绝对值。目前所有结果都只有一个
训练 seed，也不能据此声称统计显著性。

## 3. 原生 velocity 单头基线

训练目标为线性 flow 的原生速度：

```text
x_t = (1 - t) * epsilon + t * x
v   = x - epsilon
```

| step | FID | sFID | IS |
|---:|---:|---:|---:|
| 50K | 142.6490 | 73.6372 | 11.3228 |
| 60K | 130.7967 | 73.1066 | 12.4126 |
| 80K | 116.0442 | 72.4244 | 14.3577 |
| 100K | 106.1064 | 71.0320 | 16.0777 |
| 120K | 99.4306 | 70.1554 | 17.4834 |
| 180K | 86.9341 | 69.4012 | 20.3537 |
| 240K | 79.9044 | 69.0897 | 22.6320 |
| 300K | 73.9829 | 68.7939 | 24.3396 |
| 400K | 68.6537 | 68.8580 | 26.3078 |
| 500K | 64.6750 | 68.6517 | 27.9932 |
| 600K | 62.4233 | 68.6669 | 28.6254 |
| 700K | 68.5173 | 76.8913 | 24.9817 |
| 800K | 61.0080 | 68.9874 | 30.2009 |
| 800K，低显存复评 | **61.0016** | 68.9871 | 30.2042 |

700K 同时出现 FID、sFID 和 IS 的反向变化，是整条曲线中的异常点。在重新采样复核前，
不应使用它推断训练动力学。800K 的两次评估使用同一 checkpoint，FID 仅相差
`0.0064`，说明 800K 的约 `61.00` 结果可重复；它们不能算作两个独立 seed。

原始结果目录：

```text
/home/zhoushunyu/data/eqvae/imagenet_sit_flow/fid5k/
/home/zhoushunyu/data/eqvae/imagenet_sit_flow/fid5k_lowmem_v1/
```

## 4. 单头 prediction-target 对照

`native` 表示直接回归对应 target；`velocity` 表示网络输出 target 后，将 prediction 和
真实 native target 使用同一个公式转换到 velocity space 再计算 loss。

| prediction target / loss | 100K FID | 200K FID | 300K FID | 400K FID |
|---|---:|---:|---:|---:|
| `velocity / velocity` | 106.1064 | 未评估 | 73.9829 | **68.6537** |
| `epsilon / native` | **104.7138** | **87.2789** | 未训练 | 未训练 |
| `x / native` | 118.6746 | 99.4994 | 未训练 | 未训练 |
| `x / velocity`，floor=`0.05` | 122.7828 | 94.3210 | 83.6212 | 77.6033 |

完整指标如下。

| prediction target | loss | step | FID | sFID | IS |
|---|---|---:|---:|---:|---:|
| epsilon | native | 100K | 104.7138 | 69.2893 | 15.6712 |
| epsilon | native | 200K | 87.2789 | 67.3440 | 20.0699 |
| x | native | 100K | 118.6746 | 68.7453 | 14.6032 |
| x | native | 200K | 99.4994 | 70.3495 | 17.8818 |
| x | velocity，floor=`0.05` | 100K | 122.7828 | 71.8471 | 14.5650 |
| x | velocity，floor=`0.05` | 200K | 94.3210 | 69.7244 | 19.4958 |
| x | velocity，floor=`0.05` | 300K | 83.6212 | 69.3388 | 22.1621 |
| x | velocity，floor=`0.05` | 400K | 77.6033 | 69.2131 | 24.2136 |

当前证据只支持以下事实：在已经匹配的 checkpoint 上，原生 velocity 仍然最好。
JiT-style `x/velocity` 从 100K 到 400K 持续改善，但 400K FID 仍比 velocity 400K 高
`8.9495`。这里保留 SiT 的均匀时间采样，因此该实验不是完整 JiT recipe。

原始结果目录：

```text
/home/zhoushunyu/data/eqvae/imagenet_sit_flow/fid5k_single-target_epsilon-native_seed0/
/home/zhoushunyu/data/eqvae/imagenet_sit_flow/fid5k_single-target_x-native_seed0/
/home/zhoushunyu/data/eqvae/imagenet_sit_flow/fid5k_single-target_x-velocity-floor0p05_seed0/
```

## 5. Dynamic Dual-Output 对照

该模型共享一个 SiT-S/2 trunk，同时输出 clean `x`、`epsilon` 和动态 gate。它是在
continuous linear flow 上进行的 DDO-inspired adaptation，不是 CVPR 2022 离散
VP-DDPM/DDIM 方法的逐式复现。

| checkpoint | 采样路径 | FID | sFID | IS |
|---:|---|---:|---:|---:|
| 400K | learned dynamic gate | 73.8274 | 69.0210 | 24.2918 |
| 450K | only epsilon head | **70.7303** | **68.4541** | 24.3442 |
| 450K | learned dynamic gate | 71.7942 | 68.9646 | 24.7589 |
| 450K | only x head | 72.8160 | 69.6784 | **24.8607** |

在同一个 450K checkpoint 内，dynamic 比 x head 好 `1.0218` FID，但比 epsilon head
差 `1.0638`。同为 400K 时，dynamic 比原生 velocity 单头差 `5.1737` FID。当前结果
不支持“learned dynamic gate 能直接改善 continuous SiT rollout”。

## 6. 双头静态混合与外推

450K 双头 checkpoint 的静态速度场定义为：

```text
v_static = v_epsilon + scale * (v_x - v_epsilon)
```

因此 `scale=0` 是纯 epsilon，`scale=1` 是纯 x，`0<scale<1` 是内插，区间外是外推。
除特别标出的 `override` 外，表中均使用 `raw` endpoint mode。

| scale / 路径 | FID | sFID | IS |
|---:|---:|---:|---:|
| -0.50 | 71.1474 | 68.0082 | 24.2385 |
| -0.40 | 70.6520 | **67.9923** | 24.3512 |
| -0.30 | 70.4083 | 68.0015 | 24.5337 |
| -0.25 | 70.3015 | 68.0420 | 24.5420 |
| -0.20，raw | **70.2396** | 68.0884 | 24.5460 |
| -0.20，endpoint override | 70.3665 | 68.2234 | 24.3319 |
| -0.10 | 70.3065 | 68.1703 | 24.5214 |
| 0.00，纯 epsilon | 70.7303 | 68.4541 | 24.3442 |
| 0.25 | 70.5681 | 68.4853 | 24.5857 |
| 0.50 | 70.9939 | 68.8186 | 24.8260 |
| 0.75 | 71.8878 | 69.2115 | 24.8303 |
| 1.00，纯 x | 72.8160 | 69.6784 | **24.8607** |
| 1.25 | 73.8951 | 70.1592 | 24.4573 |
| 1.50 | 75.5002 | 70.9125 | 24.0158 |
| learned dynamic gate | 71.7942 | 68.9646 | 24.7589 |

最佳静态 FID 是 `scale=-0.20, raw` 的 `70.2396`，比纯 epsilon 低 `0.4907`，也比
learned dynamic gate 低 `1.5545`。这只是同一 checkpoint、单 seed 上的小幅筛查结果；
在复评和多 seed 之前，不能把它表述为稳定方法增益。

原始结果目录：

```text
/home/zhoushunyu/data/eqvae/imagenet_sit_flow/fid5k_dual-output_step400000_seed0/
/home/zhoushunyu/data/eqvae/imagenet_sit_flow/fid5k_dual-output_step450000_seed0/
```

## 7. 当前汇总结论

1. 当前所有已评估 SiT 条件中，最低 FID 是原生 velocity 单头 800K 的低显存复评：
   `61.0016`。
2. 在已经直接匹配的训练步数上，原生 velocity 单头优于 `x/native`、
   `epsilon/native`、JiT-style `x/velocity` 和当前双头模型。
3. 双头 checkpoint 内部，epsilon head 优于 learned dynamic gate，dynamic 又优于 x
   head；这与 DDO 原论文中 x 分支明显强于 epsilon 分支的工作区间不同。
4. 静态向 epsilon 外侧轻微外推存在一个局部最优点，但增益不足以脱离 FID-5K 和单
   seed 的测量不确定性。
5. 700K velocity 结果是未解释异常点；任何训练曲线分析都应先复评该 checkpoint。

## 8. 数据完整性与使用边界

- 38 份结果的 manifest 均确认：EMA、5000 samples、seed 0、SiT-S/2、CFG 关闭且使用
  同一个 reference；FID 数值均为有限正数。
- 800K 两份 artifact 使用同一个 checkpoint SHA256，只是显存与 FID batch 流程不同。
- 后期低显存流程和双头流程带有资源审计；早期 baseline artifact 主要依赖 sampling
  manifest 和 checkpoint SHA256。800K 的跨流程复评支持 evaluator 一致性，但不能
  自动替代所有早期 checkpoint 的逐点复评。
- smoke、预览图、没有 ADM-FID 结果的采样目录均未收入本表。
- Git 只保存本报告和代码，不保存 checkpoint、NPZ、数据集或 FID feature 文件。

进一步的方法结论应至少补充多 seed，并把候选 checkpoint 升级到 50K samples 后再进入
论文主表。
