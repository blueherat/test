# ImageNet-100 SiT-v 与 JiT-x 静态内插/外推报告

更新时间：2026-08-13

## 1. 结论摘要

本实验比较两个独立训练到 400K step 的 `SiT-S/2` 单头模型：原生 velocity
prediction（下文简称 SiT-v）与 JiT-style `x` prediction。采样时不重新训练模型，
而是在 ODE 的每一次网络求值处混合两者转换后的速度场：

```text
v_s = v_SiT-v + s * (v_JiT-x - v_SiT-v)
```

在本次 15 点、单 seed、FID-5K 配对扫描中，FID 随 `s` 从 `-1.0` 增加到 `1.5`
严格上升：

- `s=0` 的纯 SiT-v：FID `67.4735`；
- `s=1` 的纯 JiT-x：FID `75.9490`；
- 从 SiT-v 朝 JiT-x 内插会持续恶化；
- 越过 JiT-x 外推仍继续恶化；
- 沿相反方向越过 SiT-v，FID 持续改善；
- 扫描内最佳点为 `s=-1.0`，FID `59.0766`，比配对 SiT-v 低 `8.3969`
  FID，下降 `12.44%`。

这是一个强且内部一致的现象，但还不是最终方法结论。最优点位于扫描边界，说明负方向
曲线尚未闭合；实验也只有一个训练 seed 和 5000 张生成图，尚未报告 precision/recall，
因此不能据此声称该方向在不同模型、训练阶段或大样本 FID 上稳定有效。

## 2. scale 的准确含义

两个端点先各自输出 prediction，再按照其训练语义转换为同一 velocity space：

```text
SiT-v:  网络直接输出 velocity
JiT-x:  网络输出 clean x，再转换成 velocity
```

随后在采样轨迹的每个时间点使用：

```text
v_s(x_t, t) = v_v(x_t, t) + s * (v_x(x_t, t) - v_v(x_t, t))
```

因此：

| scale 区间 | 含义 |
|---|---|
| `s=0` | 纯 SiT-v |
| `s=1` | 纯 JiT-x |
| `0<s<1` | 从 SiT-v 到 JiT-x 的凸内插 |
| `s<0` | 从 JiT-x 反方向越过 SiT-v |
| `s>1` | 从 SiT-v 方向越过 JiT-x |

这里混合的是整条 ODE 轨迹上的速度场，不是对两组最终图像做像素混合，也不是对最终
latent 做一次线性组合。

## 3. 数据与模型协议

### 数据

- 数据集：ImageNet-100；
- 分辨率：`256x256`；
- 训练集：126,689 张，覆盖 100 类；
- 验证 reference：5000 张，100 类各 50 张；
- tokenizer/decoder：`stabilityai/sd-vae-ft-mse`；
- latent moments：`[8,32,32]`，前 4 通道为 mean，后 4 通道为 standard
  deviation；训练时采样得到 `[4,32,32]` latent；
- VAE scaling factor：`0.18215`；
- 公开 latent cache 使用 center crop，不包含在线 random horizontal flip。

### 两个 checkpoint

| 项目 | SiT-v | JiT-x |
|---|---|---|
| 模型 | `SiT-S/2` | `SiT-S/2` |
| step | 400K | 400K |
| prediction target | velocity | clean `x` |
| loss space | velocity | velocity |
| conversion floor | `0.001` | `0.05` |
| global batch | 256 | 256 |
| seed | 0 | 0 |
| 训练 world size | 4 | 2 |
| 采样权重 | EMA | EMA |

JiT-x 保留 SiT 的线性 flow、均匀时间采样和其余训练协议，只采用 `x` 输出并在
velocity space 计算 loss，因此它是 JiT-style prediction-target control，不是完整 JiT
训练配方。

### 采样与指标

- 每个 scale 生成 5000 张图；
- class-conditioned，但 `cfg_scale=1.0`，不启用 classifier-free guidance；
- 类别通过与官方 SiT 一致的 `torch.randint` 随机抽取，因此单次 5K 生成的类别计数
  只近似均衡；
- sampler：官方 SiT linear-path Dopri5；
- 采样精度：FP32，允许 TF32；
- evaluator：ADM TensorFlow FID evaluator；
- 所有 scale 共用相同的初始噪声和类别标签；
- 所有 scale 共用同一份 ImageNet-100 validation 5K reference。

这使 scale 之间成为严格配对比较，显著减少了不同随机样本带来的横向比较噪声。

## 4. 完整结果

`Delta FID` 以本轮配对采样中的纯 SiT-v（`s=0`）为基准；负值表示改善。FID 和
sFID 越低越好，IS 越高越好。

| scale | 区域 | FID | Delta FID | sFID | IS |
|---:|---|---:|---:|---:|---:|
| -1.00 | 越过 SiT-v | **59.0766** | **-8.3969** | **67.6016** | 27.3436 |
| -0.75 | 越过 SiT-v | 60.6991 | -6.7744 | 67.6460 | **27.5308** |
| -0.50 | 越过 SiT-v | 62.8293 | -4.6442 | 67.8832 | 27.2172 |
| -0.30 | 越过 SiT-v | 64.4349 | -3.0386 | 68.1291 | 26.8944 |
| -0.20 | 越过 SiT-v | 65.2661 | -2.2074 | 68.3305 | 26.9170 |
| -0.10 | 越过 SiT-v | 66.2498 | -1.2237 | 68.5542 | 26.9688 |
| -0.05 | 越过 SiT-v | 66.8738 | -0.5997 | 68.7252 | 26.8646 |
| 0.00 | 纯 SiT-v | 67.4735 | 0.0000 | 68.8711 | 26.6705 |
| 0.10 | 内插 | 68.1899 | +0.7164 | 68.9454 | 26.3105 |
| 0.25 | 内插 | 69.6031 | +2.1296 | 69.2619 | 25.5868 |
| 0.50 | 内插 | 71.9368 | +4.4633 | 69.5818 | 24.8997 |
| 0.75 | 内插 | 74.2443 | +6.7707 | 69.8115 | 24.4043 |
| 1.00 | 纯 JiT-x | 75.9490 | +8.4755 | 69.8197 | 23.9173 |
| 1.25 | 越过 JiT-x | 77.4831 | +10.0096 | 69.6494 | 23.3673 |
| 1.50 | 越过 JiT-x | 79.0005 | +11.5269 | 69.3721 | 22.9250 |

便携原始表见
[`docs/data/imagenet100_sit_v_jit_x_static_sweep_400k.csv`](data/imagenet100_sit_v_jit_x_static_sweep_400k.csv)。

## 5. 当前能确认什么

### 5.1 JiT-x 不是本轮的强端点

纯 JiT-x 比纯 SiT-v 高 `8.4755` FID。当前模型与训练预算下，不能把 JiT-x 当作
strong predictor，再从 SiT-v 朝 JiT-x 外推；数据明确显示该方向会恶化生成分布。

### 5.2 两模型差值确实形成了有用的闭环方向

从 `s=0` 到 `s=-1`，FID 在每个已测点都改善，sFID 总体同步改善，IS 的最佳点位于
相邻的 `s=-0.75`。这说明 `v_JiT-x-v_SiT-v` 不只是无结构噪声；它的负方向在当前
闭环 ODE 中系统性地改变了生成分布。

### 5.3 这更像 weak-to-strong 外推，而不是 prediction-target 外推成功

因为 400K 时 SiT-v 明显优于 JiT-x，`s<0` 的实际含义是：

```text
JiT-x（较弱） -> SiT-v（较强） -> 越过 SiT-v
```

这种方向与 AutoGuidance 式 weak-to-strong 外推在形式上相似。它不能单独证明
“velocity target 天生更接近某个可继续外推的理想方向”，也不能证明任意两个
prediction targets 之间的差都能提供 guidance。

## 6. 尚不能确认什么

### 6.1 尚未找到最佳 scale

FID 在整个已测区间内随 `s` 单调上升，最佳点恰好是左边界 `s=-1`。因此本实验只说明
负方向值得继续检查，没有证明 `-1` 是最优 scale。更负的 scale 可能继续改善，也可能
突然造成轨迹不稳定、精度提高但覆盖下降。

### 6.2 尚未区分质量提高与覆盖收缩

当前只有 FID、sFID 和 IS，没有 precision、recall、class coverage 或样本重复率。
负方向可能改善视觉集中度，也可能牺牲分布覆盖；仅凭 FID-5K 不能区分。

### 6.3 尚未证明跨 seed、跨 step 稳定

两个模型都只有训练 seed 0，且只在 400K checkpoint 上完成这次配对扫描。当前正在进行
的 1M 续训可以检验：随着两个端点都变强，该方向是否保持、缩小、反转或消失。

### 6.4 不能与历史独立采样的绝对 FID 直接拼表

本轮配对采样的 `s=0/1` 分别为 `67.4735/75.9490`；历史独立 400K 评估分别为
`68.6537/77.6033`。checkpoint 相同，但生成噪声、标签抽样和采样 world size 协议不同，
FID-5K 本身也有采样波动。解释 scale 效果时应只使用本轮内部配对差值，不能把两套
绝对数交叉相减。

## 7. 与早期双头静态扫描的区别

早期实验在同一个 450K dynamic dual-output checkpoint 内混合 epsilon head 与 x head，
最佳点为 epsilon 外侧 `s=-0.20`，FID 只比纯 epsilon 改善 `0.4907`。本报告使用的是
两个独立训练的单头模型，方向定义为 SiT-v 到 JiT-x，最佳边界点比 SiT-v 改善
`8.3969` FID。

两者的 checkpoint、训练目标、端点质量和 scale 语义均不同，不能合并成一条曲线。

## 8. 数据完整性审计

已完成以下复核：

- 15/15 条 CSV 记录与各条件 `fid5k_adm_results.json` 逐项一致；
- 15/15 个 sampling manifest 均声明 5000 个样本、EMA、无 guidance；
- 所有 scale 的 noise SHA256 完全相同；
- 所有 scale 的 label SHA256 完全相同；
- 两个 checkpoint 的模型、数据 manifest、step 和官方 SiT 源版本相容；
- CSV 中的最佳条件与汇总 JSON 的最佳条件一致。

关键哈希：

```text
SiT-v checkpoint:
fa9a5d1c67783021035aa4ccc1d5eb09e5e928e170ee87c9d40f30c3e50320d5

JiT-x checkpoint:
24ece36139dcb14710a547085195450b93ab6bf473ef02f3a7651243b26a4f9f

shared noise fingerprint:
bb460ea94ef337bca3b15f07d0d136d2fe286168c5afef96aeda2e4757f5c143

shared label fingerprint:
dd55c4ee2ce9e86812f5dd9dbf7262467ac6fda3d319e21ec8716ce7292f6e46
```

本机正式源数据：

```text
/home/zhoushunyu/data/eqvae/imagenet_sit_flow/
  fid5k_static_pair_v_to_jit_x_step400000_seed0/
    static_pair_v_to_jit_x_fid5k.csv
    static_pair_v_to_jit_x_fid5k.json
```

仓库只保存本报告、便携 CSV、代码与测试，不保存 checkpoint、生成样本 NPZ、ImageNet、
VAE latent cache 或 FID reference。

## 9. 后续验收标准

这条现象要升级为可靠方法证据，至少需要：

1. 在 1M SiT-v 与 1M JiT-x 上重新扫描，并确认最佳方向没有反转；
2. 将负 scale 扩展到边界之外，找到内部最优点或明确的失稳拐点；
3. 对候选 scale 增加 precision/recall、class coverage 与重复率；
4. 至少补 3 个训练 seed，或采用独立采样 seed 的重复 5K/最终 50K 评估；
5. 检查速度场 RMS、方向 cosine 与 ODE NFE，排除结果只来自无约束幅值放大；
6. 只有在独立验证设置上仍稳定改善，才把该方向称为可复用 guidance。

## 10. 复现入口

- 速度场语义：[`experiments/imagenet100_sit_static_pair.py`](../experiments/imagenet100_sit_static_pair.py)
- 配对采样：[`experiments/sample_imagenet100_sit_static_pair_fid.py`](../experiments/sample_imagenet100_sit_static_pair_fid.py)
- FID 编排：[`experiments/run_imagenet100_sit_static_pair_fid5k.py`](../experiments/run_imagenet100_sit_static_pair_fid5k.py)
- 本轮命令：[`experiments/run_imagenet100_sit_static_pair_v_jit_x_400k.sh`](../experiments/run_imagenet100_sit_static_pair_v_jit_x_400k.sh)
