# SiT v800-v500 权重外推便携数据

该目录保存 SiT-S/2 `v800` strong 与 `v500` weak 的 EMA 权重外推实验所需的轻量结果。派生 checkpoint、生成样本 NPZ、日志和 FID 临时文件仍在本机数据盘，不写入 Git。

## 实验定义

权重外推只生成一个模型：

```text
theta_gamma = theta_800 + gamma * (theta_800 - theta_500)
```

配对的速度场 AutoGuidance 每次求值运行两个模型：

```text
v_gamma(z,t) = v_800(z,t) + gamma * (v_800(z,t) - v_500(z,t))
```

两个 checkpoint 使用相同 SiT-S/2 架构、ImageNet-100 latent cache、训练 seed 和协议，只相差训练进度。所有正式比较均使用 EMA 权重。

## 文件

| 文件 | 内容 |
|---|---|
| `checkpoint_provenance.json` | 10 个派生 checkpoint 的公式、源权重 SHA256、派生权重 SHA256 与字节数 |
| `small_scale_fid1k.csv/json/png` | `gamma={0,0.01,0.05,0.1}` 的严格配对 FID-1K 结果 |
| `large_scale_fid1k.csv/json/png` | `gamma={0.5,1,1.5,2,2.5,3}` 的严格配对 FID-1K 结果 |
| `small_scale_same_state.csv/json` | 小尺度时两种模型在同一 `(z,t)` 上的速度与增量比较 |
| `large_scale_same_state.csv/json` | 大尺度时两种模型在同一 `(z,t)` 上的速度与增量比较 |

CSV/JSON 中的本机绝对路径只用于 provenance；换机器后可能无法访问，不影响聚合数值。

## 协议

- strong：native velocity SiT-S/2 `v800` EMA；
- weak：native velocity SiT-S/2 `v500` EMA；
- dataset：ImageNet-100；
- sampler：Dopri5，FP32/TF32，CFG=1；
- metric：ADM FID/sFID/IS；
- screening：1,000 samples，sample seed 0；
- 所有条件共享 noise SHA256 `b693d3cc...c7b8`；
- 所有条件共享 label SHA256 `76fcd0fc...0758`。

## 头部结果

| `gamma` | 权重外推 FID | 速度外推 FID | 权重相对自身 `gamma=0` |
|---:|---:|---:|---:|
| 0 | 86.8140 | 86.8070 | 0 |
| 0.01 | 86.7754 | 86.7510 | -0.0386 |
| 0.05 | **86.7020** | 86.6353 | **-0.1120** |
| 0.10 | 88.5788 | **86.5191** | +1.7648 |
| 0.50 | 215.3378 | 84.8532 | +128.5238 |
| 1.00 | 346.7866 | 82.1577 | +259.9726 |
| 3.00 | 461.3254 | 74.4795 | +374.5114 |

`gamma=0.01/0.05` 的变化量很小，只能视为 FID-1K 筛查信号。`gamma=0.1` 已出现明显退化，而 `gamma>=0.5` 的权重外推全部失效。

## 一致性检查

- `gamma=0` 的派生 state dict 与 strong EMA 逐 tensor 精确相等；
- 权重公式在抽查 tensor 上最大数值误差为 0；
- 每个权重/速度条件的 noise 与 label SHA256 一致；
- 两条 `gamma=0` 采样代码路径的 FID 只差 `0.0070`；输出像素有 `93.47%` 逐位相等，平均绝对差为 `0.0668/255`；
- 相关测试共 15 项通过；
- checkpoint、NPZ、Numpy label、日志均未进入 Git。
