# SiT 800K nominal-path frozen guidance 机制验证

## 研究问题

本轮只回答一个问题：为什么把 AutoGuidance gap 固定在 strong model 的 baseline trajectory 上，仍能保留 closed AutoGuidance 的大部分 FID 收益？

记 strong field 为 `S`，weak field 为 `W`，两者差值为：

```text
g(z,t) = S(z,t) - W(z,t)
```

四条关键轨迹是：

```text
baseline: z_b' = S(z_b,t)
frozen:   z_f' = S(z_f,t) + gamma * g(z_b,t)
replay:   z_r' = S(z_b,t) + gamma * g(z_b,t)
closed:   z_c' = S(z_c,t) + gamma * g(z_c,t)
```

`frozen` 只冻结 gap 的求值位置，仍在当前 guided state 上计算 `S(z_f,t)`。`replay` 才同时去掉 strong field 的当前状态响应。

## 实验设置

- 数据与指标：ImageNet-100，ADM FID-5K；donor 信息来源消融先用 FID-1K 筛查。
- strong model：`v800`，原生 velocity SiT-S/2 的 800K EMA checkpoint。
- weak model 1：`x800`，x-output + velocity-space loss 的 800K EMA checkpoint。
- weak model 2：`v500`，同一 native-velocity run 的 500K EMA checkpoint。
- 引导系数：`gamma=1`。
- 采样：fp32、TF32、Dopri5、`atol=1e-6`、`rtol=1e-3`、CFG=1。
- 几何诊断：每类 2 个随机 seed，每个 seed 512 个配对样本。
- 正式 FID：2 个配对采样 seed，每个条件 5000 张；同 seed 内 noise、class label 和 reference 完全相同。
- 按用户最新要求，本轮不做 400K 复验。

## 结论

当前证据支持的机制不是“gap 完全不随状态变化”，也不是“固定 gap 自己就足以改善生成”。更准确的结论是：

1. `g(z_b,t)` 在 baseline 到 frozen 的实际偏移范围内保留稳定的主方向，因此它可以充当样本和类别特异的 nominal correction。
2. frozen 仍然是闭环系统：`S(z_f,t)` 对当前状态的响应负责把 nominal correction 运输成有效终点位移。
3. 若连 strong field 也固定成 replay，大部分 FID 收益消失；因此 current-state strong response 是必要组成，而不只是 closed 比 frozen 多出来的微调。
4. closed 在线重算 gap 的主要有效部分是相对 nominal gap 的正交方向修正。只恢复平行强度变化几乎无效，只恢复方向变化则接近或达到 closed 的 FID。
5. nominal gap 不是通用时间增益：替换对应类别或初始噪声都会明显破坏 FID，说明它绑定具体条件和具体生成轨迹。

## 方向稳定性

在 frozen trajectory 上比较 `g(z_b,t)` 与 `g(z_f,t)`，两个 seed 的均值如下：

| family | `t=0.6` cosine | `t=0.8` cosine | `t=0.9` cosine | `t=0.95` cosine |
|---|---:|---:|---:|---:|
| `v800-v500` | 0.843 | 0.833 | 0.835 | 0.845 |
| `v800-x800` | 0.783 | 0.757 | 0.750 | 0.759 |

同 target 的 `v800-v500` 更刚性；prediction-target gap 对状态更敏感。但两者都没有在实际 frozen 偏移后变成随机或反向方向。

沿 `z_b + alpha(z_f-z_b)` 扫描 `alpha={0,.25,.5,.75,1}` 时，cosine 和 projection coefficient 都平滑变化，没有只在两个端点偶然对齐。

## Strong-field feedback 对照

正式 FID-5K 的双 seed 均值：

| family | baseline | frozen | replay | closed |
|---|---:|---:|---:|---:|
| `v800-x800` | 60.485 | 56.565 | 60.218 | 54.982 |
| `v800-v500` | 60.485 | 55.690 | 59.446 | 55.151 |

`x800` replay 只比 baseline 低 0.268 FID，`v500` replay 低 1.039 FID；两者都远弱于 frozen。sFID/IS 也没有支持 replay 复现 frozen：x replay 的两项都变差，v replay 只有 IS 小幅改善。

这排除了“把 nominal gap 沿时间简单积分就能得到 frozen 收益”的解释。

## 精确动力学解释

令 `delta_f=z_f-z_b`，则 frozen 与 baseline 的差满足：

```text
delta_f' = S(z_b + delta_f,t) - S(z_b,t) + gamma * g(z_b,t)
```

利用微积分基本定理，可以精确写成：

```text
delta_f' = [integral_0^1 J_S(z_b + a*delta_f,t) da] * delta_f
           + gamma * g(z_b,t)
```

这里没有小 `gamma` 或一阶 Taylor 假设。第一项是 strong flow 沿实际位移的 secant response，第二项才是 nominal gap。

replay 则满足：

```text
delta_r' = gamma * g(z_b,t)
```

实验中 strong-field change 的 RMS 在中后段约为 nominal gap RMS 的 `1.4-1.9x`（x800）和 `2.1-2.8x`（v500），但它与 nominal gap 的 cosine 接近 0 或略为负值。也就是说，strong field 不是简单放大 nominal gap，而是在当前状态上对它进行显著的旋转和运输。

端点上，frozen 与 closed 的位移 cosine 为：

| family | seed 0 | seed 1 |
|---|---:|---:|
| `v800-x800` | 0.878 | 0.880 |
| `v800-v500` | 0.940 | 0.943 |

所以 frozen 的终点作用与 closed 高度一致，但这种一致性是 nominal correction 经 strong flow 闭环运输后的结果，不是 frozen gap 单独的性质。

## 强度与方向投影

在每个当前 state 上按整张 latent、逐样本分解：

```text
g_current = alpha * g_nominal + g_orth
```

构造两个自洽闭环消融：

```text
gain_only:      S(z,t) + gamma * alpha * g_nominal
direction_only: S(z,t) + gamma * (g_nominal + g_orth)
```

正式 FID-5K：

| family | seed | frozen | gain-only | direction-only | closed |
|---|---:|---:|---:|---:|---:|
| `v800-x800` | 0 | 57.031 | 56.962 | 55.592 | 55.258 |
| `v800-x800` | 1 | 56.100 | 55.905 | 55.005 | 54.705 |
| `v800-v500` | 0 | 56.221 | 56.523 | 55.155 | 55.569 |
| `v800-v500` | 1 | 55.158 | 55.481 | 54.406 | 54.734 |

两组 x800 中，direction-only 恢复了 frozen 到 closed 数值 FID 差异的 78.5% 和 81.1%；gain-only 只恢复 3.9% 和 13.9%。两组 v500 中，gain-only 都比 frozen 更差，而 direction-only 与 closed 相当并在本次 FID-5K 上略低。

由于 FID 非线性且只有 5K 样本，不能把这些比例解释成“机制贡献率”，也不应宣称 direction-only 严格优于 closed。可靠结论是：主指标在两个采样 seed 上都把有效的在线修正指向 `g_orth`，而不是 `alpha`。sFID/IS 对 x800 的细小排序较混合，对 v500 则基本支持同一趋势。

## Nominal gap 的信息来源

FID-1K donor 筛查：

| family | paired | 换类别 | 换噪声 | 类别和噪声都换 |
|---|---:|---:|---:|---:|
| `v800-x800` | 82.203 | 89.979 | 88.354 | 89.000 |
| `v800-v500` | 82.596 | 87.847 | 87.705 | 87.413 |

换类别或换初始噪声都会增加约 5-8 FID。这个结果目前只是一组 1K 强筛查，但效应远大于 paired runner 的数值误差，说明有效的 nominal gap 同时依赖类别和实例轨迹。

## 证据边界

- 这是两个采样 seed，不是多个训练 seed；所有 checkpoint 来自同一个训练 run。
- FID-5K 用于内部配对机制比较，不等同于正式 ImageNet FID-50K。
- donor 消融只有 1K，不能据此比较类别信息和实例信息谁更重要。
- x800 包含其既有 denominator floor；本轮问题是 frozen transfer 机制，不重新解释该 floor。
- gain-only 与 direction-only 会各自改变后续 state，因此二者的 FID 不能线性相加。
- 没有发现数据泄露：模型只生成随机噪声出发的样本，ImageNet-100 validation 统计只用于最终 ADM 指标；同 seed 的输入哈希完全配对。

## 验证与文件

- 相关测试：61 项通过。
- 独立数据复算：检查了 67,584 行几何原始记录，预期行数完整、主键无重复、projection valid rate 为 100%、汇总数值无 NaN/Inf。
- 正式投影实验包含 2 个 family x 2 个 seed x 5 个条件，共 20 行；每个 seed 内 noise/label fingerprint 完全一致。
- 8 组新增正式采样均为 5,000 张，100 个类别全部覆盖；逐条件类别计数范围为 31-73。
- 共检查 56 份 sampling/FID resource audit，均返回 0、无 violation、峰值低于 8192 MiB。
- 400K 复验按用户最新要求不属于本轮验收范围。
- Git 便携汇总包：`docs/data/imagenet100_sit_800k_nominal_guidance_transfer/`
- 结果根目录：`/home/zhoushunyu/data/eqvae/imagenet_sit_flow/nominal_guidance_transfer_800k_v1`
- 几何汇总：`summary/nominal_transfer_compact.csv`
- replay 表：`causal_screen/summary/replay_fid5k.csv`
- 正式投影表：`causal_screen/summary/projection_fid5k.csv`
- donor 表：`causal_screen/summary/donor_fid1k.csv`
- 正式投影图：`causal_screen/summary/projection_fid5k.png`
