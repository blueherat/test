# RAEv2 上的 PFR 迁移：指数重定时、正式协议与初步结果

状态：**公式和实现审计通过；FID-1K 的正信号未通过正式 FID-5K，OU 证书方向也未在
两个独立 FID-1K bank 中超过 raw PFR。RAEv2 迁移当前应判为阴性。**

## 一、实验问题

SiT-S/2 上，PFR 在等 full-forward-equivalent 预算下将 ordinary IG 的 FID-5K 从
`40.912` 降到 `37.530`，相对改善 `8.27%`。本轮不假设这个幅度能够原样迁移，而是检验
同一构造能否跨越以下变化：

- SiT 的低维 SD-VAE latent 改为 RAEv2 的 `1024 x 16 x 16` representation latent；
- native velocity prediction 改为 clean prediction；
- 数据方向从 SiT 的 `t: 0 -> 1` 改为 RAEv2 的 `t: 1 -> 0`；
- 独立/后训练内部头改为 RAEv2 官方联合训练的 depth-8 base head。

只有公式、默认采样协议和内部头计算都保持 faithful，跨模型正结果才有意义。

## 二、官方协议审计

本轮使用 RAEv2 官方 `imagenet-dinov3l-k7.yaml` 对应的：

- checkpoint step `100080`，EMA 参数；
- `DiTwDDTHeadIG`，base depth `8`；
- clean-`x` prediction；
- shifted 100-step Euler；
- CFG `1.0`；
- ordinary IG scale `1.78`，区间 `t in [0.1, 1]`；
- 官方 stage-1 decoder 与 normalization statistics；
- 每类均衡标签、条件间相同初始噪声和标签。

单独执行 prefix 的 base-head 输出已与完整官方 forward 返回的 base 输出做逐元素检查，结果
bitwise identical。PFR 因而只省去未被 base head 使用的 suffix blocks，没有改变 weak predictor。

## 三、反向时间坐标下的精确公式

RAEv2 的 linear bridge 为

\[
z_t=(1-t)x+t\epsilon,
\]

其中 `t=1` 是高斯噪声，`t=0` 是数据。模型预测 clean endpoint，采样使用

\[
v_t(z)=\mathbb E[\epsilon-x\mid z_t=z]
      =\frac{z-\hat x_t(z)}{t}.
\]

在 PFR/IG 实际作用的 `t >= 0.1` 区间内，官方 `t_eps=0.05` 不参与分母，因此以下代数不受
clamp 影响。由 `x=z-tv_t` 可得 marginal score

\[
s_t(z)=\nabla_z\log p_t(z)
      =-\frac{z+(1-t)v_t(z)}{t}.
\]

定义数据相对噪声的赔率

\[
r(t)=\frac{1-t}{t},
\]

以及相对标准高斯 score `s_phi(z)=-z` 的 normalized shape field

\[
m_t(z)=\frac{s_t(z)-s_\phi(z)}{r(t)}=-(z+v_t(z)).
\]

于是任意时刻的 score 都精确写成

\[
s_t=s_\phi+r(t)m_t.
\]

这里把“数据证据有多强”与“证据的空间形状是什么”分开了：`r(t)` 是已知标量，`m_t`
是模型所表示的 normalized probability landscape。

## 四、唯一的 Gaussian-centered 指数重定时

若一个理想化 probability path 只是在同一能量景观上逐渐增加数据证据，则存在与时间无关的
`m(z)`，使

\[
s_t=s_\phi+r(t)m.
\]

当 `m=grad F` 时，它对应指数族射线

\[
p_t(z)\propto \phi(z)\exp(r(t)F(z)).
\]

这不是声称真实 diffusion marginals 必须属于该指数族，而是定义一个可检验的零模型：
**生成只会提高同一能量景观的强度，不会改变景观形状。**

把未来数据侧时刻 `tau<t` 的 score 放回当前信息强度，保持该零模型不变的 affine map 唯一为

\[
\mathcal R_{t\leftarrow\tau}(s_\tau)
=s_\phi+\frac{r(t)}{r(\tau)}(s_\tau-s_\phi).
\]

它满足 identity 与 semigroup，并且只依赖赔率而不依赖时间表的参数化。当前 score 偏离这条
指数射线的有限 defect 是

\[
\delta_{t,\tau}
=s_t-\mathcal R_{t\leftarrow\tau}(s_\tau)
=r(t)\,[v_\tau-v_t].
\]

因此 `delta=0` 当且仅当两个时刻看到同一个 normalized shape field；它不是 Euler 截断误差，
也不是 trajectory curvature，而是 probability landscape 超出纯“升温/降温”变化的形变。

这个重定时的唯一性证明很短。若要求 affine map 固定 Gaussian score，则只能写成

\[
T(s)=s_\phi+a(s-s_\phi).
\]

对任意 shape field `m`，又要求它把 `s_phi+r(tau)m` 精确映到 `s_phi+r(t)m`，便必须有

\[
a\,r(\tau)=r(t),
\]

即 `a=r(t)/r(tau)`。该比值自动满足

\[
\mathcal R_{t\leftarrow u}\,\mathcal R_{u\leftarrow\tau}
=\mathcal R_{t\leftarrow\tau}.
\]

因此，只要接受“Gaussian-centered、affine、保留固定 landscape”这三个要求，当前 operator
不是一种方便的选择，而是唯一选择。

## 五、PFR 与 ordinary IG

记官方 full head 为 strong `S`，depth-8 base head 为 weak `W`，ordinary IG score 为

\[
s_{IG}=s_{W,t}+\beta(s_{S,t}-s_{W,t}),\qquad \beta=1.78.
\]

PFR 加入 weak path 的指数重定时 defect：

\[
s_{PFR}=s_{IG}+\beta\rho
\left[s_{W,t}-\mathcal R_{t\leftarrow\tau}(s_{W,\tau})\right].
\]

换回 RAEv2 的反向采样 velocity 后恰好是

\[
v_{PFR}
=v_{W,t}+\beta(v_{S,t}-v_{W,t})
+\beta\rho(v_{W,t}-v_{W,\tau}).
\]

符号看似与 SiT 相反，是因为 RAEv2 从 `t=1` 向 `t=0` 积分；换到 score space 后加入的都是
同一个正向 defect。

另一个等价写法揭示了方法含义。定义

\[
s_{neg}=(1-\rho)s_{W,t}+\rho\mathcal R_{t\leftarrow\tau}(s_{W,\tau}),
\]

则

\[
s_{PFR}=s_{W,t}+\beta(s_{S,t}-s_{neg}).
\]

也就是说，ordinary IG 使用“当前 weak”作为负参考；PFR 将它部分替换为“获得更多数据信息、
再剥掉平凡 SNR 增益后”的 counterfactual weak reference。

## 六、两个独立 FID-1K bank

旧仓库评估器使用同一个 ADM reference NPZ。下表仅用于同 bank 配对筛查：

| 条件 | seed 20260903 | seed 20260904 | 均值 | 相对 IG 均值改善 |
|---|---:|---:|---:|---:|
| full | 41.388 | 41.347 | 41.368 | -0.395 |
| ordinary IG | 41.164 | 40.782 | 40.973 | 0.000 |
| `h=1/64, rho=0.1` | 40.699 | 40.220 | 40.460 | 0.513 |
| `h=1/64, rho=0.2` | 40.563 | 40.349 | 40.456 | 0.517 |
| `h=1/32, rho=0.05` | 40.686 | 40.180 | **40.433** | **0.540** |
| `h=1/32, rho=0.1` | 40.663 | 40.301 | 40.482 | 0.490 |

正确符号的四个温和条件在两个 bank 上全部优于 ordinary IG。反号 `h=1/32,rho=-0.1`
在 seed 20260903 得到 `43.391`；过强的 `h=1/32,rho=0.2` 得到 `41.502`。因此响应具有
正确符号与有限强度 turnover，不是任意扰动都改善。

固定乘积 `h*rho` 的条件近似等价：

- `(1/64,0.1)` 与 `(1/32,0.05)` 的逐 seed FID 差仅 `0.013 / 0.040`；
- `(1/64,0.2)` 与 `(1/32,0.1)` 的逐 seed差为 `-0.100 / 0.048`。

这支持当前温和设置处于有限差分的局部导数区间，而不是一个孤立超参数点。

## 七、官方 evaluator 复核

RAEv2 当前仓库的 stage-2 评价使用 `nanogen-evals/fd_evaluator` 与
`imagenet_256_fid_stats`。它与旧 evaluator 的绝对 FID 相差约 2，但关键排序一致。

| bank | ordinary IG | `h=1/32,rho=0.05` | PFR 改善 |
|---|---:|---:|---:|
| seed 20260903 | 38.919 | 38.418 | **0.501** |
| seed 20260904 | 38.397 | 37.890 | **0.507** |

seed 20260904 的 full 为 `39.304`，故该 bank 中 ordinary IG 相对 full 改善 `0.906`，PFR
相对 ordinary IG 再改善 `0.507`，PFR 相对 full 总改善 `1.413`。三者必须分开报告。

较强的 `h=1/64,rho=0.2` 在该 bank 的官方 FID 为 `38.194`，只比 IG 好 `0.203`，但 IS
升得更多。因此正式 5K 预注册候选选用跨 evaluator 更稳定的 `h=1/32,rho=0.05`，不追逐
单次最低旧 FID。

正式 FID-5K 随后否定了这个 FID-1K 正信号：

| 条件 | FID-5K | IS |
|---|---:|---:|
| ordinary IG | **7.034546** | 154.5394 |
| raw PFR，`h=1/32,rho=0.05` | 7.224213 | **167.8552** |

两路来自同一个 seed、同一均衡标签集和同一官方 evaluator。raw PFR 的 IS 上升，但 FID
恶化 `0.189666`。因此不能把两个经过多条件筛选的 FID-1K bank 当成分布质量改善；它们至多
说明该扰动会稳定改变质量/覆盖折衷。这里也再次说明 FID 的绝对值会强烈依赖样本数：1K 与
5K 数值不能纵向直接比较，只能在同一 bank 内比较条件。

## 八、OU 一阶证书与幅值保留的跨模型反证

SiT 上进一步得到一个无新增可调系数的构造：用 OU 一阶消失矩证书决定修正方向，同时保留
raw revision 的逐样本 RMS。为检验它是否为跨模型规律，本轮将完全相同的原则原样迁移到
RAEv2；沿用 raw PFR 的 `h=1/32,rho=0.05`，未重新扫参。

两个新条件为：

- `OU-common`：把 raw revision 投影到 strong-head OU 一阶 defect；
- `OU-polar`：保持该投影方向，但恢复 raw revision 的逐样本范数。

官方 FID-1K 结果如下：

| 条件 | seed 20260903 | seed 20260904 | 均值 | 相对 ordinary IG 均值改善 |
|---|---:|---:|---:|---:|
| ordinary IG | 38.919309 | 38.397378 | 38.658344 | 0.000000 |
| raw PFR | **38.416922** | **37.890022** | **38.153472** | **0.504872** |
| OU-common | 38.607826 | 38.112188 | 38.360007 | 0.298337 |
| OU-polar | 38.696531 | 37.961418 | 38.328975 | 0.329369 |

`OU-polar` 在 seed 20260904 中比 `OU-common` 好 `0.150770`，说明恢复幅值确实纠正了
一部分投影衰减；但两 seed 平均仍比 raw PFR 差 `0.175503`。第一 seed 中 polar 也略差于
common，说明幅值恢复的微小排序本身尚不稳定。

该结果给出两个边界：

1. OU 一阶 defect 是一个数学上可识别的 probability-shape certificate，但它不是“投影后
   必然提高任意生成模型质量”的充分条件；
2. SiT 上被证书删除的分量在 RAEv2 的高维 DINO latent、联合训练 depth-8 weak head 中可能
   仍携带有用信息。不同 representation 和 weak-head construction 会改变可用修正子空间。

OU 两路每 1K 样本使用 `22144` 次 full-model call 和 `12672` 次 prefix call，而 raw PFR
使用 `12800 + 12672`，ordinary IG 使用 `12800 + 0`。所以这组只是机制迁移筛查，不是等计算
比较。鉴于 raw PFR 本身已经未通过正式 FID-5K，而 OU 两路在两个 FID-1K bank 中又稳定落后
raw PFR，本轮不再为 RAEv2 追加 OU FID-5K 或事后调参。

## 九、当前机制证据与边界

64 个 ordinary-IG rollout 的几何审计显示，weak/strong defect cosine 从噪声侧约 `0.74`
逐步降到数据侧约 `0.07`，共同能量比例从约 `56%` 降到约 `0.5%`。这否定了一个过强说法：

> PFR 必须依靠 strong 与 weak 在所有模型、所有时间共享同一个 temporal-curvature direction。

SiT 上“shared defect”是一个强的充分机制；它不是 RAEv2 正结果的必要条件。若正式 5K 保留
增益，更稳妥的跨模型理论对象应是：

> Gaussian-centered exponential ray 给出唯一的跨时间可比基准；PFR 使用相对该基准的
> normalized landscape deformation 构造 temporal contrast。具体哪些 defect component 有益，
> 由模型与 representation 决定。

目前没有定理保证沿该 defect 必然降低 FID；FID、IS 与 KID 也可能呈现 precision/recall
权衡。正式结论必须等待 FID-5K、等计算对照，并保留任何指标分歧。

## 十、理论导出的下一种参数化

raw-time horizon `h` 会随 noise schedule 改变含义。更自然的是固定 information-time distance

\[
\lambda(t)=\log r(t),\qquad \lambda(\tau)=\lambda(t)+\Delta\lambda.
\]

RAEv2 中它给出闭式

\[
\tau=\frac{t}{t+e^{\Delta\lambda}(1-t)}.
\]

此时重定时系数恒为 `exp(-Delta lambda)`，并且小距离下

\[
\delta_{t,\tau}\approx
-\Delta\lambda\,r(t)\,\partial_\lambda m_t.
\]

这把任意 `(h,rho)` 二维调参改成一个 schedule-invariant 的信息距离。代码保留 raw-time
默认行为，并新增独立 `log_odds` 模式；只有其跨 bank 表现不弱于 raw-time 版本，才会把它
提升为方法设计，而不是仅作为更漂亮的记号。
