# ImageNet-100 SiT 前瞻物质导数 Internal Guidance

## 一、结论

本轮只研究 Internal Guidance（IG）。我们没有把 Foresight Fixed Point
Guidance（FSG）的 CFG 固定点语义直接搬到 IG，而是先对它的“一次前向预测、一次弱场
逆推”做精确分解，再删除其中与现有 IG 重复的局部差值。留下的唯一新量是弱头沿预测
生成路径的未来变化：

\[
\boxed{
G_{\mathrm{FMD}}(z,t)
=G(z,t)+\eta\left[W_4(z,t)-W_4(z+hG(z,t),t+h)\right]
}
\]

其中

\[
h=\min(H,0.5-t),
\]

而原 depth-4 IG 为

\[
G(z,t)=S(z,t)+\gamma(t)\left[S(z,t)-W_4(z,t)\right].
\]

`S` 是 800K SiT-S/2 EMA 的完整 velocity field，`W4` 是冻结 backbone 后训练
50K step 的 depth-4 velocity readout。原最优调度保持不变：

\[
\gamma(t)=
\begin{cases}
0.6,&0\le t<0.25,\\
0.7,&0.25\le t<0.5,\\
0,&0.5\le t\le1.
\end{cases}
\]

正式 ImageNet-100 FID-5K 配对结果为：

| sampling seed | depth4 IG | FMD-IG | FID 改善 |
|---:|---:|---:|---:|
| 0 | 39.953584 | **37.944441** | **-2.009143** |
| 1 | 39.939969 | **37.929577** | **-2.010392** |
| 均值 | 39.946776 | **37.937009** | **-2.009767（-5.03%）** |

两套独立采样 bank 上，FID 改善几乎相同；sFID 均值从 `69.925170` 降到
`68.040809`，IS 均值从 `36.852156` 升到 `37.793234`。这超过了本轮严格配对的
depth4 最优解，也低于仓库此前所有已登记的 IG FID-5K 结果。

这仍然是一个训练 seed、两个 sampling seed、FID-5K 的内部研究结论，不等同于
ImageNet-1K FID-50K 或多训练 seed 结论。

## 二、为什么不能直接照搬 FSG

### 2.1 IG 与 CFG 的语义不同

FSG 的关键直觉是：先沿有条件/引导路径向前看，再用无条件模型逆推，通过固定点迭代
校准 CFG 路径。CFG 中有明确的 conditional/unconditional 分布语义。

IG 中，`W4` 只是同一 strong model 的中间 readout。它不是一个无条件分布，也没有理由
要求 strong 与 weak 路径在某个“黄金条件路径”上达到 CFG 式固定点。因此，本轮只借用
FSG 的**路径前瞻操作**，不继承其分布语义。

### 2.2 一次 Euler 往返的精确分解

从当前 `(z,t)` 出发，先用 IG 场做一步前向：

\[
z^+=z+hG(z,t).
\]

再用未来点的 weak field 做一步逆向：

\[
\widetilde z=z^+-hW_4(z^+,t+h).
\]

往返位移除以 `h` 后为

\[
\frac{\widetilde z-z}{h}
=G(z,t)-W_4(z+hG(z,t),t+h).
\]

加减当前弱场 `W4(z,t)`：

\[
\begin{aligned}
G-W_4(z+hG,t+h)
&=\underbrace{G-W_4(z,t)}_{\text{局部项}}
+\underbrace{W_4(z,t)-W_4(z+hG,t+h)}_{\text{未来项}}.
\end{aligned}
\]

因为

\[
G-W_4=(1+\gamma)(S-W_4),
\]

第一项没有提供新的路径信息，只是再次放大已经调好的 `S-W4`。实验也证实：直接端点
外推、匹配一阶强度的 probe、完整 round trip、原式 FSG analog、分段 budget 和
early-only analog 均未超过 depth4 IG。

因此，成功条件只保留第二项。

## 三、这个新项到底是什么

定义 Euler 预测射线

\[
\ell(\tau)=\left(z+\tau G(z,t),\ t+\tau\right),\qquad 0\le\tau\le h.
\]

对可微弱场，微积分基本定理给出精确恒等式：

\[
\begin{aligned}
W_4(z,t)-W_4(z+hG,t+h)
=-\int_0^h
\left[
\partial_tW_4+J_zW_4\,G(z,t)
\right]_{\ell(\tau)}d\tau.
\end{aligned}
\]

所以它不是另一次静态 `strong - weak` 外推，而是弱场沿当前 IG 预测方向的**有限路径
变化积分**。当 `h` 很小时：

\[
W_4(z,t)-W_4(z+hG,t+h)
=-h\left(\partial_tW_4+J_zW_4G\right)+O(h^2).
\]

于是

\[
\boxed{
G_{\mathrm{FMD}}
=G-\kappa D_GW_4+O(\kappa H),
\qquad \kappa=\eta H
}
\]

其中

\[
D_GW_4=\partial_tW_4+J_zW_4G
\]

是弱场沿 guided characteristic 的物质导数。这就是 “Foresight Material-Derivative
IG” 名称的来源。

在 `t -> 0.5` 时使用

\[
h=\min(H,0.5-t)
\]

使实际增益

\[
\kappa(t)=\eta h
\]

连续衰减为零，避免未来查询跨入原本关闭 IG 的后半程。

这段推导只证明新项的数学来源，不证明它必然改善 FID。质量增益由后面的配对实验
建立。

## 四、FID-1K 机制证据

### 4.1 最优点

同一个历史 FID-1K noise/label bank 下：

| 方法 | `H` | `eta` | `kappa=H*eta` | FID-1K |
|---|---:|---:|---:|---:|
| depth4 IG | - | 0 | 0 | 64.850875 |
| FMD-IG | 0.03125 | 1.0 | 0.03125 | 63.360230 |
| FMD-IG | 0.046875 | 0.6 | 0.028125 | **63.270378** |
| FMD-IG | 0.0625 | 0.4 | 0.025 | 63.335434 |
| FMD-IG | 0.125 | 0.225 | 0.028125 | 63.917506 |

最优点相对配对 depth4 IG 改善 `1.580497` FID（`2.44%`）。

### 4.2 参数按 `kappa=eta*H` 近似塌缩

三个较短 horizon 的最佳区域均落在

\[
\kappa\approx0.025\sim0.03125.
\]

这正是小 `H` 展开预测的控制变量，而不是 `H` 或 `eta` 各自独立决定效果。相同
`kappa` 下，`H=0.125` 和 `0.1875` 变差，符合有限差分截断误差
`O(kappa*H)` 随 horizon 增大的解释。

完整网格保存在
`docs/data/imagenet100_sit_foresight_material_ig/fid1k_horizon_strength.csv`。

### 4.3 符号对照

固定 `H=0.125`：

| `eta` | FID-1K |
|---:|---:|
| 0 | 64.853061 |
| +0.25 | **63.965763** |
| +0.50 | 65.661300 |
| +1.00 | 73.950375 |
| -0.25 | 68.478295 |
| -0.50 | 75.342295 |
| -1.00 | 97.514698 |

FSG 分解导出的正号被数据明确选择；相反符号从最小幅度起就显著恶化。正向过大也会
过冲，因此该项不是越强越好。

## 五、正式 FID-5K

协议：ImageNet-100、SiT-S/2 800K EMA、SD-VAE、无 CFG、Dopri5、`rtol=1e-3`、
`atol=1e-6`、每个 sampling seed 生成 5,000 张，ADM ImageNet-100 validation-5K
reference。每个 seed 内 baseline 与候选使用完全相同的初始噪声和类别标签。

| seed | 方法 | FID ↓ | sFID ↓ | IS ↑ |
|---:|---|---:|---:|---:|
| 0 | depth4 IG | 39.953584 | 69.936335 | 36.835426 |
| 0 | FMD-IG, `H=.046875, eta=.6` | **37.944441** | **68.050170** | **37.758644** |
| 1 | depth4 IG | 39.939969 | 69.914004 | 36.868885 |
| 1 | FMD-IG, `H=.046875, eta=.6` | **37.929577** | **68.031449** | **37.827824** |

配对哈希：

| seed | noise SHA256 | label SHA256 |
|---:|---|---|
| 0 | `0ea1ae6701039845f0596ad3387b7f35480ce9c79f3907437804ef679ffd2636` | `57849a94ad38e74bda68272bd08e273a7de143b43b6c5979dddc9e10bade8feb` |
| 1 | `255a20ae36953f03db7de9717748d1a0f50974074b52b02ba0c4750234ddab23` | `a478940abd1f0d9b2d09cf0687a504312835c3c69b227710619cc9d0b8f6a9bc` |

另一个邻近候选 `H=.0625, eta=.4` 在 seed0 得到 FID `38.058305`，也明显优于
配对 baseline。这说明正结果并不依赖唯一一个孤立超参数点。

## 六、计算量与等价性

未来查询只需要 `W4`，因此实现只运行 backbone 前 4/12 个 transformer blocks，再接
训练好的 depth4 head；不运行无用的后 8 层。

真实 checkpoint、batch=3、`t={.03,.23,.47}` 的逐元素验证结果为：

```text
bitwise_equal = true
max_abs_diff = 0
```

旧的完整 suffix 与新的 prefix 实现生成的 FID-1K preview SHA256 完全相同：

```text
6f942921b3d52f46d00e260004af1e5e082f4d9bc72f2e20b8b4d00ed35cbcf4
```

单次 batch=8 microbenchmark 中，完整 paired forward 为约 `5.115 ms`，depth4 prefix
future query 为约 `1.999 ms`，即 future query 约为完整 forward 的 `39.1%`。按此折算：

- FID-1K：depth4 IG 为 8,572 次完整调用等价量；FMD-IG 约为 9,929，增加约 `15.8%`；
- FID-5K 两 seed 均值：约增加 `16.4%` 完整调用等价量。

额外 NFE 本身不能解释质量改善。只收紧 depth4 IG 的 Dopri5 容差：

| depth4 IG | NFE | FID-1K |
|---|---:|---:|
| `rtol=1e-3` | 8,572 | **64.850875** |
| `rtol=3e-4` | 14,740 | 65.154781 |
| `rtol=1e-4` | 20,986 | 65.204089 |

更多求解器调用没有降低 FID，反而略微恶化。

## 七、哪些结论成立，哪些还不成立

当前证据支持：

1. 直接把 CFG 的 FSG 固定点操作搬到 IG 不成立；IG 缺少相同的分布语义，而且完整
   往返中包含一个重复放大 local gap 的项。
2. 删除重复局部项后，弱头沿预测 IG 路径的有限变化是稳定有效的新信息。
3. 最优超参数在短 horizon 下近似服从 `kappa=eta*H`，与物质导数展开一致。
4. 该方法在两个独立 sampling seed 上同时改善 FID、sFID 和 IS，并超过既有 depth4
   配对最优解。

当前证据不支持：

1. 不能声称存在 IG 的 CFG 式“黄金固定点”。
2. 不能声称 `-D_GW4` 是真实误差或 Bayes correction；它只是可计算的路径前瞻量。
3. 不能由一个训练 seed、ImageNet-100 和 FID-5K 推广到其他模型、数据集或正式
   FID-50K。
4. 两个 sampling seed 不是两个训练 seed，也不是统计置信区间。

## 八、代码与数据

- 主程序：`experiments/run_imagenet100_sit_path_extrapolated_ig.py`
- 数学算子：`experiments/internal_guidance_path_extrapolation.py`
- prefix weak-head evaluation：`experiments/imagenet100_sit_multiscale_models.py`
- 单元测试：
  - `tests/test_internal_guidance_path_extrapolation.py`
  - `tests/test_imagenet100_sit_path_extrapolated_ig.py`
- 便携数据：`docs/data/imagenet100_sit_foresight_material_ig/`

## 九、拆分实验：真正发生的是强场演化项消去

前面的“物质导数”写法仍然只说明了成功项是什么，没有解释为什么这个组合会成功。
为了拆开它，在同一个未来点

\[
(z^+,t^+)=(z+hG,t+h)
\]

记

\[
S^+=S(z^+,t^+),\qquad W^+=W(z^+,t^+),\qquad g=S-W.
\]

弱场未来漂移可以精确分成

\[
\begin{aligned}
W-W^+
&=\underbrace{(g^+-g)}_{\text{gap change}}
 +\underbrace{(S-S^+)}_{\text{negative strong change}}.
\end{aligned}
\]

因为

\[
g^+-g=(S^+-S)-(W^+-W),
\]

第二项恰好把第一项中的强场演化量 `S^+-S` 完全消掉，只留下

\[
-(W^+-W)=W-W^+.
\]

这不是近似，也不依赖小 `h`。FID-1K 的因果拆分为：

| 条件 | FID-1K | 相对 baseline |
|---|---:|---:|
| depth4 IG | 64.852163 | 0 |
| `gap_change` | 66.869202 | +2.017039 |
| `gap_change`，RMS 匹配 | 66.626069 | +1.773906 |
| `strong_curvature` | 65.988144 | +1.135981 |
| `strong_curvature`，RMS 匹配 | 65.874068 | +1.021905 |
| 两项精确相加 | **63.270674** | **-1.581489** |

所以不能把结果解释成“两个各自有益的方向叠加”。两个分量在相同系数下都单独有害，
即使匹配组合项的 RMS 仍有害；有益的是它们的**代数消去结构**。当前数据支持的最窄
结论是：若目标是分离 pure weak evolution，那么 `gap change` 中的 `S^+-S` 是一个必须
消去的代数混杂项；消去后得到的 pure weak evolution 在本实验中有效。这里的“混杂”是
相对于该分解目标而言，并不证明 `S^+-S` 在一般生成动力学中是垃圾，也不证明任意模型
上的 weak evolution 都有益。

完整数据保存在
`docs/data/imagenet100_sit_foresight_material_ig/fid1k_decomposition.csv`。

## 十、更简洁的形式：Future-Weak Reference IG

原 FMD 条件为

\[
G_{\eta}=S+\gamma(S-W)+\eta(W-W^+).
\]

令

\[
\alpha=\frac{\eta}{\gamma},
\]

则可以精确改写为

\[
\boxed{
G_{\alpha}
=S+\gamma\left[S-\bar W_{\alpha}^{+}\right],
\qquad
\bar W_{\alpha}^{+}=(1-\alpha)W+\alpha W^+.
}
\]

因此，原方法并不需要被定义为“额外加一个物质导数”。它只是把普通 IG 中的当前弱
参考 `W`，逐渐替换为沿当前 guided characteristic 查询到的未来弱参考 `W^+`。

最简洁的 `alpha=1` 情况为

\[
\boxed{
G_{\mathrm{FWR}}(z,t)
=S(z,t)+\gamma(t)
\left[
S(z,t)-W(z+hG(z,t),t+h)
\right].
}
\]

这里称为 **Future-Weak Reference IG（FWR-IG）**。它只剩一个新超参数 `H`；
`h=min(H,0.5-t)` 仍用于防止未来查询跨过 IG 的关闭边界。

固定 `H=.046875` 的 alpha 扫描显示：

| `alpha` | FID-1K |
|---:|---:|
| 0，普通 IG | 64.851541 |
| .50 | 64.288947 |
| .75 | 63.407627 |
| .875 | 63.460761 |
| **1.00** | **63.350265** |
| 1.125 | 63.448452 |
| 1.25 | 63.772422 |
| 1.50 | 63.920000 |

最佳位置在 `alpha=1` 附近，而不是依赖精细的 current/future mixture。随后固定
`alpha=1` 扫 horizon：

| `H` | FID-1K |
|---:|---:|
| .015625 | 64.116869 |
| .0234375 | 63.955996 |
| **.03125** | **63.266269** |
| .0390625 | 63.297213 |
| .046875 | 63.349647 |
| .0546875 | 63.428203 |
| .0625 | 63.755598 |
| .078125 | 64.252683 |
| .09375 | 65.291841 |

一参数形式在 `H=.03125` 略优于原两参数 FMD 的本轮配对值 `63.270674`。短 horizon
先改善、过长 horizon 再恶化，说明收益不是“查询得越远越好”；它符合短期 anticipatory
reference 有益、长程 Euler 预测与跨时间场比较逐渐失真的图景。

正式 FID-5K 显示了一个 FID-1K 无法分辨的细微差别：

| sampling seed | depth4 IG | FWR-IG，`H=.03125, alpha=1` | 原 FMD-IG |
|---:|---:|---:|---:|
| 0 | 39.953584 | 38.193172 | **37.944441** |
| 1 | 39.939969 | 38.180135 | **37.929577** |
| 均值 | 39.946776 | 38.186654 | **37.937009** |

FWR-IG 相对 baseline 平均改善 `1.760123` FID（`4.41%`），说明一参数未来弱参考保留
了大部分收益；但它稳定地比原 FMD 慢约 `0.249645` FID。原 FMD 使用固定
`eta=.6`，对应第一段 `alpha=1`、第二段 `alpha=.6/.7≈.857`，因此数据提示第二段的
部分 current/future mixture 仍有小而可重复的贡献。故 FWR 是更干净的机制形式，当前
最佳质量仍属于原来的分段有效 alpha，而不是强行宣称简化式完全取代原式。

完整数据保存在
`docs/data/imagenet100_sit_foresight_material_ig/fid1k_lookahead_alpha.csv` 和
`docs/data/imagenet100_sit_foresight_material_ig/fid1k_lookahead_horizon.csv`；正式
5K 对照保存在 `fid5k_future_weak_reference.csv`。

## 十一、理论上应该怎样理解，哪些话不能说

### 11.1 一个 anchored implicit-reference 方程

把 strong field 视为生成动力学的 anchor，把 weak head 只视为 IG 的 negative/reference
branch，可以定义一个短期自洽方程：

\[
V^*
=S+\gamma
\left[
S-W(z+hV^*,t+h)
\right].
\]

普通 IG 用当前位置的 `W(z,t)`，相当于完全不让 reference 看到这次控制将把状态带到
哪里。FWR-IG 先用普通 IG 得到 predictor `G`，再计算

\[
V^{(1)}
=S+\gamma[S-W(z+hG,t+h)],
\]

它可以被理解为上述 implicit-reference 方程的一次 Picard/extragradient-style 更新。
这里的意义只是“reference 对所选择的短期运动自洽”，不是 CFG 的条件/无条件黄金路径，
也没有固定点收敛或最优生成质量定理。

不过，这个 reference fixed point 本身可以得到一个干净的适定性结论。固定 `(z,t)`，
定义

\[
\mathcal T(V)
=S+\gamma[S-W(z+hV,t+h)].
\]

若 `W` 关于状态是 `L_W`-Lipschitz，并且

\[
q=|\gamma|hL_W<1,
\]

则

\[
\|\mathcal T(V_1)-\mathcal T(V_2)\|
\le q\|V_1-V_2\|.
\]

因此 `T` 有唯一固定点，且以普通 IG `V^(0)=G` 初始化的 Picard 序列满足

\[
\|V^{(k)}-V^*\|\le q^k\|G-V^*\|.
\]

这解释了为什么短 horizon 是一个数学上自然的区域，也给出了“看得过远”会失去稳定
保证的明确条件。需要再次强调：该结论只证明 self-consistent reference 方程可解且迭代
稳定，不证明它的固定点比 ordinary IG 有更好的 FID；后者仍是经验事实。

### 11.2 物质导数是局部极限，不再是方法本体

当 `h` 很小时，

\[
W^+
=W+h(\partial_tW+J_zW\,G)+O(h^2),
\]

因此

\[
G_{\mathrm{FWR}}
=G-\gamma hD_GW+O(h^2).
\]

所以旧的 material-derivative 解释仍然正确，但它只是 Future-Weak Reference 的局部
展开。有限 `H` 下真正实现的是一次未来 reference query，而不是显式估计 Jacobian 或
假设线性响应。

### 11.3 与已有 lookahead 方法的边界

已有 diffusion guidance 工作会在同一个 latent 上改变 denoiser 的查询时间，或通过完整
未来去噪链估计终点 reward；FWR-IG 与它们的关键区别是：未来点同时改变**状态和时间**，
状态沿当前 IG characteristic 预测，并且只替换 internal weak/reference branch。这个区别
必须在相关工作中明确保留；目前结果还不足以宣称一般性的最优 lookahead guidance。

### 11.4 当前最合理、但尚未证明的机制假设

普通 IG 使用的 `W(z,t)` 是一个 myopic reference：它评价的是采取 guidance 之前的状态。
FWR 使用 `W(z+hG,t+h)`，让弱参考先看到 guidance 即将访问的、稍高信噪比的状态。
拆分实验说明，直接使用 future gap 会混入 `S^+-S`；精确消去 strong evolution 后，留下
的正是 weak reference 自身对这次短期运动的响应。这个“reference lag correction”与数据
一致，但要把它升级为质量定理，还需要给 weak head 一个可验证的误差/密度语义；当前
不能把 `W-W^+` 直接称为真实误差方向或 Bayes correction。

## 十二、从完整 characteristic 到 oblique posterior revision

继续把时间变化与状态变化拆开后，最佳查询点并不位于完整 guided characteristic 上。
定义

\[
q_{\alpha,h}(z,t)
=\left(z+\alpha hG(z,t),t+h\right),
\]

以及

\[
C_{\alpha,h}^{W}(z,t)
=W(z,t)-W\left(q_{\alpha,h}(z,t)\right).
\]

最终场为

\[
\boxed{
V(z,t)=G(z,t)+\eta C_{\alpha,h}^{W}(z,t)
}
\]

其中

\[
h=\min(H,0.5-t).
\]

沿冻结方向

\[
\ell(s)=(z+\alpha sG(z,t),t+s)
\]

有精确积分恒等式

\[
C_{\alpha,h}^{W}
=-
\int_0^h
\left[
\partial_tW+\alpha J_zW\,G(z,t)
\right]_{\ell(s)}ds.
\]

因此小 `h` 时

\[
C_{\alpha,h}^{W}
=-h(\partial_tW+\alpha J_zW\,G)+O(h^2).
\]

这里必须称为 **frozen-direction oblique derivative**。只有在 `alpha=1`、`G` 等于真实
边缘速度且取局部极限时，它才退化成标准 characteristic material derivative。正式最优
参数是

\[
H=0.02734375,\qquad \eta=1.65,\qquad \alpha=0.25.
\]

`alpha=.25` 不是事后从三维网格偶然挑出的比例。此前将时间项和状态项独立加权时，最佳
系数为 `1.2` 与 `.3`，其比值同样是 `.25`；随后联合 oblique 查询才再次选中 `.25`。

正式 FID-5K 为：

| sampling seed | FID | sFID | IS |
|---:|---:|---:|---:|
| 0 | **36.504419** | 69.073810 | 40.017986 |
| 1 | **36.496253** | 69.056045 | 40.025127 |
| 均值 | **36.500336** | **69.064928** | **40.021557** |

相对 depth4 IG 的均值 `39.946776`，FID 降低 `3.446440`，即 `8.63%`；相对上一版
完整-characteristic FMD-IG 的 `37.937009`，再降低 `1.436673`，即 `3.79%`。

## 十三、线性 bridge 下的 Bayes 速度与真实曲率来源

设

\[
Z_t=(1-t)E+tX=E+tU,\qquad U=X-E,
\]

其中 `E` 是 source noise，`X` 是 data endpoint。velocity MSE 的 population minimizer
为

\[
\boxed{v_t^*(z)=\mathbb E[U\mid Z_t=z].}
\]

这来自平方损失条件期望投影，是 Flow Matching / stochastic interpolant 的标准结果。
本实验的 `S` 与 `W` 都是对这个条件速度的有限模型估计；`W` 额外受 frozen depth-4
特征与轻量 readout 函数类限制。因此下文把它称为“弱后验速度估计”，但不假定它已经
等于 population conditional expectation。

边缘速度本身为什么会沿路径变化，可以由一个严格的动量方程回答。令

\[
\Sigma_t(z)=\operatorname{Cov}(U\mid Z_t=z).
\]

联合密度满足速度为 `U` 的 Liouville 方程。对 `U` 积分分别得到连续性方程与动量方程：

\[
\partial_tp+\nabla\cdot(pv^*)=0,
\]

\[
\partial_t(pv^*)
+\nabla\cdot\left[p(v^*v^{*\top}+\Sigma)\right]=0.
\]

利用第一式消去 `partial_t p` 后得到

\[
\boxed{
(\partial_t+v^*\cdot\nabla)v^*
=-\frac1p\nabla\cdot(p\Sigma).
}
\]

这说明即使每条 conditional path 的 `U=X-E` 完全恒定，许多 conditional trajectories
在同一点叠加后，边缘 Bayes velocity 仍会因为条件速度不确定性而弯曲。该结论也见于
[An Eulerian Perspective on Straight-Line Sampling](https://arxiv.org/abs/2510.11657)
与 [Isokinetic Flow Matching](https://arxiv.org/abs/2604.04491)。

若取理想局部极限 `W=G=v*`、`alpha=1`，则

\[
C_{1,h}^{W}
=\frac{h}{p}\nabla\cdot(p\Sigma)+O(h^2).
\]

所以 finite weak revision 测到的并非纯数值误差；其 population 对应量包含真实的后验
不确定性曲率。当前最优 `alpha=.25` 与 finite `h` 不满足上述理想极限，因此这个公式只
提供统计来源，不构成“该修正必然降低 FID”的定理。

## 十四、端点后验对比不变量

对任意 velocity estimate `W`，定义其在线性 bridge 下诱导的端点估计：

\[
\widehat X_W(z,t)=z+(1-t)W(z,t),
\]

\[
\widehat E_W(z,t)=z-tW(z,t).
\]

当 `W=v*` 时，这两式分别严格等于
`E[X|Z_t=z]` 与 `E[E|Z_t=z]`。对当前点 `p=(z,t)` 和任意查询点
`q=(z',tau)`，定义

\[
C_X=\widehat X_W(p)-\widehat X_W(q),
\]

\[
C_{-E}=\widehat E_W(q)-\widehat E_W(p).
\]

直接代数化简得到

\[
\boxed{C_X+C_{-E}=W(p)-W(q).}
\]

这个恒等式对任意 `W`、任意两个时空点都成立，不需要 Bayes 最优、小 `h` 或 trajectory
假设。

更关键的是它具有唯一性。一般线性组合为

\[
aC_X+bC_{-E}.
\]

其中人工查询位移 `z-z'` 的系数恰好为

\[
a-b.
\]

所以要让结果不含共同坐标位移，必须且只需

\[
a=b.
\]

于是除整体尺度外，唯一的 query-displacement-invariant endpoint contrast 就是

\[
C_X+C_{-E}=W(p)-W(q).
\]

直觉上，单独比较 clean endpoint 或 noise endpoint 会把两件事情混在一起：模型对端点
分解的判断发生了变化，以及我们人为把查询坐标从 `z` 移到了 `z'`。等权端点对比把后一
项严格约掉，只保留模型对同一个 `U=X-E` 的预测修正。

FID-1K 因果拆分为：

| 修正项 | FID-1K |
|---|---:|
| 完整 invariant velocity contrast | **61.9093** |
| clean endpoint change | 63.6491 |
| clean，逐样本 RMS 匹配 | 63.5848 |
| negative-noise endpoint change | 64.8581 |
| negative-noise，逐样本 RMS 匹配 | 66.0669 |

clean 项平均 RMS 约 `.051`，已经包含完整项 `.055` 的绝大多数欧氏能量；negative-noise
项平均 RMS 仅约 `.021`，两者平均 cosine 接近零。即便如此，clean 单项仍比完整对比差
约 `1.74 FID`，证明结果不能由“最大能量分量”解释。

进一步令

\[
C_\lambda=C_X+\lambda C_{-E}
=C_1+(\lambda-1)C_{-E}.
\]

七点权重扫描为：

| `lambda` | FID-1K |
|---:|---:|
| .50 | 62.6798 |
| .75 | 62.2147 |
| .875 | 62.2095 |
| **1.00** | **61.9100** |
| 1.125 | 61.9124 |
| 1.25 | 62.0114 |
| 1.50 | 62.0779 |

理论唯一点 `lambda=1` 位于本轮经验最低点。`1.125` 与它只差 `.0024`，FID-1K 无法
统计地区分二者，因此不能声称存在尖锐唯一极小值；但从两侧偏离更远均恶化，并且单项与
RMS 控制一致支持坐标消去结构。

## 十五、几个看似更“正规”的解释为何被实验排除

### 15.1 不是无穷小导数估计越准越好

缩短 horizon 并相应放大 `eta` 的 directional-derivative limit sweep，最佳仅为
`61.9646`，没有超过 finite `h=.02734375` 的 `61.9103`。

### 15.2 finite curvature 不是应该删除的截断误差

用半步 Richardson 构造一阶方向项，再令 `lambda` 控制 finite-horizon 曲率：

| curvature weight | FID-1K |
|---:|---:|
| 0，去掉二阶 finite term | 62.4584 |
| .5 | 62.0626 |
| .75 | 62.0682 |
| **1，原 finite operator** | **61.9089** |
| 1.25 | 62.2497 |
| 1.5 | 62.5487 |
| 2 | 62.6557 |

因此有限尺度本身是有用信号，不应把当前方法包装成“更准确的数值微分”。

### 15.3 self-consistent fixed point 不等于质量最优

以

\[
V_0=G,
\qquad
V_{k+1}=G+\eta[W(z,t)-W(z+\alpha hV_k,t+h)]
\]

做 Picard 重算时，`K=2/3` 分别得到 `62.1353/62.1725`，均差于一次 predictor query。
收缩映射只能保证方程适定或迭代收敛，不能保证该固定点对应更好的生成分布。

### 15.4 不能只用 `eta/gamma` 解释成统一弱参考外推率

固定 dimensionless forecast ratio `r=eta/gamma(t)` 的扫描最佳为 `62.0091`，没有超过
固定 velocity-correction coefficient `eta=1.65`。因此 correction 与原 static IG gap
是两个不同作用项，不应强行绑成一个比例。

## 十六、时间贡献与当前方法定义

保持 invariant endpoint contrast、`H/alpha` 不变，只分开两段系数：

| `(eta_early, eta_mid)` | FID-1K |
|---|---:|
| `(1.65, 0)` | 62.0239 |
| `(0, 1.65)` | 65.2974 |
| `(1.35, 1.35)` | 62.1328 |
| `(1.65, 1.35)` | 62.0032 |
| **`(1.65, 1.65)`** | **61.9090** |
| `(1.65, 1.95)` | 61.9962 |
| `(1.95, 1.65)` | 62.0830 |

主要增益来自 `[0,.25)`，但 `[.25,.5)` 的同强度修正还能稳定补充约 `.11 FID`；独立
拉高或压低两段都没有超过常数 `eta`。

在此基础上，将原 IG 的两段 `gamma=(.6,.7)` 分别乘以 `.9/1/1.1`。九点局部网格中，
未经重调的 `(1,1)` 仍以 `61.9100` 最低；其余条件位于 `61.9654` 到 `62.3290`。因此
PR-IG 的收益不是通过补偿原 IG 的明显 coefficient miscalibration 获得的。

综合严格结构与实验，当前最准确的方法名是 **Posterior-Revision Internal Guidance
(PR-IG)**：

\[
\boxed{
\begin{aligned}
G(z,t)&=S(z,t)+\gamma(t)[S(z,t)-W(z,t)],\\
q&=(z+\alpha hG(z,t),t+h),\\
V(z,t)&=G(z,t)+\eta[W(z,t)-W(q)],
\end{aligned}
}
\]

其中 `W(z,t)-W(q)` 同时具有三种互相一致但层次不同的解释：

1. **严格代数层**：它是唯一消除 query coordinate displacement 的线性端点后验对比；
2. **局部微分层**：它是弱速度场的 finite oblique posterior revision；
3. **population 统计层**：在理想 characteristic 极限中，其来源包含条件速度协方差导致的
   边缘流曲率。

只有第一层对任意有限模型严格成立。第二层是精确路径积分加局部展开。第三层需要 Bayes
最优与 characteristic 极限。FID 改善则是实验结论，不由前三式自动保证。

相关边界：原始 [Internal Guidance](https://arxiv.org/abs/2512.24176) 只在同一时空点
外推 deep/internal outputs；[Self-Guidance](https://arxiv.org/abs/2412.05827) 比较同一
状态在不同噪声时间的模型输出；PR-IG 使用 internal weak head，并同时做时间与部分状态的
oblique query，再以 endpoint-invariant contrast 更新原 IG。
