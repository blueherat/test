# RAE 生成研究 Claim Matrix

## 结论

截至 2026 年 7 月，围绕“什么 latent 更适合生成”的常见解释已经非常拥挤。现有实验不能再支持一个宽泛的
“跨层可预测性决定生成友好性”主张。它最值得保留的用途，是帮助定位一个更具体的断层：

> stage-2 在 latent 中产生什么方向的误差，以及 decoder 对这些方向有多敏感，是两个不同问题。

当前没有已经成立的 ICLR 级方法空缺。最值得进入下一轮低成本验证的候选，是**生成误差与 autoencoder 循环
不一致方向的耦合**。这比直接声称
“切向/法向失配”更严谨：少量增强或近邻只能覆盖部分真实变化，其正交补并不自动等于完整数据流形的法向空间。
在这个假设通过冻结模型诊断之前，不训练 adapter、decoder 或 DiT。

## 1. 已有工作的覆盖范围

| 研究维度 | 已有强证据或方法 | 当前实验与其重叠 | 剩余问题 |
|---|---|---|---|
| stage-2 容量 | RAE、RAE scaling、When Worse is Better | 小 `DiTDH-S` 的方向学习差异 | 同一 tokenizer 的误差方向是否随模型宽度改变 |
| 协方差条件数 | Preconditioned Flow Matching | FxLMS toy、白化和分频收敛 | 基本已覆盖，不宜作为主线 |
| 全局语义 | VA-VAE、MAETok、RAE、AlignTok | 跨层 predictability 可能代理语义 | 控制线性可分性后是否仍有独立作用 |
| patch 空间结构 | REPA、iREPA、RAEv2 | predictability 可能代理空间组织 | 控制 LDS/空间自相似后是否仍有独立作用 |
| latent 频谱 | Improving Diffusability、DC-AE 1.5 | 频带和方向加权实验 | 基本已覆盖，除非出现新的因果量 |
| corruption 鲁棒性 | RAE noise decoder、l-DeTok、LV-RAE | adapter 及噪声鲁棒设想 | 各向同性噪声是否错误压制数据支持方向 |
| decoder perceptual geometry | LPL、LV-RAE、latent pullback metric | 24-block sensitivity atlas | stage-2 实际误差是否落在 decoder 高增益方向 |
| 多层 encoder 信息 | VFM-VAE、RAEv2 | 跨层线性可预测性与 layerwise atlas | predictability 是否超出简单多层融合收益 |
| 几何等变 | EQ-VAE、VFM-VAE | 旋转/翻转和群作用诊断 | final RAE 没有干净群表示；不宜强行施加 |
| off-manifold generation | PS-VAE、LV-RAE | decoder OOD、条纹与 sensitivity atlas | full RAE 中直接量化误差方向，而非再声称发现 off-manifold |
| 局部流形几何 | geometry-preserving AE、latent vector field、相关 Jacobian 工作 | 尚未测量生成误差的 cycle consistency | cycle rejection 能否预测 decoder 伪影和可干预性 |

## 2. 现有证据应如何归类

### 已确认的事实

1. 当前 DINOv2-B RAE decoder 对不同等范数 latent 方向的响应差异很大。
2. 24 个固定子空间中，held-out 跨层可预测性比方差更能解释 decoder sensitivity。
3. 静态 predictability metric 和 decoder-atlas oracle 都不能近似逐样本 LPL gradient。
4. SPC 在 5 个 seed 上均未同时改善 FID 和 KID，平均 FID 反而恶化约 `6.30`。
5. 小 `DiTDH-S` 在高噪声和低噪声阶段呈现不同方向偏好。

### 尚未确认的解释

- predictability 代表稳定语义；
- predictability 是生成模型的统一信任轴；
- 高 decoder sensitivity 就是生成失败的主因；
- 高噪声到低噪声存在已识别的因果“接力”；
- 上述现象可从 `DiTDH-S` 外推到论文规模模型。

## 3. 四个候选主张的客观状态

| 候选主张 | 当前状态 | 判断 |
|---|---|---|
| 跨层可预测性是普适的 generative-friendly 指标 | 未控制空间、语义和流形切向性 | 证据不足，暂不作为主线 |
| 静态 decoder metric 可替代 perceptual loss | oracle 和 predictability proxy 均失败 | 已被当前实验否定 |
| 小模型中的方向偏好揭示普适 diffusion 机制 | 存在明显 stage-2 容量混杂 | 未跨尺度前不能成立 |
| 生成误差的 cycle-rejected 分量被 decoder 选择性放大 | 与现象相容；PS-VAE 已覆盖宽泛 off-manifold 主张 | 可做冻结诊断，不能直接当新颖 claim |

## 4. 下一步：不训练的三道门

### Gate A：`E(D(z))` 能否作为经验局部投影

定义冻结循环：

```text
F(z) = E(D(z))
Delta_F(z, delta) = F(z + delta) - F(z)
```

使用两类数据支持的 secant `delta_data`：同一图像的轻量增强差分，以及高语义相似近邻的 latent 差分。
再构造维数、RMS 和频谱匹配的随机方向 `delta_rand`。比较：

```text
retention(delta) = ||Delta_F(z, delta)|| / ||delta||
alignment(delta) = cosine(Delta_F(z, delta), delta)
```

验收门槛：

- clean latent 的固定点误差 `||F(z)-z||/||z||` 稳定且显著小于生成 latent；
- `delta_data` 的 retention 和 alignment 显著高于匹配随机方向；
- `eps=0.5%` 与 `1%` 的排序一致，bootstrap 95% CI 不由少数样本驱动；
- 轻量增强与近邻 secant 得到相同趋势。

若循环不能区分数据支持方向和匹配随机方向，则 `F` 不能当作局部投影，立即停止这一解释。

### Gate B：真实 stage-2 误差是否被循环拒绝

在冻结的官方或已有生成 checkpoint 上，对同一批样本、多个噪声时刻计算 clean estimate 误差：

```text
e_t = z0_hat(t) - z
cycle_retention = ||F(z + e_t) - F(z)|| / ||e_t||
cycle_alignment = cosine(F(z + e_t) - F(z), e_t)
```

同时计算该误差经过 decoder 后的视觉变化，并与等范数的 data secant 和匹配随机扰动比较。对无配对的无条件生成
latent `z_gen`，另报告 `||F(z_gen)-z_gen||/||z_gen||` 相对 clean latent 的分布偏移。

验收门槛：

- stage-2 误差的 cycle retention/alignment 显著低于 data secant 基线；
- cycle rejection 或 decoder 放大至少有一项随采样后期上升；
- 结果在不少于 3 个 checkpoint 或 seed 上方向一致；
- 使用 S 与更宽模型时结论不反转。

额外做一个无需训练的强检验：比较 `D(z_gen)` 与 `D(F(z_gen))`。必须用 clean reconstruction 同样经过一次循环，
校准额外 encode-decode 自身造成的画质损失。若 cycle 后处理在校准后仍不能改善任何生成指标，它不完全否定
off-manifold 现象，但会显著降低其作为方法主线的价值。

### Gate C：predictability 是否还有独立信息

在相同子空间上联合控制：

- latent variance；
- token mean 占比；
- patch 空间自相似或 LDS；
- 简单语义 probing；
- cycle retention/alignment；
- basis family。

只在 held-out 图像、held-out basis family 和跨 encoder 设置中报告 partial effect。

验收门槛：控制后 predictability 对 decoder gain 或 stage-2 error 的增量 `R^2 >= 0.10`，且 leave-one-family-out
系数符号稳定。未通过则把 predictability 降级为已有结构的代理量，不再研究它本身。

## 5. 通过 Gate 后唯一允许的方法实验

若 Gate A 和 B 同时通过，才考虑一个**cycle-rejected 方向选择性 decoder robustness** 对照。方向由冻结 teacher
autoencoder 的局部循环响应定义，避免训练中的 moving target：

```text
L = L_clean_reconstruction
  + lambda * distance(D_student(z + eps*u_rejected), D_teacher(z))
```

encoder 与 stage-2 冻结，只微调 decoder。关键对照不是“无正则”，而是与相同 `eps`、相同算力的各向同性
latent noise augmentation 比较。循环保留的数据支持方向不施加不变性约束，以免抹去真实语义变化和细节。

方法实验的启动条件：

- Gate A/B 全部通过；
- 与 PS-VAE、l-DeTok、RAE noise augmentation、LV-RAE 的差异能用一句话准确说明；
- 预先固定 perturbation norm、loss 权重和停止规则；
- 先做 reconstruction、off-manifold robustness 和已有 checkpoint 的 5k generation gate。

方法实验的成功门槛：

- 相对各向同性 noise baseline，clean rFID 不更差；
- 对真实 stage-2 endpoint latent 的 decoded FID/KID 更好；
- 至少 3 seeds 方向一致；
- 最终 50k gFID 有实质改善，而非只改善局部 proxy。

## 6. 推荐研究顺序

1. 只实现 Gate A，规模 `128-256` 张 ImageNet validation，预计小时级而非训练级成本。
2. Gate A 通过后复用已有 rollout/checkpoint 做 Gate B，不训练新模型。
3. 并行完成 Gate C，正式决定是否关闭 cross-layer predictability 叙事。
4. 只有 A/B 通过才设计 decoder 微调；C 是否通过只决定 predictability 是否保留为论文机制。

这条路线仍然紧贴生成模型：它直接研究 stage-2 的真实 latent 误差为何经 decoder 变成图像伪影，并以 gFID
作为最终验收，而不是把研究再次带回抽象的群结构或纯表示分析。
