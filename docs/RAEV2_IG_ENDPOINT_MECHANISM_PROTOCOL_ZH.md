# RAEv2 Internal Guidance 终点机制验证协议

## 当前已知与尚未知

现有两组 5000 样本实验可靠地说明：

- 在中低噪声时刻，官方 IG 的采样状态比 Full 更容易与训练路径状态区分。
- 同一状态直接经过冻结 decoder 后，IG 的 Inception 统计反而更接近图像参考。
- 官方 IG 的 5000 样本终点 FID 在两个 seed 上都优于 Full。
- RAEv2 的 `x` prediction 参数化和 shifted solver grid 会把 IG 的局部状态更新集中到最后几个有效步骤。

这些证据目前只能称为 **latent 探针与 decoder-Inception 度量反转**。它们尚未证明：

- IG 的终点 latent 仍比 Full 更偏离 encoder latent 分布；
- IG 当前一步的 clean prediction 本身更好；
- 最后几步的数值放大导致了终点 FID 改善；
- IG 显式或有意利用了 decoder 几何。

## 实验一：Predicted-clean 2x2

在完全相同的初始噪声和类别下生成两条状态轨迹：

- `x_full(t)`：IG scale 为 1；
- `x_ig(t)`：官方 IG scale 为 1.78。

在每个状态上同时读取模型的 Full 和 Base clean prediction，并构造官方 IG prediction：

```text
full_on_full = Full(x_full, t)
ig_on_full   = Base(x_full, t) + 1.78 * (Full(x_full, t) - Base(x_full, t))
full_on_ig   = Full(x_ig, t)
ig_on_ig     = Base(x_ig, t) + 1.78 * (Full(x_ig, t) - Base(x_ig, t))
```

四组 latent 全部经过同一个冻结 RAE decoder 和同一个 Inception extractor。

### 因果拆分

- `ig_on_full - full_on_full`：Full 状态上只改变当前预测头。
- `ig_on_ig - full_on_ig`：IG 状态上只改变当前预测头。
- `full_on_ig - full_on_full`：固定 Full 头，只改变此前累积的轨迹。
- `ig_on_ig - ig_on_full`：固定 IG 头，只改变此前累积的轨迹。
- `ig_on_ig - full_on_full`：官方 on-policy 总效应。

在 `t=1` 两条轨迹的状态必须完全相同；相同 head 的 Inception 特征也必须逐元素相同。否则实验无效。

## 实验二：终点 latent AUC 与矩统计

比较：

```text
p_0 = E(x_real)
q_0_full = Full 采样终点
q_0_ig = 官方 IG 采样终点
```

使用按 ImageNet 类别完全隔离的 train/test split 拟合 diagonal LDA，并在未见类别上报告 AUC。同时报告：

- latent 均值偏移；
- 对角方差相对偏移；
- IG/Full 相对真实 latent 的总方差比。

AUC 只表示一个训练集可泛化的线性区分方向，不等价于完整概率距离。矩统计用于判断结果是否主要由简单的均值或方差变化解释。

## 实验三：便宜的统计混杂排除

- 汇总 decoder raw min/max 和 clamp 到 `[0,1]` 的像素比例。
- 对已保存的 Inception 特征计算标准 polynomial KID。
- 计算 kNN manifold precision/recall，判断 FID 改善是否以 recall 或多样性为代价。
- 所有后处理复用已保存特征，不重新采样，也不改变主实验样本。

## 预先固定的解释规则

### 结果 A：终点 latent 更远，same-state IG predicted-clean 更好

支持：IG 不是训练路径分布校正，而是当前双头方向在 decoder 图像度量下更有利。

仍不支持：decoder-aware。decoder 没有参与 IG 训练，更准确的名称是 decoder-compatible 或 decoder-mediated。

下一步：做 latent moment matching 和 decoder/noise-training 对照。

### 结果 B：终点 latent 更远，same-state IG 更差，但 on-policy 或最终图像更好

支持：当前局部扰动必须经过后续动力学或轨迹反馈才能产生收益。

下一步：做 IG 时间窗口、单步干预和 solver/NFE 因果实验。

### 结果 C：终点 latent 更近

说明中途的度量反转不是终点机制。IG 可能先偏离、后修正 latent 分布。

下一步：定位回归发生在哪几个 solver step，而不能宣称终点 decoder 反转。

### 结果 D：FID 提升同时 precision 上升、recall 明显下降

说明 IG 可能主要做质量倾斜或模态收缩，而不是更准确地拟合完整数据分布。

下一步：报告 precision-recall 权衡，并避免只优化 FID。

## 暂缓项

在上述链条闭合前，不启动 controller、PID/MPC、复杂 guidance schedule 或多 backbone 大扫荡。`t/h` 补偿只可作为固定网格诊断；若后续研究连续时间调度，优先使用不依赖 NFE 的 `t` 补偿，并用控制能量匹配不同 schedule。

## 2026-08-02 最终运行状态

自动流水线的所有阶段均以退出码 0 完成：两个 5000 样本 endpoint runs、两个 predicted-clean 2x2 runs、跨 seed 汇总、KID 和 precision/recall。两个 endpoint runs 的 `t=1` 硬对照均逐元素通过。

### Endpoint latent

| seed | Full AUC | IG AUC | IG - Full | 95% class-cluster bootstrap CI |
|---:|---:|---:|---:|---:|
| `20260801` | 0.717025 | 0.884885 | +0.167860 | [0.146625, 0.189302] |
| `20260802` | 0.735767 | 0.887111 | +0.151344 | [0.129886, 0.172766] |
| mean | 0.726396 | 0.885998 | +0.159602 | - |

IG 终点在两个 seed 上都比 Full 更容易与真实 encoder latent 区分，因此不支持“IG 把生成 latent 分布纠正回训练路径分布”。但矩统计显示这种偏离不是简单的整体方差恶化：

- Full 的平均方差比约为 `0.9662`，IG 为 `0.9963`，IG 更接近真实 latent 的总方差；
- Full 的对角方差相对误差约为 `0.0633`，IG 为 `0.0583`；
- Full 的均值偏移 RMS 约为 `0.0396`，IG 增至 `0.0529`。

当前最准确的表述是：IG 修复了 Full 的欠方差，同时引入更强、可跨未见类别泛化的方向性均值偏移。FID 改善更像结构化质量倾斜，而不是完整分布校正或简单 mode collapse。该结论只覆盖线性探针、均值和对角方差，不等价于完整概率距离。

### Predicted-clean 2x2

在 `t≈0.14/0.20/0.41`，固定 Full 头、只替换为 IG 历史状态时，跨 seed 平均 ImageNet FID 分别变化：

```text
-0.4786 / -0.4510 / -0.3625
```

在实际 IG 状态上继续把当前 Full 头替换为 IG 头时，FID 分别变化：

```text
+0.0434 / +0.0540 / +0.0089
```

因此主要收益稳定来自此前累积的轨迹历史；低噪声当前 IG 头没有即时收益，反而轻微过冲。KID 的方向与此一致，但差值很小，主要作为支持性证据。precision/recall 变化幅度小且存在 seed 波动，只能排除明显的大规模 mode collapse，尚不能把收益归因于单一 precision 或 recall 变化。

综合结果符合预注册结果 B：IG 不是逐步把状态拉回训练路径的局部误差校正器，而是通过累积、状态依赖的轨迹反馈到达 decoder/FID 更偏好的区域。

### 可重入续跑

统一入口：

```bash
python experiments/run_raev2_ig_endpoint_pipeline.py
```

后台会话：`raev2_ig_endpoint_pipeline_v1`（已正常完成并退出）。

状态文件：

```text
~/data/eqvae/experiments/raev2_ig_endpoint_pipeline_v1/pipeline_status.json
```

每个阶段在自己的输出目录写入 `pipeline_run.log` 和 `exit_code`。续跑器只有在退出码为 0 且所有必要产物存在时才跳过该阶段；失败会立即停止，不会用残缺结果继续汇总。固定顺序为：

1. 两个 seed 的 endpoint latent AUC 与矩统计；
2. endpoint 跨 seed 汇总；
3. 两个 seed 的 predicted-clean 2x2；
4. predicted-clean 跨 seed 汇总；
5. 两个 seed 的 post-hoc KID 与 precision/recall。

## 暂存的数值解释边界

对 RAEv2 源码的复核表明，当前两个输出并不是同一算子的相邻迭代：

- Base 在第 8 个 encoder block 后截取，并使用独立的 `base_final_layer`；
- Full 继续通过全部 28 个 encoder blocks、`s_projector`、2 个 decoder blocks 和独立的 `final_layer`；
- 两条读出路径不等深，也不共享 readout。

因此，当前 `Full - Base` 可以作为经验 guidance direction，但不能直接套用 Richardson、Aitken 或固定点收敛率公式。若以后检验“深度迭代加速”假设，至少需要三个等间隔、共享或严格校准 readout 的输出，并先验证连续差值共线、范数收缩和 held-out 校准；不能为了符合理论而先加 loss 强行制造这些现象。

附件提出的 sampler impulse 分解在当前 `x` prediction + Euler sampler 下是正确的。同一状态上的 clean prediction 改变量 `delta_x0` 对下一状态的直接影响为：

```text
delta_x_next = (h / max(t, 0.05)) * delta_x0
```

但这只刻画单步数值增益，不保证方向有益，也不等于完整轨迹的最终影响。若现有复验闭合后继续，优先做等 impulse 的正向、反向和匹配范数随机方向对照；暂不实现自适应 `w(t)`、Aitken 或 recurrent tail。

统计审计还补充了两点：

- 跨 seed AUC 置信区间按 held-out ImageNet 类别整组 bootstrap，而不是把同类图片当独立样本；
- 汇总图横轴根据实际时刻自动确定，保证真正的 `t=0` endpoint 不会被隐藏。
