# 噪声分辨的生成责任曲线：预注册协议

## 1. 研究问题

本研究不把 DINOv2、SigLIP 或 VAE latent 当作候选方法，而只把现有模型
当作测量工具。目标是先判断：条件 latent 在随机 decoder 的不同噪声阶段
是否承担可区分的生成责任。

对同一真实图像 `x0`、同一噪声 `epsilon` 和同一噪声时刻 `t`，比较：

- `real`：decoder 接收与 `x0` 配对的 latent；
- `null`：decoder 接收训练时定义的空 latent；
- `shuffle`：decoder 接收同一 batch 内另一张图像的真实 latent。

所有非图像条件（类别、文本、分辨率和退化强度）必须保持完全相同。
定义逐样本预测损失：

```text
L_real(t)    = loss(target, prediction(x_t, t, z))
L_null(t)    = loss(target, prediction(x_t, t, null))
L_shuffle(t) = loss(target, prediction(x_t, t, shuffle(z)))
```

以及：

```text
Delta_null(t)    = L_null(t) - L_real(t)
Delta_shuffle(t) = L_shuffle(t) - L_real(t)
```

`Delta_shuffle` 是主指标，因为 shuffled latent 仍来自真实边缘分布；
`Delta_null` 只有在 checkpoint 的训练确实包含 latent dropout 时才能解释。

## 2. 实验顺序

1. 在公开 PiD checkpoint 上做不训练的快速筛查。
2. 若筛查显示指标可测且通过实现复核，在小型从头训练模型上做受控验证。
3. 只有受控验证通过，才训练两个候选 bottleneck 的 latent prior。
4. 只有完整两阶段生成通过固定总算力门槛，才做端到端联合训练。

后一步不得在前一步未通过时提前启动。

## 3. 现有模型筛查的预注册预测

比较对象按可获得 checkpoint 决定，优先顺序为：

- PiD + DINOv2 RAE latent；
- PiD + SigLIP latent；
- PiD + FLUX/SD3 VAE latent。

预测 P1：所有有效条件模型的 `Delta_shuffle` 平均值应大于零。

预测 P2：VAE latent 在低噪声阶段仍提供明显帮助；语义 latent 的相对帮助
更集中在中高噪声阶段。这里比较归一化曲线形状，不直接比较不同 checkpoint
的绝对 MSE。

预测 P3：`shuffle` 比 `null` 更保守。若 `Delta_null` 很大而
`Delta_shuffle` 接近零，不解释成 latent 信息丰富，而解释为 null/OOD 或
decoder 只使用 latent 的总体统计。

预测 P4：将正确 latent 重复作为对照时，两次预测和损失必须在数值精度内
一致；改变 batch 顺序后，聚合结果不得改变。

## 4. PiD 结果的解释边界

公开 DINOv2/SigLIP PiD 是四步蒸馏 checkpoint。它只能提供离散支持时刻的
条件敏感性曲线，不能单独视为连续时间 MMSE 或互信息估计。只有满足以下
条件时，`null` 分支才可使用：

- 对应 checkpoint 的训练配置包含 latent condition dropout；或
- 官方蒸馏目标明确训练了该无条件分支。

若无法从配置或 checkpoint 元数据证明，则只报告 `shuffle`，并把 `null`
标记为 OOD sensitivity。

## 5. 实现复核清单

任何预注册预测失败后，必须暂停后续实验，依次检查：

1. 三个分支是否复用了完全相同的 `x_t`、噪声和时刻；
2. target 是否符合 checkpoint 的参数化（`x0`、velocity 或 epsilon）；
3. latent 的归一化、空间尺寸、通道顺序和退化强度是否一致；
4. shuffle 是否为无固定点置换，标签/文本是否保持不变；
5. null 是否确实在训练支持内；
6. 蒸馏模型是否只支持固定时刻；
7. 结论是否由少量离群样本、分辨率或损失尺度造成；
8. 至少重复两个 seed，并查看逐样本 paired difference。

## 6. 硬停止规则

若预测失败且上述检查全部通过，则把它记为真实反常现象，立即停止后续
训练，不通过增加模型、数据、epoch、权重或事后重定义指标挽救。此时只做：

- 固化代码、环境、checkpoint 哈希和原始逐样本结果；
- 记录预测、实测、复核证据和仍无法解释的理论缺口；
- 总结该现象对“latent 负责中高噪声语义、decoder 负责低噪声细节”假设的
  支持或否定。

## 7. 进入小型受控实验的门槛

现有模型筛查只有同时满足以下条件才通过：

- 至少两个 latent 家族的 `Delta_shuffle` 在重复 seed 下稳定为正；
- 曲线不是 latent 范数、模型输出尺度或 null OOD 的平凡结果；
- 至少观察到一种可复现的曲线形状差异；
- identity、batch permutation 和相同噪声控制全部通过。

通过只表示该诊断值得在受控模型中验证，不表示论文假设已经成立。
