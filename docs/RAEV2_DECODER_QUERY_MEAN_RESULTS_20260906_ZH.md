# 两层 DDT query 均值：结构诊断完成

日期：2026-09-06。**固定的 query 信息删除在 RAEv2 上实现正确，并产生明显不同于原 Internal Guidance 的方向。** 它没有提供误差兼容或图像质量证据，本轮不据此选择 gain、时间窗口或启动 1K。至少 5% 的公平成本 FID 目标仍未完成。

## 固定机制与执行

依据 [PAG / SEG 阅读](RAEV2_GUIDANCE_READING_ATTENTION_WEAK_20260906_ZH.md)与[冻结协议](RAEV2_DECODER_QUERY_MEAN_PROTOCOL_20260906_ZH.md)，只在全部两个 DDT decoder blocks 中，将原 q RMSNorm 和 RoPE 后的 Q 替换为每样本、每 head 的 token 均值 JQ，不重新归一化。28 层 encoder、原 Full 读出及各弱层自身的 K/V 计算保持不变；第二弱层从第一弱层输出重新计算 Q/K/V。

JQ 是行常量矩阵集合上的唯一 Frobenius 投影；没有 mask 或额外位置 bias 时，弱 attention 行是同一输入原 attention 行的反向 KL 重心。这是局部信息删除的保证，不是完整网络能量曲率、weak/Full 误差共线或 FID 保证。

输入保持为固定 10 个时间快照、teacher/rollout 两域、8 个 ID，共 20 个 B8 输入、160 条相关记录。t=1 两域输入重复；不把它们当成独立样本。历史状态使用旧 FP32 guidance 合成和 TF32on；本轮仅读取其状态和身份，重新计算当前 EMA100080、FP32 权重、BF16 autocast、TF32off 的 Full/Base/Weak。未生成新噪声、轨迹、图像或 FID。

CPU 实现检查 10 项通过，其中最初 8 项另由独立审阅者执行通过；新增两项覆盖实际调用计数、漏算/重复调用与 hook 移除。随后 v1 启动在任何 CUDA 调用前遇到文件哈希 helper 的路径类型错误。仅将两处字符串路径包装成 Path，保留 v1 请求和失败记录，再以相同公式、输入、精度准备 v2。最终 v2 完整退出成功。

## 结构结果

全部 20 个输入的显式 Full/Base 与原 `model.forward` 输出 dtype 和数值逐位相同；首个弱 decoder 的 K/V 与 Full 首层逐位相同。全部实际模型、encoder、decoder q/MLP 与读出调用次数均与冻结计划一致。

共保存 10240 条分支—层—图像—head 统计，其中弱分支 5120 条：

| 固定检查 | 结果 |
|---|---:|
| 弱 Q 的 query 行最大差 | 全部 0 |
| 实际 SDPA 输出的 query 行最大差 | 全部 0 |
| 显式参照 attention 概率的 query 行最大差 | 全部 0 |
| 弱 attention 到 uniform keys 的 KL | 最小 0.00413028；均值 0.14513548；最大 1.13944976 |
| 实际有效行与理论反向 KL 重心的 L1 差 | 均值 0.00056694；最大 0.00167759 |
| 实际有效行到该重心的 KL | 最大 0.00000200855 |

这验证的是：各 query 位置的检索分布相同，但 keys 仍有偏好。残差、MLP、调制和完整 encoder 仍能保留空间信息，不能称整个弱模型输出为常量图。原 post-RoPE Q/K 为 FP32，SDPA autocast 的有效输入为 BF16；概率统计由这些有效 Q/K 在 CPU FP64 重算，不能称直接读取 fused kernel 内部概率。求均值与 BF16 舍入不交换，理论重心存在上述有限精度差异。

## 与原 gap 的关系

令 Δ=Full−Weak，d=Full−Base；两者均为本轮 BF16 head 转 FP32 后的差，尚未定义 native BF16 guidance 合成。全部 160 条的 Δ、d 均非零，逐条余弦范围为 0.08188–0.39370，范数比为 0.45247–4.56047。

汇总定义为 `sqrt(Σ||Δ||²/Σ||d||²)` 和 `Σ〈Δ,d〉/sqrt(Σ||Δ||²Σ||d||²)`，得到 **范数比 2.15912、余弦 0.18254**。将每条 Δ 分解到其自身 d 方向与正交补后，正交能量占全部 Δ 能量的 **95.3291%**。该百分比不是对一个统一方向作投影，也不是独立重复实验的显著性。

每个时间、每域全部 8 条记录的固定汇总如下：

| t | teacher 范数比 | teacher 余弦 | rollout 范数比 | rollout 余弦 |
|---:|---:|---:|---:|---:|
| 1.00000 | 0.58915 | 0.13522 | 0.58915 | 0.13522 |
| 0.90021 | 1.15553 | 0.26165 | 1.15839 | 0.23359 |
| 0.79758 | 1.33533 | 0.19693 | 1.32983 | 0.18918 |
| 0.70498 | 1.45190 | 0.17082 | 1.43933 | 0.17602 |
| 0.60377 | 1.61289 | 0.15829 | 1.60543 | 0.18238 |
| 0.49718 | 1.88492 | 0.15461 | 1.88197 | 0.20361 |
| 0.41026 | 2.23658 | 0.14969 | 2.23981 | 0.22627 |
| 0.29630 | 3.03069 | 0.14190 | 3.01369 | 0.24793 |
| 0.19835 | 4.01598 | 0.15649 | 3.81938 | 0.26105 |
| 0.07477 | 4.24593 | 0.18256 | 3.73122 | 0.30010 |

结论是结构响应并非原 gap 的简单缩放。后段相对原 gap 更大的响应还提醒：不能照搬一个常数外推就宣称消除了原误差，也不能看到表后选窗口或按时间压系数。它可能删除必要的信息，也可能暴露另一种有用偏置；本次未读取 clean X 或测量误差相关性，不能区分这两种解释。

## 成本与证据

v2 的 imports 后至 summary 前 wall 为 **31.53674 s**，CPU process time 为 **57.57553 s**；包括哈希、加载、前向、统计和 I/O。模型加载为 5.35617 s。20 个共享 encoder 合计 1.06499 s，原 Full 准备、decoder/读出和 Base 读出合计 0.19685 s；20 个额外弱两层 decoder/读出为 **0.18160 s**。该弱阶段与显式原分支总量的比值约 14.39%，但分段同步、hook、首批初始化和小规模输入影响计时，不能直接作为部署吞吐或更多采样步数的预算。

额外原生 parity forward 为 1.03771 s；CPU attention/方向统计为 9.00644 s；parity/trace 传输为 1.31174 s，写输出为 1.40066 s。这些是研究成本，不能混入或隐藏成实际 guidance 必须执行的工作。两次 CPU prepare 的 wall 分别为 9.83045 / 9.81634 s，均无模型调用；v1 校验失败的单次外部 wall 未计量，不补造总数。CPU 测试、独立复核和阅读还需分别计入研究投入。

实测调用为 20 个显式共享 encoder、20 个原生参考 forward；encoder block 共 1120 次，显式 Full / Weak / 原生参考 decoder block 各 40 次，Full 读出含弱支与参考共 60 次，Base 读出共 40 次。所有 hook 在运行结束移除。未使用 stage1 图像 decoder。

完整结果在 [decoder_query_mean_v2](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/decoder_query_mean_v2)。request SHA 为 `d7ac7fb4a2bfdcda86244e474ac26e46d406d0073c3dd42e7b0229c606af2c17`，summary SHA 为 `530d6eb9cb843da9dc6191430b3bc9a97034438e3d1fc7f14629280ae1f9bd07`；最终 runner SHA 为 `2f6313482f0752a5b8c5fb1e6bd0f24d852156f3007ab4e7c071b92472cc8296`，协议 SHA 为 `c5570ebdcaed48f104b13302bdadb73569056efca6d7565dee2a71894fe9fab0`。v1 原请求、源码和校验失败另行保留。

独立 [review.json](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/decoder_query_mean_review_v1/review.json)已完成：全部 160 行的 11 项方向量从两份向量文件重建，最大绝对差 `2.91e−11`；20 批 parity、10240 条 attention 身份、实际 hook 调用次数、成本和 t=1 重复向量全部一致。复核 wall 为 1.40699 s，未重新调用模型。review SHA 为 `82c622d66ac0d09339553f2e9ad4ca6bfbedd260cf9b7e1cbf1fcaf0250deaf5`。

全部 Δ/d 向量已保存，可独立重算其方向量；单独 F/B/W 输出和 Q/K/SDPA 原张量没有落盘，因而独立复核不能从向量差重建单头范数，也不能仅凭 attention CSV 重算 KL。对这些项已区分独立算术重建与身份、源码和一致性检查。
