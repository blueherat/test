# 单标量控制的子空间竞争：探索协议

状态：机制线索，非论文突破。由终点误差抵消 pilot 的负结果转向。

## 已有 oracle 对照

冻结连续螺旋 D512/H128 的三个 D0 双头，200-step Heun，每 seed 4096 个新样本。
同一标量最小化全维 Bayes velocity error 时，必须同时处理真实二维子空间和
510 维法向噪声。分别在正交子空间求解两个 clipped scalar gate，只改变允许的
控制类，不改变头或监督信息。三 seed ambient SWD 从约 .101/.085/.089 降到
.047/.046/.054。随机二维子空间得到 .101/.085/.089，几乎无收益。

仅用 intrinsic gate 控制全维，二维 SWD 改善但全维 SWD 恶化到 2.7–2.9；
仅用 normal gate 控制全维，则法向清除良好而二维 SWD 恶化。

这支持真实子空间的控制竞争，不证明 D4 胜过 D3 的全部机制，也不证明新方法。
两个 gate 的 oracle 类严格包含标量类，局部最优误差下降本来就可预期；有价值
的待验证点是这种分离能否在可学习、非 oracle 的场景改善最终分布。

## 下一项冻结的可学习 pilot

- 沿用三个已训练 D0，不更新其参数；新增小 gate 网络。
- 用独立训练样本估计 rank-2 PCA 投影，不使用真实 embedding basis 或 Bayes 速度。
  rank=2 来自本 toy 的已知设置，不能声称具有一般数据上的自动选秩能力。
- 三个配对控制：一个 scalar gate、PCA split 两个 gate、随机子空间 split。
  网络均输出两个 logit；scalar 使用两者平均，使参数和主干计算匹配。
- 统一安全参数化 sigmoid(log((1-t)/t)+logit)，完整 ambient velocity MSE，
  相同 clean/noise/time minibatches、初始化、Adam 学习率 2e-4。
- 固定 5000 updates，batch 1024，hidden128、depth2；不按评估选 checkpoint。
- 新采样 bank，与 oracle pilot 不同；固定 4096 samples、200-step Heun。
- 同时报告 ambient/intrinsic SWD 和 off-subspace RMS；不允许只报告二维优势。
- 优先要求三个训练 seed 的 PCA split 均胜 scalar，并优于随机 split。
  若不成立，记录 oracle-to-learned gap，不通过评估调参掩盖。
- oracle 及 learned 都只是 toy；实际 RAE 上的可实现子空间、额外成本、相关工作
  与至少两个真实模型的独立质量验证仍未解决。

原始数据：`~/data/eqvae/experiments/terminal_defect_20260907/`。

## 宽度反证实验（上一 pilot 完成后冻结）

将 D512 的 hidden128 改为 hidden512，消除最后线性输出的硬 rank<512 限制。
保留三个原 seed、depth4、15000 updates、batch2048、原初始化/训练数据种子和
优化器设置，重训 D0/D2/D4 三模型。此比较匹配步数，不匹配算力，不主张训练
效率优势。随后原样运行 5000-update scalar/PCA/random gate pilot，使用新 bank。
如果相同方法在宽模型不再改善，或原 safe 已经优于新方法，明确记录，不把
窄模型结果直接迁移为普遍贡献。检查训练是否收敛仍是另一问题，宽度足够
不自动代表网络学到了精确 Bayes 速度。

## 训练成熟度对照（宽模型结果后冻结）

三个 H512 checkpoint 均固定再训练 15000 updates，到总计 30000；恢复原
AdamW 状态，不重新初始化，不按 FID 选择停止点。原 checkpoint 未存 RNG，
因此使用明确记录的新独立 minibatch 流，不声称是 bitwise uninterrupted resume。
之后再从相同初始化、相同步数拟合 scalar/PCA/random gates，使用新采样 bank。
目的：检查收益是否仅由尚未充分训练的基础头产生；30K 本身也不自动代表收敛。

## 弯曲支撑的反证 pilot

使用仓库 v4 的 curvature=.5 随机 Fourier 嵌入，H512、depth4、15000 updates，
首先只做 seed20260831。该设置的 clean 支撑不再是一个二维平面；仍测试原样
rank-2 PCA gate，不以真实局部切空间替换它。线性 Bayes oracle 在该设置下
明确禁用。原 `off_subspace_rms` 字段此时表示 embed(decode(x)) 的一致性残差，
不是到非线性流形的最短距离。主比较仍为完整 ambient SWD。单 seed 只筛选
方法能否离开精确线性低秩假设，不用于总体成功主张。
