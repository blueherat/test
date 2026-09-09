# 分区控制：成熟度、控制类与弯曲支撑

状态：机制候选继续保留，仍无真实图像质量突破或论文完成结论。
接续 [首轮报告](SUBSPACE_GATE_PILOT_RESULTS_20260908_ZH.md)。

## 30K 续训没有消除优势

三个 H512 模型恢复原 optimizer，各再训练固定 15000 updates，总计30000。
使用记录在 config 中的新 minibatch RNG，不声称 bitwise uninterrupted resume。
之后同协议训练5000步 gate，并用新8192-sample bank复核。下表为 ambient SWD。

| 方法 | seed20260831 | seed20260901 | seed20260902 |
|---|---:|---:|---:|
| 原 safe gate | .065340 | .076024 | .044812 |
| 新 scalar | .071172 | .075363 | .047693 |
| PCA 双 gate | **.040055** | **.048831** | **.028008** |
| 纯 x | .116504 | .131752 | .083722 |
| projected x | .092524 | .110787 | .063936 |
| projected scalar | .071586 | .075165 | .048072 |

PCA 相对 scalar 降幅43.7%、35.2%、41.3%，同时法向残留下降。
这排除了“只在原15K停止点出现”的解释，但训练loss仍在下降，不能宣称全局收敛。
数据：`maturity_pilot.csv`、`maturity_confirmation.csv`；原始模型和运行历史在
`~/data/eqvae/experiments/subspace_gate_maturity_20260908/`。

## 控制类比之前认为的更重要

在窄模型同一 oracle bank，逐坐标 oracle 的 ambient SWD 三seed为
.049755/.031018/.043113，平均优于原PCA子空间 oracle的.047409/.045625/.053755。
以两头为直径端点的球内 Bayes 投影更达到 .012734/.025184/.020495。
因此不能声称数据PCA分区是获得oracle收益的必要条件；一般可行控制类的扩大
就可能带来更大改善。球oracle直接使用Bayes速度，不是已得到的实用算法。
相应几何关系见 [代数边界](SUBSPACE_GATE_ALGEBRA_ZH.md)。

随后训练逐坐标gate：hidden76、512输出，共86696参数，对照原PCA gate86530
参数；时间安全参数化、训练数据流、5000updates与评估bank保持一致。
三seed coordinate ambient SWD为.099496/.111431/.074330；PCA为
.068182/.085001/.069826。逐坐标gate比scalar略好，但仍三seed落后PCA。
这支持当前参数预算下PCA分区更容易学习，不证明所有坐标gate都逊于PCA。
数据：`control_class_oracle.csv`、`coordinate_learned.csv`。

## 弯曲支撑的单seed pilot

复用仓库v4的随机Fourier嵌入，curvature=.5，D512/H512，seed20260831，
固定15000updates，之后原样训练rank2 PCA/scalar/random gates。
禁止使用只对线性分布正确的Bayes oracle；测试覆盖此禁用行为。

| 方法 | 首次4K ambient SWD | 独立8K ambient SWD |
|---|---:|---:|
| safe | .054067 | .054244 |
| scalar | .056452 | .056930 |
| PCA双gate | **.043470** | **.041914** |
| random | .056443 | 未重复 |
| 纯x | 未测 | .101903 |
| projected x | 未测 | .358218 |
| projected scalar | 未测 | .311341 |

独立8K中PCA比scalar全维SWD降低26.4%，但embedding consistency RMS为
.05350，对照scalar .04841，约恶化10.5%。完整分布指标改善不意味着每个几何
指标改善。这里的旧字段`off_subspace_rms`实际是`embed(decode(x))`一致性残差，
不是最近点流形距离；不得把曲面强行称作二维线性子空间。

首次曲面config的文字data_protocol继承了旧curvature-0描述，但数值curvature=.5、
实际源码和运行均为非线性嵌入；已保留原配置并另写protocol_note.json说明，未
覆盖历史。后续runner已修正自动描述。数据：`curved_pilot.csv`、
`curved_confirmation.csv`。

其余两个曲面seed已完成同协议训练及独立8192-sample复核：

| 方法 | seed20260831 | seed20260901 | seed20260902 |
|---|---:|---:|---:|
| scalar | .056930 | .060858 | .056848 |
| PCA | .041914 | .052137 | .049778 |
| safe | .054244 | .060353 | .056118 |
| projected scalar | .311341 | .272290 | .284920 |

PCA相对scalar降低26.4%、14.3%、12.4%；但三个种子的embedding consistency残差
都比scalar高，不能将整体SWD改善解释为几何一致性单调改善。完整行已写入同名CSV。
目标不收缩为toy报告：真实SiT的state-dependent可学习gate已完成训练，holdout MSE
没有改善，最终图像质量正在验证；详见[SIT协议](SIT_SUBSPACE_GATE_PROTOCOL_20260908_ZH.md)。

## 当前运行与下一步

- 曲面三seed训练、固定5000update gate和独立采样均已完成，不按评估选参数。
- 真实SiT下一步使用冻结450K双头、训练集PCA basis，缓存配对监督后比较
  同参数scalar/PCA的state-dependent gate，再与已有空间gate/两个纯头比较
  最终生成质量。不能把局部MSE改善当作成功，也不能仅以局部MSE变化小否定FID。
- 新颖性、充分训练强基线、真实图像跨模型证据和完整论文稿仍未完成。
