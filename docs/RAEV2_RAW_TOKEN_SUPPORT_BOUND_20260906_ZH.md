# RAEv2 raw token 确定性支撑界：固定缓存检查

**全部 184,320 个被检验的 token 范数均小于 64；最大为 46.65022205。这个粗支撑球不会改变这些已见查询，不据此新增性能实验。** 本次没有模型、GPU、采样、裁剪、解码或 FID 调用，也没有修改正在运行的反射实验。

K7 raw token 是七层无仿射 LayerNorm 特征的均值，加上最后所选层的空间均值。理想算术下，单个归一化特征范数不超过 √1024=32，故三角不等式给出逐 token 的确定性界 `||a_j||≤64`，同时通道和为零。它不是固定球壳，也不是经验半径。仿射变换与条件期望可交换，所以真实 clean posterior mean 也属于这个凸支撑集合；有限模型预测是否违反它需要测量。该界不能直接施加给含 Gaussian 噪声的状态。

先冻结 [request.json](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/raw_token_support_bound_v1/request.json)，SHA `0e881a32898d0cc94601397313f276d89fc74a1a68de0bbad60780f854b98faf`，再进行张量运算。固定输入为：

- 历史 `normal_noise_audit_seed202609071/states/step_{000,047,067,077,084,089,092,095,097,099}.pt` 全十个快照的 teacher/rollout F/B，全部 8 个 ID，共 160 行。原 heads 为 BF16 后直接升格存 FP32；旧前向 TF32 开启，rollout 使用旧 FP32 guidance 算术。
- 当前 `query_mean_error_compatibility_v1/{F,B}.npy` 全部 80 行 teacher，原生 BF16 前向、TF32 关闭、存储 FP32。行顺序与原 `per_image_risk.csv`、request 的 ID、类别、source row、时间逐项核验。
- 唯一官方 `stats.pt`，SHA `40e57d9d38a267dc258043c081094d276382c29ebccc1bd8c38f4dd11e81ba77`。没有新真实数据半径估计。

在任何范数计算之前，全部 480 个 F/B 头行、**125,829,120 个标量**均通过 FP32→BF16→FP32 精确往返检查。之后才用 CPU BF16 逐操作重建窗口 `[.1,1]` 内的 `B+1.78*(F−B)`，再转 FP32；窗口外为 F。这是来自保存 heads 的原生算术重建，不能称为本轮新测得的 CUDA guided 输出。

固定采用 FP64 `sigma=sqrt(FP64(var)+1e-5)`、`a=sigma*FP64(prediction)+FP64(mean)`，逐 token 对全部 1024 个通道求平方和，与 **4096=4×1024** 比较。严格大于该界才记为越界，没有数值 margin、拟合阈值、半径搜索或样本筛选。

| 缓存与域 | 头 | token 范数均值 | 最大范数 | 越界数 / token 数 |
|---|---|---:|---:|---:|
| 历史 teacher | F | 42.602658 | 46.246117 | 0 / 20,480 |
| 历史 teacher | B | 42.399221 | 46.161213 | 0 / 20,480 |
| 历史 teacher | 重建 IG | 42.951647 | 46.650222 | 0 / 20,480 |
| 历史 rollout | F | 42.365798 | 46.293517 | 0 / 20,480 |
| 历史 rollout | B | 42.153697 | 45.713970 | 0 / 20,480 |
| 历史 rollout | 重建 IG | 42.717388 | 46.293517 | 0 / 20,480 |
| 当前 teacher | F | 42.602658 | 46.246117 | 0 / 20,480 |
| 当前 teacher | B | 42.399221 | 46.161213 | 0 / 20,480 |
| 当前 teacher | 重建 IG | 42.951647 | 46.650222 | 0 / 20,480 |

全局最大 norm² 为 **2176.24321738**，小于 4096。全部逐行及 pooled 越界比例、正超出量的均值／最大值／总和均为零。当前与历史 teacher 在表中摘要一致；这些行重复使用同八张图及十个时间，t=1 的两个历史域也重复，不能当作独立确认或 240 个独立样本。没有推断未保存的当前 rollout，更没有给出 FID 上界。

完整 [token_norm_squared.npy](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/raw_token_support_bound_v1/token_norm_squared.npy) 为 FP64 `[240,3,16,16]`；[per_state_head.csv](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/raw_token_support_bound_v1/per_state_head.csv) 保留全部 720 个状态×头行、身份及范数／超出量统计。另存所有逐时间 pooled 量、240 行身份及逐头 BF16 精确性结果。程序以逐状态统计重新汇总计数／最大值，核对 pooled 结果；这是内部一致性检查，不冒称独立原始数据重算。

[summary.json](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/raw_token_support_bound_v1/summary.json) SHA 为 `026a3c298015e1769980afb8afa5ec72c8816e726c4736b4855685b171137030`。23 个输入／源码文件实际 SHA 核验，共 673,754,274 字节；运行前再次核验。准备 wall **0.888772 秒**，CPU 计算阶段 wall **4.025990 秒**、进程 CPU **8.512037 秒**。后者含身份复查、NumPy/Torch 导入、计算及主要产物写盘，止于最终 summary 写入前，不含解释器启动与退出。未加载或重新哈希十余 GB 的模型权重，权重身份沿用被核验的源 request。运行脚本为 `raw_token_support_bound_v1/audit_cpu.py`，SHA `e205fbd9bd2a9c2035405315afbb9c645345ff9d91162747773690cc0e2407cc`。

这里的确定性支撑与旧 energy-ball 的经验总体二阶矩预算不同。在归一化坐标中，它是 `||sigma*x+mu||≤64` 的椭球约束；raw 径向裁剪再标准化，不能直接声称是归一化欧氏投影或具有该度量的风险收缩。本次只检查界是否活跃，未执行任何这种映射。

反射实验的 BF16 边界也应保持明确：其实际有限精度反射未严格构成二元素群，微小的反射往返输入误差仍可能伴随非零 guided 不变性残差。理想群平均的风险恒等式因此不能直接宣称为当前实现的严格保证；不变性残差本身也不是理想算子误差或 FID 上界。本次没有重跑反射、改公式或据此选择新参数。
