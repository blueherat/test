# RAEv2 完整终点伴随响应：8 图结果与独立 CPU 复核

日期：2026-09-06。**执行完整成功，机制检验阴性。** 固定控制下，FP32 平均终点响应仅为线性预测的 **23.5503%**；原生 BF16/uint8 响应为 **−15.1659%**，方向相反。停止这一实现的部署及蒸馏准入，不用 λ 扫描、裁剪、时间窗口、样本重加权或训练挽救。8 图不支持 FID 或全分布质量结论，也未完成方法总成本的公平比较。

本记录依据运行前冻结的[机制检验协议](RAEV2_ENDPOINT_ADJOINT_RESPONSE_PROTOCOL_20260906_ZH.md)、原始输出，以及本次独立 CPU 核验。此次只重算缓存与哈希，没有调用 GPU、模型、FID 或重跑此前通过的 10 项测试。

固定设置为 seed `202609111`，类别/全局 ID `[0,142,285,428,570,713,856,999]`；所有 worker 各做一次同形状 `[8,1024,16,16]` CUDA FP32 噪声抽取，按固定位置取 B1，四卡各两图。100 步 shift 8、`t_eps=.05`、原生区间 `[.1,1]` 使用 `B+1.78(F−B)`；Full 与 Base 来自同一次主干调用。FP32 连续替代主干/decoder/Inception 禁用 TF32；原生使用 BF16 主干及 decoder、FP32 状态更新、BF16 `clamp*255→uint8` 导出。参数仅保留一份 FP32 常驻拷贝，通过 autocast 切换计算精度。 本 pilot 采用相同原生算术，但 batch 为 B1；正式质量评价使用 B8，不能称为逐位复现 B8 生产采样。

终点量为 `ψ(f,c)=〈m_c−m,f−m〉/2048`。原型固定来自真实图 seed `20260801`；缓存 `fixed_prototypes.npz` 的 directions 已经除以 2048，复核没有再除一次。扰动 `δ=0.0018097258402418833` 来自历史 A→B source−IG 对齐差。历史 bank 的 IG 混合及像素量化与当前原生协议存在差异；这里只把 δ 用作冻结扰动尺度，不能称为已知当前 8 图或原生总体存在同样缺口。

收集得到 `G=mean_i Σ_k h_k ||b_ik||²=2.606953444155491`，唯一冻结 `λ=δ/G=0.0006941918522937546`。每步先做 Euler，再加 `h_k u_ik`，其中 `u_ik=λ b_ik`。所有 8 图 collect 完成后才 CPU finalize；此后只执行一次完整 replay。

平均结果如下；Δ 为 controlled−同精度 baseline，最后一列以固定 δ 为分母。ψ 从 32 份保存的 Inception 特征重新计算，和原始逐图摘要最大绝对误差为 `1.3877787807814457e−17`。

| 精度 | 基准 ψ | 控制后 ψ | 实测平均 Δψ | Δψ/δ |
|---|---:|---:|---:|---:|
| FP32 连续替代 | 0.040865163417 | 0.041291359964 | 0.000426196546 | 23.550338% |
| 原生 BF16/uint8 | 0.039345636688 | 0.039071175935 | -0.000274460753 | -15.165875% |

线性平均预测为 `δ=0.0018097258402418833`。FP32 平均预测误差为 `−0.0013835292937882816`。按实际使用的 FP32 `u=λb` 累积的平均控制能量 `mean_i Σh||u||²=1.2562968887907915e−6`，与理想 `δ²/G=1.2562969331813844e−6` 接近；相应伴随线性预测为 `0.001809725808269025`。**这些量在将 `h*u` 加入状态之前计算，不等同于浮点状态加法后的有效位移或实测终点响应。** λ 由 G 定义使理想预测等于 δ，这个恒等式自身不是机制成功证据。

全部 8 图逐图结果如下，没有选择、丢弃或替换输出。完整精度、逐图 Δ/δ、控制能量见下方 CSV。

| ID | FP32 基准 ψ | FP32 控制 ψ | FP32 Δψ | 原生基准 ψ | 原生控制 ψ | 原生 Δψ |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.02800306557 | 0.0298377077 | 0.00183464213 | 0.02704322798 | 0.02067689699 | -0.006366330981 |
| 142 | 0.04103742784 | 0.04103811901 | 6.911690774e-07 | 0.0408113131 | 0.04194083829 | 0.001129525187 |
| 285 | 0.03605407291 | 0.03605406435 | -8.561335453e-09 | 0.03633651245 | 0.03633087077 | -5.641681257e-06 |
| 428 | 0.06516756583 | 0.06517524429 | 7.678465727e-06 | 0.06843459562 | 0.06824563193 | -0.0001889636839 |
| 570 | 0.04397754915 | 0.04554356904 | 0.001566019885 | 0.04272963476 | 0.04797298478 | 0.005243350017 |
| 713 | 0.02107899598 | 0.021078937 | -5.898388743e-08 | 0.02126987817 | 0.02150827192 | 0.0002383937459 |
| 856 | 0.03593046648 | 0.03593047215 | 5.664876596e-09 | 0.03537481403 | 0.03548559307 | 0.0001107790364 |
| 999 | 0.05567216357 | 0.05567276617 | 6.026023991e-07 | 0.04276511739 | 0.04040831973 | -0.002356797664 |

完整离散 suffix 的索引经归档源码静态核对：终点 `ψ.sum()` 对 `z[100]` 求梯度；倒序 `k=99,…,0` 先保存当前 `b_k`，它位于 **successor `z[k+1]`**，随后仅在 `k>0` 对 `T_k` 做 VJP。实际链为 `T_99,…,T_1`，共 99 次，不多乘 `T_0`，不截断后续路径。Gram 在每图内对所有坐标求和，再对 8 图求均值；终点梯度没有额外 batch 平均因子。源码与计数一致；此次没有重新执行 VJP 或用新步长做有限差分检查。

局部 Gram 分配明显集中。以下是每图 `G_i=Σh||b_ik||²`，占比的分母为全部 8 图的 Gram 总和；它没有把各图重新加权。

| ID | G_i | 占总 Gram | 按 FP32 u 累积的控制能量 |
|---:|---:|---:|---:|
| 0 | 18.27128164 | 87.608400% | 8.804972842e-06 |
| 142 | 0.01193118442 | 0.057208% | 5.749665342e-09 |
| 285 | 7.131362483e-05 | 0.000342% | 3.436620058e-11 |
| 428 | 0.06203194863 | 0.297435% | 2.989333938e-08 |
| 570 | 2.494568476 | 11.961129% | 1.202138313e-06 |
| 713 | 0.001915742752 | 0.009186% | 9.232008593e-10 |
| 856 | 0.0004613291606 | 0.002212% | 2.223155885e-10 |
| 999 | 0.01336592092 | 0.064088% | 6.441068177e-09 |

ID 0 占 **87.608400%**，ID 570 占 **11.961129%**，合计约 99.57%。按实际采样顺序统计，前 10 步 `k=0…9`、即从 `t=1` 到约 `t=.98630136`，占 **47.841654%**。所有 100 个时刻均保存；下表按固定十步分箱完整列出，占比是 `Σ_i h_k||b_ik||² / Σ_i G_i`，不是等时间长度密度。

| 步索引 k（含端点） | 占总 Gram |
|---|---:|
| 0–9 | 47.841654% |
| 10–19 | 37.659102% |
| 20–29 | 13.532920% |
| 30–39 | 0.912115% |
| 40–49 | 0.022560% |
| 50–59 | 0.011120% |
| 60–69 | 0.006495% |
| 70–79 | 0.003746% |
| 80–89 | 0.003104% |
| 90–99 | 0.007184% |

这只是冻结队列、基准轨迹附近的局部响应集中。不能据此声称某类样本或某段时间普遍更重要，也不据此另设窗口、时间增益或样本权重。每图每步的 `||b_k||`、`||b_k||²`、积分贡献与控制能量全部归档。

数值失配至少需要区分两个层次：同一 FP32 连续替代的有限重放没有兑现一阶预测；迁移到原生计算后平均方向又发生反转。一阶最小能量结论约束的是局部线性化，未保证这个有限扰动、浮点乘加以及原生精度下的响应。当前数据没有隔离非线性、FP32 舍入、BF16 轨迹差异、decoder 差异和 uint8 量化的各自作用，因此不把失败单独归因于任何一项，也不把它写成反驳所有终点控制理论。

首图保存的原生 uint8 特征与连续包装器输入 `uint8/255` 的坐标检查通过，最大绝对差 `2.7418136596679688e−6`，`rtol=atol=2e−5`。独立复核重算了这对已保存特征的差；没有重新调用特征提取器。这个检查只支持 Inception 坐标一致，不证明 BF16 或 uint8 可微，也不证明两种精度的轨迹接近。原生均值为负已足以拒绝当前实现的部署/蒸馏准入；没有追加 λ 搜索或训练。

调用成本在逐图 execution、worker summary 与总摘要三层相符：

| 工作 | 主干 forward | 主干输入 VJP | decoder forward | Inception forward | decoder / Inception 输入 VJP |
|---|---:|---:|---:|---:|---:|
| 8 图 collect 与首图坐标核对 | 1592 | 792 | 9 | 10 | 8 / 8 |
| 三组 8 图 replay | 2400 | 0 | 24 | 24 | 0 / 0 |
| 合计 | **3992** | **792** | **33** | **34** | **8 / 8** |

每图 collect 为 100 次 rollout forward 加 99 次 VJP 重算 forward；三组 replay 为 300 次 forward，共 499 次/图。B1 下调用数与 sample-forward 数相同。Base head 与 Full 共享主干 forward，不能重复计算成两次独立主干调用。RAE 构造时载入但不用的 DINOv3 encoder 已删除，其载入时间及权重身份保留，encoder forward 为零。FID 调用、训练更新均为零。

四卡并发的时间边界如下；各 worker 墙钟和是进程占用时间总和，不能当作端到端延迟。下表 collect/replay 的“阶段”包括 B1 处理与记录，不含其前面的权重校验和模型加载。

| 阶段 | worker 外层墙钟和 | 其中源码/权重校验和 | 其中模型载入和 | 内层阶段墙钟和 |
|---|---:|---:|---:|---:|
| collect，各 4 worker | 282.602415 s | 54.362897 s | 43.927585 s | 153.736002 s |
| replay，各 4 worker | 199.334195 s | 54.369254 s | 43.170496 s | 76.028343 s |

外层 driver 墙钟 **132.266372 s**，含 collect、CPU finalize、replay、CPU summarize 和调度；不含此前 prepare、driver 自身在 `main()` 前的 imports，以及最后摘要写入和退出。每个子进程的外层时间从启动前至 `wait4` 返回，包含该子进程 imports 和退出；finalize **4.423860 s**，summarize **3.610309 s**。这些阶段已包含在 driver 中，不能再相加一次。

prepare 的内部进程快照为墙钟 **25.227655 s**、CPU **19.196968 s**：准确起点在 `import time` 之后、其余 imports（含 NumPy/Torch）之前；终点在写 request 之前，缺最后 JSON 写入、解释器退出和父 shell 启动，故不是完整外层耗时。worker 内部 process_cost 使用同一起点和类似末尾边界。完整逐 worker 墙钟/CPU、三项模型载入时间及显存见 timing_evidence。CUDA event 数值是含 stream 空闲间隙的跨度，不能冒充 kernel 忙时或直接转成模型 FLOPs。

最大单 worker GPU allocated 为 **6267868160 bytes**（约 5.837 GiB），reserved 为 **6385827840 bytes**（约 5.947 GiB），包括模型载入。所有 100 份伴随每图数据体积为 100 MiB；101 份基准状态每图为 101 MiB，另有 NPY 头。保存这类完整开环控制所需的收集与反向成本已经发生，不能称为免费在线 guidance，也没有据此完成与普通采样的公平总成本比较。

独立 CPU 复核一次通过 **572 项记录/数值检查**，实际读取并核验 **235 个唯一文件、15,331,833,658 bytes** 的 SHA256，覆盖原始 request、prototype/ref、当前和归档源码、配置及模型权重、worker/逐图记录、噪声与全部输出。32 个终点（16 个 FP32 连续像素、16 个原生 uint8 像素）均保留；8 份完整噪声及 RNG 状态逐位一致。重算 808 个 collect 状态哈希，以及四组终点的初始/最终状态哈希。replay 中间状态只保存了各自 101 条哈希，未在本次 CPU 复核中重建其张量或动力学。

全局噪声原始字节 SHA 为 `9112d1dfb5acd8b7df73d56e9f4af47584b6b30a7958fb93f8127c15ba8b7500`。request SHA 为 `eec577a37763439471e2f4530d1e7810dfc3825e30cdf4ad54f7ee4f98d7b4bd`；frozen control SHA 为 `11aa47b831fa666c93be54849daa0437f7d0e3a280fed016e4f5c181b8827e56`。所有其余完整 SHA 与路径见归档清单。

本次独立 CPU 数值/哈希核验实测墙钟 **16.270826 s**、CPU **16.262202 s**、峰值 RSS **296849408 bytes**。计时包括其 NumPy import、实际文件读取、重算与前置证据/CSV 写入；不含最后 results/checks/manifest 序列化、退出、本报告编写和此前人工只读检查，不能当作本次人工研究的总成本。此次 GPU/model/test 调用均为零。

可复核交付物：

- [独立 CPU 脚本](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/endpoint_adjoint_response_v1/review/audit_saved_response.py)、[结果与边界](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/endpoint_adjoint_response_v1/review/results.json)、[逐项核验](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/endpoint_adjoint_response_v1/review/checks.json)。
- [全部逐图 ψ、Δ/δ、Gram、能量](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/endpoint_adjoint_response_v1/review/per_image.csv)、[全部 8×100 时刻伴随范数与积分贡献](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/endpoint_adjoint_response_v1/review/per_image_per_step.csv)。
- [时间原始证据与聚合](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/endpoint_adjoint_response_v1/review/timing_evidence.json)、[带行号的冻结源码证据](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/endpoint_adjoint_response_v1/review/source_evidence.json)。
- [输入/资产全部 SHA](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/endpoint_adjoint_response_v1/review/sha256_evidence.json)、[本次复核产物 manifest](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/endpoint_adjoint_response_v1/review/manifest.json)。
- [原运行 request](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/endpoint_adjoint_response_v1/request.json)、[外层执行记录](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/endpoint_adjoint_response_v1/execution_summary.json)、[原 response summary](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/endpoint_adjoint_response_v1/response_summary.json)、[唯一冻结控制](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/endpoint_adjoint_response_v1/frozen_control.json)。

补充：事后静态舍入诊断（不改变冻结结论）。root 另在 CPU 上对保存的 FP32 基准 successor `s=z[k+1]` 计算 `d=float32(h_k*float32(λb_k))`、`float32(s+d)`。每图对 100 步全部坐标等权平均，FP32 加法后数值不变的比例为 **87.1978%–99.8648%**；直接比较这两个状态的原始 BF16 cast，不变率为 **99.99847%–99.99995%**。ID 0 的加法舍入误差平方和 / 期望增量平方和约 **0.1843**，ID 570 约 **0.5512**，其余约 **0.86–0.98**。这里的分母是 `Σ||d||²`，与前述 `Σh||u||²` 控制能量定义不同。

这支持“某些控制位于 FP32 状态加法的舍入尺度附近”这一有限判断；它只在**保存的基准状态**做单步算术，不是完整受控轨迹的不变率。直接 BF16 cast 也不是主干的完整输入计算或 Jacobian；坐标不变比例不等于终点影响比例，不能单独解释原生符号反转。没有据此修改 λ、增加量化方案、重放、训练或计算 FID。诊断实测墙钟 **5.622398 s**、CPU **10.979609 s**，包含 `import time` 后的其余 imports 和缓存读取，缺最后结果写入/退出，零 GPU/模型调用。见[诊断脚本](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/endpoint_adjoint_response_v1/rounding_diagnostic.py)、[逐时刻与逐图结果](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/endpoint_adjoint_response_v1/rounding_diagnostic.json)；这两份新增文件的 SHA 单独保存在 [rounding_evidence](/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/endpoint_adjoint_response_v1/review/rounding_evidence.json)。本次报告只读核对其源码、冻结 request/control/source 身份及已保存摘要，没有重跑此诊断。
