# 轻微阴性1K补至5K：便携证据包

本包保存用户授权的四个固定guidance候选从原1K补到5K的完整小型证据。原始根目录是 `/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/scale_extension_5k_v1`。复制保持JSON/CSV的原字节；采样、训练和评价均没有因归档重新运行。**最终复核、初审恢复证据和merge成本注释均已到齐；本包按root最后通知定版，文件身份见 [portable_manifest.json](portable_manifest.json)。Git提交由root完成。**

完整结论见[5K补测结题报告](../../RAEV2_MILD_NEGATIVE_5K_CLOSEOUT_20260907_ZH.md)。本包只保存证据与读取边界，不复制主报告造成循环身份。

## 样本与基线不能混用

| family | 固定候选 | 同family基线 | 原1K seed |
|---|---|---|---|
| legacy | global proximal、旧全局energy ball | legacy official100 | 202609066 |
| native_global | native全局energy ball | native official100 | 202609121 |
| reflection | affine reflection100 | native official100与成本对照official201 | 202609131 |

三个family分别按自己的冻结计划与元数据配对，不能因为新增seed相同就混用legacy/native样本。legacy采用FP32 guidance/Euler、TF32=true和逐B8噪声生成；native采用原生BF16内部IG、FP32状态、TF32=false和完整1K噪声生成。原1K采样身份见 `historical/` 和各family merge plan，不依本表单独认定配对。

新增四个cohort的seed为202609171、202609172、202609173、202609174；每个cohort固定N=1000、B8、1000类各一图。能量球始终在单个N=1000 cohort内计算，没有改成N=5000相互作用算法。

- **old1k**：原先已看过并用于选入补测的筛查样本。
- **new4k**：仅新增四个固定cohort，每类4图，排除原筛查样本。
- **pooled5k**：原1K和新增4K合并，每类5图；与new4k重叠，二者不是独立复现。

## 如何复查

1. [冻结协议副本](protocol/RAEV2_MILD_NEGATIVE_5K_EXTENSION_PROTOCOL_20260907_ZH.md)和[总plan](extension/plan.json)固定候选、成本规则、数据、代码及方法边界。`extension/plans/`、两个extension manifest保留各native cohort身份；`extension/launch.json`及四个`cohort_*/execution.json`记录实际启动和退出。
2. `extension/cohort_*/`保留全部候选、official100/201、两种parity的request、summary、batch manifest、sampling input/noise manifest、step metrics及诊断。不筛选正面结果；较差候选与成本对照同样保留。
3. [原成本审计](extension/cost_review_v1/summary.json)与[解盲前成本决定](extension/cost_release.json)保留T/W口径。reflection使用同一official201覆盖五个cohort；未使用条件补跑helper。legacy没有分离的轨迹T，不能据相同NFE宣称所有额外成本都已闭合。
4. `extension/evaluation_preparation_v1/`的三个`*_merge_plan.json`分别固定legacy、native_global和reflection的配对关系；各`*_merged/summary.json`及各臂`new4k/pooled5k/summary.json`保留合并核验。各`official_evaluation.csv/json`是统一官方评价结果；`feature_subsets/`保留从官方特征按原行切出的old1k/new4k计算、记录和输入身份。
5. 最终结果在 `extension/final_results_v1/`；最终独立核验是 [采样/成本复核](extension/final_sampling_cost_independent_review.json)与 [FID复核](extension/evaluation_preparation_v1/final_fid_independent_review.json)。[merge执行记录](extension/merge_execution.json)保留外层wall的测量范围注释，避免把有序回收子进程的观察延迟误称为精确进程成本。

FID复核的398项必要检查通过，new4K/pooled5K独立公式重算与保存评价的最大绝对差为 `3.3777425301195763e−12`。`all_checks_passed=false`仍被原样保留：8项旧1K特征的紧阈值诊断不相等，旧1K FID最大数值差 `0.00018038692066113526`。这不是独立重新提取Inception，也没有以新公式替换历史FID，不能说全部比较完全一致。

同目录的 [初审JSON](extension/evaluation_preparation_v1/final_fid_independent_review_v1.json)和 [初审审核器源码](extension/evaluation_preparation_v1/final_fid_independent_review_v1_source.py)完整保留。初审曾把new4K的局部ID误期望为1000…4999；按合并规范，它应重新编号为0…3999，原cohort/source位置另有记录。修正只发生在审核预期，生成图像和评价指标未改变。最终复核分别记录必要检查与旧特征诊断，初审失败不会被最终通过记录覆盖。

## 原1K与大文件在哪里

`historical/`实际复制本次计划引用的九个原始物理臂的metadata，包括reflection历史cost-only200；`historical_context/`复制四个历史实验根目录的JSON/CSV、旧FID和两种native原parity元数据。历史context里可能含原实验其他候选的结果，它们仅提供来源，不属于此次新增5K候选。旧理论/实现导航仍见[研究总索引](../../RAEV2_RESEARCH_ARCHIVE_INDEX_20260906_ZH.md)。

`portable_manifest.json`为每份复制文件列出原始/解析路径、SHA-256、字节数、包内相对路径及复制核验。约3MB的legacy global request包含完整校准元数据，保留原字节而未裁剪。`recorded_large_asset_identities.json`只汇集元数据已经记录的大文件身份及首个来源/引用计数，**归档时不重新读取这些大二进制，也不把原始tensor哈希改称为文件SHA**。

模型、校准pt、完整图像npz、逐batch数组及Inception特征npz保持原址，没有复制进本包。原始执行日志仅记录路径和stat信息于 `raw_log_paths.json`，未复制/重新hash日志内容。源码保留仓库原文件和request里的冻结身份；各输出`sources/`下的重复源码不再次复制到本包。

本目录的[导出脚本](export_portable.py)限定JSON/CSV、冻结协议和明确历史目录，逐份读取来源与副本校验字节。`snapshot`不包含待定的最终复核/成本注释；只有root发出最终就绪通知后才执行`finalize --root-finalization-approved`，最终包拒绝覆盖。
