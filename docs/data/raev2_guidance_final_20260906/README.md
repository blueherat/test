# RAEv2 guidance 最终轻量数据包

本包保存预先列明的已完成实验的小型原始记录，不重新选择方法、seed 或 FID 最好的 arm。当前包含 188 个逐字节复制文件，共 3,366,966 字节。已显式加入已完成的固定第一轮 1K screen。

## 从哪里看

- [研究导航](../../RAEV2_RESEARCH_ARCHIVE_INDEX_20260906_ZH.md) 串起理论、实现、结果与大资产原址。
- [manifest.json](manifest.json) 逐文件列出源逻辑路径、解析后真实路径、字节数、SHA-256、包内相对路径。每份小文件复制后都重新检查包内 SHA。
- [recorded_asset_identities.json](recorded_asset_identities.json) 从已复制 JSON 提取已有路径/摘要记录。它们是**记录身份，未本轮重读**；某些摘要可能针对 raw tensor bytes，必须回到原始记录判断，不能当作本轮完整文件核验。
- `files/` 按原 RAEv2 实验根目录保留结构，可直接比较 request、summary、FID、成本与逐时风险。

## 已纳入的固定实验

1. affine reflection：official100、reflection100、cost-only official200、最终 official201、parity、成本选择、统一 FID、独立审核。
2. spatial energy balls：official/global/spatial100、parity、成本冻结、统一 FID 与逐样本审核摘要。
3. observable potential 5K：全部三个 arm 和四 shard、合并/恢复元数据、统一评价与独立分析。
4. endpoint adjoint：执行、冻结控制、响应、量化诊断和独立 review；没有 FID。
5. query-mean error compatibility：逐图/逐时风险、固定输入身份、独立重建；没有 FID。
6. paired bridge：pilot/train/validate/rollout、独立归一化与实际速度单位汇总、原父进程状态和后续恢复链；不把负回归结果或矩改善称为质量收益。

## 明确保留的缺口

- 不复制模型、完整样本、latent/feature bank、PDF、图像、batch 文件、大日志或大于 1 MB 的单文件。paired-bridge `teacher_regression.csv`（约 3.74 MB）与详细 draw ledger 留在原址；小型逐时/逐类汇总已纳入。
- 本轮只 hash 被复制的小文件与归档脚本，不扫描或 hash GB 级资产，不运行 GPU/模型/FID。大资产清单是已有元数据索引，不能冒充完整备份或新的数据真实性审计。
- paired-bridge 原训练父进程消失造成外层精确 wall time 缺口；原 `pipeline_status.json` 保留其历史不完整状态，后继 `pipeline_resume_status.json` 只证明后续验证/rollout 恢复。训练没有重复。
- observable-potential 的 `failure.json` 是此前中断记录；完成状态看后继 `execution_summary.json`，不删除失败历史。历史准备/数据选取成本未全部闭合，当前推理匹配不等于总成本达标。
- 本包不重导入已有 9 月 5 日 Git 数据包，也未覆盖全部早期 RAE/SiT/AdvFD/DiT 大资产。当前任务的 ≥5% 相对 FID 与独立确认目标仍未因此达成。

## 后续显式增量

```bash
python experiments/archive_raev2_guidance_final_data.py --include-completed-screen
```

该开关只允许同一固定 `paired_bridge_v1/screen_v1`，要求 `screen_execution.json` 已 complete，并保留全部 cost-only 中间 arm。没有新 seed 或扫描选择。已有复制文件必须字节相同，否则拒绝覆盖。聚合 manifest 的前版写入 `manifest_history/` 后才追加，新版记录父摘要；这不改冻结源文件。默认再次执行只验证已有包，不移除已加入的 screen。
