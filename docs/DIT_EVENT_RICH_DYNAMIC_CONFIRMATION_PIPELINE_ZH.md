# DiT 事件富集动态确认流水线（v3 B/C）

## 当前状态

代码与源码锁已经完成，但没有运行真实轨迹、打开真实标签或计算真实 B/C 结果。

- 科学协议：`dit_event_rich_confirmation_protocol_lock_v3`
- 动态流水线源码锁：`dit_event_rich_dynamic_confirmation_source_lock_v4`
- dynamic contract identity：`ee57e5dfe94d96c2455450bf8c3f747ff330f8017d25c3248bf6f1dc978027e0`
- source manifest identity：`3e1402cadeb4fac46ce7f5390056c4014ad13a0af95c8a5e4b06a1aab0a0a161`
- 对接的盲标源码锁：`dit_event_rich_review_pipeline_source_lock_v7`，identity `b015b4c173db1e09da2608ccb5bd3958be185fa876e304e4f80f316a098050c8`
- v1/v2/v3 动态源码锁保留为历史版本；v3 在 v2 的实际 review handoff、单候选源码绑定、确定性 VAE 和多任务单次模型加载之上，又把 anchor plan 的完整字段、Wilson forecast 与 GO 决策重放设为强制条件；v4 再把最终盲标源码锁 v7 的 identity 和 consensus-runner source hash 固定进 Stage A 来源链。

这套实现严格属于 scientific v3，只包含已经冻结的 B 与 C。未来若 v4 引入新的 exact e-process E，不能把 E 偷塞进本锁；必须重新冻结科学协议、源码合同、E 的独立产品、Stage-A 家族门控、Stage-B 多重检验和干预权限。

## 数据流

1. discovery 与 anchor 的 endpoint-only 盲标完成后，冻结 selector 产生唯一的 `anchor_plan.json`。
2. 轨迹 worker 只采样 `active_union_classes × (1100..1119, 1200..1327)`；若 B/C 都是 STOP，不允许建立 full-trace pool。
3. 每个 `(global_seed,class_id)` 是独立的 SHA-256 派生 RNG 流；每个 worker 可以一次加载模型/VAE 后处理一个任务分片。
4. 每条轨迹只保存最终 endpoint 与 B/C 必需的最小内部量：
   - B 类所需的 9 个 `pred_xstart[4,32,32]`；
   - C 类所需的 q2 50 个 `pred_xstart` 第 3 通道和对应 `alpha_bar`；
   - 不保存候选分数、标签、表征或外部指标。
5. B 与 C 分别生成独立目录。每个 `scores.csv` 精确只有：`phase,global_seed,class_id,唯一候选列`。任何 `label`、`raw_consensus_label`、另一候选列或 Inception/DINO/FID/embedding 列都会失败。
6. Stage A 只打开最终盲标锁的 manifest、completion 和 `aggregate_counts.json`：
   - B：blur/fusion clear-bad ≥ 15、clean-good ≥ 60，且至少 3 个有事件并有 clean-good 的可比较类；
   - C：全部 clear-bad ≥ 30，其余要求相同；
   - 两者独立开门。
7. Stage B 是另一个进程。它才打开 `evaluation_labels.csv`，并且只打开已经过门的候选产品。未过门候选固定 `raw p=1`，其路径不会被检查、解析、`stat`、hash 或打开。
8. 通过门控的候选按冻结方向计算类内 pair-count 加权、tie-aware AUC；100,000 次置换共用同一批完整 128-global-seed block permutation；恰对 B/C 两项做 Holm。
9. 20 个 calibration seeds 只产生类别特异阈值：B 的 alpha=.10/.05 分别为第 19/20 顺序统计量，C 为第 2/1 顺序统计量，确认集使用严格不等式。
10. 输出只有聚合统计，不输出逐行分数、rank、置换样本、图像或轨迹。只有 B 通过原始全部门槛时才能授权后续 blur/fusion 干预；C 单独通过也没有干预权限。

## 关键程序

- `experiments/dit_event_rich_dynamic_contract.py`：协议、anchor plan、动态轴、单列产品与外部指标禁用合同。
- `experiments/sample_dit_event_rich_dynamic_traces.py`：`sample-one`、单次加载多任务 `sample-tasks`、精确整池 `finalize/validate`。
- `experiments/extract_dit_event_rich_candidate_product.py`：B-only 或 C-only 产品。
- `experiments/evaluate_dit_event_rich_dynamic_confirmation.py`：分进程 Stage A / Stage B。
- `experiments/freeze_dit_event_rich_dynamic_confirmation_sources.py`：冻结与验证源码锁。

真实运行必须从 v4 锁内的 `sources/` 调用，并先取得完整、不可变且可重放 GO 决策的 anchor plan。任务 JSON 的固定格式为：

```json
{
  "schema_version": 1,
  "anchor_plan_identity_sha256": "...",
  "tasks": [
    {"phase": "calibration", "global_seed": 1100, "class_id": 160}
  ],
  "tasks_sha256": "canonical SHA-256 of tasks"
}
```

任务可分给 4 张 GPU，分片、任务顺序、重启和 resume 都不改变科学 RNG。pool 只有在完整动态轴逐对 hash 复核后才会生成最终 receipt。

## 已完成的无真实数据验证

- B/C 数学公式与列 schema 自测通过。
- pair-keyed RNG known-answer、最小 trace shape/dtype/step axis 和 task-file poison 测试通过。
- `label`/`raw_consensus_label`、双候选列、Inception/DINO/FID/embedding 输入 poison 测试通过。
- 用 768 行合成 confirmation 标签完成真实 review artifact 结构的 Stage-A 聚合重放。
- 完成一项通过、一项失败的端到端合成测试：B 产品被打开，C 使用带 `DINO` 字样的不存在路径，Stage B 成功且未触碰该路径，C 的 raw p 精确为 1。
- 冻结 v4 的四组自测与 manifest/exact-tree 验证通过；未进行 GPU 采样。
