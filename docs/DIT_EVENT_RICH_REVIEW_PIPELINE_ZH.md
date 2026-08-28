# Event-rich 端点盲评管线（canonical v7）

## 当前状态

冻结锁：`experiments/locks/dit_event_rich_review_pipeline_source_lock_v7`

- source-lock identity：`b015b4c173db1e09da2608ccb5bd3958be185fa876e304e4f80f316a098050c8`
- 基于 event-rich protocol v3：`04e933793992e2a7ce62aa4ac66836412f3c4f221cce731f2e072da97e892dd7`
- 当前只有代码、schema、空白表和合成自测，没有真实 expert/reviewer/adjudicator 结果。
- `ready_for_real_sampling=false`；此锁本身不授权正式采样或发放正式评审图片。
- v1–v6 均完整保留，均没有真实结果；各版为何被 supersede 已逐项冻结在 v7 的 `review_contract.json` 中。

## 不可自动化的前置输入

1. 两个真正独立且合格的 expert 分别填写 20 个可见 anchor 的 ratification 表；任一项不 ratify 就停止并修改协议。
2. 两个独立 expert curator 分别建立 60 项 primary hidden gold 和 60 项完全不重叠的 reserve hidden gold；分歧项交给第三个独立 resolver。软件不会假称这些角色是人类，也不会伪造结果。
3. 三个 reviewer 和两个 adjudicator 共五个独立角色完成隐藏资格表。每个人相对 gold 的 clear-bad recall 与 non-clear-bad specificity 均须至少 0.80；十个角色对的 positive agreement 均须至少 0.60、binary Cohen kappa 均须至少 0.50。
4. 任一门槛失败即 `STOP`，本表永久消耗；替换或重新训练失败角色后，整个五人 panel 只能使用预先冻结且不重叠的 reserve 表重新资格审查。

这些角色可由合格人工团队或合格外部盲评服务承担；当前尚未完成。

## 正式标签流程

每个 discovery、anchor、confirmation 阶段都依次执行：

1. 锁定实际存在且哈希完整的端点 cohort 和精确轴；
2. 仅在五角色资格锁为 PASS 后，生成三个彼此隔离的 reviewer delivery；
3. 锁定三份 endpoint-only review；
4. 将“任一 reviewer 给 severity≥2”的全集与等量、冻结随机抽取的零阳性 decoy 混合，生成两个彼此隔离的 adjudicator delivery；
5. 锁定双裁决并形成 consensus。

规则被写死为：

- 3/3 clear-bad 永不降级；
- 2/3 clear-bad 只有两位 adjudicator 都独立判为 non-clear-bad 才降级；
- 1/3 阳性以及零阳性 decoy，只有两位 adjudicator 都独立判为 clear-bad 才升级；
- 单个 adjudicator 不能改变最终标签；任何实际改变必须有两份 component 与书面定位理由。

各角色只获得自己的 `delivery/<role_slot>`。reviewer 看不到 B/C、类风险排名、轨迹、指标、阈值、embedding 或其他投票；adjudicator 还看不到 reviewer 身份、票数、票源、trigger/decoy 身份和另一位裁决结果。

## 下游唯一精简接口

最终 consensus 同时输出详细审计表和 `evaluation_labels.csv`。后者精确列为：

```text
phase,global_seed,class_id,final_severity,blur_component
```

其中 `final_severity` 仅允许 `clean_good`、`mild_or_disputed`、`clear_bad`；`blur_component` 必须是 `0/1`，且 `1` 只允许出现在最终 clear-bad。这与冻结的 `select_dit_event_rich_classes.py` 输入合同一致。

confirmation consensus 的 manifest identity 还会绑定权威 anchor plan identity；空 GO union 会直接停止，不能建立 confirmation review pack。

## 验证

```bash
PYTHONDONTWRITEBYTECODE=1 python experiments/selftest_dit_event_rich_review_pipeline.py \
  --source-lock experiments/locks/dit_event_rich_review_pipeline_source_lock_v7
```

该自测只使用临时合成逻辑，不创建科学标签；它覆盖毒化 score/label 列、缺失/重复/额外行、资格门、JSON 顺序、3/3/2/3/1/3/decoy 共识规则和 frozen-lock 完整性。
