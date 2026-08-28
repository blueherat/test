# Scientific v4.2.1：端点采样与外部盲评锁

## 当前结论

端点采样与盲评基础设施已经重新绑定到 scientific v4.2.1 和 method v2.2；旧的
v4.1 锁保持不变，不能换 manifest 冒充新版本。当前仍是设计锁，不包含任何真实端点、
真实标签或质量结果，也不授权 GPU 采样。

- scientific protocol：`af65e362cd8c543f898a3dbeb3a7b4478940966b48bfef689db7afdbad8d97d2`
- method v2.2：`cc4dc5e7c06c25f4d8567a42fb4f0387097a6296c587543830bfeaa4771f6921`
- endpoint source lock：`experiments/locks/dit_scientific_v4_2_1_endpoint_sampling_source_lock_v1`
  - sampling protocol：`afb468cd092942fffdee35d267e3dd5bd61623a778c7a099ddfd22a9a535f499`
  - source manifest identity：`ca01effe354104f878244f7d0493dccb839659a93fb4dd8862ef41a93610b1ff`
- review source lock：`experiments/locks/dit_scientific_v4_2_1_review_pipeline_source_lock_v1`
  - source identity：`8b4bab84604c4e7b8e41b7483aa1116c4a4e381d464aace393c5adfaeee3450f`
  - inner manifest identity：`b2b77bacfeef311d0ac30b6149bf0ab475858b2f4c943a901a51cccfd0f8fc2c`
  - review contract：`f6d11ff5840a63613014c1084794bccaf1d148200fe3d7b756e31c3913fe899b`

## 逻辑分片与物理 GPU 已解耦

1008 个 discovery 端点仍是固定的 84 类 × 12 seeds，并按冻结顺序切成四个逻辑分片
0、1、2、3，每片 252 对。逻辑分片决定任务；物理 GPU 只决定任务在哪里运行。

未来把相同 launcher 重新绑定进不可覆盖的 execution-ready source lock 后，可以只运行
任意固定子集，例如：

```bash
python <execution-ready-source-lock>/sources/run_dit_event_rich_endpoint_screen.py \
  --source-lock <execution-ready-source-lock> \
  --logical-workers 0,2 \
  --gpus 3
```

这表示逻辑分片 0、2 在物理 GPU 3 上依次运行。执行计划不记录这次物理映射；任务文件、
输出目录和 `(global_seed,class_id)` 派生 RNG 都只由冻结逻辑轴决定。因此换空闲卡、改变
启动批次或恢复顺序不会改变样本身份和随机流。

每个完成的逻辑分片单独产生并复核 receipt。少于四个 receipt 时，程序只报告已完成和
剩余逻辑分片，不会调用全 1008 轴验证，也不会发布 `pool_manifest.json` 或
`pool_completion.json`。四个 receipt 全部存在并逐片重验后，才允许整池验证和发布。

## 方法与外部裁判的防火墙

这些 endpoint 和视觉标签只用于构造、富集和评判外部验证集。FID、Inception、DINO、
CLIP、人工标签、端点图像或表征距离都不能进入 B、E、阈值、Q*、排序、触发或回滚。
method v2.2 只读取冻结生成器内部轨迹允许的量。

confirmation 的 `review_endpoints` 与 `method_traces` 必须是两个精确 sibling 目录。盲评
验证器递归拒绝 trace、B/E、score、metric、feature、embedding、threshold、alert 等
内部载荷或字段。

## 已通过的无真实数据检查

- endpoint：1008 对精确轴、四个固定逻辑分片、任意子集、单卡顺序调度、pair RNG
  known-answer、task-file resume/mutation、exact-tree 与隔离 provenance 验证通过；未加载
  模型或使用 GPU。
- review：外部 cwd、空 `PYTHONPATH` 下 source/provenance 25 项与完整合成 E2E 13 项通过；
  poison 列、树混用、缺失/重复/额外行及单裁决越权均 fail closed。
- 冻结前审计的三个真实 endpoint 输出前缀均为 0 个完成 pair、0 个 partial pair、0 张
  endpoint。

## 仍然阻塞真实执行的条件

1. 新的 v4.2.1 dynamic/selector/evaluator 源锁必须绑定上述两个 identity。
2. 两位独立 expert 完成 20 个可见 anchor ratification。
3. 独立 curator/resolver 建立互不重叠的 primary/reserve hidden gold。
4. 三位 reviewer 与两位 adjudicator 在隐藏资格表上通过全部门槛。
5. 另建不可覆盖的 execution-authorization receipt，绑定 scientific、method、endpoint、
   review、dynamic、selector/evaluator 和资格锁。
6. 真实执行时还必须有可用 GPU；当前设计锁的 `execution_ready=false`，不能直接启动采样。
