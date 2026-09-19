# 原大队列自动接续


最新执行门槛：按随后指令，先完成[五个机制候选](SIT_MEASURE_GUIDANCE_PROTOCOL_20260912_ZH.md)及审计，再补旧优先批次，最后才启动本接续器。本文件最初描述的直接恢复尚未启动即被该顺序覆盖。
2026-09-12 用户明确要求：“自动恢复大队列，把之前的跑完”。这条指令覆盖 9 月 11 日优先批次结束后继续暂停的安排。

执行顺序为：当前 100 组优先批次完成并释放锁后，恢复原 709 组 control 队列；全部配置与最终审计完成后，恢复原 201 组 APG 扩展。已有提交由原采样器校验后跳过，优先批次写入的结果也计入各自原队列。

优先批次结束时，control 预计已有 324/709 组，随后补 385 组；APG 已有 76/201 组，随后补 125 组。因此接续阶段预计新跑 510 组，每组 1K，实际启动前再次按提交记录计算。没有新增参数网格或自动 5K。

接续器是独立后台进程，不修改正在使用的冻结采样源码、原请求或旧协议。它等待优先调度器正常完成并释放控制锁，再归档由旧调度器写入的暂停标记，调用原生队列入口。两个大队列按顺序运行，不同时分配 GPU。

每个阶段继续执行原有来源、模型、输入、历史提交、四卡运行与最终结果审计。若当前优先批次失败、出现新的暂停指令，或原队列恢复后未正常完成，接续器记录原因并停止后续启动。

运行目录为 /home/zhoushunyu/data/eqvae/experiments/sit_broad_resume_20260912。plan.json 保存授权与接续范围；status.json 保存当前状态；controller.log 是接续日志，control.log / apg.log 分别记录原队列进程输出。

启动命令为 python -m experiments.resume_sit_broad_20260912 --run，后台 tmux 为 sit_broad_resume_0912。如需暂停全部后续执行，使用 python -m experiments.resume_sit_broad_20260912 --stop-after-current；当前配置提交后停止。

[当前接续状态](/home/zhoushunyu/data/eqvae/experiments/sit_broad_resume_20260912/status.json) · [优先批次](SIT_REFINED_PRIORITY_EXECUTION_20260911_ZH.md) · [709 组结果](SIT_CONTROL_53_IDEAS_RESULTS_20260910_ZH.md) · [201 组结果](SIT_APG_EXTENSION_RESULTS_20260911_ZH.md)
