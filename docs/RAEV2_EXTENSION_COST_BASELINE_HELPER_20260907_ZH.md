# 5K补测：条件成本基线helper

本文件只说明已冻结补测协议中的必要成本恢复操作，不新增guidance方法。helper为 [run_raev2_extension_cost_baseline.py](../experiments/run_raev2_extension_cost_baseline.py)。准备期间仅完成CPU自检和编译；没有创建实际补跑plan，也没有启动GPU。

`prepare`只接受实际完成的 `scale_extension_5k_v1/cost_review_v1/summary.json`，要求显式给出审核后的SHA；核对审计输入、原四个worker全部terminal成功、冻结source，并从原始summary重新精确求和T/W，检查预设K建议一致。若201同时覆盖两项成本，则不创建目录或plan。若不覆盖，则只准备五个seed统一K的official作业。

GPU布局保持原cohort计划：GPU0→cohort1、GPU1→cohort2、GPU2→cohort3、GPU3→cohort4→cohort0。旧cohort0的计划已固定GPU3，不能改到GPU0。cohort1…4复用各自原生 `reflection_parity16`；cohort0先在新目录执行包装器的100步16图parity，再执行officialK。旧seed不重新生成reflection候选。

从仓库根目录运行；两个SHA占位符必须由实际完成且审核过的文件身份替换，不能用这些文字直接启动：

```bash
/home/zhoushunyu/miniconda3/envs/myenv/bin/python experiments/run_raev2_extension_cost_baseline.py self-test

/home/zhoushunyu/miniconda3/envs/myenv/bin/python experiments/run_raev2_extension_cost_baseline.py prepare --cost-summary-sha256 ACTUAL_COST_SUMMARY_SHA256

/home/zhoushunyu/miniconda3/envs/myenv/bin/python experiments/run_raev2_extension_cost_baseline.py launch --plan-sha256 REVIEWED_HELPER_PLAN_SHA256
```

默认新目录为 `scale_extension_5k_v1/cost_baseline_v1/`。`prepare`输出plan身份但不启动；`launch`另需该plan的SHA并创建不可复用的lock。四个detached worker各自记录argv、PID/PPID、GPU、起止UTC、退出码、外层wall与summary SHA；GPU3顺序运行三项子作业。已有任何输出或lock都拒绝重启/覆盖；失败保留原状，不能删除lock后当作首次运行。

完整执行记录在 `gpu_0…3/execution.json`；正式1000图summary在 `cohort_0…4/officialK/summary.json`，cohort0新增parity在 `cohort_0/reflection_parity16/`。所有原201和历史200结果仍保持原址、计入累计投入。worker末尾验证样本数、seed、步数、mode及主模型调用数；原包装器还执行自己的权重/source和parity身份检查。

helper不读FID、特征或图像分数，不合并图像、不评价，也不宣称新K已经覆盖实测成本。**新K全部完成后，root仍须重新核对五个cohort的 pooled T/W、parity/noise身份、额外成本和实际终止状态；如K仍不足，保持成本未匹配状态，按原协议作新的显式决定。** 该helper只支持从初始201审计导出的这一次预备，不接受伪造为初始审计的后续K结果。

准备阶段已通过的CPU检查覆盖：未完成/已放行评价/已启动新作业的审计拒绝；binary64 `nextafter` 边界；五个cohort统一K且完整；新cohort复用原parity；旧cohort0在GPU3的parity先于official；输出路径互异。真实GPU运行与成本覆盖尚未发生，不能由这些检查推断。
