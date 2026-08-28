#!/usr/bin/env python3
"""Build the portable report payload from the frozen, self-contained SQL evidence table."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


REPORT_TITLE = "DiT v2.2 内部信号复核：E 退役，分支共识首版未通过"
REPORT_SOURCE_PATH = (
    "reports/dit_v22_internal_signal_reassessment/queries/report_analysis.sql"
)
METHOD_SOURCE_PATH = "docs/DIT_V22_INTERNAL_SIGNAL_REASSESSMENT_ZH.md"


def _run_sql(sql_text: str) -> list[dict[str, Any]]:
    completed = subprocess.run(
        ["sqlite3", "-json", ":memory:"],
        input=sql_text,
        text=True,
        check=True,
        capture_output=True,
    )
    rows = json.loads(completed.stdout)
    if not isinstance(rows, list) or len(rows) != 43:
        raise RuntimeError(f"Expected 43 reviewed SQL rows, received {len(rows)}")
    return rows


def _matched_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        if row["section"] != "matched_path_difference":
            continue
        result.append(
            {
                "pair_label": row["group_name"],
                "pair_index": row["pair_index"],
                "joint_success_count": row["joint_success_count"],
                "b_only_success_count": row["b_only_success_count"],
                "comparisons_per_path": row["comparisons_per_path"],
                "joint_rate": row["joint_success_count"]
                / row["comparisons_per_path"],
                "b_only_rate": row["b_only_success_count"]
                / row["comparisons_per_path"],
                "difference": row["difference"],
                "guard_satisfied": bool(row["opportunity_guard_satisfied"]),
                "result_id": row["result_id"],
            }
        )
    if len(result) != 8:
        raise RuntimeError(f"Expected 8 matched-path rows, received {len(result)}")
    return result


def _repair_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        if row["section"] != "strict_repair":
            continue
        result.append(
            {
                "row_order": row["row_order"],
                "group_name": row["group_name"],
                "rollback_step": row["rollback_step"],
                "numerator": row["numerator"],
                "denominator": row["denominator"],
                "rate": row["rate"],
                "opportunity_guard": "失败",
                "interpretation": row["note"],
                "result_id": row["result_id"],
            }
        )
    return result


def _audit_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    section_labels = {
        "blind_alarm_audit": "E 警报盲审",
        "opportunity_path": "修复机会门",
        "analysis_correction": "协议更正",
    }
    result = []
    for row in rows:
        if row["section"] not in section_labels:
            continue
        result.append(
            {
                "row_order": row["row_order"],
                "stage": section_labels[row["section"]],
                "group_name": row["group_name"],
                "numerator": row["numerator"],
                "denominator": row["denominator"],
                "rate": row["rate"],
                "note": row["note"],
                "result_id": row["result_id"],
            }
        )
    return result


def _branch_rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rank_labels = {
        1: "rank1 · medoid",
        2: "rank2",
        3: "rank3",
        4: "rank4 · max-outlier",
    }
    result = []
    for row in rows:
        if row["section"] != "branch_rank":
            continue
        result.append(
            {
                "row_order": row["row_order"],
                "horizon": row["horizon"],
                "horizon_label": f"h={row['horizon']}",
                "centrality_order": row["centrality_order"],
                "rank_label": rank_labels[row["centrality_order"]],
                "success_count": row["numerator"],
                "job_count": row["denominator"],
                "rate": row["rate"],
                "random_rate": row["comparator_rate"],
                "difference": row["difference"],
                "note": row["note"],
                "result_id": row["result_id"],
            }
        )
    if len(result) != 12:
        raise RuntimeError(f"Expected 12 branch-rank rows, received {len(result)}")
    return result


def _branch_audit_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        include = (
            row["section"] in {"branch_detection", "branch_safety"}
            or (
                row["section"] == "branch_rank"
                and (
                    row["centrality_order"] == 4
                    or (row["horizon"] == 10 and row["centrality_order"] == 1)
                )
            )
        )
        if not include:
            continue
        status = "冻结首版失败" if row["row_order"] in {64, 80} else "事后线索/安全审计"
        selected.append(
            {
                "row_order": row["row_order"],
                "metric": row["group_name"],
                "horizon": row["horizon"],
                "observed": row["rate"],
                "comparator": row["comparator_rate"],
                "difference": row["difference"],
                "status": status,
                "note": row["note"],
                "result_id": row["result_id"],
            }
        )
    if len(selected) != 7:
        raise RuntimeError(f"Expected 7 branch-audit rows, received {len(selected)}")
    return selected


def _build_payload(sql_text: str, rows: list[dict[str, Any]], generated_at: str) -> dict[str, Any]:
    report_source = {
        "id": "report_analysis",
        "label": "内部信号复核分析 SQL",
        "path": REPORT_SOURCE_PATH,
    }
    method_source = {
        "id": "method_note",
        "label": "内部信号复核与下一方法说明",
        "path": METHOD_SOURCE_PATH,
    }

    matched_rows = _matched_rows(rows)
    repair_rows = _repair_rows(rows)
    audit_rows = _audit_rows(rows)
    branch_rank_rows = _branch_rank_rows(rows)
    branch_audit_rows = _branch_audit_rows(rows)

    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": REPORT_TITLE,
            "description": "跨尺度 e-process、分支共识首版及反向短暂离群线索的盲化复核与下一轮冻结验证。",
            "generatedAt": generated_at,
            "charts": [
                {
                    "id": "matched_strict_repair_difference",
                    "title": "8 个匹配路径的严格修复率差",
                    "subtitle": "Joint E+B − B-only；每条路径含 8 个匿名比较；机会门失败，区间仅作事后稳定性描述",
                    "showDescription": True,
                    "type": "bar",
                    "dataset": "matched_path_differences",
                    "sourceId": "report_analysis",
                    "encodings": {
                        "x": {
                            "field": "pair_label",
                            "type": "ordinal",
                            "label": "匹配路径对",
                        },
                        "y": {
                            "field": "difference",
                            "type": "quantitative",
                            "label": "严格修复率差",
                            "format": "percent",
                        },
                        "tooltip": [
                            {"field": "pair_index", "label": "匹配对"},
                            {
                                "field": "joint_success_count",
                                "label": "Joint 成功数",
                            },
                            {
                                "field": "b_only_success_count",
                                "label": "B-only 成功数",
                            },
                            {
                                "field": "comparisons_per_path",
                                "label": "每路径比较数",
                            },
                            {
                                "field": "difference",
                                "label": "Joint − B-only",
                                "format": "percent",
                            },
                        ],
                    },
                    "xAxisTitle": "匹配路径对",
                    "yAxisTitle": "Joint E+B − B-only",
                    "valueFormat": "percent",
                    "unit": "比例点",
                    "intent": "comparison",
                    "question": "匹配路径是否显示 E 对严格修复率的增量优势？",
                    "rationale": "有符号柱形图保留零点，并逐对展示方向，避免被不匹配的 pooled rate 误导。",
                    "comparisonContext": {
                        "paired": True,
                        "comparisonsPerPath": 8,
                        "guardSatisfied": False,
                        "confirmatory": False,
                    },
                    "referenceLines": [{"value": 0, "label": "无差异"}],
                    "maxRows": 8,
                    "layout": "full",
                },
                {
                    "id": "branch_rank_success_by_horizon",
                    "title": "四个内部 rank 的严格修复率",
                    "subtitle": "9 条 opportunity path、18 个 job；虚线为同四条 scout 均匀选择的 37.5% 期望",
                    "showDescription": True,
                    "type": "bar",
                    "dataset": "branch_rank_success",
                    "sourceId": "report_analysis",
                    "encodings": {
                        "x": {
                            "field": "horizon_label",
                            "type": "ordinal",
                            "label": "短程 horizon",
                        },
                        "y": {
                            "field": "rate",
                            "type": "quantitative",
                            "label": "严格修复率",
                            "format": "percent",
                        },
                        "color": {
                            "field": "rank_label",
                            "type": "nominal",
                            "label": "内部中心性 rank",
                        },
                        "tooltip": [
                            {"field": "horizon", "label": "horizon"},
                            {"field": "rank_label", "label": "内部 rank"},
                            {"field": "success_count", "label": "成功数"},
                            {"field": "job_count", "label": "job 数"},
                            {"field": "rate", "label": "修复率", "format": "percent"},
                            {"field": "difference", "label": "相对随机", "format": "percent"},
                        ],
                    },
                    "xAxisTitle": "短程 horizon",
                    "yAxisTitle": "严格修复率",
                    "valueFormat": "percent",
                    "intent": "comparison",
                    "question": "medoid 与 max-outlier 的方向是否跨 horizon 稳定？",
                    "rationale": "同时展示全部四个 rank，避免只报告揭盲后最好的 rank4 而隐藏 max-of-four 选择偏差。",
                    "comparisonContext": {
                        "pairedWithinJob": True,
                        "opportunityPaths": 9,
                        "jobs": 18,
                        "posthocDirection": True,
                        "confirmatory": False,
                    },
                    "referenceLines": [{"value": 0.375, "label": "同预算均匀随机"}],
                    "maxRows": 12,
                    "layout": "full",
                }
            ],
            "tables": [
                {
                    "id": "audit_evidence",
                    "title": "E 警报、机会门与协议更正",
                    "subtitle": "这些结果只评价内部候选量，不进入在线采样规则。",
                    "showDescription": True,
                    "dataset": "audit_evidence",
                    "sourceId": "report_analysis",
                    "defaultSort": {"field": "row_order", "direction": "asc"},
                    "density": "comfortable",
                    "layout": "full",
                    "columns": [
                        {"field": "row_order", "label": "顺序", "type": "number"},
                        {"field": "stage", "label": "审计阶段", "type": "text"},
                        {"field": "group_name", "label": "事件或组别", "type": "text"},
                        {"field": "numerator", "label": "分子", "type": "number"},
                        {"field": "denominator", "label": "分母", "type": "number"},
                        {"field": "rate", "label": "比例", "type": "percent", "format": "percent"},
                        {"field": "note", "label": "定义/限制", "type": "text"},
                    ],
                },
                {
                    "id": "strict_repair_summary",
                    "title": "协议一致的严格修复汇总",
                    "subtitle": "成功要求模糊明确减轻、语义与构图保持，且 fresh 未出现同等严重缺陷。",
                    "showDescription": True,
                    "dataset": "strict_repair_summary",
                    "sourceId": "report_analysis",
                    "defaultSort": {"field": "row_order", "direction": "asc"},
                    "density": "comfortable",
                    "layout": "full",
                    "columns": [
                        {"field": "row_order", "label": "顺序", "type": "number"},
                        {"field": "group_name", "label": "组别/范围", "type": "text"},
                        {"field": "rollback_step", "label": "回滚步", "type": "number"},
                        {"field": "numerator", "label": "成功数", "type": "number"},
                        {"field": "denominator", "label": "比较数", "type": "number"},
                        {"field": "rate", "label": "严格修复率", "type": "percent", "format": "percent"},
                        {"field": "opportunity_guard", "label": "机会门", "type": "text"},
                    ],
                },
                {
                    "id": "branch_hypothesis_audit",
                    "title": "分支共识首版与反向线索审计",
                    "subtitle": "medoid/high-O 是冻结失败；max-outlier 与安全结果均为揭盲后审计。",
                    "showDescription": True,
                    "dataset": "branch_hypothesis_audit",
                    "sourceId": "report_analysis",
                    "defaultSort": {"field": "metric", "direction": "asc"},
                    "density": "comfortable",
                    "layout": "full",
                    "columns": [
                        {"field": "metric", "label": "指标", "type": "text"},
                        {"field": "horizon", "label": "horizon", "type": "number"},
                        {"field": "observed", "label": "观察值", "type": "percent", "format": "percent"},
                        {"field": "comparator", "label": "比较值", "type": "percent", "format": "percent"},
                        {"field": "difference", "label": "差值", "type": "percent", "format": "percent"},
                        {"field": "status", "label": "证据状态", "type": "text"},
                        {"field": "note", "label": "限制", "type": "text"},
                    ],
                },
            ],
            "sources": [report_source, method_source],
            "blocks": [
                {
                    "id": "title",
                    "type": "markdown",
                    "body": f"# {REPORT_TITLE}",
                },
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "sourceId": "report_analysis",
                    "body": (
                        "## 技术摘要\n\n"
                        "跨尺度路径比值 `E` 的鞅记账是正确的，但两轮盲化检验均未建立它与可见坏图或可修复性的对应关系。"
                        "通用警报审计中，严格 clear-bad 为 `1/26` 对 `2/26`；修复 pilot 的机会门又以 `3/8` 对 `6/8` 失败。"
                        "因此当前 `E` 应退出质量报警、回滚与 guidance；`B` 只保留为弱表型量。"
                        "随后冻结的分支共识首版也没有通过：`h=10` medoid 严格修复为 `5/18=27.8%`，低于同四条 scout 均匀随机的 `37.5%`；较高 attempt-0 `O` 的 opportunity AUC 仅 `0.286`。"
                        "外部视觉标签、FID、DINO、CLIP 或 Inception 只充当封存后的裁判，绝不进入 selector、触发器或阈值。"
                    ),
                },
                {
                    "id": "key_findings",
                    "type": "markdown",
                    "sourceId": "report_analysis",
                    "body": (
                        "## 关键发现\n\n"
                        "第一版 analyzer 把“偏好 fresh”误当成“模糊明确减轻”，多算了 17 个成功。协议一致更正后，Joint E+B 为 `10/64`，B-only 为 `17/64`。"
                        "机会内 pooled rate 看似是 `43.5%` 对 `35.4%`，但它来自 3 条对 6 条不同路径，不能归因给 `E`。真正的一一匹配差值中，4 对为负、0 对为正、4 对持平。"
                        "在分支 rank 揭盲后，`h=10` 最离群分支是 `10/18=55.6%`，比随机期望高 `18.1` 个百分点；但这是看到 medoid 失败后才提出的反向解释，且 `h=5` 仅 `+6.9`、`h=20` 仅 `+1.4` 个百分点。"
                        "非机会样本的 clear-bad 与 preservation 安全读数也都略差于随机，所以它只能触发一次新数据上的窄验证，不能被写成已成立的方法。"
                    ),
                },
                {
                    "id": "matched_chart",
                    "type": "chart",
                    "chartId": "matched_strict_repair_difference",
                    "layout": "full",
                },
                {
                    "id": "branch_first_form_result",
                    "type": "markdown",
                    "sourceId": "report_analysis",
                    "body": (
                        "## 分支共识首版没有通过\n\n"
                        "首版具体预测是：四条短程 fresh scout 中的 medoid 更可靠，且 current 相对 fresh 共识的 `O` 越高越像可修复事故。"
                        "冻结结果方向相反：`h=10` medoid 为 `5/18`，随机期望为 `6.75/18`；higher-`O` AUC 为 `0.286`。"
                        "把符号翻过来得到 low-`O` AUC `0.714` 并不是第二份独立证据，而只是同一个失败 AUC 的代数翻转。"
                    ),
                },
                {
                    "id": "branch_rank_chart",
                    "type": "chart",
                    "chartId": "branch_rank_success_by_horizon",
                    "layout": "full",
                },
                {
                    "id": "branch_reverse_boundary",
                    "type": "markdown",
                    "sourceId": "report_analysis",
                    "body": (
                        "## 反向结果为什么仍只是线索\n\n"
                        "完整四-rank 结果必须一起看：`h=10` 从中心到离群依次是 `5, 7, 5, 10/18`，并非随 nonconformity 单调改善。"
                        "rank4 是揭盲后从四个 rank 中挑出的最好方向，样本只有 9 条独立原路径，而且增益主要集中在 class 795。"
                        "因此当前只能把它叫作“短暂 minority-branch escape 候选”，不能声称多数分支形成了错误共识盆地。"
                    ),
                },
                {
                    "id": "branch_audit_table",
                    "type": "table",
                    "tableId": "branch_hypothesis_audit",
                    "layout": "full",
                },
                {
                    "id": "scope_and_definitions",
                    "type": "markdown",
                    "sourceId": "method_note",
                    "body": (
                        "## 范围、定义与方法边界\n\n"
                        "`P` 是冻结的实际基线采样器；`Q*` 是人为跨尺度备择；`E_k` 是已实现两条 Markov 路径之间的累计似然比；`B_k` 是 predicted-clean 内部得到的低边缘/软融合持续量。"
                        "`E` 只回答路径更像 `Q*` 还是 `P`，不自动回答图像是否坏。盲评得到的 opportunity 与 successful repair 都是事后外部终点，只能验证内部指标，不能反馈到在线判定。"
                    ),
                },
                {
                    "id": "audit_table_intro",
                    "type": "markdown",
                    "sourceId": "report_analysis",
                    "body": (
                        "## 证据链审计\n\n"
                        "表中同时保留警报盲审、冻结机会门和协议更正。它们回答的是候选内部量有没有外部效度，而不是构造一个由外部分数驱动的新方法。"
                    ),
                },
                {
                    "id": "audit_table",
                    "type": "table",
                    "tableId": "audit_evidence",
                    "layout": "full",
                },
                {
                    "id": "repair_table_intro",
                    "type": "markdown",
                    "sourceId": "report_analysis",
                    "body": (
                        "## 严格修复结果\n\n"
                        "后缀重采并非完全无效：在 71 个被盲评确认存在模糊修复机会的比较中，有 27 个满足严格修复定义。"
                        "但成功和失败混在一起，当前又没有可靠内部 selector，因此不能把“偶尔可修”写成“方法已能自动改善”。"
                    ),
                },
                {
                    "id": "repair_table",
                    "type": "table",
                    "tableId": "strict_repair_summary",
                    "layout": "full",
                },
                {
                    "id": "methodology",
                    "type": "markdown",
                    "sourceId": "method_note",
                    "body": (
                        "## 实验设计与误差控制\n\n"
                        "pilot 使用 8 条 Joint E+B 与 8 条连续 `B` 强度匹配的 B-only 路径；每条在两个回滚位置保留一次精确重放和四条 fresh 后缀，共 160 个分支。"
                        "128 个 baseline–fresh 对由三名彼此隔离、看不到 seed、角色、分数和另两人回答的评审判断。统计独立单位是原路径；step 109 与 step 149 不能被当成 32 条独立路径。"
                        "分支内部提取器只打开 manifest、branch metadata 以及 `target_pred_xstart`/timestep 数组；一次逐行复核发现旧实现用 h 时刻 attempt-0 做尺度归一化，而文档声称共同前缀。"
                        "改为真正 h0 共同前缀后，32 个 job 的 max/min 排名和上述成功计数均未变化；新 prospective 实现已固定采用共同前缀。"
                    ),
                },
                {
                    "id": "uncertainty",
                    "type": "markdown",
                    "sourceId": "report_analysis",
                    "body": (
                        "## 不确定性与稳健性\n\n"
                        "机会门失败使组间 repairability 比较按冻结协议必须记为 inconclusive。配对 bootstrap `[-20.3,-3.1]` 个百分点和 leave-one-pair-out 方向一致，只是揭盲后的稳定性描述，不是确认性区间或因果效应。"
                        "本试验不能证明 `E` 会伤害图像；能支持的动作只是停止把它解释成质量触发器。max-outlier 的非机会安全审计为 clear-bad `14.3%` 对 `10.7%`、preservation `85.7%` 对 `87.5%`，样本很小但方向要求下一轮设置硬安全门。"
                    ),
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "sourceId": "method_note",
                    "body": (
                        "## 下一步：冻结检验短暂离群分支，而不是继续修补首版\n\n"
                        "新实验只读取冻结采样器内部的 predicted-clean latent：在 step149 的共同前缀上生成四条对称 scout，完成 10 次转移后计算多尺度两两距离，选择平均距离最大的分支。"
                        "主比较是同四条 scout 的均匀随机策略值，因此计算预算一致；四条都跑到终点只属于实验评估开销，部署时可在 h10 后只继续一条。"
                        "96 个 class-795 prefix 用于确认最初线索，class 207/602 各 16 个只做明显伤害哨兵。内部选择表必须在打开任何 PNG、人工标签、FID 或表征模型结果前封存。"
                    ),
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "sourceId": "method_note",
                    "body": (
                        "## 仍需回答的问题\n\n"
                        "现有 18 个 opportunity job 来自 `B/E` 富集 pilot，max-outlier 方向又是在揭盲后提出，不能代表普通基线分布。"
                        "新 suffix 虽然未生成过，但其共同 prefix 来自曾用于历史研究的第三池，因此最准确的名称是 new-suffix conditional prospective，而不是完全独立 fresh-prefix confirmation。"
                        "即使新主结果通过，它也只证明窄条件下的分支选择优势；仍缺少一个纯内部 trigger 来判断哪些 prefix 值得付出四-scout 计算，也不能据此声称 FID 改善或罕见灾难率受控。"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "matched_path_differences": matched_rows,
                "audit_evidence": audit_rows,
                "strict_repair_summary": repair_rows,
                "branch_rank_success": branch_rank_rows,
                "branch_hypothesis_audit": branch_audit_rows,
            },
        },
        "sources": [
            {
                **report_source,
                "query": {
                    "engine": "sqlite3",
                    "sql": sql_text,
                    "description": "从冻结结果标识、协议一致更正与分支假设审计中形成报告用的 43 行证据表。",
                    "language": "sql",
                    "executed_at": generated_at,
                    "tables_used": ["inline CTE report_evidence"],
                    "filters": [
                        "仅纳入冻结 E 警报盲审、repairability pilot、协议一致更正与匹配路径结果",
                        "分支 rank 方向来自 posthoc 审计，不能与冻结 medoid/high-O 主预测混同",
                        "不读取 FID、DINO、CLIP、Inception 或终点特征作为方法输入",
                    ],
                    "metric_definitions": [
                        "rate = numerator / denominator",
                        "matched difference = Joint E+B strict repair rate − matched B-only strict repair rate",
                        "strict repair requires reviewer-level blur reduction, preservation, and no fresh 2-of-3 clear-bad consensus",
                        "opportunity guard requires at least four stable opportunity paths per role",
                        "branch random baseline = uniform policy value over the same four already-computed scouts",
                        "max-outlier, h5/h20, and safety rows are posthoc descriptive evidence",
                    ],
                },
            },
            method_source,
        ],
        "package_info": {
            "report_kind": "technical_research_audit",
            "language": "zh-CN",
            "external_metrics_role": "evaluation_only",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("artifact.json"),
    )
    args = parser.parse_args()

    report_dir = Path(__file__).resolve().parent
    sql_path = report_dir / "queries" / "report_analysis.sql"
    sql_text = sql_path.read_text(encoding="utf-8")
    rows = _run_sql(sql_text)
    generated_at = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    payload = _build_payload(sql_text, rows, generated_at)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sql_rows": len(rows),
                "datasets": {
                    key: len(value)
                    for key, value in payload["snapshot"]["datasets"].items()
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
