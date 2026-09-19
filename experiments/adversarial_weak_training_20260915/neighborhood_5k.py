"""Eight fixed 5K coefficient candidates per completed endpoint head.

Sampling only. The stopped training/search supervisors remain stopped. Reuse
verified 5K results and matching 1K batches; new coefficients go straight to 5K.
Run in tmux with ``python -m ...neighborhood_5k run``; interruption is resumable.
"""

import argparse
import csv
import fcntl
import io
import math
import os
from pathlib import Path
import subprocess
import time

from . import common as c
from experiments.weak_reference_loss_20260914 import idle


ROOT = c.ROOT / "coefficient_5k_neighborhood_20260916"
STOP = ROOT / "STOP_AFTER_CURRENT"
REPORT = c.WORK / "docs/ADVERSARIAL_HEADS_5K_NEIGHBORHOOD_20260916_ZH.md"
MODULE = "experiments.adversarial_weak_training_20260915.neighborhood_5k"
REFERENCE = c.original.EXPS.parent / "imagenet_sit_flow/adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz"

# A tick is the EXTRA coefficient a / 0.025, in S + a(S - W).
# User correction: exactly eight consecutive points, spaced by 0.05.
# Anchor at the best observed 5K coefficient; include three points to its left
# and four to its right. Early pilots have only the a=1 observation available.
GROUPS = (
    ("endpoint_binary_gan_rgb_v2", "RGB 二分类 GAN 续训", 1456, "head", 46),
    ("endpoint_binary_gan_rgb_v1", "RGB 二分类 GAN", 656, "head", 46),
    ("endpoint_energy_guided_v1", "条件 energy", 416, "ema", 43),
    ("endpoint_joint_moments_v1", "图像与类别联合分布矩", 1000, "ema", 45),
    ("endpoint_moments_v1", "分布矩", 600, "head", 47),
    ("endpoint_binary_gan_v1", "VAE 真实侧二分类 GAN", 416, "ema", 48),
    ("endpoint_gan_v1", "早期四来源 GAN v1", 200, "ema", 40),
    ("endpoint_gan_v2", "早期四来源 GAN v2", 600, "ema", 40),
)


def branch(group):
    return c.ROOT / group["run"] / "quality" / f"step{group['step']:06d}_{group['weights']}"


def point(group, tick):
    return branch(group) / f"c{tick:04d}"


def compact_metric(path):
    row = c.read(path)
    keys = ("n", "tick", "coefficient", "step", "weights", "fid", "inception_score",
            "checkpoint_sha256", "samples_sha256", "reference_sha256", "evaluator")
    return dict(path=str(path), **{key: row[key] for key in keys if key in row})


def checked_metric(group, tick, plan, deep=False):
    stage = point(group, tick) / "n5000"
    path = stage / "metrics.json"
    if not path.exists():
        return None
    row = c.read(path)
    assert row["complete"] and row["valid"] and row["n"] == 5000, path
    assert row["step"] == group["step"] and row["weights"] == group["weights"], path
    assert row["tick"] == tick and row["coefficient"] == tick / 40., path
    assert row["checkpoint_sha256"] == group["checkpoint_sha256"], path
    assert row["reference_sha256"] == plan["assets"][str(REFERENCE)], path
    assert len(row["records"]) == 625 and math.isfinite(row["fid"]), path
    summary = c.read(stage / "summary.json")
    assert summary["samples_sha256"] == row["samples_sha256"], path
    assert Path(row["samples_path"]).is_file(), path
    if deep:
        assert c.sha(row["samples_path"]) == row["samples_sha256"], path
    result = compact_metric(path)
    adm = c.read(stage / "adm.json")
    result["sfid"] = adm.get("sfid")
    return result


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / "plan.json").exists():
        return c.read(ROOT / "plan.json")
    bank = c.original.model_root("sit_small") / "quality_inputs"
    assets = {str(path): c.sha(path) for path in (bank / "noise.npy", bank / "labels.npy", REFERENCE)}
    plan = dict(created_utc=c.now(), authorization="2026-09-16 用户要求每组在最优解附近挑 8 个外推系数进行 5K 采样，防止 1K 排名误差",
        coefficient="extra a: S + a*f(t)*(S-W); original f(t), 64-step Heun and batch=8 preserved",
        scope="8 completed SiT-S/2 ImageNet-100 endpoint training runs; final checkpoint and previously chosen head/EMA",
        selection="User correction: exactly 8 consecutive coefficients spaced by 0.05; best observed 5K anchor plus 3 left and 4 right; reuse matching old 5K only",
        offsets=[-0.15, -0.10, -0.05, 0., 0.05, 0.10, 0.15, 0.20],
        early_gan_selection="Only a=1 was previously tested; use that sole observed coefficient as anchor, 0.85-1.20 directly at 5K",
        excluded=dict(endpoint_moment_prepared="Preparation only: identical initial/final head fingerprints", smoke="Engineering checks, not independent trained candidates"),
        no_training=True, no_new_1k_selection=True, independent_validation=False,
        inputs="Same historical fixed 5000 latents and labels; any existing first 1000 retained exactly",
        old_stop_file=str(c.ROOT / "STOP_AFTER_CURRENT"), old_stop_sha256=c.sha(c.ROOT / "STOP_AFTER_CURRENT"),
        stop_file=str(STOP), assets=assets, groups=[], sources=c.source_receipts(),
        baseline_5k_fid=36.688847797154665, previous_adversarial_best_5k_fid=36.588818101320044)
    for run, title, step, weights, anchor_tick in GROUPS:
        ticks = tuple(anchor_tick + offset for offset in range(-6, 9, 2))
        assert len(ticks) == len(set(ticks)) == 8 and all(0 < tick <= 80 for tick in ticks)
        assert all(right - left == 2 for left, right in zip(ticks, ticks[1:]))
        checkpoint = c.ROOT / run / f"checkpoint_{step:06d}.pt"
        assert c.read(c.ROOT / run / "complete.json")["step"] == step
        group = dict(run=run, title=title, step=step, weights=weights, ticks=list(ticks),
                     anchor_tick=anchor_tick, anchor_coefficient=anchor_tick / 40., spacing=0.05,
                     coefficients=[tick / 40. for tick in ticks], checkpoint=str(checkpoint),
                     checkpoint_sha256=c.sha(checkpoint))
        group["one_k_before"] = sorted([compact_metric(path) for path in branch(group).glob("c*/n1000/metrics.json")],
                                       key=lambda row: row["fid"])
        assert group["one_k_before"], run
        historical_5k = [compact_metric(path) for path in branch(group).glob("c*/n5000/metrics.json")]
        anchor = min(historical_5k or group["one_k_before"], key=lambda row: (row["fid"], row["tick"]))
        assert anchor["tick"] == anchor_tick, (run, anchor)
        group["anchor_evidence"] = anchor
        group["existing_5k_before"] = [row for tick in ticks if (row := checked_metric(group, tick, plan, deep=True))]
        plan["groups"].append(group)
    for filename, digest in plan["sources"].items():
        source = Path(filename)
        saved = ROOT / "source_snapshot" / source.relative_to(c.WORK)
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(source.read_bytes())
        assert c.sha(saved) == digest
    c.atomic(ROOT / "plan.json", plan)
    return plan


def verify(plan):
    for filename, digest in {**plan["assets"], **plan["sources"]}.items():
        assert c.sha(filename) == digest, f"Frozen input/source changed: {filename}"
    assert c.sha(plan["old_stop_file"]) == plan["old_stop_sha256"], "Original queue stop marker changed"
    for group in plan["groups"]:
        assert c.sha(group["checkpoint"]) == group["checkpoint_sha256"]


def report(plan):
    rows, groups = [], []
    for group in plan["groups"]:
        before = {row["tick"] for row in group["existing_5k_before"]}
        one_k = {row["tick"]: row for row in group["one_k_before"]}
        finished = []
        for tick in group["ticks"]:
            metric = checked_metric(group, tick, plan)
            row = dict(run=group["run"], title=group["title"], step=group["step"], weights=group["weights"],
                tick=tick, coefficient=tick / 40., n=5000,
                state="reused" if tick in before else "complete" if metric else "pending",
                fid_1k=one_k.get(tick, {}).get("fid"), fid_5k=metric["fid"] if metric else None,
                inception_score=metric.get("inception_score") if metric else None,
                sfid=metric.get("sfid") if metric else None,
                metrics=metric["path"] if metric else str(point(group, tick) / "n5000/metrics.json"))
            rows.append(row)
            if metric:
                finished.append(row)
        groups.append(dict(run=group["run"], title=group["title"], completed=len(finished), total=8,
            best=min(finished, key=lambda row: row["fid_5k"]) if finished else None))
    completed = sum(row["fid_5k"] is not None for row in rows)
    summary = dict(updated_utc=c.now(), complete=completed == len(rows), completed=completed,
        total=len(rows), reused=sum(row["state"] == "reused" for row in rows), groups=groups,
        interpretation="5K coefficient selection on a shared fixed bank; not independent validation or a global optimum claim")
    c.atomic(ROOT / "results.json", dict(summary=summary, rows=rows))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    temporary = ROOT / "results.csv.tmp"
    temporary.write_text(buffer.getvalue())
    temporary.replace(ROOT / "results.csv")
    lines = [f"更新：{summary['updated_utc']}。已完成 {completed}/{len(rows)} 个 5K 点（其中 {summary['reused']} 个复用原结果）。", "",
        "按用户更正：每组严格 8 个连续外推系数，相邻间隔 0.05。以当前最好 5K 系数为锚点，含锚点、左侧 3 点、右侧 4 点。",
        "只采样已有最终 checkpoint，保留原先选定的 raw/EMA 权重。历史上不落在本轮网格的结果保留，但不加入本轮 8 点或重新采样。",
        "系数为额外 a：S+a·f(t)·(S−W)，沿用原时间区间、64 步 Heun、每批 8 图、同一固定 5K 输入及 ADM 参考。",
        "新系数直接采样 5K；已有 1K 批次按哈希复用，不新增 1K 筛选。5K 用于选参，不是独立留出验证。",
        "旧训练和搜索队列的停止标记保留；新队列有自己的停止标记。", "",
        f"[冻结计划]({ROOT / 'plan.json'}) · [逐点 CSV]({ROOT / 'results.csv'}) · [运行状态]({ROOT / 'status.json'})", "",
        "| 组 | 步数 / 权重 | 最优锚点 | 8 个系数（间隔 0.05） | 完成 | 当前最好 a / 5K FID |",
        "| --- | --- | --- | --- | --- | --- |"]
    for group, result in zip(plan["groups"], groups):
        best = result["best"]
        value = f"{best['coefficient']:g} / {best['fid_5k']:.6f}" if best else "—"
        coefficients = ", ".join(f"{tick / 40.:g}" for tick in group["ticks"])
        lines.append(f"| {group['title']} | {group['step']} / {group['weights']} | {group['anchor_coefficient']:g} | {coefficients} | {result['completed']}/8 | {value} |")
    lines += ["", "早期四来源 GAN 仅有 a=1 的初筛，以这个唯一观测系数为锚点，0.85–1.20 直接 5K 补充搜索。",
        "工程准备 checkpoint 的头参数未变化，不另算一组；烟雾测试不纳入。",
        "系数表冻结后不自动加点，不额外启动 1K 筛选或独立 5K 验证。",
        "若最后最优点落在区间边界，报告会保留这一事实，不声称已经找到全局最优。", ""]
    REPORT.write_text("\n".join(lines))
    return summary


def run(plan):
    lock = (ROOT / "supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def publish(phase, **values):
        c.atomic(ROOT / "status.json", dict(phase=phase, pid=os.getpid(), updated_utc=c.now(), **values))

    try:
        verify(plan)
        for index, group in enumerate(plan["groups"]):
            missing = [tick for tick in group["ticks"] if checked_metric(group, tick, plan) is None]
            if not missing:
                continue
            while True:
                if STOP.exists():
                    publish("paused", summary=report(plan))
                    return
                devices = idle.gpu_snapshot()
                if sum(idle.eligible(row) for row in devices) >= 2:
                    break
                publish("waiting_idle_gpus", group=group["run"], gpus=devices, summary=report(plan))
                time.sleep(20)
            verify(plan)
            # Distinct attempt IDs allow a failed or paused action to resume its batches.
            attempt = 1
            prefix = f"neighborhood5k_20260916_{index:02d}"
            while (c.ROOT / group["run"] / f"{prefix}_try{attempt:02d}").exists():
                attempt += 1
            action_id = f"{prefix}_try{attempt:02d}"
            action = c.ROOT / group["run"] / action_id
            command = [c.PYTHON, "-u", "-m", "experiments.adversarial_weak_training_20260915.launch",
                "--hold", "--run", group["run"], "--mode", "evaluate", "--action-id", action_id,
                "--minimum-gpus", "2", "--stop-file", str(STOP), "--",
                "--checkpoint", group["checkpoint"], "--weights", group["weights"],
                "--samples", "5000", "--ticks", *map(str, missing)]
            action.mkdir(parents=True)
            with (action / "holder.log").open("a") as stream:
                process = subprocess.Popen(command, cwd=c.WORK, stdout=stream, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL)
                c.atomic(action / "queue_launch.json", dict(pid=process.pid, command=command, utc=c.now(), plan=str(ROOT / "plan.json")))
                print(c.now(), "start", group["run"], "a=", [tick / 40. for tick in missing], flush=True)
                while process.poll() is None:
                    publish("sampling_or_scoring", group=group["run"], action=str(action),
                            child_pid=process.pid, summary=report(plan))
                    time.sleep(20)
            if STOP.exists():
                publish("paused", summary=report(plan))
                return
            if process.returncode:
                raise RuntimeError(f"Evaluation exited {process.returncode}; inspect {action / 'holder.log'} and worker.log")
            # CPU fallback scorers are normally awaited by evaluate; never mark
            # success based on the launcher exit code alone.
            for tick in missing:
                if checked_metric(group, tick, plan, deep=True) is None:
                    raise RuntimeError(f"Evaluation did not produce complete 5K metrics: {group['run']} a={tick / 40.}")
            print(c.now(), "finished", group["run"], flush=True)
        summary = report(plan)
        assert summary["complete"]
        publish("complete", summary=summary)
        c.atomic(ROOT / "complete.json", summary)
    except Exception as error:
        publish("failed", error=repr(error))
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "report", "run"))
    args = parser.parse_args()
    plan = prepare() if args.action == "prepare" else c.read(ROOT / "plan.json")
    if args.action == "run":
        run(plan)
    else:
        summary = report(plan)
        print({key: summary[key] for key in ("completed", "total", "reused")})
