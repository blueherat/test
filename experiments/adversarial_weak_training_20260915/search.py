"""CPU supervisor: wait for training, evaluate, and optionally finish fixed search.

This never resumes the stopped old queue and never silently changes a training
hyperparameter. A pilot ends with a real metric for the research decision. Full
mode follows the user's connected 0.4/0.2/0.1/0.025 search and top-two 5K rule.
"""

import argparse
import os
from pathlib import Path
import subprocess
import time
from . import common as c
from experiments.guidance_strength_sweep_20260915.planning import refine


def alive(pid):
    path = Path("/proc") / str(pid) / "stat"
    try:
        line = path.read_text()
    except FileNotFoundError:
        return False
    return line[line.rfind(")") + 2:].split()[0] != "Z"


def initial_baseline(run):
    request = c.read(run / 'request.json')
    resume = request.get('resume_checkpoint')
    if resume:
        checkpoint = Path(resume).resolve()
        step = int(checkpoint.stem.removeprefix('checkpoint_'))
        # Training restores state['head'], not the checkpoint's EMA. Both are
        # screened in the established evaluation protocol, so use the raw head
        # metric when identifying the actual starting model for continuation.
        path = checkpoint.parent / f'quality/step{step:06d}_head/c0040/n1000/metrics.json'
        if path.exists():
            row = c.read(path)
            assert row['n'] == 1000 and row['coefficient'] == 1.
            assert row['checkpoint_sha256'] == c.sha(checkpoint)
            return dict(head=f'{checkpoint.parent.name}:head', coefficient=1.,
                        fid=row['fid'], metrics=str(path), exact_initial_model=True,
                        kind='resumed_raw_head', checkpoint=str(checkpoint))
    name = Path(request['initial_head']).parent.name
    path = c.original.model_root('sit_small') / f'points/{name}__c0040/n1000/metrics.json'
    row = c.read(path)
    assert row['n'] == 1000
    result = dict(head=name, coefficient=1., fid=row['fid'], metrics=str(path),
                  exact_initial_model=not bool(resume),
                  kind='historical_reference_without_resume_metric' if resume else 'pretrained_initial_head')
    if resume:
        result['resume_checkpoint'] = str(checkpoint)
        result['limitation'] = 'No matching raw-head a=1 metric for the resumed checkpoint; this is a historical reference'
    return result


def main(args):
    run = c.ROOT / args.run
    name = f"search_step{args.step:06d}_{args.weights}" + ("_pilot" if args.pilot else "_full")
    root = run / name
    root.mkdir(parents=True, exist_ok=True)
    c.atomic(root / "launch.json", dict(pid=os.getpid(), started_utc=c.now(), args=vars(args)))
    def publish(phase, **values):
        c.atomic(root / "status.json", dict(phase=phase, updated_utc=c.now(), **values))
    while not (run / "complete.json").exists() or c.read(run / "training/status.json")["phase"] == "running":
        if c.stopped():
            publish("paused")
            return
        if (run / "training/status.json").exists():
            status = c.read(run / "training/status.json")
            if status["phase"] == "failed":
                raise RuntimeError("Training failed; inspect training/worker.log")
        progress = c.read(run / "progress.json") if (run / "progress.json").exists() else {}
        publish("waiting_training", training_step=progress.get("step"))
        time.sleep(10)
    checkpoint = run / f"checkpoint_{args.step:06d}.pt"
    assert checkpoint.exists()
    branch = run / "quality" / f"step{args.step:06d}_{args.weights}"

    def evaluate(ticks, n, action_id):
        missing = [tick for tick in ticks if not (branch / f"c{tick:04d}" / f"n{n}/metrics.json").exists()]
        if missing:
            action = run / action_id
            command = [c.PYTHON, "-m", "experiments.adversarial_weak_training_20260915.launch",
                "--run", args.run, "--mode", "evaluate", "--action-id", action_id, "--",
                "--checkpoint", str(checkpoint), "--weights", args.weights,
                "--samples", str(n), "--ticks", *map(str, missing)]
            if not (action / "launcher.json").exists():
                subprocess.run(command, cwd=c.WORK, check=True)
            while True:
                if c.stopped():
                    raise InterruptedError("New experiment stop requested")
                pending = [tick for tick in missing if not (branch / f"c{tick:04d}" / f"n{n}/metrics.json").exists()]
                publish("sampling_or_scoring", n=n, ticks=ticks, pending=pending, action_id=action_id)
                if not pending:
                    break
                if (action / "status.json").exists() and c.read(action / "status.json")["phase"] == "failed":
                    raise RuntimeError(f"Sampling failed: {action}")
                holder = c.read(action / "launcher.json")["pid"]
                if not (action / "status.json").exists() and not alive(holder):
                    raise RuntimeError(f"GPU admission/launch failed: {action}")
                for tick in pending:
                    stage = branch / f"c{tick:04d}" / f"n{n}"
                    if (stage / "score_launch.json").exists():
                        score = c.read(stage / "score_launch.json")
                        if not alive(score["pid"]) and not (stage / "metrics.json").exists():
                            raise RuntimeError(f"CPU score failed: {stage}")
                time.sleep(10)
            # GPU leases may take a few seconds to release after metrics finish.
            while (action / "status.json").exists() and c.read(action / "status.json")["phase"] == "running":
                time.sleep(2)
        return [c.read(branch / f"c{tick:04d}" / f"n{n}/metrics.json") for tick in ticks]

    try:
        if args.pilot:
            rows = evaluate([40], 1000, f"pilot_step{args.step:06d}_{args.weights}")
            baseline = initial_baseline(run)
            result = dict(complete=True, kind="fixed-coefficient development screen", rows=rows,
                initial_baseline=baseline, delta_from_initial=rows[0]['fid'] - baseline['fid'],
                baseline_real_same_coefficient_fid=64.14058231741768,
                previous_best_1k_fid=63.6736167185403,
                delta_from_real=rows[0]["fid"] - 64.14058231741768,
                updated_utc=c.now())
            c.atomic(root / "result.json", result)
            publish("complete", fid=rows[0]["fid"], delta_from_real=result["delta_from_real"])
            print(result["rows"][0]["fid"], flush=True)
            return
        ticks = list(range(0, 81, 16))
        stages, union = [], {}
        for index, width in enumerate((16, 8, 4, 1)):
            rows = evaluate(ticks, 1000, f"full_step{args.step:06d}_{args.weights}_stage{index}")
            union.update({row["tick"]: row for row in rows})
            stage = dict(width=width / 40., ticks=ticks, rows=rows)
            stages.append(stage)
            if index < 3:
                intervals, ticks = refine(rows, width, (16, 8, 4, 1)[index + 1])
                assert ticks, "No valid connected refinement intervals"
                stage["selected_intervals"] = intervals
                stage["next_ticks"] = ticks
            c.atomic(root / "stages.json", stages)
        selected = sorted([row for row in union.values() if row.get("valid", True)],
                          key=lambda row: (row["fid"], row["tick"]))[:2]
        assert len(selected) == 2
        c.atomic(root / "selected.json", dict(selected=selected, frozen_before_5k_utc=c.now()))
        results = evaluate([row["tick"] for row in selected], 5000,
                           f"full_step{args.step:06d}_{args.weights}_top2_5k")
        c.atomic(root / "result.json", dict(complete=True, stages=stages, selected=selected,
            results5k=results, previous_best_5k_fid=36.688847797154665,
            beats_previous_best=min(row["fid"] for row in results) < 36.688847797154665,
            five_k_includes_selection_one_k=True, updated_utc=c.now()))
        publish("complete", best_5k_fid=min(row["fid"] for row in results))
    except InterruptedError:
        publish("paused")
    except Exception as error:
        publish("failed", error=repr(error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--weights", choices=("ema", "head"), default="ema")
    parser.add_argument("--pilot", action="store_true")
    main(parser.parse_args())
