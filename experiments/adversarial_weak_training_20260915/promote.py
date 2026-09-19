"""Finish an authorized checkpoint evaluation without a GPU gap after its pilot.

Inspect EMA and online weights at the same coefficient. A positive candidate gets the unchanged
connected coefficient search and two 5K extensions. No training hyperparameter
is changed automatically, and a negative screen waits for research judgment.
"""

import argparse
import os
import subprocess
import time
from . import common as c
from .search import alive, initial_baseline


def main(args):
    run = c.ROOT / args.run
    baseline = initial_baseline(run)
    root = run / f"promotion_step{args.step:06d}"
    root.mkdir(parents=True, exist_ok=True)
    c.atomic(root / "launch.json", dict(pid=os.getpid(), args=vars(args), started_utc=c.now()))

    def publish(phase, **values):
        c.atomic(root / "status.json", dict(phase=phase, updated_utc=c.now(), **values))

    def complete_search(weights, pilot):
        name = f"search_step{args.step:06d}_{weights}_" + ("pilot" if pilot else "full")
        directory = run / name
        result = directory / "result.json"
        if not result.exists():
            launch = directory / "launch.json"
            child_pid = c.read(launch)["pid"] if launch.exists() else None
            if launch.exists() and not alive(c.read(launch)["pid"]):
                raise RuntimeError(f"Existing evaluation exited without a result: {directory}")
            if not launch.exists():
                command = [c.PYTHON, "-u", "-m", "experiments.adversarial_weak_training_20260915.search",
                           "--run", args.run, "--step", str(args.step), "--weights", weights]
                if pilot:
                    command.append("--pilot")
                with (root / (name + ".log")).open("a") as stream:
                    process = subprocess.Popen(command, cwd=c.WORK, stdout=stream,
                        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
                c.atomic(root / (name + "_child.json"), dict(pid=process.pid, command=command))
                child_pid = process.pid
            while not result.exists():
                if c.stopped():
                    raise InterruptedError("Stopped new endpoint task")
                status = directory / "status.json"
                current = c.read(status) if status.exists() else {}
                if current.get("phase") in ("failed", "paused"):
                    raise RuntimeError(f"Evaluation {current}")
                if not alive(child_pid) and not result.exists():
                    raise RuntimeError(f"Evaluation process exited: {directory}")
                publish("waiting_pilot" if pilot else "full_evaluation", weights=weights,
                        evaluation=current)
                time.sleep(10)
        return c.read(result)

    try:
        screens = {}
        for weights in ("ema", "head"):
            screen = complete_search(weights, True)
            screens[weights] = screen
        winner = min(screens, key=lambda key: (screens[key]["rows"][0]["fid"], key != "ema"))
        selected = winner if args.always_full or screens[winner]["rows"][0]["fid"] < baseline['fid'] else None
        c.atomic(root / "decision.json", dict(screens=screens, selected_weights=selected,
            baseline_same_coefficient_fid=baseline['fid'], initial_baseline=baseline, decided_utc=c.now(),
            always_complete_coefficient_evaluation=args.always_full,
            rule="Inspect EMA and online weights at a=1; lower 1K FID selects the checkpoint, ties prefer EMA"))
        if selected is None:
            publish("needs_research_decision", screens={k:v["rows"][0]["fid"] for k,v in screens.items()})
            return
        result = complete_search(selected, False)
        c.atomic(root / "result.json", dict(weights=selected, result=result, completed_utc=c.now()))
        publish("complete", selected_weights=selected, beats_previous_best=result["beats_previous_best"],
                best_5k_fid=min(row["fid"] for row in result["results5k"]))
    except InterruptedError:
        publish("paused")
    except Exception as error:
        publish("failed", error=repr(error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--always-full", action="store_true")
    main(parser.parse_args())
