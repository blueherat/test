"""Launch and analyze the paired semantic-conditioning gate on four GPUs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.rae_latent_cache import load_cache_manifest  # noqa: E402


CONFIG = ROOT / "experiments/configs/rae_spectral_tiny_ditdh_s_dinov2.yaml"
SUBSPACES = (
    Path.home()
    / "data/eqvae/experiments/rae_layerwise_path/"
    "gate1_imagenet_train1024_val256_mid9/subspaces.pt"
)
LATENT_CACHE = (
    Path.home()
    / "data/eqvae/cache/rae_layerwise_path_streams/seed3407_n160000_fp32"
)
RESULTS = Path.home() / "data/eqvae/experiments/rae_dual_stream_gate"
SEEDS = (3407, 4211)
MODES = ("paired", "shuffled")


def branch_name(seed: int, mode: str, steps: int) -> str:
    return f"seed{int(seed)}_{mode}_steps{int(steps)}"


def analyze_results(results: Path, seeds: tuple[int, ...] = SEEDS) -> dict[str, object]:
    rows = []
    for seed in seeds:
        payloads = {}
        for mode in MODES:
            path = results / branch_name(seed, mode, 2000) / "result.json"
            if not path.exists():
                continue
            payloads[mode] = json.loads(path.read_text(encoding="utf-8"))
        if set(payloads) != set(MODES):
            continue
        paired = payloads["paired"]
        shuffled = payloads["shuffled"]
        usage_gain = float(paired["context_usage_gain"])
        training_gain = 1.0 - float(paired["paired_context_normalized_mse"]) / float(
            shuffled["paired_context_normalized_mse"]
        )
        rows.append(
            {
                "seed": seed,
                "paired_model_paired_mse": float(paired["paired_context_normalized_mse"]),
                "paired_model_shuffled_mse": float(
                    paired["shuffled_context_normalized_mse"]
                ),
                "shuffled_model_paired_mse": float(
                    shuffled["paired_context_normalized_mse"]
                ),
                "context_usage_gain": usage_gain,
                "training_gain": training_gain,
                "pass": bool(usage_gain >= 0.10 and training_gain >= 0.10),
            }
        )
    complete = len(rows) == len(seeds)
    gate_pass = bool(complete and all(row["pass"] for row in rows))
    return {
        "status": "pass" if gate_pass else ("fail" if complete else "incomplete"),
        "complete_seeds": len(rows),
        "required_seeds": len(seeds),
        "rows": rows,
        "gate_pass": gate_pass,
        "decision": (
            "proceed_to_full_dual_stream_generation"
            if gate_pass
            else ("stop_dual_stream" if complete else "continue_gate")
        ),
    }


def render_report(result: dict[str, object]) -> str:
    lines = ["# RAE 语义-细节双流条件门控", "", f"状态：`{result['status']}`", ""]
    for row in result["rows"]:
        lines.extend(
            [
                f"## Seed {row['seed']}",
                "",
                f"- 正确 context 相对打乱 context 改善：{100 * row['context_usage_gain']:.2f}%",
                f"- 正确训练相对 shuffled 训练改善：{100 * row['training_gain']:.2f}%",
                f"- 单 seed 门控：{'通过' if row['pass'] else '未通过'}",
                "",
            ]
        )
    lines.extend(
        [
            f"最终决策：`{result['decision']}`",
            "",
            "门槛要求两个 seed 的两项改善都不低于 10%。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "run", "analyze"), default="preflight")
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--latent-cache", type=Path, default=LATENT_CACHE)
    parser.add_argument("--subspaces", type=Path, default=SUBSPACES)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-count", type=int, default=4096)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    devices = tuple(int(item) for item in args.devices.split(",") if item.strip())
    required = (CONFIG, args.subspaces, args.latent_cache / "manifest.json")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing inputs:\n" + "\n".join(missing))
    cache = load_cache_manifest(args.latent_cache)
    required_samples = int(args.steps) * int(args.batch_size) + int(args.val_count)
    if int(cache["sample_count"]) < required_samples:
        raise ValueError("latent cache is too small for train/validation split")
    args.results.mkdir(parents=True, exist_ok=True)
    protocol = {
        "question": "Can clean semantic latent predict held-out rank-16 detail flow?",
        "seeds": list(SEEDS),
        "conditions": list(MODES),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "val_count": int(args.val_count),
        "precision": "fp32",
        "gate": "context_usage_gain >= 10% and training_gain >= 10% for 2/2 seeds",
        "stop_rule": "failure stops full dual-stream generation",
    }
    protocol_path = args.results / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise RuntimeError("refusing to change the registered dual-stream protocol")
    else:
        protocol_path.write_text(
            json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(json.dumps(protocol, indent=2, ensure_ascii=False))
    if args.mode == "preflight":
        return
    if args.mode == "run":
        jobs = [(seed, mode) for seed in SEEDS for mode in MODES]
        if len(devices) < len(jobs):
            raise ValueError("the paired two-seed gate requires four devices")
        processes = []
        log_root = args.results / "logs"
        log_root.mkdir(exist_ok=True)
        for device, (seed, context_mode) in zip(devices, jobs):
            name = branch_name(seed, context_mode, int(args.steps))
            command = [
                sys.executable,
                str(ROOT / "experiments/train_rae_dual_stream_gate.py"),
                "--config",
                str(CONFIG),
                "--latent-cache",
                str(args.latent_cache),
                "--subspaces",
                str(args.subspaces),
                "--results",
                str(args.results),
                "--experiment-name",
                name,
                "--context-mode",
                context_mode,
                "--seed",
                str(seed),
                "--steps",
                str(args.steps),
                "--batch-size",
                str(args.batch_size),
                "--val-count",
                str(args.val_count),
            ]
            print(f"[{name}] CUDA {device}: {' '.join(command)}", flush=True)
            if args.dry_run:
                continue
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(device)
            environment["PYTHONUNBUFFERED"] = "1"
            handle = (log_root / f"{name}.log").open("a", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            processes.append((name, process, handle))
        failures = []
        for name, process, handle in processes:
            code = process.wait()
            handle.close()
            print(f"[{name}] exit={code}", flush=True)
            if code:
                failures.append(name)
        if failures:
            raise RuntimeError(f"dual-stream gate branches failed: {failures}")
    if not args.dry_run:
        result = analyze_results(args.results)
        (args.results / "gate.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        report = render_report(result)
        (args.results / "gate_zh.md").write_text(report, encoding="utf-8")
        print(report)


if __name__ == "__main__":
    main()
