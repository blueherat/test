"""Persistent launcher for the resumable RAEv2 scale-response study."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_raev2_scale_response_study import DEFAULT_SCALES  # noqa: E402


DEFAULT_OUTPUT = Path(
    "/home/zhoushunyu/data/eqvae/experiments/raev2_ig_scale_response/"
    "n1000_seed20260801_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--scale", action="append", type=float, dest="scales")
    parser.add_argument("--nproc-per-node", type=int, default=4)
    parser.add_argument("--per-rank-batch", type=int, default=2)
    parser.add_argument("--decode-batch", type=int, default=4)
    parser.add_argument("--encode-batch", type=int, default=4)
    parser.add_argument("--min-available-memory-gib", type=float, default=32.0)
    parser.add_argument("--resource-poll-seconds", type=float, default=60.0)
    parser.add_argument("--retry-delay-seconds", type=float, default=120.0)
    parser.add_argument("--max-attempts", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    torchrun = Path(sys.executable).resolve().parent / "torchrun"
    command = [
        str(torchrun if torchrun.is_file() else "torchrun"),
        "--standalone",
        f"--nproc_per_node={int(args.nproc_per_node)}",
        "experiments/run_raev2_scale_response_study.py",
        "--output-dir",
        str(args.output_dir.expanduser().resolve()),
        "--samples",
        str(int(args.samples)),
        "--seed",
        str(int(args.seed)),
        "--per-rank-batch",
        str(int(args.per_rank_batch)),
        "--decode-batch",
        str(int(args.decode_batch)),
        "--encode-batch",
        str(int(args.encode_batch)),
        "--log-every-batches",
        "10",
        "--bootstrap-repeats",
        "500",
        "--sketch-dim",
        "16",
        "--metric-batch",
        "8",
    ]
    for scale in args.scales or DEFAULT_SCALES:
        command.extend(("--scale", str(float(scale))))
    return command


def available_memory_gib(meminfo_path: Path = Path("/proc/meminfo")) -> float:
    for line in meminfo_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            fields = line.split()
            if len(fields) < 2:
                break
            return float(fields[1]) / (1024.0**2)
    raise RuntimeError(f"MemAvailable is missing from {meminfo_path}")


def write_status(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def wait_for_host_memory(
    *,
    minimum_gib: float,
    poll_seconds: float,
    log: object,
    status_path: Path,
    command: list[str],
) -> None:
    while True:
        available_gib = available_memory_gib()
        if available_gib >= minimum_gib:
            return
        checked = datetime.now(timezone.utc).isoformat()
        message = (
            f"[{checked}] waiting for host memory: {available_gib:.2f} GiB "
            f"available, need {minimum_gib:.2f} GiB\n"
        )
        log.write(message)
        log.flush()
        write_status(
            status_path,
            {
                "state": "waiting_for_memory",
                "checked_utc": checked,
                "available_memory_gib": available_gib,
                "required_memory_gib": minimum_gib,
                "command": command,
            },
        )
        time.sleep(poll_seconds)


def main() -> None:
    args = parse_args()
    if args.min_available_memory_gib <= 0:
        raise ValueError("--min-available-memory-gib must be positive")
    if args.resource_poll_seconds <= 0 or args.retry_delay_seconds <= 0:
        raise ValueError("poll and retry delays must be positive")
    if args.max_attempts <= 0:
        raise ValueError("--max-attempts must be positive")
    output_dir = args.output_dir.expanduser().resolve()
    command = build_command(args)
    if args.dry_run:
        print(json.dumps(command, indent=2))
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "pipeline.log"
    status_path = output_dir / "pipeline_status.json"
    attempts: list[dict[str, object]] = []
    result_code = 1
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        for attempt in range(1, int(args.max_attempts) + 1):
            wait_for_host_memory(
                minimum_gib=float(args.min_available_memory_gib),
                poll_seconds=float(args.resource_poll_seconds),
                log=log,
                status_path=status_path,
                command=command,
            )
            started = datetime.now(timezone.utc).isoformat()
            log.write(
                f"\n[{started}] attempt {attempt}/{args.max_attempts}: "
                f"{json.dumps(command)}\n"
            )
            environment = os.environ.copy()
            environment.setdefault(
                "MPLCONFIGDIR", "/tmp/matplotlib-raev2-scale-response"
            )
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            finished = datetime.now(timezone.utc).isoformat()
            result_code = int(result.returncode)
            attempts.append(
                {
                    "attempt": attempt,
                    "started_utc": started,
                    "finished_utc": finished,
                    "exit_code": result_code,
                }
            )
            write_status(
                status_path,
                {
                    "state": "complete" if result_code == 0 else "retrying",
                    "finished_utc": finished,
                    "exit_code": result_code,
                    "attempts": attempts,
                    "command": command,
                    "log": str(log_path),
                },
            )
            if result_code == 0:
                break
            if attempt < int(args.max_attempts):
                log.write(
                    f"[{finished}] attempt failed with exit {result_code}; "
                    f"retrying after {args.retry_delay_seconds:g} seconds\n"
                )
                time.sleep(float(args.retry_delay_seconds))
    if result_code != 0:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
        payload["state"] = "failed"
        write_status(status_path, payload)
    raise SystemExit(result_code)


if __name__ == "__main__":
    main()
