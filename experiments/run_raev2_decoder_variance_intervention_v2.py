#!/usr/bin/env python3
"""Safe launcher for the RAEv2 decoder feature-variance intervention.

Place this file next to:
    experiments/run_raev2_decoder_variance_intervention.py

Why this launcher exists
------------------------
The original experiment intentionally rejects endpoint caches produced under a
mismatched protocol (for example, reusing a 1k cache for a 5k run). This
launcher gives every distinct command-line protocol its own output directory,
so 1k/5k runs and other configuration changes cannot collide accidentally.

It does not alter sampling, decoder hooks, variance amplification, FID, or any
other experiment logic. It only resolves a collision-safe --output-dir and
then executes the original program in the same torchrun worker process.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
from typing import Iterable


ORIGINAL_SCRIPT_NAME = "run_raev2_decoder_variance_intervention.py"
WRAPPER_ONLY_FLAGS = {"--v2-no-auto-output-suffix"}


def _find_option(args: list[str], option: str) -> tuple[int, str] | None:
    """Return (index, value) for --name value or --name=value."""
    for index, token in enumerate(args):
        if token == option:
            if index + 1 >= len(args):
                raise SystemExit(f"{option} requires a value")
            return index, args[index + 1]
        prefix = option + "="
        if token.startswith(prefix):
            return index, token[len(prefix) :]
    return None


def _replace_option(args: list[str], option: str, value: str) -> list[str]:
    result = list(args)
    found = _find_option(result, option)
    if found is None:
        result.extend([option, value])
        return result

    index, _ = found
    token = result[index]
    if token == option:
        result[index + 1] = value
    else:
        result[index] = f"{option}={value}"
    return result


def _without_wrapper_flags(args: Iterable[str]) -> list[str]:
    return [token for token in args if token not in WRAPPER_ONLY_FLAGS]


def _canonical_protocol_args(args: list[str]) -> list[str]:
    """Build a stable signature from all arguments except output location.

    Including evaluation/intervention arguments may create more cache copies
    than strictly necessary, but it guarantees safety and avoids guessing which
    options the original script considers protocol-critical.
    """
    canonical: list[str] = []
    skip_next = False
    for token in args:
        if skip_next:
            skip_next = False
            continue
        if token == "--output-dir":
            skip_next = True
            continue
        if token.startswith("--output-dir="):
            continue
        if token == "--overwrite-cache":
            # This flag controls cache policy rather than the experiment itself.
            continue
        canonical.append(token)

    canonical.extend(
        [
            f"WORLD_SIZE={os.environ.get('WORLD_SIZE', '1')}",
            "launcher_protocol=raev2_decoder_variance_v2",
        ]
    )
    return canonical


def _make_suffix(args: list[str]) -> str:
    canonical = _canonical_protocol_args(args)
    digest = hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:10]

    sample_count = _find_option(args, "--sample-count")
    layer_index = _find_option(args, "--layer-index")
    sample_tag = f"n{sample_count[1]}" if sample_count else "ndefault"
    layer_tag = f"l{layer_index[1]}" if layer_index else "lauto"
    return f"{sample_tag}_{layer_tag}_cfg{digest}"


def main() -> None:
    original = Path(__file__).resolve().with_name(ORIGINAL_SCRIPT_NAME)
    if not original.is_file():
        raise SystemExit(
            "Cannot find the original experiment script next to this launcher:\n"
            f"  expected: {original}\n"
            "Put both files in the repository's experiments/ directory."
        )

    raw_args = sys.argv[1:]
    auto_suffix = "--v2-no-auto-output-suffix" not in raw_args
    forwarded = _without_wrapper_flags(raw_args)

    output_option = _find_option(forwarded, "--output-dir")
    if output_option is None:
        raise SystemExit("The experiment requires --output-dir PATH")

    requested_output = Path(output_option[1]).expanduser()
    if auto_suffix:
        suffix = _make_suffix(forwarded)
        resolved_output = requested_output.parent / f"{requested_output.name}__{suffix}"
    else:
        resolved_output = requested_output

    forwarded = _replace_option(forwarded, "--output-dir", str(resolved_output))

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank == 0:
        print("[RAEv2 variance v2 launcher]", flush=True)
        print(f"  requested output : {requested_output}", flush=True)
        print(f"  resolved output  : {resolved_output}", flush=True)
        print(f"  original script  : {original}", flush=True)
        print(
            "  command          : "
            + " ".join(shlex.quote(part) for part in [sys.executable, str(original), *forwarded]),
            flush=True,
        )

    # Replace this worker process instead of nesting a second distributed launch.
    os.execv(sys.executable, [sys.executable, str(original), *forwarded])


if __name__ == "__main__":
    main()