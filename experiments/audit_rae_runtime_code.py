"""Audit the patched external RAE worktree used by strict LPL experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAE_ROOT = ROOT / "external/RAE"
EXPECTED_COMMIT = "a4d18c4db766419cbe7cb8c02cd9f7ceb0ec9041"

ALLOWED_TRACKED_PATCHES = {
    "src/sample_ddp.py": "strict device, label-pairing and RNG audit I/O",
    "src/stage1/__init__.py": "local adapter export; base RAE remains selected",
    "src/stage1/rae.py": "Transformers decoder-config compatibility",
    "src/train.py": "unused by the strict LPL trainer",
    "src/utils/resume_utils.py": "unused by the strict LPL trainer",
    "src/utils/train_utils.py": "ImageNet parquet reader used by strict LPL",
}
ALLOWED_UNTRACKED_PREFIXES = (
    ".adapter_cache/",
    "src/stage1/adapted_rae.py",
)
CRITICAL_CLEAN_PATHS = (
    "src/stage1/decoders",
    "src/stage1/encoders",
    "src/stage2",
    "src/utils/model_utils.py",
    "src/utils/optim_utils.py",
)


def run_git(root: Path, *args: str, text: bool = False) -> bytes | str:
    return subprocess.check_output(
        ("git", *args),
        cwd=root,
        text=text,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def status_rows(root: Path) -> list[dict[str, str]]:
    output = run_git(root, "status", "--porcelain=v1", "-z")
    rows = []
    for entry in output.split(b"\0"):
        if not entry:
            continue
        decoded = entry.decode("utf-8")
        rows.append({"status": decoded[:2], "path": decoded[3:]})
    return rows


def is_allowed_untracked(path: str) -> bool:
    return any(
        path == prefix.rstrip("/") or path.startswith(prefix)
        for prefix in ALLOWED_UNTRACKED_PREFIXES
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rae-root", type=Path, default=DEFAULT_RAE_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.home()
        / "data/eqvae/experiments/rae_lpl_authenticity/rae_runtime_code_audit.json",
    )
    args = parser.parse_args()

    root = args.rae_root.expanduser().resolve()
    commit = str(run_git(root, "rev-parse", "HEAD", text=True)).strip()
    status = status_rows(root)
    errors = []
    if commit != EXPECTED_COMMIT:
        errors.append(f"unexpected RAE commit: {commit}")

    for row in status:
        path = row["path"]
        if row["status"] == "??":
            if not is_allowed_untracked(path):
                errors.append(f"unexpected untracked RAE path: {path}")
        elif path not in ALLOWED_TRACKED_PATCHES:
            errors.append(f"unexpected tracked RAE modification: {row['status']} {path}")

    critical_rows = []
    for path in CRITICAL_CLEAN_PATHS:
        changed = subprocess.run(
            ("git", "diff", "--quiet", "HEAD", "--", path),
            cwd=root,
            check=False,
        ).returncode != 0
        critical_rows.append({"path": path, "matches_commit": not changed})
        if changed:
            errors.append(f"critical numerical RAE path differs from commit: {path}")

    patch_rows = []
    for path, role in sorted(ALLOWED_TRACKED_PATCHES.items()):
        local = root / path
        official = run_git(root, "show", f"HEAD:{path}")
        difference = run_git(root, "diff", "--binary", "HEAD", "--", path)
        patch_rows.append(
            {
                "path": path,
                "role": role,
                "local_sha256": sha256_file(local),
                "official_blob_sha256": sha256_bytes(official),
                "diff_sha256": sha256_bytes(difference),
                "diff_bytes": len(difference),
            }
        )

    result = {
        "passed": not errors,
        "rae_root": str(root),
        "expected_commit": EXPECTED_COMMIT,
        "actual_commit": commit,
        "description": (
            "official RAE commit plus audited compatibility, dataset-I/O and "
            "strict paired-sampling patches"
        ),
        "status": status,
        "allowed_patch_hashes": patch_rows,
        "critical_numerical_paths": critical_rows,
        "errors": errors,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
